import os
import cv2
import math
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as T
from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.cluster import DBSCAN
from resynthesizer import resynthesize
from rich.progress import Progress
import click
from matplotlib import pyplot as plt
import traceback


template_cache = {}
sift = cv2.SIFT_create()

def search_single_match(template_path: str, image_path: str, min_match: int = 10, flann_index_kdtree: int = 0):
    # check if template path already in template_cache
    kp_template = None
    des_template = None
    img_template = None
    if template_path not in template_cache:
        img_template = cv2.imread('template.png', 0)
        kp_template, des_template = sift.detectAndCompute(img_template, None)
        template_cache[template_path] = (kp_template, des_template, img_template)
    else:
        kp_template, des_template, img_template = template_cache[template_path]

    image = cv2.imread(image_path, 0)
    kp_image, des_image = sift.detectAndCompute(image, None)
    index_params = dict(algorithm=flann_index_kdtree, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des_template, des_image, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
    if len(good) > min_match:
        template_pts = np.float32([kp_template[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        image_pts = np.float32([kp_image[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(template_pts, image_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
        h, w = img_template.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        #image = cv2.polylines(image, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
        return np.int32(dst).reshape((dst.shape[0],dst.shape[2]))
    else:
        return None

def search_multiple_matches(template_path: str, image_path: str, min_match: int = 10, 
                          max_matches: int = 10, flann_index_kdtree: int = 1):
    """
    Find multiple instances of template in image using iterative homography.
    """
    # Load or get cached template data
    if template_path not in template_cache:
        img_template = cv2.imread(template_path, 0)
        kp_template, des_template = sift.detectAndCompute(img_template, None)
        template_cache[template_path] = (kp_template, des_template, img_template)
    else:
        kp_template, des_template, img_template = template_cache[template_path]

    image = cv2.imread(image_path, 0)
    kp_image, des_image = sift.detectAndCompute(image, None)
    
    # FLANN matcher
    index_params = dict(algorithm=flann_index_kdtree, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    all_matches = []
    
    # Create mutable copies of keypoints and descriptors
    current_kp_image = list(kp_image)  # Convert to list for easy removal
    current_des_image = des_image.copy() if des_image is not None else None
    
    h, w = img_template.shape
    
    for match_attempt in range(max_matches):
        if current_des_image is None or len(current_kp_image) < min_match:
            break
            
        matches = flann.knnMatch(des_template, current_des_image, k=2)
        good = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.7 * n.distance:
                    good.append(m)
                
        if len(good) > min_match:
            template_pts = np.float32([kp_template[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            image_pts = np.float32([current_kp_image[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            
            M, mask = cv2.findHomography(template_pts, image_pts, cv2.RANSAC, 5.0)
            
            if M is not None:
                matchesMask = mask.ravel().tolist()
                pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, M)
                
                # Convert to integer coordinates and add to results
                dst_int = np.int32(dst).reshape((dst.shape[0], dst.shape[2]))
                all_matches.append(dst_int)
                
                # Remove matched keypoints for next iteration
                good_matches_used = [good[i] for i in range(len(good)) if matchesMask[i] == 1]
                
                if good_matches_used:
                    # Get indices of used keypoints (sort in descending order for safe removal)
                    used_indices = sorted(set([m.trainIdx for m in good_matches_used]), reverse=True)
                    
                    # Remove used keypoints and descriptors
                    for idx in used_indices:
                        if idx < len(current_kp_image):
                            del current_kp_image[idx]
                        if current_des_image is not None and idx < len(current_des_image):
                            current_des_image = np.delete(current_des_image, idx, axis=0)
                    
                    # Update FLANN matcher with new descriptors
                    if current_des_image is not None and len(current_des_image) > 0:
                        flann = cv2.FlannBasedMatcher(index_params, search_params)
                    else:
                        current_des_image = None
                else:
                    break
            else:
                break
        else:
            break
    
    return all_matches if all_matches else None

def create_mask_from_coordinates(mask: np.ndarray, x1: int, y1: int, x2: int, y2: int, padding: int = 5) -> np.ndarray:
    target_size = mask.shape[:2]
    top_left_padded = (max(0, x1 - padding), max(0, y1 - padding))
    bottom_right_padded = (min(target_size[1], x2 + padding), min(target_size[0], y2 + padding))
    cv2.rectangle(mask, top_left_padded, bottom_right_padded, 255, -1)
    return mask

def create_mask_from_list(source_frame: np.ndarray, coordinates: list[tuple[int, int, int, int]]) -> np.ndarray:
    mask = np.zeros(source_frame.shape[:2], dtype=np.uint8)
    for x1, y1, x2, y2 in coordinates:
        create_mask_from_coordinates(mask, x1, y1, x2, y2)
    return mask

def apply_filter_to_mask(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    pil_img = Image.fromarray(frame)
    mask_pil = Image.fromarray(mask)
    result_pil = resynthesize(pil_img, mask_pil)
    result = np.array(result_pil)
    return result

@click.command()
@click.option("--template-dir", type=click.Path(exists=True), default="templates", help="Folder containing templates.")
@click.option("--image-dir", type=click.Path(exists=True), default="images", help="Folder containing images.")
@click.option("--output-dir", type=click.Path(exists=False), default="output", help="Folder to save processed images.")
def main(**options):
    output_dir = options["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    t_root, _, t_files = os.walk(options["template_dir"])
    root, _, files = os.walk(options["image_dir"])

if __name__ == "__main__":
    main()
