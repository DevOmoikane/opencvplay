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
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, SpinnerColumn
import click
from matplotlib import pyplot as plt
import traceback


# Check for CUDA availability
HAS_CUDA = cv2.cuda.getCudaEnabledDeviceCount() > 0
HAS_TORCH_CUDA = torch.cuda.is_available()

template_cache = {}

# Initialize appropriate feature detectors
if HAS_CUDA:
    try:
        sift_gpu = cv2.cuda.SIFT_create()
        print("Using CUDA-accelerated SIFT")
    except:
        sift_gpu = None
        sift = cv2.SIFT_create()
        print("CUDA SIFT not available, using CPU SIFT")
else:
    sift = cv2.SIFT_create()
    sift_gpu = None
    print("Using CPU SIFT")


def search_single_match(template_path: str, image_path: str, min_match: int = 10, flann_index_kdtree: int = 0):
    # check if template path already in template_cache
    kp_template = None
    des_template = None
    img_template = None
    if template_path not in template_cache:
        img_template = cv2.imread(template_path, 0)  # Fixed: was reading 'template.png' instead of template_path
        if img_template is None:
            return None
        kp_template, des_template = sift.detectAndCompute(img_template, None)
        template_cache[template_path] = (kp_template, des_template, img_template)
    else:
        kp_template, des_template, img_template = template_cache[template_path]

    image = cv2.imread(image_path, 0)
    if image is None:
        return None
    kp_image, des_image = sift.detectAndCompute(image, None)
    
    if des_template is None or des_image is None:
        return None
        
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
        if M is None:
            return None
        matchesMask = mask.ravel().tolist()
        h, w = img_template.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
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
        if img_template is None:
            return None
        kp_template, des_template = sift.detectAndCompute(img_template, None)
        template_cache[template_path] = (kp_template, des_template, img_template)
    else:
        kp_template, des_template, img_template = template_cache[template_path]

    image = cv2.imread(image_path, 0)
    if image is None:
        return None
    kp_image, des_image = sift.detectAndCompute(image, None)
    
    if des_template is None or des_image is None:
        return None
    
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

def search_multiple_matches_gpu(template_path: str, image_path: str, min_match: int = 10, 
                               max_matches: int = 10):
    """GPU-accelerated multiple template matching"""
    if not HAS_CUDA or sift_gpu is None:
        return search_multiple_matches_cpu(template_path, image_path, min_match, max_matches)
    
    try:
        # Load images to GPU
        img_template = cv2.imread(template_path, 0)
        if img_template is None:
            return None
        
        image = cv2.imread(image_path, 0)
        if image is None:
            return None
        
        # Upload to GPU
        gpu_template = cv2.cuda_GpuMat()
        gpu_template.upload(img_template)
        
        gpu_image = cv2.cuda_GpuMat()
        gpu_image.upload(image)
        
        # Detect features on GPU
        kp_template_gpu, des_template_gpu = sift_gpu.detectAndCompute(gpu_template, None)
        kp_image_gpu, des_image_gpu = sift_gpu.detectAndCompute(gpu_image, None)
        
        # Download to CPU for matching (FLANN doesn't have good GPU support)
        kp_template = sift_gpu.downloadKeypoints(kp_template_gpu)
        des_template = des_template_gpu.download()
        kp_image = sift_gpu.downloadKeypoints(kp_image_gpu)
        des_image = des_image_gpu.download()
        
        if des_template is None or des_image is None:
            return None
        
        # Continue with CPU matching (this part is fast)
        return _find_matches_with_keypoints(kp_template, des_template, kp_image, des_image, 
                                          img_template.shape, min_match, max_matches)
        
    except Exception as e:
        print(f"GPU processing failed, falling back to CPU: {e}")
        return search_multiple_matches_cpu(template_path, image_path, min_match, max_matches)

def search_multiple_matches_cpu(template_path: str, image_path: str, min_match: int = 10, 
                               max_matches: int = 10):
    """CPU version with optimizations"""
    if template_path not in template_cache:
        img_template = cv2.imread(template_path, 0)
        if img_template is None:
            return None
        kp_template, des_template = sift.detectAndCompute(img_template, None)
        template_cache[template_path] = (kp_template, des_template, img_template)
    else:
        kp_template, des_template, img_template = template_cache[template_path]

    image = cv2.imread(image_path, 0)
    if image is None:
        return None
    kp_image, des_image = sift.detectAndCompute(image, None)
    
    if des_template is None or des_image is None:
        return None
    
    return _find_matches_with_keypoints(kp_template, des_template, kp_image, des_image,
                                      img_template.shape, min_match, max_matches)

def _find_matches_with_keypoints(kp_template, des_template, kp_image, des_image, 
                                template_shape, min_match, max_matches):
    """Common matching logic for both GPU and CPU"""
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    all_matches = []
    current_kp_image = list(kp_image)
    current_des_image = des_image.copy() if des_image is not None else None
    
    h, w = template_shape
    
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
                
                dst_int = np.int32(dst).reshape((dst.shape[0], dst.shape[2]))
                all_matches.append(dst_int)
                
                good_matches_used = [good[i] for i in range(len(good)) if matchesMask[i] == 1]
                
                if good_matches_used:
                    used_indices = sorted(set([m.trainIdx for m in good_matches_used]), reverse=True)
                    
                    for idx in used_indices:
                        if idx < len(current_kp_image):
                            del current_kp_image[idx]
                        if current_des_image is not None and idx < len(current_des_image):
                            current_des_image = np.delete(current_des_image, idx, axis=0)
                    
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

def create_mask_from_polygons(image_shape: tuple, polygons: list) -> np.ndarray:
    """Create mask from polygon coordinates"""
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    for polygon in polygons:
        cv2.fillPoly(mask, [polygon], 255)
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

def get_image_files(directory: str) -> list:
    """Get all image files from directory"""
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    image_files = []
    for file in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, file)):
            ext = os.path.splitext(file)[1].lower()
            if ext in valid_extensions:
                image_files.append(file)
    return sorted(image_files)

def process_image_with_templates(image_path: str, template_files: list, template_dir: str, 
                               output_dir: str, progress, template_task, min_match: int = 10):
    """Process a single image with all templates"""
    try:
        # Read the image once
        image = cv2.imread(image_path)
        if image is None:
            # progress.console.print(f"[red]Error reading image: {os.path.basename(image_path)}[/red]")
            return
        
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        all_matches = []
        
        # Process each template
        for template_file in template_files:
            template_path = os.path.join(template_dir, template_file)
            
            # Update template progress
            progress.update(template_task, advance=1, description=f"Template: {template_file[:20]}...")
            
            # Search for multiple matches
            matches = search_multiple_matches_gpu(
                template_path, 
                image_path, 
                min_match=min_match, 
                max_matches=10
            )
            
            if matches:
                all_matches.extend(matches)
            #     progress.console.print(f"[green]Found {len(matches)} instances of {template_file} in {os.path.basename(image_path)}[/green]")
            # else:
            #     progress.console.print(f"[yellow]No matches found for {template_file} in {os.path.basename(image_path)}[/yellow]")
        
        # Apply resynthesize if we found any matches
        if all_matches:
            # Create combined mask from all matches
            mask = create_mask_from_polygons(image.shape, all_matches)
            
            # Apply resynthesize filter
            result = apply_filter_to_mask(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), mask)
            
            # Save result
            output_filename = f"{os.path.basename(image_path)}"
            output_path = os.path.join(output_dir, output_filename)
            cv2.imwrite(output_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
            
        #     progress.console.print(f"[blue]Saved processed image: {output_filename}[/blue]")
        # else:
        #     progress.console.print(f"[yellow]No matches found in {os.path.basename(image_path)}, skipping[/yellow]")
            
    except Exception as e:
        # progress.console.print(f"[red]Error processing {os.path.basename(image_path)}: {str(e)}[/red]")
        # progress.console.print(traceback.format_exc())
        pass

@click.command()
@click.option("--template-dir", type=click.Path(exists=True), default="templates", help="Folder containing templates.")
@click.option("--image-dir", type=click.Path(exists=True), default="images", help="Folder containing images.")
@click.option("--output-dir", type=click.Path(exists=False), default="output", help="Folder to save processed images.")
@click.option("--min-matches", type=int, default=10, help="Minimum number of matches required.")
@click.option("--max-matches", type=int, default=5, help="Maximum number of matches to find per template.")
def main(**options):
    template_dir = options["template_dir"]
    image_dir = options["image_dir"]
    output_dir = options["output_dir"]
    min_matches = options["min_matches"]
    max_matches = options["max_matches"]
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get template and image files
    template_files = get_image_files(template_dir)
    image_files = get_image_files(image_dir)
    
    if not template_files:
        print(f"No template images found in {template_dir}")
        return
        
    if not image_files:
        print(f"No images found in {image_dir}")
        return
    
    print(f"Found {len(template_files)} templates and {len(image_files)} images")
    
    # Configure rich progress
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=None,
        expand=False
    ) as progress:
        # Main task for images
        main_task = progress.add_task(
            f"[cyan]Processing {len(image_files)} images...", 
            total=len(image_files)
        )
        
        for image_file in image_files:
            image_path = os.path.join(image_dir, image_file)
            
            # Update main progress
            progress.update(main_task, advance=0, description=f"[cyan]Processing: {image_file[:30]}...")
            
            # Create sub-task for templates
            template_task = progress.add_task(
                f"[yellow]Checking {len(template_files)} templates...", 
                total=len(template_files)
            )
            
            # Process this image with all templates
            process_image_with_templates(
                image_path, 
                template_files, 
                template_dir, 
                output_dir, 
                progress, 
                template_task,
                min_match=min_matches
            )
            
            # Remove the template task when done
            progress.remove_task(template_task)
            
            # Advance main progress
            progress.update(main_task, advance=1)
        
        progress.console.print("[green]✓ All images processed successfully![/green]")

if __name__ == "__main__":
    main()
