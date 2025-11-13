import cv2
import numpy as np
from PIL import Image
import os
from resynthesizer import resynthesize
from rich import print

def feature_based_matching(source_path, target_path, min_matches=10):
    """
    Use feature matching (ORB) to find template with scale/rotation invariance.
    """
    # Read images
    source = cv2.imread(source_path, cv2.IMREAD_GRAYSCALE)
    target = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
    
    if source is None or target is None:
        return None
    
    # Initialize ORB detector
    orb = cv2.ORB_create(1000)  # Increase features for better matching
    
    # Find keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(source, None)
    kp2, des2 = orb.detectAndCompute(target, None)
    
    if des1 is None or des2 is None:
        return None
    
    # FLANN based matcher
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH,
                       table_number=6,
                       key_size=12,
                       multi_probe_level=1)
    search_params = dict(checks=50)
    
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    
    # Lowe's ratio test
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
    
    print(f"Found {len(good_matches)} good matches")
    
    if len(good_matches) < min_matches:
        return None
    
    # Extract location of good matches
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # Find homography
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    if M is None:
        return None
    
    # Get corners of source image in target coordinates
    h, w = source.shape
    corners = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
    transformed_corners = cv2.perspectiveTransform(corners, M)
    
    # Calculate bounding box
    x_coords = transformed_corners[:, 0, 0]
    y_coords = transformed_corners[:, 0, 1]
    
    top_left = (int(np.min(x_coords)), int(np.min(y_coords)))
    bottom_right = (int(np.max(x_coords)), int(np.max(y_coords)))
    
    return {
        'top_left': top_left,
        'bottom_right': bottom_right,
        'corners': transformed_corners,
        'confidence': len(good_matches) / len(matches) if matches else 0,
        'matches_count': len(good_matches)
    }

def pyramid_template_matching(source_path, target_path, threshold=0.7, scale_steps=10):
    """
    Efficient pyramid-based template matching.
    """
    source = cv2.imread(source_path)
    target = cv2.imread(target_path)
    
    if source is None or target is None:
        return None
    
    source_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    
    # Get dimensions
    h, w = source_gray.shape
    H, W = target_gray.shape
    
    # Calculate scale range
    min_scale = max(0.1, max(10/w, 10/h))
    max_scale = min(5.0, min(W/w, H/h))
    
    best_match = None
    best_confidence = threshold
    
    # Logarithmic scale sampling (more efficient)
    scales = np.logspace(np.log10(min_scale), np.log10(max_scale), scale_steps)
    
    for scale in scales:
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        if new_w > W or new_h > H or new_w < 10 or new_h < 10:
            continue
            
        # Resize template
        resized_source = cv2.resize(source_gray, (new_w, new_h))
        
        # Template matching
        result = cv2.matchTemplate(target_gray, resized_source, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        if max_val > best_confidence:
            best_confidence = max_val
            best_match = {
                'top_left': max_loc,
                'bottom_right': (max_loc[0] + new_w, max_loc[1] + new_h),
                'confidence': max_val,
                'scale': scale
            }
    
    return best_match

def efficient_keypoint_matching(source_path, target_path, min_matches=15):
    """
    Efficient keypoint matching with automatic scale estimation.
    """
    source = cv2.imread(source_path, cv2.IMREAD_GRAYSCALE)
    target = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
    
    if source is None or target is None:
        return None
    
    # Use AKAZE for better scale/rotation invariance
    akaze = cv2.AKAZE_create()
    kp1, des1 = akaze.detectAndCompute(source, None)
    kp2, des2 = akaze.detectAndCompute(target, None)
    
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return None
    
    # BFMatcher with default params
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1, des2, k=2)
    
    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    
    if len(good_matches) < min_matches:
        return None
    
    # Estimate scale from keypoint distances
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches])
    
    # Calculate scale from median distance ratio
    scales = []
    for i in range(len(src_pts)):
        for j in range(i+1, len(src_pts)):
            src_dist = np.linalg.norm(src_pts[i] - src_pts[j])
            dst_dist = np.linalg.norm(dst_pts[i] - dst_pts[j])
            if src_dist > 0:
                scales.append(dst_dist / src_dist)
    
    if not scales:
        return None
    
    estimated_scale = np.median(scales)
    
    # Find homography with scale constraint
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    if M is None:
        return None
    
    # Transform corners
    h, w = source.shape
    corners = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
    transformed_corners = cv2.perspectiveTransform(corners, M)
    
    x_coords = transformed_corners[:, 0, 0]
    y_coords = transformed_corners[:, 0, 1]
    
    return {
        'top_left': (int(np.min(x_coords)), int(np.min(y_coords))),
        'bottom_right': (int(np.max(x_coords)), int(np.max(y_coords))),
        'corners': transformed_corners,
        'confidence': len(good_matches) / len(kp1),
        'estimated_scale': estimated_scale,
        'matches_count': len(good_matches)
    }

def find_template_coordinates(source_path, target_path, threshold=0.8):
    """
    Find coordinates of source image in target image using template matching.
    Returns coordinates of best match or None if not found.
    """
    # Read images
    source = cv2.imread(source_path)
    target = cv2.imread(target_path)
    
    if source is None or target is None:
        print(f"Error reading images: {source_path} or {target_path}")
        return None
    
    # Try multiple scales to account for size/aspect ratio variations
    scales = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    best_match = None
    best_val = threshold
    
    for scale in np.arange(0.1, 2.0, 0.05):
        # Calculate new dimensions
        height, width = source.shape[:2]
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Skip if resized source is larger than target
        if new_width > target.shape[1] or new_height > target.shape[0]:
            continue
            
        # Resize source image
        resized_source = cv2.resize(source, (new_width, new_height))
        
        # Template matching
        result = cv2.matchTemplate(target, resized_source, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        if max_val > best_val:
            best_val = max_val
            best_match = {
                'top_left': max_loc,
                'bottom_right': (max_loc[0] + new_width, max_loc[1] + new_height),
                'confidence': max_val,
                'scale': scale
            }
    
    return best_match

def find_template_coordinates_noscale(source_path, target_path, threshold=0.8):
    """
    Find coordinates of source image in target image using template matching.
    Returns coordinates of best match or None if not found.
    """
    # Read images
    source = cv2.imread(source_path)
    target = cv2.imread(target_path)

    if source is None or target is None:
        print(f"Error reading images: {source_path} or {target_path}")
        return None

    sh, sw = source.shape[:2]
    source_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)

    # Template matching
    targetgray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    result = cv2.matchTemplate(targetgray, source_gray, cv2.TM_CCOEFF_NORMED)

    loc = np.where( result >= threshold)
    print(loc)
    for pt in zip(*loc[::-1]):
        data = {
            'top_left': pt,
            'bottom_right': (pt[0] + sw, pt[1] + sh),
            'confidence': result[pt[1]][pt[0]],
            'scale': 1.0
        }
        return data
    return None

def create_mask_from_coordinates(target_size, coordinates, padding=5):
    """
    Create a mask image with white rectangle at the specified coordinates.
    """
    mask = np.zeros(target_size[:2], dtype=np.uint8)
    
    if coordinates:
        top_left = coordinates['top_left']
        bottom_right = coordinates['bottom_right']
        
        # Add padding
        top_left_padded = (
            max(0, top_left[0] - padding),
            max(0, top_left[1] - padding)
        )
        bottom_right_padded = (
            min(target_size[1], bottom_right[0] + padding),
            min(target_size[0], bottom_right[1] + padding)
        )
        
        # Create white rectangle
        cv2.rectangle(
            mask, 
            top_left_padded, 
            bottom_right_padded, 
            255, 
            -1  # Filled rectangle
        )
    
    return Image.fromarray(mask)

def process_directory(source_image_path, directory_path, output_dir, threshold=0.8):
    """
    Process all images in directory to find source image and apply resynthesize.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Supported image extensions
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        file_ext = os.path.splitext(filename)[1].lower()
        
        if os.path.isfile(file_path) and file_ext in valid_extensions:
            print(f"Processing: {filename}")
            
            try:
                # Find coordinates of source image in target
                coordinates = find_template_coordinates(source_image_path, file_path, threshold)
                
                if coordinates:
                    print(f"  Found match with confidence: {coordinates['confidence']:.3f}")
                    
                    # Load target image for mask creation
                    target_img = cv2.imread(file_path)
                    
                    # Create mask
                    mask = create_mask_from_coordinates(target_img.shape, coordinates)
                    
                    # Convert target image to PIL format for resynthesize
                    target_pil = Image.open(file_path)
                    
                    # Apply resynthesize filter
                    result = resynthesize(target_pil, mask)
                    
                    # Save result
                    output_path = os.path.join(output_dir, f"{filename}")
                    result.save(output_path)
                    print(f"  Saved result to: {output_path}")
                   
                else:
                    print(f"  No match found in {filename}")
                    
            except Exception as e:
                print(f"  Error processing {filename}: {str(e)}")

def find_template_efficient(source_path, target_path, method='feature', min_matches=15):
    """
    Main function to find template using efficient methods.
    
    Args:
        method: 'feature' (recommended), 'pyramid', or 'keypoint'
    """
    if method == 'feature':
        return feature_based_matching(source_path, target_path, min_matches)
    elif method == 'pyramid':
        return pyramid_template_matching(source_path, target_path)
    elif method == 'keypoint':
        return efficient_keypoint_matching(source_path, target_path, min_matches)
    else:
        raise ValueError("Method must be 'feature', 'pyramid', or 'keypoint'")
    
def create_polygon_mask(target_size, corners):
    """
    Create mask from polygon corners (for feature-based matching).
    """
    mask = np.zeros(target_size[:2], dtype=np.uint8)
    corners = corners.astype(np.int32)
    cv2.fillPoly(mask, [corners], 255)
    return Image.fromarray(mask)

def create_rect_mask(target_size, coordinates, padding=5):
    """
    Create rectangular mask.
    """
    mask = np.zeros(target_size[:2], dtype=np.uint8)
    
    top_left = coordinates['top_left']
    bottom_right = coordinates['bottom_right']
    
    # Add padding
    top_left_padded = (
        max(0, top_left[0] - padding),
        max(0, top_left[1] - padding)
    )
    bottom_right_padded = (
        min(target_size[1], bottom_right[0] + padding),
        min(target_size[0], bottom_right[1] + padding)
    )
    
    cv2.rectangle(mask, top_left_padded, bottom_right_padded, 255, -1)
    return Image.fromarray(mask)

def process_directory_efficient(source_image_path, directory_path, output_dir, 
                               method='feature', min_matches=15):
    """
    Process directory using efficient template matching.
    """
    os.makedirs(output_dir, exist_ok=True)
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        file_ext = os.path.splitext(filename)[1].lower()
        
        if os.path.isfile(file_path) and file_ext in valid_extensions:
            print(f"Processing: {filename}")
            
            try:
                # Find template using efficient method
                result = find_template_efficient(source_image_path, file_path, method, min_matches)
                
                if result:
                    print(f"  Found match with confidence: {result.get('confidence', 0):.3f}")
                    print(f"  Matches: {result.get('matches_count', 'N/A')}")
                    
                    target_img = cv2.imread(file_path)
                    target_pil = Image.open(file_path)
                    
                    # Create appropriate mask
                    if 'corners' in result:
                        mask = create_polygon_mask(target_img.shape, result['corners'])
                    else:
                        mask = create_rect_mask(target_img.shape, result)
                    
                    # Apply resynthesize
                    output = resynthesize(target_pil, mask)
                    output_path = os.path.join(output_dir, f"result_{filename}")
                    output.save(output_path)
                    
                    print(f"  Saved to: {output_path}")
                else:
                    print(f"  No match found")
                    
            except Exception as e:
                print(f"  Error: {str(e)}")

def main():
    # Configuration
    source_image_path = 'logo.png'  # Your source PNG image
    directory_path = '/home/israel/my_win/Resources/Images/img/'  # Directory containing larger images
    # directory_path = './kw'
    output_directory = './output'
    confidence_threshold = 0.7  # Adjust based on your needs
    
    # Process all images in directory
    process_directory(
        source_image_path, 
        directory_path, 
        output_directory, 
        confidence_threshold
    )

    # process_directory_efficient(
    #     source_image_path=source_image_path,
    #     directory_path=directory_path,
    #     output_dir=output_directory,
    #     method='keypoint',  # Most efficient and robust
    #     min_matches=1
    # )

# Alternative function for single image processing
def process_single_image(source_path, target_path, output_path, threshold=0.8):
    """
    Process a single image pair.
    """
    # Find coordinates
    coordinates = find_template_coordinates(source_path, target_path, threshold)
    
    if not coordinates:
        print("No match found!")
        return
    
    print(f"Match found with confidence: {coordinates['confidence']:.3f}")
    
    # Load target image
    target_img = cv2.imread(target_path)
    
    # Create mask
    mask = create_mask_from_coordinates(target_img.shape, coordinates)
    
    # Convert to PIL and apply resynthesize
    target_pil = Image.open(target_path)
    result = resynthesize(target_pil, mask)
    
    # Save result
    result.save(output_path)
    print(f"Result saved to: {output_path}")

if __name__ == "__main__":
    main()
