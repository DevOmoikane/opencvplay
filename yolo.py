import os
import glob
from ultralytics import YOLO

def process_images_with_yolo(model_path, images_directory):
    """
    Process all images in a directory using YOLO model and print results
    
    Args:
        model_path (str): Path to the custom YOLO .pt model file
        images_directory (str): Path to directory containing images
    """
    
    # Load the custom YOLO model
    model = YOLO(model_path)
    
    # Get all image files from the directory (common image extensions)
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']
    image_paths = []
    
    for extension in image_extensions:
        image_paths.extend(glob.glob(os.path.join(images_directory, extension)))
        image_paths.extend(glob.glob(os.path.join(images_directory, extension.upper())))
    
    if not image_paths:
        print(f"No images found in directory: {images_directory}")
        return
    
    print(f"Found {len(image_paths)} images to process")
    print("-" * 50)
    
    # Process each image
    for image_path in image_paths:
        print(f"\nProcessing: {os.path.basename(image_path)}")
        
        # Run inference on the image
        results = model(image_path)
        
        # Process results
        for i, result in enumerate(results):
            print(f"  Result {i+1}:")
            
            # Get class names
            class_names = result.names
            
            # Check if there are any detections
            if result.boxes is not None and len(result.boxes) > 0:
                # Get bounding boxes, confidence scores, and class IDs
                boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes in [x1, y1, x2, y2] format
                confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
                class_ids = result.boxes.cls.cpu().numpy().astype(int)  # Class IDs
                
                print(f"    Detections: {len(boxes)}")
                
                # Print details for each detection
                for j, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
                    class_name = class_names[cls_id]
                    print(f"      Detection {j+1}: {class_name} (confidence: {conf:.3f})")
                    print(f"        Bounding box: [{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]")
            else:
                print("    No detections found")

if __name__ == "__main__":
    # Configuration - Update these paths according to your setup
    MODEL_PATH = "yolov8s_aa11.pt"  # Replace with your model path
    IMAGES_DIRECTORY = "/home/israel/my_win/Resources/Images/anime"     # Replace with your images directory
    
    # Process all images
    process_images_with_yolo(MODEL_PATH, IMAGES_DIRECTORY)
