import os
import glob
import click
from ultralytics import YOLO
from rich import print
import cv2

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
        
        # Load the image for drawing
        image = cv2.imread(image_path)
        if image is None:
            print(f"  Error: Could not load image {image_path}")
            continue
            
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
                
                # Print details and draw rectangles for each detection
                for j, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
                    # Convert coordinates to integers
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Draw rectangle on the image
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Add label with class name and confidence
                    label = f"{class_names[cls_id]} {conf:.2f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    
                    # Draw background for text
                    cv2.rectangle(image, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), (0, 255, 0), -1)
                    # Draw text
                    cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                    
                    print(f"      Detection {j+1}: {class_names[cls_id]} (confidence: {conf:.3f})")
                    print(f"        Bounding box: [{x1}, {y1}, {x2}, {y2}]")
            else:
                print("    No detections found")
        
        # Display the image with detections
        window_name = f"YOLO Detection - {os.path.basename(image_path)}"
        cv2.imshow(window_name, image)
        print(f"  Press any key to close the image and continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

@click.command()
@click.option("--model-path", default="yolov8s_aa11.pt", help="Path to the custom YOLO .pt model file")
@click.option("--images-directory", default="./images", help="Path to directory containing images")
def main(model_path, images_directory):
    process_images_with_yolo(model_path, images_directory)

if __name__ == "__main__":
    main()
