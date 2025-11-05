import os
import glob
from PIL import Image, ImageStat
import argparse

def get_average_border_color(image, border_width=10):
    """
    Calculate the average color of the image borders
    """
    width, height = image.size
    
    # Get border regions
    left_border = image.crop((0, 0, border_width, height))
    right_border = image.crop((width - border_width, 0, width, height))
    top_border = image.crop((0, 0, width, border_width))
    bottom_border = image.crop((0, height - border_width, width, height))
    
    # Calculate average color from all borders
    borders = [left_border, right_border, top_border, bottom_border]
    total_r, total_g, total_b = 0, 0, 0
    total_pixels = 0
    
    for border in borders:
        stat = ImageStat.Stat(border)
        total_r += stat.mean[0] * border.width * border.height
        total_g += stat.mean[1] * border.width * border.height
        total_b += stat.mean[2] * border.width * border.height
        total_pixels += border.width * border.height
    
    avg_r = int(total_r / total_pixels)
    avg_g = int(total_g / total_pixels)
    avg_b = int(total_b / total_pixels)
    
    return (avg_r, avg_g, avg_b)

def resize_with_padding(image, target_size, enable_filling=True):
    """
    Resize image maintaining aspect ratio and optionally add padding
    """
    original_width, original_height = image.size
    target_width, target_height = target_size
    
    if enable_filling:
        # Calculate scaling factor to fit within target dimensions
        scale_width = target_width / original_width
        scale_height = target_height / original_height
        scale_factor = min(scale_width, scale_height)
        
        # Calculate new dimensions
        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)
        
        # Resize image
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Create new square image with padding
        padded_image = Image.new('RGB', (target_width, target_height))
        
        # Get average border color for padding
        padding_color = get_average_border_color(resized_image)
        
        # Fill background with average border color
        padded_image.paste(padding_color, [0, 0, target_width, target_height])
        
        # Calculate position to center the resized image
        x_offset = (target_width - new_width) // 2
        y_offset = (target_height - new_height) // 2
        
        # Paste resized image onto centered position
        padded_image.paste(resized_image, (x_offset, y_offset))
        
        return padded_image
    else:
        # Without filling - simple resize to fit within target dimensions
        scale_width = target_width / original_width
        scale_height = target_height / original_height
        scale_factor = min(scale_width, scale_height)
        
        # Calculate new dimensions
        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)
        
        # Resize image
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        return resized_image

def process_directory(input_dir, output_dir, target_size, enable_filling=True):
    """
    Process all images in input directory and save resized versions to output directory
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Supported image formats
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp', '*.JPG', '*.JPEG', '*.PNG']
    image_paths = []
    
    # Collect all image files
    for extension in image_extensions:
        image_paths.extend(glob.glob(os.path.join(input_dir, extension)))
    
    if not image_paths:
        print(f"No images found in directory: {input_dir}")
        return
    
    print(f"Found {len(image_paths)} images to process")
    print(f"Target size: {target_size}x{target_size}")
    print(f"Filling enabled: {enable_filling}")
    print(f"Output directory: {output_dir}")
    print("-" * 50)
    
    processed_count = 0
    
    for image_path in image_paths:
        try:
            filename = os.path.basename(image_path)
            name, ext = os.path.splitext(filename)
            
            # Create output filename
            if enable_filling:
                output_filename = f"{name}_resized{ext}"
            else:
                output_filename = f"{name}_resized_no_fill{ext}"
            
            output_path = os.path.join(output_dir, output_filename)
            
            # Skip if output file already exists
            if os.path.exists(output_path):
                print(f"⚠️  Skipping {filename} (already exists)")
                continue
            
            # Open and process image
            with Image.open(image_path) as img:
                # Convert to RGB if necessary (for PNG with transparency)
                if img.mode in ('RGBA', 'LA', 'P'):
                    rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                    rgb_img.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                    img = rgb_img
                
                original_size = img.size
                print(f"Processing: {filename} ({original_size[0]}x{original_size[1]})")
                
                # Resize with or without padding
                resized_img = resize_with_padding(img, (target_size, target_size), enable_filling)
                
                # Save processed image
                if enable_filling:
                    resized_img.save(output_path, quality=95)
                    final_size = f"{target_size}x{target_size}"
                else:
                    resized_img.save(output_path, quality=95)
                    final_size = f"{resized_img.size[0]}x{resized_img.size[1]}"
                
                processed_count += 1
                
                print(f"  ✅ Resized to: {final_size}")
                
        except Exception as e:
            print(f"❌ Error processing {filename}: {str(e)}")
    
    print("-" * 50)
    print(f"Processing complete! {processed_count} images resized and saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Resize images with optional padding while maintaining aspect ratio')
    parser.add_argument('--input', '-i', required=True, help='Input directory containing images')
    parser.add_argument('--output', '-o', required=True, help='Output directory for resized images')
    parser.add_argument('--size', '-s', type=int, required=True, help='Target size (width and height)')
    parser.add_argument('--fill', '-f', action='store_true', help='Enable filling/padding to maintain square aspect ratio')
    parser.add_argument('--no-fill', '-n', action='store_true', help='Disable filling/padding (maintain original aspect ratio)')
    
    args = parser.parse_args()
    
    # Validate input directory exists
    if not os.path.exists(args.input):
        print(f"Error: Input directory '{args.input}' does not exist")
        return
    
    # Validate target size
    if args.size <= 0:
        print("Error: Target size must be positive")
        return
    
    # Determine filling mode
    if args.no_fill:
        enable_filling = False
    else:
        enable_filling = True  # Default to enabled
    
    process_directory(args.input, args.output, args.size, enable_filling)

# Alternative: Direct usage without command line arguments
if __name__ == "__main__":
    # You can use the script in two ways:
    
    # Method 1: Using command line arguments
    # With filling: python script.py --input ./images --output ./resized --size 640 --fill
    # Without filling: python script.py --input ./images --output ./resized --size 640 --no-fill
    main()
    
    # Method 2: Direct configuration (uncomment below and comment main() call above)
    """
    INPUT_DIRECTORY = "./images"      # Replace with your input directory
    OUTPUT_DIRECTORY = "./resized"    # Replace with your output directory
    TARGET_SIZE = 640                 # Replace with your desired size
    ENABLE_FILLING = False            # Set to True for padding, False for no padding
    
    process_directory(INPUT_DIRECTORY, OUTPUT_DIRECTORY, TARGET_SIZE, ENABLE_FILLING)
    """