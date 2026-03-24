import os
from PIL import Image, ImageEnhance

def process_image_for_ml(input_path, output_path, target_size=(224, 224)):
    """
    Resizes, enhances contrast, and pads the image to a square standard for ML.
    """
    try:
        # 1. Load the Image
        # We use RGB now because we don't need an Alpha (transparency) channel anymore
        img = Image.open(input_path).convert("RGB")

        # 1.5 Contrast Enhancement
        # Boost the contrast by 20% to help the ML model distinguish dark fabrics
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.2) 

        # 2. Rescaling (Resize while maintaining aspect ratio)
        # We shrink the image so its longest side fits the target size (e.g., 224px)
        img.thumbnail(target_size, Image.Resampling.LANCZOS)

        # 3. Padding (Create a uniform square background)
        # ML models need square images. Instead of stretching the item, we paste it 
        # in the center of a pure white square canvas.
        final_image = Image.new('RGB', target_size, (255, 255, 255))
        
        # Calculate the center position for the padding
        offset_x = (target_size[0] - img.size[0]) // 2
        offset_y = (target_size[1] - img.size[1]) // 2
        
        # Paste the resized image onto the white canvas
        final_image.paste(img, (offset_x, offset_y))

        # 4. Save the final processed image
        final_image.save(output_path, "JPEG", quality=95)
        print(f"Successfully processed: {os.path.basename(input_path)}")
        return True

    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return False

# ==========================================
# Run the pipeline on a folder of images
# ==========================================
if __name__ == "__main__":
    # Define your folders here
    INPUT_DIR = "UC PE Shirt Validation" 
    OUTPUT_DIR = "Processed UC PE Shirt Validation"

    # Create the output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Loop through all files in the input folder
    for filename in os.listdir(INPUT_DIR):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_file = os.path.join(INPUT_DIR, filename)
            
            # Force the output file to have a .jpg extension
            base_name = os.path.splitext(filename)[0]
            output_filename = f"{base_name}.jpg"
            output_file = os.path.join(OUTPUT_DIR, output_filename)
            
            # Run the preprocessing function
            process_image_for_ml(input_file, output_file)
            
    print("\nBatch preprocessing complete!")