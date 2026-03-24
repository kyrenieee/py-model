import albumentations as A
import cv2
import os
import glob

# Set your folder paths
INPUT_DIR = "Processed UC Junior High School White Polo Validation"
OUTPUT_DIR = "Augmented Processed UC Junior High School White Polo Validation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

images = glob.glob(os.path.join(INPUT_DIR, "*.jpg"))

print(f"Generating 5 escalating levels of marketplace augmentation for {len(images)} images...\n")

for img_path in images:
    image = cv2.imread(img_path)
    if image is None: 
        continue
    
    # OpenCV loads images in BGR format, Albumentations expects RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    filename = os.path.basename(img_path).split('.')[0]

    # Generate 5 levels, scaling up to your requested marketplace maximums
    for level in range(1, 6):
        
        # We divide by 5.0 so that Level 5 exactly equals your maximum requested values
        multiplier = level / 5.0 

        transform = A.Compose([
            # 1. Horizontal Flipping (Static 50% chance across all levels)
            A.HorizontalFlip(p=0.5),
            
            # 2. Rotation, 3. Shift, and 4. Zoom (Scale)
            A.ShiftScaleRotate(
                shift_limit=0.10 * multiplier,   # Max 10% shift
                scale_limit=0.10 * multiplier,   # Max 10% zoom in/out
                rotate_limit=20 * multiplier,    # Max 20 degree rotation
                border_mode=cv2.BORDER_REPLICATE, # CRITICAL: Replicates white tiles instead of adding black borders
                p=1.0
            ),
            
            # 5. Brightness & Contrast (The "Dark Fabric" Fixer)
            # Keras [0.8, 1.2] is equivalent to a 0.20 limit in Albumentations
            A.RandomBrightnessContrast(
                brightness_limit=0.20 * multiplier, 
                contrast_limit=0.20 * multiplier, 
                p=1.0
            ),
            
            # Ensure the final output remains locked at the 224x224 ML standard
            A.Resize(224, 224)
        ])

        # Apply the transformation
        augmented = transform(image=image)["image"]
        
        # Save the file with the level indicator
        save_name = f"{filename}_aug_lv{level}.jpg"
        cv2.imwrite(
            os.path.join(OUTPUT_DIR, save_name), 
            cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR) # Convert back to BGR for OpenCV to save properly
        )

print("="*50)
print(f"Done! Successfully created {len(images) * 5} augmented images.")
print("="*50)