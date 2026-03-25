import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from rembg import remove
from PIL import Image

# ==========================================
# PHASE 1: AUTOMATED BACKGROUND REMOVAL
# ==========================================

# 1. Define your folders
raw_data_dir = 'path/to/your/raw_dataset'     # Where your original photos are
clean_data_dir = 'path/to/your/clean_dataset' # Where the script should put the clean ones

# Only run the heavy background removal if the clean folder doesn't exist yet
if not os.path.exists(clean_data_dir):
    print("Clean dataset not found. Starting background removal (this happens once)...")
    
    for root, dirs, files in os.walk(raw_data_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                input_path = os.path.join(root, file)
                
                # Recreate the folder structure (e.g., 'White Polo', 'Black Pants')
                relative_path = os.path.relpath(root, raw_data_dir)
                target_folder = os.path.join(clean_data_dir, relative_path)
                os.makedirs(target_folder, exist_ok=True)
                
                output_path = os.path.join(target_folder, f"clean_{file}")
                
                try:
                    original_img = Image.open(input_path)
                    transparent_img = remove(original_img)
                    
                    # MobileNetV2 needs solid backgrounds, so we make it white
                    white_bg = Image.new("RGB", transparent_img.size, (255, 255, 255))
                    white_bg.paste(transparent_img, mask=transparent_img.split()[3])
                    white_bg.save(output_path, "JPEG")
                    print(f"Cleaned: {file}")
                except Exception as e:
                    print(f"Error processing {file}: {e}")
    print("Background removal complete!\n")
else:
    print("Clean dataset already exists! Skipping background removal phase.\n")


# ==========================================
# PHASE 2: MODEL TRAINING
# ==========================================

print("Starting model setup and training...")

# 1. Prepare and Augment your Data (Now pointing to the clean data!)
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.25,     
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

# Load Training Data
train_generator = datagen.flow_from_directory(
    clean_data_dir, # <-- Notice we are using the clean directory here
    target_size=(224, 224), 
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

# Load Validation Data
validation_generator = datagen.flow_from_directory(
    clean_data_dir, # <-- And here
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# 2. Load the MobileNetV2 Base Model
base_model = MobileNetV2(
    weights='imagenet', 
    include_top=False,  
    input_shape=(224, 224, 3)
)
base_model.trainable = False

# 3. Add Custom Layers for Your Specific Items
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)

num_classes = train_generator.num_classes 
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# 4. Compile the Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 5. Early Stopping
early_stopping = EarlyStopping(
    monitor='val_loss',       
    patience=5,               
    restore_best_weights=True 
)

# 6. Train the Model
print("Starting training epochs...")
history = model.fit(
    train_generator,
    epochs=50, 
    validation_data=validation_generator,
    callbacks=[early_stopping]
)

# 7. Save the Model
model.save('unisuki_mobilenet_model.h5')
print("Training complete and UNISUKI model saved!")
