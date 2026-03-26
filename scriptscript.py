import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from rembg import remove
from PIL import Image

# ==========================================
# PHASE 1: AUTOMATED BACKGROUND REMOVAL
# ==========================================

raw_data_dir = 'path/to/your/raw_dataset'     # Where your original photos are
clean_data_dir = 'path/to/your/clean_dataset' # Where the script puts the clean ones

# Only run the heavy background removal if the clean folder doesn't exist yet
if not os.path.exists(clean_data_dir):
    print("Clean dataset not found. Starting background removal (this happens once)...")
    
    for root, dirs, files in os.walk(raw_data_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                input_path = os.path.join(root, file)
                
                # Recreate the folder structure for UNISUKI items
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

# 1. Prepare and Augment your Data (Pointing to the clean data)
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.25,     
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

train_generator = datagen.flow_from_directory(
    clean_data_dir, 
    target_size=(224, 224), 
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    clean_data_dir, 
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

# 3. Add Custom Layers
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
print("Training complete and UNISUKI model saved!\n")


# ==========================================
# PHASE 3: EVALUATION & REPORTING
# ==========================================

print("Generating Evaluation Metrics for your Panel...")

# 1. Create a specific test generator (Must not be shuffled!)
test_generator = datagen.flow_from_directory(
    clean_data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    shuffle=False 
)

# 2. Get predictions
print("Running final predictions on validation data...")
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

# 3. Print the Classification Report
print("\n" + "="*40)
print("       UNISUKI CLASSIFICATION REPORT")
print("="*40)
report = classification_report(true_classes, predicted_classes, target_names=class_labels)
print(report)

# 4. Generate and Plot the Confusion Matrix
print("\nGenerating Confusion Matrix visual...")
cm = confusion_matrix(true_classes, predicted_classes)

plt.figure(figsize=(10, 8)) 
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_labels, yticklabels=class_labels)

plt.title('UNISUKI Image Recognition - Confusion Matrix', fontsize=16)
plt.ylabel('Actual Category', fontsize=12)
plt.xlabel('Predicted Category', fontsize=12)
plt.xticks(rotation=45) 
plt.tight_layout()

# Save the matrix as a picture for your presentation
plt.savefig('unisuki_confusion_matrix.png', dpi=300)
print("Confusion matrix saved to your folder as 'unisuki_confusion_matrix.png'!")

plt.show()
