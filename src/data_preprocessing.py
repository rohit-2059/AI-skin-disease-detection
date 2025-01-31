import os
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Set dataset paths
DATASET_DIR = "dataset/"
IMAGE_PARTS = ["HAM10000_images_part_1", "HAM10000_images_part_2"]
CSV_PATH = os.path.join(DATASET_DIR, "HAM10000_metadata.csv")

# Load metadata
df = pd.read_csv(CSV_PATH)

# Image Processing
IMG_SIZE = 128  # Resize images to 128x128

def load_images(df):
    images = []
    labels = []
    
    for img_name, label in zip(df['image_id'], df['dx']):
        img_path = None
        # Check both image folders
        for part in IMAGE_PARTS:
            part_path = os.path.join(DATASET_DIR, part, img_name + ".jpg")
            if os.path.exists(part_path):
                img_path = part_path
                break  # Stop once we find the image in one of the parts
        
        # If image doesn't exist in any part, skip it
        if img_path is None:
            print(f"Warning: Image {img_name}.jpg not found in any folder.")
            continue

        img = cv2.imread(img_path)
        
        # If image cannot be read, skip
        if img is None:
            print(f"Warning: Failed to read {img_path}.")
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # Resize image
        img = img / 255.0  # Normalize pixel values
        images.append(img)
        labels.append(label)

    return np.array(images), np.array(labels)

# Load dataset images
X, y = load_images(df)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Check if files are being saved
if not os.path.exists("dataset"):
    os.mkdir("dataset")

# Save processed data for model training
print("Saving preprocessed data...")
np.save("dataset/X_train.npy", X_train)
np.save("dataset/X_test.npy", X_test)
np.save("dataset/y_train.npy", y_train)
np.save("dataset/y_test.npy", y_test)
np.save("dataset/label_mapping.npy", label_encoder.classes_)

print("Data preprocessing complete. Files saved in 'dataset/'")
