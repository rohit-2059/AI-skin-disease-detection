import cv2
import tensorflow as tf
import numpy as np
import os

# Set image size and model path
IMG_SIZE = 128  # Resize images to 128x128
MODEL_PATH = "dataset/skin_disease_model.h5"
IMAGE_PARTS = ["HAM10000_images_part_1", "HAM10000_images_part_2"]

# Class labels for your dataset (update this list with the correct labels)
class_labels = [
    "Akne", "Basal cell carcinoma", "Dermatofibroma", "Melanocytic nevi", "Melanoma", 
    "Pigmented benign keratosis", "Squamous cell carcinoma"
]

# Load the pre-trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Function to make predictions
def predict_image(image_name):
    img_path = None
    
    # Check both image folders
    for part in IMAGE_PARTS:
        part_path = os.path.join("dataset", part, image_name)
        if os.path.exists(part_path):
            img_path = part_path
            break
    
    # If image is not found, return an error message
    if img_path is None:
        print(f"Error: Image {image_name} not found in any folder.")
        return None

    # Read the image
    img = cv2.imread(img_path)

    # Check if the image is loaded successfully
    if img is None:
        print(f"Error: Unable to load image at {img_path}")
        return None

    # Resize and normalize the image
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0  # Normalize pixel values

    # Prepare the image for prediction
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(img)
    return prediction

# Replace "ISIC_0027419.jpg" with the image name you want to predict
image_name = "ISIC_0029316.jpg" # Example: Use your own image name here

# Get prediction
prediction = predict_image(image_name)

# Display the result
if prediction is not None:
    # Get the index of the highest probability
    predicted_class_index = np.argmax(prediction)

    # Get the label of the predicted class
    predicted_class_label = class_labels[predicted_class_index]

    # Print prediction and label
    print(f"Prediction probabilities: {prediction}")
    print(f"Predicted class: {predicted_class_label}")
