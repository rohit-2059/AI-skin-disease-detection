# DermaSense.ai
Many people face difficulty in accessing dermatologists for basic skin check-ups due to high consultation fees, limited availability, or geographical constraints. This can lead to delayed diagnosis and worsening of skin conditions.

An AI-powered system that allows users to upload images of their skin conditions. The AI model, trained using deep learning techniques, will analyze the image and provide potential diagnoses, helping users take the next steps, such as consulting a specialist if necessary.

Tech Stack
Backend & AI Model: Python, TensorFlow
Frontend: HTML, CSS, JavaScript
3D Visualization: Three.js (for enhanced user experience)
Resources
Dataset: Dermatology dataset containing labeled skin disease images for training the AI model.
HAM10000 Dataset (A popular dataset for skin disease classification)

1. Data Collection & Preprocessing
Obtain a high-quality dermatology dataset.
Perform data cleaning (resizing, augmentation, balancing classes).

2. AI Model Development
Use TensorFlow/Keras to build and train a Convolutional Neural Network (CNN).
Fine-tune the model for better accuracy.

3. Used Twilio SMS API for message based support.

4. Check sample.pdf for sample images of the project. 

Code for Dataset:
import kagglehub
path = kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")
print("Path to dataset files:", path) 
                  OR
Download From: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
