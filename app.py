from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
import tensorflow as tf
import cv2
import numpy as np
from twilio.rest import Client

app = Flask(__name__)

TWILIO_ACCOUNT_SID = "AC327d16ccdb54d5d440cda64e5f11b9d5"
TWILIO_AUTH_TOKEN = "6b64a8915760e3a4938fd68ce2ea0f65"
TWILIO_PHONE_NUMBER = "+18454489267"

# Model path and upload folder
MODEL_PATH = "dataset/skin_disease_model.h5"
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#
# Additional information for each disease
disease_info = {
    "Melanoma": {
        "description": "Melanoma is a form of skin cancer that develops from the pigment-producing cells known as melanocytes.",
        "cause": "It is often caused by UV radiation from the sun or tanning beds.",
        "treatment": "Treatment typically involves surgery to remove the melanoma, immunotherapy, or targeted therapies.",
        "cureable": "If detected early, melanoma can be treated and cured."
    },
    "Nevus": {
        "description": "A nevus is a mole or growth on the skin, which is usually benign.",
        "cause": "It can be genetic or due to sun exposure.",
        "treatment": "In most cases, no treatment is required unless it changes in appearance, in which case it may need to be removed.",
        "cureable": "Benign nevi are harmless and do not require a cure."
    },
    "Basal Cell Carcinoma": {
        "description": "Basal Cell Carcinoma is a type of skin cancer that arises from the basal cells in the skin's outer layer.",
        "cause": "It is primarily caused by prolonged exposure to UV rays.",
        "treatment": "Treatment includes surgical removal of the tumor or other therapies like cryotherapy or radiation.",
        "cureable": "It is highly treatable and rarely spreads to other parts of the body."
    },
    "Actinic Keratosis": {
        "description": "Actinic Keratosis is a precancerous growth that can develop on sun-damaged skin.",
        "cause": "It is caused by long-term exposure to UV radiation.",
        "treatment": "It can be treated with cryotherapy, topical creams, or laser therapy.",
        "cureable": "If caught early, it is treatable and can be cured."
    },
    "Squamous Cell Carcinoma": {
        "description": "Squamous Cell Carcinoma is a type of skin cancer that originates in the squamous cells.",
        "cause": "Prolonged sun exposure and UV radiation are common causes.",
        "treatment": "Treatment options include surgical removal, radiation therapy, or topical chemotherapy.",
        "cureable": "It is treatable, and most cases can be cured if detected early."
    },
    "Vascular": {
        "description": "Vascular skin conditions include various disorders related to blood vessels, such as spider veins or hemangiomas.",
        "cause": "Vascular conditions can be hereditary or caused by trauma, sun exposure, or aging.",
        "treatment": "Treatment options vary and include laser therapy, sclerotherapy, or surgery.",
        "cureable": "Treatment can help manage symptoms, but some conditions may not be fully curable."
    }
}

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Load the pre-trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Class labels (update with your actual labels)
class_labels = ["Melanoma", "Nevus", "Basal Cell Carcinoma", "Actinic Keratosis", "Squamous Cell Carcinoma", "Vascular"]

# Prediction function
def predict_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))  # Resize to match the model's input size
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = img / 255.0  # Normalize the image
    predictions = model.predict(img)
    predicted_class = class_labels[np.argmax(predictions)]
    return predicted_class

def send_sms(phone_number, prediction, details):
    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    
    message_body = f"Skin Disease Analysis Result:\n\n" \
                   f"Prediction: {prediction}\n" \
                   f"Description: {details['description']}\n" \
                   f"Cause: {details['cause']}\n" \
                   f"Treatment: {details['treatment']}\n" \
                   f"Curable: {details['cureable']}\n"

    message = client.messages.create(
        body=message_body,
        from_=TWILIO_PHONE_NUMBER,
        to=phone_number
    )

    print(f"SMS sent! SID: {message.sid}")
# Route for home page and form
@app.route('/')
def home():
    return render_template('index.html')

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return redirect(request.url)

    image = request.files['image']

    if image.filename == '':
        return redirect(request.url)

    if image:
        # Save the uploaded image
        filename = image.filename
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(image_path)

        # Make prediction
        prediction = predict_image(image_path)

        # Get the disease information from the dictionary
        disease_details = disease_info.get(prediction, {"description": "No information available.",
    "cause": "Unknown.",
    "treatment": "Consult a doctor.",
    "cureable": "Not determined."})
        phone_number = request.form.get("phone")
        if phone_number:
            send_sms(phone_number, prediction, disease_details)

        return render_template('index.html', 
                               prediction=prediction, 
                               filename=filename,
                               disease_details=disease_details)


if __name__ == '__main__':
    app.run(debug=True)
