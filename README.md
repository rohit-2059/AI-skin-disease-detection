# DermaSense.ai — AI Skin Disease Detection

## Overview

DermaSense.ai helps users detect skin diseases using AI image analysis, making dermatology more accessible. Users upload skin images, and the system predicts possible diagnoses using a deep learning model trained on real-world dermatology datasets.

## Features

- **AI-powered diagnosis:** Upload a skin image and instantly receive a disease prediction.
- **Disease details:** See description, cause, treatment, and curability for each prediction.
- **SMS support:** Optionally receive results and recommendations via SMS using Twilio.
- **Modern UI:** Interactive web interface with 3D visualization (Three.js).
- **Sample results:** See `sample.pdf` for example outputs.

## Project Structure

- **Main app:** [`app.py`](app.py) — Flask server; handles image upload, prediction, SMS, and web interface.
- **Training scripts:** [`src/train_model.py`](src/train_model.py) — builds and trains the CNN.
- **Preprocessing:** [`src/data_preprocessing.py`](src/data_preprocessing.py) — loads, cleans, splits data.
- **Prediction:** [`src/predict.py`](src/predict.py) — loads model and makes predictions on new images.
- **Frontend:** [`templates/index.html`](templates/index.html), [`static/css/styles.css`](static/css/styles.css) — modern responsive UI with Three.js animation.
- **Resources:** `HAM10000_metadata.csv`, `hmnist_8_8_RGB.csv`, `hmnist_8_8_L.csv` — dataset files.

## Model Details

- **Preprocessing:** Images are loaded, resized to 128x128, normalized, and labels encoded.
- **Training:** CNN built with TensorFlow/Keras; see `src/train_model.py` for architecture.
- **Model file:** Saved as `dataset/skin_disease_model.h5`.
- **Prediction:** Uploaded image is preprocessed and passed to the model; returns most probable skin disease class.

## Disease Classes

- Melanoma
- Nevus
- Basal Cell Carcinoma
- Actinic Keratosis
- Squamous Cell Carcinoma
- Vascular conditions

Each class has associated medical details returned in the UI and SMS.

## Web Application Usage

1. **Run the app:**
   ```bash
   pip install -r requirement.txt
   python app.py
   ```
2. **Open browser:** Visit `http://localhost:5000`
3. **Upload image:** Choose a skin image (jpg/png), optionally enter phone number for SMS.
4. **View results:** See diagnosis plus information, treatment, and uploaded image.
5. **Sample output:** See [`sample.pdf`](sample.pdf)

## Dataset

- **Main:** [HAM10000 Skin Lesion Dataset](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
- **Download example:**
  ```python
  import kagglehub
  path = kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")
  print("Path to dataset files:", path)
  ```

## Requirements

See [`requirement.txt`](requirement.txt):
- tensorflow, numpy, pandas, matplotlib, seaborn, opencv-python, scikit-learn, flask, kaggle

## Contributing

1. Fork the repo, clone, and create a feature branch.
2. Open a pull request for improvements or new features.

## License

MIT License

## Contact

For questions, open an issue or reach out via [LinkedIn](https://www.linkedin.com/in/-rohit-khandelwal-).
