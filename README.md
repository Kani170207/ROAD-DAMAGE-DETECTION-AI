# 🚧 Road Damage Detection Using Deep Learning

## Project Overview
Road infrastructure is critical for safe transportation, but road surfaces often deteriorate due to traffic, weather conditions, and poor maintenance. Manual inspection of road damage such as potholes and cracks is time-consuming and inefficient.

This project develops an AI-powered road damage detection system that automatically classifies road surface images into damage categories using deep learning. The system uses a MobileNetV2 transfer learning model to detect potholes, cracks, and manholes from images.

A web application built with Streamlit allows users to upload road images and receive real-time predictions along with confidence scores and Grad-CAM visualizations for explainable AI predictions.

---

## Features
- Real-time road damage classification
- Detection of potholes, cracks, and manholes
- Confidence score for predictions
- Grad-CAM heatmap visualization for explainability
- Streamlit web application for easy interaction
- Cloud deployment support

---

## Technologies Used
- Python
- TensorFlow / Keras
- MobileNetV2 (Transfer Learning)
- OpenCV
- NumPy
- Streamlit
- Grad-CAM

---

## Project Structure


road-damage-detection/
│
├── app/
│ └── app.py
│
├── dataset/
│ ├── train/
│ ├── val/
│ └── test/
│
├── models/
│ └── road_damage_model.h5
│
├── src/
│ ├── prepare_dataset.py
│ ├── preprocessing.py
│ └── train_model.py
│
├── requirements.txt
└── README.md


---

## Model
The project uses **MobileNetV2 transfer learning** to classify road damage images into three categories:

- Crack
- Manhole
- Pothole

The model is trained on a labeled road damage dataset and saved as:


models/road_damage_model.h5


---

## Installation

Clone the repository:


git clone https://github.com/yourusername/road-damage-detection.git


Navigate to the project directory:


cd road-damage-detection


Create a virtual environment:


python -m venv venv


Activate the environment:

Windows:

venv\Scripts\activate


Install dependencies:


pip install -r requirements.txt


---

## Run the Application

Start the Streamlit app:


python -m streamlit run app/app.py


Open the browser and go to:


http://localhost:8501


Upload a road image and the system will detect the damage type.

---

## Results

The system provides:

- Real-time road damage classification
- Prediction confidence score
- Grad-CAM heatmap visualization
- Web-based interface for image upload and analysis

---

## Future Improvements

- Integrate object detection for precise damage localization
- Deploy the system for mobile applications
- Integrate with smart city infrastructure monitoring systems
- Expand dataset for improved model accuracy

---

## Author
Kanish V
