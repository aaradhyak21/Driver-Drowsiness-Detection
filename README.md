# Driver Drowsiness Detection System

This project aims to detect driver drowsiness based on eye, mouth, and head position. The model uses facial landmark points to determine if the driver is yawning, drowsy, or has a tilted head, and provides an alert accordingly.

## Features
- **Eye Aspect Ratio (EAR)**: Used to detect drowsiness by monitoring eye closure.
- **Mouth Aspect Ratio (MAR)**: Used to detect yawning.
- **Head Tilt Detection**: Measures the angle of head tilt to detect potential drowsiness or fatigue.

## Dataset
- https://drive.google.com/drive/folders/1hjdPerTAgYc2ONyXyTWdg-dXamZI0MXS

## Model
- The model was trained using scikit-learn and is saved as a **.pkl** file. The trained model is loaded during the execution to make predictions.

## How It Works
- **Face Detection:** The system detects faces using MediaPipe's FaceMesh model.
- **Landmark Calculation:** It calculates eye, mouth, and head landmarks.
- **Drowsiness Detection:** Based on the extracted features (EAR, MAR, and head tilt), the system determines the driver's status.
- **Alert System:** If the driver is detected as yawning, drowsy, or having a tilted head, an alert sound is played.

## Dependencies
- Python 3.x
- OpenCV
- MediaPipe
- scikit-learn
- joblib
- numpy
- winsound

## Acknowledgments
- **MediaPipe** for the face mesh model.
- **scikit-learn** for the model training.
- **Dataset source** Face 5,000 Images Dataset
