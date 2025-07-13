from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
import io
from PIL import Image
from helper import eye_aspect_ratio, mouth_aspect_ratio
from head_pos import detect_head_tilt
import joblib

app = Flask(__name__)
model = joblib.load("Drowsiness-Model.pkl")

EAR_THRESHOLD = 0.25
MAR_THRESHOLD = 1.35

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [13, 14, 17, 18, 19, 20, 23, 24]

import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    data_url = request.json['image']
    header, encoded = data_url.split(",", 1)
    decoded = base64.b64decode(encoded)
    img = Image.open(io.BytesIO(decoded)).convert("RGB")
    frame = np.array(img)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                height, width, _ = frame.shape
                landmarks = [(int(lm.x * width), int(lm.y * height)) for lm in face_landmarks.landmark]

                left_eye = np.array([landmarks[i] for i in LEFT_EYE])
                right_eye = np.array([landmarks[i] for i in RIGHT_EYE])
                mouth = np.array([landmarks[i] for i in MOUTH])

                ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
                mar = mouth_aspect_ratio(mouth)
                is_tilted, tilt_angle = detect_head_tilt(landmarks)

                result = "Alert"
                if mar > MAR_THRESHOLD:
                    result = "Yawning"
                elif ear < EAR_THRESHOLD:
                    result = "Drowsy"
                elif is_tilted:
                    result = f"Head Tilt ({tilt_angle:.2f})"

                return jsonify({"status": result, "ear": round(ear, 2), "mar": round(mar, 2), "tilt": round(tilt_angle, 2)})
    return jsonify({"status": "No face detected", "ear": 0, "mar": 0, "tilt": 0})