from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import dlib
import base64
import os
import urllib.request
import io
from PIL import Image
from helper import compute_ear, compute_mar
from head_pos import is_head_tilted

app = Flask(__name__)

PREDICTOR_PATH = "model/shape_predictor_68_face_landmarks.dat"
PREDICTOR_URL = "https://huggingface.co/spaces/bhadresh-savani/dlib-shape-predictor/resolve/main/shape_predictor_68_face_landmarks.dat"

def download_predictor():
    if not os.path.exists(PREDICTOR_PATH):
        os.makedirs(os.path.dirname(PREDICTOR_PATH), exist_ok=True)
        urllib.request.urlretrieve(PREDICTOR_URL, PREDICTOR_PATH)

download_predictor()

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

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
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    faces = detector(gray)
    for face in faces:
        shape = predictor(gray, face)
        landmarks = np.array([[p.x, p.y] for p in shape.parts()])
        left_eye = landmarks[36:42]
        right_eye = landmarks[42:48]
        mouth = landmarks[60:68]
        nose = landmarks[27:36]

        ear = (compute_ear(left_eye) + compute_ear(right_eye)) / 2.0
        mar = compute_mar(mouth)
        head_tilted = is_head_tilted(nose)

        result = "Alert"
        if mar > 0.6:
            result = "Yawning"
        elif ear < 0.2:
            result = "Drowsy"
        elif head_tilted:
            result = "Head Tilted"

        return jsonify({"status": result, "ear": round(ear, 2), "mar": round(mar, 2)})
    return jsonify({"status": "No face detected", "ear": 0, "mar": 0})