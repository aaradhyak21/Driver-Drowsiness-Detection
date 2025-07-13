from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
import io
from PIL import Image

app = Flask(__name__)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    data_url = request.json['image']
    _, encoded = data_url.split(",", 1)
    decoded = base64.b64decode(encoded)
    img = Image.open(io.BytesIO(decoded)).convert("RGB")
    frame = np.array(img)
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray)
        mouth = mouth_cascade.detectMultiScale(roi_gray, 1.7, 11)

        ear = 0.3 if len(eyes) >= 2 else 1.0
        mar = 0.7 if len(mouth) >= 1 else 0.0

        status = "Alert"
        if mar > 0.6:
            status = "Yawning"
        elif ear < 0.2:
            status = "Drowsy"

        return jsonify({"status": status, "ear": round(ear, 2), "mar": round(mar, 2)})

    return jsonify({"status": "No face detected", "ear": 0, "mar": 0})
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
