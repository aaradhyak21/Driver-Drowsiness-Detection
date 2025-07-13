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

        ear = 1.0
        mar = 0.0

        if len(eyes) >= 2:
            # Take the two most confident eye detections
            eyes = sorted(eyes, key=lambda e: e[2]*e[3], reverse=True)[:2]
            eye_h = np.mean([eh for (_, _, _, eh) in eyes])
            eye_w = np.mean([ew for (_, ew, _, _) in eyes])
            ear = eye_h / eye_w if eye_w else 1.0

        if len(mouth) >= 1:
            mx, my, mw, mh = mouth[0]
            mar = mh / mw if mw else 0.0

        status = "Alert"
        if mar > 0.6:
            status = "Yawning"
        elif ear < 0.2:
            status = "Drowsy"

        return jsonify({"status": status, "ear": round(ear, 2), "mar": round(mar, 2)})

    return jsonify({"status": "No face detected", "ear": 0, "mar": 0})

# Render-compatible run
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
