import gradio as gr
import cv2
import dlib
import numpy as np
from PIL import Image
from helper import compute_ear, compute_mar
from head_pos import is_head_tilted

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def process(image):
    img_rgb = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
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
        tilted = is_head_tilted(nose)

        # Annotate
        for (x, y) in np.concatenate([left_eye, right_eye, mouth]):
            cv2.circle(img_rgb, (x, y), 2, (0, 255, 0), -1)

        if mar > 0.6:
            status = "Yawning"
        elif ear < 0.2:
            status = "Drowsy"
        elif tilted:
            status = "Head Tilted"
        else:
            status = "Alert"

        text = f"Status: {status} | EAR: {ear:.2f} | MAR: {mar:.2f}"
        cv2.putText(img_rgb, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        break
    else:
        status = "No face detected"
        cv2.putText(img_rgb, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

    return Image.fromarray(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB))

demo = gr.Interface(
    fn=process,
    inputs=gr.Image(type="pil", label="Upload or Take a Snapshot"),
    outputs=gr.Image(type="pil", label="Processed Output"),
    title="Driver Drowsiness Detection (EAR, MAR, Head Tilt)",
    description="Upload an image or take a webcam snapshot. Detects yawning, drowsiness, and head tilt using facial landmarks."
)

if __name__ == "__main__":
    demo.launch()