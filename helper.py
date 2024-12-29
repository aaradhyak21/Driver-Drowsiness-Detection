import cv2
import os
import numpy as np
import mediapipe as mp
import pandas as pd

mp_face_mesh = mp.solutions.face_mesh

# To compute EAR
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# To compute MAR
def mouth_aspect_ratio(mouth):
    A = np.linalg.norm(mouth[1] - mouth[5])
    B = np.linalg.norm(mouth[2] - mouth[4])
    C = np.linalg.norm(mouth[0] - mouth[3])
    mar = (A + B) / (2.0 * C)
    return mar

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [13, 14, 17, 18, 19, 20, 23, 24]

EAR_THRESHOLD = 0.25
MAR_THRESHOLD = 1.35

dataset_dir = 'Dataset/'

with mp_face_mesh.FaceMesh(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as face_mesh:
    
    features = []
    labels_list = []
    
    for img_name in os.listdir(dataset_dir):
        if img_name.endswith('.jpg'):
            img_path = os.path.join(dataset_dir, img_name)
            
            if not os.path.exists(img_path):
                print(f"Image not found: {img_path}")
                continue
            
            image = cv2.imread(img_path)
            
            if image is None:
                print(f"Failed to load image: {img_path}")
                continue
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image_rgb)
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    left_eye = np.array([(face_landmarks.landmark[i].x, face_landmarks.landmark[i].y) for i in LEFT_EYE])
                    right_eye = np.array([(face_landmarks.landmark[i].x, face_landmarks.landmark[i].y) for i in RIGHT_EYE])
                    
                    ear_left = eye_aspect_ratio(left_eye)
                    ear_right = eye_aspect_ratio(right_eye)
                    ear = (ear_left + ear_right) / 2
                    
                    mouth = np.array([(face_landmarks.landmark[i].x, face_landmarks.landmark[i].y) for i in MOUTH])
                    mar = mouth_aspect_ratio(mouth)
                    
                    features.append([ear, mar])
                    
                    if ear < EAR_THRESHOLD and mar > MAR_THRESHOLD:
                        labels_list.append(1)
                    else:
                        labels_list.append(0)
                        
    df = pd.DataFrame(features, columns=["EAR", "MAR"])
    df['Label'] = labels_list
    
    df.to_csv("Dataset.csv", index = False)
    
    print("Feature extraction is completed. Data saved to 'Dataset.csv'.")