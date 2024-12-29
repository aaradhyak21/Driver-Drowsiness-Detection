import cv2
import joblib
import winsound
import numpy as np
import mediapipe as mp
from helper import eye_aspect_ratio, mouth_aspect_ratio
from head_pos import detect_head_tilt

model = joblib.load("Drowsiness-Model.pkl")

mp_face_mesh = mp.solutions.face_mesh

EAR_THRESHOLD = 0.25
MAR_THRESHOLD = 1.4
TILT_THRESHOLD = 89.1
Total_Duration = 10

# Landmark indices for eyes and mouth
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [13, 14, 17, 18, 19, 20, 23, 24]

cam = cv2.VideoCapture(0)

ear_frame_counter = 0
mar_frame_counter = 0
tilt_counter = 0

with mp_face_mesh.FaceMesh(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as face_mesh:
    while cam.isOpened():
        ret, frame = cam.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
    
                left_eye = np.array([(face_landmarks.landmark[i].x, face_landmarks.landmark[i].y) for i in LEFT_EYE])
                right_eye = np.array([(face_landmarks.landmark[i].x, face_landmarks.landmark[i].y) for i in RIGHT_EYE])
               
                ear_left = eye_aspect_ratio(left_eye)
                ear_right = eye_aspect_ratio(right_eye)
                ear = (ear_left + ear_right) / 2
                
                mouth = np.array([(face_landmarks.landmark[i].x, face_landmarks.landmark[i].y) for i in MOUTH])
                mar = mouth_aspect_ratio(mouth)
                
                landmarks = [(lm.x * frame.shape[1], lm.y * frame.shape[0]) for lm in face_landmarks.landmark]
                is_tilted, tilt_angle = detect_head_tilt(landmarks)
                
                features = np.array([[ear, mar]])
                
                prediction = model.predict(features)[0]
                
                if mar < MAR_THRESHOLD:
                    mar_frame_counter += 1
                    if mar_frame_counter >= Total_Duration:
                        status = "Yawning"
                        winsound.Beep(1500, 500)
                    
                elif ear < EAR_THRESHOLD:
                    ear_frame_counter += 1
                    if ear_frame_counter >= Total_Duration:
                        status = "Drowsy"
                        winsound.Beep(1000, 500)
                
                elif tilt_angle < 0 or tilt_angle >= TILT_THRESHOLD:
                    tilt_counter += 1
                    if tilt_counter >= Total_Duration:
                        status = f"Head Tilt: {tilt_angle:.2f}"
                        winsound.Beep(2000, 500)
                    
                else:
                    ear_frame_counter, mar_frame_counter, tilt_counter = 0, 0, 0
                    status = "Alert"

                status_color = (0, 255, 0) if status == "Alert" else (0, 0, 255)
                
                cv2.putText(frame, f"Status: {status}", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
                
                cv2.putText(frame, f"EAR: {ear:.2f}", (50, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                cv2.putText(frame, f"MAR: {mar:.2f}", (50, 150), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                if is_tilted:
                    cv2.putText(frame, f"Tilt Angle: {tilt_angle:.2f}", (50, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (225, 225, 255), 2)
                
        cv2.imshow('Drowsiness Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
cam.release()
cv2.destroyAllWindows()