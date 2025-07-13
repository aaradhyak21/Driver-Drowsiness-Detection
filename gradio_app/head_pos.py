import cv2
import mediapipe as mp
import numpy as np
from math import atan2, degrees

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

def calculate_head_tilt_angle(landmarks):
    """
    Calculate the head tilt angle using MediaPipe landmarks.
    """
    # Extract relevant points
    nose_tip = landmarks[1]  # Nose tip
    left_eye = landmarks[33]  # Left eye outer corner
    right_eye = landmarks[263]  # Right eye outer corner
    
    # Calculate the midpoint of the eyes
    mid_eye_x = (left_eye[0] + right_eye[0]) / 2
    mid_eye_y = (left_eye[1] + right_eye[1]) / 2
    
    # Calculate the tilt angle
    delta_y = nose_tip[1] - mid_eye_y
    
    angle = degrees(atan2(delta_y, 1))
    return angle

def detect_head_tilt(landmarks):
    """
    Detect if head tilt exceeds a threshold angle.
    """
    angle = calculate_head_tilt_angle(landmarks)
    THRESHOLD = 15  # Define the threshold for head tilt in degrees
    
    if abs(angle) > THRESHOLD:
        return True, angle  # Head tilt detected
    return False, angle  # Normal posture
