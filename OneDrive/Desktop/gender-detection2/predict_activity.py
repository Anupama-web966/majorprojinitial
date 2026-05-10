import cv2
import numpy as np
import mediapipe as mp
import joblib

# Load trained model
model = joblib.load("activity_model.pkl")

# MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

# Image path
image_path = input("Enter image path: ")

image = cv2.imread(image_path)

if image is None:
    print("Could not read image")
    exit()

rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

result = pose.process(rgb)

if result.pose_landmarks:

    landmarks = []

    for lm in result.pose_landmarks.landmark:

        landmarks.extend([lm.x, lm.y, lm.z])

    X = np.array([landmarks])

    prediction = model.predict(X)

    print("Predicted Activity:", prediction[0])

else:
    print("No human pose detected")