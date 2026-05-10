import cv2
import numpy as np
import mediapipe as mp
import joblib

# Load trained model
model = joblib.load("activity_model.pkl")

# MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

# Input image
image_path = input("Enter image path: ")

image = cv2.imread(image_path)

if image is None:
    print("Could not read image")
    exit()

# Convert to RGB
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Pose detection
result = pose.process(rgb)

activity = "No Pose Detected"

if result.pose_landmarks:

    landmarks = []

    for lm in result.pose_landmarks.landmark:

        landmarks.extend([lm.x, lm.y, lm.z])

    X = np.array([landmarks])

    prediction = model.predict(X)

    activity = prediction[0]

    # Draw pose landmarks
    mp.solutions.drawing_utils.draw_landmarks(
        image,
        result.pose_landmarks,
        mp_pose.POSE_CONNECTIONS
    )

# Color based on prediction
color = (0, 255, 0)

if activity != "safe":
    color = (0, 0, 255)

# Display prediction
cv2.putText(
    image,
    f"Activity: {activity}",
    (20, 50),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    color,
    3
)

# Show image
cv2.imshow("Activity Detection", image)

cv2.waitKey(0)
cv2.destroyAllWindows()