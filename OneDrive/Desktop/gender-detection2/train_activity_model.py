import os
import cv2
import numpy as np
import mediapipe as mp
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import joblib

# Dataset path
DATASET_PATH = "activity_dataset"

# MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

X = []
y = []

# Read dataset folders
for label in os.listdir(DATASET_PATH):

    label_path = os.path.join(DATASET_PATH, label)

    if not os.path.isdir(label_path):
        continue

    print(f"Processing: {label}")

    for image_name in os.listdir(label_path):

        image_path = os.path.join(label_path, image_name)

        image = cv2.imread(image_path)

        if image is None:
            continue

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        result = pose.process(rgb)

        if result.pose_landmarks:

            landmarks = []

            for lm in result.pose_landmarks.landmark:

                landmarks.extend([lm.x, lm.y, lm.z])

            X.append(landmarks)
            y.append(label)

# Convert to numpy
X = np.array(X)
y = np.array(y)

print("Dataset Loaded")
print("Samples:", len(X))

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    max_iter=500
)

model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

# Save model
joblib.dump(model, "activity_model.pkl")

print("Model Saved Successfully")