import cv2
import numpy as np
import mediapipe as mp
import speech_recognition as sr
import joblib
import requests
import time

# =========================
# TELEGRAM CONFIG
# =========================

BOT_TOKEN = "8424334136:AAFyp3bBumrAOwYhERdL4gYxxLXK0QZI_NY"
CHAT_ID = "976435954"

# =========================
# LOAD ACTIVITY MODEL
# =========================

model = joblib.load("activity_model.pkl")

# =========================
# MEDIAPIPE SETUP
# =========================

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# =========================
# SPEECH RECOGNITION
# =========================

recognizer = sr.Recognizer()
mic = sr.Microphone()

ALERT_WORDS = [
    "help",
    "save me",
    "leave me",
    "stop",
    "don't touch me",
    "harassment"
]

# =========================
# TELEGRAM ALERT FUNCTION
# =========================

def send_telegram_alert(message):

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

    data = {
        "chat_id": CHAT_ID,
        "text": message
    }

    requests.post(url, data=data)

# =========================
# START VOICE LISTENING
# =========================

print("🎤 Listening for emergency words...")

while True:

    with mic as source:

        recognizer.adjust_for_ambient_noise(source)

        print("Listening...")

        audio = recognizer.listen(source)

    try:

        text = recognizer.recognize_google(audio)
        text = text.lower()

        print("You said:", text)

        triggered = False

        for word in ALERT_WORDS:

            if word in text:

                print("🚨 ALERT WORD DETECTED:", word)
                triggered = True
                break

        # =========================
        # START CAMERA IF TRIGGERED
        # =========================

        if triggered:

            cap = cv2.VideoCapture(0)

            print("📷 Camera Started")

            while True:

                ret, frame = cap.read()

                if not ret:
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                result = pose.process(rgb)

                activity = "safe"

                if result.pose_landmarks:

                    landmarks = []

                    for lm in result.pose_landmarks.landmark:

                        landmarks.extend([lm.x, lm.y, lm.z])

                    X = np.array([landmarks])

                    prediction = model.predict(X)

                    activity = prediction[0]

                    # Draw pose landmarks
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame,
                        result.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS
                    )

                # Show prediction
                cv2.putText(
                    frame,
                    f"Activity: {activity}",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )

                cv2.imshow("Smart Surveillance", frame)

                # =========================
                # SEND ALERT IF UNSAFE
                # =========================

                if activity != "safe":

                    print("🚨 UNSAFE ACTIVITY DETECTED")

                    send_telegram_alert(
                        f"🚨 ALERT!\nUnsafe activity detected: {activity}"
                    )

                    time.sleep(5)

                # Press Q to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()

    except sr.UnknownValueError:

        print("Could not understand")

    except sr.RequestError:

        print("API unavailable")