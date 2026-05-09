import requests
import os
# --- PROTOBUF FIX: Must be at the very top ---
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import cv2
import numpy as np
import winsound 

frame_width = 1280
# --- TELEGRAM SETTINGS ---
BOT_TOKEN = "8424334136:AAFyp3bBumrAOwYhERdL4gYxxLXK0QZI_NY"
CHAT_ID = "976435954"

# --- 1. DYNAMIC MEDIAPIPE IMPORT ---
try:
    import mediapipe as mp
    try:
        mp_hands = mp.solutions.hands
        mp_draw = mp.solutions.drawing_utils
    except AttributeError:
        import mediapipe.python.solutions.hands as mp_hands
        import mediapipe.python.solutions.drawing_utils as mp_draw
except ImportError:
    print("CRITICAL: MediaPipe is still not installed.")
    exit()

# --- 2. SETTINGS & FILE PATHS ---
FACE_PROTO = "weights/deploy.prototxt"
FACE_MODEL = "weights/res10_300x300_ssd_iter_140000_fp16.caffemodel"
GENDER_MODEL = "weights/deploy_gender.prototxt"
GENDER_PROTO = "weights/gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
GENDER_LIST = ['Male', 'Female']

# --- 3. INITIALIZE MODELS ---
face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)
gender_net = cv2.dnn.readNetFromCaffe(GENDER_MODEL, GENDER_PROTO)

hands_detector = mp_hands.Hands(
    static_image_mode=False, 
    max_num_hands=2, 
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# --- 4. UTILITY & LOGIC FUNCTIONS ---
def send_telegram_alert():

    message = """
🚨 WOMEN SAFETY ALERT 🚨

Potential emergency detected.

✔ Female detected
✔ Help gesture detected

Immediate attention required.
"""

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

    data = {
        "chat_id": CHAT_ID,
        "text": message
    }

    requests.post(url, data=data)

def get_faces(frame):
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177.0, 123.0))
    face_net.setInput(blob)
    output = np.squeeze(face_net.forward())
    faces = []
    if len(output.shape) == 1: output = np.expand_dims(output, axis=0)
    for i in range(output.shape[0]):
        confidence = output[i, 2]
        if confidence > 0.5:
            box = output[i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            start_x, start_y, end_x, end_y = box.astype(int)
            faces.append((max(0, start_x), max(0, start_y), end_x, end_y))
    return faces

def detect_help_gesture(hand_landmarks):
    tips = [8, 12, 16, 20]
    knuckles = [5, 9, 13, 17]
    fingers_folded = all(hand_landmarks.landmark[t].y > hand_landmarks.landmark[k].y for t, k in zip(tips, knuckles))
    thumb_tip = hand_landmarks.landmark[4]
    k5 = hand_landmarks.landmark[5]
    k17 = hand_landmarks.landmark[17]
    thumb_tucked = min(k5.x, k17.x) < thumb_tip.x < max(k5.x, k17.x)
    return fingers_folded and thumb_tucked

def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    (h, w) = image.shape[:2]
    if width is None and height is None: return image
    r = width / float(w) if width else height / float(h)
    dim = (width, int(h * r)) if width else (int(w * r), height)
    return cv2.resize(image, dim, interpolation=inter)

# --- 5. IMAGE PREDICTION FUNCTION ---

def predict_gender(input_path: str):
    """Detects gender from image with features for baldness-aware detection"""
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Could not load image at {input_path}")
        return
    
    frame = img.copy()
    if frame.shape[1] > frame_width: 
        frame = image_resize(frame, width=frame_width)
    
    faces = get_faces(frame)
    padding = 20 # Expanded padding to catch facial bone structure
    
    for (sx, sy, ex, ey) in faces:
        face_img = frame[max(0, sy-padding):min(ey+padding, frame.shape[0]-1),
                         max(0, sx-padding):min(ex+padding, frame.shape[1]-1)]
        
        if face_img.size == 0: continue
        
        blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        gender_net.setInput(blob)
        gender = GENDER_LIST[gender_net.forward()[0].argmax()]
        
        color = (255, 0, 0) if gender == "Male" else (147, 20, 255)
        cv2.rectangle(frame, (sx, sy), (ex, ey), color, 2)
        cv2.putText(frame, gender, (sx, sy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Image Predictor - Press any key to close", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# --- 6. LIVE PREDICTION FUNCTION ---

def predict_live():
    cap = cv2.VideoCapture(0)
    padding = 25 # Higher padding for live video to maintain accuracy during movement

    while True:
        ret, frame = cap.read()
        if not ret: break

        woman_present = False
        
        faces = get_faces(frame)
        for (sx, sy, ex, ey) in faces:
            face_crop = frame[max(0, sy-padding):min(ey+padding, frame.shape[0]-1),
                              max(0, sx-padding):min(ex+padding, frame.shape[1]-1)]
            
            if face_crop.size == 0: continue

            blob = cv2.dnn.blobFromImage(face_crop, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            gender_net.setInput(blob)
            gender = GENDER_LIST[gender_net.forward()[0].argmax()]
            
            if gender == "Female":
                woman_present = True
            
            color = (255, 0, 0) if gender == "Male" else (147, 20, 255)
            cv2.rectangle(frame, (sx, sy), (ex, ey), color, 2)
            cv2.putText(frame, f"{gender}", (sx, sy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands_detector.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
                
                if detect_help_gesture(hand_lms) and woman_present:
                    # Red header bar
                    cv2.rectangle(frame, (0,0), (frame.shape[1], 80), (0,0,255), -1)
                    cv2.putText(frame, "HELP ALERT!", (frame.shape[1]//3, 55), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 4)
                    winsound.Beep(1000, 100)
                    send_telegram_alert()

        cv2.imshow('Detection System - Press Q to Quit', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

# --- 7. MAIN SWITCH ---

if __name__ == "__main__":
    #for live 
    predict_live()
    #testing the sequence of the images
    predict_gender("test.jpg")
    predict_gender("test2.png")
    predict_gender("test3.png")
    predict_gender("test4.png")
    predict_gender("test5.png")
    predict_gender("test6.png")
    predict_gender("test7.png")
    predict_gender("test8.png")
    predict_gender("test9.jpeg")
