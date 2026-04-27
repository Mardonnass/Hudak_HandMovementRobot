import cv2
import mediapipe as mp
import pickle
import numpy as np
import serial
import time
from tensorflow.keras.models import load_model

# ---------------------------
# Load trained model
# ---------------------------
model = load_model("MODEL2.keras")

with open("ENKODIK.pkl", "rb") as f:
    le = pickle.load(f)

# ---------------------------
#SERIAL SETUP
# ---------------------------
ser = serial.Serial("COM5", 9600, timeout=1)
time.sleep(2)

# ---------------------------
# Mediapipe setup
# ---------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ---------------------------
# Rectangle settings (same as FIRST CODE)
# ---------------------------
RECT_W = 400
RECT_H = 400

# ---------------------------
# Function to extract landmarks
# ---------------------------
def extract_landmarks(hand_landmarks):
    coords = []
    for lm in hand_landmarks.landmark:
        coords.extend([lm.x, lm.y, lm.z])
    return coords

# ---------------------------
# OpenCV camera
# ---------------------------
cap = cv2.VideoCapture(0)

last_gesture = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1280, 720))
    h, w, _ = frame.shape

    x1 = w // 2 - RECT_W // 2
    y1 = h // 2 - RECT_H // 2
    x2 = w // 2 + RECT_W // 2
    y2 = h // 2 + RECT_H // 2

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    hand_features = [np.zeros(63), np.zeros(63)]

    if result.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(
                result.multi_hand_landmarks,
                result.multi_handedness):

            idx = 0 if handedness.classification[0].label == "Left" else 1
            hand_features[idx] = np.array(extract_landmarks(hand_landmarks))

            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    features = np.concatenate(hand_features).reshape(1, -1)

    prediction = model.predict(features, verbose=0)
    max_prob = np.max(prediction)
    class_id = np.argmax(prediction)

    if max_prob < 0.4 or (result.multi_hand_landmarks is None):
        gesture = "None"
    else:
        gesture = le.inverse_transform([class_id])[0]


    if gesture != last_gesture:
        if gesture == "FORWARD":
            ser.write(b"F\n")
        elif gesture == "BACKWARD":
            ser.write(b"B\n")
        elif gesture == "Left":
            ser.write(b"L\n")
        elif gesture == "Right":
            ser.write(b"R\n")
        else:
            ser.write(b"S\n")

        print("Sent:", gesture)
        last_gesture = gesture

    # ✅ DISPLAY (INSIDE LOOP)
    cv2.putText(frame, f"Gesture: {gesture}", (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Hand Gesture Test", frame)

    # ✅ REQUIRED
    if cv2.waitKey(1) & 0xFF == 27:
        break

# cleanup
cap.release()
ser.close()
cv2.destroyAllWindows()