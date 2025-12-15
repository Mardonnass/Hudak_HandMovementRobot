import cv2
import mediapipe as mp
import pickle
import numpy as np
from tensorflow.keras.models import load_model

# ---------------------------
# Load trained model and label encoder
# ---------------------------
model = load_model("MojKlasfir.h5")
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

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
# Function to extract landmarks (x, y, z)
# ---------------------------
def extract_landmarks(hand_landmarks):
    coords = []
    for lm in hand_landmarks.landmark:
        coords.extend([lm.x, lm.y, lm.z])
    return coords

# ---------------------------
# OpenCV live loop
# ---------------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    # Initialize 2x63 = 126 features for left and right hands
    hand_features = [np.zeros(63), np.zeros(63)]  # [Left, Right]

    if result.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
            idx = 0 if handedness.classification[0].label == "Left" else 1
            hand_features[idx] = np.array(extract_landmarks(hand_landmarks))
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Flatten both hands into single array (126 features)
    features = np.concatenate(hand_features)

    # Make prediction (reshape to (1, 126))
    features = features.reshape(1, -1)
    prediction = model.predict(features, verbose=0)
    max_prob = np.max(prediction)
    class_id = np.argmax(prediction)

    # If confidence is too low OR no hands detected
    if max_prob < 0.6 or (result.multi_hand_landmarks is None):
        gesture = "None"
    else:
        gesture = le.inverse_transform([class_id])[0]

    # Display gesture on frame
    cv2.putText(frame, f"Gesture: {gesture}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Hand Gesture Test", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
