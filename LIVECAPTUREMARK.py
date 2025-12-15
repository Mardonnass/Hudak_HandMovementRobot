import cv2
import mediapipe as mp
import csv
import time

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# CSV HEADER: 63 values per hand × 2 hands + label
HEADER = ["label"] + [f"L_{axis}{i}" for i in range(21) for axis in ("x","y","z")] + \
                    [f"R_{axis}{i}" for i in range(21) for axis in ("x","y","z")]

CSV_PATH = "PrvyREAL+.csv"

# Make CSV if missing
try:
    with open(CSV_PATH, "x", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
except FileExistsError:
    pass

# Ask once before camera opens
label = input("Enter label for this recording (ex: forward, stop, left, right): ")
savedsamples = 0
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5
) as hands:

    print("\nPress SPACE to save one sample.")
    print("Press ESC to exit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Make camera bigger
        frame = cv2.resize(frame, (1280, 720))

        # Prepare RGB frame
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        # Prepare empty arrays (63 zeros per hand)
        left_hand = [0]*63
        right_hand = [0]*63

        if result.multi_hand_landmarks and result.multi_handedness:
            for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):

                # Fix flipping (MediaPipe is mirrored)
                label_hand = handedness.classification[0].label
                corrected = "Left" if label_hand == "Right" else "Right"

                # Extract flattened 21 × 3 = 63 values
                flat = []
                for lm in hand_landmarks.landmark:
                    flat.extend([lm.x, lm.y, lm.z])

                if corrected == "Left":
                    left_hand = flat
                elif corrected == "Right":
                    right_hand = flat

                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

        cv2.putText(frame, f"Label: {label}", (20, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        cv2.putText(frame, "Press SPACE to capture", (20, 280),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        
        cv2.putText(frame, "Press ESC to end", (20, 320),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        cv2.imshow("Hand Capture", frame)

        key = cv2.waitKey(10) & 0xFF

        if key == 27:  # ESC quits
            break

        if key == 32:  # SPACE saves sample
            row = [label] + left_hand + right_hand
            with open(CSV_PATH, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(row)
            
            savedsamples+=1
            print(f'Saved {savedsamples} sample!')

cap.release()
cv2.destroyAllWindows()