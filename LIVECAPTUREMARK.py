import cv2
import mediapipe as mp
import csv
import time

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

HEADER = ["label"] + [f"L_{axis}{i}" for i in range(21) for axis in ("x","y","z")] + \
                    [f"R_{axis}{i}" for i in range(21) for axis in ("x","y","z")]

CSV_PATH = "Try1.csv"

# Create CSV if not exists
try:
    with open(CSV_PATH, "x", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
except FileExistsError:
    pass

label = input("Enter label (FORWARD, BACKWARD, LEFT...): ")
savedsamples = 0

cap = cv2.VideoCapture(0)

RECT_W = 400
RECT_H = 400

# ⏱ cooldown to prevent duplicate frames
last_capture_time = 0
COOLDOWN = 0.5  # seconds

with mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5
) as hands:

    print("\nPress SPACE to save sample")
    print("Press ESC to exit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (1280, 720))
        h, w, _ = frame.shape

        # Rectangle
        x1 = w // 2 - RECT_W // 2
        y1 = h // 2 - RECT_H // 2
        x2 = w // 2 + RECT_W // 2
        y2 = h // 2 + RECT_H // 2

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        left_hand = [0] * 63
        right_hand = [0] * 63

        hand_inside = False  # ✅ check if hand is inside rectangle

        if result.multi_hand_landmarks and result.multi_handedness:
            for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):

                label_hand = handedness.classification[0].label
                corrected = "Left" if label_hand == "Right" else "Right"

                flat = []
                xs, ys = [], []

                for lm in hand_landmarks.landmark:
                    flat.extend([lm.x, lm.y, lm.z])

                    # convert to pixel coords
                    px = int(lm.x * w)
                    py = int(lm.y * h)

                    xs.append(px)
                    ys.append(py)

                # ✅ Check if hand is inside rectangle (bounding box)
                if min(xs) > x1 and max(xs) < x2 and min(ys) > y1 and max(ys) < y2:
                    hand_inside = True

                if corrected == "Left":
                    left_hand = flat
                elif corrected == "Right":
                    right_hand = flat

                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

        # UI text
        cv2.putText(frame, f"Label: {label}", (20, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.putText(frame, "Press SPACE to capture", (20, 280),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.putText(frame, "Press ESC to end", (20, 320),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if not hand_inside:
            cv2.putText(frame, "Hand NOT inside box", (20, 360),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Hand Capture", frame)

        key = cv2.waitKey(10) & 0xFF

        if key == 27:
            break

        if key == 32:
            current_time = time.time()

            # ✅ cooldown check
            if current_time - last_capture_time < COOLDOWN:
                print("Too fast! Wait a bit...")
                continue

            # ✅ must have hand inside rectangle
            if not hand_inside:
                print("Hand not inside rectangle!")
                continue

            # ✅ must detect at least one hand
            if result.multi_hand_landmarks is None:
                print("No hand detected!")
                continue

            row = [label] + left_hand + right_hand

            with open(CSV_PATH, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(row)

            savedsamples += 1
            last_capture_time = current_time

            print(f"Saved sample #{savedsamples}")

cap.release()
cv2.destroyAllWindows()