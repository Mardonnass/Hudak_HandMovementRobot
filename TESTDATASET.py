import mediapipe as mp
import cv2
import pandas as pd
import csv
import os

mp_hands = mp.solutions.hands

def extract_from_csv(label_csv, image_folder, output_csv):
    # Load your CSV with columns: filename,label
    df = pd.read_csv(label_csv)

    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.65
    )

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)

        # Write header (optional)
        header = []
        for i in range(21):
            header += [f"x{i+1}", f"y{i+1}", f"z{i+1}"]
        header += ["label"]
        writer.writerow(header)

        for _, row in df.iterrows():
            filename = row["filename"]
            label = row["label"]

            img_path = os.path.join(image_folder, filename)

            img = cv2.imread(img_path)
            if img is None:
                print("File missing:", img_path)
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            if not results.multi_hand_landmarks:
                print("No hand detected:", filename)
                continue

            # Extract 21 landmark triples
            landmarks = []
            lm = results.multi_hand_landmarks[0]
            for p in lm.landmark:
                landmarks += [p.x, p.y, p.z]

            writer.writerow(landmarks + [label])

    hands.close()
    print("DONE! Created:", output_csv)



extract_from_csv(
    label_csv="MockDatasetCSV.csv",
    image_folder="MockDataset/",
    output_csv="gestures.csv"
)
