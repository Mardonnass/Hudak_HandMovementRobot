import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
import pickle
from collections import Counter

# -------------------------------
# 1. Load and preprocess CSV
# -------------------------------
csv_file = 'Try1.csv'
df = pd.read_csv(csv_file)

# Shuffle dataset
df = shuffle(df, random_state=42)

# Split features and labels
X = df.drop(columns=["label"]).values
y = df["label"].values

print("X shape:", X.shape)
print("Labels:", np.unique(y))

# -------------------------------
# 2. Normalize EACH HAND separately
# -------------------------------
def normalize_landmarks(landmarks):
    landmarks = landmarks.reshape(2, 21, 3)  # 2 hands

    normalized = []

    for hand in landmarks:
        base = hand[0]  # wrist
        hand = hand - base

        scale = np.linalg.norm(hand[9] - hand[0])  # wrist → middle finger base
        if scale > 0:
            hand = hand / scale

        normalized.append(hand.flatten())

    return np.concatenate(normalized)

X = np.array([normalize_landmarks(row) for row in X])

# -------------------------------
# 3. Encode labels
# -------------------------------
le = LabelEncoder()
y = le.fit_transform(y)

with open("ENKODIK.pkl", "wb") as f:
    pickle.dump(le, f)

print("Label encoder saved!")
print(Counter(y))
# -------------------------------
# 4. Train/test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2 , random_state=42
)

# -------------------------------
# 5. Build improved model
# -------------------------------
def build_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_shape,)),

        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),

        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

input_shape = X_train.shape[1]
num_classes = len(np.unique(y))

model = build_model(input_shape, num_classes)

# -------------------------------
# 6. Compile model
# -------------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# -------------------------------
# 7. Early stopping (IMPORTANT)
# -------------------------------
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# -------------------------------
# 8. Train model
# -------------------------------
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    batch_size=64,
    epochs=40,
    callbacks=[early_stop]
)

# -------------------------------
# 9. Save model
# -------------------------------
model.save("MODEL2.keras")
print("Model saved successfully!")

# -------------------------------
# 10. Inference helper
# -------------------------------
def predict_gesture(landmarks):
    landmarks = normalize_landmarks(landmarks)
    landmarks = np.array(landmarks).reshape(1, -1)

    pred = model.predict(landmarks, verbose=0)
    pred_class = np.argmax(pred)

    return le.inverse_transform([pred_class])[0]