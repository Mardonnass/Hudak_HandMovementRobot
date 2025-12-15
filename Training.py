import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
import pickle

# -------------------------------
# 1. Load and preprocess CSV
# -------------------------------
csv_file = 'PrvyREAL.csv'  # CSV format: x0,y0,...,x20,y20,label
df = pd.read_csv(csv_file)

# Shuffle dataset
df = shuffle(df, random_state=42)

# Split features and labels
X = df.drop(columns=["label"]).values  # landmarks (42 values)
print(X.shape)
y = df["label"].values
print(y)   

# -------------------------------
# 1. Load and preprocess CSV
# -------------------------------
csv_file = 'PrvyREAL.csv'
df = pd.read_csv(csv_file)

# Split features and labels
X = df.drop(columns=["label"]).values
y = df["label"].values

# Normalize each row (each sample)
def normalize_landmarks(landmarks):
    landmarks = landmarks.reshape(-1, 3)  # 21 landmarks × 2 hands = 42 points total
    # You can decide whether to normalize each hand separately.
    # For simplicity, normalize whole vector relative to first landmark (wrist)
    base = landmarks[0]
    landmarks -= base
    scale = np.linalg.norm(landmarks[9] - landmarks[0])  # wrist → middle finger base
    if scale > 0:
        landmarks /= scale
    return landmarks.flatten()

X = np.array([normalize_landmarks(row) for row in X])

# Encode labels (TEXT → NUMBER)
le = LabelEncoder()
y = le.fit_transform(y)
print(np.unique(y))
# Save encoder for webcam inference
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("Saved label_encoder.pkl")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 2. Build TensorFlow model
# -------------------------------
def build_model(input_shape, num_classes, hidden_units=256, dropout_rate=0.2):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_shape,)),
        tf.keras.layers.Dense(hidden_units, activation='relu'),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(hidden_units, activation='relu'),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

input_shape = X_train.shape[1]
num_classes = len(np.unique(y))

model = build_model(input_shape, num_classes)

# -------------------------------
# 3. Compile model
# -------------------------------
learning_rate = 0.001
batch_size = 128
epochs = 500

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# -------------------------------
# 4. Train the model
# -------------------------------
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    batch_size=batch_size,
    epochs=epochs
)

# -------------------------------
# 5. Save the trained model
# -------------------------------
model.save("MojKlasfir.h5")
print("Model saved successfully!")

# -------------------------------
# 6. Inference helper function
# -------------------------------
def predict_gesture(landmarks):
    landmarks = np.array(landmarks).reshape(1, -1)
    pred_class = np.argmax(model.predict(landmarks))
    return le.inverse_transform([pred_class])[0]
