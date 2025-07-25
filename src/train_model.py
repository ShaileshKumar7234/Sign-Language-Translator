import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder

# Paths
PROCESSED_DIR = 'data/processed'
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

# Load data
X_train = pd.read_csv(os.path.join(PROCESSED_DIR, 'X_train.csv')).values
X_test = pd.read_csv(os.path.join(PROCESSED_DIR, 'X_test.csv')).values
y_train = pd.read_csv(os.path.join(PROCESSED_DIR, 'y_train.csv')).values.ravel()
y_test = pd.read_csv(os.path.join(PROCESSED_DIR, 'y_test.csv')).values.ravel()

# Encode labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)
num_classes = len(label_encoder.classes_)

# Normalize features
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Build the model
model = models.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train_encoded, epochs=25, batch_size=32, validation_data=(X_test, y_test_encoded))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test_encoded)
print(f"[✅] Test Accuracy: {accuracy:.4f}")

# Save the model
model_path = os.path.join(MODEL_DIR, 'asl_model.h5')
model.save(model_path)
print(f"[💾] Model saved at {model_path}")

# Save label map
label_map = {i: label for i, label in enumerate(label_encoder.classes_)}
np.save(os.path.join(MODEL_DIR, 'label_map.npy'), label_map)
print("[✅] Label map saved at models/label_map.npy")
