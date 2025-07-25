import cv2
import numpy as np
import os
import pandas as pd
import mediapipe as mp

# Force CPU only (optional, in case of GPU-related bugs)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Create data folder if it doesn't exist
DATA_DIR = "data/raw"
os.makedirs(DATA_DIR, exist_ok=True)

# Ask user for label
label = input("Enter the label for this gesture (e.g., A, B, Hello): ").strip().upper()
csv_file = os.path.join(DATA_DIR, f"{label}.csv")

# Start webcam
print("📷 Opening webcam...")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("❌ Cannot access webcam")
    exit()
else:
    print("✅ Webcam started. Press 's' to save a sample, 'q' to quit.")

data = []

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    frame = cv2.flip(frame, 1)  # Mirror image
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Flatten 21 landmarks into a 63-dim array (x, y, z)
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            landmarks.append(label)

            # Overlay label
            cv2.putText(frame, f"Label: {label}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("ASL Data Capture", frame)

    key = cv2.waitKey(1)
    if key == ord('s') and result.multi_hand_landmarks:
        print("✅ Sample captured")
        data.append(landmarks)
    elif key == ord('q'):
        print("👋 Exiting and saving data...")
        break

cap.release()
cv2.destroyAllWindows()

# Save collected data
if data:
    df = pd.DataFrame(data)
    df.to_csv(csv_file, mode='a', header=not os.path.exists(csv_file), index=False)
    print(f"💾 Saved {len(data)} samples to {csv_file}")
else:
    print("⚠️ No data collected.")
