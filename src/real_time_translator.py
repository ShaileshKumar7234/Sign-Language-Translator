import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import pyttsx3
import time
import csv
from datetime import datetime
# Load model
model = tf.keras.models.load_model('models/asl_model.h5')

# Load class labels from training data
import pandas as pd
y_train = pd.read_csv('data/processed/y_train.csv')
class_names = sorted(y_train.iloc[:, 0].unique())

# Init MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Init TTS
engine = pyttsx3.init()
last_spoken = ""
last_time = 0

# Start webcam
cap = cv2.VideoCapture(0)
print("[INFO] Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            prediction = model.predict(np.array([landmarks]))[0]
            class_id = np.argmax(prediction)
            class_label = class_names[class_id]
            confidence = prediction[class_id]

            if confidence > 0.50:
                cv2.putText(frame, f"{class_label} ({confidence:.2f})", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

                # Speak only if changed or time elapsed
                if class_label != last_spoken or (time.time() - last_time) > 2:
                    engine.say(class_label)
                    engine.runAndWait()
                    last_spoken = class_label
                    last_time = time.time()

    cv2.imshow("ASL Real-Time Translator", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



# Initialize CSV log file (optional)
LOG_PATH = "logs/transcript.csv"
os.makedirs("logs", exist_ok=True)
if not os.path.exists(LOG_PATH):
    with open(LOG_PATH, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Gesture"])