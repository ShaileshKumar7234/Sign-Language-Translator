import tkinter as tk
from tkinter import ttk
from threading import Thread
import cv2
import PIL.Image, PIL.ImageTk
import numpy as np
import tensorflow as tf
import mediapipe as mp
import pyttsx3
import time

# Load model and label encoder
print("[🔁] Loading model...")
model = tf.keras.models.load_model('models/asl_model.h5')
label_map = np.load('models/label_map.npy', allow_pickle=True).item()
print("[✅] Model and labels loaded.")

# MediaPipe and TTS
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils
engine = pyttsx3.init()

class ASLApp:
    def __init__(self, window):
        self.window = window
        self.window.title("ASL Translator")
        self.running = False

        # Webcam
        self.video = cv2.VideoCapture(0)
        if not self.video.isOpened():
            raise RuntimeError("Could not open webcam.")

        # Canvas
        self.canvas = tk.Canvas(window, width=640, height=480)
        self.canvas.pack()

        # Buttons
        self.btn_start = ttk.Button(window, text="Start", command=self.start)
        self.btn_start.pack(pady=10)

        self.btn_stop = ttk.Button(window, text="Stop", command=self.stop)
        self.btn_stop.pack()

        # Subtitle
        self.subtitle = tk.Label(window, text="", font=("Helvetica", 20))
        self.subtitle.pack(pady=10)

        self.last_spoken = ""
        self.last_time = 0

    def start(self):
        if not self.running:
            print("[▶️] Starting ASL detection...")
            self.running = True
            self.thread = Thread(target=self.update, daemon=True)
            self.thread.start()

    def stop(self):
        print("[⏹️] Stopping ASL detection...")
        self.running = False
        self.subtitle.config(text="")
        self.video.release()
        cv2.destroyAllWindows()

    def update(self):
        while self.running:
            ret, frame = self.video.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        landmarks.extend([lm.x, lm.y, lm.z])

                    print(f"[📏] Landmark length: {len(landmarks)}")
                    print(f"[📐] Expected input shape: {model.input_shape}")

                    if len(landmarks) == model.input_shape[1]:
                        prediction = model.predict(np.array([landmarks]), verbose=0)
                        class_id = np.argmax(prediction)
                        confidence = prediction[0][class_id]
                        label = label_map[class_id]

                        print(f"[🔊] Predicting: {label} with confidence {confidence:.2f}")
                        self.subtitle.config(text=f"{label} ({confidence:.2f})")

                        # Speak the label
                        if confidence > 0.85 and (label != self.last_spoken or (time.time() - self.last_time) > 2):
                            engine.say(label)
                            engine.runAndWait()
                            self.last_spoken = label
                            self.last_time = time.time()
                    else:
                        self.subtitle.config(text="Landmark mismatch")
            else:
                self.subtitle.config(text="No hand detected")

            img = PIL.Image.fromarray(rgb)
            imgtk = PIL.ImageTk.PhotoImage(image=img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            self.canvas.image = imgtk

# Run GUI
if __name__ == "__main__":
    print("[🚀] Launching ASL GUI...")
    root = tk.Tk()
    app = ASLApp(root)
    root.mainloop()
    print("[👋] GUI closed.")
