# utils/speech.py

import pyttsx3

# Initialize TTS engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

def speak_text(text):
    print(f"[🔊] Speaking: {text}")
    engine.say(text)
    engine.runAndWait()
