# 🧏‍♀️ Sign-Language-Translator  
**Bridging Silence with Seamless Sign Language Communication**

---

![repo-top-language](https://img.shields.io/github/languages/top/ShaileshKumar7234/Sign-Language-Translator)
![last-commit](https://img.shields.io/github/last-commit/ShaileshKumar7234/Sign-Language-Translator)
![repo-language-count](https://img.shields.io/github/languages/count/ShaileshKumar7234/Sign-Language-Translator)

---

## 🔧 Built With

- 🐍 **Python**
- 📄 **Markdown**
- 🤖 TensorFlow / Keras
- 🖼️ OpenCV
- 🖐️ MediaPipe
- 🎙️ pyttsx3
- 🪟 Tkinter

---

## 📚 Table of Contents

- [📌 Overview](#overview)
- [🚀 Getting Started](#getting-started)
  - [✅ Prerequisites](#prerequisites)
  - [📦 Installation](#installation)
- [▶️ Usage](#usage)
- [🧪 Testing](#testing)
- [📄 License](#license)

---

## 📌 Overview

**Sign-Language-Translator** is a powerful developer tool designed to facilitate **real-time American Sign Language (ASL)** interpretation using **computer vision** and **machine learning** techniques.

It provides a complete pipeline—from **data collection and labeling** to **model training** and **live gesture recognition**—making sign language **accessible**, **interactive**, and **voice-enabled**.

---

### 💡 Why Sign-Language-Translator?

This project streamlines the development of sign language communication systems. The core features include:

- 🧩 **Gesture Data Collection**: Capture and label hand landmarks via webcam for dataset creation.  
- 🧠 **Model Training & Evaluation**: Build and evaluate neural network classifiers for gesture recognition.  
- 🖥️ **Real-Time Recognition**: Live hand detection and classification with instant on-screen captions.  
- 🎙️ **Voice Output**: Converts recognized gestures into speech using offline text-to-speech.  
- 🛠️ **Modular Design**: Clean, scalable, and adaptable to new gestures, datasets, or models.

---

## 🚀 Getting Started

### ✅ Prerequisites

This project requires the following:

- Python (≥ 3.8 recommended)
- Conda (Anaconda/Miniconda)
- Webcam

### 📦 Installation

#### 1. Clone the repository

```bash
git clone https://github.com/ShaileshKumar7234/Sign-Language-Translator
cd Sign-Language-Translator
```

#### 2. Create a conda environment

```bash
conda env create -f conda.yml
```

> This installs all required packages including TensorFlow, OpenCV, pyttsx3, MediaPipe, and more.

---

## ▶️ Usage

Activate the environment and run the application:

```bash
conda activate asl_env
python src/gui_app.py
```

This will:

- Open a webcam feed in a GUI window
- Detect hand signs in real-time
- Display the translated letter as a caption
- Speak the letter aloud using TTS

---

## 🧪 Testing

If unit tests are added in the future (using `pytest` or `unittest`), run them like this:

```bash
conda activate asl_env
pytest
```

> 🔧 *Currently, test coverage is under development.*

---

## 📁 Project Structure

```
Sign-Language-Translator/
├── models/
│   ├── asl_model.h5
│   └── label_map.npy
│
├── src/
│   ├── gui_app.py
│   ├── train_model.py
│   ├── generate_label_map.py
│   └── inspect_label_map.py
│
├── data/                # Optional: for training samples
├── conda.yml            # Conda environment file
└── README.md
```

---

## 📄 License

This project is open-source and intended for educational, personal, and non-commercial use.  
For commercial applications, please seek prior permission from the author.

---

## 🙋‍♂️ Author

**Shailesh Kumar**  
🔗 GitHub: [@ShaileshKumar7234](https://github.com/ShaileshKumar7234)
