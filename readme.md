# 🤟 AI Sign Language Translator using Deep Learning

## 📌 Project Overview

The AI Sign Language Translator is a Deep Learning-based computer vision project that recognizes American Sign Language (ASL) hand gestures in real time using a webcam.

The system uses **MobileNetV2** for image classification and **MediaPipe** for hand detection. It predicts the corresponding alphabet and can convert recognized gestures into text.

---

## 🚀 Features

- Real-time hand detection using MediaPipe
- Deep Learning model using MobileNetV2
- Live webcam prediction
- Alphabet recognition (A-Z)
- Bounding box around detected hand
- Hand landmark visualization
- High accuracy image classification
- Easy to train with custom datasets

---

## 🛠️ Technologies Used

- Python 3.10
- TensorFlow / Keras
- OpenCV
- MediaPipe
- NumPy
- Matplotlib

---

## 📂 Project Structure

```
AI Sign Language Translator
│
├── dataset
│   ├── asl_alphabet_train
│   │   ├── A
│   │   ├── B
│   │   ├── C
│   │   └── ...
│   │
│   └── asl_alphabet_test
│
├── models
│   ├── best_model.keras
│   ├── final_model.keras
│   ├── accuracy.png
│   └── loss.png
│
├── preprocess.py
├── train.py
├── predict.py
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

Dataset Used:

**ASL Alphabet Dataset**

Classes:

- A - Z
- del
- nothing
- space

Total Classes: **29**

---

## ⚙️ Model Architecture

- MobileNetV2 (Pre-trained on ImageNet)
- Global Average Pooling
- Dense Layer (256 neurons)
- Dropout Layer
- Softmax Output Layer

Loss Function:

```
Categorical Crossentropy
```

Optimizer:

```
Adam
```

---

## 🔥 Training

Run:

```bash
python train.py
```

After training, the following files are generated:

- best_model.keras
- final_model.keras
- accuracy.png
- loss.png

---

## ▶️ Prediction

Run:

```bash
python predict.py
```

The webcam will open.

The application:

- Detects the hand
- Draws landmarks
- Predicts the sign
- Displays the recognized alphabet

---

## 📈 Results

- Deep Learning based classification
- Real-time webcam prediction
- MediaPipe hand tracking
- Lightweight MobileNetV2 architecture

---

## 📷 Future Improvements

- Sentence generation
- Voice output
- Streamlit dashboard
- Multi-hand detection
- Word prediction
- Translation to different languages

---

## 💡 Applications

- Communication for deaf and mute individuals
- Smart education
- Human-computer interaction
- Gesture-controlled systems
- Assistive AI

---

## 👨‍💻 Author

**Thoram Sudeeptha**

B.Tech – Artificial Intelligence & Machine Learning

---

## ⭐ Acknowledgements

- TensorFlow
- OpenCV
- MediaPipe
- Kaggle ASL Alphabet Dataset