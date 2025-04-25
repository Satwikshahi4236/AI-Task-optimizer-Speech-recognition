# Emotion Detection System: Multi-Modal (Face, Voice, Text)

import cv2
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
from transformers import pipeline
import tkinter as tk
from tkinter import filedialog, Label, Button
import sounddevice as sd
import soundfile as sf

face_model = load_model('models/facial_emotion_model.h5')
text_pipeline = pipeline("sentiment-analysis")
voice_model = load_model('models/voice_emotion_model.h5')

emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
def predict_face_emotion(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (48, 48)) / 255.0
        roi = roi.reshape(1, 48, 48, 1)
        prediction = face_model.predict(roi)
        return emotions[np.argmax(prediction)]
    return "No face detected"


def predict_voice_emotion(file_path):
    y, sr = librosa.load(file_path, duration=3, offset=0.5)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc = np.mean(mfcc.T, axis=0)
    mfcc = mfcc.reshape(1, -1)
    prediction = voice_model.predict(mfcc)
    return emotions[np.argmax(prediction)]


def predict_text_emotion(text):
    result = text_pipeline(text)[0]
    return result['label']


class EmotionGUI:
    def __init__(self, root):
        self.root = root
        root.title("Multi-Modal Emotion Detection")

        self.label = Label(root, text="Choose input mode:")
        self.label.pack()

        self.face_button = Button(root, text="Detect from Image", command=self.load_face_image)
        self.face_button.pack()

        self.voice_button = Button(root, text="Detect from Voice", command=self.load_voice_file)
        self.voice_button.pack()

        self.text_button = Button(root, text="Detect from Text", command=self.get_text_input)
        self.text_button.pack()

        self.result_label = Label(root, text="")
        self.result_label.pack()

    def load_face_image(self):
        file_path = filedialog.askopenfilename()
        emotion = predict_face_emotion(file_path)
        self.result_label.config(text=f"Detected Emotion (Face): {emotion}")

    def load_voice_file(self):
        file_path = filedialog.askopenfilename()
        emotion = predict_voice_emotion(file_path)
        self.result_label.config(text=f"Detected Emotion (Voice): {emotion}")

    def get_text_input(self):
        input_win = tk.Toplevel(self.root)
        input_win.title("Text Input")
        tk.Label(input_win, text="Enter text:").pack()
        text_entry = tk.Entry(input_win)
        text_entry.pack()
        def process_text():
            text = text_entry.get()
            emotion = predict_text_emotion(text)
            self.result_label.config(text=f"Detected Emotion (Text): {emotion}")
            input_win.destroy()
        tk.Button(input_win, text="Submit", command=process_text).pack()


if __name__ == "__main__":
    root = tk.Tk()
    gui = EmotionGUI(root)
    root.mainloop()
