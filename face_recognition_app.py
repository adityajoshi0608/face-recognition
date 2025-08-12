import cv2
import numpy as np
import os
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import messagebox

def train_recognizer(data_dir='faces'):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    current_id = 0
    label_ids = {}
    x_train, y_labels = [], []

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith("jpg") or file.endswith("png"):
                path = os.path.join(root, file)
                label = os.path.basename(root)

                if label not in label_ids:
                    label_ids[label] = current_id
                    current_id += 1
                
                id_ = label_ids[label]

                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=4)
                for (x, y, w, h) in faces:
                    roi = img[y:y+h, x:x+w]
                    x_train.append(roi)
                    y_labels.append(id_)
    
    recognizer.train(x_train, np.array(y_labels))
    return recognizer, label_ids

class FaceRecognitionApp:
    def __init__(self, root):
        self.cap = None
        self.root = root
        self.root.title("Face Recognition App")
        
        self.recognizer, self.label_ids = train_recognizer()
        self.id_labels = {v: k for k, v in self.label_ids.items()}
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        self.cap = cv2.VideoCapture(0)
        self.panel = tk.Label(root)
        self.panel.pack()

        self.capture_btn = tk.Button(root, text="Capture and Recognize", command=self.capture_and_recognize)
        self.capture_btn.pack()

        self.update_frame()

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            self.panel.imgtk = imgtk
            self.panel.config(image=imgtk)
        self.root.after(10, self.update_frame)

    def capture_and_recognize(self):
        ret, frame = self.cap.read()
        if not ret:
            messagebox.showerror("Error", "Failed to capture image")
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            id_, conf = self.recognizer.predict(roi_gray)
            name = self.id_labels.get(id_, "Unknown")
            if conf < 80:
                label = f"{name} ({int(conf)})"
            else:
                label = "Unknown"

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        cv2.imshow("Recognition Result", frame)
        cv2.waitKey(3000)
        cv2.destroyWindow("Recognition Result")

def __del__(self):
    if hasattr(self, 'cap') and self.cap and self.cap.isOpened():
        self.cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()