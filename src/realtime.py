# src/realtime.py
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array

def run_realtime_emotion_recognition(model_path, emotions_list):
    # Load the pre-trained emotion detection model
    emotion_classifier = load_model(model_path, compile=False)
    # Initialize face detection
    face_detection = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cv2.namedWindow('Your_Face')
    camera = cv2.VideoCapture(0)
    while True:
        ret, frame = camera.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detection.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                                  minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
        for (fX, fY, fW, fH) in faces:
            roi_gray = gray[fY:fY+fH, fX:fX+fW]
            roi_gray = cv2.resize(roi_gray, (100, 100))
            roi_gray = roi_gray.astype("float") / 255.0
            roi_gray = img_to_array(roi_gray)
            roi_gray = np.expand_dims(roi_gray, axis=0)
            preds = emotion_classifier.predict(roi_gray)[0]
            emotion_label = emotions_list[np.argmax(preds)]
            emotion_prob = np.max(preds) * 100
            cv2.rectangle(frame, (fX, fY), (fX+fW, fY+fH), (0,255,0), 2)
            cv2.putText(frame, f"{emotion_label}: {emotion_prob:.2f}%", (fX, fY-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 2)
        cv2.imshow('Your_Face', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    camera.release()
    cv2.destroyAllWindows()
