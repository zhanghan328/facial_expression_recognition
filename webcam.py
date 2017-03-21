#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 17:12:23 2017

@author: wind
"""
import cv2
import numpy as np
from keras.models import model_from_json

CASCADE_PATH = "haarcascade_frontalface_default.xml"
cascade = cv2.CascadeClassifier(CASCADE_PATH)
emotion = ['Angry', 'Fear', 'Happy','Sad', 'Surprise', 'Neutral']

#load model 
# load json and create model arch
json_file = open('model.json','r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# load weights into new model
model.load_weights('model.h5')

def predict_emotion(gray_face):
    resized_img = cv2.resize(gray_face, (48,48), interpolation = cv2.INTER_AREA)
    image = resized_img.reshape(1, 1, 48, 48)
    results = model.predict(image, batch_size=1, verbose=1)
    return results


capture = cv2.VideoCapture(0)


while True:
    flag, frame = capture.read()
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY,1)
    rects = cascade.detectMultiScale(
        img_gray,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(48, 48),
        flags= cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in rects:
        face_image = img_gray[y:y+h,x:x+w]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        results = predict_emotion(face_image)

        print emotion[np.argmax(results)]

    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()