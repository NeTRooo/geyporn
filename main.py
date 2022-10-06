import cv2
import math
import sys
import numpy as np
import time
import os

CONFIDENCE = 0.5
SCORE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

vid = cv2.VideoCapture(0)
while True:
    ret, frame = vid.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, z, h) in faces:

        cv2.rectangle(frame, (x, y), (x+z, y+h), (255, 0, 0), 2)
       #cv2.putText()
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyALLWindows()