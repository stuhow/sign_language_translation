import os
import tensorflow as tf
import cv2
import mediapipe as mp
from keras.models import load_model
import numpy as np
import time
from preprocessing import crop_image

model = load_model('models/model.h5')


cap = cv2.VideoCapture(0)
_, frame = cap.read()
h, w, c = frame.shape


while True:
    _, frame = cap.read()

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    image = crop_image(frame)

    cv2.imshow("Image", image)

cap.release()
cv2.destroyAllWindows()
