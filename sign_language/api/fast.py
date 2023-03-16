from fastapi import FastAPI
from keras.models import load_model
import os
import cv2
import numpy as np
import tensorflow as tf

app = FastAPI()


app.state.model = load_model(os.environ['MODEL'])


img = cv2.imread('/home/stuart/code/stuhow/sign_language_translation/processed_images/all_images/A/Image_A485.jpg')

@app.get("/")
def index():
    return {"status": "ok"}

@app.get('/predict')
def predict():
    prediction_list = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["del", "space"]
    cropped_image = img.reshape(1, 56, 56, 3)
    prediction = app.state.model.predict(cropped_image)
    letter = np.argmax(prediction[0], axis=0)
    print(prediction)
    print(letter)
    return {'prediction': prediction_list[letter]}
