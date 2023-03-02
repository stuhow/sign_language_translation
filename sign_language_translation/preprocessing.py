import cv2
import mediapipe as mp
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import utils
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import random

def make_image_square(x_max, x_min, y_max, y_min, h, w):
    '''used in below'''
    x_diff = x_max - x_min
    y_diff = y_max - y_min

    if y_diff > x_diff:

        length_diff =  y_diff - x_diff

        half_length_diff_max = round(length_diff/2)
        half_length_diff_min = length_diff-half_length_diff_max

        x_max = half_length_diff_max + x_max
        x_min = x_min - half_length_diff_min
        if x_min < 0:
            x_max += abs(x_min)
        if x_max > w:
            x_min -= x_max

    elif x_diff > y_diff:
        length_diff =  x_diff -  y_diff

        half_length_diff_max = round(length_diff/2)
        half_length_diff_min = length_diff-half_length_diff_max

        y_max = half_length_diff_max + y_max
        y_min = y_min - half_length_diff_min

        if y_min < 0:
            y_max += abs(y_min)
        if y_max > h:
            y_min -= y_max


    return x_max, x_min, y_max, y_min

def crop_image(image):

    mp_hands = mp.solutions.hands

    # For static images:
    mp_model = mp_hands.Hands(
        static_image_mode=True, # only static images
        max_num_hands=2, # max 2 hands detection
        min_detection_confidence=0.5) # detection confidence

    results = mp_model.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    h, w, c = image.shape # get image shape

    hand_landmarks = results.multi_hand_landmarks
    crop_image = np.array([[[0]]])
    if hand_landmarks:
        for handLMs in hand_landmarks:
            x_max = 0
            y_max = 0
            x_min = w
            y_min = h
            for lm in handLMs.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                if x > x_max:
                    x_max = x
                if x < x_min:
                    x_min = x
                if y > y_max:
                    y_max = y
                if y < y_min:
                    y_min = y
            y_min -= round(h/10)
            y_max += round(h/10)
            x_min -= round(h/10)
            x_max += round(h/10)

        x_max, x_min, y_max, y_min = make_image_square(x_max, x_min, y_max, y_min, h, w)

        crop_image = image[y_min:y_max, x_min:x_max]

        if x_max > w:
            crop_image = np.array([[[0]]])
        if x_min < 0:
            crop_image = np.array([[[0]]])
        if y_max > h:
            crop_image = np.array([[[0]]])
        if y_min < 0:
            crop_image = np.array([[[0]]])

    mp_model.close()
    return crop_image

def balancing(X, y):
    m = pd.Series(y).value_counts()[-1]
    unique_values = pd.Series(y).unique()
    x_sampled = []
    y_sampled = []
    for i in unique_values:
        start = y.index(i)
        end = len(y) - y[::-1].index(i)
        x1 = X[start:end]
        y1 = y[start:end]
        x_sub, y_sub = zip(*random.sample(list(zip(x1,y1)), m))
        x_sampled.extend(x_sub)
        y_sampled.extend(y_sub)

    return x_sampled, y_sampled


def preprocessing(X, y):
    """Normalise our images and categorically encode our labels"""
    LE = LabelEncoder()
    X = np.array(X)
    X = X/255
    y = LE.fit_transform(y)
    y = utils.to_categorical(y, num_classes = 29)
    return X, y

def train_val_test_split(X, y):
	X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.4)
	X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size = 0.5)
	return X_train, X_val, X_test, y_train, y_val, y_test


def backgroud_removal(img):
    """Receives the croped image and proceeds
    to remove the background, leaving only the hand layout.
    """

    # initialize mediapipe
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

    #creating white background
    imgWhite = np.ones((img.shape[0],img.shape[1],3),np.uint8)*255

    # extract segmented mask
    results = selfie_segmentation.process(img)

    #condition to apply the mask
    condition = np.stack(
      (results.segmentation_mask,) * 3, axis=-1) > 0.6

    #merging croped img with the white background
    noBackground = np.where(condition, img, imgWhite)

    return noBackground
