import cv2
import mediapipe as mp
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import utils
from sklearn.model_selection import train_test_split
import numpy as np

def make_image_square(x_max, x_min, y_max, y_min):
    '''function used as a helper function in crop images to return a square image'''
    x_diff = x_max - x_min
    y_diff = y_max - y_min

    if y_diff > x_diff:

        length_diff =  y_diff - x_diff

        half_length_diff_max = round(length_diff/2)
        half_length_diff_min = length_diff-half_length_diff_max

        x_max = half_length_diff_max + x_max
        x_min = x_min - half_length_diff_min


    elif x_diff > y_diff:
        length_diff =  x_diff -  y_diff

        half_length_diff_max = round(length_diff/2)
        half_length_diff_min = length_diff-half_length_diff_max

        y_max = half_length_diff_max + y_max
        y_min = y_min - half_length_diff_min


    return x_max, x_min, y_max, y_min

def crop_image(image):
    '''a function to crop an image to just show their hands'''

    mp_hands = mp.solutions.hands

    # For static images:
    mp_model = mp_hands.Hands(
        static_image_mode=True, # only static images
        max_num_hands=2, # max 2 hands detection
        min_detection_confidence=0.5) # detection confidence

    results = mp_model.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    h, w, c = image.shape # get image shape

    hand_landmarks = results.multi_hand_landmarks

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
                y_min -= 20
                y_max += 20
                x_min -= 20
                x_max += 20

                x_max, x_min, y_max, y_min = make_image_square(x_max, x_min, y_max, y_min)

                crop_img = image[y_min:y_max, x_min:x_max]

    else:
        pass

    return crop_img


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
