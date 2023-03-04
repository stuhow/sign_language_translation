import streamlit as st
import copy

import cv2
import numpy as np
import mediapipe as mp
from tensorflow import device
from keras.models import load_model
from tensorflow import config
import os


def main():
    model = get_model()
    cap_width = 640
    cap_height = 480

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)

    hands = load_mediapipe_model()
    
    prediction_list = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["del", "space"]

    if st.button("Start"):
        with st.spinner('Loading...'):
            while True:
                ret, image = cap.read()
                if not ret:
                    break
                image = cv2.flip(image, 1)
                debug_image = copy.deepcopy(image)

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(image)

                cropped_image = 0
                shape = 0

                if results.multi_hand_landmarks is not None:
                    for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                        results.multi_handedness):
                        brect = calc_bounding_rect(debug_image, hand_landmarks)
                        debug_image = draw_bounding_rect(debug_image, brect)
                        cropped_image = debug_image[brect[3]:brect[2], brect[1]:brect[0]]
                try:
                    cropped_image = cv2.resize(cropped_image, (56, 56))
                    cropped_image = backgroud_removal(cropped_image)
                    cropped_image = cropped_image.reshape(1, 56, 56, 3)
                except:
                    pass
                predict = None
                prediction = None
                proba = None
                if cropped_image.shape == (1, 56, 56, 3):
                    predict = model.predict(cropped_image)
                    prediction = np.argmax(predict[0], axis = -1)
                    proba = max(predict[0])
                    st.write(f"Prediction: {prediction_list[prediction]}, p_value = {proba}")
                st.image(debug_image)

    cap.release()
    cv2.destroyAllWindows()


def get_model():
    local_path = os.environ['MODEL']
    model = load_model(local_path)
    return model

def load_mediapipe_model():
    max_num_hands = 1
    min_detection_confidence = 0.5
    min_tracking_confidence = 0.5

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=max_num_hands,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    return hands


def calc_bounding_rect(image, hand_landmarks):
    h, w, c = image.shape 

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

            y_min -= round(h/20)
            y_max += round(h/20)
            x_min -= round(w/20)
            x_max += round(w/20)

        x_max, x_min, y_max, y_min = make_image_square(x_max, x_min, y_max, y_min, h, w)
        return [x_max, x_min, y_max, y_min]


def make_image_square(x_max, x_min, y_max, y_min, h, w):
    '''used in calculating bounding rect'''
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


def draw_bounding_rect(image, brect):
    cv2.rectangle(image, (brect[1], brect[3]), (brect[0], brect[2]),
        (0, 255, 0), 2)

    return image


def backgroud_removal(img):
    """Receives the croped image and proceeds
    to remove the background, leaving only the hand layout.
    """
    if img.shape[2] < 3:
        return img
    # initialize mediapipe
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
    #creating white background
    imgWhite = np.ones((img.shape[0],img.shape[1],3),np.uint8)*255
    # extract segmented mask
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = selfie_segmentation.process(img)
    #condition to apply the mask
    condition = np.stack(
      (results.segmentation_mask,) * 3, axis=-1) > 0.6
    #merging croped img with the white background
    noBackground = np.where(condition, img, imgWhite)
    selfie_segmentation.close()
    return noBackground


if __name__ == '__main__':
    main()
