import streamlit as st
import copy
import cv2
import numpy as np
import pandas as pd
from random import randrange
import time
import mediapipe as mp
from tensorflow import device
from google.cloud import storage
from keras.models import load_model
from tensorflow import config
import os
import av
from rembg import remove
from PIL import Image
from streamlit_webrtc import (
    RTCConfiguration,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)

config.run_functions_eagerly(True)

# ss = st.session_state.get(option='A')

# if "option" not in st.session_state:
# 	st.session_state.option = "A"

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)
list_of_predictions = []
# counter = 0
test_prob = None
#
def app_sign_language_detection(_model, _mp_model, _option):

    class signs(VideoProcessorBase):
        def __init__(self) -> None:

            self.model = _model
            self.hands = _mp_model
            self.option = _option
            self.top3 = None


        def update_status(self,option):
            if self.option != option:
                self.option = option

        def get_predict(self,cropped_img):
            if cropped_img.shape == (1, 56, 56, 3):
                print('entered if shape statement')
                predict = self.model.predict(cropped_img)[0]
                global list_of_predictions
                list_of_predictions.append(predict)
                if len(list_of_predictions) > 5:
                    del list_of_predictions[0]
                predict_mean = np.mean(np.array(list_of_predictions), axis = 0)
                top3 = np.argsort(predict_mean)[-3:]
                top3 = list(reversed(top3))

            return top3,predict_mean

        def draw_and_predict(self, image):
            prediction_list = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["del", "space"]
            print(f'initial print after defining function')
            # print(image)
            image = cv2.flip(image, 1)
            debug_image = copy.deepcopy(image)
            option = self.option
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.hands.process(image)
            # print(results.multi_hand_landmarks)
            cropped_image = None
            shape = 0

            if results.multi_hand_landmarks is not None:
                for hand_landmarks in results.multi_hand_landmarks:
                    brect = calc_bounding_rect(debug_image, hand_landmarks)
                    debug_image = draw_bounding_rect(debug_image, brect)
                    cropped_image = debug_image[brect[3]:brect[2], brect[1]:brect[0]]
            else:
                print('No hand found')

            try:
                cropped_image = cv2.resize(cropped_image, (56, 56))
                cropped_image = backgroud_removal(cropped_image)
                cropped_image = cropped_image/255
                cropped_image = cropped_image.reshape(1, 56, 56, 3)
            except:
                print('No hand found')

            predict = None
            # global counter
            # if counter % 10 != 0:
            #     return debug_image

            try:
                # if cropped_image.shape == (1, 56, 56, 3):
                #     print('entered if shape statement')
                #     predict = self.model.predict(cropped_image)[0]
                #     global list_of_predictions
                #     list_of_predictions.append(predict)
                #     if len(list_of_predictions) > 5:
                #         del list_of_predictions[0]
                #     predict_mean = np.mean(np.array(list_of_predictions), axis = 0)
                #     top3 = np.argsort(predict_mean)[-3:]
                #     top3 = list(reversed(top3))
                top3,predict_mean = self.get_predict(cropped_image)
                global test_prob
                test_prob = top3[0]
                debug_image = print_prob([predict_mean[i] for i in top3], [prediction_list[i] for i in top3], debug_image,option)

                return debug_image

            except:
                cv2.putText(debug_image, f"No hand detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
                return debug_image




        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            # global counter
            # counter += 1
            image = frame.to_ndarray(format='rgb24')
            annotated_image = self.draw_and_predict(image)
            return av.VideoFrame.from_ndarray(annotated_image,format='rgb24')

    webrtc_ctx = webrtc_streamer(
    key="sign_language",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=signs,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
    )
    if webrtc_ctx.video_processor:

        webrtc_ctx.video_processor.update_status(_option)
        return _option



@st.cache_resource
def load_cloud_model():
    bucket_name = os.environ.get("BUCKET")
    model_name = os.environ.get("MODEL_NAME")
    model_dir = os.environ.get("MODEL_DIR")

    # Create a client object for Google Cloud Storage
    client = storage.Client()

    # Get a bucket object for the bucket
    bucket = client.get_bucket(bucket_name)

    # Get a blob object for the Keras model file
    blob = bucket.blob(model_name)

    # Download the Keras model file to a local file
    blob.download_to_filename(model_dir + model_name)

    model = load_model(model_dir + model_name)

    return model


@st.cache_resource
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
    x_max = 0
    y_max = 0
    x_min = w
    y_min = h
    for lm in hand_landmarks.landmark:
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

def print_prob(predict, letters, debug_image,option):

    colours = [(0, 244, 127),
                (250, 176, 55),
                (223, 198, 106),
                (208, 157, 74),
                (78, 174, 107)]
    output_frame = debug_image.copy()
    for num, prob in enumerate(predict):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colours[num], -1)
        cv2.putText(output_frame, letters[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    if option == letters[0]:
        cv2.putText(output_frame, f"Congrats! you're doing {letters[0]} with {round(predict[0]*100)}% Accuracy ",
                    (80, 450),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (79, 235, 52), 2, cv2.LINE_AA)

    else:
        cv2.putText(output_frame, f"Sorry, you're doing {letters[0]} instead of {option}", (80, 450),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (235, 52, 52), 2, cv2.LINE_AA)

    return output_frame



st.set_page_config(
            page_title="SignIntell",


            # page_icon="🐍",
            layout="centered", # wide
            initial_sidebar_state="auto")


# Upload models to the page, first thing when opening!
model = load_cloud_model()
mp_model = load_mediapipe_model()


def about():
    st.write('Welcome to our Sign Language detection system')
    st.markdown("""
     **About our app**
    - We are attempting to improve the communication all around.
    - Our app is a real time Sign Language detection, using a live camera the
    user can practice hand signs by selecting the desired letter.
    - Our system will detect the hand sign being made and evaluate accordingly.""")

object_detection_page = "SignLingo"
about_page = "About SignIntell"

app_mode = st.sidebar.selectbox(
    "Choose the app mode",
    [
        object_detection_page,
        about_page
    ],)


st.subheader(app_mode)


@st.cache_data
def get_select_box_data():

    return list(" ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["del", "space"]



# grid to place the example image in the middle
def grid(img):

    col1,col2,col3 = st.columns(3)

    with col2:
        place_holder = st.image(img)
    return place_holder


# grid to place the example image in the middle
def grid(img):

    col1,col2,col3 = st.columns(3)

    with col2:
        place_holder = st.image(img)
    return place_holder


def about_sign_lingo():
    st.markdown('**Welcome to **SignLingo**, the Sign Language training app**')
    st.markdown("""
                This is a real-time experience when practicing sign language,
                follow the steps below and try!""")

    if st.button("Instructions"):
        mkdown_holder = st.markdown("""
                    - First select which sign within the available ones you want to try.
                    - If you have no clue on the shape, click on the Get a hint button, that will give you an example of the sign.
                    - The example image is only available for 5 seconds, so if you need more time just click once more.
                    - When ready, click on the start button.
                    - As soon as the camera opens, put your hand in a position where our system detects it, and try the sign you chose!
                    - Pay atention to the feedback answer on the bottom of the camera screen:
                        - If you make it correctly, you should see a green statement giving you the accuracy of your sign!
                        - If not, a red one will appear telling you which sign our system is detecting and which one you should aim to do.
                    - Finally, whenever you want to try a new one, just select it from the dropdown menu.

                    """)
        if st.button("Close"):
            mkdown_holder.empty()


def obj_detection():

    about_sign_lingo()
    df = get_select_box_data()
    opt_holder = " "

    #asking the user to select a letter to be predicted for comparison.
    option = st.selectbox('**Select a Sign to practice**', df)

    #if the selectbox returns a letter different than  " ", main function is called.
    if option != opt_holder:
        opt_holder = app_sign_language_detection(model, mp_model,option)
        if st.button("Get a hint!"):
            info = st.info(f"This is the shape of  {option}")
            img = Image.open(f"{os.environ.get('EXAMPLES')}/{option}/{option}.jpg")
            img = remove(img)
            place_holder = grid(img)
            time.sleep(5)
            place_holder.empty()
            info.empty()


# pre-loading the model before calling the main function

if app_mode == object_detection_page:
    obj_detection()


if app_mode == about_page:
    about()
