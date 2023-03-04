import cv2
from sklearn.model_selection import train_test_split
from sign_language.ml_logic.preprocessing import crop_image
import os
from sign_language.ml_logic.preprocessing import backgroud_removal


def get_images(directory):
    """get images local if in same directory as collab notebook"""
    images = []
    labels = []
    directory_list = sorted(os.listdir(directory))
    for i in range(len(directory_list)):
        print(f"Getting images of {directory_list[i]}:")
        for image in os.listdir(directory + "/" + directory_list[i])[:2]:
            img = cv2.imread(directory + "/" + directory_list[i] + "/" + image)
            images.append(img)
            labels.append(directory_list[i])
    return images, labels
