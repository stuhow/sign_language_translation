import cv2
from preprocessing import crop_image
import os

def get_images(directory):
    """get images local if in same directory as collab notebook"""
    images = []
    labels = []

    directory_list = sorted(os.listdir(directory))
    for i in range(len(directory_list)):
        print(f"Getting images of {directory_list[i]}:")
        for image in os.listdir(directory + "/" + directory_list[i])[:10]:
            img = cv2.imread(directory + "/" + directory_list[i] + "/" + image)
            img = crop_image(img)
            if img.shape[0] > 1:
                img = cv2.resize(img, (56, 56))
                images.append(img)
                labels.append(directory_list[i])

    return images, labels
