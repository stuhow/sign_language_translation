import cv2
from preprocessing import crop_image
import os
from preprocessing import backgroud_removal

def process_images(directory,saving_dir):
    """get images local if in same directory as collab notebook"""
    directory_list = sorted(os.listdir(directory))
    success_counter = 0
    for i in range(len(directory_list)):
        print(f"Getting images of {directory_list[i]}:")
        for image in os.listdir(directory + "/" + directory_list[i]):
            img = cv2.imread(directory + "/" + directory_list[i] + "/" + image)
            img = crop_image(img)
            img = backgroud_removal(img)
            if img.shape[0] > 1:
                img = cv2.resize(img, (56, 56))
                try:
                    os.mkdir(f"{saving_dir}/{directory_list[i]}")
                except:
                    pass
                cv2.imwrite(f"{saving_dir}/{directory_list[i]}/Image_{image}",img)
    print("Complete")

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
            labels.append(directory[i])
    return images, labels
