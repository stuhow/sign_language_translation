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
    for image in os.listdir(directory + "/" + directory_list[i])[:100]:
      img = cv2.imread(directory + "/" + directory_list[i] + "/" + image)
      # print(img.shape)
      # img = crop_image(image)
      img = cv2.resize(img, (28, 28))
      images.append(img)
      labels.append(directory_list[i])
    # print(f"Got images of {directory_list[i]}: {len(images)}")
  return images, labels
