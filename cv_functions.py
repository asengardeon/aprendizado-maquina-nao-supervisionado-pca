import os

from cv2 import cv2


def convert_to_gray_scale(image):
   return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def equalize_histogram(image):
    return cv2.equalizeHist(image)

def resize_image(image):
    return cv2.resize(image, (80, 80), interpolation=cv2.INTER_AREA)

def read_image_data(folder, filename):
    image = cv2.imread(os.path.join(folder, filename))
    if image is not None:
        gray_image = convert_to_gray_scale(image)
        # gray_image = equalize_histogram(gray_image)
        gray_image = resize_image(gray_image)
        return gray_image
    return None

