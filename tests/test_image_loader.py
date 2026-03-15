from services.image_loader import load_image
import pytest

import cv2
import os

def load_image(image_path):

    if not os.path.exists(image_path):
        raise FileNotFoundError("Image not found.")

    image = cv2.imread(image_path)

    return image
