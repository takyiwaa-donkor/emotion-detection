import cv2
import numpy as np

def preprocess_face(image):

    image = image.astype("uint8")

    image = cv2.resize(image,(224,224))

    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    gray = gray / 255.0

    return gray