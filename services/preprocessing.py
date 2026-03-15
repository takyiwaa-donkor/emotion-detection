import cv2
import numpy as np

def preprocess_face(image):
    """
    Resize and normalize face image
    """

    if image is None:
        raise ValueError("No face detected")

    image = image.astype(np.uint8)
    image = cv2.resize(image, (48, 48))
    image = image / 255.0

    return image