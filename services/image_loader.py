import cv2
import os


def load_image(image_path):
    """
    Loads an image from disk.
    """

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"{image_path} not found")

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Image could not be loaded.")

    return image