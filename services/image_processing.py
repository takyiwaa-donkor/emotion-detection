import cv2

def resize_image(image, size=(224, 224)):
    """
    Resize image for model input.
    """

    return cv2.resize(image, size)


def convert_to_gray(image):
    """
    Convert image to grayscale.
    """

    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)