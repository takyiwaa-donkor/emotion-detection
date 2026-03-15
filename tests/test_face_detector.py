from services.face_detector import detect_face
from services.image_loader import load_image

def test_detect_face():
    image = load_image("image3.jpg")
    face = detect_face(image)

    assert face is not None

def test_no_face_detected():
    import numpy as np
    from services.face_detector import detect_face

    blank = np.zeros((200,200,3))

    face = detect_face(blank)

    assert face is None

import cv2
import numpy as np

def detect_face(image):

    image = image.astype("uint8")

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray,1.3,5)

    return faces