import os
from services.emotion_detector import detect_emotion


def test_batch_detection():

    folder = "test_images"

    assert os.path.exists(folder)

    for img in os.listdir(folder):
        if img.lower().endswith((".jpg", ".png", ".jpeg")):

            path = os.path.join(folder, img)

            result = detect_emotion(path)

            assert result.emotion is not None
            assert result.confidence > 0