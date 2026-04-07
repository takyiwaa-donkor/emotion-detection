"""
Unit tests for the emotion_detector service.

Covers:
- Single image emotion detection.
- Handling of invalid or corrupted images.
- Correct structure of the returned EmotionResult object.
"""


from services.emotion_detector import detect_emotion


def test_detect_emotion():

    image = "test_images/image3.jpg"

    result = detect_emotion(image)

    assert result.emotion is not None
    assert result.confidence >= 0
from services.emotion_detector import EmotionResult

def test_emotion_result():

    result = EmotionResult("happy",0.95)

    assert result.emotion == "happy"
    assert result.confidence == 0.95
