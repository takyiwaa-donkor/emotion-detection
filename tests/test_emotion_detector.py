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
