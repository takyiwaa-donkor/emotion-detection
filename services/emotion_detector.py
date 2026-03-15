from deepface import DeepFace


class EmotionResult:

    def __init__(self, emotion, confidence):
        self.emotion = emotion
        self.confidence = confidence


def detect_emotion(image_path):
    """
    Detect emotion from a facial image
    """

    result = DeepFace.analyze(
        img_path=image_path,
        actions=["emotion"],
        enforce_detection=False
    )

    emotion = result[0]["dominant_emotion"]
    confidence = max(result[0]["emotion"].values())

    return EmotionResult(emotion, confidence)