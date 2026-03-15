class EmotionResult:
    """
    Represents the result of an emotion prediction.
    """

    def __init__(self, emotion, confidence):
        self.emotion = emotion
        self.confidence = confidence

    def __str__(self):
        return f"Emotion: {self.emotion} | Confidence: {self.confidence:.2f}"