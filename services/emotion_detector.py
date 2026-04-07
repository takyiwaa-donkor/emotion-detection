"""
Emotion detection service module.

This module provides functions for performing emotion recognition using
DeepFace and TensorFlow. It handles image loading, preprocessing,
model inference, and packaging results into structured objects.
"""

from deepface import DeepFace


class EmotionResult:

    def __init__(self, emotion, confidence):
        self.emotion = emotion
        self.confidence = confidence


def detect_emotion(image_path):
    """
    Detects the dominant emotion in a given image.

    Parameters:
        image_path (str): Path to the input image file.

    Returns:
        EmotionResult: A dataclass containing:
            - emotion (str): Predicted emotion label.
            - confidence (float): Confidence score (0–100).
            - image_path (str): Original image path.

    Raises:
        FileNotFoundError: If the image does not exist.
        ValueError: If no face is detected in the image.
    """

    result = DeepFace.analyze(
        img_path=image_path,
        actions=["emotion"],
        enforce_detection=False
    )

    emotion = result[0]["dominant_emotion"]
    confidence = max(result[0]["emotion"].values())

    return EmotionResult(emotion, confidence)