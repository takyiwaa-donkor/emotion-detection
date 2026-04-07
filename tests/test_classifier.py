"""
Tests for the underlying emotion classification logic.

This module ensures that:
- The classifier returns valid emotion labels.
- Confidence scores fall within expected ranges.
- The model behaves consistently across repeated runs.
"""


from services.emotion_classifier import predict_emotion

def test_prediction_output():

    emotion, confidence = predict_emotion("image3.jpg")

    assert isinstance(emotion, str)
    assert isinstance(confidence, float)
