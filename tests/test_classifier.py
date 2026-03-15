from services.emotion_classifier import predict_emotion

def test_prediction_output():

    emotion, confidence = predict_emotion("image3.jpg")

    assert isinstance(emotion, str)
    assert isinstance(confidence, float)
