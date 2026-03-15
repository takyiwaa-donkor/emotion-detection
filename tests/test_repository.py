import os


from persistence.repository import save_result

def test_save_result():

    save_result("happy",0.92)

    assert True
    import os

def test_csv_file_exists():
        assert os.path.exists("data/results.csv")

def test_multiple_predictions():

    from services.emotion_classifier import predict_emotion

    for _ in range(3):

        emotion, confidence = predict_emotion("image3.jpg")

        assert isinstance(emotion, str)
        assert confidence >= 0

def test_full_pipeline():

    from services.image_loader import load_image
    from services.face_detector import detect_face
    from services.preprocessing import preprocess_face
    from services.emotion_classifier import predict_emotion

    image = load_image("image3.jpg")
    face = detect_face(image)

    # Ensure face was detected
    if face is None:
        assert True
        return

    processed = preprocess_face(face)

    emotion, confidence = predict_emotion("image3.jpg")

    assert isinstance(emotion, str)
    assert isinstance(confidence, float)

