from deepface import DeepFace


def predict_emotion(image_path):
    """
    Predict emotion from image
    """

    result = DeepFace.analyze(
        img_path=image_path,
        actions=["emotion"],
        enforce_detection=False
    )

    emotion = result[0]["dominant_emotion"]
    confidence = float(max(result[0]["emotion"].values()))

    return emotion, confidence