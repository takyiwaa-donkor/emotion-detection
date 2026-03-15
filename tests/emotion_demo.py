from services.emotion_detector import detect_emotion

images = {
    "Happy": "test_images/image3.jpg",
    "Sad": "test_images/image5.jpg",
    "Neutral": "test_images/image4.jpg",
    "Angry": "test_images/"
             ""
             ""
             ""
             "image1.jpg",
    "Surprise": "test_images/image6.jpg",
    "Disgust": "test_images/image2.jpg"

}

for label, path in images.items():

    try:
        result = detect_emotion(path)

        print(f"Image: {label}")
        print(f"Detected Emotion: {result.emotion}")
        print(f"Confidence: {result.confidence:.2f}%")
        print("--------------------------")

    except Exception as e:
        print("Error with", path, e)