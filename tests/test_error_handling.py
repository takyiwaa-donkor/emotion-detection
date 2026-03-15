from services.emotion_detector import detect_emotion

try:
    detect_emotion("not_existing.jpg")
except Exception as e:
    print("Handled error:", e)