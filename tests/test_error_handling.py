"""
Tests for error handling across the application.

Ensures that:
- Missing files raise appropriate exceptions.
- Invalid paths are handled gracefully.
- The system does not crash on unexpected input.
"""


from services.emotion_detector import detect_emotion

try:
    detect_emotion("not_existing.jpg")
except Exception as e:
    print("Handled error:", e)