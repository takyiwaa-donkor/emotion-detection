"""
Performance tests for the Emotion Detection System.

Measures:
- Average processing time per image.
- Total time for batch processing.
- Model inference speed and stability.

Used to ensure the system meets performance expectations.
"""


import os
import time
from services.emotion_detector import detect_emotion


def test_performance():

    folder = "test_images"

    assert os.path.exists(folder)

    count = 0
    start = time.time()

    for img in os.listdir(folder):
        if img.lower().endswith((".jpg", ".png", ".jpeg")):

            path = os.path.join(folder, img)

            detect_emotion(path)
            count += 1

    end = time.time()

    total = end - start
    avg = total / count if count > 0 else 0

    print("\nImages processed:", count)
    print("Total time:", total)
    print("Average time per image:", avg)

    assert count > 0