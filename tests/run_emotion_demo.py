"""
Runs a visual demonstration of emotion detection.

Loads a sample image, performs emotion analysis, and displays the
result using Matplotlib. Intended for quick visual verification.
"""



import cv2
import matplotlib.pyplot as plt
from services.emotion_detector import detect_emotion

path = "../test_images/image5.jpg"

result = detect_emotion(path)

img = cv2.imread(path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img)
plt.title(f"Emotion: {result.emotion} ({result.confidence:.2f}%)")
plt.axis("off")
plt.show()