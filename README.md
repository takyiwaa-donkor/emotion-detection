# How the Emotion Detector Works

This document explains how the emotion detection system in this project works, from an input image to a predicted emotion and confidence score.

## Local Set up

Add a requirements.txt in the repo root (same folder as main.py) so it matches what main.py tells users to install.
The old DOCS/requirements.txt has invalid lines (matplotlib.pyplot as plt, shutil); either fix that file or remove
duplicates so there’s one canonical list.

Suggested requirements.txt (repo root)
deepface
tensorflow
tf-keras
opencv-python
matplotlib
numpy
pandas
pytest
---

## Prerequisites
- **Python 3.12** — TensorFlow (used by DeepFace) does not support Python 3.13+ yet. Use 3.12 for a smooth install.
### Install
1. Clone this repository and open a terminal in the project root (the folder that contains `main.py`).
2. Create and activate a virtual environment:
   **macOS / Linux**
   ```bash
   python3.12 -m venv venv
   source venv/bin/activate
   
## Windows (Command Prompt)
python3.12 -m venv venv
venv\Scripts\activate

3. Install dependencies:
   pip install --upgrade pip
   pip install -r requirements.txt

## Run
From the project root, with the virtual environment activated:
python main.py
You’ll get a text menu: single-image detection, batch folder, view past results, or exit. 
For single image, the app may open a matplotlib 
window to show the photo — use a machine with a display (or adapt the code for headless use).

First run
The first time you detect emotions, DeepFace may download model weights 
(and related files) to a cache directory on your machine (often under ~/.deepface/). That requires a network connection once; later runs are faster.

Run tests (optional)
pytest

## Troubleshooting
ModuleNotFoundError for deepface (or similar) — Activate the venv and run pip install -r requirements.txt again from the project root.
Python version errors when installing TensorFlow — Use Python 3.12, not 3.13+.



## Overview

The emotion detector takes an **image path** as input and returns:

- **Emotion** — One of a fixed set of labels (e.g. angry, happy, sad, surprise, neutral, fear, disgust).
- **Confidence** — A score (typically 0–100%) indicating how confident the model is in that prediction.

The implementation is built on the **DeepFace** library, which handles face detection and emotion classification in one call. The project also includes supporting modules (image loading, Haar-based face detection, preprocessing) that can be used for other pipelines or experiments.

---

## High-Level Pipeline


```
Image file (path)
       ↓
  Load image (optional; DeepFace can load from path)
       ↓
  Face detection (find face region in the image)
       ↓
  Preprocessing (align / resize / normalize for the model)
       ↓
  Emotion model (neural network predicts probabilities per emotion)
       ↓
  Dominant emotion + confidence → EmotionResult
```

In the current app, **all of these steps are performed inside DeepFace** when you call `detect_emotion(image_path)`.

---

## Main Entry Point: `detect_emotion()`

The function used by the CLI and demos lives in **`services/emotion_detector.py`**:

```python
def detect_emotion(image_path):
    result = DeepFace.analyze(
        img_path=image_path,
        actions=["emotion"],
        enforce_detection=False
    )
    emotion = result[0]["dominant_emotion"]
    confidence = max(result[0]["emotion"].values())
    return EmotionResult(emotion, confidence)
```

### What this does

1. **`DeepFace.analyze()`**  
   - Loads the image from `image_path`.  
   - Runs **face detection** (e.g. RetinaFace or another backend) to find at least one face.  
   - If `enforce_detection=False`, it still tries to run the emotion model even when no face is found (you may get less meaningful results).  
   - Runs the **emotion** model on the detected face(s).

2. **`result`**  
   - A list of per-face results. `result[0]` is the first face.  
   - `result[0]["dominant_emotion"]` is the label with the highest score (e.g. `"angry"`, `"happy"`).  
   - `result[0]["emotion"]` is a dictionary of emotion labels to raw scores (e.g. `{"angry": 97.6, "happy": 0.5, ...}`).

3. **Confidence**  
   - Taken as the **maximum** of those scores (the score of the dominant emotion).  
   - Often used as a percentage (e.g. 97.62%).

4. **Return value**  
   - An **`EmotionResult`** object with `.emotion` and `.confidence`, which the CLI and other scripts use to display or save results.

---

## What DeepFace Does Internally

When `actions=["emotion"]` is used, DeepFace typically:

1. **Detects faces**  
   Uses a face detector (e.g. RetinaFace) to find bounding boxes and optionally facial landmarks.

2. **Aligns / crops the face**  
   Prepares a consistent crop of the face for the emotion model.

3. **Runs the emotion model**  
   Uses a pre-trained model (e.g. a small CNN trained on a facial expression dataset) that outputs a probability or score for each emotion class.

4. **Returns**  
   Per-face: `dominant_emotion` and the full `emotion` score dictionary.

The **model weights** (e.g. `facial_expression_model_weights.h5`) are downloaded on first use to a cache directory (e.g. `~/.deepface/weights/`). That file contains the trained neural network used for emotion classification.

---

## Emotion Labels

The set of emotions depends on the model DeepFace uses (often a FER-style model). Typical labels include:

- **angry**
- **disgust**
- **fear**
- **happy**
- **sad**
- **surprise**
- **neutral**

The detector returns exactly one **dominant** emotion per face (the one with the highest score) plus a confidence value derived from that score.

---

## Supporting Modules in This Project

The repo contains extra utilities that are **not** required for the current `detect_emotion()` path but support other scripts and tests:

| Module | Purpose |
|--------|--------|
| **`services/image_loader.py`** | Loads an image from disk with basic validation (file exists, readable). Used in tests and demos that pass image arrays instead of paths. |
| **`services/face_detector.py`** | Uses OpenCV’s **Haar Cascade** (`haarcascade_frontalface_default.xml`) to find one face in an image and return a cropped face region. Alternative to DeepFace’s built-in detector. |
| **`services/preprocessing.py`** | Resizes a face image to 48×48 and normalizes pixel values to [0, 1]. Useful if you feed custom crops into another model. |
| **`services/emotion_classifier.py`** | Thin wrapper around DeepFace that returns `(emotion, confidence)` instead of an `EmotionResult` object. Same underlying pipeline. |
| **`domain/emotion_result.py`** | Defines the `EmotionResult` type (emotion + confidence) and a string representation. |

The **CLI** (`main.py`) only uses **`services/emotion_detector.detect_emotion()`**; the rest are available for batch scripts, tests, or future extensions.

---

## Data Flow in the CLI

1. User chooses “Detect Emotion (Single Image)” or “Detect Emotions (Batch Folder)” and provides a path (file or folder).
2. For each image, the app calls **`detect_emotion(path)`**.
3. DeepFace loads the image, finds the face, runs the emotion model, and returns per-face results.
4. The app reads **dominant_emotion** and **max(emotion.values())** and builds an **`EmotionResult`**.
5. The result is printed and optionally appended to **`data/results.csv`** (path, emotion, confidence).

---

## Limitations and Tips


- **One face per image**
  The current implementation only processes the first detected face (result[0]), so if an image contains multiple people with different emotions, 
  only the dominant/first face is analysed. Supporting multi‑face detection would require iterating over all detected faces and classifying each one individually.

- **Image quality**  
  Clear, front-facing faces under reasonable lighting give more stable and interpretable confidence scores.

- **Model bias**  
  Pre-trained emotion models can reflect biases from their training data (e.g. culture, demographics). Use results as a signal, not a ground truth.

- **First run**  
  The first time you run the detector, DeepFace may download the emotion model weights; after that, runs are faster.

- **`enforce_detection=False`**  
  With this setting, DeepFace still tries to analyze the image even if no face is found, which can produce low-confidence or misleading labels. For production, you may want `enforce_detection=True` and handle “no face” explicitly.

---

## Summary

| Step | Where it happens | Output |
|------|------------------|--------|
| Load image | DeepFace (from path) | Pixel array |
| Detect face | DeepFace (e.g. RetinaFace) | Face region(s) |
| Preprocess | DeepFace (internal) | Model-ready crop |
| Emotion model | DeepFace (e.g. FER weights) | Scores per emotion |
| Final result | `emotion_detector.detect_emotion()` | `EmotionResult(emotion, confidence)` |

The emotion detector is therefore a thin wrapper around **DeepFace’s analyze API** with a fixed `actions=["emotion"]` and a simple result type (`EmotionResult`) used by the rest of the application.
