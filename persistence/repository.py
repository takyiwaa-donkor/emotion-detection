"""
Persistence module for saving and retrieving emotion detection results.

This module handles:
- Writing detection results to a CSV file.
- Maintaining an in-memory list of results for quick access.
- Ensuring the data directory exists before writing.
"""

import csv
import os
from datetime import datetime   # <-- Correct import


# In-memory storage for quick retrieval
results = []


def save_result(image_path: str, emotion: str, confidence: float):
    """
    Saves a single emotion detection result to both CSV and in-memory storage.

    Parameters:
        image_path (str): Path to the processed image.
        emotion (str): Predicted emotion label.
        confidence (float): Confidence score (0–100).

    Behavior:
        - Ensures the 'data' directory exists.
        - Appends the result to 'data/results.csv'.
        - Stores the result in the in-memory list for fast access.
    """
    os.makedirs("data", exist_ok=True)

    csv_path = os.path.join("data", "results.csv")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Save to CSV
    with open(csv_path, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, image_path, emotion, confidence])

    # Save to in-memory list
    results.append((timestamp, image_path, emotion, confidence))


def get_all_results():
    """
    Retrieves all stored emotion detection results.

    Returns:
        list[tuple]: A list of tuples in the format:
            (timestamp, image_path, emotion, confidence)
    """
    return results
