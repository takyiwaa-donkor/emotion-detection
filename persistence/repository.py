import csv
import os

import os
import csv

results = []

def save_result(image_path, emotion, confidence):
    os.makedirs("data", exist_ok=True)

    # Save to CSV
    with open("data/results.csv", "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([image_path, emotion, confidence])

    # Save to in-memory list
    results.append((image_path, emotion, confidence))

def get_all_results():
    return results

