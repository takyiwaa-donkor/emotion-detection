import matplotlib.pyplot as plt
import matplotlib.image as mpimg


import os
import sys
import csv
import shutil
try:
    from services.emotion_detector import detect_emotion
except ModuleNotFoundError as e:
    if "deepface" in str(e).lower():
        print("DeepFace is not installed. Set up the project first:")
        print("  1. Create a virtual environment:  python3.12 -m venv venv")
        print("  2. Activate it:                   source venv/bin/activate   (macOS/Linux)")
        print("                                     or  venv\\Scripts\\activate   (Windows)")
        print("  3. Install dependencies:          pip install -r requirements.txt")
        print("\nYou need Python 3.12 (TensorFlow does not support 3.13+).")
        sys.exit(1)
    raise


RESULTS_FILE = "data/results.csv"


def detect_single_image():
    path = input("Enter image path: ").strip()

    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    try:
        result = detect_emotion(path)

        print(f"\nDetected Emotion: {result.emotion}")
        print(f"Confidence: {result.confidence:.2f}%")

        # Display the image
        img = mpimg.imread(path)
        plt.imshow(img)
        plt.title(f"{result.emotion} ({result.confidence:.2f}%)")
        plt.axis("off")
        plt.show()

        # Save result to CSV
        save_result(path, result.emotion, result.confidence)

        # --- Copy image into categorized folder ---
        output_folder = "categorized_results"
        os.makedirs(output_folder, exist_ok=True)

        emotion_folder = os.path.join(output_folder, result.emotion)
        os.makedirs(emotion_folder, exist_ok=True)

        dst_name = f"{result.emotion}_{int(result.confidence)}_{os.path.basename(path)}"
        dst_path = os.path.join(emotion_folder, dst_name)

        shutil.copy2(path, dst_path)
        print(f"Copied image to: {dst_path}")

    except Exception as e:
        print(f"Error: {e}")


def detect_batch_images():
    folder = input("Enter folder path: ").strip()

    if not os.path.isdir(folder):
        print(f"Folder not found: {folder}")
        return

    supported = (".jpg", ".jpeg", ".png", ".bmp")
    images = [f for f in os.listdir(folder) if f.lower().endswith(supported)]

    if not images:
        print("No supported images found in folder.")
        return

    print(f"\nProcessing {len(images)} image(s)...\n")
    print(f"{'File':<30}{'Emotion':<15}{'Confidence':<10}")
    print("-" * 55)

    for filename in sorted(images):
        path = os.path.join(folder, filename)

        try:
            result = detect_emotion(path)

            print(f"{filename:<30}{result.emotion:<15}{result.confidence:.2f}%")

            # New code: categorize into another folder
            output_folder = "categorized_results"  # base folder for categorized images
            os.makedirs(output_folder, exist_ok=True)  # create if doesn't exist

            # Create folder for this emotion
            emotion_folder = os.path.join(output_folder, result.emotion)
            os.makedirs(emotion_folder, exist_ok=True)

            # Destination filename with confidence to avoid conflicts
            dst_name = f"{result.emotion}_{int(result.confidence)}_{filename}"
            dst_path = os.path.join(emotion_folder, dst_name)

            shutil.copy2(path, dst_path)
            print(f"Copied to: {dst_path}")

        except Exception as e:
            print(f"{filename:<30}Error: {e}")


def view_results():
    if not os.path.exists(RESULTS_FILE):
        print("No results found.")
        return

    with open(RESULTS_FILE, "r") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        print("No results found.")
        return

    print(f"\n{'Image':<35}{'Emotion':<15}{'Confidence':<10}")
    print("-" * 60)
    for row in rows:
        if len(row) == 3:
            print(f"{row[0]:<35}{row[1]:<15}{row[2]}")


def save_result(image_path, emotion, confidence):
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(RESULTS_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([image_path, emotion, f"{confidence:.5f}"])


def main():
    while True:
        print("\n" + "=" * 35)
        print("     EMOTION DETECTION")
        print("=" * 35)
        print("1. Detect Emotion (Single Image)")
        print("2. Detect Emotions (Batch Folder)")
        print("3. View Past Results")
        print("4. Exit")

        choice = input("Select option: ").strip()

        if choice == "1":
            detect_single_image()
        elif choice == "2":
            detect_batch_images()
        elif choice == "3":
            view_results()
        elif choice == "4":
            print("Goodbye!")
            break
        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main()
