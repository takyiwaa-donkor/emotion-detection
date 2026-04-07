"""
Application controller for coordinating emotion detection workflows.

This module acts as the intermediary between:
- The emotion detection service (business logic)
- The persistence layer (CSV logging + in-memory storage)
- The CLI or any future UI (GUI, API, web interface)

The controller ensures that the application follows a clean separation
of concerns by keeping high-level workflow logic separate from
model inference and data storage.
"""

from services.emotion_detector import detect_emotion
from persistence.repository import save_result, get_all_results


class EmotionController:
    """
        Controller class responsible for managing emotion analysis operations.

        This class provides high-level methods that:
        - Accept input from the user interface (CLI or GUI)
        - Call the emotion detection service
        - Save results using the persistence layer
        - Return structured results back to the UI

        It ensures that the UI layer does not directly interact with
        lower-level services or storage components.
        """

    def analyze_image(self, image_path: str):
        """
                Analyzes a single image and stores the detection result.

                Parameters:
                    image_path (str): Path to the image file to be analyzed.

                Returns:
                    EmotionResult: A structured object containing:
                        - emotion (str): Predicted emotion label
                        - confidence (float): Confidence score
                        - image_path (str): Original image path

                Workflow:
                    1. Calls the emotion detection service.
                    2. Saves the result using the persistence layer.
                    3. Returns the result to the caller.

                Raises:
                    FileNotFoundError: If the image path does not exist.
                    ValueError: If no face is detected in the image.
                """

        result = detect_emotion(image_path)

        save_result(
            image_path,
            result.emotion,
            result.confidence
        )

        return result

    def show_results(self):
        """
                Retrieves all previously stored emotion detection results.

                Returns:
                    list[tuple]: A list of tuples in the format:
                        (image_path, emotion, confidence)

                Notes:
                    - Results are retrieved from in-memory storage.
                    - CSV persistence ensures results are not lost between sessions.
                """

        return get_all_results()