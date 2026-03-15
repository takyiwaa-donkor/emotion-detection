from services.emotion_detector import detect_emotion
from persistence.repository import save_result, get_all_results


class EmotionController:

    def analyze_image(self, image_path):

        result = detect_emotion(image_path)

        save_result(
            image_path,
            result.emotion,
            result.confidence
        )

        return result

    def show_results(self):

        return get_all_results()