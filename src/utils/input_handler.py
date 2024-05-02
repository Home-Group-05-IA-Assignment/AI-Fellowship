from langdetect import detect
import os
from dotenv import load_dotenv
import google.generativeai as genai


class TextHandler:
    """
    Manages text-related operations including language detection and providing human-readable
    descriptions based on emotion prediction outcomes. Note: Translation method is a placeholder
    and requires actual implementation for production use.

    Attributes:
        emotion_mapping (dict): Maps predicted emotion class IDs to tuples
                                containing the emotion as a string and a descriptive message.

    Methods:
        __init__: Initializes the TextHandler with predefined emotion mappings.

        translate_from_spanish(text: str) -> str:
            Translates Spanish text to English. Placeholder method.

        detect_language(text: str) -> (str, str):
            Detects the language of the input text. Suggests translation for Spanish texts.

        get_emotion_description(predicted_class_id: int, input_language: str = 'en') -> (str, str):
            Retrieves a human-readable emotion and description based on a predicted class ID.
            Translates the description to Spanish if input text was in Spanish.
    """

    def __init__(self):
        """
        Initializes the TextHandler instance by setting up predefined emotion mappings and
        loading necessary configurations, such as the API key for translation services.
        """
        self.emotion_mapping = {
            0: ("Sadness", "I sense sadness in this text."),
            1: ("Joy", "This text expresses joy."),
            2: ("Love", "Love is in the air."),
            3: ("Anger", "There's noticeable anger here."),
            4: ("Fear", "A sentiment of fear is detected."),
            5: ("Surprise", "What a surprise in this message!")
        }

        load_dotenv()
        self.GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        self.gemini_model = genai.GenerativeModel('gemini-pro')

    def translate_from_spanish(self, text: str) -> str:
        """
        Translates Spanish text to English using Gemini's model. Currently acts as a placeholder
        and should be implemented or integrated with actual translation services.

        Args:
            text (str): Spanish text to be translated to English.

        Returns:
            str: Translated text in English.
        """
        chat = self.gemini_model.start_chat(history=[])
        parameter = "Translate this text to English without adding anything else: "
        # Here, assume `send_message` effectively translates the input text.
        response = chat.send_message(parameter + text)
        return response.text

    def detect_language(self, text: str) -> (str, str):
        """
        Determines the language of the provided text using langdetect. Suggests translation
        if the detected language is Spanish.

        Args:
            text (str): Text to perform language detection on.

        Returns:
            Tuple[str, str]: Detected language (ISO 639-1 code) and the text, translated if Spanish.
        """
        try:
            input_language = detect(text)
            if input_language == 'es':
                text = self.translate_from_spanish(text)
            return input_language, text
        except Exception as e:
            print(f"Error detecting language: {e}")
            return 'unknown', text

    def get_emotion_description(self, predicted_class_id: int, input_language: str = 'en') -> (str, str):
        """
        Matches a predicted emotion class ID to its corresponding human-readable emotion and
        description. Optionally translates the description to Spanish.

        Args:
            predicted_class_id (int): ID of the predicted emotion.
            input_language (str): Original language of input text. Defaults to 'en'.

        Returns:
            Tuple[str, str]: Emotion and its descriptive message, translated to Spanish if applicable.
        """
        emotion, description = self.emotion_mapping[predicted_class_id]
        if input_language == 'es':
            description = self.translate_from_spanish(description)
        return emotion, description