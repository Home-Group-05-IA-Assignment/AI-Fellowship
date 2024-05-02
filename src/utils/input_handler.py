from langdetect import detect
import os
from dotenv import load_dotenv
import google.generativeai as genai

from services.gemini_service import ChatService


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

        detect_language(text: str) -> str:
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
        self.chat = ChatService()

    def translate_from_spanish(self, text):
        """
        Translates Spanish text to English using Gemini's model. Currently acts as a placeholder
        and should be implemented or integrated with actual translation services.

        Args:
            text (str): Spanish text to be translated to English.

        Returns:
            str: Translated text in English.
        """
        chat_started = self.chat.start_chat()

        parameter = "Translate this text to English without adding anything else: "
        # Here, assume `send_message` effectively translates the input text.
        response = self.chat.send_message(parameter, text, chat_started)

        return response

    def detect_language(self, text):
        """
        Determines the language of the provided text using langdetect. Suggests translation
        if the detected language is Spanish.

        Args:
            text (str): Text to perform language detection on.

        Returns:
            str: The text, translated if Spanish.
        """
        try:
            input_language = detect(text)
            if input_language == 'es':
                text = self.translate_from_spanish(text)
            return text
        except Exception as e:
            print(f"Error detecting language: {e}")
            return text

    def get_emotion_description(self, predicted_class_id):
        """
        Matches a predicted emotion class ID to its corresponding human-readable emotion and
        description. Optionally translates the description to Spanish.

        Args:
            predicted_class_id (int): ID of the predicted emotion.

        Returns:
            Tuple[str, str]: Emotion and its descriptive message, translated to Spanish if applicable.
        """

        return self.emotion_mapping[predicted_class_id]
