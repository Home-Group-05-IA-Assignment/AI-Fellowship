from typing import Tuple, Any

from models.emotion_model import IEmotionPredictor
from models.logistic_model import EmotionLogisticPredictor
from utils.input_handler import TextHandler
from utils.text_preprocessor import TextPreprocessor


class EmotionAnalysisService:
    """
    A service class that performs emotion analysis on given text inputs.
    Instead of creating a specific emotion predictor, this service receives any implementation of IEmotionPredictor
    through its constructor (dependency injection).

    Attributes:
        emotion_predictor (IEmotionPredictor): An emotion predictor that is capable of determining emotions in text.
    """

    def __init__(self, emotion_predictor: IEmotionPredictor):
        """
        Initializes the service with a specific implementation of IEmotionPredictor.

        Args:
            emotion_predictor (IEmotionPredictor): An implementation of the emotion predictor interface.
        """

        self.emotion_predictor: IEmotionPredictor = emotion_predictor
        self.text_handler = TextHandler()
        self.text_processor = TextPreprocessor()

    def analyze_text(self, text):
        """
        Uses the injected emotion_predictor to analyze the given text, translate, process and determine its emotion.

        Args:
            text (str): The text to analyze.

        Returns:
            Tuple[int, float]: A tuple containing the class ID and the probability of the predicted
                               emotion from the provided text.
            input_language (str): Original language of input text. Defaults to 'en'.
        """

        # Language detection and optional translation
        translated = self.text_handler.detect_language(text)

        is_logistic = isinstance(self.emotion_predictor, EmotionLogisticPredictor)
        processed_text = self.text_processor.preprocess_text(translated, is_logistic)
        
        # Prediction
        prediction, percentage = self.emotion_predictor.predict_emotion(processed_text)
        # Mapping label and description
        prediction_label, description_label = self.text_handler.get_emotion_description(prediction)

        return prediction_label, description_label, percentage
