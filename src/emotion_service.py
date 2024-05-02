from typing import Tuple

from src.ai_controller import text_processor


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
        self.emotion_predictor = emotion_predictor

    def analyze_text(self, text: str) -> Tuple[int, float]:
        """
        Uses the injected emotion_predictor to analyze the given text and determine its emotion.

        Args:
            text (str): The text to analyze.

        Returns:
            Tuple[int, float]: A tuple containing the class ID and the probability of the predicted
                               emotion from the provided text.
        """

        processed_text = text_processor.preprocess_text(text)
        self.emotion_predictor.predict_emotion(text)
        return
