from abc import ABC, abstractmethod
from typing import Tuple


class IEmotionPredictor(ABC):
    """
    Abstract class (interface) for emotion predictors. Defines a common method signature for predicting emotions.
    """

    @abstractmethod
    def predict_emotion(self, text: str) -> Tuple[int, float]:
        """
        Abstract method to predict the emotion of a given piece of text.

        Args:
            text (str): The input text to predict the emotion for.

        Returns:
            Tuple[int, float]: A tuple containing the class ID and the probability of the predicted
                           emotion from the provided text.
        """
        pass