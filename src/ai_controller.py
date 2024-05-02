from src.models.GRU_model import GRUEmotionPredictor
from src.models.bert_model import EmotionBERTPredictor
from src.models.emotion_model import IEmotionPredictor
from src.services.emotion_analysis_service import EmotionAnalysisService

from src.models.logistic_model import EmotionLogisticPredictor


class EmotionController:
    def __init__(self):
        """
        Initialize the EmotionController instance, setting up available models and an optional initial service setup.
        """
        self.model_options = {
            0: EmotionLogisticPredictor(),
            1: GRUEmotionPredictor(),
            2: EmotionBERTPredictor()
        }

    def run_analysis(self, chosen_model, text):
        """
        Conducts emotion analysis on text provided by the user, utilizing the selected predictive model.
        """

        self.service = EmotionAnalysisService(chosen_model)

        prediction_label, description_label, percentage = self.service.analyze_text(text)

        return prediction_label, description_label, percentage
