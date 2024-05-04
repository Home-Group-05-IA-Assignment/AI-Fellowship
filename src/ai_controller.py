from models.bert_model import EmotionBERTPredictor
from services.emotion_service import EmotionAnalysisService
from models.logistic_model import EmotionLogisticPredictor
from services.gemini_service import ChatService


class EmotionController:
    def __init__(self):
        """
        Initialize the EmotionController instance, setting up available models and an optional initial service setup.
        """
        self.service = None
        self.model_options = {
            0: EmotionLogisticPredictor(),
            1: EmotionBERTPredictor()
        }
        self.gemini_service = ChatService()

    def run_analysis(self, chosen_model, text):
        
        """
        Conducts emotion analysis on text provided by the user, utilizing the selected predictive model.
        """
        if chosen_model == 0:
            self.service = EmotionAnalysisService(EmotionLogisticPredictor())
        else:
            self.service = EmotionAnalysisService(EmotionBERTPredictor())
        print(self.service)
        prediction_label, description_label, percentage = self.service.analyze_text(text)

        return prediction_label, description_label, percentage

    def gemini_controller(self, parameter, message):
        chat = self.gemini_service.start_chat()
        parameter += f" {message}"
        response = chat.send_message(parameter)
        return response.text

