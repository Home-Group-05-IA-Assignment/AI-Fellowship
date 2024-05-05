from models.bert_model import EmotionBERTPredictor
from services.emotion_service import EmotionAnalysisService
from models.logistic_model import EmotionLogisticPredictor
from services.gemini_service import ChatService
from services.gru_service import GruModelService

import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')


class EmotionController:
    def __init__(self):
        """
        Initialize the EmotionController instance, setting up available models and an optional initial service setup.
        """
        self.service = None
        self.model_options = {
            0: EmotionLogisticPredictor(),
            1: EmotionBERTPredictor(),
            2: GruModelService()
            
        }
        self.gemini_service = ChatService()

    def run_analysis(self, chosen_model, text):
        
        """
        Conducts emotion analysis on text provided by the user, utilizing the selected predictive model.
        """
        if chosen_model == 0:
            self.service = EmotionAnalysisService(EmotionLogisticPredictor())
        elif chosen_model == 1:
            self.service = EmotionAnalysisService(EmotionBERTPredictor())
        elif chosen_model == 2:
            self.service = GruModelService()
            return self.service.analyze_text(text)
        
        prediction_label, description_label, percentage = self.service.analyze_text(text)

        return prediction_label, description_label, percentage

    def gemini_controller(self, parameter, message):
        chat = self.gemini_service.start_chat()
        parameter += f" {message}"
        response = chat.send_message(parameter)
        return response.text
    

    def restartChat(self):
        self.gemini_service = ChatService()
        return "Chat restarted. How can I help you today?"