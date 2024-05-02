from src.models.GRU_model import GRUEmotionPredictor
from src.models.bert_model import EmotionBERTPredictor
from src.services.emotion_service import EmotionAnalysisService
import textwrap
from IPython.display import Markdown
from src.models.logistic_model import EmotionLogisticPredictor
from src.services.gemini_service import ChatService


class EmotionController:
    def __init__(self):
        """
        Initialize the EmotionController instance, setting up available models and an optional initial service setup.
        """
        self.service = None
        self.model_options = {
            0: EmotionLogisticPredictor(),
            1: GRUEmotionPredictor(),
            2: EmotionBERTPredictor()
        }
        self.gemini_service = ChatService()

    def run_analysis(self, chosen_model, text):
        """
        Conducts emotion analysis on text provided by the user, utilizing the selected predictive model.
        """

        self.service = EmotionAnalysisService(chosen_model)

        prediction_label, description_label, percentage = self.service.analyze_text(text)

        return prediction_label, description_label, percentage

    def to_markdown(self, text):
        text_replaced = text.replace('•', '  *')
        return Markdown(textwrap.indent(text_replaced, '> ', predicate=lambda _: True))

    def gemini_controller(self, parameter, message):
        chat = self.gemini_service.start_chat()
        response = chat.send_message(parameter, message, chat)

        return response
