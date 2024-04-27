from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class EmotionPredictor:
    """
    A class for predicting emotions from a text input using a pre-trained model from Hugging Face.
    sadness (0), joy (1), love (2), anger (3), fear (4) and surprise (5)
    """

    def __init__(self, model_id):
        """
        Initializes the emotion predictor by loading the model and tokenizer.

        Args:
            model_id (str): The identifier for the repository on Hugging Face where the model is stored.
        """
        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Load the model
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id)

    def predict_emotion(self, text):
        """
        Predicts the emotion of a given text.

        Args:
            text (str): The text to predict the emotion for.

        Returns:
            int: The predicted class ID of the emotion.
        """
        # Tokenize the input text
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)

        # Perform the prediction
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Extract and return the predicted class ID
        predicted_class_id = outputs.logits.argmax(dim=1).item()

        return predicted_class_id