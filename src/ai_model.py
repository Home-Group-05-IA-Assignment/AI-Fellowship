from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


class EmotionPredictor:
    """
    A class designed to predict emotions from textual input using pre-trained models
    from the Hugging Face Transformers library.

    This class simplifies the process of emotion prediction by abstracting the complexities
    of model loading, text tokenization, and inference. It is designed to work with any
    sequence classification models available on Hugging Face's model hub that are purposed
    for emotion analysis.

    Attributes:
        tokenizer (AutoTokenizer): Tokenizer loaded from the specified model_id, responsible
                                   for converting text input into a format suitable for the model.
        model (AutoModelForSequenceClassification): The pre-trained model loaded from Hugging Face,
                                                    used for predicting emotions from text.
        device (torch.device): The device (CPU/GPU) where the model and inputs are allocated for inference.

    Methods:
        __init__(model_id: str)
            Initializes the emotion predictor class by loading the tokenizer and model based on model_id.
            Automatically detects and uses GPU for inference if available.

        predict_emotion(text: str) -> int
            Predicts the emotion of a given piece of text. Returns the class ID of the predicted emotion.
    """

    def __init__(self, model_id="Valwolfor/distilbert_emotions_fellowship"):
        """
        Initializes the EmotionPredictor instance by loading the model and tokenizer from
        the specified Hugging Face model repository identifier (model_id). It also sets up the
        device for inference (auto-selects GPU if available, otherwise uses CPU).

        Args:
            model_id (str): A string identifier for the Hugging Face repository where
                            the model and tokenizer are hosted. Example: "bert-base-uncased".
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id).to(self.device)

    def predict_emotion(self, text):
        """
        Tokenizes the input text and performs an inference to predict the emotion. This method
        automatically manages the device allocation for the input and utilizes the loaded model
        to predict the emotion, returning the identified class ID.

        Args:
            text (str): The textual input for which the emotion is to be predicted.

        Returns:
            int: The class ID representing the predicted emotion from the provided text.
        """
        # Tokenize the input text and move tensors to the appropriate device
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(
            self.device)

        # Perform inference without gradient calculations
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Extract and return the predicted class ID
        predicted_class_id = outputs.logits.argmax(dim=1).item()
        return predicted_class_id