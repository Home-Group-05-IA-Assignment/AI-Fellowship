import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import json
from keras.preprocessing.text import tokenizer_from_json
from typing import Tuple

from src.models.emotion_model import IEmotionPredictor


class GRUEmotionPredictor(IEmotionPredictor):
    """
    Initialize the GRU emotion predictor by loading a model, tokenizer, and maxlen value.

    This class implements the IEmotionPredictor interface for predicting emotions from given text inputs.
    It utilizes a pre-trained GRU (Gated Recurrent Unit) neural network model along with a tokenizer for text processing.

    Args:
    -----
    model_path : str
        The path to the saved model file.
    tokenizer_path : str
        The path to the saved tokenizer JSON file.

    The model_path is expected to point to a .h5 file for a Keras model, while the tokenizer path
    should be a .json file containing the tokenizer used during training. The maxlen value specifies the
    maximum length of sequences used for padding inputs.
    """

    def __init__(self, model_path="./src/models/model-repository/gru-model/gru_model.h5",
                 tokenizer_path="./src/models/model-repository/gru-model/tokenizer.json"):
        """
        Initialize the GRU emotion predictor.
        """
        # Load the model
        self.model = load_model(model_path)

        # Load the tokenizer
        with open(tokenizer_path) as f:
            data = json.load(f)
            self.tokenizer = tokenizer_from_json(data)

        # Load maxlen
        self.maxlen = 79

    def predict_emotion(self, text: str) -> Tuple[int, float]:
        """
        Predicts the emotion of a given piece of text, returning the class ID and the probability of the predicted emotion.

        This method takes a text input, processes it using the pre-loaded tokenizer,
        pads it to the correct sequence length, and then performs inference using the loaded GRU model.

        Args:
        -----
        text : str
            The input text for which the emotion is to be predicted.

        Returns:
        -------
        Tuple[int, float]
            A tuple containing the class ID (integer) of the predicted emotion and the probability (float)
            for that prediction. The class ID corresponds to one of the predefined emotion categories.

        The method processes the input text by tokenizing it, padding the resulting sequence according to the
        loaded 'maxlen' value, and then performing inference with the GRU model. The returned class ID represents
        the predicted emotion category, and the probability indicates the model's confidence in this prediction.
        """
        # Prepare the text
        sequence = self.tokenizer.texts_to_sequences([text])
        padded_sequence = pad_sequences(sequence, maxlen=self.maxlen, padding='post')

        # Perform the prediction
        probabilities = self.model.predict(padded_sequence).flatten()

        # Get the ID of the class with the highest probability
        predicted_class_id = np.argmax(probabilities)

        # Get the probability of the predicted class
        predicted_probability = np.max(probabilities)

        return predicted_class_id, predicted_probability
