import pickle

from models.emotion_model import IEmotionPredictor


class EmotionLogisticPredictor(IEmotionPredictor):
    """
    A class for loading pretrained models and predicting emotions for a given text.

    Attributes:
    ----------
    tfidf_vectorizer : The TF-IDF vectorizer model.
    emotions_model : The logistic regression model for emotion prediction.
    emotion_classification : A dictionary for mapping numeric predictions to emotion labels.
    """

    def __init__(self):
        """Load pretrained models."""
        with open('./models/model-repository/logistic-reg-model/tfidf_vectorizer.pkl', 'rb') as f_tfidf:
            self.tfidf_vectorizer = pickle.load(f_tfidf)

        with open('./models/model-repository/logistic-reg-model/logisticRegModel.pkl', 'rb') as f_logreg:
            self.emotions_model = pickle.load(f_logreg)

    def predict_emotion(self, text: str):
        """
        Predict emotions for a single text.

        Parameters:
        ----------
        text : str
            A string to predict emotions for tokenized.

        Returns:
        -------
        Tuple[int, float]: A tuple containing the class ID and the probability of the predicted
                           emotion from the provided text.
        """
        if isinstance(text, list):
            text = ' '.join(text)

        if not text.strip():
            raise ValueError("The input text is empty.")

        if isinstance(text, str):
            texts = [text]
        else:
            texts = text

        # Vectorize the text
        X = self.tfidf_vectorizer.transform(texts)

        # Predict probabilities for each emotion
        predicted_probabilities = self.emotions_model.predict_proba(X)

        # Find the index (class ID) of the maximum probability and its value
        predicted_class_id = predicted_probabilities.argmax(axis=1)[0]
        predicted_probability = predicted_probabilities.max(axis=1)[0]

        return predicted_class_id, predicted_probability
