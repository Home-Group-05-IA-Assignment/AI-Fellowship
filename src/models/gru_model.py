import joblib
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

class EmotionGruPredictor():

    def __init__(self):
        """Load pretrained models."""
        self.model = load_model('./src/models/model-repository/gru-model/modelo_gru.h5')
        self.tokenizer = joblib.load('./src/models/model-repository/gru-model/tokenizer.joblib')
        self.emotion_classification = {0:'Sadness',1:'Joy',2:'Love',3:'Anger',4:'Fear',5:'Surprise'}

    def analyze_text(self, text):
        """
        Predicts the emotion of the given text.
        """
        text_secuencia = self.tokenizer.texts_to_sequences([text])
        text_padded = pad_sequences(text_secuencia, maxlen=79, padding='post')
        prediction = self.model.predict(text_padded)

        emotion_probabilities = {}
        for idx, prob in enumerate(prediction[0]):
            emotion = self.emotion_classification[idx]
            probability_percentage = round(prob * 100, 4)
            emotion_probabilities[emotion] = probability_percentage

        return emotion_probabilities