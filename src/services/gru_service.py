from nltk.corpus import stopwords
import re
from models.gru_model import EmotionGruPredictor

class GruModelService:

    def analyze_text(self, text):
        """
        Uses the injected emotion_predictor to analyze the given text, translate, process and determine its emotion.

        Args:
            text (str): The text to analyze.

        Returns:
            Tuple[int, float]: A tuple containing the class ID and the probability of the predicted
                               emotion from the provided text.
            input_language (str): Original language of input text. Defaults to 'en'.
        """
        
        # Paso 1: Eliminar URLs
        text = re.sub(r'http\S+', '', text)

        # Paso 2: Eliminar caracteres especiales
        text = re.sub(r'[^\w\s]', '', text)

        # Paso 3: Reducir múltiples espacios en blanco a uno solo
        text = re.sub(r'\s+', ' ', text)

        # Paso 4: Eliminar números
        text = re.sub(r'\d+', '', text)

        # Paso 5: Eliminar caracteres no alfabéticos (dejar solo letras)
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # Paso 6: Convertir texto a minúsculas
        text = text.lower()

        # Paso 7: Eliminar stop words
        stop = stopwords.words('english')
        text = ' '.join([word for word in text.split() if word not in stop])

        return EmotionGruPredictor().analyze_text(text)