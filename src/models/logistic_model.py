import pickle
import pandas as pd
from collections import defaultdict
import pickle
from models.emotion_model import IEmotionPredictor


class EmotionLogisticPredictor(IEmotionPredictor):
    """
    A class for loading pretrained models and predicting emotions for a given text.

    Attributes:
    ----------
    tfidf_vectorizer : The TF-IDF vectorizer model.
    LogisticRegressionModel : The logistic regression model for emotion prediction.
    emotion_classification : A dictionary for mapping numeric predictions to emotion labels.
    """

    def characteristicsExtraction(self,X_val):
    # Characteristics Extraction. 
    #We use the pretrained model of tfidf_vectorizer
        X_val  = pd.DataFrame(self.tfidf_vectorizer.transform(X_val).toarray())
        X_val.columns = self.tfidf_vectorizer.get_feature_names_out()
        return X_val
    

    def LogisticRegressionModel(self,text_dataframe):
            X_test = self.characteristicsExtraction(text_dataframe['t_text'])
            #returns a dataframe with all amotions
            return self.getResults(X_val=X_test,modelName=self.emotions_model)

    def __init__(self):
        """Load pretrained models."""
        with open('./src/models/model-repository/logistic-reg-model/tfidf_vectorizer.pkl', 'rb') as f_tfidf:
        #with open('./models/model-repository/logistic-reg-model/tfidf_vectorizer.pkl','rb') as f_tfidf:
            self.tfidf_vectorizer = pickle.load(f_tfidf)

        with open('./src/models/model-repository/logistic-reg-model/logisticRegModel.pkl', 'rb') as f_logreg:
        #with open('./models/model-repository/logistic-reg-model/logisticRegModel.pkl','rb') as f_logreg:
            self.emotions_model = pickle.load(f_logreg)

    def getEmotionsArrays(self,modelName,dataframe):
        emotion_classification = {0:'Sadness',1:'Joy',2:'Love',3:'Anger',4:'Fear',5:'Surprise'}
        X_val = self.characteristicsExtraction(dataframe['t_text'])
        num_array = modelName.predict(X_val)
        emotions_array = []
        for i in range(len(num_array)):
            emotions_array.append(emotion_classification[num_array[i]])
        return emotions_array

    def getResults(self,X_val,modelName):
        emotion_classification = {0:'Sadness',1:'Joy',2:'Love',3:'Anger',4:'Fear',5:'Surprise'}
  # Create a dict to save average probabilities for each emotion
        emotion_probabilities = defaultdict(list)
        
        predicted_probabilities = modelName.predict_proba(X_val)
        

# Add probabilities for each emotion
 
        for probabilities in predicted_probabilities:
            for emotion, probability in zip(modelName.classes_, probabilities):
                emotion_probabilities[emotion].append(probability)

# Get the average of each emotion
        average_emotion_probabilities = {}
        for emotion, probabilities in emotion_probabilities.items():
            average_probability = sum(probabilities) / len(probabilities)
            e = emotion#emotion_classification[emotion]
            average_emotion_probabilities[e] = round(average_probability,3)

# Save results in a dataframe
        
        result = pd.DataFrame.from_dict(average_emotion_probabilities, orient='index', columns=['avg_prob'])
        return result


    def predict_emotion(self, text: pd.DataFrame):
        """
        Predict emotions for a  text.

        Parameters:
        ----------
        text : str
            A string to predict emotions for tokenized.

        Returns:
        -------
        Tuple[int, float]: A tuple containing the class ID and the probability of the predicted
                           emotion from the provided text.
                           may return a dataframe or a dict with the emotion and each prob
        """
    

        """We supose text is a dataframe which is returned from text_preprocessor"""
        
        df = self.LogisticRegressionModel(text)
        
        #return the index and the max probability
        predicted_class_id = df.avg_prob.idxmax()
        predicted_probability = df.iloc[predicted_class_id].avg_prob
        print('Retrieving from Logistic Regression')

        return predicted_class_id, predicted_probability
