import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from collections import defaultdict
import pickle
from text_preprocessorLog import tokenizeDF
from text_preprocessorLog import readFromTxt

def characteristicsExtraction(X_val):
    # Characteristics Extraction. 
    #We use the pretrained model of tfidf_vectorizer
    with open('./../ia-model/clean-data/tfidf_vectorizer.pkl', 'rb') as f:  
        try:
            tfidf_vectorizer  = pickle.load(f)
            X_val  = pd.DataFrame(tfidf_vectorizer.transform(X_val).toarray())
            X_val.columns = tfidf_vectorizer.get_feature_names_out()
            return X_val
        except UnicodeDecodeError:
        # Try other encodings (e.g., 'latin-1') if utf-8 fails
            print(f'Found: {UnicodeDecodeError}')

def LogisticRegressionModel(dataframe):
    with open('./../ia-model/clean-data/logisticRegModel.pkl', 'rb') as f:
        try:
            emotions3_model = pickle.load(f)
            X_test = characteristicsExtraction(dataframe['t_text'])
            print('--------')
            y_test = emotions3_model.predict(X_test)
            return getResults(prediction=y_test,X_val=X_test,modelName=emotions3_model)
        except UnicodeDecodeError:
        # Try other encodings (e.g., 'latin-1') if utf-8 fails
            print(f'Found: {UnicodeDecodeError}')
        



def getResults(prediction,X_val,modelName):
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
      e = emotion_classification[emotion]
      average_emotion_probabilities[e] = round(average_probability*100,3)

# Save results in a dataframe
  result = pd.DataFrame.from_dict(average_emotion_probabilities, orient='index', columns=['Average Probability'])
  return result

def run():
    df = readFromTxt('./../test.txt')
    #print(df.head(len(df)))
    df = tokenizeDF(df)
    #print('*******')
    #print(df.head(len(df)))
    #print('--------------')
    #print('chars')
    print(LogisticRegressionModel(df).head(6))



if __name__ == '__main__':
    run()