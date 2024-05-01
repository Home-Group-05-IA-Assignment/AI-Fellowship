import pandas as pd
import re

from string import punctuation
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.stem import WordNetLemmatizer
from spacy.lang.en.stop_words import STOP_WORDS
from wordcloud import STOPWORDS

import pickle
#nltk.download('punkt')
#nltk.download('wordnet')
#nltk.download('stopwords')

def readFromTxt(route):
    #first u have to open  the file and seperate every line like below:
    file = open(route, "r")
    lines = file.readlines()
    file.close()
    # remove /n at the end of each line
    for index, line in enumerate(lines):
      lines[index] = line.strip()
    #creating a dataframe
    df_result = pd.DataFrame(columns=('text', 'label'))
    #Assigning rows to the dataframe
    df_result['text'] = lines
    return df_result

def tokenizeDF(dataframe):
    # lowercase, digits and extra-spaces
    dataframe["t_text"] = dataframe["text"].str.lower()
    dataframe["t_text"] = dataframe["t_text"].apply(lambda x: re.sub(r"\d+","",x))
    dataframe["t_text"] = dataframe["t_text"].apply(lambda x: re.sub(r"\s+"," ",x))

    # links and special characters
    dataframe["t_text"] = dataframe["t_text"].apply(lambda x: re.sub(r"http\S+","",x))
    dataframe["t_text"] = dataframe["t_text"].apply(lambda x: re.sub(r"[^\w\s]","",x))

    #Tokenization and removing puntuation
    dataframe["t_text"] = dataframe["t_text"].apply(lambda x:word_tokenize(x))
    dataframe["t_text"] = dataframe["t_text"].apply(lambda items:[item for item in items if item not in punctuation])

    #Stop words
    stop_words = stopwords.words("english")
    com_stop_words = stop_words + list(STOP_WORDS) + list(STOPWORDS)
    print(len(com_stop_words))
    print(len(set(com_stop_words)))

    dataframe["t_text"] = dataframe["t_text"].apply(lambda words:[word for word in words if word not in com_stop_words])
    #Lemmatization
    lem = WordNetLemmatizer()
    dataframe["t_text"] = dataframe["t_text"].apply(lambda words:" ".join([lem.lemmatize(word,pos="v") for word in words]))
    return dataframe
    


def run():
    df = readFromTxt('./../test.txt')
    print(df.head())
    print('tokenize')
    df = tokenizeDF(df)
    print(df.head())

if '__main__' == __name__:
    run()