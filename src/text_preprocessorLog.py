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


def readFromTxt(route):
    #first u have to open  the file and seperate every line like below:
    file = open(route, "r")
    lines = file.readlines()
    file.close()
    #creating a dataframe
    text = []
    df_result = pd.DataFrame(columns=('text', 't_text'))
    # remove /n at the end of each line
    for line in lines:
         line = line.strip()
         pattern = r"\s+" 
         filtered_line = re.sub(pattern, " ", line)
         if filtered_line !="" or filtered_line!=" " and(len(filtered_line)!=0):
            text.append(filtered_line)
    #Assigning rows to the dataframe
    df_result['text'] = text     
    
    return df_result

def readFromStr(s):
    df_result = pd.DataFrame(columns=('text', 't_text'))
    text = []
    subtext = ""
    #Separate into substrings when it finds a , or . or \n
    #then add it to the df
    for word in s:
        if word != "," and word != "." and word != "\n":
            subtext+=word
        else:
            text.append(subtext)
            subtext = "" 
    df_result['text'] = text
    return df_result

def tokenizeDF(dataframe):
    #drop NaN values
    dataframe = dataframe.dropna(subset=["text"])
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
    

    dataframe["t_text"] = dataframe["t_text"].apply(lambda words:[word for word in words if word not in com_stop_words])
    #Lemmatization
    lem = WordNetLemmatizer()
    dataframe["t_text"] = dataframe["t_text"].apply(lambda words:" ".join([lem.lemmatize(word,pos="v") for word in words]))

    #Finally delete rows with empty data
    dataframe.drop(dataframe.loc[dataframe["t_text"]==""].index,inplace=True)
    return dataframe
    

