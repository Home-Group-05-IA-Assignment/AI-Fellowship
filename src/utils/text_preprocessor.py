import re
import emoji
import pandas as pd
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from string import punctuation
from nltk.corpus import wordnet
from spacy.lang.en.stop_words import STOP_WORDS
from wordcloud import STOPWORDS


class TextPreprocessor:
    """
    A class dedicated to preprocessing textual data. The preprocessing steps include
    normalization (lowercasing), cleaning (removing symbols and extra spaces), stopping
    (removal of stopwords), stemming, emoji conversion to text, and expansions of chat
    abbreviations.

    Attributes:
        stop_words (set): A set of stopwords for English, to be removed from text.
        ps (PorterStemmer): An instance of the NLTK PorterStemmer for word stemming.

    Methods:
        __init__()
            Constructor of the TextPreprocessor class. Initializes with English
            stopwords and a PorterStemmer instance.

        preprocess_text(text: str) -> str
            Performs all preprocessing steps on the provided text string and returns
            the cleaned text.

        replace_chat_words(text: str) -> str
            Replaces abbreviations often found in chat messages with their full forms
            using a predefined JSON file as a source for the mappings.

        readFromStr(str) -> pandas.DataFrame
            Reads the lowered string and converts it into a Dataframe. Necessary for logModel
        tokenizeDF(pandas.DataFrame) -> pandas.DataFrame
            Cleans and tokenizes the Datafrme. Necessary for logModel
    """

    def __init__(self):
        """
        Initializes the text preprocessor by loading English stopwords from NLTK and
        creating an instance of the NLTK PorterStemmer to be used for word stemming
        in the preprocessing pipeline.
        """
        self.stop_words = set(stopwords.words('english'))
        self.ps = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()

    def convertDF_getProbs(self, s):
        df_result = pd.DataFrame(columns=('text', 't_text'))
        text = []
        subtext = ""
        # Separate into substrings when it finds a , or . or \n
        # then add it to the df
        for word in s:
            if word != "," and word != "." and word != "\n":
                subtext += word
            else:
                text.append(subtext)
                subtext = ""
        df_result['text'] = text
        df_result = self.tokenizeDF(df_result)

        # run logistic model and return probabilities
        return df_result

    """
    Enables reading from strings and returns a dataframe with the data (used for logistic model)
    """

    def readFromStr(self, s):
        df_result = pd.DataFrame(columns=('text', 't_text'))
        text = []
        subtext = ""
        # Separate into substrings when it finds a , or . or \n
        # then add it to the df
        for word in s:
            if word != "," and word != "." and word != "\n":
                subtext += word
            else:
                text.append(subtext)
                subtext = ""
        df_result['text'] = text
        return df_result

    """
    Tokenizes and cleans the Dataframe from readFromStr (used for logistic model)
    """

    def tokenizeDF(self, dataframe):

        # drop NaN values
        dataframe = dataframe.dropna(subset=["text"])
        # lowercase, digits and extra-spaces
        dataframe["t_text"] = dataframe["text"].astype(str).str.lower()
        dataframe["t_text"] = dataframe["t_text"].apply(lambda x: re.sub(r"\d+", "", x))
        dataframe["t_text"] = dataframe["t_text"].apply(lambda x: re.sub(r"\s+", " ", x))

        # links and special characters
        dataframe["t_text"] = dataframe["t_text"].apply(lambda x: re.sub(r"http\S+", "", x))
        dataframe["t_text"] = dataframe["t_text"].apply(lambda x: re.sub(r"[^\w\s]", "", x))

        # Tokenization and removing puntuation
        dataframe["t_text"] = dataframe["t_text"].apply(lambda x: word_tokenize(x))
        dataframe["t_text"] = dataframe["t_text"].apply(
            lambda items: [item for item in items if item not in punctuation])

        # Stop words
        stop_words = stopwords.words("english")
        com_stop_words = stop_words + list(STOP_WORDS) + list(STOPWORDS)

        dataframe["t_text"] = dataframe["t_text"].apply(
            lambda words: [word for word in words if word not in com_stop_words])
        # Lemmatization
        lem = WordNetLemmatizer()
        dataframe["t_text"] = dataframe["t_text"].apply(
            lambda words: " ".join([lem.lemmatize(word, pos="v") for word in words]))

        # Finally delete rows with empty data
        dataframe.drop(dataframe.loc[dataframe["t_text"] == ""].index, inplace=True)
        return dataframe

    def preprocess_text(self, text, is_logistic=False):
        """
        Conducts a series of preprocessing steps on the input text including normalization,
        cleaning, stopping, stemming, and emoji to text conversion. Also expands chat
        abbreviations found in the input text.

        Args:
            text (str): The English text to preprocess.

        Returns:
            str: The preprocessed text.
            :param text:
            :param is_logistic:
        """
        if is_logistic:

            # tokens = word_tokenize(text)
            # filtered_tokens = [word for word in tokens if word not in self.stop_words]
            # text = [self.lemmatizer.lemmatize(word) for word in filtered_tokens]
            text = self.readFromStr(text)
            """Text now is a Dataframe"""
            text = self.tokenizeDF(text)
        else:
            # Initial preprocessing step to expand chat abbreviations
            text = self.replace_chat_words(text)

            # Converting to lowercase, removing extra spaces, symbols, and converting emojis
            text = text.lower()
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'[^\w\s]', '', text)
            text = emoji.demojize(text)

            # Removing stop words and stemming the remainders
            text = " ".join([self.ps.stem(word) for word in text.split() if word not in self.stop_words])
        return text

    @staticmethod
    def replace_chat_words(text):
        """
        Expands abbreviations commonly found in chat messages with their full forms as per
        a predefined abbreviations.json file. This process is case-insensitive.

        Args:
            text (str): The text containing possible chat abbreviations.

        Returns:../../ai-models/clean-data/abbreviations.json
            str: Text with chat abbreviations expanded.
        """

        replacements = {
            "AFAIK": "As Far As I Know",
            "AFK": "Away From Keyboard",
            "ASAP": "As Soon As Possible",
            "ATK": "At The Keyboard",
            "ATM": "At The Moment",
            "A3": "Anytime, Anywhere, Anyplace",
            "BAK": "Back At Keyboard",
            "BBL": "Be Back Later",
            "BBS": "Be Back Soon",
            "BFN": "Bye For Now",
            "B4N": "Bye For Now",
            "BRB": "Be Right Back",
            "BRT": "Be Right There",
            "BTW": "By The Way",
            "B4": "Before",
            "B4N": "Bye For Now",
            "CU": "See You",
            "CUL8R": "See You Later",
            "CYA": "See You",
            "FAQ": "Frequently Asked Questions",
            "FC": "Fingers Crossed",
            "FWIW": "For What It's Worth",
            "FYI": "For Your Information",
            "GAL": "Get A Life",
            "GG": "Good Game",
            "GN": "Good Night",
            "GMTA": "Great Minds Think Alike",
            "GR8": "Great!",
            "G9": "Genius",
            "IC": "I See",
            "ICQ": "I Seek you (also a chat program)",
            "ILU": "ILU: I Love You",
            "IMHO": "In My Honest/Humble Opinion",
            "IMO": "In My Opinion",
            "IOW": "In Other Words",
            "IRL": "In Real Life",
            "KISS": "Keep It Simple, Stupid",
            "LDR": "Long Distance Relationship",
            "LMAO": "Laugh My A.. Off",
            "LOL": "Laughing Out Loud",
            "LTNS": "Long Time No See",
            "L8R": "Later",
            "MTE": "My Thoughts Exactly",
            "M8": "Mate",
            "NRN": "No Reply Necessary",
            "OIC": "Oh I See",
            "PITA": "Pain In The A..",
            "PRT": "Party",
            "PRW": "Parents Are Watching",
            "QPSA?": "Que Pasa?",
            "ROFL": "Rolling On The Floor Laughing",
            "ROFLOL": "Rolling On The Floor Laughing Out Loud",
            "ROTFLMAO": "Rolling On The Floor Laughing My A.. Off",
            "SK8": "Skate",
            "STATS": "Your sex and age",
            "ASL": "Age, Sex, Location",
            "THX": "Thank You",
            "TTFN": "Ta-Ta For Now!",
            "TTYL": "Talk To You Later",
            "U": "You",
            "U2": "You Too",
            "U4E": "Yours For Ever",
            "WB": "Welcome Back",
            "WTF": "What The F...",
            "WTG": "Way To Go!",
            "WUF": "Where Are You From?",
            "W8": "Wait...",
            "7K": "Sick:-D Laugher",
            "TFW": "That feeling when",
            "MFW": "My face when",
            "MRW": "My reaction when",
            "IFYP": "I feel your pain",
            "TNTL": "Trying not to laugh",
            "JK": "Just kidding",
            "IDC": "I don't care",
            "ILY": "I love you",
            "IMU": "I miss you",
            "ADIH": "Another day in hell",
            "ZZZ": "Sleeping, bored, tired",
            "WYWH": "Wish you were here",
            "TIME": "Tears in my eyes",
            "BAE": "Before anyone else",
            "FIMH": "Forever in my heart",
            "BSAAW": "Big smile and a wink",
            "BWL": "Bursting with laughter",
            "BFF": "Best friends forever",
            "CSL": "Can't stop laughing"
        }

        text_upper = text.upper()
        for chat_word, full_form in replacements.items():
            # Replace the abbreviation in the text with its full form
            text_upper = text_upper.replace(chat_word, full_form)

        return text_upper.lower()
