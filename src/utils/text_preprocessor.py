import re
import emoji
import pandas as pd
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords


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
    """

    def __init__(self):
        """
        Initializes the text preprocessor by loading English stopwords from NLTK and
        creating an instance of the NLTK PorterStemmer to be used for word stemming
        in the preprocessing pipeline.
        """
        self.stop_words = set(stopwords.words('english'))
        self.ps = PorterStemmer()

    def preprocess_text(self, text):
        """
        Conducts a series of preprocessing steps on the input text including normalization,
        cleaning, stopping, stemming, and emoji to text conversion. Also expands chat
        abbreviations found in the input text.

        Args:
            text (str): The English text to preprocess.

        Returns:
            str: The preprocessed text.
        """
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

    def replace_chat_words(self, text):
        """
        Expands abbreviations commonly found in chat messages with their full forms as per
        a predefined abbreviations.json file. This process is case-insensitive.

        Args:
            text (str): The text containing possible chat abbreviations.

        Returns:
            str: Text with chat abbreviations expanded.
        """
        replacements_df = pd.read_json('../../ai-models/clean-data/abbreviations.json', orient='index')
        replacements_df.index = replacements_df.index.str.upper()
        replacements = replacements_df[0].to_dict()

        text_upper = text.upper()
        for chat_word, full_form in replacements.items():
            text_upper = text_upper.replace(chat_word, full_form)

        return text_upper.lower()