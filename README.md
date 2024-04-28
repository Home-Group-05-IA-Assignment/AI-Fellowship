IA-Fellowship: Emotion and Stress Recognition Application
Description

IA-Fellowship is an innovative application designed to accurately recognize emotions and stress levels embedded within textual data. By harnessing cutting-edge natural language processing (NLP) and machine learning techniques, it facilitates the identification and analysis of various human emotional states conveyed through text.
Environment Setup

To ensure a consistent and isolated development environment for this project, ia-venv (a Python virtual environment) is utilized. Follow these steps to set up your environment:

# Install virtualenv if you haven't
pip install virtualenv

# Create a virtual environment
virtualenv ia-venv

# Activate the virtual environment
# For Windows
ia-venv\Scripts\activate
# For macOS and Linux:
source ia-venv/bin/activate

# Install the required dependencies
pip install -r requirements.txt

Data Preparation

The initial stage of the project involves rigorous data cleaning and preprocessing to prepare the text data for emotion analysis. Key steps include:

    Removing null values and duplicate entries.
    Trimming extra spaces and removing symbols and punctuation.
    Lowercasing all text to maintain uniformity.
    Expanding English abbreviations to ensure clarity and consistency.
    Categorizing emotions into six distinct classes: sadness (0), joy (1), love (2), anger (3), fear (4), and surprise (5).
    Calculating text length, removing emojis and 'stop words', and applying stemming to simplify the text.

Key Files and Directories

    abbreviations.json: This JSON file contains mappings from common English abbreviations to their expanded forms, used in text preprocessing to enhance clarity.
    hg-05-ia.ipynb: A Jupyter notebook that outlines the data cleaning process, available for review on Google Colab.
    text.csv: Raw textual data before cleaning.
    cleaned_data.csv: Processed data post-cleaning.

The API

The core of this application is exposed through a Flask-based API, which serves endpoints for emotion prediction. To utilize the API:

    Ensure the application is running locally at: http://127.0.0.1:5000.
    Make POST requests to /predict with a JSON payload containing the text to be analyzed, e.g.,

{
  "text": "Feeling great after a long walk."
}

    Analyzed emotions and their descriptions are returned in JSON format.

Project Structure Overview

    The EmotionPredictor class handles the prediction logic using a pre-trained BERT model, focusing on emotion classification.
    TextHandler is responsible for language detection, translation (placeholder), and providing human-readable descriptions of the predicted emotions.
    TextPreprocessor performs initial text cleaning, including expanding abbreviations, removing stopwords, and stemming.

Final Notes

This application, part of the broader IA-Fellowship initiative, aims to push the boundaries of emotion and stress analysis
using AI. Ongoing development and improvements are focused on enhancing accuracy, expanding language support, and refining
the model for better real-world applicability.