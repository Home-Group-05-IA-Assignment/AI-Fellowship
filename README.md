# AI-Fellowship: Emotion and Stress Recognition Application

## Description

IA-Fellowship is an application designed to accurately recognize emotions and stress levels embedded within textual data. By harnessing cutting-edge natural language processing (NLP) and machine learning techniques, it facilitates the identification and analysis of various human emotional states conveyed through text.

## Environment Setup

The all models was trained with Notebook in Google Colab with T4. The Notebook there are in the repository.

The first model **BERT** (Bidirectional Encoder Representations from Transformers) is a transformer-based machine learning technique for natural language processing (NLP) pre-training. Developed by Google, BERT captures the context of words in a sentence in all directions (bidirectionally),This model is pre-trained on a large corpus of text and then fine-tuned for specific tasks like question answering, sentiment analysis, and language understanding. Then was also uploaded to this [HuggingFace](https://huggingface.co/Valwolfor/distilbert_emotions_fellowship/). This model return the class of the emotion and the probability. We used _Distil Bert_ with a lost of .02, trained in three epoch.

The second model **Logistic Regression** tailored for recognizing various human emotions from textual data. The process encompasses data preparation, model training, and serialization to enable easy future use without the need to retrain.

The **Bidirectional Gated Recurrent Unit (Bi-GRU)** model is an advanced tool in Natural Language Processing (NLP), enhanced by its ability to understand and process sequential data such as text. By combining the strengths of Closed Recurrent Units (GRU) with a bidirectional approach, Bi-GRU models can capture contextual information of both past and future states within a sequence. This allows them to achieve exceptional performance in text classification tasks, with a high degree of accuracy, especially when overfitting is avoided by regularization techniques such as Dropout. Proper data preprocessing is crucial for optimal model performance, which includes tokenizing and converting text into sequences of uniform length. During model evaluation, the accuracy metric is used to measure its ability to correctly classify test examples. A Bi-GRU model with an accuracy of 94% indicates a strong ability to generalize to unseen data, making it a valuable tool for a wide range of PLN applications.

**Key Features**

Bidirectionality: Unlike traditional GRU models that process data in a single direction, Bi-GRUs analyze data in both forward and backward directions. This allows the model to have a more comprehensive understanding of the input sequence.

GRU Cells: Incorporates GRU cells to efficiently manage information flow, aiding the model in solving vanishing gradient problems and making it capable of handling long-range dependencies within text.

To ensure a consistent and isolated development environment for this project (a Python virtual environment) is utilized. Follow these steps to set up your environment:

```
#Install virtualenv if you haven't
pip install virtualenv

#Create a virtual environment
virtualenv ia-venv

# Activate the virtual environment
# For Windows
ia-venv\Scripts\activate
# For macOS and Linux:
source ia-venv/bin/activate

# Install the required dependencies
pip install -r requirements.txt
```
Dockerfile

```
#Build image from docker file in the dir where is the Dockerfile.
docker build -t <id-image> .

# Run the container
 docker run -p 8000:8000 -dit <id-container>

#You can execute commands into of container 
docker exec -it <id-container> bash


#Into the container
streamlit run ./src/frontend.py --server.port 8000

```

## Data Preparation

[Dataset](https://www.kaggle.com/datasets/nelgiriyewithana/emotions)

The initial stage of the project involves rigorous data cleaning and preprocessing to prepare the text data for emotion analysis. Key steps include:

- Removing null values and duplicate entries.
- Trimming extra spaces and removing symbols and punctuation.
- Lowercasing all text to maintain uniformity.
- Expanding English abbreviations to ensure clarity and consistency.
- Categorizing emotions into six distinct classes: sadness (0), joy (1), love (2), anger (3), fear (4), and surprise (5).
- Calculating text length, removing emojis and 'stop words', and applying stemming to simplify the text.

## Key Files and Directories

- **abbreviations.json**: This JSON file contains mappings from common English abbreviations to their expanded forms, used in text preprocessing to enhance clarity.
- **\*.ipynb**: A Jupyter notebook that outlines the data cleaning process, available for review on Google Colab.
- **text.csv**: Raw dataset before cleaning.
- **Models files**: the model trained and ready to be used by the final user.

## Project Structure Overview

Complementing the EmotionPredictor is the TextHandler, responsible for preliminary language processing tasks such as detecting the language of the input text and facilitating its translation (currently a placeholder for future development). It further enriches the output by attaching human-readable descriptions to the predicted emotions, enhancing interpretability for end users.

Meanwhile, the TextPreprocessor plays a critical role in refining the input data. By expanding abbreviations, excising stopwords, and applying stemming techniques, it ensures the text is in an optimal form for analysis, improving the accuracy of emotion detection.

From an architectural standpoint, our project adopts a modular approach, utilizing an IEmotionPredictor interface with an abstract method to seamlessly integrate the emotion prediction models into our service. This design pattern not only enhances the maintainability of the codebase but also simplifies the integration of additional models in the future.

On the frontend, we use Streamlit to craft an interactive and user-friendly interface, organized into three main tabs:

**Emotion Analysis Tab**: Allows the user to input text, which is then processed to display a DataFrame showcasing the percentage representation of detected emotions, leveraging the GRU model for this purpose and one class with its probability for the others models.

**Gemmini Conversation Tab**: Takes the output from the first tab to facilitate a conversation with the Gemini App. This innovative feature aims to simulate a responsive interaction based on the emotional cues gleaned from the user's input.

**Text Analysis Tab**: Focuses on a deeper textual analysis using the Logistic Regression model. The output, alongside a Word Cloud visualization, offers the user a comprehensive view of the thematic and emotional elements present in their input.

This streamlined process, from robust backend logic to a dynamic and engaging frontend, exemplifies our commitment to delivering a holistic tool for emotion and sentiment analysis. By bridging cutting-edge AI models with intuitive interface design, we aim to provide users with valuable insights into the emotional undertones of textual data.

### Final Notes

This application, part of the broader IA-Fellowship initiative, aims to push the boundaries of emotion and stress analysis using AI. Ongoing development and improvements are focused on enhancing accuracy, expanding language support, and refining the models for better real-world applicability. Having both models, BERT and logistic regression, offers complementary benefits: while BERT excels at capturing complex linguistic patterns, logistic regression provides interpretable probability scores for each emotion, enhancing the overall robustness and interpretability of the emotion classification system.
