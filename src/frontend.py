import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
from ai_controller import EmotionController
from utils import text_preprocessor
from models import logistic_model

controller = EmotionController()
st.session_state['send_button_disabled'] = True

st.title("Do you know how to control your emotions? The first step is to identify them")

firstIn = False
send_button_disabled = True

model_options = {
    "Logistic Regression": 0,
    "Bidirectional Encoder Representations from Transformers-BERT": 1,
    "GRU": 2,
}

model_information = {
    "Logistic Regression": "Simplicity, efficiency, handling of categorical variables, classification probabilities, robustness. Disadvantage: Limited interpretability in complex cases.",
    "Bidirectional Encoder Representations from Transformers-BERT": "Powerful Contextual Understanding: BERT excels at capturing complex relationships between words in a sentence, leading to highly accurate classifications based on context. However has a high computational complexity.",
    "GRU": "GRU models offer more interpretability, allowing you to better understand what features contribute to classification decisions. However has a limited contextual awareness"
}


def main():
    global firstIn
    global send_button_disabled
    emotion = ""

    parameter = "Take on the role of a counselor to provide reading recommendations (do not share links) and techniques for controlling emotions, especially inviting the person to engage in offline activities, as well as giving advice and listening to the user. Based on the following text, what recommendations would you make: "

    tab1, tab2, tab3 = st.tabs(["Explore Your Emotions", "Need Help?", "Explore your text"])

    prediction_label, description_label, percentage = "", "", ""
    with tab1:
        st.header("We'll tell you what emotion you're feeling based on your text")
        model_choice = st.selectbox("Select the model you want to work with: ", list(model_options.keys()),
                                    help="Hover for more information")
        if model_choice:
            # Show the model description
            st.warning(f" 🤓{model_information[model_choice]}")
        text = st.text_area("Enter your text here")
        # Add the extra information for the user

        if st.button("Identify emotion"):
            if text.strip():
                print(f'model choice is {model_choice}')
                try:
                    if model_choice == "GRU":

                        res = controller.run_analysis(model_options[model_choice], text)

                        prediction_label = max(res, key=res.get)
                        percentage = res[prediction_label] / 100

                        df_emotions = pd.DataFrame(list(res.items()), columns=['Emotion', 'Probability (%)'])
                        st.table(df_emotions)

                    else:
                        prediction_label, description_label, percentage = controller.run_analysis(
                            model_options[model_choice], text)

                    st.write(
                        f"The emotion you're feeling is: {prediction_label}, the probability: {percentage:.3%}. If you want to delve a little deeper, go to the second tab.")
                    firstIn = True

                except Exception as e:
                    print(e)
                    st.write(
                        f"We had problems reading your text. Please write something longer or try with another model.")
            else:
                st.write("Please enter some text to analyze.")

    with tab2:
        st.warning("If you leave the tab, the conversation will be cleared")

        message = st.text_area("Enter your message for Gemini here")

        st.write(
            f"Hello! I'm Gemini, your personal assistant for controlling your emotions. How can I help you today? Your emotion was {prediction_label} and its probability was {percentage:.2}")

        if st.button("Send"):
            parameter += f"the evaluation of the emotion was {prediction_label} and the probability {percentage}. The user write this {message}"
            response = controller.gemini_controller(parameter, message)
            st.write(response)

        if firstIn:
            controller.restartChat()
            firstIn = False
            response = controller.gemini_controller(parameter, emotion)
            st.write(response)
    with tab3:
        st.warning(
            "🤓 We use Logistic Regression to analyze your text. However its response may not be 100% accurate or perfect. ")
        if text == "":
            st.write(
                'We need some text to analyze, please write something in "explore your emotions" and come back here.')
        else:
            try:
                processor = text_preprocessor.TextPreprocessor()
                df = processor.convertDF_getProbs(s=text)
                # add the emotions to the dataframe
                model = logistic_model.EmotionLogisticPredictor()
                probs = model.getEmotionsArrays(dataframe=df, modelName=model.emotions_model)
                df['emotion'] = probs
                df = df.loc[:, ["text", "emotion"]]
                st.write(
                    'This is the information you provided to us. Please expand the container in the rigth down corner')
                st.dataframe(df, use_container_width=True, height=5)
            except Exception as e:
                print(e)
                st.write("We had problems reading your text. Please write something longer")
            st.write('Wordcloud derived from your text')
            # Create and generate a word cloud image:
            wordcloud = WordCloud(background_color='white', height=600, width=800).generate(text)
            # supress warning
            fig, ax = plt.subplots()
            ax.imshow(wordcloud, interpolation='nearest')
            ax.axis("off")
            # Hide any extra axes or labels from matplotlib
            plt.tight_layout()
            # Display the word cloud image in Streamlit
            st.pyplot(fig)


def get_response_text(generate_content_response):
    # Making sure candidates are available
    if not generate_content_response.result.candidates:
        return "No candidates found."

    # Taking the first candidate. Adjust as needed if expecting multiple candidates
    first_candidate = generate_content_response.result.candidates[0]

    # Checking for parts with text available
    if not first_candidate['content']['parts']:
        return "Candidate does not contain parts with text."

    # Concatenating the text from all available parts
    response_text = ""
    for part in first_candidate['content']['parts']:
        response_text += part['text'] + "\n"  # Assuming each part has text to concatenate

    return response_text


if __name__ == "__main__":
    main()
