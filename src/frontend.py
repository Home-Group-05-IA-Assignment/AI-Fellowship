import streamlit as st

from ai_controller import EmotionController

controller = EmotionController()
st.session_state['send_button_disabled'] = True

st.title("Do you know how to control your emotions? The first step is to identify them")

firstIn = False
send_button_disabled = True

model_options = {
    "Logistic Regression": 0,
    "Bidirectional Encoder Representations from Transformers-BERT": 1
}

def main():
    global firstIn
    global send_button_disabled
    emotion = ""

    parameter = "Take on the role of a counselor to provide reading recommendations (do not share links) and techniques for controlling emotions, especially inviting the person to engage in offline activities, as well as giving advice and listening to the user. Based on the following text, what recommendations would you make: "

    tab1, tab2,tab3 = st.tabs(["Explore Your Emotions", "Need Help?","Explore your text"])

    prediction_label, description_label, percentage = "", "", ""
    with tab1:
        st.header("We'll tell you what emotion you're feeling based on your text")
        model_choice = st.selectbox("Select the model you want to work with: ", list(model_options.keys()))
        text = st.text_area("Enter your text here")

        #
        if st.button("Identify emotion"):
            if text.strip():

                try:
                    prediction_label, description_label, percentage = controller.run_analysis(model_options[model_choice], text)
                    f"The emotion you're feeling is: {prediction_label}, the probability: {percentage:.2%}, {description_label}. If you want to delve a little deeper, go to the second tab.")
                except:
                    st.write(f"We had problems reading your text. Please write something longer or try with another model.")
            else:
                st.write("Please enter some text to analyze.")

    with tab2:
        st.warning("If you leave the tab, the conversation will be cleared")

        st.write(f"Hello! I'm Gemini, your personal assistant for controlling your emotions. How can I help you today? Your emotion was {prediction_label} and its probability was {percentage:.2}")

        message = st.text_area("Enter your message for Gemini here")

        if st.button("Send"):
            parameter += f"the evaluation of the emotion was {prediction_label} and the probability {percentage}. The user write this {message}"
            response = controller.gemini_controller(parameter, message)
            st.write(response)

        if firstIn:
            firstIn = False
            response = controller.gemini_controller(parameter, emotion)
            st.write(response)
    with tab3:
        st.warning("🤓 We use Gemini API to analyze your text. However its response may not be 100% accurate or perfect. Please be careful about the recommendations of the generated text")
        if text == "":
            st.write('We need some text to analyze, please write something in "explore your emotions" and come back here.')
        else:
            st.write('it is nice to meet you')
    


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
