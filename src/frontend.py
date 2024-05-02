import streamlit as st

from src.ai_controller import EmotionController

controller = EmotionController()
st.session_state['send_button_disabled'] = True

st.title("¿Sabes como controlar tus emociones? El primer paso es identificarlas")

firstIn = False
send_button_disabled = True

model_options = {
    "Logistic Regression": 0,
    "Gated Recurrent Unit-GRU": 1,
    "Bidirectional Encoder Representations from Transformers-BERT": 2
}


def main():
    global firstIn
    global send_button_disabled
    emotion = ""
    parameter = "Toma el papel de un psicologo para dar recomendaciones de articulos y tecnicas para controlar las emociones, especialmente invitando a la persona a hacer actividades offline, ademas de dar consejos y escuchar al usuario. Segun el siguiente texto que recomendaciones harias: "

    tab1, tab2 = st.tabs(["Explora tus emociones", "¿Quieres ayuda?"])

    with tab1:
        st.header("Según tu texto te diremos qué emoción estás sintiendo")

        # Dropbox button
        model_choice = st.selectbox("Seleciona el modelo con el que quieres trabajar: ", list(model_options.keys()))

        text = st.text_area("Escribe aquí tu texto")

        prediction_label, description_label, percentage = controller.run_analysis(model_choice, text)
        emotion = prediction_label

        if st.button("Identificar emoción"):

            st.write(f"La emoción que estás sintiendo es: {prediction_label}, la probabilidad: {percentage*100}%, {description_label}")
            firstIn = True

    with tab2:
        st.warning("Si sales de la pestaña se borrara la conversación")

        st.write("¡Hola! Soy Gemini, tu asistente personal para controlar tus emociones. ¿En qué puedo ayudarte hoy?")

        message = st.text_area("Escribe aquí tu mensaje para Gemini")

        if st.button("Enviar"):
            response = controller.gemini_controller(parameter, message)
            st.write(response)

        if firstIn:
            firstIn = False
            response = controller.gemini_controller(parameter, emotion)
            st.write(response)


if __name__ == "__main__":
    main()
