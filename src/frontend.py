import streamlit as st

from ai_controller import EmotionController

controller = EmotionController()
st.session_state['send_button_disabled'] = True

st.title("¿Sabes como controlar tus emociones? El primer paso es identificarlas")

firstIn = False
send_button_disabled = True

model_options = {
    "Logistic Regression": 0,
    "Bidirectional Encoder Representations from Transformers-BERT": 1
}

#comenzar
def main():
    global firstIn
    global send_button_disabled
    emotion = ""
    parameter = "Toma el papel de un consejero para dar recomendaciones de lecturas(no compartas enlaces) y tecnicas para controlar las emociones, especialmente invitando a la persona a hacer actividades offline, ademas de dar consejos y escuchar al usuario. Segun el siguiente texto que recomendaciones harias: "

    tab1, tab2, tab3 = st.tabs(["Explora tus emociones", "¿Quieres ayuda?","Explora tu texto"])
    prediction_label, description_label, percentage = "", "", ""
    with tab1:
        st.header("Según tu texto te diremos qué emoción estás sintiendo")
        model_choice = st.selectbox("Selecciona el modelo con el que quieres trabajar: ", list(model_options.keys()))
        text = st.text_area("Escribe aquí tu texto")

        #
        if st.button("Identificar emoción"):
            if text.strip():

                try:
                    prediction_label, description_label, percentage = controller.run_analysis(model_options[model_choice], text)
                    st.write(f"La emoción que estás sintiendo es: {prediction_label}, la probabilidad: {percentage:.2%}, {description_label}. Si quieres profundizar un poco más ve a la segunda pestaña.")
                except:
                    st.write(f"Por favor escribe algo más largo. O intenta con otro modelo")
            else:
                st.write("Por favor, introduce algo de texto para analizar.")

    with tab2:
        st.warning("Si sales de la pestaña se borrara la conversación")

        st.write(f"¡Hola! Soy Gemini, tu asistente personal para controlar tus emociones. ¿En qué puedo ayudarte hoy? Tu emoción fue {prediction_label} y su probabilidad fue de {percentage:.2}")

        message = st.text_area("Escribe aquí tu mensaje para Gemini")

        if st.button("Enviar"):
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
    # Asegurándose de que hay candidatos disponibles
    if not generate_content_response.result.candidates:
        return "No se encontraron candidatos."

    # Tomando el primer candidato. Ajusta según sea necesario si esperas múltiples candidatos
    first_candidate = generate_content_response.result.candidates[0]

    # Verificando que haya partes con texto disponible
    if not first_candidate['content']['parts']:
        return "El candidato no contiene partes con texto."

    # Concatenando el texto de todas las partes disponibles
    response_text = ""
    for part in first_candidate['content']['parts']:
        response_text += part['text'] + "\n"  # Asumiendo que cada parte tiene un texto a concatenar

    return response_text

if __name__ == "__main__":
    main()
