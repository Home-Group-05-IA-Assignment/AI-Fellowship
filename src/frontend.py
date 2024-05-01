import streamlit as st
import backend as bk

st.session_state['send_button_disabled'] = True

st.title("¿Sabes como controlar tus emociones? El primer paso es identificarlas")

firstIn = False
send_button_disabled = True

def main():
    global firstIn
    global send_button_disabled
    emotion = ""
    parameters = "Toma el papel de un psicologo para dar recomendaciones de articulos y tecnicas para controlar las emociones, especialmente invitando a la persona a hacer actividades offline, ademas de dar consejos y escuchar al usuario. Segun el siguiente texto que recomendaciones harias: "

    tab1, tab2 = st.tabs(["Explora tus emociones", "¿Quieres ayuda?"])

    with tab1:
        st.header("Según tu texto te diremos qué emoción estás sintiendo")
        text = st.text_area("Escribe aquí tu texto")

        if st.button("Identificar emoción"):
            emotion = text
            st.write(f"La emoción que estás sintiendo es: {emotion}")
            firstIn = True

    with tab2:
        chat = bk.start_chat()
        st.warning("Si sales de la pestaña se borrara la conversación")

        st.write("¡Hola! Soy Gemini, tu asistente personal para controlar tus emociones. ¿En qué puedo ayudarte hoy?")

        message = st.text_area("Escribe aquí tu mensaje para Gemini")

        if st.button("Enviar"):
            response = bk.send_message(message, chat)
            st.write(response)

        if firstIn:
            firstIn = False
            firstChat = parameters + emotion
            response = bk.send_message(firstChat, chat)
            st.write(response)



if __name__ == "__main__":
    main()