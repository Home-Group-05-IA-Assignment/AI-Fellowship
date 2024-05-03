from ai_controller import EmotionController

# Assuming the implementation of EmotionController and gemini_controller remains unchanged
controller = EmotionController()

model_options = {
    "Logistic Regression": 0,
    "Gated Recurrent Unit-GRU": 1,
    "Bidirectional Encoder Representations from Transformers-BERT": 2
}

def main():
    print("¿Sabes cómo controlar tus emociones? El primer paso es identificarlas.")
    emotion = ""
    parameter = "Toma el papel de un psicólogo para dar recomendaciones de artículos y técnicas para controlar las emociones, especialmente invitando a la persona a hacer actividades offline, además de dar consejos y escuchar al usuario. Según el siguiente texto que recomendaciones harías: "

    # Simulating Streamlit's tab functionality
    print("\n[1] Explora tus emociones")
    print("[2] ¿Quieres ayuda?")
    choice = input("Seleccione una opción (1 o 2): ")

    if choice == "1":
        print("\nSegún tu texto te diremos qué emoción estás sintiendo.")
        model_choice = input("Selecciona el modelo con el que quieres trabajar (Logistic Regression, Gated Recurrent Unit-GRU, Bidirectional Encoder Representations from Transformers-BERT): ")

        text = input("Escribe aquí tu texto: ")

        prediction_label, description_label, percentage = controller.run_analysis(model_options[model_choice], text)
        emotion = prediction_label

        print(f"La emoción que estás sintiendo es: {prediction_label}, con una probabilidad del {percentage*100}%, {description_label}")

    elif choice == "2":
        message = input("Escribe aquí tu mensaje para Gemini: ")
        response = controller.gemini_controller(parameter, message)
        print(response)

        if emotion:  # Assuming there was a previous emotion identified
            response = controller.gemini_controller(parameter, emotion)
            print(response)

if __name__ == "__main__":
    main()