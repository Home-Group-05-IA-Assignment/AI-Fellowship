from flask import Flask, request, jsonify
from ai_model import EmotionPredictor
from input_handler import TextHandler
from text_preprocessor import TextPreprocessor

# Initialize the Flask application
app = Flask(__name__)

# Initialize the emotion prediction model, input handler, and text preprocessor
emotion_predictor = EmotionPredictor(model_id="Valwolfor/distilbert_emotions_fellowship")
input_handler = TextHandler()
text_processor = TextPreprocessor()


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predicts the emotion from the input text.
    Note: Language detection and translation (e.g., Spanish to English) should be handled separately.

    Expected JSON format in request:
    {
        "text": "<input_text_here>"
    }

    Response JSON format:
    {
        "emotion": "<predicted_emotion>",
        "description": "<description_of_emotion>"
    }

    Returns:
        - 400 Bad Request if "text" is missing from the request.
        - JSON response with predicted emotion and description.
    """

    if not request.json or 'text' not in request.json:
        return jsonify({'error': 'Invalid request, "text" field is expected.'}), 400

    text = request.json['text']
    # Language detection and optional translation (implementation needed)

    processed_text = text_processor.preprocess_text(text)

    prediction = emotion_predictor.predict_emotion(processed_text)
    emotion, description = input_handler.get_emotion_description(prediction)

    return jsonify({'emotion': emotion, 'description': description})


if __name__ == '__main__':
    app.run(debug=True)