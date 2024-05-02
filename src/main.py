from src.models.logistic_model import EmotionLogisticPredictor

emotion_predictor = EmotionLogisticPredictor()
texts = "Hi i am so happy about being, the first text to be processed in this exciting model."
predictions = emotion_predictor.predict_emotions(texts)
print(predictions)