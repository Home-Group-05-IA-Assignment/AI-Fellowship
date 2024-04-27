from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
# Repository in HuggingFace trained.
ID_REPOSITORY = "Valwolfor/distilbert_emotions_fellowship"

# Use tokenizer of original train.
tokenizer = AutoTokenizer.from_pretrained(ID_REPOSITORY)

# Model trained.
model = AutoModelForSequenceClassification.from_pretrained(ID_REPOSITORY)

texts = [
    "I feel so alone even when I'm surrounded by people.",
    "The news of his passing brought tears to my eyes.",
    "Everything feels gray and bleak these days.",
    "Losing my job has left me feeling completely hopeless.",
    "It's hard to say goodbye to someone you love.",
    "Winning the competition filled me with an indescribable happiness.",
    "Seeing her smile is the best part of my day.",
    "I couldn't stop laughing at the joke he told.",
    "The sun is shining, and I feel fantastic!",
    "Hearing the good news made my heart leap with joy.",
    "Every moment with you is a treasure I hold dear.",
    "I feel a warmth in my heart whenever I think of you.",
    "Meeting you was the best thing that ever happened to me.",
    "Your love gives me the strength to face any challenge.",
    "Holding your hand feels like coming home.",
    "Their unfair decision made my blood boil with anger.",
    "I can't believe you would betray my trust like this!",
    "Every word he said was like adding fuel to the fire.",
    "I'm tired of being ignored and taken for granted.",
    "Canceling our plans at the last minute really infuriated me.",
    "Walking alone at night sends shivers down my spine.",
    "The thought of losing you terrifies me more than anything.",
    "Hearing those strange noises made my heart race.",
    "I'm petrified of speaking in front of large crowds.",
    "The uncertainty of the future fills me with dread.",
    "I was taken aback by the unexpected gift.",
    "Seeing them jump out of the dark gave me quite a shock.",
    "The plot twist at the end of the movie was completely unforeseen.",
    "Getting the promotion was a pleasant surprise I hadn't anticipated.",
    "Their sudden appearance at the party was a total surprise."
]

# Tokenizing inputs
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=128)

# Predictions
with torch.no_grad():
    outputs = model(**inputs)


logits = outputs.logits

# Get id of Class for every text
predicted_class_ids = logits.argmax(dim=1)

print("ID de clase predicha:", predicted_class_ids.tolist())
