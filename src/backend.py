import google.generativeai as genai
import os
from dotenv import load_dotenv
import textwrap
from IPython.display import Markdown

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=GOOGLE_API_KEY)

model = genai.GenerativeModel('gemini-pro')


def to_markdown(text):
    text = text.replace('•', '  *')
    return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))


def start_chat():
    chat = model.start_chat(history=[])
    return chat


def send_message(message, chat):
    response = chat.send_message(message)
    return response.text
