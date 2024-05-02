import os

import google.generativeai as genai
from dotenv import load_dotenv


class ChatService:
    """
    A service class for managing chat interactions using the generative AI model from Google's genai library.

    Attributes:
        model (genai.GenerativeModel): An instance of the GenerativeModel ready to facilitate chat interactions.

    Methods:
        __init__(self, model_name: str)
            Initializes the ChatService with a specific GenerativeModel.

        start_chat(self) -> genai.Chat
            Initiates a new chat session and returns the Chat object.

        send_message(self, parameter: str, message: str, chat: genai.Chat) -> str
            Sends a message within a chat session and returns the response text.
    """

    def __init__(self, model_name: str = 'gemini-pro'):
        """
        Initializes the ChatService instance by configuring the genai library with an API key and loading
        a GenerativeModel based on the provided model name.
        """

        load_dotenv()
        self.GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        genai.configure(api_key=self.GOOGLE_API_KEY)
        self.model = genai.GenerativeModel(model_name)

    def start_chat(self):
        """
        Starts a new chat by initializing a chat session with the model.

        Returns:
            A genai.Chat instance representing the new chat session.
        """
        chat = self.model.start_chat(history=[])
        return chat

    def send_message(self, parameter: str, message: str, chat) -> str:
        """
        Sends a message to the chat session and returns the received response text.

        Args:
            parameter (str): Additional parameter to be added to the message.
            message (str): The message text to send.
            chat (genai.Chat): The chat instance to which the message should be sent.

        Returns:
            The text of the response from the chat session.
        """
        response = chat.send_message(f"{parameter} {message}")
        return response.text
