import os
from openai import AzureOpenAI

from dotenv import load_dotenv


def load_env():
    load_dotenv()
    api_key = os.environ.get("AZURE_OPENAI_APIKEY", None)
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", None)
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT_2", None)
    deployment = os.environ.get(
        "AZURE_OPENAI_DEPLOYMENT", "gpt-4o")  # default to gpt-4o
    return api_key, api_version, endpoint, deployment


class AzureInterface:
    def __init__(self, api_key: str, api_version: str, azure_endpoint: str, deployment: str):
        self.api_key = api_key
        self.api_version = api_version
        self.azure_endpoint = azure_endpoint
        self.deployment = deployment
        self.client = self.create_client()

    def create_client(self):
        client = AzureOpenAI(
            azure_endpoint=self.azure_endpoint,
            azure_deployment=self.deployment,
            api_version=self.api_version,
            api_key=self.api_key
        )
        return client

    @staticmethod
    def generate_prompt_message(role: str, content: str, system_input: str = "You are a helpful assistant.") -> list[dict]:
        """
        Generates a list of dictionaries to be used as a prompt for a model. Each dictionary represents a message from either the system or a specified role.

        Args:
            role (str): The role of the message sender (e.g., 'user', 'assistant').
            content (str): The message content for the specified role.
            system_input (str): The message from the system. Defaults to "You are a helpful assistant.".

        Returns:
            list[dict]: A list of dictionaries with the roles 'system' and the specified role and their corresponding messages.

        Example:
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hey there! How are you?"}
            ]
        """
        return [
            {"role": "system", "content": system_input.strip()},
            {"role": role, "content": content.strip()}
        ]

    def create_chat_completion(self, messages: list[dict], max_tokens: int = 4096, temperature: float = 0.45):
        """
        Generates a chat-based completion using the specified parameters.

        This method interacts with an AI language model to generate a response based on a sequence of input messages.
        The function utilizes the chat completion API of the client's deployed model, allowing fine-tuning of the response
        length and creativity through the `max_tokens` and `temperature` parameters.

        Args:
            messages (list[dict]): A list of message dictionaries representing the conversation history.
                Each dictionary should typically contain a 'role' key with a value of either 'user' or 'assistant',
                and a 'content' key containing the message text.
            max_tokens (int, optional): The maximum number of tokens the model should generate in the response. 
                Tokens can be as short as one character or as long as one word. Default is 1337.
            temperature (float, optional): A value that controls the randomness of the output. 
                Lower values (e.g., 0.2) make the output more deterministic and focused, while higher values (e.g., 0.8) 
                increase creativity and variety. Default is 0.45.

        Returns:
            str: The generated text content of the response, as produced by the model.

        Example:
            messages = [
                {"role": "user", "content": "What is the capital of France?"},
                {"role": "assistant", "content": "The capital of France is Paris."}
            ]
            response = create_chat_completion(messages, max_tokens=50, temperature=0.7)
            print(response)  # Output might be "Paris is the capital city of France, known for its culture, art, and history."

        Raises:
            Any exceptions raised by the underlying API client will propagate through this method.

        Notes:
            - The `max_tokens` parameter includes both input and output tokens, so the effective length of the generated
            response may be slightly shorter than this value.
            - Adjusting the `temperature` parameter allows you to balance between predictable and creative responses,
            depending on the use case.

        """
        response = self.client.chat.completions.create(
            messages=messages,
            model=self.deployment,
            max_tokens=max_tokens,
            temperature=temperature)
        return response.choices[0].message.content

    def create_message_list(self, roles: str | list[str], messages: list[str]):
        if isinstance(roles, str):
            _list = [{"role": roles, "content": message}
                     for message in messages[1:]]
            return [messages[0]].extend(_list)
        elif isinstance(roles, list):
            _list = [{"role": role, "content": message}
                     for role, message in zip(roles[1:], messages[1:])]
            return [messages[0]].extend(_list)


if __name__ == "__main__":
    api_key, api_version, endpoint, deployment = load_env()

    # create the azure AzureOpenAI interface in order to connect to the model deployment
    az = AzureInterface(api_key, api_version, endpoint, deployment)

    message = az.generate_prompt_message(
        "user", "give me a simple python3 script that will help me interact with you")
    prompt = az.create_chat_completion(message)
    print(prompt)
