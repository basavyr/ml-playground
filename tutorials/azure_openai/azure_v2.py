import os
from openai import AzureOpenAI

from dotenv import load_dotenv


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
    def generate_initial_message(user_input: str = "Hey there! How are you?") -> list[dict]:
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": f'{user_input.strip()}'
            }
        ]
        return messages

    def create_chat_completion(self, user_input: str, max_tokens: int = 1337, temperature: float = 0.45):
        messages = AzureInterface.generate_initial_message(user_input)
        response = self.client.chat.completions.create(
            messages=messages,
            model=self.deployment,
            max_tokens=max_tokens,
            temperature=temperature)
        print(response.choices[0].message.content)


if __name__ == "__main__":
    load_dotenv()

    api_key = os.environ.get("AZURE_OPENAI_APIKEY", None)
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", None)
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT_2", None)
    deployment = os.environ.get(
        "AZURE_OPENAI_DEPLOYMENT", "gpt-4o")  # default to gpt-4o

    # create the azure AzureOpenAI interface in order to connect to the model deployment
    az = AzureInterface(api_key, api_version, endpoint, deployment)

    az.create_chat_completion(
        "give me a simple python3 script that will help me interact with you")
