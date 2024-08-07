#!/usr/bin/env python
import os
import requests


def get_azure_env() -> tuple[str, str]:
    # https://stackoverflow.com/a/65694371
    api_key = os.environ.get("AZURE_OPENAI_APIKEY", None)
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", None)

    # try to read the env file manually
    if api_key is None or endpoint is None:
        print('The Azure API Key and Endpoint are not configured. Please make sure you have .env file with these variables ($AZURE_OPENAI_APIKEY , $AZURE_OPENAI_ENDPOINT)')
        try:
            with open(".env", 'r') as reader:
                data = reader.readlines()
                api_key = data[0].strip()[data[0].find("=")+1:]
                # the endpoint might contain an extra "=" in the api version query
                endpoint = data[1].strip()[data[1].find("=")+1:]
        except Exception:
            exit(1)

    return api_key, endpoint


class AzureChat:
    def __init__(self, api_key: str, api_endpoint: str):
        self.api_key = api_key
        self.endpoint = api_endpoint
        self.headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key,
        }

    def generate_prompt_response(self, user_input: str):
        prompt_message = self.generate_payload(user_input)
        try:
            response = requests.post(
                self.endpoint, headers=self.headers, json=prompt_message)
            response.raise_for_status()
        except requests.RequestException as e:
            raise SystemExit(f"Failed to make the request. Error: {e}")

        if response.status_code in [200, 201, 203]:
            resp_as_json = response.json()
            resp_message = resp_as_json["choices"][0]["message"]["content"]
            return resp_message
        return None

    def generate_payload(self, user_input: str = "Hey there! Where are you hosted and by which company?", temperature: float = 0.45, max_tokens: int = 1200) -> dict:
        """
        - uses the `user_input` string to initialize the payload used in the `POST` request to the Azure OpenAI deployment
        - the payload will contain a `messages` dictionary, in which `system`, `user`, and `assistant` contents will be configured
        - the actual prompt message that a user will require an answer to is provided by the `"role": "user"` content
        """
        prompt_obj = {
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "You are an AI assistant that helps people find information."
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"{user_input.strip()}"
                        }
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "Hello! How can I assist you today?"
                        }
                    ]
                }
            ],
            "temperature": temperature,
            "top_p": 0.95,
            "max_tokens": max_tokens
        }

        return prompt_obj


if __name__ == "__main__":
    API_KEY, API_ENDPOINT = get_azure_env()

    chat = AzureChat(API_KEY, API_ENDPOINT)

    MESSAGE = "give me a simple python3 script that will help me interact with you"

    print(chat.generate_prompt_response(MESSAGE))
