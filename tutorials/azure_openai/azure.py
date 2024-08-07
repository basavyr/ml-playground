#!/usr/bin/env python
import os
import requests


def get_azure_env() -> tuple[str, str]:
    # https://stackoverflow.com/a/65694371
    api_key = os.environ.get("AZURE_OPENAI_APIKEY", None)
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", None)

    # try to read the env file manually
    if api_key is None or endpoint is None:
        print('The Azure API Key and Endpoint are not configured. Please make sure you have .env file with these variables')
        try:
            with open(".env", 'r') as reader:
                data = reader.readlines()
                api_key = data[0].strip()[data[0].find("=")+1:]
                # the endpoint might contain an extra "=" in the api version query
                endpoint = data[1].strip()[data[1].find("=")+1:]
        except Exception:
            exit(1)

    return api_key, endpoint


def prompt(prompt_message: str = "Hey there! Where are you hosted and by which company?", temperature: float = 0.45, max_tokens: int = 1200) -> dict:
    """
    - uses the `prompt_message` string to initialize the payload used in the `POST` request to the Azure OpenAI deployment
    - the payload will contain a `message` object, in which `system`, `user`, and `assistant` contents will be configured
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
                        "text": f"{prompt_message.strip()}"
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

    HEADERS = {
        "Content-Type": "application/json",
        "api-key": API_KEY,
    }

    ######### prompt #########
    MESSAGE = "give me a simple python3 script that will help me interact with you"
    payload = prompt(MESSAGE)
    ##########################

    try:
        response = requests.post(API_ENDPOINT, headers=HEADERS, json=payload)
        response.raise_for_status()
    except requests.RequestException as e:
        raise SystemExit(f"Failed to make the request. Error: {e}")

    resp_as_json = response.json()
    resp_message = resp_as_json["choices"][0]["message"]["content"]
    print(resp_message)
