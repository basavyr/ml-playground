#!/usr/bin/env python
import os
import requests

# https://stackoverflow.com/a/65694371
API_KEY = os.environ.get("AZURE_OPENAI_APIKEY", None)
API_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", None)

if API_ENDPOINT is None or API_ENDPOINT is None:
    print('The Azure API Key and Endpoint are not configured. Please make sure you have .env file with these variables')
    exit(1)

headers = {
    "Content-Type": "application/json",
    "api-key": API_KEY,
}


def prompt(prompt_message: str = "Hey there! Where are you hosted and by which company?"): return {
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
                    "text": f"{prompt_message}"
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
    "temperature": 0.7,
    "top_p": 0.95,
    "max_tokens": 800
}


######### prompt #########
MESSAGE = "give me a simple python3 script that will help me interact with you"
payload = prompt(MESSAGE)
##########################

try:
    response = requests.post(API_ENDPOINT, headers=headers, json=payload)
    # Will raise an HTTPError if the HTTP request returned an unsuccessful status code
    response.raise_for_status()
except requests.RequestException as e:
    raise SystemExit(f"Failed to make the request. Error: {e}")


resp_as_json = response.json()
resp_message = resp_as_json["choices"][0]["message"]["content"]
print(resp_message)
