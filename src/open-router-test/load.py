import requests
import json
import dotenv
import os
dotenv.load_dotenv()

def process_model_resp(resp: dict):
	resp_message = resp['choices'][0]['message']
	return resp_message['content']


def create_prompt(user_content:str):
	return [{"role" : "user", "content" : f'{user_content}'}]

API_KEY = os.environ.get("OPEN_ROUTER_APIKEY")
prompt=create_prompt("What is a GPU?")

response = requests.post(
  url="https://openrouter.ai/api/v1/chat/completions",
  headers={
    "Authorization": f"Bearer {API_KEY}",
  },
  data=json.dumps({
    "model": "meta-llama/llama-3.1-8b-instruct:free", # Optional
    "messages": prompt,
    "top_p": 1,
    "temperature": 0.2,
    "frequency_penalty": 0,
    "presence_penalty": 0,
    "repetition_penalty": 1,
    "top_k": 0,
  })
)


resp = response.json()


model_output = process_model_resp(resp)

print(model_output)
