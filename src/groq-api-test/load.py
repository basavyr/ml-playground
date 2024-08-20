import os

from groq import Groq
import dotenv

dotenv.load_dotenv()


client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

models = ["mixtral-8x7b-32768","llama3-groq-70b-8192-tool-use-preview","llama-3.1-70b-versatile","llama3-70b-8192","llama-3.1-8b-instant","llama3-8b-8192"]


def create_user_prompt(user_content: str):
	return [{"role": "user", "content": f'{user_content}'}]


prompt = create_user_prompt("What is self-attention in the context of transformers?")

chat_completion = client.chat.completions.create(
    messages=prompt,
    model=models[2],
)


def save_prompt_output(prompt_response:str):
	with open('model.output','w+') as writer:
		writer.write(prompt_response)


model_output = chat_completion.choices[0].message.content

save_prompt_output(model_output)
