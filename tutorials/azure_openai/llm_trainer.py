
import azure_v2 as az2
import random


import pandas as pd


def process_examples(prev_examples: list):
    # Initialize lists to store prompts and responses
    prompts = []
    responses = []

    # Parse out prompts and responses from examples
    for example in prev_examples:
        try:
            split_example = example.split('-----------')
            prompts.append(split_example[1].strip())
            responses.append(split_example[2].strip())
        except:
            pass

    # Create a DataFrame
    df = pd.DataFrame({
        'prompt': prompts,
        'response': responses
    })

    df.head()

    # Save the dataframes to .jsonl files
    df.to_json('train.jsonl', orient='records', lines=True)


def generate_messages(model_description: str, prev_examples: list):
    messages = [
        {
            "role": "system",
            "content": f"""You are generating data which will be used to train a machine learning model.\n\nYou will be given a high-level description of the model we want to train and from that, you will generate data samples, each with a prompt/response pair.\n\nYou will do so in this format:\n```\nprompt\n-----------\nresponse_goes_here\n-----------\n```\n\nOnly one prompt/response pair should be generated per turn.\n\nFor each turn, make the example slightly more complex than the last, while ensuring diversity. Coding questions can also be given.\n\nMake sure your samples are unique and diverse, yet high-quality and complex enough to train a well-performing model.\n\nHere is the type of model we want to train:\n`{model_description}`"""
        }
    ]

    K = 5
    if len(prev_examples) > 0:
        if len(prev_examples) > K:
            prev_examples = random.sample(prev_examples, K)
        for example in prev_examples:
            messages.append({
                "role": "assistant",
                "content": example.strip()
            })

    return messages


def main():
    api_key, api_version, endpoint, deployment = az2.load_env()

    az = az2.AzureInterface(api_key, api_version, endpoint, deployment)

    model_description = "A model that takes in a cybersecurity-oriented question, and responds with a well-reasoned, step-by-step response. The model is also able to solve coding tasks that are specific to cybersecurity."
    number_of_examples = 3

    prev_examples = []
    for _ in range(number_of_examples):
        messages = generate_messages(model_description, prev_examples)
        example = az.create_chat_completion(messages)
        prev_examples.append(example)

    process_examples(prev_examples)


if __name__ == "__main__":
    main()
