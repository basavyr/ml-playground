import random
import pandas as pd
import time

import azure_v2 as az2


def process_examples(prev_examples: list):
    # Initialize lists to store prompts and responses
    prompts = []
    responses = []

    # Parse out prompts and responses from examples
    for example in prev_examples:
        try:
            split_example = example.split('-----------')
            prompts.append(split_example[1].strip())
            responses.append(split_example[3].strip())
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
            "content": f"""You are generating data which will be used to train a machine learning model.\n\nYou will be given a high-level description of the model we want to train, and from that, you will generate data samples, each with a prompt/response pair.\n\nYou will do so in this format: prompt-----------$prompt_goes_here-----------response-----------$response_goes_here-----------\n\nOnly one prompt/response pair should be generated per turn.\n\nFor each turn, make the example slightly more complex than the last, while ensuring diversity.\n\nMake sure your samples are unique and diverse, yet high-quality and complex enough to train a well-performing model.\n\nHere is the type of model we want to train:\n{model_description}"""
        }
    ]

    K = 10
    if len(prev_examples) > 0:
        if len(prev_examples) >= 10:
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

    model_description = """A model that takes in a cybersecurity-oriented question, and responds with a well-reasoned, very short and concise answer. The model is also able to solve coding tasks that are specific to cybersecurity. It is aware of the latest tools that are used in the domain, such as nmap, wfuzz, and many more. Responses related to coding and command specific tasks must only contain the actual code or the actual command to be used."""
    number_of_examples = 100

    start = time.time()

    prev_examples = []
    for _ in range(number_of_examples):
        messages = generate_messages(model_description, prev_examples)
        example = az.create_chat_completion(messages)
        prev_examples.append(example)

    process_examples(prev_examples)
    print(
        f'Generating {number_of_examples} samples took: {round(time.time()-start,2)} s')


if __name__ == "__main__":
    main()
