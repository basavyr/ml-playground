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


def generate_messages(system_prompt: str | None):
    system_prompt = "You are a helpful assistant." if system_prompt is None else system_prompt
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        }
    ]

    return messages


def get_system_prompt(system_prompt_file: str):
    import os
    if not os.path.exists(system_prompt_file):
        return None
    else:
        with open(system_prompt_file, 'r+') as reader:
            data = reader.read()
            return data


def main():
    system_prompt_file = "gpt-4o.prompt"
    system_prompt = get_system_prompt(system_prompt_file)

    api_key, api_version, endpoint, deployment = az2.load_env()
    az = az2.AzureInterface(api_key, api_version, endpoint, deployment)

    start = time.time()
    messages = generate_messages(system_prompt)
    example = az.create_chat_completion(messages)

    print(example)
    print(
        f'Generating output: {round(time.time()-start,2)} s')


if __name__ == "__main__":
    main()
