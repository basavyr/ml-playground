import random
import pandas as pd
import time

import azure_v2 as az2
import datetime


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


def save_model_output(output_dir: str, output_file: str, model_output: str):
    import json
    c_date = str(datetime.datetime.now().date())
    c_time = str(datetime.datetime.now().time())
    file_suffix = f'{c_date}-{c_time}'[:-7]
    with open(f'{output_dir}/{output_file}-{file_suffix}.jsonl', 'w') as dumper:
        # model_output = model_output[len(
        #     "```jsonl\n"):-len("\n```\n")].split(r"""{"prompt":""")
        model_output = model_output.replace(
            "\n", "\\n").split(r"""{"prompt":""")
        for line in model_output[1:]:
            t = fr"""{{"prompt": {line.strip()}"""
            if t.endswith("\\n\\n"):
                dumper.write(t[:-len("\\n\\n")])
            else:
                dumper.write(t)
            dumper.write("\n")


def generate_output_file(output_dir: str, system_prompt: str):
    model_output_file = "az-output"

    messages = generate_messages(system_prompt)

    api_key, api_version, endpoint, deployment = az2.load_env()
    az = az2.AzureInterface(api_key, api_version, endpoint, deployment)

    example = az.create_chat_completion(messages)

    save_model_output(output_dir, model_output_file, example)


def main():
    volumes = "./volumes"

    system_prompt_file = "gpt-4o.prompt"
    system_prompt = get_system_prompt(system_prompt_file)

    start = time.time()

    max_iter = 2
    for idx in range(max_iter):
        generate_output_file(volumes, system_prompt)
        print(f'Generated {idx+1}/{max_iter} output files')
    print(
        f'Generating output: {round(time.time()-start,2)} s')


if __name__ == "__main__":
    main()
