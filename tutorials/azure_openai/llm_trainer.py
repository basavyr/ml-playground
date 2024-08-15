
import pandas as pd
import time

import azure_v2 as az2
import datetime


def generate_prompt_message(system_prompt: str | None) -> list[dict]:
    system_prompt = "You are a helpful assistant." if system_prompt is None else system_prompt
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        }
    ]

    return messages


def get_system_prompt(system_prompt_file: str) -> str | None:
    import os
    if not os.path.exists(system_prompt_file):
        return None
    else:
        with open(system_prompt_file, 'r+') as reader:
            data = reader.read()
            return data


def generate_prompt_response(prompt_message: str) -> str | None:
    api_key, api_version, endpoint, deployment = az2.load_env()

    az = az2.AzureInterface(api_key, api_version, endpoint, deployment)
    prompt_response = az.create_chat_completion(prompt_message)

    return prompt_response


def save_prompt(output_dir: str, prompt_response: str) -> None:
    output_file = "az-output"
    now = datetime.datetime.now()
    file_suffix = f'{now.date()}-{now.time()}'[:-7]
    with open(f'{output_dir}/{output_file}-{file_suffix}.jsonl', 'w') as dumper:
        prompt_response = prompt_response.replace(
            "\n", "\\n").split(r"""{"prompt":""")[1:]
        for line in prompt_response:
            reconstructed_line = fr"""{{"prompt": {line.strip()}"""
            if reconstructed_line.endswith("\\n\\n"):
                dumper.write(reconstructed_line[:-len("\\n\\n")])
            elif reconstructed_line.endswith("\\n"):
                dumper.write(reconstructed_line[:-len("\\n")])
            else:
                dumper.write(reconstructed_line)
            dumper.write("\n")


def main():
    OUTPUT_FILES = 5  # change the number of output files with data that should be generated

    output_dir = "./volumes"

    system_prompt_file = "gpt-4o.prompt"
    system_prompt = get_system_prompt(system_prompt_file)
    if system_prompt is None:
        print(
            f'No prompt file available. Please make sure < {system_prompt_file} > is in the correct path.')
        exit(1)

    prompt = generate_prompt_message(system_prompt)

    start = time.time()

    for idx in range(OUTPUT_FILES):
        prompt_response = generate_prompt_response(prompt)
        save_prompt(output_dir, prompt_response)
        print(f'Generated {idx+1}/{OUTPUT_FILES} output files')
    print(
        f'Generating {OUTPUT_FILES} files took: {round(time.time()-start,2)} s')


if __name__ == "__main__":
    main()
