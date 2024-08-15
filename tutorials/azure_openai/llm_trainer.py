
import pandas as pd
import time

import azure_v2 as az2
import datetime


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
    now = datetime.datetime.now()
    file_suffix = f'{now.date()}-{now.time()}'[:-7]
    with open(f'{output_dir}/{output_file}-{file_suffix}.jsonl', 'w') as dumper:
        model_output = model_output.replace(
            "\n", "\\n").split(r"""{"prompt":""")[1:]
        for line in model_output:
            reconstructed_line = fr"""{{"prompt": {line.strip()}"""
            if reconstructed_line.endswith("\\n\\n"):
                dumper.write(reconstructed_line[:-len("\\n\\n")])
            elif reconstructed_line.endswith("\\n"):
                dumper.write(reconstructed_line[:-len("\\n")])
            else:
                dumper.write(reconstructed_line)
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

    if system_prompt is None:
        print(
            f'No prompt file available. Please make sure < {system_prompt_file} > is in the correct path.')
        exit(1)

    start = time.time()

    OUTPUT_FILES = 5  # change the number of output files with data that should be generated

    for idx in range(OUTPUT_FILES):
        generate_output_file(volumes, system_prompt)
        print(f'Generated {idx+1}/{OUTPUT_FILES} output files')
    print(
        f'Generating {OUTPUT_FILES} files took: {round(time.time()-start,2)} s')


if __name__ == "__main__":
    main()
