import json

from datasets import load_dataset
from trl import SFTConfig, SFTTrainer


def prompt_format(raw_examples: list[dict]):
    output_prompts = []

    for idx in range(len(raw_examples)):
        sample = raw_examples[idx]
        instruction = sample["prompt"]
        response = sample["response"]
        text = f'''Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
            
            ### Instruction:
            {instruction}
            
            ### Response:
            {response}
            '''
        output_prompts.append(text)

    return output_prompts


def read_jsonl(file: str):
    with open(f'{file}', 'r') as reader:
        data = reader.readlines()
        json_lines = []
        for line in data:
            line_as_json = json.loads(line)
            json_lines.append(line_as_json)

    return json_lines


if __name__ == "__main__":
    data_jsonl = read_jsonl("dataset.jsonl")
    prompts = prompt_format(data_jsonl)
