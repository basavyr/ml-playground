import json

from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset

from transformers import pipeline


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


def formatting_prompts_func(example):
    output_texts = []
    for i in range(50):
        text = f"""###Below is a question or a task. Write a response that appropriately completes the request.

        ### Prompt:
        {example['prompt'][i]}
        
        ### Instruction response:
        {example['response'][i]}
        """
        output_texts.append(text)
    return output_texts


def read_jsonl(file: str):
    with open(f'{file}', 'r') as reader:
        data = reader.readlines()
        json_lines = []
        for line in data:
            line_as_json = json.loads(line)
            json_lines.append(line_as_json)

    return json_lines


def model_train(model_name: str, dataset_path: str, sft_config: SFTConfig):
    dataset = load_dataset("json", data_files=dataset_path, split="train")

    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # source: https://stackoverflow.com/questions/70544129/transformers-asking-to-pad-but-the-tokenizer-does-not-have-a-padding-token
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        # we need to resize the token embeddings
        model.resize_token_embeddings(
            len(tokenizer))  # Resize token embeddings

    # Initialize trainer with the custom configuration
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        tokenizer=tokenizer,
        train_dataset=dataset,
        formatting_func=formatting_prompts_func,  # Use your custom formatting function
    )

    trainer.train()
    trainer.save_model(f'{model_name.split("/")[-1]}-trained')


def model_eval(prompt: str, model_name: str):
    pipe = pipeline("text-generation", model=model_name)
    print(pipe(prompt)
          [0]["generated_text"])


def batch_eval(trained_model_name: str, prompt_list: list[str]):
    for prompt in prompt_list:
        print("\n-----------------------")
        model_eval(prompt, trained_model_name)
        print("-----------------------\n")


if __name__ == "__main__":
    model_name = "openai-community/gpt2"
    dataset_path = "dataset.jsonl"

    sft_config = SFTConfig(
        max_seq_length=1024,  # Adjust based on average response length
        output_dir="./gpt2-trained",
        packing=False,
        num_train_epochs=30,
    )
    model_train(model_name, dataset_path, sft_config)

    trained_model_name = "gpt2-trained"
    prompts = [
        "How does the cranking model account for pairing correlations in rotating nuclei?",
        "What role do deformation-driving orbitals play in determining nuclear shape?",
        "How does the wobbling frequency vary with increasing angular momentum in a triaxial nucleus?",
        "What is the effect of gamma softness on the rotational spectra of nuclei?",
        "How do pairing vibrations influence the moment of inertia in deformed nuclei?"
    ]
    batch_eval(trained_model_name, prompts)
