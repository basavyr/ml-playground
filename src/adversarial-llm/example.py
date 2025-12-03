import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F


# Generate a response from the model
def generate_response(input_prompt: str, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, max_length: int = 256):
    input_tokens = tokenizer(input_prompt, return_tensors="pt").input_ids

    output = model.generate(input_tokens, max_length=len(input_tokens[0]) + 20
                            )
    return tokenizer.decode(output[0], skip_special_tokens=True)


# Define a simple function to make small perturbations to tokens
def create_adversarial_example(input_prompt: str, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, epsilon: float = 1.0e-8):
    tokenized_input = tokenizer(
        input_prompt, return_tensors="pt").input_ids  # Tokenize input text

    # Generate the the embeddings for every token
    input_tokens = tokenized_input.clone().detach()
    input_embeddings = model.transformer.wte(input_tokens)

    # Generate a small perturbation
    perturbation = epsilon * torch.randn_like(input_embeddings)
    perturbed_embeddings = input_embeddings + perturbation

    # Decode perturbed embeddings back to text
    with torch.no_grad():
        outputs = model(inputs_embeds=perturbed_embeddings)
    perturbed_logits = outputs.logits
    perturbed_ids = torch.argmax(perturbed_logits, dim=-1)

    print(perturbed_ids.shape)
    print(perturbed_ids)

    import sys
    sys.exit()
    # Convert perturbed tokens back to text
    adversarial_text = tokenizer.decode(
        perturbed_ids[0], skip_special_tokens=True)
    return adversarial_text


if __name__ == "__main__":
    # Load your small LLM from Hugging Face
    model_name = "gpt2"  # Replace this with the model you're using
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Set your original prompt and expected logical response
    input_prompt = "In my car there is a dog. What is inside the car?"
    target_response = "In the car it is a dog."

    # Generate an adversarial sample
    adversarial_prompt = create_adversarial_example(
        input_prompt, model, tokenizer)
    print("Original input prompt:", input_prompt)
    print('################')
    print("Adversarial input prompt:", adversarial_prompt)
    print('################')

    # Get responses for original and adversarial text
    original_response = generate_response(input_prompt, model, tokenizer)
    adversarial_response = generate_response(
        adversarial_prompt, model, tokenizer)

    print("\nExpected Response:", target_response)
    print("Original Response:", original_response)
    print("Adversarial Response:", adversarial_response)
