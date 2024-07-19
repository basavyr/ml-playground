from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('stabilityai/stablelm-zephyr-3b')
model = AutoModelForCausalLM.from_pretrained(
    'stabilityai/stablelm-zephyr-3b',
    device_map="cpu"
) # must force to CPU otherwise issue with MPS device will arise

# github issue (my comment https://github.com/coqui-ai/TTS/issues/3758#issuecomment-2239388385)
# original issue: https://github.com/coqui-ai/TTS/issues/3758

prompt = [{'role': 'user', 'content': 'List 3 synonyms for the word "tiny"'}]
inputs = tokenizer.apply_chat_template(
    prompt,
    add_generation_prompt=True,
    return_tensors='pt'
)

tokens = model.generate(
    inputs.to(model.device),
    max_new_tokens=1024,
    temperature=0.8,
    do_sample=True
)


print(tokenizer.decode(tokens[0], skip_special_tokens=False))
