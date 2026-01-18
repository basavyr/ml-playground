import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer
from typing import List

DEFAULT_GPT = "gpt2"
DEFAULT_PHI3 = 'microsoft/Phi-3-mini-4k-instruct'


class DecoderManager:
    def __init__(self, device: torch.device, force_gpt2=False):
        self.device = device
        self.models = []
        self.model_names = []

        # Initialize tokenizer as None first
        self.tokenizer = None

        if force_gpt2:
            model_configs: List[str] = [
                DEFAULT_GPT,
                DEFAULT_GPT,
                DEFAULT_GPT]
            model_class = GPT2LMHeadModel
            tokenizer_class = GPT2Tokenizer
        else:
            model_configs: List[str] = [
                DEFAULT_PHI3,
                DEFAULT_PHI3,
                DEFAULT_PHI3]
            model_class = AutoModelForCausalLM
            tokenizer_class = AutoTokenizer

        model_desc = f"Models: 3× {model_configs[0]}"
        for model_name in model_configs:
            model = model_class.from_pretrained(model_name)
            model.to(device)
            self.models.append(model)
            # Extract model name from path
            self.model_names.append(model_name.split('/')[-1])

            # Use tokenizer from first model
            if self.tokenizer is None:
                self.tokenizer = tokenizer_class.from_pretrained(model_name)

        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Update model description for output
        self.model_desc = model_desc

    def decode_embedding(self, embedding: torch.Tensor, model_idx: int = 0) -> str:
        model = self.models[model_idx]

        # Convert embedding to text prompt by sampling features
        embedding_features = embedding.cpu().numpy()[
            :50]  # Take first 50 features
        prompt_tokens = [f"feat_{i}:{val:.2f}" for i,
                         val in enumerate(embedding_features[:10])]
        prompt = " ".join(prompt_tokens) + " generate:"

        inputs = self.tokenizer(
            prompt, return_tensors='pt', truncation=True, max_length=128).to(self.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=1
            )

        generated_text = self.tokenizer.decode(
            outputs[0], skip_special_tokens=True)

        # Extract only the generated part (after the prompt)
        if "generate:" in generated_text:
            result = generated_text.split("generate:")[1].strip()
        else:
            result = generated_text.strip()

        return result

    def decode_all(self, embeddings: List[torch.Tensor]) -> List[str]:
        results = []
        for i, embedding in enumerate(embeddings):
            try:
                result = self.decode_embedding(embedding, i)
                results.append(result)
            except Exception as e:
                results.append(f"Error generating with model {i}: {str(e)}")
        return results
