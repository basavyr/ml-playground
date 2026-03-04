import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer
from typing import List


class DecoderManager:
    def __init__(self, device: torch.device, model_names: List[str]):
        self.device = device
        self.models = []
        self.model_names = model_names
        self.tokenizer = None

        for model_name in model_names:
            try:
                # Auto-detect model class and tokenizer
                model = AutoModelForCausalLM.from_pretrained(model_name)
                tokenizer = AutoTokenizer.from_pretrained(model_name)
            except Exception as e:
                try:
                    # Fallback to GPT2 class
                    model = GPT2LMHeadModel.from_pretrained(model_name)
                    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
                except Exception as e:
                    print(f"Error loading model {model_name}: {e}")
                    raise

            model.to(device)
            self.models.append(model)

            # Use tokenizer from first model for all models
            if self.tokenizer is None:
                self.tokenizer = tokenizer

        self.tokenizer.pad_token = self.tokenizer.eos_token

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
