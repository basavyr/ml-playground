from typing import Tuple, List, Dict, Any
import torch
from models.embedders import EmbeddingManager
from models.decoders import DecoderManager


class EmbeddingSplitter:
    def __init__(self, device: torch.device, complexity_threshold: int = 50, force_gpt2: bool = False):
        self.device = device
        self.complexity_threshold = complexity_threshold
        self.embedder = EmbeddingManager(device)
        self.decoder = DecoderManager(device, force_gpt2=force_gpt2)
        
    def process(self, prompt: str) -> Dict[str, Any]:
        # Step 1: Generate initial embedding E
        initial_embedding = self.embedder.encode_primary(prompt)
        
        # Step 2: Assess complexity and decide whether to split
        token_count = self._get_token_count(prompt)
        should_split = self._assess_complexity(token_count)
        
        if should_split:
            # Step 3: Create 3 sub-embeddings E1, E2, E3 using semantic chunking
            sub_embeddings = self.embedder.create_sub_embeddings(initial_embedding, prompt)
        else:
            # Use the original embedding split into 3 parts
            sub_embeddings = self._split_embedding_simple(initial_embedding)
        
        # Step 4: Decode each sub-embedding with different GPT-2 models
        outputs = self.decoder.decode_all(list(sub_embeddings))
        
        return {
            'prompt': prompt,
            'initial_embedding': initial_embedding,
            'sub_embeddings': sub_embeddings,
            'outputs': outputs,
            'token_count': token_count,
            'should_split': should_split,
            'model_names': self.decoder.model_names
        }
    
    def _get_token_count(self, text: str) -> int:
        """Get token count for complexity assessment."""
        tokens = self.embedder.primary_model.tokenizer(text, truncation=True)
        return len(tokens['input_ids'])
    
    def _assess_complexity(self, token_count: int) -> bool:
        """Assess if embedding is complex enough to warrant splitting."""
        return token_count > self.complexity_threshold
    
    def _split_embedding_simple(self, embedding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Simple splitting when complexity is low - split embedding into thirds."""
        if embedding.dim() == 1:
            # For 1D embeddings, create three similar embeddings
            e1 = embedding.clone()
            e2 = embedding.clone() * 0.9  # Slight variation
            e3 = embedding.clone() * 1.1  # Slight variation
        else:
            # For higher dimensional embeddings, split along first dimension
            third = embedding.size(0) // 3
            e1 = embedding[:third] if third > 0 else embedding[:1]
            e2 = embedding[third:2*third] if 2*third <= embedding.size(0) else embedding[third:third+1]
            e3 = embedding[2*third:] if 2*third < embedding.size(0) else embedding[-1:]
            
            # Ensure all have same dimension by padding/interpolation
            target_size = embedding.size(0)
            if e1.size(0) != target_size:
                e1 = torch.nn.functional.interpolate(e1.unsqueeze(0).unsqueeze(0), size=target_size, mode='linear').squeeze()
            if e2.size(0) != target_size:
                e2 = torch.nn.functional.interpolate(e2.unsqueeze(0).unsqueeze(0), size=target_size, mode='linear').squeeze()
            if e3.size(0) != target_size:
                e3 = torch.nn.functional.interpolate(e3.unsqueeze(0).unsqueeze(0), size=target_size, mode='linear').squeeze()
        
        return e1, e2, e3
    
    def get_embedding_info(self, embeddings: List[torch.Tensor]) -> List[str]:
        info = []
        for i, emb in enumerate(embeddings, 1):
            shape_str = f"E{i}: {emb.shape}"
            size_mb = float(emb.numel() * emb.element_size()) / (1024**2)
            info.append(f"{shape_str}, {size_mb:.4f}MB")
        return info