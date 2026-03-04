import os
import time
from typing import Tuple, List
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer


class EmbeddingManager:
    def __init__(self, device: torch.device, embedding_model_name: str):
        self.device = device
        
        # Primary embedder - nomic-embed-text-v1.5 (768d)
        self.primary_model = SentenceTransformer(embedding_model_name, device=str(device), trust_remote_code=True)
        
        # No longer using sub-models - using same primary model for all sub-embeddings
        self.sub_tokenizer = self.primary_model.tokenizer
    
    def encode_primary(self, text: str) -> torch.Tensor:
        return self.primary_model.encode(text, convert_to_tensor=True)
    
    def create_sub_embeddings(self, primary_embedding: torch.Tensor, original_text: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Get token-level embeddings from primary model
        token_embeddings = self._get_token_embeddings(original_text)
        
        # Apply semantic chunking with cosine similarity
        chunk_indices = self._semantic_chunking(token_embeddings)
        
        # Create sub-embeddings from semantic chunks using the SAME primary model
        sub_embeddings = self._create_chunk_embeddings_from_tokens(chunk_indices, token_embeddings)
        
        return sub_embeddings[0], sub_embeddings[1], sub_embeddings[2]
    
    def _get_token_embeddings(self, text: str) -> torch.Tensor:
        """Get token-level embeddings from primary model."""
        # Use the primary model's tokenizer and encoder
        tokens = self.primary_model.tokenizer(text, return_tensors='pt', truncation=True, max_length=8192).to(self.device)
        
        with torch.no_grad():
            # Get token embeddings from the transformer
            outputs = self.primary_model._modules['0'].auto_model(**tokens)
            token_embeddings = outputs.last_hidden_state.squeeze(0)  # [T, d] where d=768
            
        return token_embeddings
    
    def _semantic_chunking(self, token_embeddings: torch.Tensor) -> List[int]:
        """Find split points using cosine similarity between adjacent tokens."""
        if len(token_embeddings) < 3:
            # If too short, split roughly into thirds
            third = len(token_embeddings) // 3
            return [third, 2 * third]
        
        # Calculate cosine similarities between adjacent tokens
        threshold = 0.45
        similarities = []
        for i in range(len(token_embeddings) - 1):
            sim = F.cosine_similarity(
                token_embeddings[i:i+1], 
                token_embeddings[i+1:i+2], 
                dim=-1
            )
            similarities.append(sim.item())
        
        # Find split points where similarity drops below threshold
        split_points = []
        for i, sim in enumerate(similarities):
            if sim < threshold:
                split_points.append(i + 1)  # Split after this token
        
        # If no good split points found, split roughly into thirds
        if len(split_points) < 2:
            third = len(token_embeddings) // 3
            split_points = [third, 2 * third]
        elif len(split_points) > 2:
            # Keep only the 2 best split points (lowest similarities)
            sorted_points = sorted([(i, similarities[i-1]) for i in split_points], key=lambda x: x[1])
            split_points = sorted([x[0] for x in sorted_points[:2]])
            split_points.sort()
        
        return split_points
    
    def _create_chunk_embeddings_from_tokens(self, split_indices: List[int], token_embeddings: torch.Tensor) -> List[torch.Tensor]:
        """Create embeddings for each semantic chunk using the SAME primary model."""
        # Convert split indices to chunk ranges
        chunks = []
        prev_idx = 0
        
        for split_idx in split_indices:
            # Extract token embeddings for this chunk
            chunk_embeddings = token_embeddings[prev_idx:split_idx]
            chunks.append(chunk_embeddings)
            prev_idx = split_idx
        
        # Handle remaining tokens
        if prev_idx < len(token_embeddings):
            remaining_embeddings = token_embeddings[prev_idx:]
            chunks.append(remaining_embeddings)
        
        # Ensure we have exactly 3 chunks
        while len(chunks) < 3:
            # Split the largest chunk
            largest_idx = max(range(len(chunks)), key=lambda i: chunks[i].size(0))
            large_chunk = chunks[largest_idx]
            mid = large_chunk.size(0) // 2
            chunks[largest_idx] = large_chunk[:mid]
            chunks.insert(largest_idx + 1, large_chunk[mid:])
        
        chunks = chunks[:3]
        
        # Convert each chunk to single embedding vector using mean pooling
        sub_embeddings = []
        for chunk_emb in chunks:
            if chunk_emb.size(0) == 0:
                # Fallback for empty chunks
                chunk_emb = torch.zeros(1, 768, device=self.device)
            
            # Mean pool to get single vector representation
            sub_emb = chunk_emb.mean(dim=0)
            sub_embeddings.append(sub_emb)
        
        return sub_embeddings