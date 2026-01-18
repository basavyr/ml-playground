import time
import torch
from typing import Dict, Any


def get_device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else 
                       "cuda" if torch.cuda.is_available() else "cpu")


def calculate_bandwidth(tensors: list, duration: float) -> float:
    total_mb = sum(float(t.numel() * t.element_size()) / (1024**2) for t in tensors)
    return total_mb / duration


def measure_performance(func, *args, **kwargs) -> tuple:
    start = time.monotonic_ns()
    result = func(*args, **kwargs)
    duration = (time.monotonic_ns() - start) / 1e9
    return result, duration


def calculate_cosine_similarity(tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
    """Calculate cosine similarity between two tensors."""
    # Ensure both tensors are 1D and on same device
    if tensor1.dim() > 1:
        tensor1 = tensor1.flatten()
    if tensor2.dim() > 1:
        tensor2 = tensor2.flatten()
    
    # Move to CPU for calculation
    tensor1 = tensor1.cpu()
    tensor2 = tensor2.cpu()
    
    # Calculate cosine similarity
    similarity = torch.nn.functional.cosine_similarity(tensor1.unsqueeze(0), tensor2.unsqueeze(0), dim=1)
    return similarity.item()

def format_tensor_with_similarity(tensor: torch.Tensor, original_tensor: torch.Tensor, name: str) -> None:
    """Show similarity with original embedding."""
    similarity = calculate_cosine_similarity(tensor, original_tensor)
    print(f"{name}: similarity = {similarity:.4f}")

def format_results(results: Dict[str, Any], duration: float, bandwidth: float) -> None:
    print("=== RESULTS ===")
    print(f"Prompt: \"{results['prompt']}\"")
    print(f"Tokens: {results['token_count']} | Splitting: {'Yes' if results.get('should_split', False) else 'No'}")
    
    no_split = results.get('no_split', False)
    
    if not no_split:
        print(f"\n--- SUB-EMBEDDING SIMILARITY ---")
        sub_names = ["E1", "E2", "E3"]
        sub_embeddings = results.get('sub_embeddings', [])
        for i, (emb, name) in enumerate(zip(sub_embeddings, sub_names)):
            format_tensor_with_similarity(emb, results['initial_embedding'], name)
    
    print(f"\n--- GENERATED RESPONSES ---")
    model_names = results.get('model_names', ['Unknown'])
    outputs = results.get('outputs', [])
    
    if no_split:
        print(f"Y ({model_names[0]}): \"{outputs[0]}\"")
    else:
        for i, (output, model_name) in enumerate(zip(outputs, model_names)):
            print(f"Y{i+1} ({model_name}): \"{output}\"")
    
    print(f"\n--- PERFORMANCE ---")
    print(f"Time: {duration:.2f}s | Bandwidth: {bandwidth:.2f}MB/s")
    
    if no_split:
        emb = results['initial_embedding']
        total_memory = emb.numel() * emb.element_size()
    else:
        sub_embeddings = results.get('sub_embeddings', [])
        total_memory = sum(emb.numel() * emb.element_size() for emb in sub_embeddings)
        
    total_memory_mb = total_memory / (1024**2)
    print(f"Memory: {total_memory_mb:.2f}MB")