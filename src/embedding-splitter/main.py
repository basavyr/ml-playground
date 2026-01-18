import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
from utils import get_device, measure_performance, calculate_bandwidth, format_results
from models.splitter import EmbeddingSplitter

# Model configuration
EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5"
GPT2_MODEL = "gpt2"
PHI3_MODEL = 'microsoft/Phi-3-mini-4k-instruct'


def main():
    parser = argparse.ArgumentParser(description='Embedding Splitter POC')
    parser.add_argument('prompt', nargs='?', default='The future of artificial intelligence is fascinating and will transform many industries including healthcare, finance, transportation, education, manufacturing, agriculture, energy, retail, entertainment, and government services through advanced machine learning algorithms, neural networks, deep learning frameworks, natural language processing, computer vision, robotics, and autonomous systems that will revolutionize how we work, live, and interact with technology in the coming decades.', 
                       help='Input prompt for processing')
    parser.add_argument('--threshold', type=int, default=50, 
                       help='Token count threshold for complexity-based splitting')
    parser.add_argument('--forcegpt2', action='store_true',
                       help='Force use of 3 GPT2-large instances instead of modern models')
    args = parser.parse_args()
    
    # Determine decoder models based on flag
    if args.forcegpt2:
        decoder_models = [GPT2_MODEL, GPT2_MODEL, GPT2_MODEL]
    else:
        decoder_models = [PHI3_MODEL, PHI3_MODEL, PHI3_MODEL]
    
    device = get_device()
    splitter = EmbeddingSplitter(device, complexity_threshold=args.threshold, embedding_model=EMBEDDING_MODEL, decoder_models=decoder_models)
    
    # Print configuration
    print(f"=== CONFIGURATION ===")
    print(f"Device: {device}")
    print(f"Embedding Dimension: {splitter.embedder.primary_model.get_sentence_embedding_dimension()}")
    print(f"Complexity Threshold: {args.threshold} tokens")
    print(f"Embedding Model: {EMBEDDING_MODEL}")
    # Get unique model name for display
    unique_model = splitter.decoder.model_names[0] if splitter.decoder.model_names else "unknown"
    print(f"Decoder Models: 3× {unique_model}")
    print()
    
    # Process with performance tracking
    results, duration = measure_performance(splitter.process, args.prompt)
    
    # Calculate bandwidth using the embeddings
    all_tensors = [results['initial_embedding']] + list(results['sub_embeddings'])
    bandwidth = calculate_bandwidth(all_tensors, duration)
    
    # Display results
    format_results(results, duration, bandwidth)


if __name__ == "__main__":
    main()