import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
import numpy as np

import os

DEBUG = os.getenv("DEBUG", "0")


torch.manual_seed(1)

# Sample data
raw_data = ['hello', 'openai', 'data', 'python', 'world hello', 'python', 'code', 'hello', 'python', 'text',
            'hello', 'gicu2', 'code', 'data', 'openai', 'code', 'python', 'gicu', 'openai', 'code',
            'hello world', 'data', 'data', 'gicu', 'world hello', 'hello world', 'world hello', 'openai',
            'openai', 'openai', 'gicu2', 'hello world', 'gicu', 'hello', 'text', 'data', 'gicu', 'text',
            'data', 'python', 'data', 'text', 'gicu2', 'gicu2', 'text', 'hello world', 'hello', 'world hello',
            'hello world', 'hello', 'data', 'code', 'hello world', 'python', 'code', 'world hello',
            'world hello', 'python', 'openai', 'gicu', 'world hello', 'hello', 'openai', 'code', 'gicu2',
            'hello', 'hello world', 'text', 'gicu2', 'code', 'gicu', 'gicu2', 'hello world', 'gicu', 'gicu', "gicu", "test", "iphone"]


def build_word_to_idx(raw_data: list[str] | str) -> tuple[dict, torch.Tensor]:
    """
    basic tokenization method, where each word in a list of strings will become a token
    - constructs a vocabulary with all the words and their respective indices
    - 
    """
    # add support for list of strings or just strings
    if isinstance(raw_data, list):
        words = " ".join(raw_data).split()
    elif isinstance(raw_data, str):
        words = raw_data.split(" ")
    counter = Counter(words)
    # Counter({'hello': 23, 'world': 15, 'openai': 8, 'data': 8, 'code': 8, 'gicu': 8, 'python': 7, 'gicu2': 7, 'text': 6})
    words = {word: idx for idx, word in enumerate(counter)}
    indices = torch.tensor([v for _, v in words.items()], dtype=torch.long)
    return words, indices


# Define the embedding model
class EmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(EmbeddingModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, x):
        return self.embedding(x)


if __name__ == "__main__":
    EMBEDDING_DIM = 3  # size of an embedding vector
    # The words are first transformed into a set of tokens, where every integer represents a word.
    # For each token, the embedding layer will encode that token into a vector.
    # This vector belongs to a high-dimensional vector space, with the dimensionality specified by EMBEDDING_DIM.
    words, indices = build_word_to_idx(raw_data)
    vocab_size = len(words)
    batch_size = 3

    # Create the dataset and dataloader
    tensor_dataset = TensorDataset(indices)
    dataloader = DataLoader(
        tensor_dataset, batch_size=batch_size, shuffle=True)

    # Initialize the model
    model = EmbeddingModel(vocab_size, EMBEDDING_DIM)

    def test_embedding(model: nn.Module):
        raw_list = ["gicu", "test", "iphone"]
        words, indices = build_word_to_idx(raw_list)
        output = model(indices)
        for idx, word in enumerate(words):
            print(f'{word} -> {output[idx]}')

    # for idx, batch_list in enumerate(dataloader):
    #     batch = batch_list[0]
    #     output = model(batch)
    #     print(output)

    test_embedding(model)