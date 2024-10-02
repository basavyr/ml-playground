import torch
import torch.nn as nn
import torch.nn.functional as F


import utils

EMBEDDING_DIM = 10


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super(Embedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor):
        x = self.embedding(x)
        return x


class Tokenizer():
    tokenized_symbols = r' `~!@#$%^&*()-_=+[{]}\|;:\'",<.>/?'

    def __init__(self, dictionary: list[str]):
        self.dictionary = dictionary
        self.num_embeddings = len(dictionary)
        self.word_indices = self.word_to_idx()

    def deconstruct_input(self, input_string):
        return utils.split_input_by_symbols(
            input_string, self.tokenized_symbols)

    def word_to_idx(self):
        word_indices = {word: idx for idx, word in enumerate(self.dictionary)}
        word_indices.update({"<UNK>": len(word_indices)})
        return word_indices

    def tokenize(self, input_string: str) -> list[int]:
        deconstructed = self.deconstruct_input(input_string)

        token_ids = [self.word_indices.get(
            word, self.word_indices['<UNK>']) for word in deconstructed]

        return token_ids

    def to_tensor(self, input_string: str) -> torch.Tensor:
        token_ids = self.tokenize(input_string)

        return torch.tensor(token_ids, dtype=torch.int)


def create_word_embedding(tokenizer: Tokenizer, embedding: Embedding, input_string: str):
    input_token_ids = tokenizer.to_tensor(input_string)
    word_embedding = embedding(input_token_ids)
    return word_embedding


if __name__ == "__main__":

    T = Tokenizer(utils.dictionary)
    num_embeddings = len(T.word_indices)
    WE = Embedding(num_embeddings, EMBEDDING_DIM)

    input_strings = [
        "Hey there, how are you?",
        "Hey there, how are you?",
        "I like coffee and good morning!",
        "What is your name, please?",
        "Thanks for coming to the office.",
        "We will watch a movie tonight.",
        "How do you code this algorithm?",
        "I love reading books on a sunny day.",
        "The computer and keyboard are broken.",
        "Are you happy with the game results?",
        "Is the office open today or closed?"
    ]

    for input in input_strings:
        we = create_word_embedding(T, WE, input)
        print(we)
