import torch
import torch.nn as nn
import torch.nn.functional as F


import utils

EMBEDDING_DIM = 10


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


if __name__ == "__main__":

    T = Tokenizer(utils.dictionary)

    input = "Hey there, how are you?"

    input_token_ids = T.to_tensor(input)

    print(input_token_ids)
