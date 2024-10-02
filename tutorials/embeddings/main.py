import torch
import torch.nn as nn
import torch.nn.functional as F


import utils

EMBEDDING_DIM = 10


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, weight_init: str, pre_trained_weights=None):
        super(Embedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

        if pre_trained_weights is not None:
            # Set the weights of the embedding layer to the pre-trained weights
            self.embedding.weight = nn.Parameter(pre_trained_weights)
            self.embedding.weight.requires_grad_(False)
        else:
            # Initialize the embedding weights based on the weight_init argument
            if weight_init == "xavier":
                nn.init.xavier_uniform_(self.embedding.weight)
            elif weight_init == "kaiming":
                nn.init.kaiming_uniform_(
                    self.embedding.weight, nonlinearity='relu')
            elif weight_init == "normal":
                nn.init.normal_(self.embedding.weight, mean=0.0, std=0.1)
            elif weight_init == "uniform":
                nn.init.uniform_(self.embedding.weight, a=-0.05,
                                 b=0.05)  # Uniform distribution
            elif weight_init == "constant":
                # All weights set to a constant value
                nn.init.constant_(self.embedding.weight, 0.5)
            elif weight_init == "orthogonal":
                # Orthogonal initialization
                nn.init.orthogonal_(self.embedding.weight)
            elif weight_init == "sparse":
                # Sparse initialization
                nn.init.sparse_(self.embedding.weight, sparsity=0.1, std=0.01)
            else:
                # Default: Apply Xavier uniform initialization if no specific method is provided
                nn.init.xavier_uniform_(self.embedding.weight)

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

    pre_trained_weights = torch.randint(
        0, 10, (num_embeddings, EMBEDDING_DIM), dtype=torch.float)
    weight_inits = ["xavier", "uniform", "kaiming",
                    "orthogonal", "normal", "constant", "sparse"]

    WE = Embedding(num_embeddings, EMBEDDING_DIM, "kaiming")

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

    word_embeddings = []
    for input in input_strings:
        we = create_word_embedding(T, WE, input)
        word_embeddings.append(we)

    for i in range(len(word_embeddings)-1):
        we1, we2 = word_embeddings[i], word_embeddings[i+1]
        if we1.shape[0] != we2.shape[0]:
            print(
                "Can't compute cosine similarity due to the word embeddings shape inconsistency")
            # TODO: add support for padding in the word embeddings such that cosine similarity can be evaluated for input strings that do not have the same token length
            # source: https://stackoverflow.com/questions/66374955/computing-cosine-distance-with-differently-shaped-tensors
        else:
            coss = F.cosine_similarity(we1, we2)
            print(coss)
