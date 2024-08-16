# based on the tutorial from Andrew Karpathy
# source: https://www.youtube.com/watch?v=kCc8FmEb1nY

from typing import Tuple
from torch.utils.data import DataLoader
import torch
import os

import llm

torch.manual_seed(1337)

DEBUG_MODE = os.getenv('DEBUG', '0')  # Default to '0' if DEBUG is not set
DEVICE = os.getenv('DEVICE', 'cpu')  # Default to '0' if DEBUG is not set

raw_dataset_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

input_file = "input.txt"


with open(input_file, 'r') as reader:
    data = reader.read()


def get_first_n_sequence(input_data: str, n: int = 100):
    """
    - Prints the first `n` characters of the input sequence in `input_data`
    """
    print(f'Loaded {len(data)} strings into the buffer')
    print(f'First {n} characters:\n{input_data[:n]}')


def get_vocab(input_data: str) -> tuple[list, int]:
    vocab = sorted(list(set(input_data)))
    vocab_size = len(vocab)
    return vocab, vocab_size


# first we need to tokenize the input strings
# the tokenization will result in assigning numbers/indices (so typically integer numbers) to the characters in the vocabulary.
# in this case we work at the character level, but methods applied at the "word" level (or even sub-word level) will be similar
###################
#    TOKENIZER    #
###################

# this tokenizer is composed of an encoder and decoder parts
# the encoder will take the character and assign a token -> encode STR_CHAR->INT_INDEX
# the decoder will take the index and look up on what character it corresponds to -> decode INT_INDEX -> STR_CHAR
# in essence these will be look-up-tables
# other tokenizers: https://github.com/google/sentencepiece
# solution from openai: https://github.com/openai/tiktoken

class Tokenizer():
    """
    A tokenizer class that handles character-level tokenization.

    Attributes:
        vocab (list of str): A list of single-character strings representing the vocabulary.
        word_to_int (dict): A dictionary mapping each character in the vocabulary to a unique integer index.
        int_to_word (dict): A dictionary mapping each integer index back to the corresponding character in the vocabulary.

    Methods:
        __init__(self, vocab):
            Initializes the Tokenizer with the given vocabulary.
            Constructs the word-to-int and int-to-word lookup tables.
    """

    def __init__(self, vocab: list[str]):
        """
        Initializes the Tokenizer with the given vocabulary.

        Args:
            vocab (list of str): A list of single-character strings representing the vocabulary.

        Initializes:
            word_to_int (dict): A dictionary mapping each character in the vocabulary to a unique integer index.
            int_to_word (dict): A dictionary mapping each integer index back to the corresponding character in the vocabulary.
        """
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.string_to_idx = {ch: idx for idx, ch in enumerate(vocab)}
        self.idx_to_string = {idx: ch for idx, ch in enumerate(vocab)}

    def encode(self, word: str):
        # the encoder will create a look up table for every character in the vocabulary and map it to a given integer index
        return [self.string_to_idx[idx] for idx in word]

    def decode(self, tokens: list[int] | torch.Tensor):
        # the decoder will take a list of integers list[int] and for every integer will need to map to the correct character
        # supports vectorization
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.detach().cpu().numpy().tolist()
            # check if the tensor is a list of lists of integers (this is to support batch sizes within vectorization)
            if isinstance(tokens[-1], list):
                sub_tokens = ["".join(self.idx_to_string[idx]
                                      for idx in token) for token in tokens]
                return sub_tokens
            return "".join(self.idx_to_string[idx] for idx in tokens)
        else:
            return "".join(self.idx_to_string[idx] for idx in tokens)


vocab, vocab_size = get_vocab(data)
if DEBUG_MODE == "1":
    print("".join(vocab))
    print(vocab_size)


tokenizer = Tokenizer(vocab)
if DEBUG_MODE == "1":
    print(tokenizer.encode("gicu"))
    print(tokenizer.decode(tokenizer.encode("gicu")))


# We need to tokenize the input data and transform it into a PyTorch tensor.
# Tensors in PyTorch are represented as high-dimensional arrays containing real, float, or integer numbers.
# These tensors are contiguous blocks of data that allow for efficient vectorized calculations.
input_tensor = torch.tensor(tokenizer.encode(
    data), dtype=torch.long, device=DEVICE)

if DEBUG_MODE == "1":
    print(input_tensor.shape)


# split the data 85-15 in training and test, respectively
N = int(round(0.85*len(input_tensor)))
EMBEDDING_SIZE = 32  # batch size in Andrew's video
# this is the actual embedding size within the embedding layer
batch_size = EMBEDDING_SIZE
context_length = 8  # also called block size


training_data = input_tensor[:N]
test_data = input_tensor[N:]

# create DataLoader objects
train_set = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_set = DataLoader(test_data, batch_size=batch_size, shuffle=True)


# now, we need to create a set of X,Y pairs
# where X will be a sequence of tokens (i.e., encoded characters from the initial data) of size "context length", meaning that for `context_length=8` we require 8 integer indices, and each index will represent the consecutive characters found in the original dataset.
# Y will be the target sequence, meaning that Y will represent the tokens that will follow right after each of the tokens from X
# Given a context sequence x1, x2, x3 from X, Y will represent the subsequent sequence x4, x5, ...
# This means that for a provided context c = x1, x2, x3, the target sequence Y follows immediately after the context.
def batch(batch_size: int, context_length: int, data_type: str = "train") -> Tuple[torch.Tensor, torch.Tensor]:
    """
    - Create a set of (X, Y) pairs for training or testing.
    - The targets `Y` represent the sequences of characters that should be followed given an input sequence X.

    - Keep in mind that input sequence will be a list of tokens (a tensor of tokens) with a fixed `context_length` that is randomly picked from the input data. There will be `batch_size` such sequences making up the entire batch of X-token-sequences
    - The Y-token sequences will be essentially created by taking the current offset position and shift by 1-token (character)

    Args:
        `batch_size (int)`: The number of samples in the batch.
        `context_length (int)`: The length of the context sequence.
        `data_type (str)`: The type of data to use ('train' or 'test').

    Returns:
        `Tuple[torch.Tensor, torch.Tensor]`: A tuple containing the input (X) and target (Y) tensors.
    """
    data = training_data if data_type == "train" else test_data

    offsets = torch.randint(len(data) - context_length, (batch_size,))

    x = torch.stack([data[offset:offset + context_length]
                    for offset in offsets])
    y = torch.stack([data[offset + 1:offset + context_length + 1]
                    for offset in offsets])

    return x, y


# using a direct copycat of Andrew's method to evaluate the loss function over a data set
# source: https://github.com/karpathy/ng-video-lecture/blob/52201428ed7b46804849dea0b3ccf0de9df1a5c3/bigram.py#L47
def estimate_loss(batch_size: int, context_length: int, num_epochs: int):
    out = {}
    model.eval()  # do not perform backpropagation
    with torch.no_grad():
        for data_type in ['train', 'test']:
            losses = torch.zeros(num_epochs)

            for k in range(num_epochs):
                # requires external method TODO: self-contained
                X, Y = batch(batch_size, context_length, data_type)

                _, loss = model(X, Y)  # only the loss value is needed
                losses[k] = loss.item()
            out[data_type] = losses.mean()
    model.train()  # set the model back into training mode
    return out


def train(model: torch.nn.Module, epochs: int, max_num_tokens: int, batch_size: int, context_length: int):
    if DEBUG_MODE == "1":
        print(x)
        print(y)
        print(tokenizer.decode(x))
        print(tokenizer.decode(y))

    # load the bigram model onto the device
    # set an optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for idx in range(epochs):
        x, y = batch(batch_size, context_length, "train")

        optimizer.zero_grad()
        # this will be a tensor of batch size, in which every sample in the batch will have dimension `context_length`/`vocabulary_size`
        # torch.Size([4, 8, 65])
        # torch.Size([batch_size, context_length, vocab_size])
        output, loss = model(x, y)

        loss.backward()
        optimizer.step()

        # if idx % 100 == 0:
        # losses = estimate_loss(batch_size, context_length, epochs)
        # print(
        #     f"step {idx}: train loss {losses['train']:.4f}, val loss {losses['test']:.4f}")
    print(loss.item())
    print(tokenizer.decode(model.generate(torch.zeros(
        (1, 1), dtype=torch.long, device=DEVICE), max_num_tokens)[0].tolist()))


model = llm.BigramLanguageModel(
    vocab_size, batch_size, context_length).to(DEVICE)
train(model, 5000, 300, batch_size, context_length)
