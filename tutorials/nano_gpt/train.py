# based on the tutorial from Andrew Karpathy
# source: https://www.youtube.com/watch?v=kCc8FmEb1nY

import os

DEBUG_MODE = os.getenv('DEBUG', '0')  # Default to '0' if DEBUG is not set


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
    unique_characters = sorted(list(set(input_data)))
    vocab_size = len(unique_characters)
    return unique_characters, vocab_size


vocab, vocab_size = get_vocab(data)

if DEBUG_MODE == "1":
    print("".join(vocab))
    print(vocab_size)


# first we need to tokenize the input strings
# the tokenization will result in assigning numbers/indices (so typically integer numbers) to the characters in the vocabulary.
# in this case we work at the character level, but methods applied at the "word" level (or even sub-word level) will be similar
###################
#    TOKENIZER    #
###################

# this tokenizer is composed of an encoder and decoder parts
# the encoder will take the character and assign a token -> encode STR_CHAR->INT_INDEX
# the decoder will take the index and look up on what character it corresponds to -> decode INT_INDEX -> STR_CHAR

string_to_idx = {ch: idx for idx, ch in enumerate(vocab)}
idx_to_string = {idx: ch for idx, ch in enumerate(vocab)}


def encode(x): return [string_to_idx[idx] for idx in x]


def decode(idx_list): return "".join(idx_to_string[idx] for idx in idx_list)


# the decoder will take a list of integers list[int] and for every integer will need to map to the correct character
print(encode("gicu"))
print(decode(encode("gicu")))
