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
    vocab = sorted(list(set(input_data)))
    vocab_size = len(vocab)
    return vocab, vocab_size


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
        return [self.string_to_idx[idx] for idx in word]

    def decode(self, tokens: list[int]):
        return "".join(self.idx_to_string[idx] for idx in tokens)


tokenizer = Tokenizer(vocab)

# the decoder will take a list of integers list[int] and for every integer will need to map to the correct character
print(tokenizer.encode("gicu"))
print(tokenizer.decode(tokenizer.encode("gicu")))
