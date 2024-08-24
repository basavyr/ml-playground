import dotenv
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

dotenv.load_dotenv()


def default_vocab():
    vocab = list(set(["What", "is", "Cybersecurity", "?"]))
    vocab = list(map(lambda s: s.lower(), vocab))

    test_string = ["hey", "what", "is", "gicu"]

    words_to_idx = {w: idx if w in vocab else "UNK" for idx,
                    w in enumerate(test_string)}
    tokens = list(
        map(lambda str: words_to_idx[str] if str in vocab else len(vocab), test_string))

    print(words_to_idx)
    print(tokens)
    vocab_size = len(vocab)

    return vocab, vocab_size, tokens


class Model(nn.Module):
    def __init__(self, vocab_size: int, embedding_size: int):
        super(Model, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_size)

    def forward(self, x: torch.Tensor):
        return self.embeddings(x)


def tokenizer(words: str):
    list_of_words = words.split(" ")
    vocab = {w: idx for idx, w in enumerate(list(set(list_of_words)))}
    vocab_size = len(vocab)
    tokens = [vocab[word] for word in list_of_words]
    return vocab, vocab_size, tokens


if __name__ == "__main__":
    default_vocab()
    # prompt = os.environ.get("PROMPT", "What is Cybersecurity ?").lower()
    # embedding_dim = 3

    # vocab, vocab_size, tokens = tokenizer(prompt)

    # net = Model(vocab_size, embedding_dim)

    # output = net(torch.tensor(tokens))

    # print(output)
