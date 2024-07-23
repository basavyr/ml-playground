import torch
import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F

import localdata as data


def pprint(obj, debug=False):
    if debug == True:
        print(obj)
    else:
        pass


class NGramLanguageModeler(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        logits = self.linear2(out)
        return logits


def generate_ngram(text_sequence: list[str], context_size: int) -> int:
    """
    - this will generate an NGRAM-like object
    - using the tutorial given from pytorch official tutorials (see file beginner_source/nlp/word_embeddings_tutorial.py)
    """
    ngram = []
    for i in range(context_size, len(text_sequence)):
        context_list = [text_sequence[i-j-1] for j in range(context_size)]
        target_word = text_sequence[i]
        context_tuple = (context_list, target_word)
        ngram.append(context_tuple)

    return ngram


def prepare_data(raw_text: str) -> tuple:
    # Prepare the data
    words = raw_text.split()
    vocab = list(set(words))
    vocab_size = len(vocab)
    word_to_idx = {word: i for i, word in enumerate(vocab)}
    idx_to_word = {i: word for i, word in enumerate(vocab)}

    return words, vocab_size, vocab, word_to_idx, idx_to_word


def train(train_data: str, num_epochs: int, embedding_dim: int, context_size: int):

    DEBUG_FLAG = False

    words, vocab_size, vocab, word_to_idx, idx_to_word = prepare_data(
        train_data)
    ngrams = generate_ngram(words, context_size)

    losses = []
    loss_function = nn.CrossEntropyLoss()
    model = NGramLanguageModeler(vocab_size, embedding_dim, context_size)
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        total_loss = 0
        for context, target in ngrams:
            context_idx = torch.tensor(
                [word_to_idx[w]for w in context], dtype=torch.long)

            pprint(f'context {context}')
            pprint(f'context idx: {context_idx}')

            y_true = torch.tensor([word_to_idx[target]], dtype=torch.long)

            # Forward pass
            model.zero_grad()
            # the model needs to have as input the id of each word from the context
            output = model(context_idx)

            # Compute loss, gradients, and update parameters
            loss = loss_function(output, y_true)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        losses.append(total_loss)
        if (epoch + 1) % 10 == 0:
            print(
                f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss/len(ngrams)}')

    torch.save(model, 'model.pth')


if __name__ == "__main__":

    train_data = data.train_data

    # Hyperparameters
    embedding_dim = 75
    context_size = 10

    # Train the model
    epochs = 1
    train(train_data, epochs, embedding_dim, context_size)
