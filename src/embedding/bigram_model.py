import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
import random

import localdata as data


def generate_text(model, start_word, num_words, word_to_idx):
    current_word = start_word
    generated_text = [current_word]
    for _ in range(num_words):
        input_var = torch.tensor([word_to_idx[current_word]], dtype=torch.long)
        output = model(input_var)
        _, predicted_idx = torch.max(output, 1)
        current_word = idx_to_word[predicted_idx.item()]
        generated_text.append(current_word)
    return ' '.join(generated_text)


def prepare_data(raw_text: str):
    words = raw_text.split()

    vocab = list(set(words))
    vocab_size = len(vocab)

    word_to_idx = {word: i for i, word in enumerate(vocab)}
    idx_to_word = {i: word for i, word in enumerate(vocab)}

    bigrams = [(words[i], words[i+1]) for i in range(len(words)-1)]
    bigram_indices = [(word_to_idx[w1], word_to_idx[w2]) for w1, w2 in bigrams]

    return vocab_size, vocab, word_to_idx, idx_to_word, bigrams, bigram_indices


# Define the model
class BigramModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(BigramModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        out = self.linear(embeds)
        return out


if __name__ == "__main__":
    vocab_size, vocab, word_to_idx, idx_to_word, bigrams, bigram_indices = prepare_data(
        data.train_data)

    # Hyperparameters
    embedding_dim = 75

    print(f'Using an embedding size: {embedding_dim}')
    print(f'Vocabulary size: {vocab_size}')

    model = BigramModel(vocab_size, embedding_dim)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    epochs = 200
    for epoch in range(epochs):
        total_loss = 0
        for context, target in bigram_indices:
            # Forward pass
            model.zero_grad()
            output = model(torch.tensor([context], dtype=torch.long))

            # Compute loss, gradients, and update parameters
            loss = loss_function(output, torch.tensor(
                [target], dtype=torch.long))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(
                f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss/len(bigram_indices)}')

    # Example of text generation
    start_word = 'marginal'
    print(generate_text(model, start_word, 10, word_to_idx))
