# encoding=utf-8
# Project: learn-pytorch
# Author: xingjunjie    github: @gavinxing
# Create Time: 29/07/2017 11:58 AM on PyCharm
# Basic template from http://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html

import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.nn.functional as F

class CBOW(nn.Module):

    def __init__(self, context_size=2, embedding_size=100, vocab_size=None):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_size)
        self.linear1 = nn.Linear(embedding_size, vocab_size)

    def forward(self, inputs):
        lookup_embeds = self.embeddings(inputs)
        embeds = lookup_embeds.sum(dim=0)
        out = self.linear1(embeds)
        out = F.log_softmax(out)
        return out



# create your model and train.  here are some functions to help you make
# the data ready for use by your module


def make_context_vector(context, word_to_ix):
    idxs = [word_to_ix[w] for w in context]
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)


# print(make_context_vector(data[0][0], word_to_ix))  # example

if __name__ == '__main__':
    CONTEXT_SIZE = 2  # 2 words to the left, 2 to the right
    EMBEDDING_SIZE = 10
    raw_text = """We are about to study the idea of a computational process.
    Computational processes are abstract beings that inhabit computers.
    As they evolve, processes manipulate other abstract things called data.
    The evolution of a process is directed by a pattern of rules
    called a program. People create programs to direct processes. In effect,
    we conjure the spirits of the computer with our spells.""".split()

    # By deriving a set from `raw_text`, we deduplicate the array
    vocab = set(raw_text)
    vocab_size = len(vocab)

    word_to_ix = {word: i for i, word in enumerate(vocab)}
    data = []
    for i in range(2, len(raw_text) - 2):
        context = [raw_text[i - 2], raw_text[i - 1],
                   raw_text[i + 1], raw_text[i + 2]]
        target = raw_text[i]
        data.append((context, target))

    loss_func = nn.CrossEntropyLoss()
    net = CBOW(CONTEXT_SIZE, embedding_size=EMBEDDING_SIZE, vocab_size=vocab_size)
    optimizer = optim.SGD(net.parameters(), lr=0.01)

    for epoch in range(100):
        total_loss = 0
        for context, target in data:
            context_var = make_context_vector(context, word_to_ix)
            net.zero_grad()
            log_probs = net(context_var)

            loss = loss_func(log_probs, autograd.Variable(
                torch.LongTensor([word_to_ix[target]])
            ))

            loss.backward()
            optimizer.step()

            total_loss += loss.data
        print(total_loss)