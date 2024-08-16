import torch
import os

DEBUG_MODE = os.getenv('DEBUG', '0')  # Default to '0' if DEBUG is not set


class Head(torch.nn.Module):
    """
    - implementation for self-attention on a single head
    """

    def __init__(self, embedding_dim: int, head_size: int, block_size: int):
        super(Head, self).__init__()
        self.key = torch.nn.Linear(embedding_dim, head_size, bias=False)
        self.query = torch.nn.Linear(embedding_dim, head_size, bias=False)
        self.value = torch.nn.Linear(embedding_dim, head_size, bias=False)
        self.register_buffer("tril", torch.tril(
            torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        w = q @ k.transpose(-2, -1) * C ** -0.5
        w = w.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        w = torch.nn.functional.softmax(w, dim=-1)

        return w @ v


class BigramLanguageModel(torch.nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, block_size: int):
        super(BigramLanguageModel, self).__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.self_attn = Head(embedding_dim, embedding_dim, block_size)

        # embedding based on the identity of the tokens
        self.token_embeddings = torch.nn.Embedding(vocab_size, embedding_dim)

        # embedding based on the position of the tokens
        self.position_embeddings = torch.nn.Embedding(
            block_size, embedding_dim)

        self.lm_head = torch.nn.Linear(embedding_dim, vocab_size)

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor = None):

        # the (B,T,C) sizes from Andrew's video
        # T refers to the length of time, i.e., the size of the context
        B, T = tokens.shape

        # size: (B,T,C)
        token_embeddings = self.token_embeddings(tokens)

        # size: (T, C)
        position_embeddings = self.position_embeddings(torch.arange(T))

        # size: (B,T,C)
        x = token_embeddings + position_embeddings
        x = self.self_attn(x)

        # size: (B,T,vocab_size)
        logits = self.lm_head(x)

        _, _, channels = logits.shape
        C = channels

        if targets is None:
            loss = None
        else:
            if DEBUG_MODE == "2":
                print(f'Before reshape')
                print(f'Targets: {targets}')
                # logits[0], that is a single tensor from all logits will have a shape defined by ('vocab_size`,`context_length`)
                # logits[0]: torch.Size([8, 65])
                print(f'Targets shape: {targets.shape}')
                print(f'Logits shape: {logits.shape}')
            # reshape the targets to have the same structure as the logits
            targets = targets.view(-1)

            # logits = logits.view(-1, logits.shape[2])
            logits = logits.view(B*T, C)

            if DEBUG_MODE == "2":
                print(f'\nAfter reshape')
                print(f'Targets shape: {targets.shape}')
                print(f'Logits shape: {logits.shape}')

            # cross entropy loss expects that the number of channels should be the second dimension
            # source: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss
            # input has to be a Tensor of size (C) for unbatched input, (minibatch,C) for batched input
            loss = torch.nn.functional.cross_entropy(logits, targets)
        return logits, loss

    # create a method for generating tokens based on the model
    def generate(self, tokens: torch.Tensor, max_new_tokens: int):
        # tokens is of shape (batch_size, context_length) tensor in the current context
        # in Andrew's video, idx is (B,T)
        for _ in range(max_new_tokens):
            # crop the tokens to the last block_size
            tokens_cond = tokens[:, -self.block_size:]
            logits, loss = self(tokens_cond)

            # from (batch_size, context_length) -> (batch_size, vocab_size)
            # from (B,T,C) to (B, C)
            logits = logits[:, -1, :]

            # size is (B,C)
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            next_tokens = torch.multinomial(
                probabilities, num_samples=1)  # (B,T+1)

            # (batch_size,tokens + 1)
            tokens = torch.cat((tokens, next_tokens), dim=1)
        # (batch_size,tokens + max_new_tokens)
        return tokens


if __name__ == "__main__":
    pass
