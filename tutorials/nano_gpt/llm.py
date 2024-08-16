import torch
import os

DEBUG_MODE = os.getenv('DEBUG', '0')  # Default to '0' if DEBUG is not set


class BigramLanguageModel(torch.nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, block_size: int):
        super(BigramLanguageModel, self).__init__()
        self.vocab_size = vocab_size

        # this embedding is made based on the identity of the tokens
        self.token_embeddings = torch.nn.Embedding(vocab_size, embedding_dim)

        self.position_embeddings = torch.nn.Embedding(
            block_size, embedding_dim)  # embedding based on the position of the table

        self.lm_head = torch.nn.Linear(embedding_dim, vocab_size)

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor = None):
        token_embeddings = self.token_embeddings(tokens)  # (B,T,C)
        logits = self.lm_head(token_embeddings)  # (B,T,vocab_size)

        # the (B,T,C) sizes from Andrew's video
        # T refers to the length of time, i.e., the size of the context
        batches, time, channels = logits.shape
        B, T, C = batches, time, channels

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
            logits, loss = self(tokens)

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
