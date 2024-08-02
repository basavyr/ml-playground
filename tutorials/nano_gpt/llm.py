import torch
import os

DEBUG_MODE = os.getenv('DEBUG', '0')  # Default to '0' if DEBUG is not set


class BigramLanguageModel(torch.nn.Module):
    def __init__(self, vocab_size: int, n_embed: int, block_size: int):
        super(BigramLanguageModel, self).__init__()
        self.vocab_size = vocab_size

        # this embedding is made based on the identity of the tokens
        self.token_embeddings = torch.nn.Embedding(vocab_size, n_embed)

        self.position_embeddings = torch.nn.Embedding(
            block_size, n_embed)  # embedding based on the position of the table

        self.fake_logits = torch.nn.Linear(n_embed, vocab_size)

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor = None):
        token_embeddings = self.token_embeddings(tokens)
        logits = self.fake_logits(token_embeddings)
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
            logits = logits.view(-1, logits.shape[2])
            if DEBUG_MODE == "2":
                print(f'\nAfter reshape')
                print(f'Targets shape: {targets.shape}')
                print(f'Logits shape: {logits.shape}')
            loss = torch.nn.functional.cross_entropy(logits, targets)
        return logits, loss

    # create a method for generating tokens based on the model
    def generate(self, tokens: torch.Tensor, max_new_tokens: int):
        # tokens is of shape (batch_size, context_length) tensor in the current context
        for _ in range(max_new_tokens):
            logits, loss = self(tokens)

            # from (batch_size, context_length) -> (batch_size, vocab_size)
            logits = logits[:, -1, :]

            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            next_tokens = torch.multinomial(
                probabilities, num_samples=1)  # (batch_size,1)

            # (batch_size,tokens +1)
            tokens = torch.cat((tokens, next_tokens), dim=1)
        return tokens


if __name__ == "__main__":
    pass
