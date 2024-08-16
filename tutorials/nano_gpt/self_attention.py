import torch


def average_previous_tokens(input: torch.Tensor):
    B, T, C = input.shape
    xbow = torch.zeros((B, T, C))

    for b in range(B):
        for t in range(T):
            x_prev = input[b, :t+1]
            xbow[b, t] = torch.mean(x_prev, 0)

    return xbow


def self_attention_1head(x: torch.Tensor, head_size: int):
    """
    SELF ATTENTION
    a token will "emit" a QUERY Q and a KEY K (two vectors Q,K)
    Q -> as a token, what am I looking for
    K -> as a token, what do I contain
    for one token, a dot product between the token query and all the keys of the other tokens
    implementation of self-attention for a single head

    More notes on self attention (source: https://colab.research.google.com/drive/1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-?usp=sharing#scrollTo=M5CvobiQ0pLr)

    Attention is a communication mechanism. Can be seen as nodes in a directed graph looking at each other and aggregating information with a weighted sum from all nodes that point to them, with data-dependent weights.
    There is no notion of space. Attention simply acts over a set of vectors. This is why we need to positionally encode tokens.
    Each example across batch dimension is of course processed completely independently and never "talk" to each other
    In an "encoder" attention block just delete the single line that does masking with tril, allowing all tokens to communicate. This block here is called a "decoder" attention block because it has triangular masking, and is usually used in auto-regressive settings, like language modeling.
    "self-attention" just means that the keys and values are produced from the same source as queries. In "cross-attention", the queries still get produced from x, but the keys and values come from some other, external source (e.g. an encoder module)
    "Scaled" attention additional divides wei by 1/sqrt(head_size). This makes it so when input Q,K are unit variance, wei will be unit variance too and Softmax will stay diffuse and not saturate too much. Illustration below
    """

    # when we do the aggregation, we do not aggregated the tokens themselves, but instead we aggregated over < value >
    key = torch.nn.Linear(C, head_size, bias=False)  # what do I look for
    query = torch.nn.Linear(C, head_size, bias=False)  # what do tokens contain
    value = torch.nn.Linear(C, head_size, bias=False)

    k = key(x)  # (B,T,16)
    q = query(x)  # (B,T,16)

    # all queries must be dot product with all the keys
    w = q @ k.transpose(-2, -1)  # (B,T,16) @ (B,16,T) -> (B,T,T)
    # scaled self attention must contain the 1/sqrt(head_size) to control the variance at init
    w = w * head_size ** -0.5
    w = w.masked_fill(torch.tril(torch.ones((T, T))) == 0, float('-inf'))
    w = torch.nn.functional.softmax(w, dim=-1)

    # this is what a token will "communicate" after performing the dot product between its query and keys
    v = value(x)  # evaluate the value

    attn = w @ v
    return w, v, attn


if __name__ == "__main__":
    torch.manual_seed(1337)
    # based on Andrew's video with the mathematical trick on self-attention
    # source: https://www.youtube.com/watch?v=kCc8FmEb1nY&t=5722s
    # he uses B,T,C letters to express the batches, the time component (i..e, hidden states or context size), and the channels (i.e., vocab size)
    # ( B, T, C ) = ( 4, 8, 2 ) means that there is a group of 4 batches with 8 tokens each, with only two channels

    batches, times, channels = 4, 8, 32
    B, T, C = batches, times, channels

    x = torch.randn(B, T, C)
    head_size = 16
    w, v, self_attn = self_attention_1head(x, head_size)
    print(self_attn.shape)
    print(w)
