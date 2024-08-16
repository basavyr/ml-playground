

import torch

torch.manual_seed(1337)


# based on Andrew's video with the mathematical trick on self-attention
# therein he uses B,T,C letters to express the batches, the time component (i..e, hidden states or context size), and the channels (vocab size)
batches, times, channels = 4, 8, 2
B, T, C = batches, times, channels
# the time component represent the number of context tokens
# ( 4,8,2 ) -> will mean that there is a group of 4 batches with 8 tokens each, with only two channels

x = torch.randn(B, T, C)


def average_previous_tokens(input: torch.Tensor):
    b, t, c = input.shape
    xbow = torch.zeros((b, t, c))

    for bb in range(b):
        for tt in range(t):
            x_prev = input[bb, :tt+1]
            xbow[bb, tt] = torch.mean(x_prev, 0)

    return xbow


def create_softmax_matrix(t: torch.tensor):
    """
    multiply a matrix `t` with the weight matrix `t_l`, which represents a weighted sum, such that every token in a time series T will be "connected" with only the previous ones, up to itself
    """
    T = t.shape[1]
    t_l = torch.tril(torch.ones((T, T)))
    t_l = t_l.masked_fill(t_l == 0, float('-inf'))
    t_l = torch.nn.functional.softmax(t_l, dim=-1)

    return t_l @ t


print(x[0])
print(average_previous_tokens(x)[0])
