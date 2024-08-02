

import torch

torch.manual_seed(1337)


# based on Andrew's video with the mathematical trick on self-attention
# therein he uses B,T,C letters to express the batches, the time component (i..e, hidden states), and the channels
batches, times, channels = 4,  8, 2
# the time component represent the number of tokens
# ( 4,8 ) -> will mean that there is a group of 4 batches with 8 tokens each

x = torch.randn(batches, times, channels)
print(x)


def create_softmax_matrix(t: torch.tensor):
    """
    multiply a matrix `t` with the weight matrix `t_l`, which represents a weighted sum, such that every token in a time series T will be "connected" with only the previous ones, up to itself
    """
    T = t.shape[1]
    t_l = torch.tril(torch.ones((T, T)))
    t_l = t_l.masked_fill(t_l == 0, float('-inf'))
    t_l = torch.nn.functional.softmax(t_l, dim=-1)

    return t_l @ t


sft = create_softmax_matrix(x)
print(sft)
