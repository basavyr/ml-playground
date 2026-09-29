import torch
import torch.nn as nn

device = torch.device("mps")


t1 = torch.randn(128, 512)
t2 = torch.randn(128, 512)


print(t1)