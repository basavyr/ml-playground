import torch
x = torch.backends.mps.is_available()
print('NPU available:')
print(x)
