# Source: https://pytorch.org/tutorials/intermediate/ensembling.html

from torch import vmap
from torch.func import functional_call
import copy
from torch.func import stack_module_state

import torch
import torch.nn as nn
import torch.nn.functional as F


torch.manual_seed(0)


class Model1(nn.Module):
    """
    Straightforward model, taking as input a tensor of shape `(mx1)`
    The input tensor will be flattened, to the dimension `n_channels*height*width`

    Params:
    - requires `n_x` -> the size of the input tensor of shape `(n_x,1)`
    """

    def __init__(self, n_x: int):
        super().__init__()
        self.fc1 = nn.Linear(n_x, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x: torch.tensor):
        # flattened tensor
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        logits = self.fc4(x)
        return logits


class Model2(nn.Module):
    """
    Straightforward model, similar to `Model1`, taking as input a tensor of shape `(mx1)`
    The input tensor will be flattened, to the dimension `n_channels*height*width`

    Params:
    - requires `n_x` -> the size of the input tensor of shape `(n_x,1)`
    """

    def __init__(self, n_x: int):
        super().__init__()
        self.fc1 = nn.Linear(n_x, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x: torch.tensor):
        # flattened tensor
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        logits = self.fc3(x)
        return logits


epochs = 100
batch_size = 64
n_channels = 1
height = 28
width = 28
n_x = n_channels*height*width

device = torch.device("mps")

num_models = 10

data = torch.randn(100, 64, 1, 28, 28, device=device)
targets = torch.randint(10, (6400,), device=device)

models = [Model1(n_x).to(device) if i % 2 == 0 else Model2(n_x).to(device)
          for i in range(num_models)]  # the models have different class

models = [Model1(n_x).to(device) for _ in range(num_models)]


minibatches = data[:num_models]
predictions_diff_minibatch_loop = [
    model(minibatch) for model, minibatch in zip(models, minibatches)]

params, buffers = stack_module_state(models)


# Construct a "stateless" version of one of the models. It is "stateless" in
# the sense that the parameters are meta Tensors and do not have storage.
base_model = copy.deepcopy(models[0])
base_model = base_model.to('meta')


def fmodel(params, buffers, x):
    return functional_call(base_model, (params, buffers), (x,))


# show the leading 'num_models' dimension
print([p.size(0) for p in params.values()])

# verify minibatch has leading dimension of size 'num_models'
assert minibatches.shape == (num_models, 64, 1, 28, 28)


predictions1_vmap = vmap(fmodel)(params, buffers, minibatches)

# verify the ``vmap`` predictions match the
assert torch.allclose(predictions1_vmap, torch.stack(
    predictions_diff_minibatch_loop), atol=1e-3, rtol=1e-5)
