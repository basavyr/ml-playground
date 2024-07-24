# this implementation is based on a tutorial from pytorch
# it uses pytorch to create a custom function that has support for forward and backward propagation

# source: beginner_source/examples_autograd/polynomial_custom_function.py
# original repo: https://github.com/basavyr/pytorch-tutorials

import torch
import math


class MyFunc(torch.autograd.Function):
    """
    A simple custom function that will just multiply by 5
    """
    @staticmethod
    def forward(ctx, input):
        # `input` type: : torch.Tensor
        ctx.save_for_backward(input)

        # y=f(x)
        f_x = input**2
        return f_x

    @staticmethod
    def backward(ctx, grad_output):
        # `grad_output` type: : torch.Tensor
        input, = ctx.saved_tensors

        # grad_output = dL/dy
        # dL/dx = dL/dy * dy/dx -> dL/dx = grad_output * dy/dx
        dy_dx = 2*input
        return grad_output*dy_dx


_type = torch.float
_device = "mps"


x = torch.linspace(-10, 10, 100, dtype=_type,
                   device=_device, requires_grad=True)


alpha = [0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 1]

for a in alpha:

    y_true = x*(a*x)

    loss_fn = torch.nn.MSELoss()

    FF = MyFunc.apply

    y_pred = FF(x)
    y_pred.retain_grad()

    print(y_pred.grad)

    loss = loss_fn(y_pred, y_true)

    loss.backward()
    print(loss.item())
