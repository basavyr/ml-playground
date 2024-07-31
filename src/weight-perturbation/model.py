
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


torch.manual_seed(1)


class MYF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return 5 * input

    @staticmethod
    def backward(ctx, grad_output):
        # this method will evaluate the gradient of the output loss function with respect to the input
        # if the input is x and the output model provides a function y=f(x)=5*x, then the gradient will need to return df/dx=5
        # typically, the gradient of the loss function w.r.t. the output is known, and then one can evaluate the gradient of the loss function w.r.t. the input x by applying the chain rule, i.e., dL/dx = dL/dy * dy/dx
        input, = ctx.saved_tensors
        return grad_output * 5


class MyNet(nn.Module):
    def __init__(self, dim: int):
        super(MyNet, self).__init__()
        self.linear1 = nn.Linear(dim, 64)
        self.linear2 = nn.Linear(64, 120)
        self.linear3 = nn.Linear(120, dim)

    def forward(self, x: torch.Tensor):
        fc1 = self.linear1(x)
        fc2 = self.linear2(F.relu(fc1))
        logits = self.linear3(F.relu(fc2))

        return logits


def train(model, num_epochs, x, y_true):
    model.train()

    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.MSELoss()

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        y_pred = model(x)
        loss = loss_fn(y_pred, y_true)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    torch.save(model, "model.pth")


def evaluate(model, n_samples, x_size):

    model = torch.load(f'model.pth')
    loss_fn = nn.MSELoss()

    model.eval()

    with torch.no_grad():
        x = torch.randint(0, 10, (n_samples, x_size), dtype=torch.float32)
        y_true = MYF.apply(x)

        y_pred = model(x)

        loss = loss_fn(y_pred, y_true)
    print(f'Loss: {loss.item()}')


if __name__ == "__main__":

    n_samples = 10
    x_size = 2
    X_train = torch.randint(0, 100, (n_samples, x_size), dtype=torch.float32)
    Y_train = 5 * X_train

    n_eval_samples = 1
    X_eval = torch.randint(
        0, 100, (n_eval_samples, x_size), dtype=torch.float32)
    Y_eval = 5 * X_eval

    model = MyNet(x_size)

    losses = []

    # for beta in range(10, 500, 10):
    #     x = torch.randint(0, 1000, (n_samples, 3), dtype=torch.float32)
    #     y_true = MYF.apply(x)
    #     loss = train(model, beta, x, y_true)
    #     losses.append({"beta": beta, "loss": loss})
    #     print({"beta": beta, "loss": loss})

    import os
    if not os.path.exists('model.pth'):
        loss = train(model, 5000, X_train, Y_train)
    evaluate(model, n_eval_samples, x_size)
