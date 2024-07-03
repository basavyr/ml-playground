import torch
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

x = torch.arange(-5, 5, 0.1).view(-1, 1)
y = 20 * x + 0.1 * torch.randn(x.size())

model = torch.nn.Linear(1, 1)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)


def train_model(model, iter):
    model.train()
    for epoch in range(iter):
        y1 = model(x)
        loss = criterion(y1, y)
        writer.add_scalar("Loss/train", loss, epoch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test_model(model, iter):
    x = torch.arange(-10, 10, 0.1).view(-1, 1)
    y = 20 * x + 0.1 * torch.randn(x.size())

    model.eval()

    for epoch in range(iter):
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        with torch.no_grad():
            y_pred = model(x)
            loss = criterion(y_pred, y)
            writer.add_scalar("Loss/train", loss, epoch)


train_model(model, 100)
test_model(model, 100)
writer.flush()

writer.close()
