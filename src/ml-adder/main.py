import torch

import torch.utils
from torch.utils.data import Dataset, DataLoader, TensorDataset

import model as m

import torch.nn as nn
import torch.optim as optim

from tqdm import trange

import data


def train(model: nn.Module, loss_fn, dataloader: DataLoader):
    optimizer = optim.SGD(model.parameters())

    model.train()
    train_loss = []
    for idx, batch in enumerate(dataloader):
        x, y_true = batch
        y = model(x)

        loss = loss_fn(y, y_true)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        train_loss.append(loss.item())

    torch.save(model, "model.pth")


def eval(model_path: str, loss_fn, eval_dataloader: DataLoader):
    model = torch.load(model_path)

    model.eval()

    eval_losses = []

    total_predictions = []
    correct_predictions = []
    with torch.no_grad():
        for idx, batch in enumerate(eval_dataloader):
            x, y_true = batch

            y = model(x)

            loss = loss_fn(y, y_true)
            eval_losses.append(loss.item())

            prediction = torch.allclose(y, y_true, atol=0.1)
            total_predictions.append(prediction)
            if prediction is True:
                correct_predictions.append(prediction)

    print(
        f'Correct predictions: {len(correct_predictions)}/{len(total_predictions)}')
    print(
        f'Model accuracy: {len(correct_predictions)/len(total_predictions)*100:.3f} %')


if __name__ == "__main__":
    batch_size = 64
    n_samples = 150000

    train_loader = data.generate_train_data(n_samples, batch_size)
    test_loader = data.generate_test_data(5000)

    config = m.AdnetConfig(
        input_size=2,
        hidden_size1=512,
        hidden_size2=1024,
        output_size=1,
        license="mit",
        repo_url="https://huggingface.co/basavyr/adnet"
    )
    model = m.Adnet_HF(config)

    loss_fn = nn.MSELoss()

    train(model, loss_fn, train_loader)

    eval("model.pth", loss_fn, test_loader)

    # save locally
    model.save_pretrained("adnet")

    # push to the hub
    model.push_to_hub("basavyr/adnet")
