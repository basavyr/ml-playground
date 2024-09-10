import torch
import torch.utils
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from tqdm import trange
from typing import Callable
import time

import model as m
import data


def train(model: nn.Module, loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], dataloader: DataLoader, num_epochs: int = 1):
    """
    Trains a given model using the provided loss function and dataloader.

    Args:
        model (nn.Module): The neural network model to train.
        loss_fn (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
            The loss function used for training, e.g., nn.CrossEntropyLoss().
        dataloader (DataLoader): The dataloader providing training batches.

    Saves:
        - The trained model as "{model.model_name}.pth".
        - The optimizer state as "{model.model_name}-optimizer_state.pth".

    Note:
        Ensure that the model has an attribute `model_name` defined for proper saving.
    """
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

    model.train()

    start = time.time()
    epoch_losses = []
    for _ in (t := trange(num_epochs)):
        running_loss = 0
        for _, batch in enumerate(dataloader):
            x, y_true = batch
            y = model(x)

            loss = loss_fn(y, y_true)
            running_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

        epoch_loss = running_loss/len(dataloader)
        epoch_losses.append(epoch_loss)
        t.set_description(f"Loss: {epoch_loss}")

    train_time = time.time()-start
    print(f'Training model finished: {train_time:.3f} s')

    torch.save(model, f"{model.model_name}.pth")
    torch.save(optimizer.state_dict(),
               f"{model.model_name}-optimizer.pth")


def eval(model_path: str, loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], eval_dataloader: DataLoader):
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
    n_samples = 500000
    num_epochs = 5

    train_loader = data.generate_train_data(n_samples, batch_size)
    test_loader = data.generate_test_data(2500)

    config = m.AdnetConfig(
        input_size=2,
        hidden_size1=512,
        hidden_size2=1024,
        output_size=1,
    )

    model = m.Adnet(config)

    loss_fn = nn.MSELoss()

    train(model, loss_fn, train_loader, num_epochs)

    eval(f"{model.model_name}.pth", loss_fn, test_loader)

    # save locally
    model.save_pretrained(model.model_name)

    # push to the hub
    model.push_to_hub(model.repo_id)
