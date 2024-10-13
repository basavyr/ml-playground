
from typing import Mapping, Dict, Any
import torch
import torch.utils.data
import torch.nn as nn

import time
import os
import sys
import json

import model as models
from data import MNIST
from config import Conv_Configs


DEFAULT_DEVICE: torch.device = torch.device("mps") if os.uname(
).sysname == "Darwin" else torch.device("cpu")


def train(training_configs: Conv_Configs):
    device = training_configs.device
    loss_fn = training_configs.loss_fn
    optimizer = training_configs.optimizer
    data = training_configs.train_loader
    epochs = training_configs.epochs
    model = training_configs.model

    report = {}
    losses = []
    best_loss = (0, torch.inf)
    accuracies = []
    best_accuracy = (0, 0)

    train_start = time.time()
    model.to(device)
    model.train()

    from tqdm import trange

    for epoch in (t := trange(0, epochs)):
        start = time.time()
        epoch_loss = 0
        predictions = 0
        for x, y_true in data:
            x, y_true = x.to(device), y_true.to(device)
            optimizer.zero_grad()

            y = model(x)
            loss = loss_fn(y, y_true)
            epoch_loss += loss.item()
            predictions += (y.argmax(dim=1) == y_true).sum().item()

            loss.backward()
            optimizer.step()

        epoch_loss /= len(data)
        losses.append(epoch_loss)
        accuracy = predictions/len(data.dataset)*100
        accuracies.append(accuracy)

        if epoch_loss < best_loss[1]:
            best_loss = (epoch, epoch_loss)
        if accuracy > best_accuracy[1]:
            best_accuracy = (epoch, accuracy)

        report[f'epoch-{epoch+1}'] = {
            "loss": epoch_loss,
            "acc": accuracy,
            "epoch_time": time.time()-start,
            "best_loss": best_loss[1],
            "best_acc": best_accuracy[1],
        }
        t.set_description(
            f'Epoch: {epoch+1} -> Loss: {epoch_loss:.3f} | Acc: {accuracy:.3f} %')

    train_time = time.time()-train_start
    # print(f'Train time: {train_time:.3f} s')
    # print(f'Accuracy: {best_accuracy}')
    # print(f'Loss: {best_loss}')

    report["loss"] = best_loss[1]
    report["acc"] = best_accuracy[1]
    report["loss_fn"] = f'{type(loss_fn).__name__}'
    report["optimizer"] = f'{type(optimizer).__name__}'
    report["train_time"] = train_time

    return report


def eval(configs: Conv_Configs):
    model = configs.model
    device = configs.device
    data = configs.eval_loader
    loss_fn = configs.loss_fn

    report = {}
    start = time.time()
    model.to(device)
    model.eval()
    running_loss = 0
    predictions = 0
    with torch.no_grad():
        for x, y_true in data:
            x, y_true = x.to(device), y_true.to(device)

            y = model(x)
            loss = loss_fn(y, y_true)
            running_loss += loss.item()

            predictions += (torch.argmax(y, dim=1) == y_true).sum().item()

    return {
        "loss": running_loss/len(data),
        "acc": float(predictions/len(data.dataset)*100),
        "eval_time": time.time()-start,
    }


if __name__ == "__main__":

    model_configs = Conv_Configs(MNIST().mnist, DEFAULT_DEVICE, epochs=10)

    report = {}
    report["training"] = train(model_configs)
    report["eval"] = eval(model_configs)
    report["hash"] = model_configs.hash

    with open("training_report.json", 'w') as dumper:
        json.dump(report, dumper)
