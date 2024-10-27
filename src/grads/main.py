
from typing import Mapping, Dict, Any
import torch
import torch.utils.data
import torch.nn as nn

import numpy as np

from tqdm import trange
import time
import os
import sys
import json

import model as models
from data import MNIST, CIFAR10, CIFAR100
from config import Conv_Configs


DEFAULT_DEVICE: torch.device = torch.device("mps") if os.uname(
).sysname == "Darwin" else torch.device("cpu")


def collect_grads(model: nn.Module, collapsed: bool = True) -> list[float]:
    grads = []
    for p in model.named_parameters():
        param_name, param_value = p
        param_grad = param_value.grad.detach().cpu().flatten().numpy()
        if param_grad is not None:
            grads.append((param_name, param_grad))
    if collapsed == True:
        from itertools import chain
        return list(chain.from_iterable([g[1] for g in grads]))
    return grads


def train(training_configs: Conv_Configs, with_grads: bool = False, save_state_dict: bool = False):
    """
    - trains a model based on a configuration

    Args:
    - `with_grads`: if set to `True` it will print grads during the training
    - `save_state_dict`: if set to `True` it will save the model locally
    """
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
            "epoch_grads": collect_grads(model),
        }
        t.set_description(
            f'Epoch: {epoch+1} -> Loss: {epoch_loss:.3f} | Acc: {accuracy:.3f} %')

    train_time = time.time()-train_start

    report["loss"] = best_loss
    report["acc"] = best_accuracy
    report["train_time"] = train_time

    if save_state_dict:
        torch.save(model, f'{model.model_name}-trained.pth')

    return report


def eval(configs: Conv_Configs):
    model = configs.model
    device = configs.device
    data = configs.eval_loader
    loss_fn = configs.loss_fn

    model.to(device)
    model.eval()
    running_loss = 0
    acc = 0
    start = time.time()
    with torch.no_grad():
        for x, y_true in data:
            x, y_true = x.to(device), y_true.to(device)

            y = model(x)

            loss = loss_fn(y, y_true)
            running_loss += loss.item()

            acc += (torch.argmax(y, dim=1) == y_true).sum().item()

    return {
        "loss": running_loss/len(data),
        "acc": float(acc/len(data.dataset)*100),
        "eval_time": time.time()-start,
    }


class DotDict(dict):
    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, attr, value):
        self[attr] = value

    def __delattr__(self, attr):
        del self[attr]


if __name__ == "__main__":

    data_configs = DotDict({
        "cifar10": DotDict({"data": CIFAR10().cifar10, "num_classes": 10, "in_channels": 3, "out_channels": 32, "height": 32, "width": 32}),
        "cifar100": DotDict({"data": CIFAR100().cifar100, "num_classes": 100, "in_channels": 3, "out_channels": 32, "height": 32, "width": 32}),
        "mnist": DotDict({"data": MNIST().mnist, "num_classes": 10, "in_channels": 1, "out_channels": 32, "height": 28, "width": 28}),
    })
    data_cfgs = data_configs.cifar10  

    model_configs = Conv_Configs(data_cfgs.data,
                                 device=DEFAULT_DEVICE,
                                 epochs=20,
                                 n_layers=5,
                                 hidden_units=256,
                                 num_classes=data_cfgs.num_classes,
                                 in_channels=data_cfgs.in_channels,
                                 out_channels=data_cfgs.out_channels,
                                 image_size=(data_cfgs.height,
                                             data_cfgs.width),
                                 maxpool=True,
                                 logits=True,
                                 use_adam=False)

    report = {}
    report["hash"] = model_configs.hash
    report["loss_fn"] = f'{type(model_configs.loss_fn).__name__}'
    report["optimizer"] = f'{type(model_configs.optimizer).__name__}'

    # --------- perform training ---------
    report["training"] = train(model_configs, save_state_dict=True)
    # --------- measure model performance ---------
    report["eval"] = eval(model_configs)

    with open("training_report.json", 'w') as dumper:
        json.dump(report, dumper)
