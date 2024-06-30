import matplotlib.pyplot as plt

import torch

from operator import itemgetter, attrgetter
import numpy as np


def prepare_tensors(x: torch.tensor, y: torch.tensor, y_pred: torch.tensor, use_argsort: bool = False) -> tuple:
    random_batch_index = torch.randint(x.shape[0], (1,)).item()
    x = x[random_batch_index]
    y = y[random_batch_index]
    y_pred = y_pred[random_batch_index]

    x_np = x.view(-1).to("cpu").numpy()
    y_np = y.view(-1).to("cpu").numpy()
    y_pred_np = y_pred.view(-1).to("cpu").numpy()

    if use_argsort:
        sorted_indices = x_np.argsort()  # give the indices
        x_sorted = x_np[sorted_indices]
        y_sorted = y_np[sorted_indices]
        y_pred_sorted = y_pred_np[sorted_indices]
    else:
        sorted_data = sorted([(x, y, y_pred)
                              for x, y, y_pred in zip(x_np, y_np, y_pred_np)], key=itemgetter(0))
        x_sorted, y_sorted, y_pred_sorted = zip(*sorted_data)
    return x_sorted, y_sorted, y_pred_sorted


def plot_tensors(x: torch.tensor, y: torch.tensor, y_pred: torch.tensor) -> None:
    """
    - plots the the predicted data in comparison with the real data for a given input tensor `x` and `y_pred` as MODEL(x)
    """
    plt.xlabel("X")
    plt.ylabel('y vs y_pred')
    x, y, y_pred = prepare_tensors(x, y, y_pred)
    plt.plot(x, y, 'ok', label="Actual")
    plt.plot(x, y_pred, '-r', label="Predicted")
    plt.title('y vs. y_pred')
    plt.legend()
    plt.savefig("plot.pdf", dpi=300, bbox_inches="tight")
    plt.close()
