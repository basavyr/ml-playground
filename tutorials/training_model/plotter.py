import matplotlib.pyplot as plt

import torch

from operator import itemgetter, attrgetter


def prepare_tensors(x: torch.tensor, y: torch.tensor, y_pred: torch.tensor) -> tuple:
    x = x[0]
    y = y[0]
    y_pred = y_pred[0]
    x_np = x.view(-1).to("cpu").numpy()
    y_np = y.view(-1).to("cpu").numpy()
    y_pred_np = y_pred.view(-1).to("cpu").numpy()

    sorted_data = sorted([(x, y, y_pred)
                          for x, y, y_pred in zip(x_np, y_np, y_pred_np)], key=itemgetter(0))

    A, B, C = zip(*sorted_data)
    return A, B, C


def plot_tensors(x: torch.tensor, y: torch.tensor, y_pred: torch.tensor) -> None:
    """
    - plots the the predicted data in comparison with the real data for a given input tensor `x` and `y_pred` as MODEL(x)
    """
    plt.xlabel("X")
    plt.ylabel('y vs y_pred')
    x, y, y_pred = prepare_tensors(x, y, y_pred)
    plt.plot(x, y, 'ok')
    plt.plot(x, y_pred, '-r')
    plt.savefig("plot.pdf", dpi=300, bbox_inches="tight")
    plt.close()
