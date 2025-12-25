#!/usr/bin/env python3
import sys
import re
import numpy as np
import matplotlib.pyplot as plt


def detect_dataset(text):
    m = re.search(r"Dataset:\s*(cifar\d+)", text, re.IGNORECASE)
    if not m:
        raise ValueError("Could not detect dataset from logs")
    return m.group(1).upper()


def parse_block(text):
    losses, accs = [], []

    pattern = re.compile(
        r"Loss:\s*\[([^\]]+)\]\s*Accuracy:\s*\[([^\]]+)\]",
        re.S,
    )

    for m in pattern.finditer(text):
        losses.append(np.fromstring(m.group(1), sep=","))
        accs.append(np.fromstring(m.group(2), sep=","))

    if not losses:
        raise ValueError("No Loss/Accuracy found in log block")

    return np.mean(losses, axis=0), np.mean(accs, axis=0)


def main():
    raw = sys.stdin.read()

    # Split scratch vs pretrained using the flag
    blocks = re.split(
        r"\nOMP_NUM_THREADS=.*?--pt_weights",
        raw,
        maxsplit=1,
        flags=re.S,
    )

    scratch_block = blocks[0]
    pretrained_block = blocks[1]

    dataset = detect_dataset(raw)

    loss_s, acc_s = parse_block(scratch_block)
    loss_p, acc_p = parse_block(pretrained_block)

    epochs = np.arange(1, len(loss_s) + 1)

    # ---- Loss plot ----
    plt.figure()
    plt.plot(epochs, loss_s, label="From scratch", linewidth=2)
    plt.plot(epochs, loss_p, label="Pretrained on CIFAR10-5M", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"ResNet18 on {dataset} — Training Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"resnet18_{dataset.lower()}_loss_scratch_vs_pretrained.png")
    plt.close()

    # ---- Accuracy plot ----
    plt.figure()
    plt.plot(epochs, acc_s, label="From scratch", linewidth=2)
    plt.plot(epochs, acc_p, label="Pretrained on CIFAR10-5M", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(f"ResNet18 on {dataset} — Training Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        f"resnet18_{dataset.lower()}_accuracy_scratch_vs_pretrained.png")
    plt.close()


if __name__ == "__main__":
    main()
