import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.flop_counter import FlopCounterMode
from torchvision.models import resnet18, resnet34, resnet50

from typing import Callable, Tuple
from tqdm import tqdm
import os
import sys
import time
import logging
from dataclasses import dataclass

# local imports
from utils import get_optimal_device, generate_log_file
from datasets import StandardDatasets, DatasetConfig
from models import LinearNet


FORCE_DOWNLOAD = os.getenv("FORCE_DOWNLOAD", "0")


log_file, generated_at = generate_log_file("neural")
logging.basicConfig(filename=log_file, level=logging.INFO,
                    format='%(asctime)s - %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger()


@dataclass
class TrainingConfigs:
    device: torch.types.Device
    batch_size: int
    loss_fn: nn.Module = torch.nn.CrossEntropyLoss()
    num_epochs: int = 10
    learning_rate: float = 1e-3


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.types.Device,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    learning_rate: float,
    num_epochs: int,
):
    generate_log_file('neural')
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
    log = logging.getLogger()
    model.to(device)
    model.train()
    log.info(
        f'Started training {model._get_name()} on <<< {device} >>> for {num_epochs} epochs')
    log.info(
        f'Dataset: {type(train_loader.dataset).__name__} | BS= {train_loader.batch_size} | lr= {learning_rate}')
    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)

    timing_epochs = []
    start_train = time.monotonic_ns()
    for epoch in range(num_epochs):
        epoch_start = time.monotonic_ns()
        epoch_loss = 0.0
        epoch_acc = 0.0
        preds = 0
        for x, y_true in tqdm(train_loader, desc=f'Training epoch {epoch+1}/{num_epochs}'):
            x, y_true = x.to(device), y_true.to(device)
            optimizer.zero_grad()

            y = model(x)

            loss = loss_fn(y, y_true)
            epoch_loss += loss.item()*x.shape[0]
            preds += (torch.argmax(y, dim=1) == y_true).sum().item()

            loss.backward()
            optimizer.step()
        epoch_duration = (time.monotonic_ns()-epoch_start)*1e-9
        timing_epochs.append(epoch_duration)
        epoch_loss /= len(train_loader.dataset)
        epoch_acc = preds/len(train_loader.dataset)*100
        log.info(
            f'Epoch {epoch+1} -> Loss = {epoch_loss:.4f} Acc = {epoch_acc:.2f} % [{epoch_duration:.2f} s]')

    train_duration = (time.monotonic_ns()-start_train)*1e-9
    avg_s_per_epoch = sum(timing_epochs)/len(timing_epochs)
    log.info(
        f'Full training duration: {train_duration:.2f} s (avg: {avg_s_per_epoch} s per epoch)')


def run_model_workflow(
    model_type: str,
    training_config: TrainingConfigs,
    dataset_config: DatasetConfig,
):
    log.info(f'{"="*20} Benchmark {model_type} architecture {"="*20}')
    # -------- dataset  --------
    dataset_helper = StandardDatasets(dataset_config.data_dir)
    dataset = dataset_helper.get_dataset(
        dataset_name=dataset_config.name,
        custom_image_folder=dataset_config.path,
        download=dataset_config.download,
        resize_to=dataset_config.resize_to,
        force_3_channels=dataset_config.force_3_channels)
    assert dataset is not None
    train_loader = DataLoader(
        dataset, batch_size=training_config.batch_size, shuffle=False)

    # -------- model  --------
    if model_type == "linear":
        model = LinearNet(num_hidden_layers=5,
                          input_size=dataset_helper.input_size,
                          hidden_dim=64,
                          output_size=dataset_helper.num_classes)
    elif model_type == "resnet18":
        model = resnet18(num_classes=dataset_helper.num_classes)
    elif model_type == "resnet34":
        model = resnet18(num_classes=dataset_helper.num_classes)
    elif model_type == "resnet50":
        model = resnet18(num_classes=dataset_helper.num_classes)
    else:
        raise ValueError("Unsupported model type")

    train_model(model=model,
                train_loader=train_loader,
                device=training_config.device,
                loss_fn=training_config.loss_fn,
                learning_rate=training_config.learning_rate,
                num_epochs=training_config.num_epochs)


if __name__ == "__main__":
    # -------- configs --------
    training_config = TrainingConfigs(device=get_optimal_device(),
                                      batch_size=128,
                                      learning_rate=0.01)
    all_ds_configs = [DatasetConfig("tiny", "./data/tiny-imagenet-200", False,  -1, False),
                      DatasetConfig("mnist", None, False, -1, True),
                      DatasetConfig("cifar10", None, False, -1, False),
                      DatasetConfig("cifar100", None, False, -1, False),]
    if FORCE_DOWNLOAD == "1":
        for ds_conf in all_ds_configs:
            ds_conf.download = True

    # mnist and cifar100 by default
    run_model_workflow("linear", training_config, all_ds_configs[1])
    run_model_workflow("resnet50", training_config, all_ds_configs[-1])
    log.info(f'System info: {os.uname().nodename}')
