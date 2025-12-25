import torch
from torch import nn as nn
from torchvision.models import resnet18, resnet34, resnet50, resnext50_32x4d
from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torchvision import transforms

import sys
import os
from dataclasses import dataclass
from typing import Callable, Any
from tqdm import tqdm, trange
import argparse
import time

DEFAULT_DATA_DIR = os.getenv("DEFAULT_DATA_DIR", None)
CIFAR10_5M_PATH = os.getenv("CIFAR10_5M_PATH", None)
assert DEFAULT_DATA_DIR is not None, "Environment variable < DEFAULT_DATA_DIR > not set."
assert CIFAR10_5M_PATH is not None, "Environment variable < CIFAR10_5M_PATH > not set."


@dataclass
class TrainingConfigs:
    model_type: str
    num_train_epochs: int
    num_ft_epochs: int
    batch_size: int
    learning_rate: float
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    use_ddp: bool
    default_seed: int = 1137


@dataclass
class DataConfigs:
    dataset_name: str
    img_size: int
    in_channels: int
    num_classes: int


class CIFAR5m(Dataset):
    def __init__(self, root_dir: str, transform: transforms.Compose | None, train: bool):
        import numpy as np
        self.root_dir = root_dir
        self.train = train
        self.transform = transform
        self.npz_files = [
            f'{root_dir}/cifar5m_part0.npz',
            f'{root_dir}/cifar5m_part1.npz',
            f'{root_dir}/cifar5m_part2.npz',
            f'{root_dir}/cifar5m_part3.npz',
            f'{root_dir}/cifar5m_part4.npz',
            f'{root_dir}/cifar5m_part5.npz']
        self.x = []
        self.y_true = []

        for npz_file in self.npz_files[:1]:
            npz_data = np.load(npz_file)
            self.x.append(npz_data['X'])
            self.y_true.extend(npz_data['Y'])
        self.x = np.vstack(self.x).reshape(-1, 3, 32, 32)
        self.x = self.x.transpose((0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, idx: int):
        x = self.x[idx]
        y_true = self.y_true[idx]

        if self.transform:
            x = self.transform(x)
        else:
            x = torch.from_numpy(x).float()
        y_true_torch = torch.tensor(y_true, dtype=torch.long)
        return x, y_true_torch

    def __len__(self):
        return len(self.x)


def ddp_setup(force_one_gpu: bool = False):
    if force_one_gpu:
        # WORLD_SIZE=1 MASTER_ADDR=localhost MASTER_PORT=12345 RANK=0 LOCAL_RANK=1 python3 train.py --dataset cifar10
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "12345")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("LOCAL_RANK", "0")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        init_process_group(backend="nccl")
    else:
        # OMP_NUM_THREADS=1 torchrun --nproc_per_node=2 train.py --model resnet18 --dataset cifar10 --train_epochs 10
        # OMP_NUM_THREADS=1 torchrun --nproc_per_node=2 train.py --dataset cifar10
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        init_process_group(backend="nccl")


def get_model_and_optimizer(model_name: str, num_classes: int):
    if model_name == "resnet18":
        model = resnet18(num_classes=num_classes)
    elif model_name == "resnet34":
        model = resnet18(num_classes=num_classes)
    elif model_name == "resnet50":
        model = resnet50(num_classes=num_classes)
    elif model_name == "resnext":
        model = resnext50_32x4d(num_classes=num_classes)
    else:
        raise ValueError("invalid model type.")
    optimizer = torch.optim.SGD(model.parameters(
    ), lr=train_conf.learning_rate, momentum=0.9, nesterov=True, weight_decay=1e-4)

    return model, optimizer


def get_cifar_dataset(dataset_type: str, train: bool = True):
    tf = transforms.Compose([])
    tf.transforms.append(transforms.ToTensor())
    if dataset_type == "cifar10":
        tf.transforms.append(transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)))
        dataset = CIFAR10(DEFAULT_DATA_DIR, train=train,
                          transform=tf, download=True)
    elif dataset_type == "cifar100":
        tf.transforms.append(transforms.Normalize(
            mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)))
        dataset = CIFAR100(DEFAULT_DATA_DIR, train=train,
                           transform=tf, download=True)
    elif dataset_type == "cifar5m":
        tf.transforms.append(transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)))
        dataset = CIFAR5m(CIFAR10_5M_PATH, transform=tf, train=train)
    else:
        raise ValueError("Invalid dataset type.")
    return dataset


class Trainer:
    def __init__(self, train_conf: TrainingConfigs, data_conf: DataConfigs, model: nn.Module, optimizer: torch.optim.Optimizer):
        os.makedirs("models", exist_ok=True)
        self.local_rank = int(os.getenv("LOCAL_RANK"))
        self.use_ddp = train_conf.use_ddp
        if self.use_ddp:
            self.gpu_id = int(os.getenv("LOCAL_RANK"))
        else:
            self.gpu_id = torch.device("cuda:0")
        self.model = model.to(self.gpu_id)
        self.optimizer = optimizer
        self.loss_fn = train_conf.loss_fn
        self.train_conf = train_conf
        self.data_conf = data_conf
        if self.use_ddp:
            self.model = DDP(self.model, device_ids=[self.gpu_id])

    def train(self, train_loader: Any, max_epochs: int, checkpoint_path: str | None):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max_epochs, eta_min=1e-5)
        train_start = time.monotonic_ns()
        train_times = []
        if self.local_rank == 0:
            print(
                f'Training model {train_conf.model_type} on device: {self.gpu_id}')
            print(
                f'Dataset: {self.data_conf.dataset_name} | Batch size: {self.train_conf.batch_size} | Checkpoint -> {checkpoint_path}')

        if self.use_ddp:
            _range = (len(train_loader.dataset) *
                      max_epochs)//int(os.getenv("WORLD_SIZE"))
            pbar = trange(0, _range, position=self.gpu_id)
        else:
            pbar = trange(0, len(train_loader.dataset)*max_epochs, position=0)

        for epoch in range(max_epochs):
            epoch_start = time.monotonic_ns()
            epoch_tloss = 0.0
            epoch_preds = 0
            for x, y_true in train_loader:
                x, y_true = x.to(self.gpu_id), y_true.to(self.gpu_id)
                self.optimizer.zero_grad()

                y = self.model(x)
                loss = self.loss_fn(y, y_true)
                epoch_tloss += loss.item()*x.shape[0]
                epoch_preds += (y.argmax(dim=1) == y_true).sum().item()

                loss.backward()
                self.optimizer.step()

                pbar.update(x.shape[0])
            scheduler.step()
            train_times.append((time.monotonic_ns()-epoch_start)/1e9)
            epoch_tloss /= len(train_loader.dataset)
            epoch_acc = epoch_preds/len(train_loader.dataset)*100

        pbar.close()
        train_time = round((time.monotonic_ns()-train_start)/1e9, 3)
        avg_epoch_time = round(sum(train_times)/len(train_times), 3)

        if self.use_ddp:
            dist.barrier()
            print(
                f'[GPU{self.gpu_id}/{os.getenv("WORLD_SIZE")}] Total time: {train_time} s (avg. {avg_epoch_time} s per epoch)')
            print(f'Final Loss/Acc: {epoch_tloss:.4f} / {epoch_acc:.2f} %')
        elif not self.use_ddp and self.local_rank == 0:
            print(
                f'Total train time: {train_time} s (avg. {avg_epoch_time} s per epoch)')
            print(f'Final Loss/Acc: {epoch_tloss:.4f} / {epoch_acc:.2f} %')
        # source how to save a model when using DPP
        # https://stackoverflow.com/questions/70386800/what-is-the-proper-way-to-checkpoint-during-training-when-using-distributed-data
        if checkpoint_path and self.local_rank == 0:
            self._save_snapshot(max_epochs, checkpoint_path)

    def _save_snapshot(self, epoch: int, path: str):
        if self.use_ddp:
            snapshot = {
                "MODEL_STATE": self.model.state_dict(),
                "RUNS": epoch,
                "DDP": self.use_ddp,
            }
        else:
            snapshot = {
                "MODEL_STATE": self.model.state_dict(),
                "RUNS": epoch,
                "DDP": self.use_ddp,
            }
        torch.save(snapshot, path)


def cifar5m_workflow(train_conf: TrainingConfigs, data_conf: DataConfigs, train_cifar5m_first: bool = True, use_pretrained_weights: bool = True):
    if train_conf.use_ddp:
        ddp_setup()
        if dist.get_rank() == 0:
            print(f'Training with DPP: WS= {dist.get_world_size()}')

    # 1 ---- train the model on CIFAR10-5m
    cifar5m_dataset = get_cifar_dataset("cifar5m", train=True)
    checkpoint_path = f"models/_proto__{train_conf.model_type}-cifar5m_{train_conf.num_train_epochs}epochs.pth"
    if train_cifar5m_first:
        if train_conf.use_ddp:
            cifar5m_loader = DataLoader(
                cifar5m_dataset,
                batch_size=train_conf.batch_size,
                shuffle=False,
                num_workers=4,
                sampler=DistributedSampler(cifar5m_dataset))
        else:
            cifar5m_loader = DataLoader(
                cifar5m_dataset,
                batch_size=train_conf.batch_size,
                shuffle=False,
                num_workers=4)
        model, optimizer = get_model_and_optimizer(train_conf.model_type, 10)
        trainer = Trainer(train_conf=train_conf,
                          data_conf=DataConfigs(
                              dataset_name="cifar5m", img_size=32, in_channels=3, num_classes=10),
                          model=model,
                          optimizer=optimizer)
        trainer.train(cifar5m_loader,
                      train_conf.num_train_epochs, checkpoint_path)

    # 2 ---- train/finetune the model on CIFARX
    train_dataset = get_cifar_dataset(data_conf.dataset_name, train=True)
    if train_conf.use_ddp:
        train_loader = DataLoader(
            train_dataset,
            batch_size=train_conf.batch_size,
            shuffle=False,
            num_workers=4,
            sampler=DistributedSampler(train_dataset))
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=train_conf.batch_size,
            shuffle=False,
            num_workers=4)
    model, optimizer = get_model_and_optimizer(train_conf.model_type, 10)
    if use_pretrained_weights:
        print(f'Loading pretrained weights from: {checkpoint_path}')
        model_pth = torch.load(checkpoint_path)
        _state_dict = model_pth['MODEL_STATE']
        _state_runs = model_pth['RUNS']
        _state_ddp = model_pth['DDP']
        # make sure the loaded state dict is compatible with the current DDP process
        assert _state_ddp == train_conf.use_ddp, "Cannot load model in the current workflow. Inconsistent DDP between checkpoint and the environment."
        if _state_ddp is False:
            print(f'Loading non-DDP weights')
            model.load_state_dict(_state_dict)
        else:
            # source: https://discuss.pytorch.org/t/failed-to-load-model-trained-by-ddp-for-inference/84841
            print(f'Loading DDP weights')
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in _state_dict.items():
                # remove 'module.' of DataParallel/DistributedDataParallel
                name = k[7:]
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
    if data_conf.num_classes != 10:
        _in_features = model.fc.in_features
        model.fc = nn.Linear(_in_features, data_conf.num_classes)
    trainer = Trainer(train_conf=train_conf,
                      data_conf=data_conf,
                      model=model,
                      optimizer=optimizer)
    trainer.train(train_loader, train_conf.num_ft_epochs,
                  f"models/_ft__{train_conf.model_type}-{data_conf.dataset_name}_{train_conf.num_ft_epochs}epochs.pth")

    if train_conf.use_ddp:
        destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Algorithm for training CIFAR10-5m and then perform finetuning on CIFAR10-5m/CIFAR10/CIFAR100")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="The learning rate")
    parser.add_argument("--model", type=str, default="resnet18",
                        help="The model used for training and fine-tuning")
    parser.add_argument("--dataset", type=str, required=True,
                        help="The dataset to be used for fine-tuning the pretrained model")
    parser.add_argument("--train_epochs", type=int, default=1,
                        help="Number of epochs for training and fine-tuning")
    parser.add_argument("--ft_epochs", type=int, default=1,
                        help="Number of epochs for training and fine-tuning")
    parser.add_argument("--ddp", default=False,
                        action="store_true", help="Use DDP")
    args = parser.parse_args()

    assert args.model in ["resnet18", "resnet34",
                          "resnet50", "resnext"], "Invalid model type."
    assert args.dataset in ["cifar10", "cifar100",
                            "cifar5m"], "Invalid dataset type."

    train_conf = TrainingConfigs(model_type=args.model,
                                 num_train_epochs=args.train_epochs,
                                 num_ft_epochs=args.ft_epochs,
                                 batch_size=128,
                                 learning_rate=args.lr,
                                 use_ddp=args.ddp,
                                 loss_fn=nn.CrossEntropyLoss())
    num_classes = 100 if args.dataset == "cifar100" else 10
    data_conf = DataConfigs(dataset_name=args.dataset,
                            img_size=32,
                            in_channels=3,
                            num_classes=num_classes)

    # OMP_NUM_THREADS=1 torchrun --nproc_per_node=2 train.py --model resnet18 --dataset cifar10 --train_epochs 3 --ft_epochs 5 --ddp
    torch.manual_seed(train_conf.default_seed)
    torch.cuda.manual_seed_all(train_conf.default_seed)
    cifar5m_workflow(train_conf=train_conf, data_conf=data_conf,
                     train_cifar5m_first=False, use_pretrained_weights=True)
