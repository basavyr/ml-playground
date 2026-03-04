from imagenet_dataset import ImageNetParquetDataset
from torch.utils.data import DataLoader
import torch
data_dir = "/home/robertp/Documents/imagenet-1k/data"
train_ds = ImageNetParquetDataset(data_dir, split="train")
val_ds   = ImageNetParquetDataset(data_dir, split="validation")
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,num_workers=1, pin_memory=False)

x, y_true = next(iter(train_loader))

x = x.to("cuda")
y_true = y_true.to("cuda")

n_iter = 1000000
for i in range(n_iter):
    x = x+ x
    x@torch.eye(224,device = "cuda")
    if i%100==0:
        print(f"iteration {i}/{n_iter} reached")
print(x.shape, y_true.shape)
