# ML Playground 🚀

A collection of tools used for learning Machine Learning and Deep Neural Network concepts.

## January 2026 Update

With the rise of agentic tools such as [Opencode](https://opencode.ai/), a few projects have been started and maintained using Opencode. The projects that are developed within an agentic style can be identified by the presence of the `AGENTS.md` files.

## November 2025 Update

The [`benchmarks/`](./benchmarks/) folder includes new implementations that are useful for testing the compute capabilities of different GPUs and accelerators.

Several key architectures are studied and measured in terms of training & inference performance:
- deep neural networks
- convolutional neural networks
- transformers (decoders only)

The first two architectures are tested in the script [`neural.py`](./benchmarks/neural.py), while the latter is tested in [`transformer.py`](./benchmarks/transformer.py).

### Benchmark metrics

Currently (December 2025), the key performance metric tracked for the Transformer implementation is the total number of **Floating-Point-Operations (FLOP)**. Moreover, since the entire training process is tracked, one can also determine the total number of operations per second, or FLOPS for short. The way of determining the actual FLOPS resides from well-established formulas, which approximate the number of operations on the accelerator. Typically, a matrix multiplication (GEMM) is considered as two FLOP, since it is a Fused-Matrix-Multiply-Accumulate op (FMA or MAC) consisting of one multiplication and one addition (hence the 1MAC=2FLOP).

There are plenty of useful resources that can help to determine the FLOP counter on transformer models. Below are several:
- https://arxiv.org/pdf/2001.08361
- https://epoch.ai/blog/backward-forward-FLOP-ratio
- https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html
- https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#math-mem

### Run Transformer Benchmark

The script [`transformer.py`](./benchmarks/transformer.py) aims at simulating a training procedure for a Decoder-only model (i.e., [`nn.TransformerDecoder`](https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerDecoder.html) where the target mask and encoder output - memory - are irrelevant). The data is synthetically generated via [`torch.randn`](https://docs.pytorch.org/docs/stable/generated/torch.randn.html).

Details on how the training was designed to be as efficient and minimal as possible are given [here](./benchmarks/docs.md).

1. Install PyTorch. **If an external GPU is available (such as NVIDIA), make sure PyTorch is installed with CUDA support**.
     - The "Get started" guide from PyTorch available [here](https://pytorch.org/get-started/locally/) shows how to install on Windows or Linux
     - ⚠️ **CUDA Toolkit must be pre-configured beforehand**
     - For Windows/Linux, the CUDA toolkit can be downloaded from [here](https://developer.nvidia.com/cuda-downloads?target_os=Windows)
     - According to the official guide's description: *please ensure that you have met the prerequisites below (e.g., `numpy`)*
2. Install `tqdm` if not available (this package is required for the progress bar during training).
3. Navigate to `benchmarks/` and run:
    ```bash
    python3 transformer.py
    ```
    An output like this should be obtained:
    ```
    python3 transformer.py
    2025-12-04 14:22:59 - Training on mps for 3 epochs.
    <<< Config >>>
    Batch Size=24 | Total samples=1000
    Sequence Length=128
    N_decoder_layers=6 | num_attn_heads=8 | d_k=384
    ================================================================================
    Epoch 1: Training transformer: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 42/42 [00:08<00:00,  5.02it/s]
    2025-12-04 14:23:08 - Epoch 1: Loss= 10.461 [8.371512 s]
    Epoch 2: Training transformer: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 42/42 [00:08<00:00,  5.07it/s]
    2025-12-04 14:23:17 - Epoch 2: Loss= 10.302 [8.277792 s]
    Epoch 3: Training transformer: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 42/42 [00:08<00:00,  5.08it/s]
    2025-12-04 14:23:25 - Epoch 3: Loss= 10.210 [8.270144 s]
    2025-12-04 14:23:25 - Full training finished in 24.920 s (8.306 s per epoch)
    2025-12-04 14:23:25 - Total operations: 96.0163 TFLOPs
    2025-12-04 14:23:25 - Achieved avg. << 3.7897 >> TFLOPS
    ```
4. Compare your metrics with other systems ☺️ Keep in mind that this script handles all the logging is done automatically, so no additional prints are required.
5. **Update**: After the benchmark is finished, available logs with the run can be checked inside the `./logs/` directory, which is created automatically after the first benchmark.

## Run DNN Benchmark

The latest update (December 2025, sha-`8750b88862248f69e5f3b0aa016d5abaaf5b5060`) added new benchmarks for deep neural networks such as ResNet18 and even bigger ones like ResNet50. These can be tested against standard datasets such as MNIST, CIFAR10, and even Tiny ImageNet 200.

The script [`neural.py`](./benchmarks/neural.py) contains the complete benchmarking workflow for these types of models.

> [!NOTE]  
> The implementation has a special helper [`StandardDatasets`](./benchmarks/utils.py), which can get the most popular datasets. Please read its docstring to understand how to use it for custom datasets (e.g., without relying on automatic download, providing custom paths, apply resize of pixel width, etc).

> [!IMPORTANT]  
> The FLOP counter for the deep neural network architecture is still under development, thus the only relevant performance indicator is **epoch time** (given a specific training configuration).

Usage is straightforward. If the datasets are not already available on the system, one can use `FORCE_DOWNLOAD=1` environment variable when running the script.
1. Navigate to `benchmarks/` and run:
    ```bash
    FORCE_DOWNLOAD=1 python3 neural.py
    ```
    or (if your datasets are already available)
    ```bash
    python3 neural.py
    ```
2. After execution, metrics can be checked inside the `./logs` directory.

> [!CAUTION]
> The dataset retriever expects a default path to keep all files. In the current version this is set to `./data`. This will assure that everything is placed within the current working directory, but separated from the rest of implementation. Git is already configured to ignore everything in that path.

See default behavior of [`datasets.py`](./benchmarks/datasets.py) below:
```python
@dataclass
class DatasetConfig:
    name: str
    path: str | None
    download: bool = False
    resize_to: int = -1
    force_3_channels: bool = False
    data_dir: str = "./data"

# then in neural.py 
...
dataset_helper = StandardDatasets("./data")
...
```