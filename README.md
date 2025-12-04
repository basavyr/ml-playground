# ML Playground 🚀

A collection of tools used for learning Machine Learning and Deep Neural Network concepts

## November 2025 Update

The [`benchmarks/`](./benchmarks/) folder includes new implementations that are useful for testing the compute capabilities of different GPUs and accelerators.

Several key architectures are studied and measured in terms of training & inference performance:
- deep neural networks
- convolutional neural networks
- transformers (decoders only)

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

## Run DNN Benchmark

TBD...