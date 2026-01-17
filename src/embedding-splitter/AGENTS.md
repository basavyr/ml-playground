## Agent Behavior Guidelines

### Smart Solutions First
- **Pretrained > Custom**: Always prefer pretrained models (ResNet, etc.) over custom architectures
- **Proven Patterns**: Use established solutions rather than reinventing
- **Flexible Design**: Implement boolean flags for model choices (`use_pretrained=True`)
- **Environment Variables**: Use `os.getenv()` for configurable paths

### Efficiency by Default
- **Performance Monitoring**: Always track bandwidth and timing
- **Device Detection**: Support MPS (Apple Silicon), CUDA, CPU automatically
- **Memory Awareness**: Calculate tensor sizes using `x.numel() * x.element_size()`
- **Graceful Degradation**: Skip corrupted data with warnings, don't fail

### Super Concise Implementation
- **Minimal Code**: Maximum functionality with minimum lines
- **No Redundancy**: Remove unnecessary abstractions
- **Direct Solutions**: Solve problems without over-engineering
- **Clear Output**: Essential metrics only (bandwidth, loss, accuracy)

## Build/Lint/Test Commands

### Python Environment
- **Dependencies**: `pip install torch torchvision pandas pillow tqdm`
- **Main script**: `python main.py`
- **Testing**: `pytest tests/` (when implemented)

### Code Quality
- **Linting**: `ruff check .`
- **Formatting**: `black .`
- **Type checking**: `mypy .`

## Smart Coding Patterns

### Import Organization
```python
# Standard library
import os
import time
from typing import Optional

# ML libraries
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18
from torchvision.transforms import Compose, Resize, ToTensor
```

### Model Selection Pattern
```python
class Model(torch.nn.Module):
    def __init__(self, use_pretrained: bool = True):
        if use_pretrained:
            self.model = resnet18(weights='DEFAULT')
            self.model.fc = torch.nn.Linear(512, num_classes)
        else:
            # Custom architecture only when necessary
```

### Performance Monitoring Pattern
```python
start = time.monotonic_ns()
total_mb = 0

for x, y in dataloader:
    total_mb += float(x.numel() * x.element_size()) / (1024**2)
    # Your training logic here

duration = (time.monotonic_ns() - start) / 1e9
bandwidth_mbps = total_mb / duration
print(f'Bandwidth: {bandwidth_mbps:.2f} MB/s')
```

### Dataset Implementation
```python
class SimpleDataset(Dataset):
    def __init__(self, data_dir: str, max_files: int = 5):
        files = sorted([f for f in os.listdir(data_dir) if f.endswith('.pt')])
        self.files = files[:max_files]
        self.data = []
        
        for f in self.files:
            data = torch.load(os.path.join(data_dir, f))
            self.data.extend([(data['images'][i], data['labels'][i]) 
                            for i in range(len(data['images']))])
```

### Error Handling Pattern
```python
try:
    # Processing logic
except Exception as e:
    print(f"Warning: Skipping item {idx}: {e}")
    continue  # Always continue, never crash
```

## Development Standards

### Device Management
```python
device = torch.device("mps" if torch.backends.mps.is_available() else 
                      "cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
```

### File Operations
- Use `tqdm` for progress: `for item in tqdm(items, desc='Processing')`
- Check existence: `if not os.path.exists(path): raise FileNotFoundError`
- Efficient naming: `f'prefix-{idx:04d}.ext'`

### Output Format
- **Essential metrics only**: bandwidth, loss, accuracy
- **Precise timing**: `time.monotonic_ns()` for high precision
- **Clean formatting**: `{value:.2f}` for floating points
- **Progress indicators**: Always use `tqdm` with descriptive labels

## Testing Strategy
- **Unit tests**: `pytest -v tests/test_dataset.py`
- **Integration tests**: Test end-to-end pipeline with small datasets
- **Performance tests**: Validate bandwidth calculations
- **Error conditions**: Test missing files, corrupted data, empty datasets

## Quick Reference

### Bandwidth Calculation
```python
tensor_mb = float(tensor.numel() * tensor.element_size()) / (1024**2)
```

### Device Selection
```python
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
```

### Efficient File Sorting
```python
files = sorted([f for f in os.listdir(dir) if condition(f)])
```

### ResNet with Custom Classes
```python
model = resnet18(weights='DEFAULT')
model.fc = torch.nn.Linear(in_features=512, out_features=num_classes)
```