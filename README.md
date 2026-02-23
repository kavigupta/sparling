# Sparling

Sparsity enforcement and adaptive sparsity update utilities for neural networks.

## Installation

```bash
pip install sparling
```

## Overview

Sparling provides a collection of `torch.nn.Module`-based sparsity layers and
adaptive sparsity update optimizers for training sparse neural networks.

### Sparsity layers

All sparsity layers extend the `Sparsity` base class (itself an `nn.Module`).
The `sparsity` property can be updated at any time, and subclasses react via
`notify_sparsity()`.

| Class | Description |
|---|---|
| `NoSparsity` | Identity pass-through |
| `SparsityForL1` | ReLU activation |
| `ChangingSparsityForL1` | ReLU with density-scaled motif loss |
| `SparsityForKL` | Sigmoid activation |
| `EnforceSparsityPerChannel` | Per-channel threshold with momentum |
| `EnforceSparsityPerChannelAccumulated` | Accumulated batches before threshold update |
| `EnforceSparsityPerChannel2D` | 2-D (N,C,H,W) wrapper |
| `EnforceSparsityPerChannel1D` | 1-D wrapper |
| `EnforceSparsityUniversally` | Single global threshold |
| `NoiseRatherThanSparsity` | Gaussian noise bottleneck |
| `SparseLayerWithBatchNorm` | BatchNorm + sparsity wrapper |
| `EnforceSparsity1D` | 3-D (N,C,L) wrapper |
| `ParallelSparsityLayers` | Applies different sparsity layers to channel subsets |

Use the `sparsity_types()` registry to construct layers from config dicts via
`dconstruct.construct`.

### Sparsity update optimizers

| Class | Description |
|---|---|
| `NoopSUO` | Does nothing (baseline) |
| `LinearThresholdAdaptiveSUO` | Accuracy-threshold-driven adaptive sparsity reduction |

Use the `suo_types()` registry for construction.

## Quick example

```python
from sparling import EnforceSparsityPerChannel, NoopSUO

sparse = EnforceSparsityPerChannel(starting_sparsity=0.9, channels=128)
# ... use sparse(x) inside your model ...
```

## Development

```bash
pip install -r requirements.txt
pip install -e .
python -m pytest tests
python -m pylint sparling tests
```

## License

MIT
