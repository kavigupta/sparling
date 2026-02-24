# Sparling

Sparsity enforcement and adaptive sparsity update utilities for neural networks.

## Installation

```bash
pip install sparling
```

## Usage

Construct a sparsity layer with batch normalization (recommended):

```python
from sparling import SparseLayerWithBatchNorm

sparse_layer = SparseLayerWithBatchNorm(
    underlying_sparsity_spec=dict(type="EnforceSparsityPerChannel2D"),
    starting_sparsity=0.9,
    channels=128,
    affine=True,
    input_dimensions=2,  # 2 for (N,C,H,W), 1 for (N,C,L)
)

# Training: calibrate thresholds on your data
sparse_layer.train()
for batch in training_batches:
    out = sparse_layer(batch)  # thresholds update via momentum

# Inference: thresholds are frozen
sparse_layer.eval()
out = sparse_layer(x)  # ~90% of values are zero
```

Wrap the optimizer with a sparsity update optimizer to adaptively increase
sparsity when the model exceeds an accuracy threshold:

```python
from sparling import LinearThresholdAdaptiveSUO

suo = LinearThresholdAdaptiveSUO(
    optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
    initial_threshold=0.9,
    minimal_threshold=0.8,
    maximal_threshold=0.95,
    threshold_decrease_per_iter=1e-5,
    minimal_update_frequency=100,
    information_multiplier=0.5,
)

# In your training loop:
suo.zero_grad()
loss.backward()
suo.step()
suo.update_sparsity(model, step=step, acc_info=dict(acc=accuracy))
```

The model should have a setter on a property called `sparsity_value` that updates the sparsity of all sparsity layers in the model. For example:

```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.sparsity_value = 0.9
        self.sparse_layer1 = SparseLayerWithBatchNorm(...)
        self.sparse_layer2 = SparseLayerWithBatchNorm(...)
    @property
    def sparsity_value(self):
        return self._sparsity_value
    @sparsity_value.setter
    def sparsity_value(self, value):
        self._sparsity_value = value
        self.sparse_layer1.sparsity = value
        self.sparse_layer2.sparsity = value
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
| `EnforceSparsityPerChannel` | Per-channel threshold with momentum |
| `EnforceSparsityPerChannelAccumulated` | Accumulated batches before threshold update |
| `EnforceSparsityPerChannel2D` | 2-D (N,C,H,W) wrapper |
| `EnforceSparsityPerChannel1D` | 1-D (N,C,L) wrapper |
| `EnforceSparsityUniversally` | Single global threshold |

There's also sparsity combinators:

| Class | Description |
|---|---|
| `SparseLayerWithBatchNorm` | BatchNorm + sparsity wrapper. Absolutely necessary for performance. |
| `ParallelSparsityLayers` | Applies different sparsity layers to channel subsets |


Use the `sparsity_types()` registry to construct layers from config dicts via
`dconstruct.construct`.

### Sparsity update optimizers

| Class | Description |
|---|---|
| `NoopSUO` | Does nothing (baseline) |
| `LinearThresholdAdaptiveSUO` | Accuracy-threshold-driven adaptive sparsity reduction |

Use the `suo_types()` registry for construction.

## Development

```bash
pip install -r requirements.txt
pip install -e .
python -m pytest tests
python -m pylint sparling tests
```

## Baselines

Simple activation-based layers that do not enforce a target sparsity level,
useful as baselines or when sparsity is controlled externally (e.g. via an L1 or KL loss).

| Class | Description |
|---|---|
| `NoSparsity` | Identity pass-through |
| `SparsityForL1` | ReLU activation |
| `ChangingSparsityForL1` | ReLU with density-scaled motif loss |
| `SparsityForKL` | Sigmoid activation |
| `NoiseRatherThanSparsity` | Gaussian noise bottleneck |

## License

MIT
