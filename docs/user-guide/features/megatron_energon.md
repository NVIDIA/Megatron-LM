# Megatron Energon

Advanced multimodal dataloader for efficient loading of text, images, video, and audio at scale.

## Overview

[**Megatron Energon**](https://github.com/NVIDIA/Megatron-Energon) is purpose-built for large-scale multimodal training with:

- **Multimodal support** - Text, images, video, audio
- **Distributed loading** - Optimized for multi-node training
- **Data blending** - Mix datasets with configurable weights
- **WebDataset format** - Efficient streaming from cloud storage
- **State management** - Save and restore training position

## Installation

```bash
pip install megatron-energon
```

## Key Features

### Data Processing

- **Packing** - Optimize sequence length utilization
- **Grouping** - Smart batching of similar-length sequences
- **Joining** - Combine multiple dataset sources
- **Object storage** - Stream from S3, GCS, Azure Blob Storage

### Production-Ready

- Distributed loading across workers and nodes
- Checkpoint data loading state
- Memory-efficient streaming
- Parallel data loading with prefetching

## Basic Usage

```python
from megatron.energon import get_train_dataset, get_loader, WorkerConfig

# Create dataset
ds = get_train_dataset(
    '/path/to/dataset',
    batch_size=32,
    shuffle_buffer_size=1000,
    worker_config=WorkerConfig.default_worker_config(),
)

# Create loader and iterate
for batch in get_loader(ds):
    # Training step
    pass
```

## Multimodal Example

```python
# Load image-text dataset
ds = get_train_dataset(
    '/path/to/multimodal/dataset',
    batch_size=32,
    worker_config=WorkerConfig(num_workers=8, prefetch_factor=2),
)

for batch in get_loader(ds):
    images = batch['image']  # Image tensors
    texts = batch['text']    # Text captions
    # Process batch
```

## Dataset Blending

Mix multiple datasets with custom weights:

```python
from megatron.energon import Blender

blended_ds = Blender([
    ('/path/to/dataset1', 0.6),  # 60%
    ('/path/to/dataset2', 0.3),  # 30%
    ('/path/to/dataset3', 0.1),  # 10%
])
```

## Configuration

### Worker Configuration

```python
WorkerConfig(
    num_workers=8,              # Parallel workers
    prefetch_factor=2,          # Batches to prefetch per worker
    persistent_workers=True,    # Keep workers alive between epochs
)
```

### Common Parameters

| Parameter | Description |
|-----------|-------------|
| `batch_size` | Samples per batch |
| `shuffle_buffer_size` | Buffer size for randomization |
| `max_samples_per_sequence` | Max samples to pack into one sequence |
| `worker_config` | Worker configuration for parallel loading |

## Integration with Megatron-LM

```python
from megatron.energon import get_train_dataset, get_loader
from megatron.training import get_args

args = get_args()

train_ds = get_train_dataset(
    args.data_path,
    batch_size=args.micro_batch_size,
)

for iteration, batch in enumerate(get_loader(train_ds)):
    loss = train_step(batch)
```

## Resources

- **[Megatron Energon GitHub](https://github.com/NVIDIA/Megatron-Energon)** - Documentation and examples
- **[Multimodal Examples](https://github.com/NVIDIA/Megatron-LM/tree/main/examples/multimodal)** - Megatron-LM multimodal training

## Next Steps

- Check [Multimodal Models](../../models/multimodal.md) for supported architectures
- See [Training Examples](../training-examples.md) for integration examples
