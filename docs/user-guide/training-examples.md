<!---
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# Training Examples

Get started with Megatron Core training using these practical examples.

## Simple Training Example

The simplest way to get started is with the basic training loop using mock data:

```bash
# Distributed training on 2 GPUs with mock data
torchrun --nproc_per_node=2 examples/run_simple_mcore_train_loop.py
```

This example:
- Runs on 2 GPUs
- Uses generated mock data (no data preparation needed)
- Demonstrates basic distributed training setup
- Perfect for testing your installation

## LLaMA-3 Training Examples

### LLaMA-3 8B with FP8

Train LLaMA-3 8B model with FP8 mixed precision on 8 GPUs:

```bash
./examples/llama/train_llama3_8b_fp8.sh
```

**Configuration:**
- 8 GPUs
- FP8 mixed precision (requires Hopper/Ada/Blackwell GPUs)
- Mock data for quick testing

### Custom LLaMA Training

For training with your own data:

```bash
torchrun --nproc_per_node=8 pretrain_gpt.py \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --num-layers 32 \
    --hidden-size 4096 \
    --num-attention-heads 32 \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
    --micro-batch-size 4 \
    --global-batch-size 32 \
    --train-iters 100000 \
    --lr 3.0e-4 \
    --min-lr 3.0e-5 \
    --lr-decay-style cosine \
    --lr-warmup-iters 2000 \
    --weight-decay 0.1 \
    --clip-grad 1.0 \
    --bf16 \
    --data-path /path/to/your/preprocessed_data \
    --split 949,50,1 \
    --save /path/to/checkpoints \
    --load /path/to/checkpoints \
    --log-interval 10 \
    --save-interval 1000 \
    --eval-interval 1000
```

## GPT-3 Training Example

Train a GPT-3 style model:

```bash
torchrun --nproc_per_node=8 pretrain_gpt.py \
    --tensor-model-parallel-size 2 \
    --pipeline-model-parallel-size 2 \
    --num-layers 24 \
    --hidden-size 2048 \
    --num-attention-heads 16 \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --micro-batch-size 2 \
    --global-batch-size 16 \
    --train-iters 100000 \
    --lr 1.5e-4 \
    --min-lr 1.0e-5 \
    --lr-decay-style cosine \
    --lr-warmup-iters 1000 \
    --weight-decay 0.1 \
    --clip-grad 1.0 \
    --fp16 \
    --data-path /path/to/preprocessed_data \
    --split 949,50,1 \
    --save /path/to/checkpoints \
    --load /path/to/checkpoints
```

## Key Training Arguments

### Model Architecture

| Argument | Description |
|----------|-------------|
| `--num-layers` | Number of transformer layers |
| `--hidden-size` | Hidden dimension size |
| `--num-attention-heads` | Number of attention heads |
| `--seq-length` | Sequence length for training |

### Training Configuration

| Argument | Description |
|----------|-------------|
| `--micro-batch-size` | Batch size per GPU |
| `--global-batch-size` | Total batch size across all GPUs |
| `--train-iters` | Number of training iterations |

### Learning Rate

| Argument | Description |
|----------|-------------|
| `--lr` | Peak learning rate |
| `--min-lr` | Minimum learning rate |
| `--lr-decay-style` | LR schedule (cosine, linear, constant) |
| `--lr-warmup-iters` | Warmup iterations |

### Mixed Precision

| Argument | Description |
|----------|-------------|
| `--fp16` | FP16 mixed precision |
| `--bf16` | BF16 mixed precision (recommended) |
| `--fp8-hybrid` | FP8 mixed precision (Hopper/Ada/Blackwell) |

### Data and Checkpointing

| Argument | Description |
|----------|-------------|
| `--data-path` | Path to preprocessed data |
| `--split` | Train/validation/test split (e.g., 949,50,1) |
| `--save` | Checkpoint save directory |
| `--load` | Checkpoint load directory |
| `--save-interval` | Save checkpoint every N iterations |

## Next Steps

- **Optimize Performance**: See [Advanced Features](features/index.md) for FSDP, distributed optimizer, and other optimizations
- **Scale Up**: Learn about [Parallelism Strategies](parallelism-guide.md) to train larger models across more GPUs
- **Prepare Data**: Follow the [Data Preparation](data-preparation.md) guide to process your own datasets
