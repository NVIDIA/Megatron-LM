{/*
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
*/}

# Training Examples

Get started with Megatron Core training using these practical examples.

## Basic Training Example

Use the basic training loop with mock data to get started:

```bash
# Distributed training on 2 GPUs with mock data
torchrun --nproc_per_node=2 examples/run_simple_mcore_train_loop.py
```

This example:

- Runs on two GPUs
- Uses generated mock data (no data preparation needed)
- Demonstrates basic distributed training setup
- Provides a quick way to verify your installation

## LLaMA-3 Training Examples

### LLaMA-3 8B with FP8

Train the LLaMA-3 8B model with FP8 mixed precision on eight GPUs:

```bash
./examples/llama/train_llama3_8b_h100_fp8.sh
```

**Configuration**

- Eight GPUs
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

## Sequence lengths in the training configuration

Dataset content limits and model execution widths have different owners:

- `cfg.data.dataset.sequence_length` controls GPT row length and supplies SFT's
  default pack width. It is not a separate real-SFT conversation truncation limit.
  The CLI supplies it with `--seq-length`.
- `cfg.model.transformer.max_seq_length_per_sample` bounds one physical model input
  row, or an entire packed THD row, including padding and expanded image tokens.
  This is the maximum for the training run, not the model's architectural capacity.
  `--max-seq-length-per-sample` sets it explicitly; GPT defaults to `--seq-length`.
- `cfg.model.seq_length` retains `--max-position-embeddings`: it sizes learned
  absolute position tables and is retained as model/export metadata for RoPE.
  It is not a dataset truncation limit or a RoPE execution bound, and is not an
  alias of the run's `max_seq_length_per_sample`. Learned tables may be larger
  than the inputs used in a particular run.
- `max_seqlen_per_dp_cp_rank` remains the separate per-rank dynamic-CP packing budget.

For example, `--sft --seq-length 2048 --max-seq-length-per-sample 8192` emits
8192-token padded packs. Real SFT preserves its existing behavior: append tokenized
conversations, then truncate the combined pack, without truncating each conversation
at 2048 tokens. Independent per-conversation truncation is outside this migration. The
container derives `SFTDatasetConfig.packed_sequence_length` during startup;
recipes should set the model bound rather than this derived dataset field. Existing
`--seq-length`-only runs retain their previous widths and packing behavior.
The SFT-specific config lives alongside `SFTDataset`; Core's `GPTDatasetConfig`
has no pack-width field.

Core's existing `GPTDatasetConfig` remains shared by the GPT, SFT, and Varlen
dataset implementations. `GPTFIMDatasetConfig` adds FIM settings. The small training-side
`SFTDatasetConfig(GPTDatasetConfig)` subclass adds only the derived physical pack
width; Varlen continues to use `GPTDatasetConfig` with an explicit dataset class.
Separating the remaining family-specific fields is deferred to a follow-up.

Fixed GPT/FIM dataset rows must match the run's execution length; an explicit
different value is rejected rather than silently selecting another runtime length.
T5 retains separate encoder and decoder input lengths. Callback datasets such as
Energon still own truncation and packing; their
entrypoints bind those inputs before `pretrain()`, and resolve the model execution
bound before runtime buffers are initialized. Fixed-width BERT, mock VLM, and
ModelOpt callbacks reject bounds inconsistent with the rows they actually emit.

`validate()` checks these constraints without mutation. Framework startup calls
`finalize()` to derive unset/owned fields before any runtime consumer reads them.

### Python API compatibility

Python SFT callers use `SFTDatasetConfig` for a pack width distinct from the
legacy `sequence_length` value. Container deserialization upgrades saved SFT sections from
`GPTDatasetConfig` to this subclass using the saved `dataset_type`, preserving
all fields. Unknown fields still fail strict loading. Varlen callers and CLI flag
names are unchanged.

For LLaVA, set the resolved text + image + padding width in
`model.transformer.max_seq_length_per_sample` before startup.

The `--decoder-seq-length` CLI flag remains accepted: VLM entrypoints translate
it into their execution width, while T5 retains an independent decoder width.
Decoder-only models do not acquire a second decoder-length config field.

Core's schedule APIs are unchanged. The training loop passes the resolved model
execution width as `seq_length`, without a separate decoder override. T5 retains
separate model/dataset encoder and decoder lengths; its entrypoint does not support
pipeline parallelism.

Direct Python callers of the migrated callback dataset providers and task encoders
must pass their length inputs explicitly instead of relying on global arguments.
The supplied application entrypoints bind these inputs before `pretrain()`; their
existing CLI invocations remain supported.

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

The following tables group common training arguments by category.

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
| `--no-check-optimizer-step-success` | Disable synchronization of optimizer-step success across model-parallel ranks before advancing the scheduler. Checking is enabled by default. |
| `--skip-sync-grad-norm-across-mp` | Skip synchronization of the reported gradient norm across model-parallel ranks; gradient clipping is unchanged. Disabled by default. |
| `--spiky-loss-factor` | Loss-spike threshold as a multiple of the observed maximum for GPT/hybrid training when `--check-for-spiky-loss` is enabled. Default: `10.0`. |
| `--external-gpu-device-mapping` | Select visible CUDA device 0 when the launcher exposes one GPU per process. Disabled by default; otherwise device selection follows the local rank. |

`TrainingConfig.check_optimizer_step_success` defaults to `True`: the training
loop combines optimizer-step success across model-parallel ranks before deciding
whether to advance the learning-rate scheduler. Disable it only when the local
optimizer results are consistent across those ranks, otherwise scheduler state
can diverge.

### Learning Rate

| Argument | Description |
|----------|-------------|
| `--lr` | Peak learning rate |
| `--min-lr` | Minimum learning rate |
| `--lr-decay-style` | LR schedule (cosine, linear, constant) |
| `--lr-warmup-iters` | Warmup iterations |
| `--max-steps` | Full LR/weight-decay schedule horizon for a shortened iteration-based run. Overrides `--lr-decay-iters` without changing `--train-iters`; must be positive and at least `--train-iters`. Unset by default. |

For example, `--train-iters 100 --max-steps 10000` runs 100 iterations using a
10,000-iteration schedule. Fractional warmup follows the full LR horizon. This
option is not supported with sample-based training.

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
| `--split` | Train/validation/test split (for example, 949,50,1) |
| `--save` | Checkpoint save directory |
| `--load` | Checkpoint load directory |
| `--save-interval` | Save checkpoint every N iterations |

## Next Steps

- **Optimize Performance**: Refer to [Advanced Features](features/index.md) for FSDP, the distributed optimizer, and other optimizations
- **Scale Up**: Refer to [Parallelism Strategies](parallelism-guide.md) to train larger models across more GPUs
- **Prepare Data**: Follow the [Data Preparation](data-preparation.md) guide to process your own datasets
