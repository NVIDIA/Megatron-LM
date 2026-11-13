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

- `cfg.data.dataset.sequence_length` limits each GPT/SFT sequence's content. The CLI
  supplies it with `--seq-length`.
- `cfg.model.transformer.max_seq_length_per_sample` bounds one physical model input
  row, or an entire packed THD row, including padding and expanded image tokens.
  `--max-seq-length-per-sample` sets it explicitly; GPT defaults to `--seq-length`.
- `cfg.model.seq_length` is the model's position capacity (`--max-position-embeddings`),
  not a dataset truncation limit. RoPE settings are unchanged.
- `max_seqlen_per_dp_cp_rank` remains the separate per-rank dynamic-CP packing budget.

For example, `--sft --seq-length 2048 --max-seq-length-per-sample 8192` truncates
individual conversations at 2048 tokens and emits 8192-token padded packs. The
container derives `SFTDatasetConfig.packed_sequence_length` during startup;
recipes should set the model bound rather than this derived dataset field. Existing
`--seq-length`-only runs retain their previous widths and packing behavior.
The SFT-specific config lives alongside `SFTDataset`; Core's `GPTDatasetConfig`
has no pack-width field. Legacy SFT callers supplying a plain `GPTDatasetConfig`
retain its single `sequence_length` as the output pack width.

An oversized allocation bound does not enlarge fixed GPT dataset rows or change
their token-based batch schedule. T5 retains separate encoder and decoder input
lengths. Callback datasets such as Energon still own truncation and packing; their
entrypoints bind those inputs before `pretrain()`, and resolve the model execution
bound before runtime buffers are initialized. Fixed-width BERT, mock VLM, and
ModelOpt callbacks reject bounds inconsistent with the rows they actually emit.

`validate()` checks these constraints without mutation. Framework startup calls
`finalize()` to derive unset/owned fields before any runtime consumer reads them.

### Python API compatibility

`LLaVAModelConfig.decoder_seq_length` is removed. Python callers and saved
configurations using this field require updating; strict deserialization rejects
the removed LLaVA field. For LLaVA, move the resolved text + image + padding width
to `model.transformer.max_seq_length_per_sample` before startup. Do not substitute
the allocation bound for a fixed GPT dataset's actual row width.

The `--decoder-seq-length` CLI flag remains accepted: VLM entrypoints translate
it into their execution width, while T5 retains an independent decoder width.
Decoder-only models do not acquire a second decoder-length config field.

`megatron.core.pipeline_parallel.schedules.get_tensor_shapes()` and all three
forward/backward schedules use one resolved `seq_length`. Their optional
`decoder_seq_length` argument is deprecated and will be removed in the 0.22 release.
For compatibility, a non-None value still overrides `seq_length` in the shape helper
and non-interleaved schedule; the no-pipeline and interleaved schedules continue
to ignore it. Supplying a non-None value emits a `FutureWarning` on the logging ranks.
Callers should select the pipeline width before calling them, including when
forwarding keywords through ModelOpt's distillation shape adapter. For VLMs, pass
the expanded LM input width. When migrating an ignored argument, simply remove it.
T5 retains separate model/dataset encoder and decoder lengths; its entrypoint
does not support pipeline parallelism. External callers (including Bridge) should
stop passing the deprecated keyword before upgrading to 0.22.

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
| `--split` | Train/validation/test split (for example, 949,50,1) |
| `--save` | Checkpoint save directory |
| `--load` | Checkpoint load directory |
| `--save-interval` | Save checkpoint every N iterations |

## Next Steps

- **Optimize Performance**: Refer to [Advanced Features](features/index.md) for FSDP, the distributed optimizer, and other optimizations
- **Scale Up**: Refer to [Parallelism Strategies](parallelism-guide.md) to train larger models across more GPUs
- **Prepare Data**: Follow the [Data Preparation](data-preparation.md) guide to process your own datasets
