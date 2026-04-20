# Attention Residuals Examples

This directory contains small, reproducible launch examples for the Megatron
Core Attention Residuals implementation.

The examples assume that tokenizer and preprocessed data already exist. They do
not download model weights or datasets.

## Required Inputs

Set these paths before launching:

```bash
export TOKENIZER_MODEL=/path/to/llama3-tokenizer
export DATA_PREFIX=/path/to/megatron_text_document
```

`DATA_PREFIX` should point to a Megatron indexed dataset prefix, without the
`.bin` or `.idx` suffix.

## Baseline

```bash
WANDB_PROJECT=attention-residuals \
WANDB_EXP_NAME=baseline_1000steps \
TOKENIZER_MODEL=$TOKENIZER_MODEL \
DATA_PREFIX=$DATA_PREFIX \
TRAIN_ITERS=1000 \
LR_DECAY_ITERS=10000 \
LR_WARMUP_ITERS=100 \
./examples/attention_residuals/train_llama3_wikitext.sh baseline
```

## Full AttnRes

```bash
WANDB_PROJECT=attention-residuals \
WANDB_EXP_NAME=full_triton_bwd_1000steps \
TOKENIZER_MODEL=$TOKENIZER_MODEL \
DATA_PREFIX=$DATA_PREFIX \
TRAIN_ITERS=1000 \
LR_DECAY_ITERS=10000 \
LR_WARMUP_ITERS=100 \
ATTENTION_RESIDUAL_TYPE=full \
ATTENTION_RESIDUAL_IMPLEMENTATION=triton_bwd \
./examples/attention_residuals/train_llama3_wikitext.sh attnres
```

## Block AttnRes

```bash
WANDB_PROJECT=attention-residuals \
WANDB_EXP_NAME=block_n8_triton_bwd_1000steps \
TOKENIZER_MODEL=$TOKENIZER_MODEL \
DATA_PREFIX=$DATA_PREFIX \
TRAIN_ITERS=1000 \
LR_DECAY_ITERS=10000 \
LR_WARMUP_ITERS=100 \
ATTENTION_RESIDUAL_TYPE=block \
ATTENTION_RESIDUAL_NUM_BLOCKS=8 \
ATTENTION_RESIDUAL_IMPLEMENTATION=triton_bwd \
./examples/attention_residuals/train_llama3_wikitext.sh attnres
```

## Common Knobs

```bash
NUM_LAYERS=16
SEQ_LENGTH=1024
MICRO_BATCH_SIZE=8
GLOBAL_BATCH_SIZE=32
TP_SIZE=2
PP_SIZE=1
CP_SIZE=1
```

Use `EXIT_DURATION_IN_MINS=30` for wall-clock-limited experiments. Leave it
unset for step-limited experiments controlled by `TRAIN_ITERS`.

## Implementation Modes

```bash
ATTENTION_RESIDUAL_IMPLEMENTATION=torch
ATTENTION_RESIDUAL_IMPLEMENTATION=checkpointed
ATTENTION_RESIDUAL_IMPLEMENTATION=triton
ATTENTION_RESIDUAL_IMPLEMENTATION=triton_bwd
```

`triton_bwd` is the recommended mode when Triton is available.

## Limitations

The current implementation is intended for dense decoder-only GPT/Llama-style
models. MoE, cross-attention, pipeline parallelism greater than one stage, and
inference paths are not yet supported.
