# VERL Megatron Lite Example

This directory contains a runnable VERL external engine integration for
Megatron Lite plus a Qwen3-MoE SFT launch script.

The Python package is `verl_mlite`. It registers VERL's language-model engine
backend as `mlite`, while Megatron Lite model implementations still use
`impl=lite`.

## Layout

- `verl_mlite/engine/mlite_engine.py`: VERL `BaseEngine` implementation backed
  by `megatron.lite.runtime`.
- `verl_mlite/config/engine/mlite.yaml`: Hydra engine config for
  `engine=mlite`.
- `scripts/run_qwen3moe_sft.sh`: Qwen3-MoE SFT launcher using
  `verl.trainer.sft_trainer`.

## Prerequisites

Install or expose these packages before running:

- VERL with the new engine worker path.
  See [`REQUIRED_VERL.txt`](REQUIRED_VERL.txt) for the reference upstream
  commit.
- Megatron-LM from this repository, or another source tree via
  `MEGATRON_ROOT=/path/to/Megatron-LM`.
- Megatron Lite from this repository. The script automatically adds
  `experimental/lite` to `PYTHONPATH`.

Optional source-tree override:

```bash
export VERL_ROOT=/path/to/verl
export MEGATRON_ROOT=/path/to/Megatron-LM
```

## SFT

The SFT script expects VERL messages-format parquet input.

```bash
export MODEL_PATH=/path/to/qwen3-moe-hf
export TRAIN_FILES=/path/to/train.parquet
export VAL_FILES=/path/to/val.parquet

bash experimental/lite/examples/verl/scripts/run_qwen3moe_sft.sh
```

Useful knobs:

- `TP_SIZE`, `PP_SIZE`, `VPP_SIZE`, `CP_SIZE`, `EP_SIZE`, `ETP_SIZE`
- `TOTAL_STEPS`, `TOTAL_EPOCHS`, `TRAIN_BATCH_SIZE`, `MICRO_BATCH_SIZE`
- `MAX_TOKENS_PER_GPU`, `MAX_LENGTH`, `MESSAGES_KEY`
- `PARAM_OFFLOAD`, `OPTIMIZER_OFFLOAD`, `GRAD_OFFLOAD`
- `MLITE_MODEL_NAME=auto`, `MLITE_IMPL=lite`
- `ATTENTION_BACKEND=flash`
- `DRY_RUN=1` to print the resolved `torchrun` command without launching

For the FSDP2 optimizer primitive, keep `PARAM_OFFLOAD=False`; optimizer
offload is supported through `OPTIMIZER_OFFLOAD=True`.

Example dry run:

```bash
MODEL_PATH=/path/to/qwen3-moe-hf \
TRAIN_FILES=/path/to/train.parquet \
DRY_RUN=1 \
bash experimental/lite/examples/verl/scripts/run_qwen3moe_sft.sh
```

By default, logs, command snapshots, JSONL logger output, and checkpoints are
written under `experimental/lite/examples/verl/outputs/qwen3moe_sft`. Override
`OUTPUT_ROOT`, `LOG_FILE`, `JSONL_FILE`, `CMD_FILE`, or `CKPT_DIR` to redirect
artifacts.
