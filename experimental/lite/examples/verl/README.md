# VERL Megatron Lite Example

This directory contains a runnable VERL external engine integration for
Megatron Lite plus Qwen3.5-35B-A3B SFT and GRPO launch scripts.

The Python package is `verl_mlite`. It registers VERL's language-model engine
backend as `mlite`, while Megatron Lite model implementations still use
`impl=lite`.

## Layout

- `verl_mlite/engine/mlite_engine.py`: VERL `BaseEngine` implementation backed
  by `megatron.lite.runtime`.
- `verl_mlite/config/engine/mlite.yaml`: Hydra engine config for
  `engine=mlite`.
- `scripts/run_qwen3moe_sft.sh`: Qwen MoE SFT launcher using
  `verl.trainer.sft_trainer`.
- `scripts/run_qwen3moe_gsm8k_sft.sh`: GSM8K wrapper around the SFT launcher.
- `scripts/run_qwen3moe_gsm8k_grpo.sh`: GSM8K GRPO launcher with MLite actor
  training and a standard VERL rollout backend.

## Prerequisites

Install or expose these packages before running:

- VERL with the new engine worker path.
  See [`REQUIRED_VERL.txt`](REQUIRED_VERL.txt) for the reference upstream
  release tag.
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
export MODEL_PATH=/path/to/qwen3.5-35b-a3b-hf
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

FSDP2 supports two offload modes. `PARAM_OFFLOAD=True` and
`OPTIMIZER_OFFLOAD=True` move model parameters and optimizer state between CPU
and GPU when VERL switches execution contexts. `OPTIMIZER_OFFLOAD=True` also
sets `optim.override_optimizer_config.offload_fraction=1.0` by default, which
keeps FSDP2 optimizer update state on CPU during forward/backward to reduce GPU
memory pressure.

Example dry run:

```bash
MODEL_PATH=/path/to/qwen3.5-35b-a3b-hf \
TRAIN_FILES=/path/to/train.parquet \
DRY_RUN=1 \
bash experimental/lite/examples/verl/scripts/run_qwen3moe_sft.sh
```

By default, logs, command snapshots, JSONL logger output, and checkpoints are
written under `experimental/lite/examples/verl/outputs/qwen3moe_sft`. Override
`OUTPUT_ROOT`, `LOG_FILE`, `JSONL_FILE`, `CMD_FILE`, or `CKPT_DIR` to redirect
artifacts.

For local dry runs, prefer a temporary output directory if you do not want
command snapshots under the source tree:

```bash
OUTPUT_ROOT="$(mktemp -d)" \
MODEL_PATH=/path/to/qwen3.5-35b-a3b-hf \
TRAIN_FILES=/path/to/train.parquet \
DRY_RUN=1 \
bash experimental/lite/examples/verl/scripts/run_qwen3moe_sft.sh
```

## GSM8K SFT

Build messages-format GSM8K parquet files with VERL's SFT preprocessor:

```bash
python3 /path/to/verl/examples/data_preprocess/gsm8k_multiturn_sft.py \
  --local_save_dir ~/data/gsm8k_sft
```

Run the MLite GSM8K SFT wrapper:

```bash
MODEL_PATH=Qwen/Qwen3.5-35B-A3B \
DRY_RUN=1 \
bash experimental/lite/examples/verl/scripts/run_qwen3moe_gsm8k_sft.sh
```

The wrapper defaults to `Qwen/Qwen3.5-35B-A3B`,
`~/data/gsm8k_sft/train.parquet`, and
`~/data/gsm8k_sft/test.parquet`, then delegates to
`scripts/run_qwen3moe_sft.sh`. Override `DATASET_DIR`, `TRAIN_FILES`, or
`VAL_FILES` to use another location.

By default, GSM8K SFT artifacts are written under
`experimental/lite/examples/verl/outputs/qwen35_gsm8k_sft`.

## GSM8K GRPO

Build RL-format GSM8K parquet files with VERL's GRPO/PPO preprocessor:

```bash
python3 /path/to/verl/examples/data_preprocess/gsm8k.py \
  --local_save_dir ~/data/gsm8k
```

Run GRPO with the MLite actor and vLLM rollout:

```bash
MODEL_PATH=Qwen/Qwen3.5-35B-A3B \
DRY_RUN=1 \
bash experimental/lite/examples/verl/scripts/run_qwen3moe_gsm8k_grpo.sh
```

Useful GRPO knobs:

- `TRAIN_BATCH_SIZE`, `PPO_MINI_BATCH_SIZE`,
  `ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU`
- `MAX_PROMPT_LENGTH`, `MAX_RESPONSE_LENGTH`, `PPO_MAX_TOKEN_LEN_PER_GPU`
- `ROLLOUT_N`, `ROLLOUT_TP`, `ROLLOUT_GPU_MEMORY_UTILIZATION`
- `ACTOR_TP`, `ACTOR_PP`, `ACTOR_VPP`, `ACTOR_CP`, `ACTOR_EP`, `ACTOR_ETP`
- `PARAM_OFFLOAD`, `OPTIMIZER_OFFLOAD`, `GRAD_OFFLOAD`
- `INFER_BACKEND=vllm`

The GRPO launcher keeps the reference policy disabled by default
(`algorithm.use_kl_in_reward=False`, `actor_rollout_ref.actor.use_kl_loss=False`)
so the example exercises the current MLite actor path without expanding scope
to a separate reference model.

By default, GSM8K GRPO artifacts are written under
`experimental/lite/examples/verl/outputs/qwen35_gsm8k_grpo`.

## Smoke / Dry-Run Checks

Checked on this branch on 2026-06-07. These checks cover shell syntax,
Python import compilation, and resolved command construction only; they do not
cover end-to-end SFT or GRPO training.

- Shell syntax:
  - `bash -n experimental/lite/examples/verl/scripts/run_qwen3moe_sft.sh`
  - `bash -n experimental/lite/examples/verl/scripts/run_qwen3moe_gsm8k_sft.sh`
  - `bash -n experimental/lite/examples/verl/scripts/run_qwen3moe_gsm8k_grpo.sh`
- Python import compilation:
  - `PYTHONPYCACHEPREFIX="$(mktemp -d)" python3 -m compileall -q experimental/lite/examples/verl/verl_mlite`
- GSM8K SFT dry run:
  - `OUTPUT_ROOT="$(mktemp -d)" MODEL_PATH=Qwen/Qwen3.5-35B-A3B DRY_RUN=1 bash experimental/lite/examples/verl/scripts/run_qwen3moe_gsm8k_sft.sh`
  - Dry-run output shows `torchrun -m verl.trainer.sft_trainer`,
    `engine=mlite`, `model.path=Qwen/Qwen3.5-35B-A3B`,
    `data.train_files=${HOME}/data/gsm8k_sft/train.parquet`, and
    `data.val_files=${HOME}/data/gsm8k_sft/test.parquet`.
- GSM8K GRPO dry run:
  - `OUTPUT_ROOT="$(mktemp -d)" MODEL_PATH=Qwen/Qwen3.5-35B-A3B DRY_RUN=1 bash experimental/lite/examples/verl/scripts/run_qwen3moe_gsm8k_grpo.sh`
  - Dry-run output shows `python3 -m verl.trainer.main_ppo`,
    `actor@actor_rollout_ref.actor=mlite_actor`,
    `actor_rollout_ref.rollout.name=vllm`,
    `actor_rollout_ref.actor.engine.impl=lite`,
    `actor_rollout_ref.actor.engine.ep=8`,
    `algorithm.adv_estimator=grpo`, and `critic.enable=False`.
