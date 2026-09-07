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
- `scripts/run_deepseek_v4_4layer_alignment.sh`: four-GPU DeepSeek-V4 gate with
  EP4 mLite training and EP4 deterministic vLLM rollout.

## DeepSeek-V4 four-layer alignment gate

The release gate runs one DAPO step by default and records VERL's compact
train/inference consistency metrics. It requires the standalone
batch-invariant kernel and fails before model execution when the library is
missing or incompatible.

```bash
MODEL_PATH=/models/DeepSeek-V4-Flash-4L \
TRAIN_FILES=/data/dapo-train.parquet \
VAL_FILES=/data/dapo-val.parquet \
bash experimental/lite/examples/verl/scripts/run_deepseek_v4_4layer_alignment.sh
```

The release image installs the batch-invariant extension in the vLLM package;
`VLLM_BATCH_INVARIANT_KERNEL_LIB` is only an explicit development override.

## Prerequisites

Install or expose these packages before running:

- VERL with the new engine worker path.
  See [`REQUIRED_VERL.txt`](REQUIRED_VERL.txt) for the reference upstream
  source pin (commit).
- Megatron-LM from this repository, or another source tree via
  `MEGATRON_ROOT=/path/to/Megatron-LM`.
- Megatron Lite from this repository. The script automatically adds
  `experimental/lite` to `PYTHONPATH`.
- The examples directory is also added to `PYTHONPATH` and loads a local
  compatibility hook for known VERL/vLLM/Transformers dependency gaps.

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
- `DYNAMIC_CONTEXT_PARALLEL=True`, `MAX_SEQLEN_PER_DP_CP_RANK`, and
  `MIN_DYNAMIC_CONTEXT_PARALLEL_SIZE` enable runtime-owned dynamic CP scheduling.
  The SFT launcher sets `REQUIRE_FULL_CP_SIZE_COVERAGE=True` by default so an
  acceptance run fails unless every feasible CP size is scheduled; set it to
  `False` only for a workload that intentionally cannot cover the full range.
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

Dynamic CP keeps the same VERL engine API and is an explicitly enabled MLite
runtime plugin. It is disabled by default; enabling it also requires
`MAX_SEQLEN_PER_DP_CP_RANK`:

```bash
MODEL_PATH=/path/to/qwen3.5-35b-a3b-hf \
TRAIN_FILES=/path/to/train.parquet \
NUM_GPUS=4 TP_SIZE=1 CP_SIZE=1 EP_SIZE=1 \
DYNAMIC_CONTEXT_PARALLEL=True \
MAX_SEQLEN_PER_DP_CP_RANK=4096 \
REQUIRE_FULL_CP_SIZE_COVERAGE=True \
bash experimental/lite/examples/verl/scripts/run_qwen3moe_sft.sh
```

Dynamic CP uses the physical DP×CP pool to normalize the training loss, while
VERL's `batch_num_tokens` and loss logging retain their logical-DP view.  Do
not compare those telemetry values as though they were the training-loss
normalization denominator.

Ordinary pipeline parallelism and R2/R3 router replay are supported. Virtual
pipeline parallelism remains unsupported and fails loudly. R2/R3 also require
`moe_router_fusion=False`, because the fused router path bypasses the replay
hook; this restriction does not apply when router replay is disabled.

Dynamic CP is not unconditionally faster. Its benefit depends on a sequence
length distribution that lets the scheduler use smaller CP groups for enough
microbatches while keeping all DP×CP ranks busy. Decide with an A/B run that
keeps the checkpoint, samples, token budget, and parallel topology fixed; after
warm-up, compare both processed tokens per second and the emitted
`cp_size_histogram`. Leave it disabled when the histogram stays concentrated at
the full static CP size or throughput does not improve.

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
- `ROLLOUT_MODE=async`, `ROLLOUT_MAX_MODEL_LEN`, `ROLLOUT_MAX_NUM_BATCHED_TOKENS`
- `ROLLOUT_LIMIT_IMAGES=0`, `ROLLOUT_LIMIT_VIDEOS=0` keep the vLLM rollout
  backend in text-only mode for GSM8K by default.
- `ACTOR_TP`, `ACTOR_PP`, `ACTOR_VPP`, `ACTOR_CP`, `ACTOR_EP`, `ACTOR_ETP`
- `PARAM_OFFLOAD`, `OPTIMIZER_OFFLOAD`, `GRAD_OFFLOAD`
- `INFER_BACKEND=vllm`
- `POLICY_LOSS_MODE=vanilla` and `LOSS_AGG_MODE=seq-mean-token-sum-norm`
  select the pure GRPO baseline policy loss and aggregation mode.

The GRPO launcher keeps the reference policy disabled by default
(`algorithm.use_kl_in_reward=False`, `actor_rollout_ref.actor.use_kl_loss=False`)
so the example exercises the current MLite actor path without expanding scope
to a separate reference model. On latest verl, both the v0 and V1 trainer paths
route `actor@actor_rollout_ref.actor=mlite_actor` to the unified engine workers,
so the MLite actor is wired up correctly without any extra worker-path knob.

By default, GSM8K GRPO artifacts are written under
`experimental/lite/examples/verl/outputs/qwen35_gsm8k_grpo`.

### MXFP4 QAT four-arm launch

For the complete `QATSpec` field reference, supported-format table,
optimizer/checkpoint ordering contract, packed snapshot layout, and an
end-to-end MXFP4 QAT launch recipe, see [QAT.md](QAT.md).

Run the generalized four-arm Qwen3-MoE recipe with:

```bash
MODEL_PATH=Qwen/Qwen3-30B-A3B \
TRAIN_FILES=/path/to/dapo-math-17k.parquet \
VAL_FILES=/path/to/aime-2024.parquet \
MXFP4_QUANTIZATION_CONFIG=/path/to/mxfp4_w4a16.json \
bash experimental/lite/examples/verl/scripts/run_qwen3moe_mxfp4_qat.sh \
  --mode qat_on
```

Modes are `baseline`, `qat_off`, `qat_on`, and `r3`. The `qat_off` and
`qat_on` arms deliberately keep rollout MXFP4 identical and change only
training-side `impl_cfg.qat.enabled`. This recipe uses vLLM
compressed-tensors plus `verl.utils.qat.vllm_patch`, with paired
`actor.engine.qat` export and `rollout.qat` configuration. See
[QAT.md](QAT.md) for the exact arm semantics, validation scope, safe
exclusions, and required MXFP4 JSON schema.

### DeepSeek V4 MXFP4 QAT A/B

The DeepSeek V4 DAPO launcher exposes the same controlled training-side QAT
decision without replacing its model-specific resync path. Keep every input,
seed, topology, and rollout setting fixed, and run the two arms with only
`ENABLE_QAT` changed:

```bash
MODEL_PATH=/path/to/deepseek-v4-proxy \
TRAIN_FILES=/path/to/dapo-math-17k.parquet \
VAL_FILES=/path/to/aime-2024.parquet \
NNODES=1 NGPUS_PER_NODE=8 \
ACTOR_PP=2 ACTOR_CP=2 ACTOR_EP=2 ROLLOUT_TP=8 \
ROLLOUT_WEIGHT_BITS=4 ENABLE_R3=False ENABLE_QAT=False \
bash experimental/lite/examples/verl/scripts/run_deepseek_v4_dapo.sh

MODEL_PATH=/path/to/deepseek-v4-proxy \
TRAIN_FILES=/path/to/dapo-math-17k.parquet \
VAL_FILES=/path/to/aime-2024.parquet \
NNODES=1 NGPUS_PER_NODE=8 \
ACTOR_PP=2 ACTOR_CP=2 ACTOR_EP=2 ROLLOUT_TP=8 \
ROLLOUT_WEIGHT_BITS=4 ENABLE_R3=False ENABLE_QAT=True \
bash experimental/lite/examples/verl/scripts/run_deepseek_v4_dapo.sh
```

`ENABLE_QAT=True` registers MLite MXFP4 fake quantization on the BF16 master
weights before optimizer construction. The launcher rejects that setting unless
the rollout also uses MXFP4 (`ROLLOUT_WEIGHT_BITS=4`); it never silently pairs
MXFP4 training with FP8 rollout. The off/on arms have byte-identical rollout
arguments, and both enable rollout log-prob calculation plus rollout correction,
so compare `rollout_probs_diff` metrics as the primary train/inference
consistency signal. Reward is secondary.

An 8-GPU truncated proxy exercises the real DeepSeek V4 protocol, QAT
parametrization, checkpoint load, MXFP4 export/resync, vLLM rollout, backward,
and optimizer step. It does not establish full-scale memory capacity or
throughput for the 43-layer, 256-expert release.

## Smoke / Dry-Run Checks

Checked on this branch on 2026-06-07. These checks cover shell syntax,
Python import compilation, and resolved command construction only; they do not
cover end-to-end SFT or GRPO training.

- Shell syntax:
  - `bash -n experimental/lite/examples/verl/scripts/run_qwen3moe_sft.sh`
  - `bash -n experimental/lite/examples/verl/scripts/run_qwen3moe_gsm8k_sft.sh`
  - `bash -n experimental/lite/examples/verl/scripts/run_qwen3moe_gsm8k_grpo.sh`
  - `bash -n experimental/lite/examples/verl/scripts/run_qwen3moe_mxfp4_qat.sh`
  - `bash -n experimental/lite/examples/verl/scripts/run_deepseek_v4_dapo.sh`
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
    `algorithm.adv_estimator=grpo`, `actor_rollout_ref.actor.policy_loss.loss_mode=vanilla`,
    and `critic.enable=False`.
