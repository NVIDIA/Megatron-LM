#!/usr/bin/env bash
# Four-GPU release gate for the four-layer DeepSeek-V4 alignment checkpoint.
# The actor uses normal DeepEP; rollout uses vLLM low-latency DeepEP. Both use
# the required batch-invariant expert kernel and EP4.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -L)"

# The published alignment image installs its coherent runtime in this venv.
# Prefer it when present so invoking this script directly does not accidentally
# pick up packages from the base image or the user's site directory.
if [[ -x /opt/ds4-venv/bin/python3 ]]; then
  export PATH="/opt/ds4-venv/bin:${PATH}"
fi
export PYTHONNOUSERSITE=1

: "${MODEL_PATH:?set MODEL_PATH to the four-layer DeepSeek-V4 checkpoint}"
: "${TRAIN_FILES:?set TRAIN_FILES to DAPO-format training parquet}"
: "${VAL_FILES:?set VAL_FILES to DAPO-format validation parquet}"
test -s "${MODEL_PATH}/config.json"

export OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/../outputs/ds4_4layer_alignment}"
export RUN_NAME="${RUN_NAME:-ds4_4layer_ep4_alignment}"
export LOG_FILE="${LOG_FILE:-${OUTPUT_ROOT}/${RUN_NAME}.log}"
export JSONL_FILE="${JSONL_FILE:-${OUTPUT_ROOT}/${RUN_NAME}.jsonl}"

export NNODES="${NNODES:-1}"
export NGPUS_PER_NODE="${NGPUS_PER_NODE:-4}"
export ACTOR_PP="${ACTOR_PP:-1}"
export ACTOR_CP="${ACTOR_CP:-1}"
export ACTOR_EP="${ACTOR_EP:-4}"
export ROLLOUT_TP="${ROLLOUT_TP:-1}"
export ROLLOUT_DP="${ROLLOUT_DP:-${NGPUS_PER_NODE}}"
export ROLLOUT_EP="${ROLLOUT_EP:-${NGPUS_PER_NODE}}"
export ROLLOUT_AGENT_WORKERS="${ROLLOUT_AGENT_WORKERS:-${ROLLOUT_DP}}"
export ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-${ROLLOUT_DP}}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-4}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-2048}"
export PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-4096}"
export LOGPROB_MAX_TOKEN_LEN_PER_GPU="${LOGPROB_MAX_TOKEN_LEN_PER_GPU:-4096}"
export ROLLOUT_N="${ROLLOUT_N:-1}"
export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.55}"
export ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-2048}"
# Weight synchronization owns both an IPC buffer and a producer staging
# buffer. Keep each at 1 GiB in the deliberately dense PP1 four-GPU gate so a
# sleeping vLLM layer can be materialized without overlapping two 2-GiB
# transport buffers. This does not change model or kernel numerics.
export UPDATE_WEIGHTS_BUCKET_MEGABYTES="${UPDATE_WEIGHTS_BUCKET_MEGABYTES:-1024}"
export TOTAL_EPOCHS=1
# Step two is part of the correctness contract: it exercises optimizer update,
# actor export, rollout layerwise reload, and a second old-logprob comparison.
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-2}"
export SAVE_FREQ=-1
export TEST_FREQ=-1
export RESUME_MODE=disable
export ENABLE_R3=False
export ROLLOUT_WEIGHT_BITS=8
export ROLLOUT_MOE_BACKEND=deep_gemm

export VLLM_BATCH_INVARIANT=1
export VLLM_DS4_DECODE_KERNEL=sparse
export VERL_FULL_DETERMINISM=1
export PYTHONHASHSEED="${SEED:-42}"
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1

# Dockerfile.1 provides the validated torch-2.12/dev631 runtime. Keep the
# dependency check enabled so a direct invocation cannot silently use another
# vLLM, CUTLASS, or CUDA closure.
export VALIDATE_DS4_ENVIRONMENT="${VALIDATE_DS4_ENVIRONMENT:-1}"

# This is an NVIDIA-only recipe. Some Slurm/Pyxis versions export the ROCm
# visibility aliases alongside CUDA_VISIBLE_DEVICES; VERL correctly rejects
# that ambiguous pair. Remove the inapplicable aliases before Ray snapshots
# the worker environment.
unset HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES

set +e
bash "${SCRIPT_DIR}/run_deepseek_v4_dapo.sh" \
  "data.seed=${SEED:-42}" \
  "actor_rollout_ref.actor.engine.impl=vllm" \
  "+actor_rollout_ref.actor.engine.seed=${SEED:-42}" \
  "+actor_rollout_ref.actor.engine.full_determinism=True" \
  "actor_rollout_ref.actor.engine.attention_backend_override=null" \
  "++actor_rollout_ref.actor.engine.impl_cfg.deterministic=True" \
  "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}" \
  "actor_rollout_ref.rollout.full_determinism=True" \
  "actor_rollout_ref.rollout.seed=${SEED:-42}" \
  "actor_rollout_ref.rollout.data_parallel_size=${ROLLOUT_DP}" \
  "actor_rollout_ref.rollout.expert_parallel_size=${ROLLOUT_EP}" \
  "actor_rollout_ref.rollout.agent.num_workers=${ROLLOUT_AGENT_WORKERS}" \
  "actor_rollout_ref.rollout.max_num_seqs=${ROLLOUT_MAX_NUM_SEQS}" \
  "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${LOGPROB_MAX_TOKEN_LEN_PER_GPU}" \
  "actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=${UPDATE_WEIGHTS_BUCKET_MEGABYTES}" \
  "+actor_rollout_ref.rollout.engine_kwargs.vllm.all2all_backend=deepep_low_latency" \
  "+actor_rollout_ref.rollout.engine_kwargs.vllm.linear_backend=deep_gemm" \
  "actor_rollout_ref.rollout.engine_kwargs.vllm.moe_backend=deep_gemm" \
  "trainer.use_v1=False" \
  "trainer.logger=[console,file]" \
  "$@"
run_rc=$?
set -e

if (( run_rc != 0 )); then
  exit "${run_rc}"
fi
if [[ "${DRY_RUN:-0}" == "1" || "${COMPOSE_ONLY:-0}" == "1" ]]; then
  exit 0
fi

# Gate directly on VERL's file logger; no diagnostic dump or custom metric is
# required from VERL.
python3 "$(dirname "$0")/validate_alignment_metrics.py" \
  "${JSONL_FILE}" "${TOTAL_TRAINING_STEPS}"
