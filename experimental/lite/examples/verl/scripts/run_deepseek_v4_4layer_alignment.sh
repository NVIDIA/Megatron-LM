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
export VLLM_BATCH_INVARIANT_KERNEL_LIB="${VLLM_BATCH_INVARIANT_KERNEL_LIB:-/opt/batch-invariant-kernel/_vllm_batch_invariant_C.so}"
test -s "${MODEL_PATH}/config.json"
if [[ ! -s "${VLLM_BATCH_INVARIANT_KERNEL_LIB}" ]]; then
  echo "required batch-invariant kernel is missing: ${VLLM_BATCH_INVARIANT_KERNEL_LIB}" >&2
  exit 2
fi

export OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/../outputs/ds4_4layer_alignment}"
export RUN_NAME="${RUN_NAME:-ds4_4layer_ep4_alignment}"
export LOG_FILE="${LOG_FILE:-${OUTPUT_ROOT}/${RUN_NAME}.log}"
export VERL_TRAIN_INFER_DIFF_DUMP="${VERL_TRAIN_INFER_DIFF_DUMP:-${OUTPUT_ROOT}/train_infer_tokens.jsonl}"

export NNODES=1
export NGPUS_PER_NODE=4
export ACTOR_PP=1
export ACTOR_CP=1
export ACTOR_EP=4
export ROLLOUT_TP=1
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-4}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-2048}"
export ROLLOUT_N="${ROLLOUT_N:-1}"
export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.55}"
export ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-2048}"
# Weight synchronization owns both an IPC buffer and a producer staging
# buffer. Keep each at 1 GiB in the deliberately dense PP1 four-GPU gate so a
# sleeping vLLM layer can be materialized without overlapping two 2-GiB
# transport buffers. This does not change model or kernel numerics.
export UPDATE_WEIGHTS_BUCKET_MEGABYTES="${UPDATE_WEIGHTS_BUCKET_MEGABYTES:-1024}"
export TOTAL_EPOCHS=1
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-1}"
export SAVE_FREQ=-1
export TEST_FREQ=-1
export RESUME_MODE=disable
export ENABLE_R3=False
export ROLLOUT_WEIGHT_BITS=8
export ROLLOUT_MOE_BACKEND=deep_gemm

export VLLM_BATCH_INVARIANT=1
export VERL_ACTOR_BATCH_INVARIANT=1
export VERL_ROLLOUT_BATCH_INVARIANT=1
export VLLM_DS4_DECODE_KERNEL=sparse
export VERL_FULL_DETERMINISM=1
export VERL_DETERMINISM_SEED="${SEED:-42}"
export PYTHONHASHSEED="${SEED:-42}"
export VERL_LOCAL_TASK_RUNNER=1
export VERL_ENGINE_LAZY_IMPORTS=1
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

# Keep this release gate free of tensor dumps, forced synchronization, and
# validation probes so its timings remain representative.
unset MLITE_CUDA_SYNC_BOUNDARIES MLITE_CUDA_SYNC_BOUNDARY_STAGES
unset CUDA_LAUNCH_BLOCKING VERL_TRAIN_INFER_RAW_DUMP
export MLITE_VALIDATE_FINITE=0
export MLITE_VALIDATE_INDICES=0
export MLITE_WEIGHT_SYNC_FINGERPRINT=0
export MLITE_WEIGHT_SYNC_PROBE=0
export VERL_TRAIN_INFER_DIFF_MODE=compact
export VERL_TRAIN_INFER_TOKEN_SAMPLE_LIMIT="${VERL_TRAIN_INFER_TOKEN_SAMPLE_LIMIT:-8}"

layers='[0,1,2,3]'
set +e
bash "${SCRIPT_DIR}/run_deepseek_v4_dapo.sh" \
  "data.seed=${SEED:-42}" \
  "actor_rollout_ref.actor.engine.impl=vllm" \
  "+actor_rollout_ref.actor.engine.seed=${SEED:-42}" \
  "+actor_rollout_ref.actor.engine.full_determinism=True" \
  "actor_rollout_ref.actor.engine.attention_backend_override=null" \
  "++actor_rollout_ref.actor.engine.impl_cfg.deterministic=True" \
  "++actor_rollout_ref.actor.engine.impl_cfg.use_thd=False" \
  "++actor_rollout_ref.actor.engine.impl_cfg.use_deepep=True" \
  "++actor_rollout_ref.actor.engine.impl_cfg.max_tokens_per_rank=4096" \
  "++actor_rollout_ref.actor.engine.impl_cfg.mtp_enable=False" \
  "++actor_rollout_ref.actor.engine.impl_cfg.mtp_enable_train=False" \
  "++actor_rollout_ref.actor.engine.impl_cfg.qat=null" \
  "++actor_rollout_ref.actor.engine.impl_cfg.selector.global_layer_ids=${layers}" \
  "++actor_rollout_ref.actor.engine.impl_cfg.selector.module_names=[mhc,linear,kv_flashmla,o_proj,router_moe,deepep]" \
  "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096" \
  "actor_rollout_ref.rollout.full_determinism=True" \
  "actor_rollout_ref.rollout.seed=${SEED:-42}" \
  "actor_rollout_ref.rollout.data_parallel_size=4" \
  "actor_rollout_ref.rollout.expert_parallel_size=4" \
  "actor_rollout_ref.rollout.agent.num_workers=4" \
  "actor_rollout_ref.rollout.max_num_seqs=4" \
  "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=4096" \
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

# Gate on VERL's production train/infer metrics rather than a separate
# comparator. Every completed old-log-prob stage must be byte-exact and have
# zero K3 KL; absence of the metric is also a failure.
python3 - "${LOG_FILE}" <<'PY'
import re
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
pattern = re.compile(
    r"RL_STAGE old_log_prob_done .*?bitwise_fraction=([^\s\x1b]+) "
    r"k3_kl=([^\s\x1b]+)"
)
metrics = [(float(a), float(b)) for a, b in pattern.findall(log_path.read_text())]
if not metrics:
    raise SystemExit(f"no VERL train/infer alignment metric found in {log_path}")
bad = [(index + 1, bitwise, k3) for index, (bitwise, k3) in enumerate(metrics)
       if bitwise != 1.0 or k3 != 0.0]
if bad:
    raise SystemExit(f"DS4 train/infer alignment gate failed: {bad}")
print(f"DS4_4L_ALIGNMENT_EXACT steps={len(metrics)} bitwise_fraction=1.0 k3_kl=0.0")
PY
