#!/usr/bin/env bash
# DeepSeek-V4 DAPO launcher.
#
# Validated hero dependency contract (CW H100, 2026-07-17):
#   VERL: 6a937b63
#   Python: 3.12
#   PyTorch: 2.12.0a0 nv26.05, CUDA 13.2
#   vLLM: 0.25.1
#   Transformer Engine: >= 2.15.0
#   nvidia-cudnn-frontend: >= 1.27.0, with DSA q_causal_offsets
#   FlashInfer: 0.6.13
#   nvidia-cutlass-dsl: 4.5.2
#   TileLang: 0.1.9
#
# validate_deepseek_v4_dapo.py enforces this contract before a real run.
set -euo pipefail

# ---------------------------------------------------------------------------
# Key experiment knobs
# ---------------------------------------------------------------------------

# Rollout routed-expert weights: choose 4 (MXFP4 + Marlin) or
# 8 (FP8 + FlashInfer CUTLASS). Dense rollout weights remain FP8 in both modes.
ROLLOUT_WEIGHT_BITS="${ROLLOUT_WEIGHT_BITS:-8}"

# Router replay (R3): record vLLM's routed-expert choices and replay them in
# the actor when recomputing log-probs. Disable only for parity/debug A/B runs.
ENABLE_R3="${ENABLE_R3:-True}"

# ---------------------------------------------------------------------------
# Run inputs and geometry
# ---------------------------------------------------------------------------

: "${MODEL_PATH:?set MODEL_PATH to the official DeepSeek-V4 checkpoint}"
DAPO_DATA_DIR="${DAPO_DATA_DIR:-}"
TRAIN_FILES="${TRAIN_FILES:-${DAPO_DATA_DIR:+${DAPO_DATA_DIR}/dapo-math-17k.parquet}}"
VAL_FILES="${VAL_FILES:-${DAPO_DATA_DIR:+${DAPO_DATA_DIR}/aime-2024.parquet}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"

NNODES="${NNODES:-1}"
NGPUS_PER_NODE="${NGPUS_PER_NODE:-${NPROC_PER_NODE:-8}}"
ACTOR_PP="${ACTOR_PP:-4}"
ACTOR_CP="${ACTOR_CP:-4}"
ACTOR_EP="${ACTOR_EP:-8}"
ROLLOUT_TP="${ROLLOUT_TP:-8}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-32}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-6144}"
ROLLOUT_N="${ROLLOUT_N:-8}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.60}"

PARAM_OFFLOAD="${PARAM_OFFLOAD:-True}"
OPTIMIZER_OFFLOAD="${OPTIMIZER_OFFLOAD:-True}"
GRAD_OFFLOAD="${GRAD_OFFLOAD:-False}"

TOTAL_EPOCHS="${TOTAL_EPOCHS:-15}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-null}"
SAVE_FREQ="${SAVE_FREQ:-20}"
TEST_FREQ="${TEST_FREQ:-5}"
RESUME_MODE="${RESUME_MODE:-auto}"
RESUME_FROM_PATH="${RESUME_FROM_PATH:-null}"

# ---------------------------------------------------------------------------
# Repository paths and fixed recipe
# ---------------------------------------------------------------------------

[[ "${VERBOSE:-0}" == "1" ]] && set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -L)"
EXAMPLE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -L)"
LITE_ROOT="$(cd "${EXAMPLE_ROOT}/../.." && pwd -L)"
REPO_ROOT="$(cd "${LITE_ROOT}/../.." && pwd -L)"
VALIDATOR="${SCRIPT_DIR}/validate_deepseek_v4_dapo.py"
CHAT_TEMPLATE_FILE="${SCRIPT_DIR}/deepseek_v4_chat_template.jinja"
OUTPUT_ROOT="${OUTPUT_ROOT:-${EXAMPLE_ROOT}/outputs/ds4_dapo}"

add_pythonpath() {
  if [[ -n "${1:-}" ]]; then
    export PYTHONPATH="${1}:${PYTHONPATH:-}"
  fi
}

add_pythonpath "${EXAMPLE_ROOT}"
add_pythonpath "${LITE_ROOT}"
add_pythonpath "${REPO_ROOT}"
add_pythonpath "${VERL_ROOT:-}"
add_pythonpath "${MEGATRON_ROOT:-}"

export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_USE_V1=1
export VERL_VLLM_FP8_QUANT_ENABLED=1

case "${ROLLOUT_WEIGHT_BITS}" in
  4)
    ROLLOUT_RESYNC_FORMAT=mxfp4
    ROLLOUT_EXPERT_DTYPE=fp4
    ROLLOUT_MOE_BACKEND=marlin
    ROLLOUT_SCALE_FMT=ue8m0
    ;;
  8)
    ROLLOUT_RESYNC_FORMAT=block_fp8
    ROLLOUT_EXPERT_DTYPE=fp8
    ROLLOUT_MOE_BACKEND=flashinfer_cutlass
    ROLLOUT_SCALE_FMT=float32
    ;;
  *)
    echo "ROLLOUT_WEIGHT_BITS must be 4 or 8, got ${ROLLOUT_WEIGHT_BITS}" >&2
    exit 2
    ;;
esac

case "${ENABLE_R3,,}" in
  true|1|yes|on)
    ENABLE_R3=True
    ROUTER_REPLAY_MODE=R3
    ENABLE_ROLLOUT_ROUTING_REPLAY=True
    R3_TAG=on
    ;;
  false|0|no|off)
    ENABLE_R3=False
    ROUTER_REPLAY_MODE=disabled
    ENABLE_ROLLOUT_ROUTING_REPLAY=False
    R3_TAG=off
    ;;
  *)
    echo "ENABLE_R3 must be a boolean, got ${ENABLE_R3}" >&2
    exit 2
    ;;
esac

: "${TRAIN_FILES:?set TRAIN_FILES or DAPO_DATA_DIR}"
: "${VAL_FILES:?set VAL_FILES or DAPO_DATA_DIR}"

MAX_SEQ_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
DEFAULT_RUN_NAME="ds4_dapo_pp${ACTOR_PP}_ep${ACTOR_EP}_cp${ACTOR_CP}"
DEFAULT_RUN_NAME+="_rtp${ROLLOUT_TP}_w${ROLLOUT_WEIGHT_BITS}_r3${R3_TAG}"
RUN_NAME="${RUN_NAME:-${DEFAULT_RUN_NAME}}"
CKPT_DIR="${CKPT_DIR:-${OUTPUT_ROOT}/checkpoints/${RUN_NAME}}"
LOG_FILE="${LOG_FILE:-${OUTPUT_ROOT}/${RUN_NAME}.log}"
JSONL_FILE="${JSONL_FILE:-${OUTPUT_ROOT}/${RUN_NAME}.jsonl}"
CMD_FILE="${CMD_FILE:-${OUTPUT_ROOT}/${RUN_NAME}.cmd.sh}"

mkdir -p \
  "${OUTPUT_ROOT}" \
  "${CKPT_DIR}" \
  "$(dirname "${LOG_FILE}")" \
  "$(dirname "${JSONL_FILE}")" \
  "$(dirname "${CMD_FILE}")"
export VERL_FILE_LOGGER_PATH="${JSONL_FILE}"

python3 "${VALIDATOR}" geometry \
  --model-config "${MODEL_PATH}/config.json" \
  --rollout-tp "${ROLLOUT_TP}"

DS4_CHAT_TEMPLATE="${DEEPSEEK_V4_FLASH_CHAT_TEMPLATE:-$(<"${CHAT_TEMPLATE_FILE}")}"

# ---------------------------------------------------------------------------
# Hydra configuration
# ---------------------------------------------------------------------------

ALGORITHM=(
  "algorithm.adv_estimator=grpo"
  "algorithm.use_kl_in_reward=False"
  "algorithm.kl_ctrl.kl_coef=0.0"
  "algorithm.rollout_correction.bypass_mode=False"
  "algorithm.norm_adv_by_std_in_grpo=False"
)

# Use veRL's native RLHFDataset. Inputs must already follow its rule-reward
# schema (prompt, data_source, and reward_model.ground_truth).
DATA=(
  "data.train_files=${TRAIN_FILES}"
  "data.val_files=${VAL_FILES}"
  "data.train_batch_size=${TRAIN_BATCH_SIZE}"
  "data.prompt_key=prompt"
  "data.return_raw_chat=True"
  "data.max_prompt_length=${MAX_PROMPT_LENGTH}"
  "data.max_response_length=${MAX_RESPONSE_LENGTH}"
  "data.filter_overlong_prompts=True"
  "+data.apply_chat_template_kwargs.chat_template='${DS4_CHAT_TEMPLATE}'"
  "data.truncation=error"
  "data.dataloader_num_workers=8"
)

MODEL=(
  "actor_rollout_ref.model.path=${MODEL_PATH}"
  "actor_rollout_ref.model.trust_remote_code=True"
  "actor_rollout_ref.model.use_fused_kernels=True"
  "actor_rollout_ref.model.custom_chat_template='${DS4_CHAT_TEMPLATE}'"
)

ACTOR=(
  "actor@actor_rollout_ref.actor=mlite_actor"
  "actor_rollout_ref.actor.optim.lr=1e-6"
  "actor_rollout_ref.actor.optim.weight_decay=0.1"
  "actor_rollout_ref.actor.optim.betas=[0.9,0.95]"
  "actor_rollout_ref.actor.optim.clip_grad=1.0"
  "actor_rollout_ref.actor.optim.lr_warmup_steps=0"
  "actor_rollout_ref.actor.optim.lr_warmup_init=0"
  "actor_rollout_ref.actor.optim.lr_decay_style=constant"
  "actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}"
  "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1"
  "actor_rollout_ref.actor.use_dynamic_bsz=True"
  "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${MAX_SEQ_LEN}"
  "actor_rollout_ref.actor.use_kl_loss=False"
  "actor_rollout_ref.actor.kl_loss_coef=0.0"
  "actor_rollout_ref.actor.entropy_coeff=0"
  "actor_rollout_ref.actor.policy_loss.loss_mode=vanilla"
  "actor_rollout_ref.actor.clip_ratio_low=0.2"
  "actor_rollout_ref.actor.clip_ratio_high=0.28"
  "actor_rollout_ref.actor.clip_ratio_c=10.0"
  "actor_rollout_ref.actor.loss_agg_mode=token-mean"
  "actor_rollout_ref.actor.engine.dtype=bfloat16"
  "actor_rollout_ref.actor.engine.model_name=deepseek_v4"
  "actor_rollout_ref.actor.engine.impl=lite"
  "actor_rollout_ref.actor.engine.tp=1"
  "actor_rollout_ref.actor.engine.pp=${ACTOR_PP}"
  "actor_rollout_ref.actor.engine.vpp=1"
  "actor_rollout_ref.actor.engine.cp=${ACTOR_CP}"
  "actor_rollout_ref.actor.engine.ep=${ACTOR_EP}"
  "actor_rollout_ref.actor.engine.etp=1"
  "actor_rollout_ref.actor.engine.param_offload=${PARAM_OFFLOAD}"
  "actor_rollout_ref.actor.engine.optimizer_offload=${OPTIMIZER_OFFLOAD}"
  "actor_rollout_ref.actor.engine.grad_offload=${GRAD_OFFLOAD}"
  "actor_rollout_ref.actor.engine.attention_backend_override=fused"
  "actor_rollout_ref.actor.engine.impl_cfg.use_thd=True"
  "+actor_rollout_ref.actor.engine.impl_cfg.optimizer=fsdp2"
  "actor_rollout_ref.actor.engine.load_hf_weights=True"
  "+actor_rollout_ref.actor.engine.cross_entropy_fusion=True"
  "actor_rollout_ref.actor.engine.resync_format=${ROLLOUT_RESYNC_FORMAT}"
  "+actor_rollout_ref.actor.engine.resync_config.expert_dtype=${ROLLOUT_EXPERT_DTYPE}"
  "+actor_rollout_ref.actor.engine.impl_cfg.recompute=full"
  "+actor_rollout_ref.actor.engine.impl_cfg.mtp_enable=True"
  "+actor_rollout_ref.actor.engine.impl_cfg.mtp_enable_train=True"
  "actor_rollout_ref.actor.engine.router_replay_mode=${ROUTER_REPLAY_MODE}"
)

if [[ "${OPTIMIZER_OFFLOAD}" =~ ^(True|true|1)$ ]]; then
  ACTOR+=(
    "+actor_rollout_ref.actor.optim.override_optimizer_config.offload_fraction=1.0"
    "+actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True"
    "+actor_rollout_ref.actor.optim.override_optimizer_config.decoupled_weight_decay=True"
  )
fi

VLLM_CONFIG="actor_rollout_ref.rollout.engine_kwargs.vllm"
VLLM_QUANT_CONFIG="${VLLM_CONFIG}.hf_overrides.quantization_config"
VLLM_WORKER_EXTENSION="verl.workers.rollout.vllm_rollout.utils"
VLLM_WORKER_EXTENSION+=".vLLMColocateWorkerExtension"

ROLLOUT=(
  "actor_rollout_ref.rollout.name=vllm"
  "actor_rollout_ref.rollout.mode=async"
  "actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}"
  "actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION}"
  "actor_rollout_ref.rollout.n=${ROLLOUT_N}"
  "actor_rollout_ref.rollout.calculate_log_probs=True"
  "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True"
  "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${MAX_SEQ_LEN}"
  "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1"
  "actor_rollout_ref.rollout.prompt_length=${MAX_PROMPT_LENGTH}"
  "actor_rollout_ref.rollout.response_length=${MAX_RESPONSE_LENGTH}"
  "actor_rollout_ref.rollout.max_model_len=${MAX_SEQ_LEN}"
  "actor_rollout_ref.rollout.max_num_seqs=32"
  "actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_SEQ_LEN}"
  "actor_rollout_ref.rollout.temperature=1.0"
  "actor_rollout_ref.rollout.top_p=1.0"
  "actor_rollout_ref.rollout.top_k=-1"
  "actor_rollout_ref.rollout.val_kwargs.temperature=0.0"
  "actor_rollout_ref.rollout.val_kwargs.top_p=1.0"
  "actor_rollout_ref.rollout.val_kwargs.do_sample=False"
  "actor_rollout_ref.rollout.val_kwargs.n=1"
  "actor_rollout_ref.rollout.free_cache_engine=True"
  "actor_rollout_ref.rollout.load_format=dummy"
  "actor_rollout_ref.rollout.enable_rollout_routing_replay=${ENABLE_ROLLOUT_ROUTING_REPLAY}"
  "actor_rollout_ref.rollout.enforce_eager=True"
  "+${VLLM_CONFIG}.disable_custom_all_reduce=True"
  "+${VLLM_CONFIG}.worker_extension_cls=${VLLM_WORKER_EXTENSION}"
  "+${VLLM_CONFIG}.kv_cache_dtype=fp8"
  "+${VLLM_CONFIG}.moe_backend=${ROLLOUT_MOE_BACKEND}"
  "+${VLLM_CONFIG}.hf_overrides.expert_dtype=${ROLLOUT_EXPERT_DTYPE}"
  "+${VLLM_QUANT_CONFIG}.activation_scheme=dynamic"
  "+${VLLM_QUANT_CONFIG}.fmt=e4m3"
  "+${VLLM_QUANT_CONFIG}.quant_method=fp8"
  "+${VLLM_QUANT_CONFIG}.scale_fmt=${ROLLOUT_SCALE_FMT}"
  "+${VLLM_QUANT_CONFIG}.weight_block_size=[128,128]"
)

TRAINER=(
  "critic.enable=False"
  "trainer.balance_batch=True"
  "trainer.logger=[console,file]"
  "trainer.project_name=verl-mlite-ds4-dapo"
  "trainer.experiment_name=${RUN_NAME}"
  "trainer.n_gpus_per_node=${NGPUS_PER_NODE}"
  "trainer.nnodes=${NNODES}"
  "trainer.save_freq=${SAVE_FREQ}"
  "trainer.test_freq=${TEST_FREQ}"
  "trainer.total_epochs=${TOTAL_EPOCHS}"
  "trainer.total_training_steps=${TOTAL_TRAINING_STEPS}"
  "trainer.resume_mode=${RESUME_MODE}"
  "trainer.resume_from_path=${RESUME_FROM_PATH}"
  "trainer.default_local_dir=${CKPT_DIR}"
  "trainer.val_before_train=False"
  "trainer.log_val_generations=10"
)

REWARD=(
  "+reward.reward_kwargs.overlong_buffer_cfg.enable=True"
  "+reward.reward_kwargs.overlong_buffer_cfg.len=4096"
  "+reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0"
  "+reward.reward_kwargs.overlong_buffer_cfg.log=False"
)

COMMAND=(
  python3 -m verl.trainer.main_ppo
  "hydra.searchpath=[pkg://verl_mlite.config]"
  "${ALGORITHM[@]}"
  "${DATA[@]}"
  "${MODEL[@]}"
  "${ACTOR[@]}"
  "${ROLLOUT[@]}"
  "${TRAINER[@]}"
  "${REWARD[@]}"
  "$@"
)

printf '%q ' "${COMMAND[@]}" >"${CMD_FILE}"
printf '\n' >>"${CMD_FILE}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi

python3 "${VALIDATOR}" environment

echo "[ds4-dapo] weights=expert-w${ROLLOUT_WEIGHT_BITS}/dense-w8 r3=${ENABLE_R3}"
echo "[ds4-dapo] train=${TRAIN_FILES} val=${VAL_FILES} cmd=${CMD_FILE}"

set +e
"${COMMAND[@]}" 2>&1 | tee "${LOG_FILE}"
cmd_rc="${PIPESTATUS[0]}"
set -e
exit "${cmd_rc}"
