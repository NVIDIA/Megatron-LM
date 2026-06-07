#!/usr/bin/env bash
set -euo pipefail

if [[ "${VERBOSE:-0}" == "1" ]]; then
  set -x
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -L)"
EXAMPLE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -L)"
LITE_ROOT="$(cd "${EXAMPLE_ROOT}/../.." && pwd -L)"
REPO_ROOT="$(cd "${LITE_ROOT}/../.." && pwd -L)"

add_pythonpath() {
  local path="${1:-}"
  if [[ -n "${path}" ]]; then
    export PYTHONPATH="${path}:${PYTHONPATH:-}"
  fi
}

add_pythonpath "${EXAMPLE_ROOT}"
add_pythonpath "${LITE_ROOT}"
add_pythonpath "${REPO_ROOT}"
add_pythonpath "${VERL_ROOT:-}"
add_pythonpath "${MEGATRON_ROOT:-}"

export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  unset ROCR_VISIBLE_DEVICES
  unset HIP_VISIBLE_DEVICES
fi

DATASET_DIR="${DATASET_DIR:-${HOME}/data/gsm8k}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-35B-A3B}"
TRAIN_FILES="${TRAIN_FILES:-${DATASET_DIR}/train.parquet}"
VAL_FILES="${VAL_FILES:-${DATASET_DIR}/test.parquet}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${EXAMPLE_ROOT}/outputs/qwen35_gsm8k_grpo}"
PROJECT_NAME="${PROJECT_NAME:-verl-mlite-qwen35-gsm8k-grpo}"
INFER_BACKEND="${INFER_BACKEND:-vllm}"

NNODES="${NNODES:-1}"
NGPUS_PER_NODE="${NGPUS_PER_NODE:-${NPROC_PER_NODE:-8}}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-32}"
ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU="${ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-512}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1024}"
PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-8192}"

ROLLOUT_N="${ROLLOUT_N:-5}"
ROLLOUT_MODE="${ROLLOUT_MODE:-async}"
ROLLOUT_TP="${ROLLOUT_TP:-2}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.6}"
ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}"
ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU="${ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-${PPO_MAX_TOKEN_LEN_PER_GPU}}"
ROLLOUT_MAX_MODEL_LEN="${ROLLOUT_MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}"
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-1024}"
ROLLOUT_DEFAULT_MAX_NUM_BATCHED_TOKENS="$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))"
if (( ROLLOUT_DEFAULT_MAX_NUM_BATCHED_TOKENS < ROLLOUT_MAX_NUM_SEQS )); then
  ROLLOUT_DEFAULT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_SEQS}"
fi
ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-${ROLLOUT_DEFAULT_MAX_NUM_BATCHED_TOKENS}}"
if (( ROLLOUT_MAX_NUM_BATCHED_TOKENS < ROLLOUT_MAX_NUM_SEQS )); then
  ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_SEQS}"
fi
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.0}"
ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-1.0}"
ROLLOUT_TOP_K="${ROLLOUT_TOP_K:--1}"
ROLLOUT_LIMIT_IMAGES="${ROLLOUT_LIMIT_IMAGES:-0}"
ROLLOUT_LIMIT_VIDEOS="${ROLLOUT_LIMIT_VIDEOS:-0}"
VAL_TEMPERATURE="${VAL_TEMPERATURE:-0.0}"
VAL_TOP_P="${VAL_TOP_P:-1.0}"
VAL_DO_SAMPLE="${VAL_DO_SAMPLE:-False}"
VAL_N="${VAL_N:-1}"

ACTOR_TP="${ACTOR_TP:-2}"
ACTOR_PP="${ACTOR_PP:-1}"
ACTOR_VPP="${ACTOR_VPP:-null}"
ACTOR_CP="${ACTOR_CP:-1}"
ACTOR_EP="${ACTOR_EP:-8}"
ACTOR_ETP="${ACTOR_ETP:-1}"
DTYPE="${DTYPE:-bfloat16}"
MLITE_MODEL_NAME="${MLITE_MODEL_NAME:-auto}"
MLITE_IMPL="${MLITE_IMPL:-lite}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-flash}"

ACTOR_LR="${ACTOR_LR:-1e-6}"
POLICY_LOSS_MODE="${POLICY_LOSS_MODE:-vanilla}"
LOSS_AGG_MODE="${LOSS_AGG_MODE:-seq-mean-token-sum-norm}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
BETAS="${BETAS:-[0.9,0.95]}"
CLIP_GRAD="${CLIP_GRAD:-1.0}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-0}"
LR_DECAY_STYLE="${LR_DECAY_STYLE:-constant}"
ENTROPY_COEFF="${ENTROPY_COEFF:-0}"
USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-True}"
PARAM_OFFLOAD="${PARAM_OFFLOAD:-False}"
OPTIMIZER_OFFLOAD="${OPTIMIZER_OFFLOAD:-True}"
GRAD_OFFLOAD="${GRAD_OFFLOAD:-False}"
OPTIMIZER_STATE_OFFLOAD_FRACTION="${OPTIMIZER_STATE_OFFLOAD_FRACTION:-1.0}"
USE_PRECISION_AWARE_OPTIMIZER="${USE_PRECISION_AWARE_OPTIMIZER:-True}"
DECOUPLED_WEIGHT_DECAY="${DECOUPLED_WEIGHT_DECAY:-True}"

TOTAL_EPOCHS="${TOTAL_EPOCHS:-15}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-null}"
SAVE_FREQ="${SAVE_FREQ:-20}"
TEST_FREQ="${TEST_FREQ:-5}"
RESUME_MODE="${RESUME_MODE:-auto}"
LOG_VAL_GENERATIONS="${LOG_VAL_GENERATIONS:-10}"
LOGGER="${LOGGER:-[console,file]}"
USE_LEGACY_WORKER_IMPL="${USE_LEGACY_WORKER_IMPL:-disable}"
DRY_RUN="${DRY_RUN:-0}"
EXTRA_ARGS=("$@")

if [[ "${INFER_BACKEND}" != "vllm" && "${INFER_BACKEND}" != "sglang" && "${INFER_BACKEND}" != "trtllm" ]]; then
  echo "Unsupported INFER_BACKEND=${INFER_BACKEND}. Expected vllm, sglang, or trtllm." >&2
  exit 1
fi

if [[ "${INFER_BACKEND}" == "vllm" ]]; then
  export VLLM_USE_V1="${VLLM_USE_V1:-1}"
  export VLLM_ALLREDUCE_USE_SYMM_MEM="${VLLM_ALLREDUCE_USE_SYMM_MEM:-0}"
fi

MLITE_VPP_SIZE="${ACTOR_VPP}"
if [[ "${MLITE_VPP_SIZE}" == "null" ]]; then
  MLITE_VPP_SIZE=1
fi

RUN_NAME="${RUN_NAME:-qwen35_gsm8k_grpo_mlite_${INFER_BACKEND}_tp${ACTOR_TP}_pp${ACTOR_PP}_cp${ACTOR_CP}_ep${ACTOR_EP}}"
CKPT_DIR="${CKPT_DIR:-${OUTPUT_ROOT}/checkpoints/${RUN_NAME}}"
LOG_FILE="${LOG_FILE:-${OUTPUT_ROOT}/${RUN_NAME}.log}"
JSONL_FILE="${JSONL_FILE:-${OUTPUT_ROOT}/${RUN_NAME}.jsonl}"
CMD_FILE="${CMD_FILE:-${OUTPUT_ROOT}/${RUN_NAME}.cmd.sh}"

mkdir -p "${OUTPUT_ROOT}" "${CKPT_DIR}" "$(dirname "${LOG_FILE}")" "$(dirname "${JSONL_FILE}")" "$(dirname "${CMD_FILE}")"
export VERL_FILE_LOGGER_PATH="${JSONL_FILE}"

CACHE_ROOT="${VERL_MLITE_CACHE_ROOT:-${TMPDIR:-/tmp}/verl_mlite}"
mkdir -p "${CACHE_ROOT}/pycache_${USER:-user}" "${CACHE_ROOT}/torchinductor_${USER:-user}" "${CACHE_ROOT}/triton_${USER:-user}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-${CACHE_ROOT}/pycache_${USER:-user}}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${CACHE_ROOT}/torchinductor_${USER:-user}}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${CACHE_ROOT}/triton_${USER:-user}}"

ALGORITHM=(
  "algorithm.adv_estimator=grpo"
  "algorithm.use_kl_in_reward=False"
  "algorithm.kl_ctrl.kl_coef=0.0"
  "algorithm.rollout_correction.bypass_mode=True"
  "algorithm.norm_adv_by_std_in_grpo=False"
)

DATA=(
  "data.train_files=${TRAIN_FILES}"
  "data.val_files=${VAL_FILES}"
  "data.train_batch_size=${TRAIN_BATCH_SIZE}"
  "data.prompt_key=prompt"
  "data.return_raw_chat=True"
  "data.max_prompt_length=${MAX_PROMPT_LENGTH}"
  "data.max_response_length=${MAX_RESPONSE_LENGTH}"
  "data.filter_overlong_prompts=True"
  "data.truncation=error"
)

MODEL=(
  "actor_rollout_ref.model.path=${MODEL_PATH}"
  "actor_rollout_ref.model.trust_remote_code=True"
  "actor_rollout_ref.model.use_remove_padding=True"
  "actor_rollout_ref.model.use_fused_kernels=False"
)

ACTOR=(
  "actor@actor_rollout_ref.actor=mlite_actor"
  "actor_rollout_ref.actor.optim.lr=${ACTOR_LR}"
  "actor_rollout_ref.actor.optim.weight_decay=${WEIGHT_DECAY}"
  "actor_rollout_ref.actor.optim.betas=${BETAS}"
  "actor_rollout_ref.actor.optim.clip_grad=${CLIP_GRAD}"
  "actor_rollout_ref.actor.optim.lr_warmup_steps=${LR_WARMUP_STEPS}"
  "actor_rollout_ref.actor.optim.lr_warmup_init=0"
  "actor_rollout_ref.actor.optim.lr_decay_style=${LR_DECAY_STYLE}"
  "actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}"
  "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU}"
  "actor_rollout_ref.actor.use_dynamic_bsz=${USE_DYNAMIC_BSZ}"
  "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}"
  "actor_rollout_ref.actor.use_kl_loss=False"
  "actor_rollout_ref.actor.kl_loss_coef=0.0"
  "actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF}"
  "actor_rollout_ref.actor.policy_loss.loss_mode=${POLICY_LOSS_MODE}"
  "actor_rollout_ref.actor.loss_agg_mode=${LOSS_AGG_MODE}"
  "actor_rollout_ref.actor.engine.dtype=${DTYPE}"
  "actor_rollout_ref.actor.engine.model_name=${MLITE_MODEL_NAME}"
  "actor_rollout_ref.actor.engine.impl=${MLITE_IMPL}"
  "actor_rollout_ref.actor.engine.tp=${ACTOR_TP}"
  "actor_rollout_ref.actor.engine.pp=${ACTOR_PP}"
  "actor_rollout_ref.actor.engine.vpp=${MLITE_VPP_SIZE}"
  "actor_rollout_ref.actor.engine.cp=${ACTOR_CP}"
  "actor_rollout_ref.actor.engine.ep=${ACTOR_EP}"
  "actor_rollout_ref.actor.engine.etp=${ACTOR_ETP}"
  "actor_rollout_ref.actor.engine.param_offload=${PARAM_OFFLOAD}"
  "actor_rollout_ref.actor.engine.optimizer_offload=${OPTIMIZER_OFFLOAD}"
  "actor_rollout_ref.actor.engine.grad_offload=${GRAD_OFFLOAD}"
  "actor_rollout_ref.actor.engine.attention_backend_override=${ATTENTION_BACKEND}"
  "actor_rollout_ref.actor.engine.impl_cfg.use_thd=True"
)

if [[ "${OPTIMIZER_OFFLOAD}" == "True" || "${OPTIMIZER_OFFLOAD}" == "true" || "${OPTIMIZER_OFFLOAD}" == "1" ]]; then
  ACTOR+=(
    "+actor_rollout_ref.actor.optim.override_optimizer_config.offload_fraction=${OPTIMIZER_STATE_OFFLOAD_FRACTION}"
    "+actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=${USE_PRECISION_AWARE_OPTIMIZER}"
    "+actor_rollout_ref.actor.optim.override_optimizer_config.decoupled_weight_decay=${DECOUPLED_WEIGHT_DECAY}"
  )
fi

ROLLOUT=(
  "actor_rollout_ref.rollout.name=${INFER_BACKEND}"
  "actor_rollout_ref.rollout.mode=${ROLLOUT_MODE}"
  "actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}"
  "actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION}"
  "actor_rollout_ref.rollout.n=${ROLLOUT_N}"
  "actor_rollout_ref.rollout.calculate_log_probs=True"
  "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${USE_DYNAMIC_BSZ}"
  "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU}"
  "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}"
  "actor_rollout_ref.rollout.prompt_length=${MAX_PROMPT_LENGTH}"
  "actor_rollout_ref.rollout.response_length=${MAX_RESPONSE_LENGTH}"
  "actor_rollout_ref.rollout.max_model_len=${ROLLOUT_MAX_MODEL_LEN}"
  "actor_rollout_ref.rollout.max_num_seqs=${ROLLOUT_MAX_NUM_SEQS}"
  "actor_rollout_ref.rollout.max_num_batched_tokens=${ROLLOUT_MAX_NUM_BATCHED_TOKENS}"
  "actor_rollout_ref.rollout.temperature=${ROLLOUT_TEMPERATURE}"
  "actor_rollout_ref.rollout.top_p=${ROLLOUT_TOP_P}"
  "actor_rollout_ref.rollout.top_k=${ROLLOUT_TOP_K}"
  "actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMPERATURE}"
  "actor_rollout_ref.rollout.val_kwargs.top_p=${VAL_TOP_P}"
  "actor_rollout_ref.rollout.val_kwargs.do_sample=${VAL_DO_SAMPLE}"
  "actor_rollout_ref.rollout.val_kwargs.n=${VAL_N}"
  "actor_rollout_ref.rollout.free_cache_engine=True"
)

if [[ "${INFER_BACKEND}" == "vllm" ]]; then
  ROLLOUT+=(
    "+actor_rollout_ref.rollout.engine_kwargs.vllm.limit_mm_per_prompt.image=${ROLLOUT_LIMIT_IMAGES}"
    "+actor_rollout_ref.rollout.engine_kwargs.vllm.limit_mm_per_prompt.video=${ROLLOUT_LIMIT_VIDEOS}"
  )
fi

TRAINER=(
  "critic.enable=False"
  "trainer.balance_batch=True"
  "trainer.logger=${LOGGER}"
  "trainer.project_name=${PROJECT_NAME}"
  "trainer.experiment_name=${RUN_NAME}"
  "trainer.n_gpus_per_node=${NGPUS_PER_NODE}"
  "trainer.nnodes=${NNODES}"
  "trainer.save_freq=${SAVE_FREQ}"
  "trainer.test_freq=${TEST_FREQ}"
  "trainer.total_epochs=${TOTAL_EPOCHS}"
  "trainer.total_training_steps=${TOTAL_TRAINING_STEPS}"
  "trainer.resume_mode=${RESUME_MODE}"
  "trainer.default_local_dir=${CKPT_DIR}"
  "trainer.val_before_train=False"
  "trainer.log_val_generations=${LOG_VAL_GENERATIONS}"
  "trainer.use_legacy_worker_impl=${USE_LEGACY_WORKER_IMPL}"
)

COMMAND=(
  python3
  -m
  verl.trainer.main_ppo
  "hydra.searchpath=[pkg://verl_mlite.config]"
  "${ALGORITHM[@]}"
  "${DATA[@]}"
  "${MODEL[@]}"
  "${ACTOR[@]}"
  "${ROLLOUT[@]}"
  "${TRAINER[@]}"
  "${EXTRA_ARGS[@]}"
)

printf '%q ' "${COMMAND[@]}" > "${CMD_FILE}"
printf '\n' >> "${CMD_FILE}"

if [[ "${DRY_RUN}" == "1" ]]; then
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi

echo "[mlite] output_root=${OUTPUT_ROOT}"
echo "[mlite] log=${LOG_FILE}"
echo "[mlite] jsonl=${JSONL_FILE}"
echo "[mlite] cmd=${CMD_FILE}"

set +e
"${COMMAND[@]}" 2>&1 | tee "${LOG_FILE}"
cmd_rc="${PIPESTATUS[0]}"
set -e
exit "${cmd_rc}"
