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

: "${MODEL_PATH:?set MODEL_PATH to a Hugging Face checkpoint directory or model id}"
: "${TRAIN_FILES:?set TRAIN_FILES to a messages parquet path or comma-separated parquet paths}"

BACKEND="${BACKEND:-mlite}"
VAL_FILES="${VAL_FILES:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${EXAMPLE_ROOT}/outputs/qwen3moe_sft}"
PROJECT_NAME="${PROJECT_NAME:-verl-mlite-qwen3moe-sft}"

NUM_GPUS="${NUM_GPUS:-${NPROC_PER_NODE:-8}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${NUM_GPUS}}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"

TOTAL_STEPS="${TOTAL_STEPS:-100}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
SAVE_FREQ="${SAVE_FREQ:-${TOTAL_STEPS}}"
TEST_FREQ="${TEST_FREQ:--1}"
RESUME_MODE="${RESUME_MODE:-disable}"
RESUME_FROM_PATH="${RESUME_FROM_PATH:-null}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-8192}"
MAX_LENGTH="${MAX_LENGTH:-${MAX_TOKENS_PER_GPU}}"
PAD_MODE="${PAD_MODE:-no_padding}"
USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-True}"
USE_REMOVE_PADDING="${USE_REMOVE_PADDING:-True}"
IGNORE_INPUT_IDS_MISMATCH="${IGNORE_INPUT_IDS_MISMATCH:-True}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-True}"
MESSAGES_KEY="${MESSAGES_KEY:-messages}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SEED="${SEED:-1}"

TP_SIZE="${TP_SIZE:-2}"
PP_SIZE="${PP_SIZE:-1}"
VPP_SIZE="${VPP_SIZE:-null}"
CP_SIZE="${CP_SIZE:-1}"
EP_SIZE="${EP_SIZE:-8}"
ETP_SIZE="${ETP_SIZE:-1}"
DTYPE="${DTYPE:-bfloat16}"
MLITE_MODEL_NAME="${MLITE_MODEL_NAME:-auto}"
MLITE_IMPL="${MLITE_IMPL:-lite}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-flash}"
# Optimizer backend:
# - distopt (default): Megatron-Core DDP + distributed optimizer.
# - fsdp2: Megatron Lite FSDP2 wrapper + optimizer.
MLITE_OPTIMIZER_BACKEND="${MLITE_OPTIMIZER_BACKEND:-distopt}"

LR="${LR:-1e-5}"
MIN_LR="${MIN_LR:-${LR}}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
BETAS="${BETAS:-[0.9,0.95]}"
CLIP_GRAD="${CLIP_GRAD:-1.0}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-0}"
LR_DECAY_STYLE="${LR_DECAY_STYLE:-constant}"

PARAM_OFFLOAD="${PARAM_OFFLOAD:-False}"
OPTIMIZER_OFFLOAD="${OPTIMIZER_OFFLOAD:-True}"
GRAD_OFFLOAD="${GRAD_OFFLOAD:-False}"
OPTIMIZER_STATE_OFFLOAD_FRACTION="${OPTIMIZER_STATE_OFFLOAD_FRACTION:-1.0}"
USE_PRECISION_AWARE_OPTIMIZER="${USE_PRECISION_AWARE_OPTIMIZER:-True}"
DECOUPLED_WEIGHT_DECAY="${DECOUPLED_WEIGHT_DECAY:-True}"
DRY_RUN="${DRY_RUN:-0}"
EXTRA_ARGS=("$@")

if [[ "${BACKEND}" != "mlite" ]]; then
  echo "Unsupported BACKEND=${BACKEND}. This example is for BACKEND=mlite." >&2
  exit 1
fi

if [[ "${PAD_MODE}" != "no_padding" ]]; then
  echo "Megatron Lite VERL example currently supports PAD_MODE=no_padding only." >&2
  exit 1
fi

case "${MLITE_OPTIMIZER_BACKEND}" in
  distopt)
    MLITE_IMPL_OPTIMIZER="mc"
    ;;
  fsdp2)
    MLITE_IMPL_OPTIMIZER="fsdp2"
    ;;
  *)
    echo "Unsupported MLITE_OPTIMIZER_BACKEND=${MLITE_OPTIMIZER_BACKEND}. Expected distopt or fsdp2." >&2
    exit 1
    ;;
esac

MLITE_VPP_SIZE="${VPP_SIZE}"
if [[ "${MLITE_VPP_SIZE}" == "null" ]]; then
  MLITE_VPP_SIZE=1
fi

RUN_NAME="${RUN_NAME:-qwen3moe_sft_mlite_tp${TP_SIZE}_pp${PP_SIZE}_cp${CP_SIZE}_ep${EP_SIZE}_etp${ETP_SIZE}}"
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

COMMON_ARGS=(
  "data.train_files=${TRAIN_FILES}"
  "data.train_batch_size=${TRAIN_BATCH_SIZE}"
  "data.micro_batch_size_per_gpu=${MICRO_BATCH_SIZE}"
  "data.use_dynamic_bsz=${USE_DYNAMIC_BSZ}"
  "data.max_token_len_per_gpu=${MAX_TOKENS_PER_GPU}"
  "data.max_length=${MAX_LENGTH}"
  "data.pad_mode=${PAD_MODE}"
  "data.truncation=error"
  "data.messages_key=${MESSAGES_KEY}"
  "data.ignore_input_ids_mismatch=${IGNORE_INPUT_IDS_MISMATCH}"
  "data.num_workers=${NUM_WORKERS}"
  "model=hf_model"
  "model.path=${MODEL_PATH}"
  "model.trust_remote_code=${TRUST_REMOTE_CODE}"
  "model.use_remove_padding=${USE_REMOVE_PADDING}"
  "optim=megatron"
  "optim.lr=${LR}"
  "optim.min_lr=${MIN_LR}"
  "optim.weight_decay=${WEIGHT_DECAY}"
  "optim.betas=${BETAS}"
  "optim.clip_grad=${CLIP_GRAD}"
  "optim.lr_warmup_steps=${LR_WARMUP_STEPS}"
  "optim.lr_warmup_init=0"
  "optim.lr_decay_style=${LR_DECAY_STYLE}"
  "trainer.logger=[console,file]"
  "trainer.project_name=${PROJECT_NAME}"
  "trainer.experiment_name=${RUN_NAME}"
  "trainer.default_local_dir=${CKPT_DIR}"
  "trainer.total_epochs=${TOTAL_EPOCHS}"
  "trainer.total_training_steps=${TOTAL_STEPS}"
  "trainer.save_freq=${SAVE_FREQ}"
  "trainer.test_freq=${TEST_FREQ}"
  "trainer.seed=${SEED}"
  "trainer.resume_mode=${RESUME_MODE}"
  "trainer.resume_from_path=${RESUME_FROM_PATH}"
  "trainer.nnodes=${NNODES}"
  "trainer.n_gpus_per_node=${NPROC_PER_NODE}"
  "checkpoint.save_contents=[model,optimizer,extra]"
)

if [[ -n "${VAL_FILES}" ]]; then
  COMMON_ARGS+=("data.val_files=${VAL_FILES}")
fi

BACKEND_ARGS=(
  "hydra.searchpath=[pkg://verl_mlite.config]"
  "engine=mlite"
  "engine.dtype=${DTYPE}"
  "engine.model_name=${MLITE_MODEL_NAME}"
  "engine.impl=${MLITE_IMPL}"
  "engine.tp=${TP_SIZE}"
  "engine.pp=${PP_SIZE}"
  "engine.vpp=${MLITE_VPP_SIZE}"
  "engine.cp=${CP_SIZE}"
  "engine.ep=${EP_SIZE}"
  "engine.etp=${ETP_SIZE}"
  "engine.param_offload=${PARAM_OFFLOAD}"
  "engine.optimizer_offload=${OPTIMIZER_OFFLOAD}"
  "engine.grad_offload=${GRAD_OFFLOAD}"
  "engine.attention_backend_override=${ATTENTION_BACKEND}"
  "engine.impl_cfg.use_thd=True"
  "+engine.impl_cfg.optimizer=${MLITE_IMPL_OPTIMIZER}"
)

if [[ "${OPTIMIZER_OFFLOAD}" == "True" || "${OPTIMIZER_OFFLOAD}" == "true" || "${OPTIMIZER_OFFLOAD}" == "1" ]]; then
  BACKEND_ARGS+=(
    "+optim.override_optimizer_config.offload_fraction=${OPTIMIZER_STATE_OFFLOAD_FRACTION}"
    "+optim.override_optimizer_config.use_precision_aware_optimizer=${USE_PRECISION_AWARE_OPTIMIZER}"
    "+optim.override_optimizer_config.decoupled_weight_decay=${DECOUPLED_WEIGHT_DECAY}"
  )
fi

COMMAND=(
  torchrun
  --nnodes="${NNODES}"
  --node_rank="${NODE_RANK}"
  --master_addr="${MASTER_ADDR}"
  --master_port="${MASTER_PORT}"
  --nproc_per_node="${NPROC_PER_NODE}"
  -m
  verl.trainer.sft_trainer
  "${COMMON_ARGS[@]}"
  "${BACKEND_ARGS[@]}"
  "${EXTRA_ARGS[@]}"
)

printf '%q ' "${COMMAND[@]}" > "${CMD_FILE}"
printf '\n' >> "${CMD_FILE}"

if [[ "${DRY_RUN}" == "1" ]]; then
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi

echo "[${BACKEND}] output_root=${OUTPUT_ROOT}"
echo "[${BACKEND}] log=${LOG_FILE}"
echo "[${BACKEND}] jsonl=${JSONL_FILE}"
echo "[${BACKEND}] cmd=${CMD_FILE}"
echo "[${BACKEND}] optimizer_backend=${MLITE_OPTIMIZER_BACKEND} impl_cfg.optimizer=${MLITE_IMPL_OPTIMIZER}"

set +e
"${COMMAND[@]}" 2>&1 | tee "${LOG_FILE}"
cmd_rc="${PIPESTATUS[0]}"
set -e
exit "${cmd_rc}"
