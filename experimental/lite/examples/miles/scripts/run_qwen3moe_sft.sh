#!/usr/bin/env bash
#SBATCH --job-name=mlite_miles_sft
#SBATCH --partition=batch
#SBATCH --account=coreai_devtech_all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=2010G
#SBATCH --time=00:45:00
#SBATCH --output=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code/env/mlite_miles_sft-%j.log

# Qwen3 MoE SFT with miles using either the Megatron Lite patch or native Megatron.
set -euo pipefail

if [[ "${VERBOSE:-0}" == "1" ]]; then set -x; fi

resolve_script_path() {
   if [[ -n "${MLITE_MILES_SCRIPT_PATH:-}" ]]; then
      readlink -f "${MLITE_MILES_SCRIPT_PATH}"
      return
   fi
   if [[ -n "${SLURM_JOB_ID:-}" ]] && command -v scontrol >/dev/null 2>&1; then
      local command_path
      command_path="$(
         scontrol show job "${SLURM_JOB_ID}" | tr ' ' '\n' | sed -n 's/^Command=//p' | head -1
      )"
      if [[ -n "${command_path}" && -r "${command_path}" ]]; then
         readlink -f "${command_path}"
         return
      fi
   fi
   readlink -f "${BASH_SOURCE[0]}"
}

SCRIPT_PATH="$(resolve_script_path)"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code/env/miles.sqsh}"
CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-/lustre:/lustre}"
DRY_RUN="${DRY_RUN:-0}"
unset PYTORCH_CUDA_ALLOC_CONF PYTORCH_ALLOC_CONF

if [[ "${IN_MILES_CONTAINER:-0}" != "1" ]]; then
   if [[ ! -r "${CONTAINER_IMAGE}" ]]; then
      echo "Container image not readable: ${CONTAINER_IMAGE}" >&2
      exit 2
   fi

   SRUN_CMD=(
      srun
      --container-image="${CONTAINER_IMAGE}"
      --container-mounts="${CONTAINER_MOUNTS}"
      --container-workdir=/
      env IN_MILES_CONTAINER=1 MLITE_MILES_SCRIPT_PATH="${SCRIPT_PATH}"
      bash "${SCRIPT_PATH}"
   )

   if [[ "${DRY_RUN}" == "1" ]]; then
      printf 'outer: '
      printf '%q ' "${SRUN_CMD[@]}"
      printf '\n'
      IN_MILES_CONTAINER=1 DRY_RUN=1 bash "${SCRIPT_PATH}"
      exit 0
   fi

   if [[ -z "${SLURM_JOB_ID:-}" ]]; then
      echo "Submit this script with sbatch, or run DRY_RUN=1 bash ${SCRIPT_PATH} for a local command check." >&2
      exit 2
   fi

   exec "${SRUN_CMD[@]}"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -L)"
EXAMPLE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -L)"
LITE_ROOT="$(cd "${EXAMPLE_ROOT}/../.." && pwd -L)"
REPO_ROOT="$(cd "${LITE_ROOT}/../.." && pwd -L)"

add_pythonpath() { [[ -n "${1:-}" ]] && export PYTHONPATH="${1}:${PYTHONPATH:-}"; }

MILES_ROOT="${MILES_ROOT:-/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code/miles}"
MEGATRON_ROOT="${MEGATRON_ROOT:-/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code/megatron_lite/Megatron-LM}"
MODEL_PATH="${MODEL_PATH:-/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/shunyad/models/Qwen/Qwen3-30B-A3B}"
TRAIN_DATA="${TRAIN_DATA:?Set TRAIN_DATA to a messages parquet file for miles SFT.}"

add_pythonpath "${EXAMPLE_ROOT}"
add_pythonpath "${LITE_ROOT}/examples"
add_pythonpath "${LITE_ROOT}"
add_pythonpath "${REPO_ROOT}"
add_pythonpath "${MEGATRON_ROOT}"
add_pythonpath "${MILES_ROOT}"

MODEL_SCRIPT="${MODEL_SCRIPT:-${MILES_ROOT}/scripts/models/qwen3-30B-A3B.sh}"
NUM_GPUS="${NUM_GPUS:-8}"
TRAIN_BACKEND="${TRAIN_BACKEND:-mlite}"
case "${TRAIN_BACKEND}" in
   mlite|megatron) ;;
   *) echo "TRAIN_BACKEND must be 'mlite' or 'megatron'." >&2; exit 2 ;;
esac
RUN_ROOT="${RUN_ROOT:-/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code/env/mlite_miles_runs}"
SAVE_DIR="${SAVE_DIR:-${RUN_ROOT}/qwen3moe_sft_${TRAIN_BACKEND}/${SLURM_JOB_ID:-dryrun}}"
LOG_DIR="${LOG_DIR:-${RUN_ROOT}/logs}"

TP_SIZE="${TP_SIZE:-2}"
PP_SIZE="${PP_SIZE:-1}"
CP_SIZE="${CP_SIZE:-1}"
EP_SIZE="${EP_SIZE:-8}"
ETP_SIZE="${ETP_SIZE:-1}"
MLITE_MODEL_NAME="${MLITE_MODEL_NAME:-qwen3_moe}"
MLITE_OPTIMIZER_BACKEND="${MLITE_OPTIMIZER_BACKEND:-dist_opt}"
OPTIMIZER_OFFLOAD="${OPTIMIZER_OFFLOAD:-1}"
PARAM_OFFLOAD="${PARAM_OFFLOAD:-0}"
if [[ -z "${MEGATRON_TO_HF_MODE:-}" ]]; then
   if [[ "${TRAIN_BACKEND}" == "mlite" ]]; then
      MEGATRON_TO_HF_MODE="raw"
   else
      MEGATRON_TO_HF_MODE="bridge"
   fi
fi

LR="${LR:-1e-5}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-128}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-9216}"
NUM_EPOCH="${NUM_EPOCH:-1}"
NUM_ROLLOUT="${NUM_ROLLOUT:-3}"
SAVE_INTERVAL="${SAVE_INTERVAL:-100000}"

mkdir -p "${SAVE_DIR}" "${LOG_DIR}"
source "${MODEL_SCRIPT}"

export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export PYTHONUNBUFFERED=1
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export no_proxy="${no_proxy:-127.0.0.1,${MASTER_ADDR}}"

BACKEND_ARGS=(
   --train-backend megatron
   --model-name qwen3_moe
   --megatron-to-hf-mode "${MEGATRON_TO_HF_MODE}"
)
if [[ "${TRAIN_BACKEND}" == "mlite" ]]; then
   BACKEND_ARGS+=(
      --mlite-backend-patch
      --mlite-model-name "${MLITE_MODEL_NAME}"
      --mlite-impl lite
      --mlite-optimizer-backend "${MLITE_OPTIMIZER_BACKEND}"
   )
   if [[ "${OPTIMIZER_OFFLOAD}" == "1" || "${OPTIMIZER_OFFLOAD}" == "true" || "${OPTIMIZER_OFFLOAD}" == "True" ]]; then
      BACKEND_ARGS+=(--mlite-optimizer-offload)
   fi
   if [[ "${PARAM_OFFLOAD}" == "1" || "${PARAM_OFFLOAD}" == "true" || "${PARAM_OFFLOAD}" == "True" ]]; then
      BACKEND_ARGS+=(--mlite-param-offload)
   fi
fi

CKPT_ARGS=(
   --hf-checkpoint "${MODEL_PATH}"
   --save "${SAVE_DIR}"
   --save-interval "${SAVE_INTERVAL}"
)

SFT_ARGS=(
   --rollout-function-path miles.rollout.sft_rollout.generate_rollout
   --prompt-data "${TRAIN_DATA}"
   --input-key messages
   --rollout-shuffle
   --num-epoch "${NUM_EPOCH}"
   --num-rollout "${NUM_ROLLOUT}"
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
   --global-batch-size "${GLOBAL_BATCH_SIZE}"
   --loss-type sft_loss
   --calculate-per-token-loss
   --disable-compute-advantages-and-returns
   --debug-train-only
)

PERF_ARGS=(
   --tensor-model-parallel-size "${TP_SIZE}"
   --sequence-parallel
   --pipeline-model-parallel-size "${PP_SIZE}"
   --context-parallel-size "${CP_SIZE}"
   --expert-model-parallel-size "${EP_SIZE}"
   --expert-tensor-parallel-size "${ETP_SIZE}"
   --use-dynamic-batch-size
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr "${LR}"
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.95
   --clip-grad 1.0
)
if [[ "${TRAIN_BACKEND}" == "megatron" ]] && [[ "${OPTIMIZER_OFFLOAD}" == "1" || "${OPTIMIZER_OFFLOAD}" == "true" || "${OPTIMIZER_OFFLOAD}" == "True" ]]; then
   OPTIMIZER_ARGS+=(
      --optimizer-cpu-offload
      --overlap-cpu-optimizer-d2h-h2d
      --use-precision-aware-optimizer
   )
fi

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --attention-backend flash
   --log-throughput
)

JOB_ARGS=(
   python3 -m miles_mlite.launch
   --actor-num-nodes 1
   --actor-num-gpus-per-node "${NUM_GPUS}"
   --num-gpus-per-node "${NUM_GPUS}"
   "${MODEL_ARGS[@]}"
   "${BACKEND_ARGS[@]}"
   "${CKPT_ARGS[@]}"
   "${SFT_ARGS[@]}"
   "${OPTIMIZER_ARGS[@]}"
   "${PERF_ARGS[@]}"
   "${MISC_ARGS[@]}"
)

RUNTIME_ENV_JSON="$(
python3 - <<PY
import json, os
paths = ["${EXAMPLE_ROOT}", "${LITE_ROOT}/examples", "${LITE_ROOT}", "${REPO_ROOT}", "${MILES_ROOT}", "${MEGATRON_ROOT}"]
env = {
    "PYTHONPATH": ":".join(paths + [os.environ.get("PYTHONPATH", "")]),
    "CUDA_DEVICE_MAX_CONNECTIONS": os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS", "1"),
}
print(json.dumps({"env_vars": env}))
PY
)"

if [[ "${DRY_RUN}" == "1" ]]; then
   printf 'inner ray start: ray start --head --node-ip-address=%q --num-gpus=%q --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=%q\n' "${MASTER_ADDR}" "${NUM_GPUS}" "${RAY_DASHBOARD_PORT:-8265}"
   printf 'inner ray job: ray job submit --address=%q --runtime-env-json=%q -- ' "${RAY_ADDRESS:-http://127.0.0.1:8265}" "${RUNTIME_ENV_JSON}"
   printf '%q ' "${JOB_ARGS[@]}"
   printf '\n'
   exit 0
fi

trap 'ray stop --force || true' EXIT
ray stop --force || true
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${NUM_GPUS}" --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port="${RAY_DASHBOARD_PORT:-8265}"
set +e
ray job submit \
   --address "${RAY_ADDRESS:-http://127.0.0.1:8265}" \
   --runtime-env-json "${RUNTIME_ENV_JSON}" \
   -- \
   "${JOB_ARGS[@]}"
rc=$?
set -e
echo "MILES_${TRAIN_BACKEND^^}_SFT_DONE rc=${rc}"
exit "${rc}"
