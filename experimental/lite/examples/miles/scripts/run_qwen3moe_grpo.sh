#!/usr/bin/env bash
#SBATCH --job-name=mlite_miles_grpo
#SBATCH --partition=batch
#SBATCH --account=coreai_devtech_all
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=1000G
#SBATCH --time=04:00:00
#SBATCH --output=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code/env/mlite_miles_grpo-%j.log

# Qwen3 MoE GRPO using miles' qwen3-next-80B-A3B 8-GPU recipe, with only
# the training backend swapped to Megatron Lite.
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
RAY_PORT="${RAY_PORT:-6379}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
RAY_EXPECTED_NODES="${RAY_EXPECTED_NODES:-2}"
SYSTEM_PATH="${MLITE_SYSTEM_PATH:-/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin}"
SYSTEM_CC="${MLITE_CC:-/usr/bin/gcc}"
SYSTEM_CXX="${MLITE_CXX:-/usr/bin/g++}"
SYSTEM_CPATH="${MLITE_CPATH:-/usr/include:/usr/include/x86_64-linux-gnu}"

MLITE_SAVE_OPTIMIZER_CHECKPOINT="${MLITE_SAVE_OPTIMIZER_CHECKPOINT:-0}"
MLITE_SAVE_RNG_CHECKPOINT="${MLITE_SAVE_RNG_CHECKPOINT:-0}"
MLITE_LOAD_OPTIMIZER_CHECKPOINT="${MLITE_LOAD_OPTIMIZER_CHECKPOINT:-0}"
MLITE_LOAD_RNG_CHECKPOINT="${MLITE_LOAD_RNG_CHECKPOINT:-0}"
MLITE_VERIFY_CHECKPOINT_LOAD="${MLITE_VERIFY_CHECKPOINT_LOAD:-1}"
MLITE_DELETE_CHECKPOINT_AFTER_LOAD="${MLITE_DELETE_CHECKPOINT_AFTER_LOAD:-1}"
MLITE_RESET_ROLLOUT_AFTER_LOAD="${MLITE_RESET_ROLLOUT_AFTER_LOAD:-1}"

if [[ "${IN_MILES_CONTAINER:-0}" != "1" ]]; then
   if [[ ! -r "${CONTAINER_IMAGE}" ]]; then
      echo "Container image not readable: ${CONTAINER_IMAGE}" >&2
      exit 2
   fi
   if [[ -z "${MASTER_ADDR:-}" && -n "${SLURM_JOB_NODELIST:-}" ]] && command -v scontrol >/dev/null 2>&1; then
      MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)"
   fi
   MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"

   SRUN_CMD=(
      srun
      --nodes=2
      --ntasks=2
      --ntasks-per-node=1
      --container-image="${CONTAINER_IMAGE}"
      --container-mounts="${CONTAINER_MOUNTS}"
      --container-workdir=/
      env
      -u CONDA_PREFIX
      -u CONDA_DEFAULT_ENV
      -u CONDA_EXE
      -u CONDA_PYTHON_EXE
      -u CONDA_SHLVL
      -u _CE_CONDA
      -u _CE_M
      PATH="${SYSTEM_PATH}"
      CC="${SYSTEM_CC}"
      CXX="${SYSTEM_CXX}"
      CPATH="${SYSTEM_CPATH}"
      IN_MILES_CONTAINER=1
      MLITE_MILES_SCRIPT_PATH="${SCRIPT_PATH}"
      MASTER_ADDR="${MASTER_ADDR}"
      RAY_PORT="${RAY_PORT}"
      RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT}"
      RAY_EXPECTED_NODES="${RAY_EXPECTED_NODES}"
      MLITE_SAVE_OPTIMIZER_CHECKPOINT="${MLITE_SAVE_OPTIMIZER_CHECKPOINT}"
      MLITE_SAVE_RNG_CHECKPOINT="${MLITE_SAVE_RNG_CHECKPOINT}"
      MLITE_LOAD_OPTIMIZER_CHECKPOINT="${MLITE_LOAD_OPTIMIZER_CHECKPOINT}"
      MLITE_LOAD_RNG_CHECKPOINT="${MLITE_LOAD_RNG_CHECKPOINT}"
      MLITE_VERIFY_CHECKPOINT_LOAD="${MLITE_VERIFY_CHECKPOINT_LOAD}"
      MLITE_DELETE_CHECKPOINT_AFTER_LOAD="${MLITE_DELETE_CHECKPOINT_AFTER_LOAD}"
      MLITE_RESET_ROLLOUT_AFTER_LOAD="${MLITE_RESET_ROLLOUT_AFTER_LOAD}"
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

set -ex

export PATH="${SYSTEM_PATH}"
export CC="${SYSTEM_CC}"
export CXX="${SYSTEM_CXX}"
export CPATH="${SYSTEM_CPATH}"
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_EXE CONDA_PYTHON_EXE CONDA_SHLVL _CE_CONDA _CE_M

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd -L)"
EXAMPLE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -L)"
LITE_ROOT="$(cd "${EXAMPLE_ROOT}/../.." && pwd -L)"
REPO_ROOT="$(cd "${LITE_ROOT}/../.." && pwd -L)"

MILES_ROOT="${MILES_ROOT:-/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code/miles}"
MEGATRON_ROOT="${MEGATRON_ROOT:-/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code/megatron_lite/Megatron-LM}"
MODEL_PATH="${MODEL_PATH:-/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/shunyad/models/Qwen/Qwen3-30B-A3B}"
RUN_ROOT="${RUN_ROOT:-/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code/env/mlite_miles_runs}"
DAPO_MATH_DATA="${DAPO_MATH_DATA:-/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/achartier/dapo-math-17k/dapo-math-17k.jsonl}"
LOAD_DIR="${LOAD_DIR:-${RUN_ROOT}/qwen3moe_sft_mlite/13021984}"
SAVE_DIR="${SAVE_DIR:-${RUN_ROOT}/qwen3moe_grpo_mlite/${SLURM_JOB_ID:-dryrun}}"
REF_LOAD="${REF_LOAD:-${MODEL_PATH}}"

if [[ ! -d "${MODEL_PATH}" ]]; then
   echo "MODEL_PATH does not exist: ${MODEL_PATH}" >&2
   exit 2
fi
if [[ ! -s "${DAPO_MATH_DATA}" ]]; then
   echo "DAPO_MATH_DATA does not exist: ${DAPO_MATH_DATA}" >&2
   exit 2
fi
if [[ ! -d "${LOAD_DIR}" ]]; then
   echo "LOAD_DIR does not exist: ${LOAD_DIR}" >&2
   exit 2
fi
mkdir -p "${SAVE_DIR}"

if [[ "${DRY_RUN}" == "1" ]]; then
   NVLINK_COUNT=0
   HAS_NVLINK=0
else
   NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
   if [[ "${NVLINK_COUNT}" -gt 0 ]]; then
      HAS_NVLINK=1
   else
      HAS_NVLINK=0
   fi
fi
echo "HAS_NVLINK: ${HAS_NVLINK} (detected ${NVLINK_COUNT} NVLink references)"

source "${MILES_ROOT}/scripts/models/qwen3-30B-A3B.sh"

export PYTHONBUFFERED=16
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export RAY_PORT RAY_DASHBOARD_PORT RAY_EXPECTED_NODES
export no_proxy="127.0.0.1,${MASTER_ADDR}"
export MLITE_SAVE_OPTIMIZER_CHECKPOINT MLITE_SAVE_RNG_CHECKPOINT
export MLITE_LOAD_OPTIMIZER_CHECKPOINT MLITE_LOAD_RNG_CHECKPOINT
export MLITE_VERIFY_CHECKPOINT_LOAD MLITE_DELETE_CHECKPOINT_AFTER_LOAD
export MLITE_RESET_ROLLOUT_AFTER_LOAD

add_pythonpath() { [[ -n "${1:-}" ]] && export PYTHONPATH="${1}:${PYTHONPATH:-}"; }
add_pythonpath "${EXAMPLE_ROOT}"
add_pythonpath "${LITE_ROOT}/examples"
add_pythonpath "${LITE_ROOT}"
add_pythonpath "${REPO_ROOT}"
add_pythonpath "${MEGATRON_ROOT}"
add_pythonpath "${MILES_ROOT}"

CKPT_ARGS=(
   --hf-checkpoint "${MODEL_PATH}"
   --ref-load "${REF_LOAD}"
   --load "${LOAD_DIR}"
   --save "${SAVE_DIR}"
   --save-interval 20
)

ROLLOUT_ARGS=(
   --prompt-data "${DAPO_MATH_DATA}"
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   --num-rollout "${NUM_ROLLOUT:-5}"
   --rollout-batch-size 16
   --n-samples-per-prompt 4
   --rollout-max-response-len 8192
   --rollout-temperature 0.8
   --global-batch-size 64
   --balance-data
   --custom-rollout-log-function-path miles_mlite.reward_log.log_rollout_data
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 8
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 2048
)

GRPO_ARGS=(
   --advantage-estimator gspo
   --use-rollout-logprobs
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.00
   --eps-clip 4e-4
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

BACKEND_ARGS=(
   --train-backend megatron
   --model-name qwen3_moe
   --megatron-to-hf-mode raw
   --mlite-backend-patch
   --mlite-model-name qwen3_moe
   --mlite-impl lite
   --mlite-optimizer-backend dist_opt
   --mlite-optimizer-offload
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --rollout-num-gpus 8
   --sglang-mem-fraction-static 0.8
   --sglang-ep-size 1
   --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 128)
   --update-weight-buffer-size $((1024 * 1024 * 1024 * 4))
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --moe-token-dispatcher-type alltoall
   --log-passrate
)

JOB_ARGS=(
   python3 -m miles_mlite.launch
   --actor-num-nodes 1
   --actor-num-gpus-per-node 8
   --num-gpus-per-node 8
   "${MODEL_ARGS[@]}"
   "${CKPT_ARGS[@]}"
   "${ROLLOUT_ARGS[@]}"
   "${OPTIMIZER_ARGS[@]}"
   "${GRPO_ARGS[@]}"
   "${PERF_ARGS[@]}"
   "${SGLANG_ARGS[@]}"
   "${MISC_ARGS[@]}"
   "${BACKEND_ARGS[@]}"
)

RUNTIME_ENV_JSON="$(
python3 - <<PY
import json
import os

paths = [
    "${EXAMPLE_ROOT}",
    "${LITE_ROOT}/examples",
    "${LITE_ROOT}",
    "${REPO_ROOT}",
    "${MILES_ROOT}",
    "${MEGATRON_ROOT}",
]
env = {
    "PYTHONPATH": ":".join(paths + [os.environ.get("PYTHONPATH", "")]),
    "PATH": os.environ.get("PATH", "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"),
    "CC": os.environ.get("CC", "/usr/bin/gcc"),
    "CXX": os.environ.get("CXX", "/usr/bin/g++"),
    "CPATH": os.environ.get("CPATH", "/usr/include:/usr/include/x86_64-linux-gnu"),
    "CUDA_DEVICE_MAX_CONNECTIONS": os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS", "1"),
    "NCCL_NVLS_ENABLE": "${HAS_NVLINK}",
    "MASTER_ADDR": os.environ.get("MASTER_ADDR", "127.0.0.1"),
    "MLITE_SAVE_OPTIMIZER_CHECKPOINT": os.environ.get("MLITE_SAVE_OPTIMIZER_CHECKPOINT", "0"),
    "MLITE_SAVE_RNG_CHECKPOINT": os.environ.get("MLITE_SAVE_RNG_CHECKPOINT", "0"),
    "MLITE_LOAD_OPTIMIZER_CHECKPOINT": os.environ.get("MLITE_LOAD_OPTIMIZER_CHECKPOINT", "0"),
    "MLITE_LOAD_RNG_CHECKPOINT": os.environ.get("MLITE_LOAD_RNG_CHECKPOINT", "0"),
    "MLITE_VERIFY_CHECKPOINT_LOAD": os.environ.get("MLITE_VERIFY_CHECKPOINT_LOAD", "1"),
    "MLITE_DELETE_CHECKPOINT_AFTER_LOAD": os.environ.get("MLITE_DELETE_CHECKPOINT_AFTER_LOAD", "1"),
    "MLITE_RESET_ROLLOUT_AFTER_LOAD": os.environ.get("MLITE_RESET_ROLLOUT_AFTER_LOAD", "1"),
    "no_proxy": os.environ.get("no_proxy", "127.0.0.1,localhost"),
}
print(json.dumps({"env_vars": env}))
PY
)"

if [[ "${DRY_RUN}" == "1" ]]; then
   printf 'inner ray head: ray start --head --node-ip-address=%q --port=%q --num-gpus=8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=%q\n' \
      "${MASTER_ADDR}" "${RAY_PORT}" "${RAY_DASHBOARD_PORT}"
   printf 'inner ray worker: ray start --address=%q --num-gpus=8 --node-ip-address=<worker> --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=%q\n' \
      "${MASTER_ADDR}:${RAY_PORT}" "${RAY_DASHBOARD_PORT}"
   printf 'inner ray job: ray job submit --address=%q --runtime-env-json=%q -- ' \
      "${RAY_ADDRESS:-http://127.0.0.1:${RAY_DASHBOARD_PORT}}" "${RUNTIME_ENV_JSON}"
   printf '%q ' "${JOB_ARGS[@]}"
   printf '\n'
   exit 0
fi

RAY_NODE_RANK="${SLURM_PROCID:-0}"
RAY_DONE_FILE="${RUN_ROOT}/.mlite_miles_grpo_${SLURM_JOB_ID:-manual}.done"
NODE_ADDR="$(hostname -s)"

ray stop --force || true
pkill -9 sglang || true
pkill -9 ray || true
sleep 3
ray stop --force || true

cleanup() {
   if [[ "${RAY_NODE_RANK}" == "0" ]]; then
      touch "${RAY_DONE_FILE}" || true
   fi
   ray stop --force || true
}
trap cleanup EXIT

if [[ "${RAY_NODE_RANK}" != "0" ]]; then
   for attempt in $(seq 1 120); do
      if ray start \
         --address="${MASTER_ADDR}:${RAY_PORT}" \
         --num-gpus 8 \
         --node-ip-address "${NODE_ADDR}" \
         --disable-usage-stats \
         --dashboard-host=0.0.0.0 \
         --dashboard-port="${RAY_DASHBOARD_PORT}"; then
         break
      fi
      if [[ "${attempt}" == "120" ]]; then
         echo "Failed to join Ray head at ${MASTER_ADDR}:${RAY_PORT}" >&2
         exit 1
      fi
      sleep 2
   done
   while [[ ! -f "${RAY_DONE_FILE}" ]]; do
      sleep 10
   done
   exit 0
fi

rm -f "${RAY_DONE_FILE}"

ray start \
   --head \
   --node-ip-address "${MASTER_ADDR}" \
   --port="${RAY_PORT}" \
   --num-gpus 8 \
   --disable-usage-stats \
   --dashboard-host=0.0.0.0 \
   --dashboard-port="${RAY_DASHBOARD_PORT}"

python3 - <<'PY'
import os
import time

import ray

expected = int(os.environ.get("RAY_EXPECTED_NODES", "1"))
address = f"{os.environ['MASTER_ADDR']}:{os.environ['RAY_PORT']}"
ray.init(address=address)
deadline = time.time() + 300
while True:
    alive = [node for node in ray.nodes() if node.get("Alive")]
    print(f"RAY_CLUSTER_NODES {len(alive)}/{expected}", flush=True)
    if len(alive) >= expected:
        break
    if time.time() > deadline:
        raise SystemExit(f"Timed out waiting for Ray nodes: {len(alive)}/{expected}")
    time.sleep(2)
ray.shutdown()
PY

set +e
ray job submit \
   --address "${RAY_ADDRESS:-http://127.0.0.1:${RAY_DASHBOARD_PORT}}" \
   --runtime-env-json "${RUNTIME_ENV_JSON}" \
   -- \
   "${JOB_ARGS[@]}"
rc=$?
set -e
echo "MILES_MLITE_GRPO_DONE rc=${rc}"
exit "${rc}"
