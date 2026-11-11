#!/bin/bash
#SBATCH -t 4:00:00
#SBATCH -N 1
# GPU partitions reject GPU-less jobs; taking all 4 also keeps the node (and its CPUs) to ourselves.
#SBATCH --gres=gpu:4
#
# Collect natural-length MATH-500 rollouts (collect_rollouts.py) from its own node against a server
# started by server/launch_minf.sh without BENCH. Waits for the server, writes rollouts.jsonl and
# rollouts.log into the server's experiment directory, then touches <exp dir>/done (shuts the
# server down) unless KEEP_SERVER=1.
#
# Run from the Megatron-LM repo root:
#   source inference-bench/cluster.env
#   SERVER_JOB=$(sbatch --parsable --export=ALL,EXP_NAME=rollouts_superv3,MODEL=superv3 inference-bench/server/launch_minf.sh)
#   sbatch --dependency=after:${SERVER_JOB} --export=ALL,SERVER_JOB=${SERVER_JOB} inference-bench/rollouts/run_rollouts.sh
#
# Knobs: SERVER_JOB (required), DATASET (math500 | hard_math, math500), SAMPLES (rollouts per
# problem, 4), CONCURRENCY (in flight, 64),
# MAX_TOKENS (safety cap, 131072), LIMIT (first N problems, 0 = all), KEEP_SERVER (0).

SCRIPT=inference-bench/rollouts/run_rollouts.sh

if [ -z "${INFERENCE_BENCH_INNER:-}" ]; then
    cd "${SLURM_SUBMIT_DIR:-$PWD}"
    source "${INFERENCE_BENCH_CONFIG:-inference-bench/cluster.env}"
    : "${SERVER_JOB:?set SERVER_JOB to the launch_minf.sh job id}"
    export INFERENCE_BENCH_INNER=1
    COPY="inference-bench/experiments/.rollouts_${SLURM_JOB_ID:-$$}.sh"
    mkdir -p inference-bench/experiments && cp "${SCRIPT}" "${COPY}"
    srun -N 1 -n 1 \
        --container-image "${INFERENCE_BENCH_SERVE_IMAGE:?set in cluster.env}" \
        --container-mounts "${INFERENCE_BENCH_MOUNTS:?set in cluster.env}" \
        --container-workdir="${PWD}" bash "${COPY}"
    status=$?
    rm -f "${COPY}"
    exit ${status}
fi

EXP_DIR=
until [ -n "${EXP_DIR}" ]; do
    EXP_DIR=$(dirname "$(grep -lx "SLURM_JOB_ID=${SERVER_JOB}" inference-bench/experiments/*/config.env 2>/dev/null | head -1)" 2>/dev/null)
    [ "${EXP_DIR}" == "." ] && EXP_DIR=
    [ -z "${EXP_DIR}" ] && sleep 10
done
echo "Server experiment directory: ${EXP_DIR}"
until [ -f "${EXP_DIR}/host" ]; do
    if [ -f "${EXP_DIR}/done" ]; then
        echo "Server exited before becoming ready; see ${EXP_DIR}/server_rank0.log"
        exit 1
    fi
    sleep 10
done
SERVER="http://$(cat "${EXP_DIR}/host"):5000"
echo "Rollouts from $(hostname) against ${SERVER}"
OUT="${EXP_DIR}/rollouts_${DATASET:-math500}"
echo "ROLLOUT_JOB=${SLURM_JOB_ID} DATASET=${DATASET:-math500} SAMPLES=${SAMPLES:-4} CONCURRENCY=${CONCURRENCY:-64} MAX_TOKENS=${MAX_TOKENS:-131072} LIMIT=${LIMIT:-0}" >> "${EXP_DIR}/config.env"

ulimit -u unlimited 2>/dev/null
python3 -u inference-bench/rollouts/collect_rollouts.py --server "${SERVER}" --dataset "${DATASET:-math500}" \
    --out "${OUT}.jsonl" --samples "${SAMPLES:-4}" --concurrency "${CONCURRENCY:-64}" \
    --max-tokens "${MAX_TOKENS:-131072}" --limit "${LIMIT:-0}" 2>&1 | tee "${OUT}.log"
status=${PIPESTATUS[0]}
[ "${KEEP_SERVER:-0}" == "1" ] || touch "${EXP_DIR}/done"
exit ${status}
