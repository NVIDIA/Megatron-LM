#!/bin/bash
# One-node server check, run as one task per GPU inside the serving container (e.g. a container job
# with 4 tasks on 1 node): every task runs one server rank of launch_minf.sh (its inner mode); rank 0
# waits for the server, runs sanity.py (greedy checkable answers) and a short steady_client level,
# then touches <exp dir>/done, which shuts every rank down.
#
# Knobs (environment): EXP_NAME (check), MODEL (superv3), TP (4), EP (4), BENCH concurrency (128),
# STEADY_DATASET (wildchat), STEADY_WINDOWS (2), STEADY_WINDOW (300 s), plus any launch_minf.sh
# setting (e.g. CUDA_GRAPH_MAX_TOKENS). Output: inference-bench/experiments/<job id>_<EXP_NAME>/
# with server_rank*.log, sanity.log, steady.jsonl.

set -u
cd /home/sidsingh/megatron-lm
source inference-bench/cluster.env
export MODEL=${MODEL:-superv3} TP=${TP:-4} EP=${EP:-4} ETP=1 BUFFER_GB=${BUFFER_GB:-50}
export MAMBA_RATIO=${MAMBA_RATIO:-0.4} PREFIX_CACHING=0 COMBINE_PRECISION=bf16 LOG_INTERVAL=19
export INFERENCE_BENCH_INNER=1 EXP_NAME=${EXP_NAME:-check}
export EXP_DIR="inference-bench/experiments/${SLURM_JOB_ID}_${EXP_NAME}"
mkdir -p "${EXP_DIR}"

# launch_minf.sh's rank 0 runs client.py itself when BENCH is set; the clients here run below.
BENCH= bash inference-bench/server/launch_minf.sh &
server=$!
if [ "${SLURM_PROCID}" == "0" ]; then
    until [ -f "${EXP_DIR}/host" ] || [ -f "${EXP_DIR}/done" ]; do sleep 5; done
    if [ -f "${EXP_DIR}/host" ]; then
        timeout 900 python3 -u inference-bench/server/sanity.py --server http://localhost:5000 \
            --batch 16 > "${EXP_DIR}/sanity.log" 2>&1
        python3 -u inference-bench/server/steady_client.py --server http://localhost:5000 \
            --concurrency "${BENCH:-128}" --dataset "${STEADY_DATASET:-wildchat}" \
            --windows "${STEADY_WINDOWS:-2}" --window "${STEADY_WINDOW:-300}" --warmup 60 \
            --out "${EXP_DIR}/steady.jsonl" > "${EXP_DIR}/bench.log" 2>&1
    fi
    touch "${EXP_DIR}/done"
fi
wait "${server}"
