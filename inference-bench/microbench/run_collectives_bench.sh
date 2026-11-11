#!/bin/bash
#SBATCH -t 0:30:00
#SBATCH --gres=gpu:4
#
# NVLS all-gather-v / reduce-scatter-v vs NCCL microbenchmark (collectives_bench.py), one process
# per GPU. Multi-node runs need all nodes in one NVLink domain for the NVLS kernels: pass
# `--segment N` with `-N N`.
#
# Run from the Megatron-LM repo root:
#   source inference-bench/cluster.env
#   sbatch -N 1 inference-bench/microbench/run_collectives_bench.sh
#   sbatch -N 4 --segment 4 inference-bench/microbench/run_collectives_bench.sh
#
# Knobs: HIDDEN (default "1024 2048 4096"), TOKENS (local tokens per GPU; default 1 .. 2048).
# Output: inference-bench/microbench/results/collectives_<gpus>gpu_<jobid>.{csv,log}

SCRIPT=inference-bench/microbench/run_collectives_bench.sh

if [ -z "${COLLECTIVES_BENCH_INNER:-}" ]; then
    cd "${SLURM_SUBMIT_DIR:-$PWD}"
    source "${INFERENCE_BENCH_CONFIG:-inference-bench/cluster.env}"
    NUM_NODES=${SLURM_JOB_NUM_NODES:-1}
    mkdir -p inference-bench/microbench/results
    export OUT="inference-bench/microbench/results/collectives_$((4 * NUM_NODES))gpu_${SLURM_JOB_ID:-local}"
    export MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -1)
    export MASTER_PORT=${MASTER_PORT:-29511}
    export COLLECTIVES_BENCH_INNER=1
    srun -l -N "${NUM_NODES}" -n $((4 * NUM_NODES)) --ntasks-per-node 4 --cpus-per-task 36 \
        --container-image "${INFERENCE_BENCH_SERVE_IMAGE:?set in cluster.env}" \
        --container-mounts "${INFERENCE_BENCH_MOUNTS:?set in cluster.env}" \
        --container-workdir="${PWD}" bash "${SCRIPT}" 2>&1 | tee "${OUT}.log"
    exit "${PIPESTATUS[0]}"
fi

# One rank.
export RANK=${SLURM_PROCID} LOCAL_RANK=${SLURM_LOCALID} WORLD_SIZE=${SLURM_NTASKS}
export TRITON_CACHE_DIR=/tmp/triton_cache_${RANK}
# The script lives under inference-bench/; put the repo root on the path for `import megatron`.
export PYTHONPATH=${PWD}${PYTHONPATH:+:${PYTHONPATH}}
python3 -u inference-bench/microbench/collectives_bench.py \
    --hidden ${HIDDEN:-1024 2048 4096} \
    --tokens ${TOKENS:-1 2 4 8 16 32 64 128 256 512 1024 2048} \
    --csv "${OUT}.csv"
