#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Do not use this script directly with sbatch. Use it from interactive or let sbatch handle it.
#SBATCH -p batch --account=llmservice_fm_text -t 00:30:00 --nodes=1 --ntasks-per-node=8 --gpus-per-node=8 --job-name=benchmark-refit

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_FWD_LAYERNORM_SM_MARGIN=16
export NVTE_BWD_LAYERNORM_SM_MARGIN=16

# Check if we're in an interactive job
IS_INTERACTIVE=false
if [ "$SLURM_JOB_PARTITION" == "interactive" ] || [ "$SLURM_JOB_PARTITION" == "interactive_singlenode" ]; then
    IS_INTERACTIVE=true
fi

# Determine script path and set PYTHONPATH
SCRIPT_FPATH=$(realpath "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_FPATH")
BENCHMARK_SCRIPT="${SCRIPT_DIR}/benchmark_refit.py"

# Find megatron-rl directory (go up from examples/rl to repo root)
REPO_ROOT=$(dirname $(dirname "$SCRIPT_DIR"))
MEGATRON_RL_DIR="${REPO_ROOT}"
export PYTHONPATH="${MEGATRON_RL_DIR}:${PYTHONPATH:-}"

echo "PYTHONPATH: $PYTHONPATH"
echo "Interactive mode: $IS_INTERACTIVE"

# Container image
if [ -z "${CONTAINER_IMAGE}" ]; then
    SQSH_PATH="/lustre/fsw/portfolios/llmservice/users/ksanthanam/images/adlr+megatron+mamba-dynamic-engine.sqsh"
    if [ -f $SQSH_PATH ]; then
        CONTAINER_IMAGE=$SQSH_PATH
    else
        CONTAINER_IMAGE="/lustre/fsw/portfolios/llmservice/users/ksanthanam/images/adlr+megatron+mamba-dynamic-engine.sqsh"
    fi
fi

# Setup output directory
if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="${SCRIPT_DIR}/benchmark_results"
fi
mkdir -p "${OUTPUT_DIR}/logs"
LOG_DIR="${OUTPUT_DIR}/logs"

# Generate run ID
if [ -z "$RUN_ID" ]; then
    DATETIME=$(date +'%y%m%d_%H%M%S')
    RUN_ID="benchmark_${DATETIME}_${SLURM_JOB_ID:-local}"
fi
LOG_FILE="${LOG_DIR}/${RUN_ID}.log"

echo "Logs: $LOG_FILE"
echo "Arguments: $*"

# Run based on mode (same pattern as train_grpo.sh)
if [ $IS_INTERACTIVE = true ]; then
    echo "Running with torchrun"
    run_cmd="torchrun --nproc-per-node ${SLURM_GPUS_ON_NODE:-8} ${BENCHMARK_SCRIPT} $*"
    echo "Command: ${run_cmd}"
    eval $run_cmd 2>&1 | tee "$LOG_FILE"
    exit_code=${PIPESTATUS[0]}
else
    echo "Running with srun"
    run_cmd="python -u ${BENCHMARK_SCRIPT} $*"
    echo "Command: ${run_cmd}"

    MOUNTS="/home:/home,/lustre:/lustre"
    echo "Container: $CONTAINER_IMAGE"

    if [ -n "$SLURM_JOB_ID" ]; then
        scontrol show job $SLURM_JOB_ID | tee -a "${LOG_DIR}/job_info_${RUN_ID}.log"
    fi

    startup_command="srun -l \
        --verbose \
        --container-image \"${CONTAINER_IMAGE}\" \
        --container-mounts \"$MOUNTS\" \
        --output=${LOG_FILE} \
        sh -c \"export PYTHONPATH=${PYTHONPATH}; ${run_cmd}\""

    echo "$startup_command" | tee -a "${LOG_DIR}/startup_command_${RUN_ID}.log"
    eval $startup_command
    exit_code=$?
fi

if [ $exit_code -eq 0 ]; then
    echo "Benchmark completed successfully! Results: $LOG_FILE"
else
    echo "Benchmark failed with exit code: $exit_code. Check: $LOG_FILE"
fi

exit $exit_code
