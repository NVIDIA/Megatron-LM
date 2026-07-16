#!/bin/bash
set -euxo pipefail

# Path to Megatron-MoE-Scripts
export WORKSPACE=$(dirname "$(readlink -f "$0")")

# Benchmarking configurations (must be set)
export MODEL=${MODEL:-"your_own_model"}
export CLUSTER=${CLUSTER:-"template"}
export MCORE_RELEASE_VERSION=${MCORE_RELEASE_VERSION:-"your_own_megatron_version"} # Version and release info
export MEGATRON_PATH=${MEGATRON_PATH:-"$(cd "${WORKSPACE}/../.." && pwd)"} # Path to Megatron-LM (defaults to repo root; this script lives under examples/moe-modelzoo/)
export CONTAINER_IMAGE=${CONTAINER_IMAGE:-"your_own_container_image"} # Path to .sqsh or docker image url

# Load common configurations
source "${WORKSPACE}/runtime_configs/benchmarking/common.conf"
# Load model-specific configurations
source "${WORKSPACE}/runtime_configs/benchmarking/runtime.conf"
# Load cluster configurations
source "${WORKSPACE}/cluster_configs/benchmarking/${CLUSTER}.conf"

# Initialize training parameters
TRAINING_PARAMS=${TRAINING_PARAMS:-""}

# Process training parameters
if [[ -f ${TRAINING_PARAMS_PATH} ]]; then
    envsubst < ${TRAINING_PARAMS_PATH} > ${TRAINING_PARAMS_PATH}.tmp
    TRAINING_PARAMS_PATH=${TRAINING_PARAMS_PATH}.tmp
else
    echo "Error: TRAINING_PARAMS_PATH does not exist: ${TRAINING_PARAMS_PATH}."
    exit 1
fi

# Extract training parameters to export
TRAINING_PARAMS_FROM_CONFIG=$(yq '... comments="" | .MODEL_ARGS | to_entries | .[] | 
    select(.value != "false") | 
    with(select(.value == "true"); .value = "") | 
    [.key + " " + .value] | join("")' ${TRAINING_PARAMS_PATH} | tr '\n' ' ')
TRAINING_PARAMS="${TRAINING_PARAMS} ${TRAINING_PARAMS_FROM_CONFIG}"

# Append any command line arguments to TRAINING_PARAMS
if [[ $# -gt 0 ]]; then
    TRAINING_PARAMS="${TRAINING_PARAMS} $@"
fi

# Extract environment variables to export
ENV_VARS=$(yq '... comments="" | .ENV_VARS | to_entries | .[] | [.key + "=" + .value] | join(" ")' ${TRAINING_PARAMS_PATH})
while IFS='=' read -r KEY VALUE; do
    if [[ -n ${KEY} ]]; then
        export "${KEY}"="${VALUE}"
        echo "${KEY}=${VALUE}"
    fi
done < <(echo "${ENV_VARS}" | tr ' ' '\n')

# Virtual pipeline parallelism arguments
if [[ ${VPP} -gt 1 ]]; then
    if [[ ! "${TRAINING_PARAMS}" =~ "--pipeline-model-parallel-layout" ]] && \
       [[ ! "${TRAINING_PARAMS}" =~ "--num-virtual-stages-per-pipeline-rank" ]] && \
       [[ ! "${TRAINING_PARAMS}" =~ "--num-layers-per-virtual-pipeline-stage" ]]; then
        TRAINING_PARAMS="${TRAINING_PARAMS} --num-virtual-stages-per-pipeline-rank ${VPP}"
    fi
fi

# Uneven pipeline parallelism arguments
if [[ $((NUM_LAYERS % PP)) -ne 0 ]]; then
    if [[ ! "${TRAINING_PARAMS}" =~ "--pipeline-model-parallel-layout" ]]; then
        TRAINING_PARAMS="${TRAINING_PARAMS} --decoder-first-pipeline-num-layers ${PP_FIRST} --decoder-last-pipeline-num-layers ${PP_LAST}"
    fi
fi

OPTIMIZER_OFFLOAD=${OPTIMIZER_OFFLOAD:-0}
if [[ ${OPTIMIZER_OFFLOAD} == 1 ]]; then
    TRAINING_PARAMS="${TRAINING_PARAMS} --optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d"
fi

DISPATCHER=${DISPATCHER:-"deepep"}
if [[ ${DISPATCHER} == "alltoall" ]]; then
    TRAINING_PARAMS="${TRAINING_PARAMS} --moe-token-dispatcher-type alltoall"
elif [[ ${DISPATCHER} == "deepep" ]]; then
    TRAINING_PARAMS="${TRAINING_PARAMS} --moe-token-dispatcher-type flex --moe-flex-dispatcher-backend deepep"
elif [[ ${DISPATCHER} == "hybridep" ]]; then
    TRAINING_PARAMS="${TRAINING_PARAMS} --moe-token-dispatcher-type flex --moe-flex-dispatcher-backend hybridep --moe-hybridep-num-sms 32"
fi

# FP8 arguments
if [[ ${PR} == "fp8" ]]; then
    TRAINING_PARAMS="${TRAINING_PARAMS} --fp8-recipe blockwise --fp8-format e4m3"
    if [[ ${OPTIMIZER_OFFLOAD} == 0 ]]; then
        TRAINING_PARAMS="${TRAINING_PARAMS} --fp8-param-gather" # Optimizer CPU offload does not support fp8 param gather now.
    fi
    TRAINING_PARAMS="${TRAINING_PARAMS} --use-precision-aware-optimizer --main-grads-dtype fp32 --main-params-dtype fp32 --exp-avg-dtype bf16 --exp-avg-sq-dtype bf16"
    TRAINING_PARAMS="${TRAINING_PARAMS} --moe-router-padding-for-fp8"
fi

if [[ ${PR} == "bf16" ]]; then
    TRAINING_PARAMS="${TRAINING_PARAMS} --tp-comm-overlap"
fi

if [[ ${PR} == "mxfp8" ]]; then
    TRAINING_PARAMS="${TRAINING_PARAMS} --fp8-recipe mxfp8 --fp8-format e4m3"
    if [[ ${OPTIMIZER_OFFLOAD} == 0 ]]; then
        TRAINING_PARAMS="${TRAINING_PARAMS} --fp8-param-gather --reuse-grad-buf-for-mxfp8-param-ag" # Optimizer CPU offload does not support fp8 param gather now.
        # Temporary workaround for NaN issue with mxfp8.
        TRAINING_PARAMS="${TRAINING_PARAMS} --overlap-grad-reduce --overlap-param-gather"
    fi
    TRAINING_PARAMS="${TRAINING_PARAMS} --use-precision-aware-optimizer --main-grads-dtype fp32 --main-params-dtype fp32 --exp-avg-dtype bf16 --exp-avg-sq-dtype bf16"
    TRAINING_PARAMS="${TRAINING_PARAMS} --moe-router-padding-for-quantization"
fi

# 1F1B overlapping arguments and environment variables
A2A_OVERLAP=${A2A_OVERLAP:-0}
if [[ ${A2A_OVERLAP} == 1 ]]; then
    export CUDA_DEVICE_MAX_CONNECTIONS=32
    export NVTE_FWD_LAYERNORM_SM_MARGIN=24
    export NVTE_BWD_LAYERNORM_SM_MARGIN=24
    TRAINING_PARAMS="${TRAINING_PARAMS} --delay-wgrad-compute --overlap-moe-expert-parallel-comm"
else
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    export NVTE_FWD_LAYERNORM_SM_MARGIN=0
    export NVTE_BWD_LAYERNORM_SM_MARGIN=0
fi

# Long context arguments
MAX_POSITION_EMBEDDINGS_FROM_CONFIG=$(yq '.MODEL_ARGS."--max-position-embeddings"' ${TRAINING_PARAMS_PATH})
# If --max-position-embeddings is found in the config file and SEQ_LEN is set,
# check whether SEQ_LEN is greater than --max-position-embeddings,
# if so, set --max-position-embeddings to SEQ_LEN.
if [[ -n ${MAX_POSITION_EMBEDDINGS_FROM_CONFIG} ]] && [[ -n ${SEQ_LEN} ]]; then
    if [[ ${SEQ_LEN} -gt ${MAX_POSITION_EMBEDDINGS_FROM_CONFIG} ]]; then
        TRAINING_PARAMS="${TRAINING_PARAMS} --max-position-embeddings ${SEQ_LEN}"
    fi
fi

# Profile command
if [[ ${PROFILE} == "1" ]] || [[ ${PROFILE} == "nsys" ]]; then
    NSYS_PATH="${OUTPUT_PATH}/nsys"
    DATETIME=$(date +'date_%y-%m-%d_time_%H-%M-%S')
    mkdir -p "${NSYS_PATH}"
    PROFILE_CMD="nsys profile --sample=none --cpuctxsw=none -t cuda,nvtx \
        --capture-range=cudaProfilerApi \
        --capture-range-end=stop \
        --cuda-graph-trace=node \
        -f true -x true \
        -o ${NSYS_PATH}/${MODEL}-benchmarking-${DATETIME}"
    TRAINING_PARAMS="${TRAINING_PARAMS} --profile --profile-step-start 23 --profile-step-end 25 --profile-ranks 0 "
elif [[ ${PROFILE} == "torch" ]]; then
    PROFILE_CMD=""
    TRAINING_PARAMS="${TRAINING_PARAMS} --profile --profile-step-start 23 --profile-step-end 25 --profile-ranks 0 --use-pytorch-profiler"
else
    PROFILE_CMD=""
fi

export BINDPCIE_PATH=${BINDPCIE_PATH:-""}

# Export training command
export TRAINING_CMD="${PROFILE_CMD} ${BINDPCIE_PATH} python ${TRAINING_SCRIPT_PATH} ${TRAINING_PARAMS}"

# SLURM settings
SLURM_LOGS="${OUTPUT_PATH}/slurm_logs"
mkdir -p ${SLURM_LOGS} || {
    echo "Error: Failed to create SLURM logs directory ${SLURM_LOGS}"
    exit 1
}

# Generate timestamp for job name
TIMESTAMP=$(date +'%y%m%d_%H%M%S')

# Set SBATCH_ARG based on cluster type
SBATCH_ARG=""
export GB200_CLUSTER=${GB200_CLUSTER:-0}
if [[ "${GB200_CLUSTER}" == "1" ]]; then
    N_TASKS_PER_NODE=4
    if [[ -n ${SEGMENT} ]]; then
        SBATCH_ARG+=" --segment=${SEGMENT}"
    fi
    export NVLINK_DOMAIN_SIZE=72
else
    N_TASKS_PER_NODE=8
fi

# Submit SLURM job
set +e
if [[ ${DRY_RUN:-0} -eq 1 ]]; then
    echo "=== DRY RUN - SLURM Job Script ==="
    cat <<EOF
#!/bin/bash

#SBATCH --nodes=${NNODES}
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --ntasks-per-node=${N_TASKS_PER_NODE}
#SBATCH --time=${RUN_TIME}
#SBATCH --job-name=${ACCOUNT}-moe-${RUN_NAME}-${TIMESTAMP}
#SBATCH --dependency=singleton
#SBATCH --output=${WORKSPACE}/slurm.log
#SBATCH --exclusive

export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-"/tmp/triton_cache_\${SLURM_NODEID}"}

srun \
    --mpi=pmix -l \
    --no-container-mount-home \
    --container-image=${CONTAINER_IMAGE} \
    --container-mounts=${CONTAINER_MOUNTS} \
    --container-workdir=${MEGATRON_PATH} \
    bash -c \\\${TRAINING_CMD} 2>&1 | tee ${SLURM_LOGS}/\${SLURM_JOB_ID}.log
EOF
    echo "=== End of DRY RUN ==="
    echo "=== Full Training Command ==="
    echo "${TRAINING_CMD}"
    echo "=== End of Full Training Command ==="
else
    sbatch ${SBATCH_ARG} <<EOF
#!/bin/bash

#SBATCH --nodes=${NNODES}
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --ntasks-per-node=${N_TASKS_PER_NODE}
#SBATCH --time=${RUN_TIME}
#SBATCH --job-name=${ACCOUNT}-moe-${RUN_NAME}-${TIMESTAMP}
#SBATCH --output=${WORKSPACE}/slurm.log
#SBATCH --exclusive

export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-"/tmp/triton_cache_\${SLURM_NODEID}"}

srun \
    --mpi=pmix -l \
    --no-container-mount-home \
    --container-image=${CONTAINER_IMAGE} \
    --container-mounts=${CONTAINER_MOUNTS} \
    --container-workdir=${MEGATRON_PATH} \
    bash -c \\\${TRAINING_CMD} 2>&1 | tee ${SLURM_LOGS}/\${SLURM_JOB_ID}.log
EOF
fi
set -e
