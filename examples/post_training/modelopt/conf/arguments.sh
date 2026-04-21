MLM_MODEL_CFG=$1

# Bash coloring
RED='\033[0;31m'
YELLOW='\033[0;33m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
WHITE='\033[0;37m'

# Predefined logging
MLM_ERROR="${RED}ERROR:  ${WHITE}"
MLM_WARNING="${YELLOW}WARNING:${WHITE}"

if [ -z ${SANDBOX_ENV_SETUP} ]; then
    printf "${MLM_WARNING} ${PURPLE}SANDBOX_ENV_SETUP${WHITE} is not set!\n"
else
    source ${SANDBOX_ENV_SETUP}
fi

if [ -z ${SCRIPT_DIR} ]; then
    printf "${MLM_ERROR} Variable ${PURPLE}SCRIPT_DIR${WHITE} must be set!\n"
    exit 1
fi

if [ -z ${MLM_MODEL_CFG} ]; then
    printf "${MLM_ERROR} Variable ${PURPLE}MLM_MODEL_CFG${WHITE} must be set!\n"
    exit 1
fi

if [ -z ${MLM_ENV_SETUP} ]; then
    printf "${MLM_WARNING} Variable ${PURPLE}MLM_ENV_SETUP${WHITE} not set! (only needed when launching with slurm)\n"
else
    source ${MLM_ENV_SETUP}
fi

if [ -z ${MLM_EXTRA_ARGS} ]; then
    printf "${MLM_WARNING} Use ${PURPLE}MLM_EXTRA_ARGS${WHITE} to provide additional arguments!\n"
fi

if [ -z ${MLM_WORK_DIR} ]; then
    export  MLM_WORK_DIR=/tmp/megatron_workspace
    printf "${MLM_WARNING} Variable ${PURPLE}MLM_WORK_DIR${WHITE} is set (default: ${MLM_WORK_DIR})!\n"
fi

if [ -z ${TP} ]; then
    TP=1
    printf "${MLM_WARNING} Variable ${PURPLE}TP${WHITE} not set! (default: ${TP})\n"
fi

if [ -z ${EP} ]; then
    EP=1
    printf "${MLM_WARNING} Variable ${PURPLE}EP${WHITE} not set! (default: ${EP})\n"
fi

if [ -z ${PP} ]; then
    PP=1
    printf "${MLM_WARNING} Variable ${PURPLE}PP${WHITE} not set! (default: ${PP})\n"
fi

if [ -z ${DP} ]; then
    DP=1
    printf "${MLM_WARNING} Variable ${PURPLE}DP${WHITE} not set! (default: ${DP})\n"
fi


if [ -z ${LAUNCH_SCRIPT} ]; then
    LAUNCH_SCRIPT="torchrun --nproc_per_node=$((TP * EP * PP * DP))"
fi

# Install TensorRT Model Optimizer if haven't.
if [ -z ${MLM_SKIP_INSTALL} ]; then
    pip install -r ${SCRIPT_DIR}/requirements.txt
fi

export TOKENIZERS_PARALLELISM=False
export OMP_NUM_THREADS=1
export NCCL_IB_SL=1
export NCCL_IB_TIMEOUT=22
export CUDA_DEVICE_MAX_CONNECTIONS=1

# TE specific warning
printf "${MLM_WARNING} If you see core_attention  _extra_state missing error, use --export-force-local-attention\n"

# Base model specific arguments
if [ -z ${SANDBOX_ROOT} ]; then
    source "${SCRIPT_DIR}/conf/${MLM_MODEL_CFG}.sh"
else
    source "${SANDBOX_ROOT}/conf/model/${MLM_MODEL_CFG}.sh"
fi
