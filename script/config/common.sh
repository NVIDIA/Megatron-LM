#!/bin/bash

# Common configuration file for all training scripts
# This file contains shared variables and functions used across different experiments

# Set strict error handling
set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# Function to validate required directories
validate_directories() {
    local dirs=("$@")
    for dir in "${dirs[@]}"; do
        if [[ ! -d "$dir" ]]; then
            log_error "Required directory does not exist: $dir"
            return 1
        fi
    done
}

# Function to create directories if they don't exist
ensure_directories() {
    local dirs=("$@")
    for dir in "${dirs[@]}"; do
        if [[ ! -d "$dir" ]]; then
            log_info "Creating directory: $dir"
            mkdir -p "$dir"
        fi
    done
}

# Function to validate required files
validate_files() {
    local files=("$@")
    for file in "${files[@]}"; do
        if [[ ! -f "$file" ]]; then
            log_error "Required file does not exist: $file"
            return 1
        fi
    done
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to validate GPU availability
validate_gpu() {
    if ! command_exists nvidia-smi; then
        log_error "nvidia-smi not found. Please ensure NVIDIA drivers are installed."
        return 1
    fi
    
    local gpu_count=$(nvidia-smi --list-gpus | wc -l)
    if [[ $gpu_count -eq 0 ]]; then
        log_error "No GPUs detected"
        return 1
    fi
    
    log_info "Detected $gpu_count GPU(s)"
    return 0
}

# Function to get available GPU memory
get_gpu_memory() {
    nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1
}

# Function to backup and restore files
backup_file() {
    local file="$1"
    if [[ -f "$file" ]]; then
        cp "$file" "${file}.backup.$(date +%Y%m%d_%H%M%S)"
        log_info "Backed up $file"
    fi
}

restore_file() {
    local file="$1"
    local backup_file=$(ls -t "${file}.backup."* 2>/dev/null | head -1)
    if [[ -n "$backup_file" && -f "$backup_file" ]]; then
        cp "$backup_file" "$file"
        log_info "Restored $file from $backup_file"
    fi
}

# Function to clean up on exit
cleanup_on_exit() {
    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        log_error "Script failed with exit code $exit_code"
    fi
    # Add any cleanup operations here
    exit $exit_code
}

# Set up exit trap
trap cleanup_on_exit EXIT

# Common environment variables
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export NCCL_IB_TIMEOUT=${NCCL_IB_TIMEOUT:-19}
export NVTE_FWD_LAYERNORM_SM_MARGIN=${NVTE_FWD_LAYERNORM_SM_MARGIN:-16}
export NVTE_BWD_LAYERNORM_SM_MARGIN=${NVTE_BWD_LAYERNORM_SM_MARGIN:-16}
export NCCL_P2P_NET_CHUNKSIZE=${NCCL_P2P_NET_CHUNKSIZE:-2097152}
export NCCL_AVOID_RECORD_STREAMS=${NCCL_AVOID_RECORD_STREAMS:-1}

# Common paths
MEGATRON_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PRETRAIN_SCRIPT_PATH="${MEGATRON_ROOT}/pretrain_gpt.py"
TOOLS_DIR="${MEGATRON_ROOT}/tools"

# Validate Megatron-LM structure
if [[ ! -f "$PRETRAIN_SCRIPT_PATH" ]]; then
    log_error "pretrain_gpt.py not found at $PRETRAIN_SCRIPT_PATH"
    log_error "Please ensure you are running from the Megatron-LM repository root"
    exit 1
fi

log_info "Megatron-LM root: $MEGATRON_ROOT"
