#!/bin/bash

# Model configuration file
# Contains model-specific parameters for different architectures

# Source common configuration
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

# Model configurations
declare -A MODEL_CONFIGS

# LLaMA 3 8B Configuration
MODEL_CONFIGS["llama3_8b"]="
TP_SIZE=2
CP_SIZE=1
PP_SIZE=4
NUM_LAYERS=32
HIDDEN_SIZE=4096
FFN_HIDDEN_SIZE=14336
NUM_ATTENTION_HEADS=32
NUM_QUERY_GROUPS=8
KV_CHANNELS=128
ROTARY_BASE=1000000
VOCAB_SIZE=128256
"

# LLaMA 3.2 1B Configuration
MODEL_CONFIGS["llama32_1b"]="
TP_SIZE=4
CP_SIZE=1
PP_SIZE=1
NUM_LAYERS=16
HIDDEN_SIZE=2048
FFN_HIDDEN_SIZE=8192
NUM_ATTENTION_HEADS=32
NUM_QUERY_GROUPS=8
KV_CHANNELS=128
ROTARY_BASE=500000
VOCAB_SIZE=128256
"

# Function to get model configuration
get_model_config() {
    local model_name="$1"
    if [[ -z "${MODEL_CONFIGS[$model_name]:-}" ]]; then
        log_error "Unknown model: $model_name"
        log_error "Available models: ${!MODEL_CONFIGS[*]}"
        return 1
    fi
    echo "${MODEL_CONFIGS[$model_name]}"
}

# Function to set model parameters
set_model_params() {
    local model_name="$1"
    local config=$(get_model_config "$model_name")
    
    if [[ $? -ne 0 ]]; then
        return 1
    fi
    
    # Evaluate the configuration string
    eval "$config"
    
    log_info "Loaded configuration for model: $model_name"
    log_info "TP_SIZE=$TP_SIZE, CP_SIZE=$CP_SIZE, PP_SIZE=$PP_SIZE"
    log_info "NUM_LAYERS=$NUM_LAYERS, HIDDEN_SIZE=$HIDDEN_SIZE"
}

# Function to validate model configuration
validate_model_config() {
    local model_name="$1"
    
    # Check if model configuration exists
    if [[ -z "${MODEL_CONFIGS[$model_name]:-}" ]]; then
        log_error "Model configuration not found: $model_name"
        return 1
    fi
    
    # Load and validate parameters
    set_model_params "$model_name" || return 1
    
    # Validate required parameters
    local required_params=("TP_SIZE" "CP_SIZE" "PP_SIZE" "NUM_LAYERS" "HIDDEN_SIZE" "VOCAB_SIZE")
    for param in "${required_params[@]}"; do
        if [[ -z "${!param:-}" ]]; then
            log_error "Missing required parameter: $param"
            return 1
        fi
    done
    
    log_success "Model configuration validated: $model_name"
    return 0
}

# Function to get model-specific paths
get_model_paths() {
    local model_name="$1"
    local base_path="${MEGATRON_ROOT}"
    
    case "$model_name" in
        "llama3_8b")
            echo "TOKENIZER_PATH=${base_path}/model/llama3"
            echo "CHECKPOINT_BASE=${base_path}/checkpoints/llama3_8b"
            echo "TENSORBOARD_BASE=${base_path}/tensorboard_logs/llama3_8b"
            ;;
        "llama32_1b")
            echo "TOKENIZER_PATH=${base_path}/model/llama3.2-1b"
            echo "CHECKPOINT_BASE=${base_path}/checkpoints/llama32_1b"
            echo "TENSORBOARD_BASE=${base_path}/tensorboard_logs/llama32_1b"
            ;;
        *)
            log_error "Unknown model for path generation: $model_name"
            return 1
            ;;
    esac
}
