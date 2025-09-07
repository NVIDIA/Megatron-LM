#!/bin/bash

# Training configuration file
# Contains training-specific parameters and configurations

# Source common configuration
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

# Training configurations
declare -A TRAINING_CONFIGS

# Standard training configuration
TRAINING_CONFIGS["standard"]="
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=128
SEQ_LENGTH=8192
MAX_POSITION_EMBEDDINGS=8192
TRAIN_SAMPLES=47340000
LR_DECAY_SAMPLES=47245280
LR_WARMUP_SAMPLES=94720
LR=0.00015
MIN_LR=0.00001
DECOUPLED_LR=5.0e-4
DECOUPLED_MIN_LR=4.5e-5
LR_DECAY_STYLE=cosine
CLIP_GRAD=1.0
WEIGHT_DECAY=0.1
ADAM_BETA1=0.9
ADAM_BETA2=0.95
EXIT_DURATION_MINS=235000000
"

# Fast training configuration (for testing)
TRAINING_CONFIGS["fast"]="
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=32
SEQ_LENGTH=2048
MAX_POSITION_EMBEDDINGS=2048
TRAIN_SAMPLES=1000
LR_DECAY_SAMPLES=800
LR_WARMUP_SAMPLES=100
LR=0.0001
MIN_LR=0.00001
DECOUPLED_LR=5.0e-4
DECOUPLED_MIN_LR=4.5e-5
LR_DECAY_STYLE=cosine
CLIP_GRAD=1.0
WEIGHT_DECAY=0.1
ADAM_BETA1=0.9
ADAM_BETA2=0.95
EXIT_DURATION_MINS=60
"

# Function to get training configuration
get_training_config() {
    local config_name="$1"
    if [[ -z "${TRAINING_CONFIGS[$config_name]:-}" ]]; then
        log_error "Unknown training configuration: $config_name"
        log_error "Available configurations: ${!TRAINING_CONFIGS[*]}"
        return 1
    fi
    echo "${TRAINING_CONFIGS[$config_name]}"
}

# Function to set training parameters
set_training_params() {
    local config_name="$1"
    local config=$(get_training_config "$config_name")
    
    if [[ $? -ne 0 ]]; then
        return 1
    fi
    
    # Evaluate the configuration string
    eval "$config"
    
    log_info "Loaded training configuration: $config_name"
    log_info "GLOBAL_BATCH_SIZE=$GLOBAL_BATCH_SIZE, SEQ_LENGTH=$SEQ_LENGTH"
}

# Data type configurations
declare -A DTYPE_CONFIGS

# FP8 configuration
DTYPE_CONFIGS["fp8"]="
DTYPE=fp8
FP8_FORMAT=hybrid
FP8_AMAX_HISTORY_LEN=1024
FP8_AMAX_COMPUTE_ALGO=max
FP8_PARAM_GATHER=true
"

# BF16 configuration
DTYPE_CONFIGS["bf16"]="
DTYPE=bf16
"

# FP16 configuration
DTYPE_CONFIGS["fp16"]="
DTYPE=fp16
"

# Function to get data type configuration
get_dtype_config() {
    local dtype="$1"
    if [[ -z "${DTYPE_CONFIGS[$dtype]:-}" ]]; then
        log_error "Unknown data type: $dtype"
        log_error "Available data types: ${!DTYPE_CONFIGS[*]}"
        return 1
    fi
    echo "${DTYPE_CONFIGS[$dtype]}"
}

# Function to set data type parameters
set_dtype_params() {
    local dtype="$1"
    local config=$(get_dtype_config "$dtype")
    
    if [[ $? -ne 0 ]]; then
        return 1
    fi
    
    # Evaluate the configuration string
    eval "$config"
    
    log_info "Loaded data type configuration: $dtype"
}

# Distributed training configurations
declare -A DISTRIBUTED_CONFIGS

# Single node configuration
DISTRIBUTED_CONFIGS["single_node"]="
GPUS_PER_NODE=8
NUM_NODES=1
MASTER_ADDR=localhost
MASTER_PORT=6000
NODE_RANK=0
"

# Multi-node configuration (example)
DISTRIBUTED_CONFIGS["multi_node"]="
GPUS_PER_NODE=8
NUM_NODES=2
MASTER_ADDR=192.168.1.100
MASTER_PORT=6000
NODE_RANK=0
"

# Function to get distributed configuration
get_distributed_config() {
    local config_name="$1"
    if [[ -z "${DISTRIBUTED_CONFIGS[$config_name]:-}" ]]; then
        log_error "Unknown distributed configuration: $config_name"
        log_error "Available configurations: ${!DISTRIBUTED_CONFIGS[*]}"
        return 1
    fi
    echo "${DISTRIBUTED_CONFIGS[$config_name]}"
}

# Function to set distributed parameters
set_distributed_params() {
    local config_name="$1"
    local config=$(get_distributed_config "$config_name")
    
    if [[ $? -ne 0 ]]; then
        return 1
    fi
    
    # Evaluate the configuration string
    eval "$config"
    
    # Calculate world size
    WORLD_SIZE=$(($GPUS_PER_NODE * $NUM_NODES))
    
    log_info "Loaded distributed configuration: $config_name"
    log_info "GPUS_PER_NODE=$GPUS_PER_NODE, NUM_NODES=$NUM_NODES, WORLD_SIZE=$WORLD_SIZE"
}

# Function to validate training configuration
validate_training_config() {
    local training_config="$1"
    local dtype="$2"
    local distributed_config="$3"
    
    # Validate training configuration
    set_training_params "$training_config" || return 1
    
    # Validate data type configuration
    set_dtype_params "$dtype" || return 1
    
    # Validate distributed configuration
    set_distributed_params "$distributed_config" || return 1
    
    log_success "Training configuration validated"
    return 0
}
