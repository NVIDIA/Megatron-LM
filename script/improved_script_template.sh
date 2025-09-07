#!/bin/bash

# =============================================================================
# Enhanced Training Script Template with Timestamped Logging
# =============================================================================

# Set script name and version
SCRIPT_NAME="$(basename "$0")"
SCRIPT_VERSION="1.0.0"
START_TIME=$(date '+%Y-%m-%d %H:%M:%S')
START_TIMESTAMP=$(date '+%Y%m%d_%H%M%S')

# =============================================================================
# Logging Functions
# =============================================================================

# Function to get current timestamp
get_timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

# Function to log with timestamp
log_info() {
    echo "[$(get_timestamp)] [INFO] $*"
}

log_warn() {
    echo "[$(get_timestamp)] [WARN] $*" >&2
}

log_error() {
    echo "[$(get_timestamp)] [ERROR] $*" >&2
}

log_success() {
    echo "[$(get_timestamp)] [SUCCESS] $*"
}

# Function to log script execution
log_script_start() {
    log_info "=========================================="
    log_info "Script: $SCRIPT_NAME v$SCRIPT_VERSION"
    log_info "Started at: $START_TIME"
    log_info "PID: $$"
    log_info "User: $(whoami)"
    log_info "Host: $(hostname)"
    log_info "Working Directory: $(pwd)"
    log_info "=========================================="
}

log_script_end() {
    local end_time=$(get_timestamp)
    local duration=$(($(date +%s) - $(date -d "$START_TIME" +%s)))
    log_info "=========================================="
    log_info "Script completed at: $end_time"
    log_info "Total execution time: ${duration}s"
    log_info "=========================================="
}

# Function to log configuration
log_config() {
    log_info "Configuration Summary:"
    log_info "  Model: $MODEL_NAME"
    log_info "  Dataset: $DATASET_NAME"
    log_info "  Quantization: $QUANT_TYPE"
    log_info "  Experiment: $EXPERIMENT_NAME"
    log_info "  Checkpoint Path: $CHECKPOINT_PATH"
    log_info "  TensorBoard Path: $TENSORBOARD_PATH"
    log_info "  Data Path: $DATA_PATH"
    log_info "  Tokenizer Path: $TOKENIZER_PATH"
}

# Function to log training parameters
log_training_params() {
    log_info "Training Parameters:"
    log_info "  Micro Batch Size: $MICRO_BATCH_SIZE"
    log_info "  Global Batch Size: $GLOBAL_BATCH_SIZE"
    log_info "  Sequence Length: $SEQ_LENGTH"
    log_info "  Learning Rate: $LR"
    log_info "  Min Learning Rate: $MIN_LR"
    log_info "  Train Samples: $TRAIN_SAMPLES"
    log_info "  Exit Duration: ${EXIT_DURATION_MINS} minutes"
}

# Function to log model parameters
log_model_params() {
    log_info "Model Parameters:"
    log_info "  Tensor Parallel Size: $TP_SIZE"
    log_info "  Context Parallel Size: $CP_SIZE"
    log_info "  Pipeline Parallel Size: $PP_SIZE"
    log_info "  Number of Layers: $NUM_LAYERS"
    log_info "  Hidden Size: $HIDDEN_SIZE"
    log_info "  FFN Hidden Size: $FFN_HIDDEN_SIZE"
    log_info "  Number of Attention Heads: $NUM_ATTENTION_HEADS"
    log_info "  Number of Query Groups: $NUM_QUERY_GROUPS"
    log_info "  KV Channels: $KV_CHANNELS"
    log_info "  Rotary Base: $ROTARY_BASE"
    log_info "  Vocabulary Size: $VOCAB_SIZE"
}

# Function to validate paths
validate_paths() {
    log_info "Validating paths..."
    
    # Check if data path exists
    if [[ ! -d "$(dirname "$DATA_PATH")" ]]; then
        log_error "Data directory does not exist: $(dirname "$DATA_PATH")"
        return 1
    fi
    
    # Check if tokenizer path exists
    if [[ ! -d "$TOKENIZER_PATH" ]]; then
        log_error "Tokenizer directory does not exist: $TOKENIZER_PATH"
        return 1
    fi
    
    # Create checkpoint directory if it doesn't exist
    if [[ ! -d "$(dirname "$CHECKPOINT_PATH")" ]]; then
        log_info "Creating checkpoint directory: $(dirname "$CHECKPOINT_PATH")"
        mkdir -p "$(dirname "$CHECKPOINT_PATH")"
    fi
    
    # Create tensorboard directory if it doesn't exist
    if [[ ! -d "$(dirname "$TENSORBOARD_PATH")" ]]; then
        log_info "Creating tensorboard directory: $(dirname "$TENSORBOARD_PATH")"
        mkdir -p "$(dirname "$TENSORBOARD_PATH")"
    fi
    
    log_success "Path validation completed"
    return 0
}

# Function to check system resources
check_system_resources() {
    log_info "Checking system resources..."
    
    # Check available memory
    local available_memory=$(free -h | awk '/^Mem:/ {print $7}')
    log_info "Available memory: $available_memory"
    
    # Check disk space
    local disk_usage=$(df -h . | awk 'NR==2 {print $4}')
    log_info "Available disk space: $disk_usage"
    
    # Check GPU availability
    if command -v nvidia-smi &> /dev/null; then
        log_info "GPU Information:"
        nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits | while read line; do
            log_info "  $line"
        done
    else
        log_warn "nvidia-smi not found, GPU information unavailable"
    fi
    
    log_success "System resource check completed"
}

# Function to modify quantization types using sed
modify_quantization_types() {
    local quant_type="$1"
    
    log_info "Modifying quantization types to: $quant_type"
    
    # Modify linear layer quantization
    if [[ -f "megatron/core/tensor_parallel/layers.py" ]]; then
        sed -i "s/^\([[:space:]]*custom_quant_type[[:space:]]*=[[:space:]]*\)'[^']*'/\1'$quant_type'/" \
            megatron/core/tensor_parallel/layers.py
        log_info "Modified linear layer quantization in layers.py"
    else
        log_warn "layers.py not found, skipping linear quantization modification"
    fi
    
    # Modify attention quantization
    if [[ -f "megatron/core/transformer/dot_product_attention.py" ]]; then
        sed -i "s/^\([[:space:]]*custom_quant_type[[:space:]]*=[[:space:]]*\)'[^']*'/\1'$quant_type'/" \
            megatron/core/transformer/dot_product_attention.py
        log_info "Modified attention quantization in dot_product_attention.py"
    else
        log_warn "dot_product_attention.py not found, skipping attention quantization modification"
    fi
    
    log_success "Quantization type modifications completed"
}

# Function to build and run command with enhanced logging
build_and_run_command() {
    local training_config="$1"
    local dry_run="$2"
    
    log_info "Building training command..."
    log_info "Training config: $training_config"
    log_info "Dry run: $dry_run"
    
    local cmd=(
        "torchrun" "--nproc_per_node" "8"
        "--nnodes" "1" "--node_rank" "0"
        "--master_addr" "localhost" "--master_port" "6000"
        "pretrain_gpt.py"
        
        # Model arguments
        "--use-mcore-models"
        "--num-layers" "$NUM_LAYERS"
        "--hidden-size" "$HIDDEN_SIZE"
        "--ffn-hidden-size" "$FFN_HIDDEN_SIZE"
        "--num-attention-heads" "$NUM_ATTENTION_HEADS"
        "--group-query-attention"
        "--num-query-groups" "$NUM_QUERY_GROUPS"
        "--kv-channels" "$KV_CHANNELS"
        "--seq-length" "$SEQ_LENGTH"
        "--max-position-embeddings" "$SEQ_LENGTH"
        "--position-embedding-type" "rope"
        "--rotary-base" "$ROTARY_BASE"
        "--rotary-percent" "1.0"
        "--attention-dropout" "0.0"
        "--hidden-dropout" "0.0"
        "--swiglu"
        "--init-method-std" "0.0134"
        "--attention-backend" "fused"
        "--apply-layernorm-1p"
        "--untie-embeddings-and-output-weights"
        "--disable-bias-linear"
        
        # Training arguments
        "--micro-batch-size" "$MICRO_BATCH_SIZE"
        "--global-batch-size" "$GLOBAL_BATCH_SIZE"
        "--train-samples" "$TRAIN_SAMPLES"
        "--lr" "$LR"
        "--min-lr" "$MIN_LR"
        "--lr-decay-style" "cosine"
        "--clip-grad" "1.0"
        "--weight-decay" "0.1"
        "--adam-beta1" "0.9"
        "--adam-beta2" "0.95"
        "--bf16"
        "--grad-reduce-in-bf16"
        "--cross-entropy-loss-fusion"
        "--calculate-per-token-loss"
        "--manual-gc"
        "--empty-unused-memory-level" "1"
        "--exit-duration-in-mins" "$EXIT_DURATION_MINS"
        "--use-distributed-optimizer"
        "--overlap-grad-reduce"
        "--overlap-param-gather"
        
        # Model parallelism
        "--tensor-model-parallel-size" "$TP_SIZE"
        "--context-parallel-size" "$CP_SIZE"
        "--pipeline-model-parallel-size" "$PP_SIZE"
        "--sequence-parallel"
        
        # Data arguments
        "--data-path" "$DATA_PATH"
        "--tokenizer-type" "HuggingFaceTokenizer"
        "--tokenizer-model" "$TOKENIZER_PATH"
        "--vocab-size" "$VOCAB_SIZE"
        "--split" "99,1,0"
        "--no-create-attention-mask-in-dataloader"
        "--num-workers" "1"
        
        # Logging and checkpointing
        "--log-interval" "1"
        "--eval-iters" "32"
        "--eval-interval" "100"
        "--save-interval" "1000"
        "--log-throughput"
        "--ckpt-format" "torch_dist"
        "--distributed-timeout-minutes" "60"
        "--save" "$CHECKPOINT_PATH"
        "--load" "$CHECKPOINT_PATH"
        "--tensorboard-dir" "$TENSORBOARD_PATH"
    )
    
    # Add quantization arguments
    if [[ "$QUANT_TYPE" == "fp8" ]]; then
        log_info "Adding FP8 quantization arguments..."
        cmd+=(
            "--fp8-format" "$FP8_FORMAT"
            "--fp8-amax-history-len" "1024"
            "--fp8-amax-compute-algo" "max"
        )
        
        if [[ "$LINEAR_QUANT" != "None" ]]; then
            log_info "Adding linear quantization: $LINEAR_QUANT"
            cmd+=("--linear-quantization" "$LINEAR_QUANT")
        fi
        if [[ "$ATTENTION_QUANT" != "None" ]]; then
            log_info "Adding attention quantization: $ATTENTION_QUANT"
            cmd+=("--attention-quantization" "$ATTENTION_QUANT")
        fi
    fi
    
    if [[ "$dry_run" == true ]]; then
        log_info "Dry run mode - showing command:"
        echo "${cmd[*]}"
        log_success "Command generation completed (dry run)"
    else
        log_info "Starting training execution..."
        log_info "Command: ${cmd[*]}"
        
        # Set up tensorboard logs path and timestamped logging
        export HOST_TENSORBOARD_LOGS_PATH="$TENSORBOARD_PATH"
        local log_file="${HOST_TENSORBOARD_LOGS_PATH}/training_${EXPERIMENT_NAME}_$(date +'%y-%m-%d_%H-%M-%S').log"
        
        log_info "Training logs will be saved to: $log_file"
        
        # Execute the command with timestamped logging
        if "${cmd[@]}" 2>&1 | tee "$log_file"; then
            log_success "Training completed successfully"
        else
            log_error "Training failed with exit code $?"
            return 1
        fi
    fi
}

# Function to show usage with enhanced formatting
show_usage() {
    cat << EOF
[$(get_timestamp)] [INFO] ==========================================
[$(get_timestamp)] [INFO] Training Script: $SCRIPT_NAME v$SCRIPT_VERSION
[$(get_timestamp)] [INFO] ==========================================

Usage: $0 [OPTIONS]

Training script for ${MODEL_NAME} on ${DATASET_NAME} with ${QUANT_TYPE} quantization.

Options:
    --dry-run                       Show command without executing
    --training-config CONFIG        Training configuration (standard|fast)
    --help                          Show this help message

Examples:
    $0                              # Run with default settings
    $0 --dry-run                    # Show command without executing
    $0 --training-config fast       # Run with fast configuration

Configuration:
    Model: $MODEL_NAME
    Dataset: $DATASET_NAME
    Quantization: $QUANT_TYPE
    Experiment: $EXPERIMENT_NAME

[$(get_timestamp)] [INFO] ==========================================
EOF
}

# Function to parse arguments with enhanced logging
parse_arguments() {
    local training_config="standard"
    local dry_run=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --dry-run)
                dry_run=true
                log_info "Dry run mode enabled"
                shift
                ;;
            --training-config)
                training_config="$2"
                log_info "Training config set to: $training_config"
                shift 2
                ;;
            --help)
                show_usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    log_info "Arguments parsed successfully"
    log_info "Training config: $training_config"
    log_info "Dry run: $dry_run"
}

# Main function with comprehensive logging
main() {
    # Set up error handling
    set -e
    trap 'log_error "Script failed at line $LINENO"' ERR
    trap 'log_script_end' EXIT
    
    # Start logging
    log_script_start
    
    # Parse arguments
    parse_arguments "$@"
    
    # Log configuration
    log_config
    log_model_params
    log_training_params
    
    # Validate paths
    if ! validate_paths; then
        log_error "Path validation failed"
        exit 1
    fi
    
    # Check system resources
    check_system_resources
    
    # Modify quantization types if needed
    if [[ "$QUANT_TYPE" != "bf16" ]] && [[ "$QUANT_TYPE" != "fp16" ]]; then
        # Extract quantization type from QUANT_TYPE (e.g., "linear_hifp8" -> "hifp8")
        local quant_type_to_use=""
        if [[ "$QUANT_TYPE" == *"hifp8"* ]]; then
            quant_type_to_use="hifp8"
        elif [[ "$QUANT_TYPE" == *"mxfp8"* ]]; then
            quant_type_to_use="mxfp8"
        elif [[ "$QUANT_TYPE" == *"mxfp4"* ]]; then
            quant_type_to_use="mxfp4"
        fi
        
        if [[ -n "$quant_type_to_use" ]]; then
            modify_quantization_types "$quant_type_to_use"
        fi
    fi
    
    # Build and run command
    if ! build_and_run_command "$training_config" "$dry_run"; then
        log_error "Command execution failed"
        exit 1
    fi
    
    log_success "Script completed successfully"
}

# Run main function with all arguments
main "$@"
