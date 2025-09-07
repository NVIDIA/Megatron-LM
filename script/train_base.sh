#!/bin/bash

# Base training script for Megatron-LM
# This script provides a robust foundation for training with proper error handling and validation

# Source configuration files
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config/common.sh"
source "${SCRIPT_DIR}/config/models.sh"
source "${SCRIPT_DIR}/config/training.sh"

# Default parameters
MODEL_NAME=""
TRAINING_CONFIG="standard"
DTYPE="fp8"
DISTRIBUTED_CONFIG="single_node"
EXPERIMENT_NAME=""
CHECKPOINT_PATH=""
TENSORBOARD_PATH=""
TOKENIZER_PATH=""
DATA_PATH=""
USE_MOCK_DATA=false
DRY_RUN=false

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Required Options:
    --model MODEL_NAME              Model to train (llama3_8b, llama32_1b)
    --experiment-name NAME          Name for this experiment
    --checkpoint-path PATH          Path to save checkpoints
    --tensorboard-path PATH         Path to save tensorboard logs

Data Options (choose one):
    --data-path PATH                Path to training data
    --tokenizer-path PATH           Path to tokenizer
    --use-mock-data                 Use mock data for testing

Training Options:
    --training-config CONFIG        Training configuration (standard, fast) [default: standard]
    --dtype DTYPE                   Data type (fp8, bf16, fp16) [default: fp8]
    --distributed-config CONFIG     Distributed configuration (single_node, multi_node) [default: single_node]

Other Options:
    --dry-run                       Show what would be executed without running
    --help                          Show this help message

Examples:
    # Train LLaMA 3 8B with real data
    $0 --model llama3_8b --experiment-name wikipedia_fp8 \\
       --checkpoint-path checkpoints/llama3_8b/wikipedia_fp8 \\
       --tensorboard-path tensorboard_logs/llama3_8b/wikipedia_fp8 \\
       --data-path dataset/wikipedia_processed/wikipedia_processed_text_document \\
       --tokenizer-path model/llama3

    # Train with mock data for testing
    $0 --model llama32_1b --experiment-name test_run \\
       --checkpoint-path checkpoints/llama32_1b/test \\
       --tensorboard-path tensorboard_logs/llama32_1b/test \\
       --use-mock-data --training-config fast

EOF
}

# Function to parse command line arguments
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --model)
                MODEL_NAME="$2"
                shift 2
                ;;
            --experiment-name)
                EXPERIMENT_NAME="$2"
                shift 2
                ;;
            --checkpoint-path)
                CHECKPOINT_PATH="$2"
                shift 2
                ;;
            --tensorboard-path)
                TENSORBOARD_PATH="$2"
                shift 2
                ;;
            --data-path)
                DATA_PATH="$2"
                shift 2
                ;;
            --tokenizer-path)
                TOKENIZER_PATH="$2"
                shift 2
                ;;
            --use-mock-data)
                USE_MOCK_DATA=true
                shift
                ;;
            --training-config)
                TRAINING_CONFIG="$2"
                shift 2
                ;;
            --dtype)
                DTYPE="$2"
                shift 2
                ;;
            --distributed-config)
                DISTRIBUTED_CONFIG="$2"
                shift 2
                ;;
            --dry-run)
                DRY_RUN=true
                shift
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
}

# Function to validate arguments
validate_arguments() {
    local errors=0
    
    # Check required arguments
    if [[ -z "$MODEL_NAME" ]]; then
        log_error "Model name is required (--model)"
        errors=$((errors + 1))
    fi
    
    if [[ -z "$EXPERIMENT_NAME" ]]; then
        log_error "Experiment name is required (--experiment-name)"
        errors=$((errors + 1))
    fi
    
    if [[ -z "$CHECKPOINT_PATH" ]]; then
        log_error "Checkpoint path is required (--checkpoint-path)"
        errors=$((errors + 1))
    fi
    
    if [[ -z "$TENSORBOARD_PATH" ]]; then
        log_error "Tensorboard path is required (--tensorboard-path)"
        errors=$((errors + 1))
    fi
    
    # Check data configuration
    if [[ "$USE_MOCK_DATA" == false ]]; then
        if [[ -z "$DATA_PATH" ]]; then
            log_error "Data path is required when not using mock data (--data-path)"
            errors=$((errors + 1))
        fi
        
        if [[ -z "$TOKENIZER_PATH" ]]; then
            log_error "Tokenizer path is required when not using mock data (--tokenizer-path)"
            errors=$((errors + 1))
        fi
    fi
    
    if [[ $errors -gt 0 ]]; then
        log_error "Validation failed with $errors error(s)"
        return 1
    fi
    
    return 0
}

# Function to setup paths
setup_paths() {
    # Get model-specific paths if not provided
    if [[ -z "$TOKENIZER_PATH" && "$USE_MOCK_DATA" == false ]]; then
        local model_paths=$(get_model_paths "$MODEL_NAME")
        eval "$model_paths"
        TOKENIZER_PATH="$TOKENIZER_PATH"
    fi
    
    # Create necessary directories
    ensure_directories "$(dirname "$CHECKPOINT_PATH")" "$(dirname "$TENSORBOARD_PATH")"
    
    # Create data cache directory
    DATA_CACHE_PATH="${MEGATRON_ROOT}/benchmark_cache_${MODEL_NAME}_${DTYPE}"
    ensure_directories "$DATA_CACHE_PATH"
}

# Function to build training arguments
build_training_args() {
    # Load configurations
    set_model_params "$MODEL_NAME" || return 1
    set_training_params "$TRAINING_CONFIG" || return 1
    set_dtype_params "$DTYPE" || return 1
    set_distributed_params "$DISTRIBUTED_CONFIG" || return 1
    
    # Build distributed arguments
    DISTRIBUTED_ARGS=(
        --nproc_per_node $GPUS_PER_NODE
        --nnodes $NUM_NODES
        --node_rank $NODE_RANK
        --master_addr $MASTER_ADDR
        --master_port $MASTER_PORT
    )
    
    # Build model arguments
    MODEL_ARGS=(
        --use-mcore-models
        --num-layers $NUM_LAYERS
        --hidden-size $HIDDEN_SIZE
        --ffn-hidden-size $FFN_HIDDEN_SIZE
        --num-attention-heads $NUM_ATTENTION_HEADS
        --group-query-attention
        --num-query-groups $NUM_QUERY_GROUPS
        --kv-channels $KV_CHANNELS
        --seq-length $SEQ_LENGTH
        --max-position-embeddings $MAX_POSITION_EMBEDDINGS
        --position-embedding-type rope
        --rotary-base $ROTARY_BASE
        --rotary-percent 1.0
        --attention-dropout 0.0
        --hidden-dropout 0.0
        --swiglu
        --init-method-std 0.0134
        --attention-backend fused
        --apply-layernorm-1p
        --untie-embeddings-and-output-weights
        --disable-bias-linear
    )
    
    # Add model-specific arguments
    if [[ "$MODEL_NAME" == "llama32_1b" ]]; then
        MODEL_ARGS+=(--transformer-impl local)
    fi
    
    # Build training arguments
    TRAINING_ARGS=(
        --micro-batch-size $MICRO_BATCH_SIZE
        --global-batch-size $GLOBAL_BATCH_SIZE
        --train-samples $TRAIN_SAMPLES
        --lr-decay-samples $LR_DECAY_SAMPLES
        --lr-warmup-samples $LR_WARMUP_SAMPLES
        --lr $LR
        --min-lr $MIN_LR
        --decoupled-lr $DECOUPLED_LR
        --decoupled-min-lr $DECOUPLED_MIN_LR
        --lr-decay-style $LR_DECAY_STYLE
        --clip-grad $CLIP_GRAD
        --weight-decay $WEIGHT_DECAY
        --adam-beta1 $ADAM_BETA1
        --adam-beta2 $ADAM_BETA2
        --bf16
        --grad-reduce-in-bf16
        --cross-entropy-loss-fusion
        --calculate-per-token-loss
        --manual-gc
        --empty-unused-memory-level 1
        --exit-duration-in-mins $EXIT_DURATION_MINS
        --use-distributed-optimizer
        --overlap-grad-reduce
        --overlap-param-gather
    )
    
    # Build data type arguments
    DTYPE_ARGS=()
    if [[ "$DTYPE" == "fp8" ]]; then
        DTYPE_ARGS+=(
            --fp8-format $FP8_FORMAT
            --fp8-amax-history-len $FP8_AMAX_HISTORY_LEN
            --fp8-amax-compute-algo $FP8_AMAX_COMPUTE_ALGO
        )
        if [[ "$FP8_PARAM_GATHER" == "true" ]]; then
            DTYPE_ARGS+=(--fp8-param-gather)
        fi
    fi
    
    # Build model parallelism arguments
    MODEL_PARALLEL_ARGS=(
        --tensor-model-parallel-size $TP_SIZE
        --context-parallel-size $CP_SIZE
        --pipeline-model-parallel-size $PP_SIZE
        --sequence-parallel
    )
    
    # Build data arguments
    DATA_ARGS=()
    if [[ "$USE_MOCK_DATA" == true ]]; then
        DATA_ARGS+=(
            --mock-data
            --tokenizer-type NullTokenizer
            --vocab-size $VOCAB_SIZE
            --data-cache-path $DATA_CACHE_PATH
            --tiktoken-pattern v2
            --split '99,1,0'
            --no-create-attention-mask-in-dataloader
            --no-mmap-bin-files
            --num-workers 1
        )
    else
        DATA_ARGS+=(
            --data-path "$DATA_PATH"
            --tokenizer-type HuggingFaceTokenizer
            --tokenizer-model "$TOKENIZER_PATH"
            --data-cache-path $DATA_CACHE_PATH
            --split '99,1,0'
            --no-create-attention-mask-in-dataloader
            --num-workers 1
            --vocab-size $VOCAB_SIZE
        )
    fi
    
    # Build evaluation and logging arguments
    EVAL_AND_LOGGING_ARGS=(
        --log-interval 1
        --eval-iters 32
        --eval-interval 100
        --save-interval 1000
        --log-throughput
        --ckpt-format torch_dist
        --distributed-timeout-minutes 60
        --save "$CHECKPOINT_PATH"
        --load "$CHECKPOINT_PATH"
        --tensorboard-dir "$TENSORBOARD_PATH"
    )
    
    # Add profiling for standard training
    if [[ "$TRAINING_CONFIG" == "standard" ]]; then
        EVAL_AND_LOGGING_ARGS+=(
            --profile
            --profile-step-start 4
            --profile-step-end 6
        )
    fi
}

# Function to display training configuration
display_config() {
    log_info "Training Configuration:"
    log_info "  Model: $MODEL_NAME"
    log_info "  Experiment: $EXPERIMENT_NAME"
    log_info "  Training Config: $TRAINING_CONFIG"
    log_info "  Data Type: $DTYPE"
    log_info "  Distributed Config: $DISTRIBUTED_CONFIG"
    log_info "  Checkpoint Path: $CHECKPOINT_PATH"
    log_info "  Tensorboard Path: $TENSORBOARD_PATH"
    log_info "  Use Mock Data: $USE_MOCK_DATA"
    
    if [[ "$USE_MOCK_DATA" == false ]]; then
        log_info "  Data Path: $DATA_PATH"
        log_info "  Tokenizer Path: $TOKENIZER_PATH"
    fi
    
    log_info "  Model Parameters:"
    log_info "    TP_SIZE=$TP_SIZE, CP_SIZE=$CP_SIZE, PP_SIZE=$PP_SIZE"
    log_info "    NUM_LAYERS=$NUM_LAYERS, HIDDEN_SIZE=$HIDDEN_SIZE"
    log_info "    GLOBAL_BATCH_SIZE=$GLOBAL_BATCH_SIZE, SEQ_LENGTH=$SEQ_LENGTH"
}

# Function to run training
run_training() {
    local log_file="${TENSORBOARD_PATH}/training_${EXPERIMENT_NAME}_$(date +'%y-%m-%d_%H-%M-%S').log"
    
    log_info "Starting training..."
    log_info "Log file: $log_file"
    
    # Build the complete command
    local cmd=(
        torchrun "${DISTRIBUTED_ARGS[@]}"
        "$PRETRAIN_SCRIPT_PATH"
        "${MODEL_ARGS[@]}"
        "${TRAINING_ARGS[@]}"
        "${DTYPE_ARGS[@]}"
        "${MODEL_PARALLEL_ARGS[@]}"
        "${DATA_ARGS[@]}"
        "${EVAL_AND_LOGGING_ARGS[@]}"
    )
    
    if [[ "$DRY_RUN" == true ]]; then
        log_info "Dry run - would execute:"
        printf '%s \\\n' "${cmd[@]}"
        return 0
    fi
    
    # Execute training with logging
    "${cmd[@]}" 2>&1 | tee "$log_file"
    local exit_code=${PIPESTATUS[0]}
    
    if [[ $exit_code -eq 0 ]]; then
        log_success "Training completed successfully"
    else
        log_error "Training failed with exit code $exit_code"
    fi
    
    return $exit_code
}

# Main function
main() {
    log_info "Starting Megatron-LM training script"
    
    # Parse arguments
    parse_arguments "$@"
    
    # Validate arguments
    validate_arguments || exit 1
    
    # Validate configurations
    validate_model_config "$MODEL_NAME" || exit 1
    validate_training_config "$TRAINING_CONFIG" "$DTYPE" "$DISTRIBUTED_CONFIG" || exit 1
    
    # Setup paths
    setup_paths || exit 1
    
    # Build training arguments
    build_training_args || exit 1
    
    # Display configuration
    display_config
    
    # Run training
    run_training
}

# Run main function with all arguments
main "$@"
