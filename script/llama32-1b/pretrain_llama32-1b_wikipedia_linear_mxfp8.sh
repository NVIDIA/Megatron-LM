#!/bin/bash

# Training script for llama32-1b on wikipedia with linear_mxfp8 quantization
# Linear layer quantization with mxfp8

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR/.."  # Go to Megatron-LM root

# Configuration
MODEL_NAME="llama32-1b"
DATASET_NAME="wikipedia"
QUANT_TYPE="linear_mxfp8"
EXPERIMENT_NAME="wikipedia_linear_mxfp8"
CHECKPOINT_PATH="checkpoints/${MODEL_NAME}_${EXPERIMENT_NAME}"
TENSORBOARD_PATH="tensorboard_logs/${MODEL_NAME}_${EXPERIMENT_NAME}"
DATA_PATH="dataset/wikipedia_processed/wikipedia_processed_text_document"
TOKENIZER_PATH="model/llama3.2-1b"

# Model parameters
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

# Training parameters
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=128
SEQ_LENGTH=8192
TRAIN_SAMPLES=47340000
LR=0.00015
MIN_LR=0.00001
EXIT_DURATION_MINS=235000000

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Training script for ${MODEL_NAME} on ${DATASET_NAME} with ${QUANT_TYPE} quantization.
Linear layer quantization with mxfp8

Options:
    --dry-run                       Show command without executing
    --training-config CONFIG        Training configuration (standard, fast) [default: standard]
    --checkpoint-path PATH          Override checkpoint path
    --tensorboard-path PATH         Override tensorboard path
    --data-path PATH                Override data path
    --tokenizer-path PATH           Override tokenizer path
    --help                          Show this help message

Examples:
    # Run with default settings
    $0

    # Run with dry run
    $0 --dry-run

    # Run with fast config
    $0 --training-config fast

EOF
}

# Function to parse arguments
parse_arguments() {
    local training_config="standard"
    local dry_run=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --dry-run)
                dry_run=true
                shift
                ;;
            --training-config)
                training_config="$2"
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
            --help)
                show_usage
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    # Override training parameters for fast config
    if [[ "$training_config" == "fast" ]]; then
        GLOBAL_BATCH_SIZE=32
        SEQ_LENGTH=2048
        TRAIN_SAMPLES=1000
        LR=0.0001
        EXIT_DURATION_MINS=60
    fi
    
    # Build training command
    local cmd=(
        "torchrun"
        "--nproc_per_node" "8"
        "--nnodes" "1"
        "--node_rank" "0"
        "--master_addr" "localhost"
        "--master_port" "6000"
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
    if [[ "fp8" == "fp8" ]]; then
        cmd+=(
            "--fp8-format" "mxfp8"
            "--fp8-amax-history-len" "1024"
            "--fp8-amax-compute-algo" "max"
        )
        
        if [[ "mxfp8" != "None" ]]; then
            cmd+=("--linear-quantization" "mxfp8")
        fi
        if [[ "None" != "None" ]]; then
            cmd+=("--attention-quantization" "None")
        fi
    fi
    
    if [[ "$dry_run" == true ]]; then
        echo "Training command:"
        echo "${cmd[*]}"
    else
        echo "Starting training: ${MODEL_NAME} on ${DATASET_NAME} with ${QUANT_TYPE}"
        echo "Linear layer quantization with mxfp8"
        exec "${cmd[@]}"
    fi
}

# Main function
main() {
    echo "Training script for ${MODEL_NAME} on ${DATASET_NAME} with ${QUANT_TYPE} quantization"
    echo "Linear layer quantization with mxfp8"
    parse_arguments "$@"
}

# Run main function with all arguments
main "$@"
