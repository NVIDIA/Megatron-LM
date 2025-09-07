#!/bin/bash

# Experiment launcher script
# Provides easy-to-use interface for common training experiments

# Source configuration files
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config/common.sh"

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [COMMAND] [OPTIONS]

Commands:
    list                    List available experiments
    run EXPERIMENT_NAME     Run a predefined experiment
    create EXPERIMENT_NAME  Create a new experiment configuration
    validate                Validate all experiment configurations

Predefined Experiments:
    llama3_8b_wikipedia_fp8     Train LLaMA 3 8B on Wikipedia with FP8
    llama3_8b_wikipedia_bf16    Train LLaMA 3 8B on Wikipedia with BF16
    llama32_1b_wikipedia_fp8    Train LLaMA 3.2 1B on Wikipedia with FP8
    llama32_1b_wikipedia_bf16   Train LLaMA 3.2 1B on Wikipedia with BF16
    llama3_8b_mock_fast         Quick test with LLaMA 3 8B using mock data
    llama32_1b_mock_fast        Quick test with LLaMA 3.2 1B using mock data

Options:
    --dry-run               Show what would be executed without running
    --help                  Show this help message

Examples:
    # List available experiments
    $0 list

    # Run a predefined experiment
    $0 run llama3_8b_wikipedia_fp8

    # Run with dry run
    $0 run llama3_8b_mock_fast --dry-run

EOF
}

# Experiment configurations
declare -A EXPERIMENTS

# LLaMA 3 8B Wikipedia FP8
EXPERIMENTS["llama3_8b_wikipedia_fp8"]="
MODEL_NAME=llama3_8b
TRAINING_CONFIG=standard
DTYPE=fp8
DISTRIBUTED_CONFIG=single_node
EXPERIMENT_NAME=wikipedia_fp8
CHECKPOINT_PATH=checkpoints/llama3_8b/wikipedia_fp8
TENSORBOARD_PATH=tensorboard_logs/llama3_8b/wikipedia_fp8
DATA_PATH=dataset/wikipedia_processed/wikipedia_processed_text_document
TOKENIZER_PATH=model/llama3
USE_MOCK_DATA=false
"

# LLaMA 3 8B Wikipedia BF16
EXPERIMENTS["llama3_8b_wikipedia_bf16"]="
MODEL_NAME=llama3_8b
TRAINING_CONFIG=standard
DTYPE=bf16
DISTRIBUTED_CONFIG=single_node
EXPERIMENT_NAME=wikipedia_bf16
CHECKPOINT_PATH=checkpoints/llama3_8b/wikipedia_bf16
TENSORBOARD_PATH=tensorboard_logs/llama3_8b/wikipedia_bf16
DATA_PATH=dataset/wikipedia_processed/wikipedia_processed_text_document
TOKENIZER_PATH=model/llama3
USE_MOCK_DATA=false
"

# LLaMA 3.2 1B Wikipedia FP8
EXPERIMENTS["llama32_1b_wikipedia_fp8"]="
MODEL_NAME=llama32_1b
TRAINING_CONFIG=standard
DTYPE=fp8
DISTRIBUTED_CONFIG=single_node
EXPERIMENT_NAME=wikipedia_fp8
CHECKPOINT_PATH=checkpoints/llama32_1b/wikipedia_fp8
TENSORBOARD_PATH=tensorboard_logs/llama32_1b/wikipedia_fp8
DATA_PATH=dataset/wikipedia_processed/wikipedia_processed_text_document
TOKENIZER_PATH=model/llama3.2-1b
USE_MOCK_DATA=false
"

# LLaMA 3.2 1B Wikipedia BF16
EXPERIMENTS["llama32_1b_wikipedia_bf16"]="
MODEL_NAME=llama32_1b
TRAINING_CONFIG=standard
DTYPE=bf16
DISTRIBUTED_CONFIG=single_node
EXPERIMENT_NAME=wikipedia_bf16
CHECKPOINT_PATH=checkpoints/llama32_1b/wikipedia_bf16
TENSORBOARD_PATH=tensorboard_logs/llama32_1b/wikipedia_bf16
DATA_PATH=dataset/wikipedia_processed/wikipedia_processed_text_document
TOKENIZER_PATH=model/llama3.2-1b
USE_MOCK_DATA=false
"

# LLaMA 3 8B Mock Fast
EXPERIMENTS["llama3_8b_mock_fast"]="
MODEL_NAME=llama3_8b
TRAINING_CONFIG=fast
DTYPE=fp8
DISTRIBUTED_CONFIG=single_node
EXPERIMENT_NAME=mock_fast
CHECKPOINT_PATH=checkpoints/llama3_8b/mock_fast
TENSORBOARD_PATH=tensorboard_logs/llama3_8b/mock_fast
USE_MOCK_DATA=true
"

# LLaMA 3.2 1B Mock Fast
EXPERIMENTS["llama32_1b_mock_fast"]="
MODEL_NAME=llama32_1b
TRAINING_CONFIG=fast
DTYPE=fp8
DISTRIBUTED_CONFIG=single_node
EXPERIMENT_NAME=mock_fast
CHECKPOINT_PATH=checkpoints/llama32_1b/mock_fast
TENSORBOARD_PATH=tensorboard_logs/llama32_1b/mock_fast
USE_MOCK_DATA=true
"

# Function to list experiments
list_experiments() {
    log_info "Available experiments:"
    echo
    for exp_name in "${!EXPERIMENTS[@]}"; do
        echo "  $exp_name"
    done
    echo
    log_info "Use '$0 run EXPERIMENT_NAME' to run an experiment"
}

# Function to get experiment configuration
get_experiment_config() {
    local exp_name="$1"
    if [[ -z "${EXPERIMENTS[$exp_name]:-}" ]]; then
        log_error "Unknown experiment: $exp_name"
        log_error "Available experiments: ${!EXPERIMENTS[*]}"
        return 1
    fi
    echo "${EXPERIMENTS[$exp_name]}"
}

# Function to run experiment
run_experiment() {
    local exp_name="$1"
    local dry_run="$2"
    
    log_info "Running experiment: $exp_name"
    
    # Get experiment configuration
    local config=$(get_experiment_config "$exp_name")
    if [[ $? -ne 0 ]]; then
        return 1
    fi
    
    # Evaluate configuration
    eval "$config"
    
    # Build command arguments
    local cmd_args=(
        --model "$MODEL_NAME"
        --experiment-name "$EXPERIMENT_NAME"
        --checkpoint-path "$CHECKPOINT_PATH"
        --tensorboard-path "$TENSORBOARD_PATH"
        --training-config "$TRAINING_CONFIG"
        --dtype "$DTYPE"
        --distributed-config "$DISTRIBUTED_CONFIG"
    )
    
    # Add data-specific arguments
    if [[ "$USE_MOCK_DATA" == "true" ]]; then
        cmd_args+=(--use-mock-data)
    else
        cmd_args+=(--data-path "$DATA_PATH" --tokenizer-path "$TOKENIZER_PATH")
    fi
    
    # Add dry run flag if specified
    if [[ "$dry_run" == "true" ]]; then
        cmd_args+=(--dry-run)
    fi
    
    # Run the training script
    log_info "Executing training with arguments: ${cmd_args[*]}"
    "${SCRIPT_DIR}/train_base.sh" "${cmd_args[@]}"
}

# Function to create new experiment
create_experiment() {
    local exp_name="$1"
    
    if [[ -n "${EXPERIMENTS[$exp_name]:-}" ]]; then
        log_error "Experiment '$exp_name' already exists"
        return 1
    fi
    
    log_info "Creating new experiment: $exp_name"
    log_info "This would create a new experiment configuration template"
    log_info "For now, please manually add the configuration to this script"
    
    # TODO: Implement interactive experiment creation
    return 0
}

# Function to validate all experiments
validate_experiments() {
    log_info "Validating all experiment configurations..."
    
    local errors=0
    for exp_name in "${!EXPERIMENTS[@]}"; do
        log_info "Validating experiment: $exp_name"
        
        # Get and evaluate configuration
        local config=$(get_experiment_config "$exp_name")
        if [[ $? -ne 0 ]]; then
            errors=$((errors + 1))
            continue
        fi
        
        eval "$config"
        
        # Validate required parameters
        local required_params=("MODEL_NAME" "TRAINING_CONFIG" "DTYPE" "EXPERIMENT_NAME" "CHECKPOINT_PATH" "TENSORBOARD_PATH")
        for param in "${required_params[@]}"; do
            if [[ -z "${!param:-}" ]]; then
                log_error "Missing required parameter '$param' in experiment '$exp_name'"
                errors=$((errors + 1))
            fi
        done
        
        # Validate data configuration
        if [[ "${USE_MOCK_DATA:-false}" != "true" ]]; then
            if [[ -z "${DATA_PATH:-}" || -z "${TOKENIZER_PATH:-}" ]]; then
                log_error "Missing DATA_PATH or TOKENIZER_PATH in experiment '$exp_name'"
                errors=$((errors + 1))
            fi
        fi
    done
    
    if [[ $errors -eq 0 ]]; then
        log_success "All experiment configurations are valid"
    else
        log_error "Found $errors error(s) in experiment configurations"
    fi
    
    return $errors
}

# Main function
main() {
    local command="$1"
    shift || true
    
    case "$command" in
        "list")
            list_experiments
            ;;
        "run")
            if [[ -z "$1" ]]; then
                log_error "Experiment name is required"
                show_usage
                exit 1
            fi
            
            local exp_name="$1"
            local dry_run="false"
            
            # Check for dry-run flag
            if [[ "$2" == "--dry-run" ]]; then
                dry_run="true"
            fi
            
            run_experiment "$exp_name" "$dry_run"
            ;;
        "create")
            if [[ -z "$1" ]]; then
                log_error "Experiment name is required"
                show_usage
                exit 1
            fi
            
            create_experiment "$1"
            ;;
        "validate")
            validate_experiments
            ;;
        "--help"|"help"|"")
            show_usage
            ;;
        *)
            log_error "Unknown command: $command"
            show_usage
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"
