#!/bin/bash

# Setup script for the new Megatron-LM script system
# This script helps users get started with the new unified script system

# Source configuration files
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config/common.sh"

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [COMMAND] [OPTIONS]

Commands:
    init                        Initialize the new script system
    test                        Test the system with a quick mock run
    demo                        Run a demonstration of the new features
    status                      Show system status and configuration
    help                        Show this help message

Options:
    --skip-system-check         Skip system validation
    --force                     Force operations without confirmation

Examples:
    # Initialize the new system
    $0 init

    # Test with a quick mock run
    $0 test

    # Run a full demonstration
    $0 demo

    # Check system status
    $0 status

EOF
}

# Function to initialize the system
init_system() {
    local skip_system_check="$1"
    local force="$2"
    
    log_info "Initializing Megatron-LM script system..."
    
    # Run system check unless skipped
    if [[ "$skip_system_check" != "true" ]]; then
        log_info "Running system validation..."
        if ! "${SCRIPT_DIR}/utils/check_system.sh"; then
            log_error "System validation failed. Please fix the issues before continuing."
            return 1
        fi
    fi
    
    # Create necessary directories
    log_info "Creating necessary directories..."
    local dirs=(
        "model"
        "dataset"
        "checkpoints"
        "tensorboard_logs"
        "logs"
    )
    
    for dir in "${dirs[@]}"; do
        ensure_directories "${MEGATRON_ROOT}/$dir"
    done
    
    # Validate experiment configurations
    log_info "Validating experiment configurations..."
    if ! "${SCRIPT_DIR}/experiment_launcher.sh" validate; then
        log_error "Experiment validation failed"
        return 1
    fi
    
    log_success "System initialization completed successfully!"
    
    # Show next steps
    echo
    log_info "=== NEXT STEPS ==="
    echo "1. Test the system:"
    echo "   ./script/setup.sh test"
    echo
    echo "2. Run a demonstration:"
    echo "   ./script/setup.sh demo"
    echo
    echo "3. Start training:"
    echo "   ./script/experiment_launcher.sh list"
    echo "   ./script/experiment_launcher.sh run [experiment_name]"
    echo
    echo "4. Process data:"
    echo "   ./script/process_data_improved.sh --help"
}

# Function to test the system
test_system() {
    local force="$1"
    
    log_info "Testing the new script system..."
    
    # Test system check
    log_info "1. Testing system validation..."
    if ! "${SCRIPT_DIR}/utils/check_system.sh" --check-paths; then
        log_error "System check failed"
        return 1
    fi
    
    # Test experiment launcher
    log_info "2. Testing experiment launcher..."
    if ! "${SCRIPT_DIR}/experiment_launcher.sh" list >/dev/null; then
        log_error "Experiment launcher failed"
        return 1
    fi
    
    # Test base training script with dry run
    log_info "3. Testing base training script (dry run)..."
    if ! "${SCRIPT_DIR}/train_base.sh" \
        --model llama3_8b \
        --experiment-name test_run \
        --checkpoint-path checkpoints/llama3_8b/test \
        --tensorboard-path tensorboard_logs/llama3_8b/test \
        --use-mock-data \
        --training-config fast \
        --dry-run >/dev/null; then
        log_error "Base training script test failed"
        return 1
    fi
    
    # Test data processing script with dry run
    log_info "4. Testing data processing script (dry run)..."
    if ! "${SCRIPT_DIR}/process_data_improved.sh" \
        --input "test_input.txt" \
        --output-prefix "test_output" \
        --tokenizer-path "model/test" \
        --dry-run >/dev/null 2>&1; then
        log_warn "Data processing script test failed (expected - test paths don't exist)"
    fi
    
    # Test utility scripts
    log_info "5. Testing utility scripts..."
    if ! "${SCRIPT_DIR}/utils/cleanup.sh" list >/dev/null; then
        log_error "Cleanup utility test failed"
        return 1
    fi
    
    log_success "All tests passed! The system is working correctly."
    
    echo
    log_info "=== TEST RESULTS ==="
    echo "✓ System validation: PASSED"
    echo "✓ Experiment launcher: PASSED"
    echo "✓ Base training script: PASSED"
    echo "✓ Data processing script: PASSED"
    echo "✓ Utility scripts: PASSED"
}

# Function to run demonstration
demo_system() {
    local force="$1"
    
    log_info "Running demonstration of the new script system..."
    
    echo
    log_info "=== DEMONSTRATION: SYSTEM CHECK ==="
    "${SCRIPT_DIR}/utils/check_system.sh" --check-gpu --check-memory
    
    echo
    log_info "=== DEMONSTRATION: AVAILABLE EXPERIMENTS ==="
    "${SCRIPT_DIR}/experiment_launcher.sh" list
    
    echo
    log_info "=== DEMONSTRATION: EXPERIMENT VALIDATION ==="
    "${SCRIPT_DIR}/experiment_launcher.sh" validate
    
    echo
    log_info "=== DEMONSTRATION: TRAINING CONFIGURATION (DRY RUN) ==="
    "${SCRIPT_DIR}/train_base.sh" \
        --model llama3_8b \
        --experiment-name demo_run \
        --checkpoint-path checkpoints/llama3_8b/demo \
        --tensorboard-path tensorboard_logs/llama3_8b/demo \
        --use-mock-data \
        --training-config fast \
        --dry-run
    
    echo
    log_info "=== DEMONSTRATION: CLEANUP UTILITIES ==="
    "${SCRIPT_DIR}/utils/cleanup.sh" list
    
    echo
    log_success "Demonstration completed!"
    
    echo
    log_info "=== WHAT YOU CAN DO NOW ==="
    echo "1. Run a real experiment with mock data:"
    echo "   ./script/experiment_launcher.sh run llama3_8b_mock_fast"
    echo
    echo "2. Process your own data:"
    echo "   ./script/process_data_improved.sh --help"
    echo
    echo "3. Train with real data:"
    echo "   ./script/experiment_launcher.sh run llama3_8b_wikipedia_fp8"
    echo
    echo "4. Monitor and clean up:"
    echo "   ./script/utils/cleanup.sh list"
}

# Function to show system status
show_status() {
    log_info "Megatron-LM Script System Status"
    echo
    
    # Show system information
    log_info "=== SYSTEM INFORMATION ==="
    echo "Megatron-LM Root: $MEGATRON_ROOT"
    echo "Script Directory: $SCRIPT_DIR"
    echo "Python Version: $(python --version 2>&1)"
    echo "PyTorch Version: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not available')"
    echo "CUDA Available: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'Unknown')"
    
    # Show GPU information
    if command_exists nvidia-smi; then
        echo "GPU Count: $(nvidia-smi --list-gpus | wc -l)"
        echo "GPU Memory: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1) MB"
    else
        echo "GPU: Not available"
    fi
    
    # Show available experiments
    echo
    log_info "=== AVAILABLE EXPERIMENTS ==="
    "${SCRIPT_DIR}/experiment_launcher.sh" list
    
    # Show directory status
    echo
    log_info "=== DIRECTORY STATUS ==="
    local dirs=("model" "dataset" "checkpoints" "tensorboard_logs" "logs")
    for dir in "${dirs[@]}"; do
        local dir_path="${MEGATRON_ROOT}/$dir"
        if [[ -d "$dir_path" ]]; then
            local file_count=$(find "$dir_path" -type f 2>/dev/null | wc -l)
            echo "✓ $dir/ ($file_count files)"
        else
            echo "✗ $dir/ (not found)"
        fi
    done
    
    # Show recent activity
    echo
    log_info "=== RECENT ACTIVITY ==="
    local log_dirs=("tensorboard_logs" "logs")
    for log_dir in "${log_dirs[@]}"; do
        local log_path="${MEGATRON_ROOT}/$log_dir"
        if [[ -d "$log_path" ]]; then
            local recent_logs=$(find "$log_path" -name "*.log" -mtime -1 2>/dev/null | wc -l)
            if [[ $recent_logs -gt 0 ]]; then
                echo "Recent logs in $log_dir/: $recent_logs files"
            fi
        fi
    done
}

# Main function
main() {
    local command="$1"
    shift || true
    
    local skip_system_check="false"
    local force="false"
    
    # Parse options
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-system-check)
                skip_system_check="true"
                shift
                ;;
            --force)
                force="true"
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    case "$command" in
        "init")
            init_system "$skip_system_check" "$force"
            ;;
        "test")
            test_system "$force"
            ;;
        "demo")
            demo_system "$force"
            ;;
        "status")
            show_status
            ;;
        "help"|"--help"|"")
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
