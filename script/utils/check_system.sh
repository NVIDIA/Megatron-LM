#!/bin/bash

# System check utility script
# Validates system requirements and environment for Megatron-LM training

# Source configuration files
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${SCRIPT_DIR}/config/common.sh"

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Options:
    --check-gpu                 Check GPU availability and specifications
    --check-memory              Check system memory
    --check-disk                Check disk space
    --check-dependencies        Check required dependencies
    --check-paths               Check required paths and files
    --all                       Run all checks (default)
    --help                      Show this help message

Examples:
    # Run all system checks
    $0

    # Check only GPU
    $0 --check-gpu

    # Check GPU and memory
    $0 --check-gpu --check-memory

EOF
}

# Function to check GPU
check_gpu() {
    log_info "Checking GPU availability..."
    
    if ! command_exists nvidia-smi; then
        log_error "nvidia-smi not found. NVIDIA drivers may not be installed."
        return 1
    fi
    
    local gpu_count=$(nvidia-smi --list-gpus | wc -l)
    if [[ $gpu_count -eq 0 ]]; then
        log_error "No GPUs detected"
        return 1
    fi
    
    log_success "Detected $gpu_count GPU(s)"
    
    # Get GPU information
    log_info "GPU Information:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free,driver_version --format=csv,noheader,nounits | while IFS=',' read -r index name total_mem free_mem driver; do
        local used_mem=$((total_mem - free_mem))
        local usage_percent=$((used_mem * 100 / total_mem))
        log_info "  GPU $index: $name"
        log_info "    Memory: ${used_mem}MB / ${total_mem}MB (${usage_percent}% used)"
        log_info "    Driver: $driver"
    done
    
    # Check CUDA version
    local cuda_version=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d. -f1,2)
    if [[ -n "$cuda_version" ]]; then
        log_info "CUDA Version: $cuda_version"
    fi
    
    return 0
}

# Function to check memory
check_memory() {
    log_info "Checking system memory..."
    
    if [[ -f /proc/meminfo ]]; then
        local total_mem=$(grep MemTotal /proc/meminfo | awk '{print $2}')
        local available_mem=$(grep MemAvailable /proc/meminfo | awk '{print $2}')
        local used_mem=$((total_mem - available_mem))
        
        local total_gb=$((total_mem / 1024 / 1024))
        local available_gb=$((available_mem / 1024 / 1024))
        local used_gb=$((used_mem / 1024 / 1024))
        
        log_info "System Memory:"
        log_info "  Total: ${total_gb}GB"
        log_info "  Available: ${available_gb}GB"
        log_info "  Used: ${used_gb}GB"
        
        # Check if we have enough memory for training
        if [[ $available_gb -lt 32 ]]; then
            log_warn "Available memory is less than 32GB. Large model training may fail."
        fi
    else
        log_warn "Cannot read memory information from /proc/meminfo"
    fi
    
    return 0
}

# Function to check disk space
check_disk() {
    log_info "Checking disk space..."
    
    # Check current directory
    local current_dir=$(pwd)
    local available_space=$(df "$current_dir" | tail -1 | awk '{print $4}')
    local total_space=$(df "$current_dir" | tail -1 | awk '{print $2}')
    local used_space=$((total_space - available_space))
    
    local available_gb=$((available_space / 1024 / 1024))
    local total_gb=$((total_space / 1024 / 1024))
    local used_gb=$((used_space / 1024 / 1024))
    local usage_percent=$((used_space * 100 / total_space))
    
    log_info "Disk Space ($current_dir):"
    log_info "  Total: ${total_gb}GB"
    log_info "  Available: ${available_gb}GB"
    log_info "  Used: ${used_gb}GB (${usage_percent}%)"
    
    # Check if we have enough space for training
    if [[ $available_gb -lt 100 ]]; then
        log_warn "Available disk space is less than 100GB. Training may fail due to insufficient space."
    fi
    
    return 0
}

# Function to check dependencies
check_dependencies() {
    log_info "Checking dependencies..."
    
    local missing_deps=()
    
    # Check Python
    if ! command_exists python; then
        missing_deps+=("python")
    else
        local python_version=$(python --version 2>&1 | awk '{print $2}')
        log_info "Python: $python_version"
    fi
    
    # Check PyTorch
    if python -c "import torch" 2>/dev/null; then
        local torch_version=$(python -c "import torch; print(torch.__version__)")
        log_info "PyTorch: $torch_version"
        
        # Check CUDA support
        if python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
            log_success "PyTorch CUDA support: Available"
        else
            log_warn "PyTorch CUDA support: Not available"
        fi
    else
        missing_deps+=("pytorch")
    fi
    
    # Check Megatron-LM dependencies
    local megatron_deps=("transformers" "numpy" "packaging")
    for dep in "${megatron_deps[@]}"; do
        if python -c "import $dep" 2>/dev/null; then
            log_info "$dep: Available"
        else
            missing_deps+=("$dep")
        fi
    done
    
    # Check system tools
    local system_tools=("git" "wget" "curl")
    for tool in "${system_tools[@]}"; do
        if command_exists "$tool"; then
            log_info "$tool: Available"
        else
            missing_deps+=("$tool")
        fi
    done
    
    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        log_error "Missing dependencies: ${missing_deps[*]}"
        return 1
    fi
    
    log_success "All dependencies are available"
    return 0
}

# Function to check paths
check_paths() {
    log_info "Checking required paths and files..."
    
    local missing_paths=()
    
    # Check Megatron-LM structure
    local required_files=(
        "pretrain_gpt.py"
        "tools/preprocess_data.py"
        "examples/llama/train_llama3_8b_h100_fp8.sh"
        "examples/llama/train_llama32_1b_h100_fp8.sh"
    )
    
    for file in "${required_files[@]}"; do
        if [[ -f "$MEGATRON_ROOT/$file" ]]; then
            log_info "$file: Found"
        else
            missing_paths+=("$file")
        fi
    done
    
    # Check common directories
    local common_dirs=("model" "dataset" "checkpoints" "tensorboard_logs")
    for dir in "${common_dirs[@]}"; do
        if [[ -d "$MEGATRON_ROOT/$dir" ]]; then
            log_info "$dir/: Found"
        else
            log_warn "$dir/: Not found (will be created when needed)"
        fi
    done
    
    if [[ ${#missing_paths[@]} -gt 0 ]]; then
        log_error "Missing required files: ${missing_paths[*]}"
        return 1
    fi
    
    log_success "All required paths and files are available"
    return 0
}

# Function to run all checks
run_all_checks() {
    log_info "Running comprehensive system check..."
    
    local errors=0
    
    check_gpu || errors=$((errors + 1))
    echo
    check_memory || errors=$((errors + 1))
    echo
    check_disk || errors=$((errors + 1))
    echo
    check_dependencies || errors=$((errors + 1))
    echo
    check_paths || errors=$((errors + 1))
    
    echo
    if [[ $errors -eq 0 ]]; then
        log_success "All system checks passed! System is ready for training."
    else
        log_error "System check completed with $errors error(s). Please fix the issues before training."
    fi
    
    return $errors
}

# Main function
main() {
    local check_gpu=false
    local check_memory=false
    local check_disk=false
    local check_dependencies=false
    local check_paths=false
    local run_all=true
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --check-gpu)
                check_gpu=true
                run_all=false
                shift
                ;;
            --check-memory)
                check_memory=true
                run_all=false
                shift
                ;;
            --check-disk)
                check_disk=true
                run_all=false
                shift
                ;;
            --check-dependencies)
                check_dependencies=true
                run_all=false
                shift
                ;;
            --check-paths)
                check_paths=true
                run_all=false
                shift
                ;;
            --all)
                run_all=true
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
    
    # Run specified checks
    if [[ "$run_all" == true ]]; then
        run_all_checks
    else
        local errors=0
        [[ "$check_gpu" == true ]] && { check_gpu || errors=$((errors + 1)); }
        [[ "$check_memory" == true ]] && { check_memory || errors=$((errors + 1)); }
        [[ "$check_disk" == true ]] && { check_disk || errors=$((errors + 1)); }
        [[ "$check_dependencies" == true ]] && { check_dependencies || errors=$((errors + 1)); }
        [[ "$check_paths" == true ]] && { check_paths || errors=$((errors + 1)); }
        return $errors
    fi
}

# Run main function with all arguments
main "$@"
