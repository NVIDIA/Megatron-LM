#!/bin/bash

# Improved data processing script for Megatron-LM
# Provides robust data preprocessing with better error handling and validation

# Source configuration files
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config/common.sh"

# Default parameters
INPUT_PATH=""
OUTPUT_PREFIX=""
TOKENIZER_PATH=""
TOKENIZER_TYPE="HuggingFaceTokenizer"
WORKERS=32
PARTITIONS=8
APPEND_EOD=true
DRY_RUN=false

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Required Options:
    --input PATH                 Input data path (supports glob patterns)
    --output-prefix PREFIX       Output prefix for processed data
    --tokenizer-path PATH        Path to tokenizer model

Optional Options:
    --tokenizer-type TYPE        Tokenizer type (HuggingFaceTokenizer, NullTokenizer) [default: HuggingFaceTokenizer]
    --workers N                  Number of worker processes [default: 32]
    --partitions N               Number of partitions [default: 8]
    --append-eod                 Append end-of-document token [default: true]
    --no-append-eod              Don't append end-of-document token
    --dry-run                    Show what would be executed without running
    --help                       Show this help message

Examples:
    # Process Wikipedia data
    $0 --input './dataset/dolma/**/*.json.gz' \\
       --output-prefix ./dataset/dolma_processed \\
       --tokenizer-path ./model/llama3/ \\
       --workers 32 --partitions 8

    # Process with custom settings
    $0 --input './dataset/wikipedia/*.json' \\
       --output-prefix ./dataset/wikipedia_processed \\
       --tokenizer-path ./model/llama3.2-1b/ \\
       --workers 16 --partitions 4

EOF
}

# Function to parse command line arguments
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --input)
                INPUT_PATH="$2"
                shift 2
                ;;
            --output-prefix)
                OUTPUT_PREFIX="$2"
                shift 2
                ;;
            --tokenizer-path)
                TOKENIZER_PATH="$2"
                shift 2
                ;;
            --tokenizer-type)
                TOKENIZER_TYPE="$2"
                shift 2
                ;;
            --workers)
                WORKERS="$2"
                shift 2
                ;;
            --partitions)
                PARTITIONS="$2"
                shift 2
                ;;
            --append-eod)
                APPEND_EOD=true
                shift
                ;;
            --no-append-eod)
                APPEND_EOD=false
                shift
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
    if [[ -z "$INPUT_PATH" ]]; then
        log_error "Input path is required (--input)"
        errors=$((errors + 1))
    fi
    
    if [[ -z "$OUTPUT_PREFIX" ]]; then
        log_error "Output prefix is required (--output-prefix)"
        errors=$((errors + 1))
    fi
    
    if [[ -z "$TOKENIZER_PATH" ]]; then
        log_error "Tokenizer path is required (--tokenizer-path)"
        errors=$((errors + 1))
    fi
    
    # Validate tokenizer path
    if [[ ! -d "$TOKENIZER_PATH" ]]; then
        log_error "Tokenizer path does not exist: $TOKENIZER_PATH"
        errors=$((errors + 1))
    fi
    
    # Validate numeric parameters
    if ! [[ "$WORKERS" =~ ^[0-9]+$ ]] || [[ "$WORKERS" -lt 1 ]]; then
        log_error "Workers must be a positive integer: $WORKERS"
        errors=$((errors + 1))
    fi
    
    if ! [[ "$PARTITIONS" =~ ^[0-9]+$ ]] || [[ "$PARTITIONS" -lt 1 ]]; then
        log_error "Partitions must be a positive integer: $PARTITIONS"
        errors=$((errors + 1))
    fi
    
    # Validate tokenizer type
    case "$TOKENIZER_TYPE" in
        "HuggingFaceTokenizer"|"NullTokenizer")
            ;;
        *)
            log_error "Invalid tokenizer type: $TOKENIZER_TYPE"
            log_error "Valid types: HuggingFaceTokenizer, NullTokenizer"
            errors=$((errors + 1))
            ;;
    esac
    
    if [[ $errors -gt 0 ]]; then
        log_error "Validation failed with $errors error(s)"
        return 1
    fi
    
    return 0
}

# Function to validate input data
validate_input_data() {
    log_info "Validating input data..."
    
    # Check if input path contains glob patterns
    if [[ "$INPUT_PATH" == *"*"* ]]; then
        # Expand glob pattern and check if any files match
        local files_found=false
        for file in $INPUT_PATH; do
            if [[ -f "$file" ]]; then
                files_found=true
                break
            fi
        done
        
        if [[ "$files_found" == false ]]; then
            log_error "No files found matching pattern: $INPUT_PATH"
            return 1
        fi
        
        # Count matching files
        local file_count=$(ls $INPUT_PATH 2>/dev/null | wc -l)
        log_info "Found $file_count file(s) matching pattern: $INPUT_PATH"
    else
        # Single file or directory
        if [[ ! -e "$INPUT_PATH" ]]; then
            log_error "Input path does not exist: $INPUT_PATH"
            return 1
        fi
        
        if [[ -f "$INPUT_PATH" ]]; then
            log_info "Processing single file: $INPUT_PATH"
        elif [[ -d "$INPUT_PATH" ]]; then
            log_info "Processing directory: $INPUT_PATH"
        fi
    fi
    
    return 0
}

# Function to setup output directory
setup_output_directory() {
    local output_dir=$(dirname "$OUTPUT_PREFIX")
    
    if [[ ! -d "$output_dir" ]]; then
        log_info "Creating output directory: $output_dir"
        mkdir -p "$output_dir"
    fi
    
    # Check if output files already exist
    local output_files=("${OUTPUT_PREFIX}_text_document.bin" "${OUTPUT_PREFIX}_text_document.idx")
    local existing_files=()
    
    for file in "${output_files[@]}"; do
        if [[ -f "$file" ]]; then
            existing_files+=("$file")
        fi
    done
    
    if [[ ${#existing_files[@]} -gt 0 ]]; then
        log_warn "Output files already exist:"
        for file in "${existing_files[@]}"; do
            log_warn "  $file"
        done
        log_warn "These files will be overwritten"
    fi
}

# Function to estimate processing time
estimate_processing_time() {
    log_info "Estimating processing time..."
    
    # Get total size of input data
    local total_size=0
    if [[ "$INPUT_PATH" == *"*"* ]]; then
        # Calculate size of all matching files
        for file in $INPUT_PATH; do
            if [[ -f "$file" ]]; then
                local file_size=$(stat -c%s "$file" 2>/dev/null || echo "0")
                total_size=$((total_size + file_size))
            fi
        done
    else
        if [[ -f "$INPUT_PATH" ]]; then
            total_size=$(stat -c%s "$INPUT_PATH" 2>/dev/null || echo "0")
        elif [[ -d "$INPUT_PATH" ]]; then
            total_size=$(du -sb "$INPUT_PATH" 2>/dev/null | cut -f1)
        fi
    done
    
    # Convert bytes to human readable format
    local size_mb=$((total_size / 1024 / 1024))
    local size_gb=$((size_mb / 1024))
    
    if [[ $size_gb -gt 0 ]]; then
        log_info "Total input data size: ${size_gb}GB (${size_mb}MB)"
    else
        log_info "Total input data size: ${size_mb}MB"
    fi
    
    # Rough estimation: 1GB per minute with 32 workers
    local estimated_minutes=$((size_gb * 1))
    if [[ $estimated_minutes -gt 0 ]]; then
        log_info "Estimated processing time: ~${estimated_minutes} minutes"
    else
        log_info "Estimated processing time: < 1 minute"
    fi
}

# Function to build processing command
build_processing_command() {
    local cmd=(
        python "${TOOLS_DIR}/preprocess_data.py"
        --input "$INPUT_PATH"
        --workers "$WORKERS"
        --partitions "$PARTITIONS"
        --output-prefix "$OUTPUT_PREFIX"
        --tokenizer-type "$TOKENIZER_TYPE"
        --tokenizer-model "$TOKENIZER_PATH"
    )
    
    if [[ "$APPEND_EOD" == true ]]; then
        cmd+=(--append-eod)
    fi
    
    echo "${cmd[@]}"
}

# Function to run data processing
run_processing() {
    local start_time=$(date +%s)
    local log_file="${OUTPUT_PREFIX}_processing_$(date +'%y-%m-%d_%H-%M-%S').log"
    
    log_info "Starting data processing..."
    log_info "Log file: $log_file"
    
    # Build command
    local cmd=($(build_processing_command))
    
    if [[ "$DRY_RUN" == true ]]; then
        log_info "Dry run - would execute:"
        printf '%s \\\n' "${cmd[@]}"
        return 0
    fi
    
    # Execute processing with logging
    log_info "Executing: ${cmd[*]}"
    "${cmd[@]}" 2>&1 | tee "$log_file"
    local exit_code=${PIPESTATUS[0]}
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local minutes=$((duration / 60))
    local seconds=$((duration % 60))
    
    if [[ $exit_code -eq 0 ]]; then
        log_success "Data processing completed successfully in ${minutes}m ${seconds}s"
        
        # Verify output files
        local output_files=("${OUTPUT_PREFIX}_text_document.bin" "${OUTPUT_PREFIX}_text_document.idx")
        for file in "${output_files[@]}"; do
            if [[ -f "$file" ]]; then
                local file_size=$(stat -c%s "$file" 2>/dev/null || echo "0")
                local size_mb=$((file_size / 1024 / 1024))
                log_info "Created: $file (${size_mb}MB)"
            else
                log_warn "Expected output file not found: $file"
            fi
        done
    else
        log_error "Data processing failed with exit code $exit_code"
    fi
    
    return $exit_code
}

# Function to display configuration
display_config() {
    log_info "Data Processing Configuration:"
    log_info "  Input Path: $INPUT_PATH"
    log_info "  Output Prefix: $OUTPUT_PREFIX"
    log_info "  Tokenizer Path: $TOKENIZER_PATH"
    log_info "  Tokenizer Type: $TOKENIZER_TYPE"
    log_info "  Workers: $WORKERS"
    log_info "  Partitions: $PARTITIONS"
    log_info "  Append EOD: $APPEND_EOD"
    log_info "  Dry Run: $DRY_RUN"
}

# Main function
main() {
    log_info "Starting Megatron-LM data processing script"
    
    # Parse arguments
    parse_arguments "$@"
    
    # Validate arguments
    validate_arguments || exit 1
    
    # Validate input data
    validate_input_data || exit 1
    
    # Setup output directory
    setup_output_directory || exit 1
    
    # Estimate processing time
    estimate_processing_time
    
    # Display configuration
    display_config
    
    # Run processing
    run_processing
}

# Run main function with all arguments
main "$@"
