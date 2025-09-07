#!/bin/bash

# Cleanup utility script
# Provides utilities for cleaning up training artifacts and managing disk space

# Source configuration files
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${SCRIPT_DIR}/config/common.sh"

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [COMMAND] [OPTIONS]

Commands:
    checkpoints [OPTIONS]         Clean up checkpoint files
    logs [OPTIONS]                Clean up log files
    cache [OPTIONS]               Clean up cache files
    all [OPTIONS]                 Clean up all artifacts
    list                          List cleanup candidates

Checkpoint Options:
    --older-than DAYS            Remove checkpoints older than N days [default: 7]
    --keep-latest N               Keep only the latest N checkpoints [default: 3]
    --dry-run                     Show what would be deleted without deleting

Log Options:
    --older-than DAYS            Remove logs older than N days [default: 7]
    --larger-than MB             Remove logs larger than N MB [default: 100]
    --dry-run                     Show what would be deleted without deleting

Cache Options:
    --all                         Remove all cache files
    --older-than DAYS            Remove cache files older than N days [default: 3]
    --dry-run                     Show what would be deleted without deleting

Examples:
    # List all cleanup candidates
    $0 list

    # Clean up old checkpoints (keep latest 3, remove older than 7 days)
    $0 checkpoints --keep-latest 3 --older-than 7

    # Clean up large log files
    $0 logs --larger-than 500

    # Clean up all cache files
    $0 cache --all

    # Dry run to see what would be cleaned
    $0 all --dry-run

EOF
}

# Function to find files by age
find_files_by_age() {
    local path="$1"
    local days="$2"
    find "$path" -type f -mtime +$days 2>/dev/null || true
}

# Function to find files by size
find_files_by_size() {
    local path="$1"
    local size_mb="$2"
    local size_bytes=$((size_mb * 1024 * 1024))
    find "$path" -type f -size +${size_bytes}c 2>/dev/null || true
}

# Function to get file size in MB
get_file_size_mb() {
    local file="$1"
    local size_bytes=$(stat -c%s "$file" 2>/dev/null || echo "0")
    echo $((size_bytes / 1024 / 1024))
}

# Function to format file size
format_size() {
    local size_mb="$1"
    if [[ $size_mb -gt 1024 ]]; then
        local size_gb=$((size_mb / 1024))
        echo "${size_gb}GB"
    else
        echo "${size_mb}MB"
    fi
}

# Function to clean checkpoints
cleanup_checkpoints() {
    local older_than=7
    local keep_latest=3
    local dry_run=false
    
    # Parse options
    while [[ $# -gt 0 ]]; do
        case $1 in
            --older-than)
                older_than="$2"
                shift 2
                ;;
            --keep-latest)
                keep_latest="$2"
                shift 2
                ;;
            --dry-run)
                dry_run=true
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                return 1
                ;;
        esac
    done
    
    log_info "Cleaning up checkpoints..."
    log_info "  Keep latest: $keep_latest"
    log_info "  Remove older than: $older_than days"
    log_info "  Dry run: $dry_run"
    
    local checkpoint_dirs=(
        "$MEGATRON_ROOT/checkpoints"
    )
    
    local total_size=0
    local file_count=0
    
    for checkpoint_dir in "${checkpoint_dirs[@]}"; do
        if [[ ! -d "$checkpoint_dir" ]]; then
            continue
        fi
        
        log_info "Processing checkpoint directory: $checkpoint_dir"
        
        # Find all checkpoint directories
        local model_dirs=$(find "$checkpoint_dir" -maxdepth 1 -type d -name "*" | grep -v "^$checkpoint_dir$" | sort)
        
        for model_dir in $model_dirs; do
            log_info "  Processing model: $(basename "$model_dir")"
            
            # Find checkpoint files (typically .pt files)
            local checkpoint_files=$(find "$model_dir" -name "*.pt" -o -name "*.bin" -o -name "*.safetensors" | sort -r)
            
            if [[ -z "$checkpoint_files" ]]; then
                continue
            fi
            
            # Keep latest N checkpoints
            local files_to_keep=$(echo "$checkpoint_files" | head -n $keep_latest)
            local files_to_remove=$(echo "$checkpoint_files" | tail -n +$((keep_latest + 1)))
            
            # Also remove files older than specified days
            local old_files=$(find_files_by_age "$model_dir" $older_than)
            
            # Combine files to remove (avoid duplicates)
            local all_files_to_remove=$(echo -e "$files_to_remove\n$old_files" | sort -u)
            
            for file in $all_files_to_remove; do
                if [[ -f "$file" ]]; then
                    local file_size=$(get_file_size_mb "$file")
                    local age_days=$(( ($(date +%s) - $(stat -c%Y "$file")) / 86400 ))
                    
                    log_info "    $(basename "$file") - $(format_size $file_size) - ${age_days} days old"
                    
                    if [[ "$dry_run" == false ]]; then
                        rm -f "$file"
                        log_info "      Deleted"
                    else
                        log_info "      Would delete"
                    fi
                    
                    total_size=$((total_size + file_size))
                    file_count=$((file_count + 1))
                fi
            done
        done
    done
    
    if [[ $file_count -gt 0 ]]; then
        if [[ "$dry_run" == true ]]; then
            log_info "Would remove $file_count checkpoint file(s) totaling $(format_size $total_size)"
        else
            log_success "Removed $file_count checkpoint file(s) totaling $(format_size $total_size)"
        fi
    else
        log_info "No checkpoint files to clean up"
    fi
}

# Function to clean logs
cleanup_logs() {
    local older_than=7
    local larger_than=100
    local dry_run=false
    
    # Parse options
    while [[ $# -gt 0 ]]; do
        case $1 in
            --older-than)
                older_than="$2"
                shift 2
                ;;
            --larger-than)
                larger_than="$2"
                shift 2
                ;;
            --dry-run)
                dry_run=true
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                return 1
                ;;
        esac
    done
    
    log_info "Cleaning up logs..."
    log_info "  Remove older than: $older_than days"
    log_info "  Remove larger than: $larger_than MB"
    log_info "  Dry run: $dry_run"
    
    local log_dirs=(
        "$MEGATRON_ROOT/tensorboard_logs"
        "$MEGATRON_ROOT/logs"
    )
    
    local total_size=0
    local file_count=0
    
    for log_dir in "${log_dirs[@]}"; do
        if [[ ! -d "$log_dir" ]]; then
            continue
        fi
        
        log_info "Processing log directory: $log_dir"
        
        # Find old log files
        local old_files=$(find_files_by_age "$log_dir" $older_than)
        
        # Find large log files
        local large_files=$(find_files_by_size "$log_dir" $larger_than)
        
        # Combine files to remove (avoid duplicates)
        local all_files_to_remove=$(echo -e "$old_files\n$large_files" | sort -u)
        
        for file in $all_files_to_remove; do
            if [[ -f "$file" ]]; then
                local file_size=$(get_file_size_mb "$file")
                local age_days=$(( ($(date +%s) - $(stat -c%Y "$file")) / 86400 ))
                
                log_info "  $(basename "$file") - $(format_size $file_size) - ${age_days} days old"
                
                if [[ "$dry_run" == false ]]; then
                    rm -f "$file"
                    log_info "    Deleted"
                else
                    log_info "    Would delete"
                fi
                
                total_size=$((total_size + file_size))
                file_count=$((file_count + 1))
            fi
        done
    done
    
    if [[ $file_count -gt 0 ]]; then
        if [[ "$dry_run" == true ]]; then
            log_info "Would remove $file_count log file(s) totaling $(format_size $total_size)"
        else
            log_success "Removed $file_count log file(s) totaling $(format_size $total_size)"
        fi
    else
        log_info "No log files to clean up"
    fi
}

# Function to clean cache
cleanup_cache() {
    local remove_all=false
    local older_than=3
    local dry_run=false
    
    # Parse options
    while [[ $# -gt 0 ]]; do
        case $1 in
            --all)
                remove_all=true
                shift
                ;;
            --older-than)
                older_than="$2"
                shift 2
                ;;
            --dry-run)
                dry_run=true
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                return 1
                ;;
        esac
    done
    
    log_info "Cleaning up cache..."
    log_info "  Remove all: $remove_all"
    log_info "  Remove older than: $older_than days"
    log_info "  Dry run: $dry_run"
    
    local cache_dirs=(
        "$MEGATRON_ROOT/benchmark_cache_*"
        "$MEGATRON_ROOT/.cache"
        "$MEGATRON_ROOT/__pycache__"
    )
    
    local total_size=0
    local file_count=0
    
    for cache_pattern in "${cache_dirs[@]}"; do
        for cache_dir in $cache_pattern; do
            if [[ ! -d "$cache_dir" ]]; then
                continue
            fi
            
            log_info "Processing cache directory: $cache_dir"
            
            local files_to_remove=""
            if [[ "$remove_all" == true ]]; then
                files_to_remove=$(find "$cache_dir" -type f)
            else
                files_to_remove=$(find_files_by_age "$cache_dir" $older_than)
            fi
            
            for file in $files_to_remove; do
                if [[ -f "$file" ]]; then
                    local file_size=$(get_file_size_mb "$file")
                    local age_days=$(( ($(date +%s) - $(stat -c%Y "$file")) / 86400 ))
                    
                    log_info "  $(basename "$file") - $(format_size $file_size) - ${age_days} days old"
                    
                    if [[ "$dry_run" == false ]]; then
                        rm -f "$file"
                        log_info "    Deleted"
                    else
                        log_info "    Would delete"
                    fi
                    
                    total_size=$((total_size + file_size))
                    file_count=$((file_count + 1))
                fi
            done
            
            # Remove empty directories
            if [[ "$dry_run" == false ]]; then
                find "$cache_dir" -type d -empty -delete 2>/dev/null || true
            fi
        done
    done
    
    if [[ $file_count -gt 0 ]]; then
        if [[ "$dry_run" == true ]]; then
            log_info "Would remove $file_count cache file(s) totaling $(format_size $total_size)"
        else
            log_success "Removed $file_count cache file(s) totaling $(format_size $total_size)"
        fi
    else
        log_info "No cache files to clean up"
    fi
}

# Function to list cleanup candidates
list_cleanup_candidates() {
    log_info "Listing cleanup candidates..."
    
    # Checkpoints
    log_info "Checkpoints:"
    local checkpoint_dirs=("$MEGATRON_ROOT/checkpoints")
    for checkpoint_dir in "${checkpoint_dirs[@]}"; do
        if [[ -d "$checkpoint_dir" ]]; then
            local total_size=0
            local file_count=0
            find "$checkpoint_dir" -name "*.pt" -o -name "*.bin" -o -name "*.safetensors" | while read -r file; do
                if [[ -f "$file" ]]; then
                    local file_size=$(get_file_size_mb "$file")
                    local age_days=$(( ($(date +%s) - $(stat -c%Y "$file")) / 86400 ))
                    echo "  $file - $(format_size $file_size) - ${age_days} days old"
                    total_size=$((total_size + file_size))
                    file_count=$((file_count + 1))
                fi
            done
        fi
    done
    
    # Logs
    log_info "Logs:"
    local log_dirs=("$MEGATRON_ROOT/tensorboard_logs" "$MEGATRON_ROOT/logs")
    for log_dir in "${log_dirs[@]}"; do
        if [[ -d "$log_dir" ]]; then
            find "$log_dir" -type f -name "*.log" | while read -r file; do
                if [[ -f "$file" ]]; then
                    local file_size=$(get_file_size_mb "$file")
                    local age_days=$(( ($(date +%s) - $(stat -c%Y "$file")) / 86400 ))
                    echo "  $file - $(format_size $file_size) - ${age_days} days old"
                fi
            done
        fi
    done
    
    # Cache
    log_info "Cache:"
    local cache_dirs=("$MEGATRON_ROOT/benchmark_cache_*" "$MEGATRON_ROOT/.cache")
    for cache_pattern in "${cache_dirs[@]}"; do
        for cache_dir in $cache_pattern; do
            if [[ -d "$cache_dir" ]]; then
                find "$cache_dir" -type f | while read -r file; do
                    if [[ -f "$file" ]]; then
                        local file_size=$(get_file_size_mb "$file")
                        local age_days=$(( ($(date +%s) - $(stat -c%Y "$file")) / 86400 ))
                        echo "  $file - $(format_size $file_size) - ${age_days} days old"
                    fi
                done
            fi
        done
    done
}

# Function to clean all
cleanup_all() {
    local dry_run=false
    
    # Parse options
    while [[ $# -gt 0 ]]; do
        case $1 in
            --dry-run)
                dry_run=true
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                return 1
                ;;
        esac
    done
    
    log_info "Cleaning up all artifacts..."
    
    cleanup_checkpoints --dry-run=$dry_run
    echo
    cleanup_logs --dry-run=$dry_run
    echo
    cleanup_cache --dry-run=$dry_run
}

# Main function
main() {
    local command="$1"
    shift || true
    
    case "$command" in
        "checkpoints")
            cleanup_checkpoints "$@"
            ;;
        "logs")
            cleanup_logs "$@"
            ;;
        "cache")
            cleanup_cache "$@"
            ;;
        "all")
            cleanup_all "$@"
            ;;
        "list")
            list_cleanup_candidates
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
