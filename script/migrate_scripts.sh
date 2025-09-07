#!/bin/bash

# Migration script for transitioning from old script structure to new structure
# This script helps users migrate from the old individual scripts to the new unified system

# Source configuration files
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config/common.sh"

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [COMMAND] [OPTIONS]

Commands:
    backup                    Backup old scripts
    migrate                   Migrate old scripts to new structure
    compare                   Compare old and new script functionality
    cleanup                   Remove old scripts (after migration)
    help                      Show this help message

Options:
    --dry-run                 Show what would be done without executing
    --force                   Force operations without confirmation

Examples:
    # Backup old scripts
    $0 backup

    # Compare old and new functionality
    $0 compare

    # Migrate to new structure
    $0 migrate --dry-run

    # Clean up old scripts after successful migration
    $0 cleanup --force

EOF
}

# Function to backup old scripts
backup_old_scripts() {
    local dry_run="$1"
    local force="$2"
    
    log_info "Backing up old scripts..."
    
    local backup_dir="${SCRIPT_DIR}/old_scripts_backup_$(date +%Y%m%d_%H%M%S)"
    local old_dirs=("llama31-8b" "llama32-1b")
    
    if [[ "$dry_run" == "true" ]]; then
        log_info "Dry run - would create backup at: $backup_dir"
        for dir in "${old_dirs[@]}"; do
            if [[ -d "${SCRIPT_DIR}/$dir" ]]; then
                log_info "  Would backup: $dir"
            fi
        done
        return 0
    fi
    
    # Create backup directory
    mkdir -p "$backup_dir"
    
    # Backup old script directories
    for dir in "${old_dirs[@]}"; do
        if [[ -d "${SCRIPT_DIR}/$dir" ]]; then
            log_info "Backing up: $dir"
            cp -r "${SCRIPT_DIR}/$dir" "$backup_dir/"
        fi
    done
    
    # Backup original process_data.sh
    if [[ -f "${SCRIPT_DIR}/process_data.sh" ]]; then
        log_info "Backing up: process_data.sh"
        cp "${SCRIPT_DIR}/process_data.sh" "$backup_dir/"
    fi
    
    log_success "Backup completed at: $backup_dir"
}

# Function to compare old and new functionality
compare_functionality() {
    log_info "Comparing old and new script functionality..."
    
    echo
    log_info "=== OLD SCRIPT STRUCTURE ==="
    echo "llama31-8b/"
    echo "├── pretrain_llama_mock.sh"
    echo "├── pretrain_llama_wikipedia.sh"
    echo "├── pretrain_llama_wikipedia_bf16.sh"
    echo "├── pretrain_llama_wikipedia_fp4.sh"
    echo "├── pretrain_llama_wikipedia_fp8.sh"
    echo "└── pretrain_llama_wikitext.sh"
    echo
    echo "llama32-1b/wikipedia/"
    echo "├── pretrain_llama_wikipedia.sh"
    echo "├── pretrain_llama_wikipedia_bf16.sh"
    echo "├── pretrain_llama_wikipedia_fp4.sh"
    echo "├── pretrain_llama_wikipedia_fp8.sh"
    echo "├── pretrain_llama_wikipedia_hifp8.sh"
    echo "├── pretrain_llama_wikipedia_FA_fp4.sh"
    echo "├── pretrain_llama_wikipedia_FA_fp8.sh"
    echo "├── pretrain_llama_wikipedia_FA_hifp8.sh"
    echo "├── pretrain_llama_wikipedia_FA_PV_fp4.sh"
    echo "├── pretrain_llama_wikipedia_FA_PV_fp8.sh"
    echo "└── pretrain_llama_wikipedia_FA_PV_hifp8.sh"
    echo
    echo "process_data.sh"
    
    echo
    log_info "=== NEW SCRIPT STRUCTURE ==="
    echo "config/"
    echo "├── common.sh              # Common utilities and functions"
    echo "├── models.sh              # Model configurations"
    echo "└── training.sh            # Training configurations"
    echo
    echo "utils/"
    echo "├── check_system.sh        # System validation"
    echo "└── cleanup.sh             # Cleanup utilities"
    echo
    echo "train_base.sh              # Unified training script"
    echo "experiment_launcher.sh     # Experiment launcher"
    echo "process_data_improved.sh   # Enhanced data processing"
    echo "migrate_scripts.sh         # This migration script"
    echo "README.md                  # Documentation"
    
    echo
    log_info "=== MIGRATION MAPPING ==="
    echo "Old Script                                    → New Command"
    echo "───────────────────────────────────────────────┼────────────────────────────────────────────"
    echo "llama31-8b/pretrain_llama_mock.sh             → experiment_launcher.sh run llama3_8b_mock_fast"
    echo "llama31-8b/pretrain_llama_wikipedia.sh        → experiment_launcher.sh run llama3_8b_wikipedia_fp8"
    echo "llama31-8b/pretrain_llama_wikipedia_bf16.sh   → experiment_launcher.sh run llama3_8b_wikipedia_bf16"
    echo "llama31-8b/pretrain_llama_wikipedia_fp8.sh    → experiment_launcher.sh run llama3_8b_wikipedia_fp8"
    echo "llama32-1b/wikipedia/pretrain_llama_wikipedia.sh → experiment_launcher.sh run llama32_1b_wikipedia_fp8"
    echo "llama32-1b/wikipedia/pretrain_llama_wikipedia_bf16.sh → experiment_launcher.sh run llama32_1b_wikipedia_bf16"
    echo "llama32-1b/wikipedia/pretrain_llama_wikipedia_fp8.sh → experiment_launcher.sh run llama32_1b_wikipedia_fp8"
    echo "process_data.sh                               → process_data_improved.sh"
    
    echo
    log_info "=== NEW FEATURES ==="
    echo "✓ Unified configuration system"
    echo "✓ Comprehensive error handling and validation"
    echo "✓ System health checks"
    echo "✓ Automatic cleanup utilities"
    echo "✓ Dry-run mode for testing"
    echo "✓ Detailed logging and progress tracking"
    echo "✓ Flexible experiment management"
    echo "✓ Better documentation and help system"
    
    echo
    log_info "=== BENEFITS ==="
    echo "• Reduced code duplication"
    echo "• Easier maintenance and updates"
    echo "• Better error handling and debugging"
    echo "• Consistent interface across all operations"
    echo "• Automatic system validation"
    echo "• Built-in cleanup and maintenance tools"
    echo "• Comprehensive documentation"
}

# Function to migrate scripts
migrate_scripts() {
    local dry_run="$1"
    local force="$2"
    
    log_info "Migrating to new script structure..."
    
    if [[ "$dry_run" == "true" ]]; then
        log_info "Dry run - would perform the following actions:"
        echo
        log_info "1. Validate new script structure"
        log_info "2. Create symbolic links for backward compatibility"
        log_info "3. Update any automation scripts"
        log_info "4. Provide migration instructions"
        return 0
    fi
    
    # Validate new script structure
    log_info "Validating new script structure..."
    local required_files=(
        "config/common.sh"
        "config/models.sh"
        "config/training.sh"
        "train_base.sh"
        "experiment_launcher.sh"
        "process_data_improved.sh"
        "utils/check_system.sh"
        "utils/cleanup.sh"
    )
    
    for file in "${required_files[@]}"; do
        if [[ ! -f "${SCRIPT_DIR}/$file" ]]; then
            log_error "Required file missing: $file"
            return 1
        fi
    done
    
    log_success "New script structure validated"
    
    # Create backward compatibility links
    log_info "Creating backward compatibility links..."
    
    # Create wrapper scripts for common old commands
    create_wrapper_script() {
        local old_script="$1"
        local new_command="$2"
        local wrapper_path="${SCRIPT_DIR}/$old_script"
        
        cat > "$wrapper_path" << EOF
#!/bin/bash
# Backward compatibility wrapper
# This script redirects to the new unified system

echo "Warning: This script is deprecated. Please use the new command:"
echo "  $new_command"
echo
echo "Redirecting to new command..."
echo

$new_command "\$@"
EOF
        chmod +x "$wrapper_path"
        log_info "Created wrapper: $old_script"
    }
    
    # Create wrappers for common scripts
    create_wrapper_script "pretrain_llama3_8b_wikipedia_fp8.sh" "./script/experiment_launcher.sh run llama3_8b_wikipedia_fp8"
    create_wrapper_script "pretrain_llama3_8b_wikipedia_bf16.sh" "./script/experiment_launcher.sh run llama3_8b_wikipedia_bf16"
    create_wrapper_script "pretrain_llama3_8b_mock.sh" "./script/experiment_launcher.sh run llama3_8b_mock_fast"
    create_wrapper_script "pretrain_llama32_1b_wikipedia_fp8.sh" "./script/experiment_launcher.sh run llama32_1b_wikipedia_fp8"
    create_wrapper_script "pretrain_llama32_1b_wikipedia_bf16.sh" "./script/experiment_launcher.sh run llama32_1b_wikipedia_bf16"
    create_wrapper_script "pretrain_llama32_1b_mock.sh" "./script/experiment_launcher.sh run llama32_1b_mock_fast"
    
    # Create wrapper for data processing
    create_wrapper_script "process_data_new.sh" "./script/process_data_improved.sh"
    
    log_success "Migration completed successfully"
    
    # Provide next steps
    echo
    log_info "=== NEXT STEPS ==="
    echo "1. Test the new system:"
    echo "   ./script/utils/check_system.sh"
    echo
    echo "2. Try a quick test with mock data:"
    echo "   ./script/experiment_launcher.sh run llama3_8b_mock_fast"
    echo
    echo "3. List available experiments:"
    echo "   ./script/experiment_launcher.sh list"
    echo
    echo "4. Update any automation scripts to use new commands"
    echo
    echo "5. When ready, clean up old scripts:"
    echo "   ./script/migrate_scripts.sh cleanup --force"
}

# Function to cleanup old scripts
cleanup_old_scripts() {
    local dry_run="$1"
    local force="$2"
    
    if [[ "$force" != "true" ]]; then
        log_warn "This will permanently delete old script directories."
        log_warn "Make sure you have backed them up and tested the new system."
        echo
        read -p "Are you sure you want to continue? (yes/no): " confirm
        if [[ "$confirm" != "yes" ]]; then
            log_info "Cleanup cancelled"
            return 0
        fi
    fi
    
    log_info "Cleaning up old scripts..."
    
    local old_dirs=("llama31-8b" "llama32-1b")
    local old_files=("process_data.sh")
    
    if [[ "$dry_run" == "true" ]]; then
        log_info "Dry run - would remove:"
        for dir in "${old_dirs[@]}"; do
            if [[ -d "${SCRIPT_DIR}/$dir" ]]; then
                log_info "  Directory: $dir"
            fi
        done
        for file in "${old_files[@]}"; do
            if [[ -f "${SCRIPT_DIR}/$file" ]]; then
                log_info "  File: $file"
            fi
        done
        return 0
    fi
    
    # Remove old directories
    for dir in "${old_dirs[@]}"; do
        if [[ -d "${SCRIPT_DIR}/$dir" ]]; then
            log_info "Removing directory: $dir"
            rm -rf "${SCRIPT_DIR}/$dir"
        fi
    done
    
    # Remove old files
    for file in "${old_files[@]}"; do
        if [[ -f "${SCRIPT_DIR}/$file" ]]; then
            log_info "Removing file: $file"
            rm -f "${SCRIPT_DIR}/$file"
        fi
    done
    
    log_success "Cleanup completed"
}

# Main function
main() {
    local command="$1"
    shift || true
    
    local dry_run="false"
    local force="false"
    
    # Parse options
    while [[ $# -gt 0 ]]; do
        case $1 in
            --dry-run)
                dry_run="true"
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
        "backup")
            backup_old_scripts "$dry_run" "$force"
            ;;
        "migrate")
            migrate_scripts "$dry_run" "$force"
            ;;
        "compare")
            compare_functionality
            ;;
        "cleanup")
            cleanup_old_scripts "$dry_run" "$force"
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
