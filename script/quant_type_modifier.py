#!/usr/bin/env python3
"""
Script to modify custom_quant_type in Megatron-LM source code
"""

import os
import re
import argparse
import shutil
from pathlib import Path

def backup_file(file_path):
    """Create a backup of the original file."""
    backup_path = f"{file_path}.backup"
    if not os.path.exists(backup_path):
        shutil.copy2(file_path, backup_path)
        print(f"Created backup: {backup_path}")
    else:
        print(f"Backup already exists: {backup_path}")

def modify_quant_type(file_path, line_number, new_quant_type):
    """Modify custom_quant_type in a specific line of a file."""
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist")
        return False
    
    # Create backup
    backup_file(file_path)
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Check if line number is valid
    if line_number > len(lines):
        print(f"Error: Line {line_number} does not exist in {file_path}")
        return False
    
    # Get the original line
    original_line = lines[line_number - 1].strip()
    print(f"Original line {line_number}: {original_line}")
    
    # Modify the specific line
    modified_line = re.sub(
        r"custom_quant_type = '[^']*'",
        f"custom_quant_type = '{new_quant_type}'",
        lines[line_number - 1]
    )
    
    lines[line_number - 1] = modified_line
    
    with open(file_path, 'w') as f:
        f.writelines(lines)
    
    print(f"Modified line {line_number}: {modified_line.strip()}")
    return True

def restore_backup(file_path):
    """Restore from backup file."""
    backup_path = f"{file_path}.backup"
    if os.path.exists(backup_path):
        shutil.copy2(backup_path, file_path)
        print(f"Restored from backup: {backup_path}")
    else:
        print(f"No backup found: {backup_path}")

def main():
    parser = argparse.ArgumentParser(description="Modify custom_quant_type in Megatron-LM source code")
    parser.add_argument("--linear-quant", choices=['hifp8', 'mxfp8', 'mxfp4', 'none'], 
                       help="Linear layer quantization type")
    parser.add_argument("--qk-quant", choices=['hifp8', 'mxfp8', 'mxfp4', 'none'], 
                       help="QK attention quantization type")
    parser.add_argument("--pv-quant", choices=['hifp8', 'mxfp8', 'mxfp4', 'none'], 
                       help="PV attention quantization type")
    parser.add_argument("--restore", action="store_true", help="Restore from backup files")
    parser.add_argument("--check", action="store_true", help="Check current quantization types")
    
    args = parser.parse_args()
    
    # Get Megatron-LM root directory
    script_dir = Path(__file__).parent
    megatron_root = script_dir.parent
    
    # File paths
    layers_file = megatron_root / "megatron/core/tensor_parallel/layers.py"
    attention_file = megatron_root / "megatron/core/transformer/dot_product_attention.py"
    
    if args.restore:
        print("Restoring from backup files...")
        restore_backup(layers_file)
        restore_backup(attention_file)
        return 0
    
    if args.check:
        print("Checking current quantization types...")
        print(f"\nLinear layer ({layers_file}):")
        with open(layers_file, 'r') as f:
            lines = f.readlines()
            if len(lines) >= 783:
                print(f"  Line 783: {lines[782].strip()}")
        
        print(f"\nAttention layer ({attention_file}):")
        with open(attention_file, 'r') as f:
            lines = f.readlines()
            if len(lines) >= 166:
                print(f"  Line 166 (QK): {lines[165].strip()}")
            if len(lines) >= 238:
                print(f"  Line 238 (PV): {lines[237].strip()}")
        return 0
    
    # Modify quantization types
    success = True
    
    if args.linear_quant:
        print(f"Modifying linear layer quantization to {args.linear_quant}...")
        success &= modify_quant_type(layers_file, 783, args.linear_quant)
    
    if args.qk_quant:
        print(f"Modifying QK attention quantization to {args.qk_quant}...")
        success &= modify_quant_type(attention_file, 166, args.qk_quant)
    
    if args.pv_quant:
        print(f"Modifying PV attention quantization to {args.pv_quant}...")
        success &= modify_quant_type(attention_file, 238, args.pv_quant)
    
    if not args.linear_quant and not args.qk_quant and not args.pv_quant:
        print("No quantization type specified. Use --help for usage information.")
        return 1
    
    if success:
        print("\n✅ All modifications completed successfully!")
        print("\nTo verify changes, run:")
        print("  python3 modify_quant_type.py --check")
        print("\nTo restore original files, run:")
        print("  python3 modify_quant_type.py --restore")
    else:
        print("\n❌ Some modifications failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
