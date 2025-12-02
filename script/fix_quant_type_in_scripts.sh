#!/bin/bash
# =============================================================================
# Fix Quantization Type in Training Scripts
# 
# Updates sed commands in training scripts to match the quantization type
# indicated by the filename
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

echo "Fixing quantization types in training scripts..."
echo ""

# llama32-1b scripts
echo "Processing llama32-1b scripts..."

# MXFP8 scripts
for script in script/llama32-1b/*mxfp8*.sh; do
    [ -f "$script" ] || continue
    echo "  Updating $(basename "$script") to mxfp8"
    sed -i "s/^\([[:space:]]*custom_quant_type[[:space:]]*=[[:space:]]*\)'[^']*'/\1'mxfp8'/g" \
        megatron/core/tensor_parallel/layers.py
    sed -i "s/^\([[:space:]]*custom_quant_type[[:space:]]*=[[:space:]]*\)'[^']*'/\1'mxfp8'/g" \
        megatron/core/transformer/dot_product_attention.py
    
    # Update the sed commands in the script itself
    sed -i "s|sed.*custom_quant_type.*layers\.py|sed -i \"s/^\\\\([[:space:]]*custom_quant_type[[:space:]]*=[[:space:]]*\\\\)'[^']*'/\\\\\\\\1'mxfp8'/\" \\\\\\\\\\n    megatron/core/tensor_parallel/layers.py|g" "$script" 2>/dev/null || true
    sed -i "s|'mxfp[48]'|'mxfp8'|g" "$script"
    sed -i "s|to mxfp[48]|to mxfp8|g" "$script"
done

# MXFP4 scripts
for script in script/llama32-1b/*mxfp4*.sh; do
    [ -f "$script" ] || continue
    echo "  Updating $(basename "$script") to mxfp4"
    sed -i "s|'mxfp8'|'mxfp4'|g" "$script"
    sed -i "s|to mxfp8|to mxfp4|g" "$script"
done

# HIFP8 scripts
for script in script/llama32-1b/*hifp8*.sh; do
    [ -f "$script" ] || continue
    echo "  Updating $(basename "$script") to hifp8"
    sed -i "s|'mxfp[48]'|'hifp8'|g" "$script"
    sed -i "s|to mxfp[48]|to hifp8|g" "$script"
done

echo ""
echo "Processing llama31-8b scripts..."

# MXFP8 scripts
for script in script/llama31-8b/*mxfp8*.sh; do
    [ -f "$script" ] || continue
    echo "  Updating $(basename "$script") to mxfp8"
    sed -i "s|'mxfp4'|'mxfp8'|g" "$script"
    sed -i "s|'hifp8'|'mxfp8'|g" "$script"
    sed -i "s|to mxfp4|to mxfp8|g" "$script"
    sed -i "s|to hifp8|to mxfp8|g" "$script"
done

# MXFP4 scripts
for script in script/llama31-8b/*mxfp4*.sh; do
    [ -f "$script" ] || continue
    echo "  Updating $(basename "$script") to mxfp4"
    sed -i "s|'mxfp8'|'mxfp4'|g" "$script"
    sed -i "s|'hifp8'|'mxfp4'|g" "$script"
    sed -i "s|to mxfp8|to mxfp4|g" "$script"
    sed -i "s|to hifp8|to mxfp4|g" "$script"
done

# HIFP8 scripts
for script in script/llama31-8b/*hifp8*.sh; do
    [ -f "$script" ] || continue
    echo "  Updating $(basename "$script") to hifp8"
    sed -i "s|'mxfp[48]'|'hifp8'|g" "$script"
    sed -i "s|to mxfp[48]|to hifp8|g" "$script"
done

echo ""
echo "✅ All scripts updated!"
echo ""
echo "Verification:"
echo "  MXFP8 scripts should have: custom_quant_type='mxfp8'"
echo "  MXFP4 scripts should have: custom_quant_type='mxfp4'"
echo "  HIFP8 scripts should have: custom_quant_type='hifp8'"
echo "  BF16 scripts: No modification needed"

