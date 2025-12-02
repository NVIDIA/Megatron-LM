#!/bin/bash
# =============================================================================
# Adapt Megatron-LM to use fake_quant_ops package
# 
# This script replaces all imports from 'quant' module to 'fake_quant_ops.quant'
# in Megatron-LM codebase (excluding the fake_quant_ops directory itself)
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=============================================================================="
echo "Adapting Megatron-LM to use fake_quant_ops package"
echo "=============================================================================="
echo "Project root: $PROJECT_ROOT"
echo ""

cd "$PROJECT_ROOT"

# Files to modify (excluding fake_quant_ops directory)
TARGET_FILES=(
    "megatron/core/tensor_parallel/layers.py"
    "megatron/core/transformer/dot_product_attention.py"
    "visualization/overflow/fake_quant_type_underflow_analysis.py"
    "visualization/overflow/fake_quant_underflow_analysis.py"
)

# Count modifications
TOTAL_MODIFIED=0
TOTAL_LINES_CHANGED=0

echo "Processing files..."
echo ""

for file in "${TARGET_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "⚠️  Warning: File not found: $file"
        continue
    fi
    
    echo "Processing: $file"
    
    # Create backup
    cp "$file" "${file}.bak"
    
    # Count original occurrences
    ORIG_COUNT=$(grep -c "from quant\." "$file" 2>/dev/null || echo 0)
    
    # Replace imports
    # from quant.mxfp -> from fake_quant_ops.quant.mxfp
    # from quant.hifp -> from fake_quant_ops.quant.hifp
    # from quant.bf16_operators -> from fake_quant_ops.quant.bf16_operators (if exists)
    # from quant.qtype -> from fake_quant_ops.quant.qtype
    # from quant.mxfp_scaling_test -> from fake_quant_ops.utils.mxfp_scaling_test
    
    sed -i 's|from quant\.mxfp_scaling_test|from fake_quant_ops.utils.mxfp_scaling_test|g' "$file"
    sed -i 's|from quant\.mxfp|from fake_quant_ops.quant.mxfp|g' "$file"
    sed -i 's|from quant\.hifp|from fake_quant_ops.quant.hifp|g' "$file"
    sed -i 's|from quant\.bf16_operators|from fake_quant_ops.quant.bf16_operators|g' "$file"
    sed -i 's|from quant\.qtype|from fake_quant_ops.quant.qtype|g' "$file"
    
    # Count new occurrences
    NEW_COUNT=$(grep -c "from fake_quant_ops\." "$file" 2>/dev/null || echo 0)
    
    if [ "$NEW_COUNT" -gt 0 ]; then
        echo "  ✅ Modified $NEW_COUNT import(s)"
        ((TOTAL_MODIFIED++))
        ((TOTAL_LINES_CHANGED += NEW_COUNT))
        
        # Show changes
        echo "  Changes:"
        diff -u "${file}.bak" "$file" | grep "^[-+].*from.*import" | head -10 || true
    else
        echo "  ℹ️  No changes needed"
        rm "${file}.bak"
    fi
    
    echo ""
done

echo "=============================================================================="
echo "Summary"
echo "=============================================================================="
echo "Files modified: $TOTAL_MODIFIED"
echo "Total import statements changed: $TOTAL_LINES_CHANGED"
echo ""

# Check if bf16_operators exists in fake_quant_ops
if [ ! -f "fake_quant_ops/quant/bf16_operators.py" ]; then
    echo "⚠️  Warning: fake_quant_ops/quant/bf16_operators.py not found"
    echo "   Some imports may fail. You may need to create this file or remove references."
    echo ""
fi

# Offer to remove backups
echo "Backup files created with .bak extension"
read -p "Remove backup files? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    for file in "${TARGET_FILES[@]}"; do
        if [ -f "${file}.bak" ]; then
            rm "${file}.bak"
            echo "  Removed: ${file}.bak"
        fi
    done
    echo "✅ All backup files removed"
else
    echo "ℹ️  Backup files kept. Remove manually with: find . -name '*.bak' -delete"
fi

echo ""
echo "=============================================================================="
echo "Adaptation complete!"
echo "=============================================================================="
echo ""
echo "Next steps:"
echo "1. Ensure fake_quant_ops is properly installed or in PYTHONPATH"
echo "2. If bf16_operators is referenced, create it or remove references"
echo "3. Test imports: python -c 'from fake_quant_ops.quant import mxfp, hifp'"
echo "4. Run your Megatron-LM training/inference to verify"
echo ""

