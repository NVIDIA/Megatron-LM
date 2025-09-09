#!/bin/bash
"""
One-Click Visualization Script
Automatically detects tensor files and generates visualization charts
"""

# Set default parameters
TENSOR_DIR=${1:-"./enhanced_tensor_logs"}
OUTPUT_DIR=${2:-"./draw"}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== One-Click Tensor Visualization Tool ==="
echo "Tensor Directory: $TENSOR_DIR"
echo "Output Directory: $OUTPUT_DIR"
echo "Script Directory: $SCRIPT_DIR"

# Check if tensor directory exists
if [ ! -d "$TENSOR_DIR" ]; then
    echo "Error: Tensor directory does not exist: $TENSOR_DIR"
    echo "Please ensure you have run the training script and generated tensor files"
    exit 1
fi

# Check if there are tensor files
TENSOR_COUNT=$(find "$TENSOR_DIR" -name "*.pt" | wc -l)
if [ "$TENSOR_COUNT" -eq 0 ]; then
    echo "Error: No tensor files (*.pt) found in $TENSOR_DIR"
    echo "Please ensure you have run the training script and generated tensor files"
    exit 1
fi

echo "Found $TENSOR_COUNT tensor files"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check Python environment
if ! command -v python &> /dev/null; then
    echo "Error: Python not found"
    exit 1
fi

# Check required Python packages
echo "Checking Python dependencies..."
python -c "import torch, matplotlib, numpy, pandas, seaborn" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Warning: Missing required Python packages, attempting to install..."
    pip install matplotlib numpy pandas seaborn scipy
fi

# Run enhanced quick visualization
echo "Running enhanced quick visualization..."
python "$SCRIPT_DIR/quick_visualize_enhanced.py" --tensor_dir "$TENSOR_DIR" --output_dir "$OUTPUT_DIR"

if [ $? -eq 0 ]; then
    echo "✅ Enhanced quick visualization completed"
else
    echo "❌ Enhanced quick visualization failed, trying basic version..."
    python "$SCRIPT_DIR/quick_visualize.py" --tensor_dir "$TENSOR_DIR" --output_dir "$OUTPUT_DIR"
    if [ $? -eq 0 ]; then
        echo "✅ Basic quick visualization completed"
    else
        echo "❌ Basic quick visualization also failed"
    fi
fi

# Run complete visualization (if file count is not too large)
if [ "$TENSOR_COUNT" -le 100 ]; then
    echo "Running complete visualization..."
    python "$SCRIPT_DIR/enhanced_tensor_visualizer.py" --tensor_dir "$TENSOR_DIR" --output_dir "$OUTPUT_DIR"
    
    if [ $? -eq 0 ]; then
        echo "✅ Complete visualization completed"
    else
        echo "❌ Complete visualization failed, trying basic version..."
        python "$SCRIPT_DIR/visualize_tensors.py" --tensor_dir "$TENSOR_DIR" --output_dir "$OUTPUT_DIR" --max_files 50
        if [ $? -eq 0 ]; then
            echo "✅ Basic complete visualization completed"
        else
            echo "❌ Basic complete visualization also failed"
        fi
    fi
else
    echo "⚠️  Large number of tensor files ($TENSOR_COUNT), skipping complete visualization"
    echo "   For complete visualization, please run manually:"
    echo "   python $SCRIPT_DIR/enhanced_tensor_visualizer.py --tensor_dir $TENSOR_DIR --output_dir $OUTPUT_DIR"
fi

# Display results
echo ""
echo "=== Visualization Complete ==="
echo "Output Directory: $OUTPUT_DIR"
echo "Generated files:"
find "$OUTPUT_DIR" -name "*.png" -o -name "*.txt" | head -10

# Check for main generated files
if [ -f "$OUTPUT_DIR/summary_analysis.png" ]; then
    echo ""
    echo "🎉 Main analysis chart: $OUTPUT_DIR/summary_analysis.png"
fi

if [ -f "$OUTPUT_DIR/detailed_tensor_stats.txt" ]; then
    echo "📊 Detailed statistical report: $OUTPUT_DIR/detailed_tensor_stats.txt"
fi

# Check subdirectories
if [ -d "$OUTPUT_DIR/quantization_analysis" ]; then
    echo "🔍 Quantization analysis charts: $OUTPUT_DIR/quantization_analysis/"
fi

if [ -d "$OUTPUT_DIR/attention_analysis" ]; then
    echo "🧠 Attention analysis charts: $OUTPUT_DIR/attention_analysis/"
fi

echo ""
echo "💡 Tips:"
echo "   - Check summary_analysis.png for basic statistical information"
echo "   - Check detailed_tensor_stats.txt for detailed statistical report"
echo "   - Check quantization_analysis/ directory for quantization type comparison"
echo "   - Check attention_analysis/ directory for attention layer analysis"
echo "   - Check *_quantization_comparison.png for quantization effect comparison"
echo "   - Check *_attention_analysis.png for detailed attention layer analysis"
