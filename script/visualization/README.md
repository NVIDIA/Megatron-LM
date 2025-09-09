# Visualization Scripts

This directory contains scripts and tools for visualizing tensor data.

## Script Files

### Visualization Scripts
- **`visualize_tensors.py`** - Complete tensor visualization tool
- **`quick_visualize.py`** - Quick visualization script
- **`one_click_visualize.sh`** - One-click visualization script

## Features

### 1. Complete Tensor Visualization Tool (visualize_tensors.py)
- **Distribution Plots**: Tensor value distribution histograms, box plots, Q-Q plots
- **Heatmaps**: Heatmap visualization of tensor data
- **Comparison Plots**: Comparative analysis of different quantization types
- **Statistical Plots**: Statistical information summary charts
- **Attention Analysis**: Specialized attention tensor analysis

### 2. Quick Visualization Script (quick_visualize.py)
- Generate basic statistical charts
- Quick analysis of tensor data distribution
- Generate statistical information text files

### 3. One-Click Visualization Script (one_click_visualize.sh)
- Automatically detect tensor files
- Run quick and complete visualization
- Generate all analysis charts

## Usage

### 1. One-Click Visualization (Recommended)
```bash
# Basic usage
./one_click_visualize.sh

# Custom parameters
./one_click_visualize.sh ./enhanced_tensor_logs ./draw
```

### 2. Quick Visualization
```bash
# Basic usage
python quick_visualize.py

# Custom parameters
python quick_visualize.py \
    --tensor_dir ./enhanced_tensor_logs \
    --output_dir ./draw
```

### 3. Complete Visualization
```bash
# Basic usage
python visualize_tensors.py

# Custom parameters
python visualize_tensors.py \
    --tensor_dir ./enhanced_tensor_logs \
    --output_dir ./draw \
    --max_files 50
```

## Output Files

### Directory Structure
```
draw/
├── quick_analysis.png          # Quick analysis chart
├── tensor_stats.txt           # Statistical information text
├── distributions/             # Distribution plots directory
├── heatmaps/                  # Heatmap directory
├── comparisons/               # Comparison plots directory
├── statistics/                # Statistical plots directory
└── attention_maps/            # Attention analysis plots directory
```

### Chart Types
- **quick_analysis.png**: Comprehensive analysis with 4 subplots
  - All tensor value distribution histogram
  - Quantization type distribution pie chart
  - Layer type distribution pie chart
  - Operation type distribution pie chart

- **distributions/**: Detailed tensor distribution analysis plots
- **heatmaps/**: Heatmaps of tensor data
- **comparisons/**: Comparison plots of different quantization types
- **statistics/**: Statistical information summary plots
- **attention_maps/**: Specialized attention tensor analysis plots

## Requirements

### Python Dependencies
```bash
pip install matplotlib seaborn pandas scipy
```

### Environment Variables
```bash
export TENSOR_SAVE_DIR="./enhanced_tensor_logs"
export TENSOR_SAVE_ENABLED="true"
```

## Parameter Description

### visualize_tensors.py
- `--tensor_dir`: Tensor file directory (default: ./enhanced_tensor_logs)
- `--output_dir`: Output image directory (default: ./draw)
- `--max_files`: Maximum number of files to process (default: 50)

### quick_visualize.py
- `--tensor_dir`: Tensor file directory (default: ./enhanced_tensor_logs)
- `--output_dir`: Output directory (default: ./draw)

### one_click_visualize.sh
- `$1`: Tensor file directory (default: ./enhanced_tensor_logs)
- `$2`: Output directory (default: ./draw)

## Use Cases

### 1. Quantization Research
- Analyze the impact of different quantization types on tensor distribution
- Compare tensor characteristics of forward and backward passes
- Study tensor behavior of attention and linear layers

### 2. Model Debugging
- Visualize tensor value distribution
- Detect outliers and value ranges
- Analyze statistical properties of tensors

### 3. Performance Analysis
- Compare performance of different quantization methods
- Analyze tensor memory usage patterns
- Optimize quantization strategies

## Notes

1. **File Format**: Supports .pt format tensor files
2. **Memory Usage**: Large files will be automatically sampled to avoid memory issues
3. **BFloat16 Support**: Automatically converted to Float32 to support numpy operations
4. **Font Support**: May need to install appropriate fonts for proper label display
5. **File Permissions**: Ensure scripts have execution permissions

## Troubleshooting

### Common Issues
1. **ModuleNotFoundError**: Install missing Python packages
2. **Font Warnings**: Ignore font warnings, they don't affect functionality
3. **Insufficient Memory**: Reduce max_files parameter or increase system memory
4. **File Permissions**: Use chmod +x to set execution permissions

### Debugging Tips
- Use quick_visualize.py for quick testing
- Check if tensor files are generated correctly
- Review error logs to locate issues
- Test with small datasets
