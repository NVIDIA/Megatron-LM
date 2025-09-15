# Unified Tensor Visualization Tool

This directory contains a simplified and optimized tensor visualization tool that combines the best features from all previous scripts.

## Main Script

### `tensor_visualizer.py` - Unified Tensor Visualization Tool
- **Comprehensive Analysis**: Quantization type comparison, HiFP8 distribution analysis, global statistics
- **Multi-threading Support**: Configurable worker threads for faster processing
- **Progress Tracking**: Real-time progress bars using tqdm
- **Error Handling**: Robust error handling with multiple fallback methods for loading tensors
- **Output Organization**: Well-organized output with subdirectories for different analysis types

## Features

### 1. Quantization Type Analysis
- File count comparison across quantization types (bf16, mxfp8, mxfp4, hifp8)
- Layer and sample distribution analysis
- Comprehensive comparison charts

### 2. HiFP8 Distribution Analysis
- Value distribution by layer and sample
- Statistical measures (mean, std, min, max) visualization
- Quantile analysis and range analysis
- Layer-sample heatmaps

### 3. Global Statistics
- Comprehensive statistical analysis across all dimensions
- JSON and text report generation
- Detailed breakdown by quantization type, layer, sample, and layer type

## Usage

### 1. Main Shell Script (Recommended)
```bash
# Basic usage
./run_tensor_draw.sh

# Custom parameters
./run_tensor_draw.sh ./enhanced_tensor_logs ./draw 4
```

### 2. Direct Python Script
```bash
# Basic usage
python script/visualization/tensor_visualizer.py

# Custom parameters
python script/visualization/tensor_visualizer.py \
    --tensor_dir ./enhanced_tensor_logs \
    --output_dir ./draw \
    --max_workers 4
```

## Parameters

### tensor_visualizer.py
- `--tensor_dir`: Tensor file directory (default: ./enhanced_tensor_logs)
- `--output_dir`: Output directory (default: ./draw)
- `--max_workers`: Maximum number of worker threads (default: 4)

### run_tensor_draw.sh
- `$1`: Tensor file directory (default: ./enhanced_tensor_logs)
- `$2`: Output directory (default: ./draw)
- `$3`: Max workers (default: 4)

## Output Structure

```
draw/
├── quantization_analysis/
│   └── quantization_comparison.png
├── hifp8_analysis/
│   └── hifp8_distribution_analysis.png
├── global_statistics/
│   ├── global_statistics.json
│   └── global_statistics_report.txt
└── statistics/
    └── detailed_statistics_report.txt
```

## Requirements

### Python Dependencies
```bash
pip install torch matplotlib numpy pandas seaborn scipy tqdm
```

### Environment Variables
```bash
export TENSOR_SAVE_DIR="./enhanced_tensor_logs"
export TENSOR_SAVE_ENABLED="true"
```

## Supported Data Formats

- **Quantization Types**: bf16, mxfp8, mxfp4, hifp8
- **File Format**: .pt (PyTorch tensor files)
- **Data Structure**: Enhanced tensor format with metadata
- **Layers**: 1-16 layers
- **Samples**: 0, 1, 2 samples

## Use Cases

### 1. Quantization Research
- Analyze the impact of different quantization types on tensor distribution
- Compare tensor characteristics across different quantization methods
- Study tensor behavior patterns

### 2. Model Debugging
- Visualize tensor value distributions
- Detect outliers and value ranges
- Analyze statistical properties of tensors

### 3. Performance Analysis
- Compare performance of different quantization methods
- Analyze tensor memory usage patterns
- Optimize quantization strategies

## Key Improvements

1. **Simplified Codebase**: Single unified script instead of multiple overlapping scripts
2. **Better Error Handling**: Multiple fallback methods for loading corrupted or incompatible files
3. **Optimized Performance**: Multi-threading support with configurable worker threads
4. **Comprehensive Analysis**: All essential analysis features in one tool
5. **Clean Output**: Well-organized output structure with clear file naming

## Troubleshooting

### Common Issues
1. **ModuleNotFoundError**: Install missing Python packages
2. **File Loading Errors**: The tool includes multiple fallback methods for loading files
3. **Memory Issues**: Large files are automatically sampled to prevent memory overflow
4. **Permission Errors**: Ensure proper file permissions for output directory

### Debugging Tips
- Check if tensor files are generated correctly
- Review error logs for specific issues
- Test with small datasets first
- Ensure all dependencies are installed

## Notes

1. **File Format**: Supports .pt format tensor files with enhanced metadata
2. **Memory Management**: Large files are automatically sampled to avoid memory issues
3. **BFloat16 Support**: Automatically converted to Float32 for numpy operations
4. **Progress Tracking**: Real-time progress bars for long-running operations
5. **Error Recovery**: Robust error handling with graceful degradation