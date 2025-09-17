# Tensor Visualization and Analysis Tools

This directory contains tools for analyzing and visualizing tensor data across different numerical formats (bf16, hifp8, mxfp8, mxfp4).

## Files Overview

- `overflow.py` - **Value-level analysis**: Shows percentage of tensor values that overflow/underflow within files
- `overflow_summary.py` - **File-level analysis**: Shows percentage of tensor files that have overflow/underflow issues  
- `layer_analysis.py` - Layer-specific analysis and visualization
- `distribution.py` - **Single tensor distribution**: Visualizes tensor value distribution against format's representable values
- `README.md` - This documentation file

### Key Difference
- **overflow.py**: "X% of values in this tensor file overflow/underflow"
- **overflow_summary.py**: "X% of tensor files contain overflow/underflow issues"

## Prerequisites

Make sure you have the required Python packages installed:

```bash
pip install torch numpy matplotlib
```

## Usage Examples

### 1. File-Level Analysis (Which files have issues?)

Analyze what percentage of tensor files contain overflow/underflow problems:

```python
# Run comprehensive file-level analysis
python overflow_summary.py --base-dir enhanced_tensor_logs --output-dir visualization

# Or with custom paths
python overflow_summary.py --base-dir /path/to/tensor/logs --output-dir /path/to/output
```

This will generate:
- `overflow_comprehensive_report.txt` - File-level analysis report (e.g., "15% of bf16 files have overflow")
- `overflow_detailed_results.json` - Complete results in JSON format
- `overflow_summary.csv` - Summary statistics in CSV format

### 2. Value-Level Analysis (How many values overflow?)

Analyze what percentage of tensor values overflow/underflow within specific files:

```python
# Analyze a single tensor file (shows % of values that overflow)
python overflow.py enhanced_tensor_logs/bf16/tensor_file.pt

# Analyze multiple tensor files at once
python overflow.py file1.pt file2.pt file3.pt

# Analyze all files in a directory (shows % of values in each file)
python overflow.py enhanced_tensor_logs/bf16/ --recursive

# Mix files and directories
python overflow.py tensor1.pt enhanced_tensor_logs/bf16/ tensor2.pt --recursive

# Generate CSV output
python overflow.py enhanced_tensor_logs/mxfp4/ --recursive --format csv --output mxfp4_value_results.csv
```

### 3. Single Tensor Distribution Analysis

Visualize how well a specific tensor fits within a data format's representable values:

```python
# Basic distribution analysis
python distribution.py enhanced_tensor_logs/bf16/tensor_file.pt

# With detailed statistics
python distribution.py enhanced_tensor_logs/mxfp4/tensor_file.pt --show-stats

# Custom output directory
python distribution.py tensor_file.pt --output-dir /path/to/output/
```

This will generate:
- `tensor_name.png` - Distribution plot with representable values overlay
- Console output with usability assessment (EXCELLENT/GOOD/CAUTION/POOR)

### 4. Layer-Specific Analysis

Analyze specific layers, ranks, and tensor types:

```python
# Analyze layer 0, rank 0, linear operations
python layer_analysis.py --layer 0 --rank 0 --type linear

# Analyze layer 16, rank 5, attention operations
python layer_analysis.py --layer 16 --rank 5 --type attention

# Analyze specific data format only
python layer_analysis.py --layer 0 --rank 0 --type linear --format bf16

# Custom input/output directories
python layer_analysis.py --layer 0 --rank 0 --type linear --base-dir /path/to/logs --output-dir /path/to/output
```

This will generate:
- `layer_X_rank_Y_TYPE_analysis.png` - Distribution plots for all data formats
- `layer_X_rank_Y_TYPE_report.txt` - Detailed analysis report

## Output Files Description

### Overflow Analysis Reports

#### File-Level Reports (from overflow_summary.py)
1. **overflow_comprehensive_report.txt**
   - **Primary focus**: Percentage of files with overflow/underflow issues
   - Format-specific file statistics (e.g., "25% of bf16 files have overflow")
   - Secondary reference: Overall value percentages
   - Risk assessment based on file percentages

2. **overflow_detailed_results.json**
   - Complete analysis results with both file and value statistics
   - Suitable for programmatic processing
   - Contains metadata for each analyzed file

3. **overflow_summary.csv**
   - Tabular summary focused on file-level statistics
   - Easy to import into spreadsheet applications

#### Value-Level Reports (from overflow.py)
1. **Individual file analysis**
   - **Primary focus**: Percentage of values that overflow/underflow within each file
   - Detailed statistics for each tensor file
   - File-by-file breakdown of problematic values

### Distribution Analysis Outputs (from distribution.py)

1. **Distribution Plots (PNG files)**
   - Histogram of tensor value distribution
   - **Red vertical lines**: All representable values in the data format
   - **Dark red dashed lines**: Overflow boundaries (±max_normal)
   - **Orange dotted lines**: Underflow boundaries (±min_denormal)
   - **Statistics box**: Complete analysis summary
   - **Usability assessment**: Color-coded recommendation (🟢🟡🟠🔴)

### Layer Analysis Outputs

1. **Distribution Plots (PNG files)**
   - Multi-panel plots showing tensor value distributions
   - Overlay of representable values for each format
   - Format range indicators and overflow/underflow warnings
   - Statistical information for each component

2. **Analysis Reports (TXT files)**
   - Detailed statistics for each tensor component
   - Overflow/underflow detection results
   - Recommendations based on findings

## Data Format Information

The tools understand the following numerical formats with their accurate ranges:

| Format | Description | Max Normal | Min Denormal | Supports Inf/NaN |
|--------|-------------|------------|--------------|------------------|
| bf16 | Brain Float 16-bit | ±6.55×10⁴ | 5.96×10⁻⁸ | Yes/Yes |
| hifp8 | Huawei HiFP8 E4M3 | ±3.28×10⁴ | 2.38×10⁻⁷ | Yes/Yes |
| mxfp8 | Microsoft MX FP8 E4M3 | ±4.48×10² (min normal: ±1.56×10⁻²) | 1.95×10⁻³ | No/Yes |
| mxfp4 | Microsoft MX FP4 E2M1 | ±6.00×10⁰ (min normal: ±1.0) | 5.00×10⁻¹ | No/No |

### Overflow and Underflow Definition

- **Overflow**: Values with magnitude exceeding the maximum representable value (|value| > max_normal)
- **Underflow**: Non-zero values closer to zero than the smallest representable non-zero value (0 < |value| < min_denormal)

Note: Underflow represents values that would be rounded to zero in the target format, causing precision loss.

## File Naming Convention

The tools expect tensor files to follow this naming pattern:
```
YYYYMMDD_HHMMSS_XXXX_iterNNN_TYPE_LXX_forward/backward_FORMAT_rankXX_sampleX_groupXXX_COMPONENT.pt
```

Where:
- `TYPE` = linear, attention
- `LXX` = layer number (e.g., L0, L16)
- `FORMAT` = bf16, hifp8, mxfp8, mxfp4
- `rankXX` = rank/card number
- `COMPONENT` = input, output, weight, bias, query, key, value, attention_weights, etc.

## Advanced Usage

### Programmatic Access

You can also use the modules programmatically:

```python
from overflow import analyze_file, DATA_TYPE_RANGES
from layer_analysis import find_matching_files, analyze_tensor

# Analyze a single file
result = analyze_file('/path/to/tensor.pt')
if result:
    print(f"Overflow: {result['overflow_percent']:.2f}%")

# Find files matching criteria
files = find_matching_files('enhanced_tensor_logs', layer=0, rank=0, tensor_type='linear')
```

### Batch Processing

For processing multiple configurations:

```bash
# Analyze multiple layers and ranks
for layer in {0..16}; do
    for rank in {0..7}; do
        python layer_analysis.py --layer $layer --rank $rank --type linear
        python layer_analysis.py --layer $layer --rank $rank --type attention
    done
done
```

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure PyTorch and other dependencies are installed
2. **File not found**: Check that tensor files exist in expected locations
3. **Memory issues**: For large tensors, the tools load data to CPU to conserve memory
4. **Permission errors**: Ensure write permissions for output directory

### Performance Tips

- Use `--format` option to analyze specific formats when you don't need all
- The tools automatically handle CUDA tensors by moving them to CPU
- Large batch analyses may take significant time - consider running in background

## Support

If you encounter issues or need additional features, please check:
1. File naming conventions match expected patterns
2. Required Python packages are installed
3. Input directories and files exist and are readable
4. Output directories are writable

For custom analysis requirements, the modular design allows easy extension of the existing tools.
