# Megatron-LM Scripts Documentation

This directory contains a comprehensive set of scripts for training and managing Megatron-LM experiments. The scripts have been redesigned for better robustness, maintainability, and ease of use.

## Directory Structure

```
script/
├── config/                    # Configuration files
│   ├── common.sh             # Common utilities and functions
│   ├── models.sh             # Model-specific configurations
│   └── training.sh           # Training configurations
├── utils/                     # Utility scripts
│   ├── check_system.sh       # System validation
│   └── cleanup.sh            # Cleanup utilities
├── train_base.sh             # Base training script
├── experiment_launcher.sh    # Experiment launcher
├── process_data_improved.sh  # Improved data processing
└── README.md                 # This documentation
```

## Quick Start

### 1. System Check

Before starting any training, run a system check to ensure everything is properly configured:

```bash
./script/utils/check_system.sh
```

### 2. Data Processing

Process your training data:

```bash
./script/process_data_improved.sh \
    --input './dataset/dolma/**/*.json.gz' \
    --output-prefix ./dataset/dolma_processed \
    --tokenizer-path ./model/llama3/ \
    --workers 32 --partitions 8
```

### 3. Training

#### Using the Experiment Launcher (Recommended)

List available experiments:
```bash
./script/experiment_launcher.sh list
```

Run a predefined experiment:
```bash
./script/experiment_launcher.sh run llama3_8b_wikipedia_fp8
```

#### Using the Base Training Script

For custom configurations:
```bash
./script/train_base.sh \
    --model llama3_8b \
    --experiment-name my_experiment \
    --checkpoint-path checkpoints/llama3_8b/my_experiment \
    --tensorboard-path tensorboard_logs/llama3_8b/my_experiment \
    --data-path dataset/wikipedia_processed/wikipedia_processed_text_document \
    --tokenizer-path model/llama3 \
    --training-config standard \
    --dtype fp8
```

## Scripts Overview

### Configuration Files

#### `config/common.sh`
- Common utilities and functions
- Error handling and logging
- Environment variable setup
- Path validation

#### `config/models.sh`
- Model-specific configurations (LLaMA 3 8B, LLaMA 3.2 1B)
- Model parameter definitions
- Path generation for different models

#### `config/training.sh`
- Training configurations (standard, fast)
- Data type configurations (fp8, bf16, fp16)
- Distributed training configurations

### Core Scripts

#### `train_base.sh`
The main training script with comprehensive features:

**Features:**
- Robust error handling and validation
- Support for multiple models and configurations
- Automatic path setup and validation
- Comprehensive logging
- Dry-run mode for testing

**Usage:**
```bash
./script/train_base.sh [OPTIONS]

Required Options:
    --model MODEL_NAME              Model to train (llama3_8b, llama32_1b)
    --experiment-name NAME          Name for this experiment
    --checkpoint-path PATH          Path to save checkpoints
    --tensorboard-path PATH         Path to save tensorboard logs

Data Options (choose one):
    --data-path PATH                Path to training data
    --tokenizer-path PATH           Path to tokenizer
    --use-mock-data                 Use mock data for testing

Training Options:
    --training-config CONFIG        Training configuration (standard, fast)
    --dtype DTYPE                   Data type (fp8, bf16, fp16)
    --distributed-config CONFIG     Distributed configuration (single_node, multi_node)

Other Options:
    --dry-run                       Show what would be executed without running
    --help                          Show this help message
```

#### `experiment_launcher.sh`
Simplified interface for common experiments:

**Predefined Experiments:**
- `llama3_8b_wikipedia_fp8` - Train LLaMA 3 8B on Wikipedia with FP8
- `llama3_8b_wikipedia_bf16` - Train LLaMA 3 8B on Wikipedia with BF16
- `llama32_1b_wikipedia_fp8` - Train LLaMA 3.2 1B on Wikipedia with FP8
- `llama32_1b_wikipedia_bf16` - Train LLaMA 3.2 1B on Wikipedia with BF16
- `llama3_8b_mock_fast` - Quick test with LLaMA 3 8B using mock data
- `llama32_1b_mock_fast` - Quick test with LLaMA 3.2 1B using mock data

**Usage:**
```bash
./script/experiment_launcher.sh [COMMAND] [OPTIONS]

Commands:
    list                    List available experiments
    run EXPERIMENT_NAME     Run a predefined experiment
    create EXPERIMENT_NAME  Create a new experiment configuration
    validate                Validate all experiment configurations
```

#### `process_data_improved.sh`
Enhanced data processing script:

**Features:**
- Input validation and glob pattern support
- Progress estimation
- Comprehensive error handling
- Output verification

**Usage:**
```bash
./script/process_data_improved.sh [OPTIONS]

Required Options:
    --input PATH                 Input data path (supports glob patterns)
    --output-prefix PREFIX       Output prefix for processed data
    --tokenizer-path PATH        Path to tokenizer model

Optional Options:
    --tokenizer-type TYPE        Tokenizer type (HuggingFaceTokenizer, NullTokenizer)
    --workers N                  Number of worker processes
    --partitions N               Number of partitions
    --append-eod                 Append end-of-document token
    --dry-run                    Show what would be executed without running
```

### Utility Scripts

#### `utils/check_system.sh`
System validation and health check:

**Features:**
- GPU availability and specifications
- Memory and disk space checks
- Dependency validation
- Path and file verification

**Usage:**
```bash
./script/utils/check_system.sh [OPTIONS]

Options:
    --check-gpu                 Check GPU availability and specifications
    --check-memory              Check system memory
    --check-disk                Check disk space
    --check-dependencies        Check required dependencies
    --check-paths               Check required paths and files
    --all                       Run all checks (default)
```

#### `utils/cleanup.sh`
Cleanup utilities for managing disk space:

**Features:**
- Checkpoint cleanup with retention policies
- Log file management
- Cache cleanup
- Size-based and age-based filtering

**Usage:**
```bash
./script/utils/cleanup.sh [COMMAND] [OPTIONS]

Commands:
    checkpoints [OPTIONS]         Clean up checkpoint files
    logs [OPTIONS]                Clean up log files
    cache [OPTIONS]               Clean up cache files
    all [OPTIONS]                 Clean up all artifacts
    list                          List cleanup candidates
```

## Configuration

### Model Configurations

The system supports the following models:

#### LLaMA 3 8B
- **Architecture**: 32 layers, 4096 hidden size, 14336 ffn hidden size
- **Attention**: 32 heads, 8 query groups, 128 kv channels
- **Parallelism**: TP=2, CP=1, PP=4
- **Tokenizer**: model/llama3

#### LLaMA 3.2 1B
- **Architecture**: 16 layers, 2048 hidden size, 8192 ffn hidden size
- **Attention**: 32 heads, 8 query groups, 128 kv channels
- **Parallelism**: TP=4, CP=1, PP=1
- **Tokenizer**: model/llama3.2-1b

### Training Configurations

#### Standard Configuration
- **Batch Size**: 128 global, 1 micro
- **Sequence Length**: 8192
- **Learning Rate**: 0.00015 with cosine decay
- **Training Samples**: 47,340,000
- **Exit Duration**: 235,000,000 minutes

#### Fast Configuration
- **Batch Size**: 32 global, 1 micro
- **Sequence Length**: 2048
- **Learning Rate**: 0.0001 with cosine decay
- **Training Samples**: 1,000
- **Exit Duration**: 60 minutes

### Data Type Configurations

#### FP8
- **Format**: hybrid
- **Amax History Length**: 1024
- **Amax Compute Algorithm**: max
- **Parameter Gather**: enabled

#### BF16
- **Mixed Precision**: enabled
- **Gradient Reduction**: in BF16

## Best Practices

### 1. Always Run System Check First
```bash
./script/utils/check_system.sh
```

### 2. Use Dry Run for Testing
```bash
./script/train_base.sh --dry-run [other options]
```

### 3. Start with Mock Data
```bash
./script/experiment_launcher.sh run llama3_8b_mock_fast
```

### 4. Monitor Disk Space
```bash
./script/utils/cleanup.sh list
```

### 5. Use Experiment Launcher for Common Tasks
```bash
./script/experiment_launcher.sh list
./script/experiment_launcher.sh run [experiment_name]
```

## Troubleshooting

### Common Issues

#### 1. GPU Not Detected
```bash
./script/utils/check_system.sh --check-gpu
```
Ensure NVIDIA drivers and CUDA are properly installed.

#### 2. Insufficient Memory
```bash
./script/utils/check_system.sh --check-memory
```
Consider using smaller batch sizes or model parallelism.

#### 3. Disk Space Issues
```bash
./script/utils/cleanup.sh list
./script/utils/cleanup.sh all --dry-run
```

#### 4. Missing Dependencies
```bash
./script/utils/check_system.sh --check-dependencies
```

### Log Files

All scripts generate detailed log files:
- Training logs: `tensorboard_logs/[model]/[experiment]/training_[experiment]_[timestamp].log`
- Data processing logs: `[output_prefix]_processing_[timestamp].log`

### Error Handling

The scripts include comprehensive error handling:
- Input validation
- Path verification
- Dependency checks
- Graceful failure with informative messages

## Migration from Old Scripts

The new script structure replaces the old individual scripts in:
- `script/llama31-8b/`
- `script/llama32-1b/`

### Migration Steps

1. **Backup old scripts** (optional):
   ```bash
   cp -r script/llama31-8b script/llama31-8b.backup
   cp -r script/llama32-1b script/llama32-1b.backup
   ```

2. **Use new scripts**:
   ```bash
   # Old way
   ./script/llama31-8b/pretrain_llama_wikipedia_fp8.sh
   
   # New way
   ./script/experiment_launcher.sh run llama3_8b_wikipedia_fp8
   ```

3. **Update any automation scripts** to use the new interface.

## Contributing

When adding new experiments or configurations:

1. **Add model configurations** to `config/models.sh`
2. **Add training configurations** to `config/training.sh`
3. **Add experiment definitions** to `experiment_launcher.sh`
4. **Test with dry-run** before committing
5. **Update documentation** as needed

## Support

For issues or questions:
1. Check the log files for detailed error messages
2. Run system checks to identify configuration issues
3. Use dry-run mode to test configurations
4. Review this documentation for usage examples
