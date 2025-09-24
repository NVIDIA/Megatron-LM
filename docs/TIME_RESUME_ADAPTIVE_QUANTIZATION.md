# Time-Resume Adaptive Quantization Training

## Overview

Time-resume adaptive quantization is an advanced training technique that dynamically switches between quantized (fp8/fp4) and high-precision (bf16) training based on loss thresholds. This approach combines the efficiency of quantized training with the stability of high-precision training.

## Features

- **Dynamic Precision Switching**: Automatically switches between quantized and bf16 training based on loss
- **Asynchronous Checkpoint Saving**: Non-blocking checkpoint saves to minimize training interruption
- **Window-based Training**: Organizes training into manageable windows with regular checkpoints
- **Loss-based Thresholds**: Configurable loss thresholds for precision switching
- **Recovery System**: Automatic recovery from the best available checkpoint

## Command Line Arguments

### Core Parameters

- `--time-resume`: Enable time-resume adaptive quantization training
- `--quant-loss-threshold`: Loss threshold for switching from quantized to BF16 training (default: 0.1)
- `--quant-window-size`: Number of iterations per training window (default: 5)
- `--quant-checkpoint-interval`: Checkpoint save interval within windows (default: 1)
- `--quant-fallback-strategy`: Fallback precision when quantized training fails (choices: bf16, fp16, default: bf16)
- `--quant-recovery-buffer`: Number of checkpoints to keep for recovery (default: 2)

### Additional Parameters

- `--scaling-control`: Scaling control strategy for MX quantization (choices: max, max_minus_1, default: max)

## Usage Examples

### Basic Usage

```bash
bash script/llama32-1b/pretrain_llama32-1b_wikipedia_FA_linear_mxfp4_time_resume.sh \
    "checkpoints/path" \
    "logs/path" \
    "tokenizer/path" \
    "data/path" \
    "bf16" \
    "max_minus_1" \
    "0.1" \
    "5" \
    "1" \
    "bf16" \
    "2"
```

### Custom Configuration

```bash
bash examples/llama/train_llama32_1b_h100_fp8.sh \
    --time-resume \
    --quant-loss-threshold 0.15 \
    --quant-window-size 10 \
    --quant-checkpoint-interval 2 \
    --quant-fallback-strategy bf16 \
    --quant-recovery-buffer 3 \
    --scaling-control max_minus_1
```

## How It Works

### 1. Initialization

The adaptive quantization manager is initialized at the start of training with the specified parameters.

### 2. Training Loop

During training, the system:
- Monitors loss values
- Compares current loss against the threshold
- Switches precision when needed
- Saves checkpoints asynchronously
- Manages training windows

### 3. Precision Switching Logic

- **Switch to BF16**: When loss exceeds threshold for 3 consecutive iterations
- **Switch back to Quantized**: When loss is stable and below 80% of threshold
- **Checkpoint before switch**: Always saves checkpoint before precision changes

### 4. Window Management

- Training is organized into windows of specified size
- Checkpoints are saved at window boundaries
- Recovery system maintains multiple checkpoints for rollback

## Configuration Guidelines

### Loss Threshold

- **Too High**: May not catch precision issues early enough
- **Too Low**: May cause unnecessary switching to BF16
- **Recommended**: 0.1-0.2 for most models

### Window Size

- **Small Windows**: More frequent checkpoints, higher overhead
- **Large Windows**: Less overhead, but longer recovery time
- **Recommended**: 5-10 iterations for most cases

### Checkpoint Interval

- **Frequent Saves**: Better recovery options, higher I/O overhead
- **Infrequent Saves**: Lower overhead, but limited recovery options
- **Recommended**: 1-2 iterations for critical training

## Monitoring and Debugging

### Log Messages

The system provides detailed logging:

```
[TimeResume] Adaptive quantization training enabled
[AdaptiveQuantization] Loss 0.1500 exceeds threshold 0.1000, switching to bf16
[AdaptiveQuantization] Checkpoint saved: window_1_iter100_bf16
[AdaptiveQuantization] Loss 0.0800 is stable, switching back to mxfp4
```

### Checkpoint Naming

Checkpoints are named with descriptive tags:
- `window_1_iter100_quantized`: Window 1, iteration 100, quantized training
- `switch_to_bf16_iter150`: Switch checkpoint at iteration 150 to bf16
- `window_end_2_iter200`: End of window 2 at iteration 200

## Best Practices

### 1. Initial Setup

- Start with conservative thresholds (0.1-0.15)
- Use moderate window sizes (5-10 iterations)
- Enable frequent checkpointing during early training

### 2. Monitoring

- Watch for frequent precision switching
- Monitor checkpoint storage usage
- Track training stability metrics

### 3. Optimization

- Adjust thresholds based on model behavior
- Increase window size for stable models
- Reduce checkpoint frequency for storage-constrained environments

## Troubleshooting

### Common Issues

1. **Frequent Switching**: Lower the loss threshold or increase window size
2. **Storage Issues**: Reduce checkpoint frequency or buffer size
3. **Training Instability**: Increase recovery buffer size

### Recovery

If training fails, the system can automatically recover from the most recent checkpoint:

```python
# Automatic recovery is handled by the manager
adaptive_quantization_manager.load_recovery_checkpoint()
```

## Performance Considerations

### Benefits

- **Automatic Optimization**: Reduces manual tuning of quantization parameters
- **Fault Tolerance**: Better recovery from training issues
- **Efficiency**: Combines benefits of both quantized and high-precision training

### Overhead

- **Checkpoint I/O**: Asynchronous saves minimize impact
- **Memory Usage**: Multiple checkpoints require additional storage
- **Complexity**: Slightly more complex training loop

## Integration with Existing Features

Time-resume adaptive quantization works seamlessly with:
- Tensor saving and collection
- Scaling control strategies
- Existing checkpoint systems
- Multi-GPU training setups

## Future Enhancements

Potential improvements include:
- Machine learning-based threshold adjustment
- More sophisticated precision selection
- Integration with model compression techniques
- Advanced recovery strategies
