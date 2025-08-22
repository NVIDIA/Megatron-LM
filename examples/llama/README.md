# Llama Models

## Table of contents
- [1. Overview](#1-overview)
- [2. Prerequisites](#2-prerequisites)
- [3. Training Setup](#3-training-setup)
- [4. Configuration](#4-configuration)
- [5. Test Datasets](#5-test-datasets)
- [6. FP8 Debugging](#6-fp8-debugging)

## 1. Overview
<a id="overview" name="overview"></a>

Train Llama models using FP8 precision with Megatron-Core.

## 2. Prerequisites
<a id="prerequisites" name="prerequisites"></a>

```bash
# Clone repository
export HOST_MEGATRON_LM_DIR="/path/to/your/host/megatron-lm"
git clone https://github.com/NVIDIA/Megatron-LM.git "$HOST_MEGATRON_LM_DIR"
cd "$HOST_MEGATRON_LM_DIR"
git checkout "core_r0.12.0"

# Set paths
export HOST_CHECKPOINT_PATH="./checkpoints/llama3_8b_fp8"
export HOST_TENSORBOARD_LOGS_PATH="./tensorboard_logs/llama3_8b_fp8"

# Optional: For real data
# export HOST_TOKENIZER_MODEL_PATH="/path/to/host/tokenizer.model"
# export HOST_DATA_PREFIX="/path/to/host/mydata_prefix"
```

## 3. Training Setup
<a id="training-setup" name="training-setup"></a>

### Using Mock Data
```bash
PYTORCH_IMAGE="nvcr.io/nvidia/pytorch:25.03-py3"

docker run --rm --gpus all --ipc=host --ulimit memlock=-1 \
  -v "${HOST_MEGATRON_LM_DIR}:/workspace/megatron-lm" \
  -v "${HOST_CHECKPOINT_PATH}:/workspace/checkpoints" \
  -v "${HOST_TENSORBOARD_LOGS_PATH}:/workspace/tensorboard_logs" \
  --workdir /workspace/megatron-lm \
  $PYTORCH_IMAGE \
  bash examples/llama/train_llama3_8b_h100_fp8.sh \
    /workspace/checkpoints \
    /workspace/tensorboard_logs \
  2>&1 | tee "${HOST_TENSORBOARD_LOGS_PATH}/training_mock_$(date +'%y-%m-%d_%H-%M-%S').log"
```

### Using Custom Data and Tokenizer
```bash
PYTORCH_IMAGE="nvcr.io/nvidia/pytorch:25.03-py3"

docker run --rm --gpus all --ipc=host --ulimit memlock=-1 \
  -v "${HOST_MEGATRON_LM_DIR}:/workspace/megatron-lm" \
  -v "${HOST_CHECKPOINT_PATH}:/workspace/checkpoints" \
  -v "${HOST_TENSORBOARD_LOGS_PATH}:/workspace/tensorboard_logs" \
  -v "${HOST_TOKENIZER_MODEL_PATH}:/workspace/tokenizer_model" \
  -v "$(dirname "${HOST_DATA_PREFIX}"):/workspace/data_dir" \
  --workdir /workspace/megatron-lm \
  $PYTORCH_IMAGE \
  bash examples/llama/train_llama3_8b_h100_fp8.sh \
    /workspace/checkpoints \
    /workspace/tensorboard_logs \
    /workspace/tokenizer_model \
    "/workspace/data_dir/$(basename "${HOST_DATA_PREFIX}")" \
  2>&1 | tee "${HOST_TENSORBOARD_LOGS_PATH}/training_custom_$(date +'%y-%m-%d_%H-%M-%S').log"
```

## 4. Configuration
<a id="configuration" name="configuration"></a>

Default parallelism strategy:
- Tensor Parallel: 1
- Pipeline Parallel: 1
- Context Parallel: 2

Llama-3-8B architecture:
- 32 layers
- Hidden size: 4096
- FFN hidden size: 14336
- Attention heads: 32
- Query groups: 8
- Sequence length: 8192
- RMSNorm normalization with SwiGLU and RoPE

Key training parameters:
- Micro-batch size: 1
- Global batch size: 128
- Learning rate: 1.5e-4
- Min learning rate: 1.0e-5
- Weight decay: 0.1
- FP8 format: hybrid

You can modify these parameters directly in the `train_llama3_8b_h100_fp8.sh` script.

This configuration follows those defined in NeMo Framework's performance scripts, which can be found at [https://github.com/NVIDIA/NeMo/tree/main/scripts/performance](https://github.com/NVIDIA/NeMo/tree/main/scripts/performance). 

### FP8 Performance

| Model | #-GPUs | GBS | MBS | Seq Length | TP | PP | CP | VP | EP | GA | Tokens/sec/GPU | TFLOP/sec/GPU |
|-------|--------|-----|-----|------------|----|----|----|----|----|----|----------------|---------------|
| LLAMA3-8B | 8 | 128 | 1 | 8192 | 1 | 1 | 2 | 1 | 1 | 32 | 13812 | 800 |
| LLAMA3-70B | 64 | 128 | 1 | 8192 | 4 | 8 | 1 | 5 | 1 | 64 | 1621 | 780 |
| LLAMA3-405B | 1024 | 512 | 1 | 8192 | 8 | 8 | 2 | 8 | 1 | 64 | 315 | 834 |

Legend:
- GBS: Global Batch Size
- MBS: Micro Batch Size
- TP: Tensor Parallel size
- PP: Pipeline Parallel size
- CP: Context Parallel size
- VP: Virtual Pipeline stages
- EP: Expert Parallel size
- GA: Gradient Accumulation steps

As NeMo uses Megatron-Core, for the latest performance benchmarks, please refer to the official [NeMo documentation](https://docs.nvidia.com/nemo-framework/user-guide/latest/performance/performance_summary.html).

## 5. Test Datasets
<a id="test-datasets" name="test-datasets"></a>

Recommended datasets:
1. **WikiText-103**: https://huggingface.co/datasets/Salesforce/wikitext

Preprocess datasets:
```bash
python "${HOST_MEGATRON_LM_DIR}/tools/preprocess_data.py" \
       --input your_dataset.json \
       --output-prefix test_dataset \
       --tokenizer-type HuggingFaceTokenizer \
       --tokenizer-model /path/to/tokenizer.model \
       --append-eod
```

## 6. FP8 Training Considerations
<a id="fp8-training-considerations" name="fp8-training-considerations"></a>

- **Hardware**: Requires NVIDIA Hopper, Ada, or Blackwell GPUs for FP8 support
   
- **Troubleshooting**: If you encounter NaN values or instability with FP8 training, please refer to [Transformer Engine](https://github.com/NVIDIA/TransformerEngine).
