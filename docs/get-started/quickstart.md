<!---
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# Quick Start

## Quick Installation

Install Megatron Core with pip:

1. Install Megatron Core with required dependencies:

    ```bash
    pip install --no-build-isolation megatron-core[mlm,dev]
    ```

2. Clone repository for examples:

    ```bash
    git clone https://github.com/NVIDIA/Megatron-LM.git
    cd Megatron-LM
    pip install --no-build-isolation .[mlm,dev]
    ```

That's it! You're ready to start training.

## Your First Training Run

### Simple Training Example

```bash
# Distributed training example (2 GPUs, mock data)
torchrun --nproc_per_node=2 examples/run_simple_mcore_train_loop.py
```

### LLaMA-3 Training Example

```bash
# 8 GPUs, FP8 precision, mock data
./examples/llama/train_llama3_8b_fp8.sh
```

## Data Preparation

### JSONL Data Format

```json
{"text": "Your training text here..."}
{"text": "Another training sample..."}
```

### Basic Preprocessing

```bash
python tools/preprocess_data.py \
    --input data.jsonl \
    --output-prefix processed_data \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model /path/to/tokenizer.model \
    --workers 8 \
    --append-eod
```

### Key Arguments

- `--input`: Path to input JSON/JSONL file
- `--output-prefix`: Prefix for output binary files (.bin and .idx)
- `--tokenizer-type`: Tokenizer type (`HuggingFaceTokenizer`, `GPT2BPETokenizer`, etc.)
- `--tokenizer-model`: Path to tokenizer model file
- `--workers`: Number of parallel workers for processing
- `--append-eod`: Add end-of-document token

## Next Steps

- Explore [Parallelism Strategies](../user-guide/parallelism-guide.md) to scale your training
- Learn about [Data Preparation](../user-guide/data-preparation.md) best practices
- Check out [Advanced Features](../user-guide/features/index.md) for advanced capabilities
