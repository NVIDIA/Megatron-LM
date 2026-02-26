<!---
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# Your First Training Run

This guide walks you through running your first training jobs with Megatron Core. Make sure you have completed [installation](install.md) before proceeding.

## Simple Training Example

Run a minimal distributed training loop with mock data on 2 GPUs:

```bash
torchrun --nproc_per_node=2 examples/run_simple_mcore_train_loop.py
```

## LLaMA-3 Training Example

Train a LLaMA-3 8B model with FP8 precision on 8 GPUs using mock data:

```bash
./examples/llama/train_llama3_8b_fp8.sh
```

## Data Preparation

To train on your own data, Megatron expects preprocessed binary files (`.bin` and `.idx`).

### 1. Prepare a JSONL File

Each line should contain a `text` field:

```json
{"text": "Your training text here..."}
{"text": "Another training sample..."}
```

### 2. Preprocess the Data

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
