<!---
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# Data Preparation

Preparing your data correctly is essential for successful training with Megatron Core.

## Data Format

Megatron Core expects training data in JSONL (JSON Lines) format, where each line is a JSON object:

```json
{"text": "Your training text here..."}
{"text": "Another training sample..."}
{"text": "More training data..."}
```

## Preprocessing Data

Use the `preprocess_data.py` tool to convert your JSONL data into Megatron's binary format:

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

| Argument | Description |
|----------|-------------|
| `--input` | Path to input JSON/JSONL file |
| `--output-prefix` | Prefix for output binary files (.bin and .idx) |
| `--tokenizer-type` | Tokenizer type (`HuggingFaceTokenizer`, `GPT2BPETokenizer`, etc.) |
| `--tokenizer-model` | Path to tokenizer model file |
| `--workers` | Number of parallel workers for processing |
| `--append-eod` | Add end-of-document token |

## Output Files

The preprocessing tool generates two files:
- `processed_data.bin` - Binary file containing tokenized sequences
- `processed_data.idx` - Index file for fast random access

## Using Preprocessed Data

Reference your preprocessed data in training scripts:

```bash
--data-path processed_data \
--split 949,50,1  # Train/validation/test split
```

## Common Tokenizers

### HuggingFace Tokenizers

```bash
--tokenizer-type HuggingFaceTokenizer \
--tokenizer-model /path/to/tokenizer.model
```

### GPT-2 BPE Tokenizer

```bash
--tokenizer-type GPT2BPETokenizer \
--vocab-file gpt2-vocab.json \
--merge-file gpt2-merges.txt
```
