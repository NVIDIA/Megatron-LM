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

## Finding Optimal Number of Workers

Use the `--find-optimal-num-workers` flag to find number of workers which gives the best performance in terms of preprocessed documents per second.
Script will lauch a few short data preprocessing runs with a different number of workers to define the fastest run in respect to collected performance data.

```bash
python tools/preprocess_data.py \
    --input data.jsonl \
    --output-prefix processed_data \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model /path/to/tokenizer.model \
    --workers 8 \
    --find-optimal-num-workers \
    --workers-to-check 4 8 16 32 \
    --max-documents 50000
```

**Required arguments**

| Argument | Description |
|----------|-------------|
| `--find-optimal-num-workers` | Activates search of optimal number of workers |
| `--workers-to-check` | List of possible number of workers to run |
| `--max-documents` | Number of documents to be preprocessed during each run |

**Output example**

```bash
-----------------------------------
Performance results (fastest → slowest):
1. 16 workers → avg. docs/s: 9606.6476
2. 32 workers → avg. docs/s: 9275.3284
3. 8 workers → avg. docs/s: 9151.9280
4. 4 workers → avg. docs/s: 6391.3819

-----------------------------------
The most optimal num of workers is 16 with avg. preprocessed docs/s: 9606.6476.
-----------------------------------
```

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
