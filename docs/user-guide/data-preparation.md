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

The following table summarizes the main preprocessor arguments:

| Argument | Description |
|----------|-------------|
| `--input` | Path to input JSON/JSONL file |
| `--output-prefix` | Prefix for output binary files (.bin and .idx) |
| `--tokenizer-type` | Tokenizer type (`HuggingFaceTokenizer`, `GPT2BPETokenizer`, and so on) |
| `--tokenizer-model` | Path to tokenizer model file |
| `--workers` | Number of parallel workers for processing |
| `--append-eod` | Add end-of-document token |

## Finding Optimal Number of Workers

Use the `--find-optimal-num-workers` flag to find the number of workers that gives the best performance in terms of preprocessed documents per second.
The script launches a few short data preprocessing runs with different worker counts and identifies the fastest run using the collected performance data.

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

The following table lists the arguments required for worker optimization:

| Argument | Description |
|----------|-------------|
| `--find-optimal-num-workers` | Activates search of optimal number of workers |
| `--workers-to-check` | List of possible number of workers to run |
| `--max-documents` | Number of documents to be preprocessed during each run |

**Output example**

The command prints performance results similar to the following:

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

## Fast Preprocessing with GigaToken

For large corpora, `tools/preprocess_data_fast.py` offers a faster alternative to
`preprocess_data.py`. Instead of splitting documents across `--workers` processes and adding
them one at a time, it tokenizes whole JSONL files at once with
[GigaToken](https://pypi.org/project/gigatoken/) and writes the result in a single batched
call via `IndexedDatasetBuilder.add_documents()`.

Requires the `gigatoken` package (`pip install gigatoken`).

```bash
python tools/preprocess_data_fast.py \
    --input data.jsonl \
    --output-prefix processed_data \
    --json-keys text \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model /path/to/tokenizer.model \
    --use-gigatoken \
    --append-eod
```

**Key Differences** from `preprocess_data.py`

| | `preprocess_data.py` | `preprocess_data_fast.py` |
|---|---|---|
| Tokenization | Per-document, split across `--workers` processes | Whole-file, parallelized internally by `gigatoken` |
| Dataset writing | `add_document()` per document | Batched `add_documents()` for the whole file |
| Requires | — | `gigatoken` package, `--use-gigatoken` |

**Key Arguments**

| Argument | Description |
|----------|-------------|
| `--input` | Path to input **JSONL** file |
| `--output-prefix` | Prefix for output binary files (`.bin` and `.idx`) |
| `--json-keys` | Space-separated list of **JSON** fields to tokenize; each gets its own process and output shard |
| `--tokenizer-type` / `--tokenizer-model` | Same as `preprocess_data.py` |
| `--use-gigatoken` | Enable GigaToken-accelerated tokenization (required) |
| `--append-eod` | Append an end-of-document token to each document |

**Performance**

Benchmarked with the `Qwen/Qwen3-8B` HuggingFace tokenizer, preprocessing whole **JSONL** files:

| Input size | `preprocess_data.py` | `preprocess_data_fast.py` | Speedup |
|---|---|---|---|
| 10M lines | 23 min (~7,246 docs/s) | 2.7 min (~61,728 docs/s) | ~8.5x |
| 80M lines | 214.3 min (~6,222 docs/s) | 25.3 min (~52,701 docs/s) | ~8.5x |

`preprocess_data_fast.py` was consistently **~8.5x faster** across both input sizes, driven by
whole-file GigaToken tokenization and `IndexedDatasetBuilder.add_documents()`.

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
