# Common Pile CI Dataset

This directory contains tools to create CI test datasets from the
[Common Pile](https://huggingface.co/datasets/common-pile/comma_v0.1_training_dataset)
filtered dataset, replacing the previous datasets sourced from The Pile.

## Output

The scripts produce Megatron-LM indexed binary datasets for three model families:

```
<output_dir>/
├── my-gpt3_00_text_document.{bin,idx}      # GPT  (~24 GB bin)
├── my-bert_00_text_sentence.{bin,idx}       # BERT (~25 GB bin)
├── my-t5_00_text_document.{bin,idx}         # T5   (~25 GB bin)
├── bpe/
│   ├── vocab.json                           # GPT-2 BPE vocabulary
│   └── merges.txt                           # GPT-2 BPE merges
├── vocab.txt                                # BERT WordPiece vocabulary
└── bert-large-cased-vocab.txt               # T5 (BERT-large cased) vocabulary
```

Current production location on the HPC cluster:

```
/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_mcore/mcore_ci/text/common_pile/v01_filtered_data/
```

## Quick Start (HPC)

The fastest way to recreate the dataset on the NVIDIA HPC cluster:

```bash
# 1. Copy both scripts to the remote machine
scp tools/common_pile_dataset/setup_common_pile_dataset.sh tools/common_pile_dataset/create_common_pile_ci_dataset.py \
    <user>@<hpc-host>:/tmp/

# 2. Launch as a background job (survives SSH disconnection)
ssh <user>@<hpc-host> \
    'nohup bash /tmp/setup_common_pile_dataset.sh > /tmp/dataset_creation.log 2>&1 &'

# 3. Monitor progress
ssh <user>@<hpc-host> 'tail -f /tmp/dataset_creation.log'
```

Total runtime is approximately **3-5 hours** (40 min download + ~1 hr per
preprocessing step).

## What the Setup Script Does

`setup_common_pile_dataset.sh` is a self-contained wrapper that:

1. **Finds Python 3.10+** (required by latest Megatron-LM for PEP 604 syntax).
2. **Creates a virtual environment** in `/tmp` to isolate from system packages.
3. **Clones Megatron-LM** (shallow, depth 1) for `preprocess_data.py`.
4. **Patches `megatron/training/__init__.py`** to skip heavy imports (triton,
   apex, transformer-engine) that are not needed for preprocessing.
5. **Installs pip dependencies**: `datasets`, `nltk`, `torch` (CPU-only),
   `transformers`.
6. **Redirects HuggingFace cache** to lustre (`HF_HOME`) to avoid filling up
   the 10 GB `/home` filesystem.
7. **Runs `create_common_pile_ci_dataset.py`** with production parameters.
8. **Cleans up** the temporary work directory.

## Running the Python Script Directly

If you already have a Megatron-LM checkout and the dependencies installed, you
can run the Python script directly:

```bash
# Small test run (streaming, ~10K docs, a few minutes)
python tools/common_pile_dataset/create_common_pile_ci_dataset.py \
    --output-dir /path/to/output \
    --num-documents 10000 \
    --download-vocab

# Full production run (~12M docs, matching existing shard00 sizes)
python tools/common_pile_dataset/create_common_pile_ci_dataset.py \
    --output-dir /path/to/output \
    --num-documents 12000000 \
    --keep-jsonl \
    --copy-vocab-from /lustre/.../text/the_pile
```

### Key Arguments

| Argument | Description |
|---|---|
| `--output-dir` | Where to write the output files (required) |
| `--num-documents` | Number of documents to download (default: 10000) |
| `--megatron-dir` | Path to Megatron-LM repo (default: auto-detect from script location) |
| `--copy-vocab-from` | Copy vocab files from existing `the_pile` directory |
| `--download-vocab` | Download vocab files from HuggingFace instead |
| `--existing-jsonl` | Skip download; use a pre-existing JSONL file |
| `--keep-jsonl` | Keep the intermediate JSONL after preprocessing |
| `--bulk-download` | Non-streaming download (faster but requires ~460 GB HF cache) |
| `--workers` | Number of worker processes for preprocessing (default: 4) |

## Prerequisites

- **Python 3.10+** (Megatron-LM uses PEP 604 `type | None` syntax)
- **PyTorch** (CPU-only is sufficient for preprocessing)
- **Python packages**: `datasets`, `nltk`, `transformers`
- **Disk space**:
  - Output directory: ~80 GB for final files
  - Intermediate JSONL: ~43 GB (deleted unless `--keep-jsonl`)
  - Sentence-split JSONL: ~43 GB (created during BERT preprocessing)
  - HF cache: ~1 GB (streaming mode) or ~460 GB (`--bulk-download` mode)
- **RAM**: ~18 GB peak (during BERT index finalization)

## How It Works

### Step 1: Vocabulary Files

Copies tokenizer vocabularies from the existing `the_pile` dataset, or
downloads them from HuggingFace. These are standard GPT-2 BPE and BERT
WordPiece vocabularies.

### Step 2: Download Raw Text

Streams documents from `common-pile/comma_v0.1_training_dataset` on
HuggingFace, filtering out documents shorter than 100 characters. Writes a
JSONL file with `{"text": "..."}` per line. At 12M documents this produces a
~43 GB file.

Progress is logged with ETA:

```
[  42.0%] 5,040,000/12,000,000 docs | 4,500 docs/s | 20,150.3 MB on disk | ETA: 25.8m
```

### Step 3: GPT Preprocessing

Runs `preprocess_data.py` with `GPT2BPETokenizer` and `--append-eod` to create
`my-gpt3_00_text_document.{bin,idx}`.

### Step 4: BERT Preprocessing

BERT requires sentence splitting, which is a **two-pass process** when using
`partitions=1` (the default):

1. **Pass 1**: Runs with `--split-sentences` to create a sentence-split JSONL
   (`common_pile_raw_ss.jsonl`), then returns.
2. **Pass 2**: Detects the `_ss.jsonl` file exists, skips splitting, and
   encodes to binary `my-bert_00_text_sentence.{bin,idx}`.

The BERT `.idx` file is much larger (~4 GB vs ~229 MB for GPT/T5) because it
indexes individual sentences rather than whole documents.

### Step 5: T5 Preprocessing

Runs `preprocess_data.py` with `BertWordPieceCase` tokenizer and `--append-eod`
to create `my-t5_00_text_document.{bin,idx}`.

### Step 6: Verification

Checks that all 10 expected output files exist and reports their sizes.

## Troubleshooting

### "No space left on device" during download

The HuggingFace `datasets` library caches data under `~/.cache/huggingface/` by
default. On HPC systems where `/home` is small, set `HF_HOME` to a path with
sufficient space:

```bash
export HF_HOME=/lustre/path/to/.hf_cache
```

The setup script does this automatically.

### "ModuleNotFoundError: No module named 'triton'" (or similar)

`preprocess_data.py` imports from `megatron.training`, which eagerly loads the
full training stack. The setup script patches `megatron/training/__init__.py` to
comment out the heavy imports. If running manually, apply:

```bash
sed -i 's/^from \.initialize/#from .initialize/' megatron/training/__init__.py
```

### "TypeError: unsupported operand type(s) for |: 'type' and 'NoneType'"

You need Python 3.10+. The latest Megatron-LM uses PEP 604 union syntax
(`type | None`) which is not supported in Python 3.9.

### numpy/scipy binary incompatibility

Use a virtual environment (`python3.10 -m venv venv`) to isolate from system
packages. The setup script creates one automatically.

### BERT produces no output files

BERT with `--split-sentences` and `partitions=1` requires two invocations of
`preprocess_data.py` (see Step 4 above). The script handles this automatically.

## CI Data Path Configuration

To use this dataset in CI tests, set the following paths in `model_config.yaml`:

```yaml
# GPT
--data-path: ${DATA_PATH}/text/common_pile/v01_filtered_data/my-gpt3_00_text_document
--vocab-file: ${DATA_PATH}/text/common_pile/v01_filtered_data/bpe/vocab.json
--merge-file: ${DATA_PATH}/text/common_pile/v01_filtered_data/bpe/merges.txt

# BERT
--data-path: ${DATA_PATH}/text/common_pile/v01_filtered_data/my-bert_00_text_sentence
--vocab-file: ${DATA_PATH}/text/common_pile/v01_filtered_data/vocab.txt

# T5
--data-path: ${DATA_PATH}/text/common_pile/v01_filtered_data/my-t5_00_text_document
--vocab-file: ${DATA_PATH}/text/common_pile/v01_filtered_data/bert-large-cased-vocab.txt
```

## Files

| File | Description |
|---|---|
| `create_common_pile_ci_dataset.py` | Main Python script that downloads data and runs preprocessing |
| `setup_common_pile_dataset.sh` | Self-contained bash wrapper for HPC deployment |
| `README.md` | This file |
