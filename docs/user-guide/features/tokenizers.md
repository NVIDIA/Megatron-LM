<!---
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# Tokenizers

Megatron Core provides a unified tokenizer system with a Hugging Face-style API for configuration and loading.

## Overview

The `MegatronTokenizer` class uses the same entry points as many Hugging Face workflows for loading and managing tokenizers:

- **Automatic detection** - Load tokenizer types without naming the backing library in code
- **Metadata-based configuration** - Store tokenizer settings in JSON for reuse across runs
- **Hugging Face-compatible API** - `.from_pretrained()`-style loading
- **Custom tokenizer support** - Extend with model-specific tokenization logic

## Key Features

### Unified API

Use the same API regardless of tokenizer backend (SentencePiece, Hugging Face, TikToken, and so on):

```python
from megatron.core.tokenizers import MegatronTokenizer

tokenizer = MegatronTokenizer.from_pretrained("/path/to/tokenizer")
```

### Tokenizer Metadata

Configuration is stored in a JSON metadata file containing:

- Tokenizer library (Hugging Face, SentencePiece, TikToken, and so on)
- Chat templates
- Custom tokenizer class
- Special token configurations

**Benefits**

- Set configuration once, reuse everywhere
- No repeated CLI arguments
- Share setups by copying the tokenizer directory

### Automatic Library Detection

The correct tokenizer implementation is selected automatically:

- Avoids hard-coding `SentencePieceTokenizer`, `HuggingFaceTokenizer`, and related class names in user code
- Library type is read from metadata
- Change tokenizer backends by updating metadata and paths

## Basic Usage

### Creating Tokenizer Metadata

Save tokenizer configuration for reuse:

```python
from megatron.core.tokenizers import MegatronTokenizer

# Create metadata for a SentencePiece tokenizer
MegatronTokenizer.write_metadata(
    tokenizer_path="/path/to/tokenizer.model",
    tokenizer_library="sentencepiece",
    chat_template="{% for message in messages %}{{ message.content }}{% endfor %}",
)
```

The metadata is saved as `tokenizer_metadata.json` in the tokenizer directory.

### Loading a Tokenizer

Load from a directory with metadata:

```python
from megatron.core.tokenizers import MegatronTokenizer

# Load with auto-detected configuration
tokenizer = MegatronTokenizer.from_pretrained("/path/to/tokenizer.model")
```

### Loading with Custom Metadata Path

If metadata is stored separately:

```python
tokenizer = MegatronTokenizer.from_pretrained(
    tokenizer_path="/path/to/tokenizer.model",
    metadata_path="/path/to/custom/metadata.json",
)
```

### Loading with Inline Metadata

Pass metadata as a dictionary:

```python
tokenizer = MegatronTokenizer.from_pretrained(
    tokenizer_path="GPT2BPETokenizer",
    metadata_path={"library": "megatron"},
    vocab_file="/path/to/vocab.txt",
)
```

## Advanced Usage

### Custom Tokenizer Classes

Create model-specific tokenization logic:

```python
from megatron.core.tokenizers.text import MegatronTokenizerText

class CustomTokenizer(MegatronTokenizerText):
    def encode(self, text):
        # Custom encoding logic
        return super().encode(text)

    def decode(self, tokens):
        # Custom decoding logic
        return super().decode(tokens)

# Save metadata with custom class
MegatronTokenizer.write_metadata(
    tokenizer_path="/path/to/tokenizer.model",
    tokenizer_library="sentencepiece",
    tokenizer_class=CustomTokenizer,
)
```

### TikToken Tokenizers

Configure TikToken-based tokenizers:

```python
tokenizer = MegatronTokenizer.from_pretrained(
    tokenizer_path="/path/to/tokenizer/model.json",
    metadata_path={"library": "tiktoken"},
    pattern="v2",
    num_special_tokens=1000,
)
```

### Null Tokenizer

The Null tokenizer is a lightweight, zero-I/O tokenizer that requires no model files.
It is useful in three scenarios:

1. **Performance benchmarking** with `--mock-data` where real tokenization is unnecessary.
2. **Testing** in functional tests and CI pipelines where tokenizer model files may not
   be available. The Null tokenizer removes the dependency on external files, making
   tests self-contained and portable.
3. **Pretraining with pretokenized data** where all data is already tokenized into
   `.bin`/`.idx` files. In this case the tokenizer is only needed for metadata
   (`vocab_size`, `eod`, `pad`) — not for actual tokenization. Using the Null tokenizer
   avoids redundant filesystem access at scale, which is particularly beneficial on
   shared filesystems like Lustre where thousands of ranks would otherwise all load the
   same tokenizer files.

Properties derived from `--vocab-size N`:
- `vocab_size` = `N` (the exact value passed)
- `eod` = `N - 1` (last token in the vocabulary)
- `pad` = `0`

```python
tokenizer = MegatronTokenizer.from_pretrained(
    metadata_path={"library": "null-text"},
    vocab_size=131072,
)
```

## Integration with Megatron-LM

### Using with Training Scripts

The tokenizer system works with Megatron-LM training scripts:

```bash
# Null tokenizer for benchmarking with mock data
torchrun --nproc_per_node=8 pretrain_gpt.py \
    --tokenizer-type NullTokenizer \
    --vocab-size 131072 \
    --mock-data \
    ...
```

```bash
# Null tokenizer for pretraining with pretokenized data (no tokenizer files needed)
torchrun --nproc_per_node=8 pretrain_gpt.py \
    --tokenizer-type NullTokenizer \
    --vocab-size 128256 \
    --data-path /path/to/pretokenized_data \
    ...
```

```bash
# Hugging Face tokenizer with metadata
torchrun --nproc_per_node=8 pretrain_gpt.py \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model meta-llama/Meta-Llama-3-8B \
    --tokenizer-metadata /path/to/metadata.json \
    ...
```

### Auto-Generated Metadata

If `--tokenizer-metadata` is not specified, a default metadata file is generated automatically based on the tokenizer type.

## Supported Tokenizer Libraries

The following table lists supported tokenizer backends:

| Library | Description | Use Case |
|---------|-------------|----------|
| **Hugging Face** | Transformers tokenizers | Most modern LLMs, such as LLaMA and Mistral |
| **SentencePiece** | Google's tokenizer | GPT-style models, custom vocabularies |
| **TikToken** | OpenAI's tokenizer | GPT-3.5/GPT-4 style tokenization |
| **Megatron** | Built-in tokenizers | Legacy GPT-2 BPE |
| **Null** | Zero-I/O tokenizer | Benchmarking, pretokenized data |

## Common Tokenizer Types

### LLaMA / Mistral

```python
MegatronTokenizer.write_metadata(
    tokenizer_path="/path/to/llama/tokenizer.model",
    tokenizer_library="sentencepiece",
)
```

### GPT-2

```python
MegatronTokenizer.write_metadata(
    tokenizer_path="GPT2BPETokenizer",
    tokenizer_library="megatron",
    vocab_file="/path/to/gpt2-vocab.json",
    merge_file="/path/to/gpt2-merges.txt",
)
```

## Recommendations

1. **Save metadata** - Create metadata once, then reuse across training runs
2. **Prefer Hugging Face tokenizers** - When the model ships one, it reduces integration work
3. **Test tokenization** - Verify encode and decode before long training jobs
4. **Version control metadata** - Track `tokenizer_metadata.json` with experiment configs
5. **Share tokenizer directories** - Ship model files and metadata together for reproducibility

## Next Steps

- **Prepare data**: Refer to [Data Preparation](../data-preparation.md) for preprocessing with tokenizers
- **Train models**: Refer to [Training Examples](../training-examples.md)
- **Supported models**: Refer to [Language Models](../../models/llms.md) for model-specific tokenizers
