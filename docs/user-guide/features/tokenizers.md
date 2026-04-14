<!---
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# Tokenizers

Megatron Core provides a unified tokenizer system with a HuggingFace-style API for easy tokenizer management and configuration.

## Overview

The `MegatronTokenizer` class offers a simple, familiar API for loading and managing tokenizers:

- **Automatic detection** - Load any tokenizer type without specifying the library
- **Metadata-based configuration** - Store tokenizer settings in JSON for easy reuse
- **HuggingFace-compatible API** - Familiar `.from_pretrained()` interface
- **Custom tokenizer support** - Extend with model-specific tokenization logic

## Key Features

### Unified API

Use the same API regardless of tokenizer backend (SentencePiece, HuggingFace, TikToken, etc.):

```python
from megatron.core.tokenizers import MegatronTokenizer

tokenizer = MegatronTokenizer.from_pretrained("/path/to/tokenizer")
```

### Tokenizer Metadata

Configuration is stored in a JSON metadata file containing:
- Tokenizer library (HuggingFace, SentencePiece, TikToken, etc.)
- Chat templates
- Custom tokenizer class
- Special token configurations

**Benefits:**
- Set configuration once, reuse everywhere
- No repeated CLI arguments
- Easy sharing - just copy the tokenizer directory

### Automatic Library Detection

The correct tokenizer implementation is automatically selected:
- No need to specify `SentencePieceTokenizer`, `HuggingFaceTokenizer`, etc.
- Library type detected from metadata
- Seamless switching between tokenizer backends

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

Use a null tokenizer for testing or non-text models:

```python
tokenizer = MegatronTokenizer.from_pretrained(
    metadata_path={"library": "null-text"},
    vocab_size=131072,
)
```

## Integration with Megatron-LM

### Using with Training Scripts

The tokenizer system integrates seamlessly with Megatron-LM training:

```bash
# Null tokenizer for testing
torchrun --nproc_per_node=8 pretrain_gpt.py \
    --tokenizer-type NullTokenizer \
    --vocab-size 131072 \
    ...
```

```bash
# HuggingFace tokenizer with metadata
torchrun --nproc_per_node=8 pretrain_gpt.py \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model meta-llama/Meta-Llama-3-8B \
    --tokenizer-metadata /path/to/metadata.json \
    ...
```

### Auto-Generated Metadata

If `--tokenizer-metadata` is not specified, a default metadata file is generated automatically based on the tokenizer type.

## Supported Tokenizer Libraries

| Library | Description | Use Case |
|---------|-------------|----------|
| **HuggingFace** | Transformers tokenizers | Most modern LLMs (LLaMA, Mistral, etc.) |
| **SentencePiece** | Google's tokenizer | GPT-style models, custom vocabularies |
| **TikToken** | OpenAI's tokenizer | GPT-3.5/GPT-4 style tokenization |
| **Megatron** | Built-in tokenizers | Legacy GPT-2 BPE |
| **Null** | No-op tokenizer | Testing, non-text modalities |

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

## Best Practices

1. **Always save metadata** - Create metadata once, reuse across training runs
2. **Use HuggingFace tokenizers** - When possible, for modern LLM compatibility
3. **Test tokenization** - Verify encode/decode before starting training
4. **Version control metadata** - Include `tokenizer_metadata.json` in your experiment configs
5. **Share tokenizer directories** - Include both model files and metadata for reproducibility

## Next Steps

- **Prepare Data**: See [Data Preparation](../data-preparation.md) for preprocessing with tokenizers
- **Train Models**: Use tokenizers in [Training Examples](../training-examples.md)
- **Supported Models**: Check [Language Models](../../models/llms.md) for model-specific tokenizers
