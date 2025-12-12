# New Tokenizer System

## Key Differences from the Old Tokenizer System

### 1. Hugging Face–style API

We now have a `MegatronTokenizer` class that provides a familiar, simple API similar to Hugging Face’s:

`.from_pretrained()` – Load a tokenizer from a directory or file, automatically detecting the type and settings.

`.write_metadata()` – Save tokenizer configuration (metadata) so that it can be reused without re-specifying parameters.

This eliminates the need for long initialization arguments and hard-coded settings in training scripts.

### 2. Tokenizer Metadata

A metadata file (JSON) now stores all essential tokenizer configuration in one place:
 - Tokenizer library (e.g., HuggingFace, SentencePiece, TikToken, etc.)
 - Chat templates
 - Tokenizer class

Benefits:
 - You only need to set these parameters once.
 - No more passing multiple CLI arguments for tokenizer settings.
 - Easy sharing — just copy the tokenizer directory with its metadata file.

### 3. Library Classes Are Now Internal

In the old system, you had to know which tokenizer library to use (`SentencePieceTokenizer`, `HuggingFaceTokenizer`, etc.) and instantiate it manually.

In the new system:
 - The library is automatically detected from the metadata.
 - The correct tokenizer implementation is chosen under the hood.
 - Users don’t need to manually manage tokenizer classes.

### 3. Support for Model-specific Tokenizer Classes

The system now supports:
 - Built-in LLM-specific tokenizers. 
 - Custom tokenizers: You can create your own tokenizer class by inheriting from `MegatronTokenizerText` and specify it in the `tokenizer_class` field in the metadata file.
 - This allows advanced customization while keeping defaults simple for most users.

### 4. Usage

**Creating and Saving Metadata**

```python
from megatron.core.tokenizers import MegatronTokenizer

# The metadata will be stored as a file named tokenizer_metadata.json inside the tokenizer’s directory.
MegatronTokenizer.write_metadata(
    tokenizer_path="/path/to/tokenizer.model",
    tokenizer_library="sentencepiece",
    chat_template="chat template in jinja format",
)

# To use custom tokenizer class
from megatron.core.tokenizers.text import MegatronTokenizerText

class CustomTokenizer(MegatronTokenizerText):
    ...

MegatronTokenizer.write_metadata(
    tokenizer_path="/path/to/tokenizer.model",
    tokenizer_library="sentencepiece",
    chat_template="chat template in jinja format",
    tokenizer_class=CustomTokenizer,
)

# To save metadata to another dir
MegatronTokenizer.write_metadata(
    tokenizer_path="/path/to/tokenizer.model",
    tokenizer_library="sentencepiece",
    metadata_path="/path/to/save/metadata.json",
)

```

**Restoring the tokenizer**

```python
from megatron.core.tokenizers import MegatronTokenizer

MegatronTokenizer.from_pretrained(
    tokenizer_path="/path/to/tokenizer.model",
)

# If metadata is not in tokenizer’s dir
MegatronTokenizer.from_pretrained(
    tokenizer_path="/path/to/tokenizer.model",
    metadata_path="/path/to/metadata.json",
)

# Pass metadata as dict
MegatronTokenizer.from_pretrained(
    tokenizer_path="GPT2BPETokenizer",
    metadata_path={"library": "megatron"},
    vocab_file="/path/to/vocab.txt",
)

# Pass additional params
MegatronTokenizer.from_pretrained(
    tokenizer_path="/path/to/tokenizer/model.json",
    metadata_path={"library": "tiktoken"},
    pattern="v2",
    num_special_tokens=1000,
)

# Null tokenzier
MegatronTokenizer.from_pretrained(
    metadata_path={"library": "null"},
    vocab_size=131072,
)

```

### 4. Megatron-LM pretraining compatibility

New tokenizer system is compatible with megatron-lm pretrain script. If `--tokenizer-metadata` is not specified, a default metadata file will be generated automatically.

```bash
# Null tokenizer
torchrun --nproc_per_node=1 pretrain_gpt.py \
    ... \
    --tokenizer-type NullTokenizer \
    --vocab-size 131072

# HuggingFace tokenizer with specified metadata
torchrun --nproc_per_node=1 pretrain_gpt.py \
    ... \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model meta-llama/Meta-Llama-3-8B \
    --tokenizer-metadata /path/to/metadata.json

```

The Megatron-LM pretraining script still supports the legacy tokenizer system. To enable it, simply add the `--legacy-tokenizer` flag.
