# Megatron Tokenizer API

## Architecture Overview

The tokenizer subsystem is organized into a layered architecture: a **factory** entry point, **text/vision wrappers** that expose a unified API, and **library implementations** that wrap specific tokenization backends.

```
megatron/core/tokenizers/
    megatron_tokenizer.py        # Factory: MegatronTokenizer.from_pretrained()
    base_tokenizer.py            # ABC: MegatronTokenizerBase
    text/
        text_tokenizer.py        # Wrapper: MegatronTokenizerText
        libraries/
            abstract_tokenizer.py        # ABC for all text libraries
            sentencepiece_tokenizer.py   # Google SentencePiece
            huggingface_tokenizer.py     # HuggingFace AutoTokenizer
            megatron_hf_tokenizer.py     # Megatron-native HF (BPE, WordPiece)
            tiktoken_tokenizer.py        # OpenAI TikToken
            bytelevel_tokenizer.py       # Raw byte-level
            null_tokenizer.py            # No-op (testing / integer passthrough)
        models/                  # (reserved for future model-specific configs)
        parsers/                 # Output parsers (DeepSeek-R1, Qwen3 tool, etc.)
    vision/
        vision_tokenizer.py      # Wrapper: MegatronTokenizerVision
        libraries/
            multimodal_tokenizer.py      # HF-backed multimodal
            null_multimodal_tokenizer.py # No-op multimodal
        models/
    conversation/
        conversation_tokenizer.py  # tokenize_conversation() function
        prompt_config.py           # PromptConfig dataclass + PROMPT_FORMAT_REGISTRY
    utils/
        build_tokenizer.py       # build_tokenizer(args) from CLI arguments
```

### Class Hierarchy

```
MegatronTokenizerBase (ABC)
    MegatronTokenizerText                     # wraps a library tokenizer
        library tokenizer (MegatronTokenizerTextAbstract)
            SentencePieceTokenizer
            HuggingFaceTokenizer
            MegatronHFTokenizer
            TikTokenTokenizer
            ByteLevelTokenizer
            NullTokenizer
    MegatronTokenizerVision                   # wraps a vision library tokenizer
        MegatronMultimodalTokenizer
        MegatronNullMultimodalTokenizer
```

---

## Tokenizer Libraries

Each library wraps a different tokenization backend. The library is selected via `--tokenizer-library` or the `library` field in metadata.

### Text Libraries

| Library name | Class | Backend | Use case |
|---|---|---|---|
| `sentencepiece` | `SentencePieceTokenizer` | Google SentencePiece protobuf | LLaMA, Mistral, and other SP-based models |
| `huggingface` | `HuggingFaceTokenizer` | `transformers.AutoTokenizer` | Any model on HuggingFace Hub or local HF tokenizer |
| `megatron` | `MegatronHFTokenizer` | Megatron-native BPE / WordPiece via HF | GPT-2 BPE (`GPT2BPETokenizer`), BERT WordPiece (`BertWordPieceCase`, `BertWordPieceLowerCase`) |
| `tiktoken` | `TikTokenTokenizer` | OpenAI TikToken | Models using TikToken vocabulary (e.g. custom vocab.json) |
| `byte-level` | `ByteLevelTokenizer` | Raw bytes | Byte-level models; no tokenizer model file needed |
| `null` | `NullTokenizer` | Space-split integers | Testing and debugging; tokenizes `"11 325 97"` to `[11, 325, 97]` |

### Vision Libraries

| Library name | Class | Backend | Use case |
|---|---|---|---|
| `multimodal` | `MegatronMultimodalTokenizer` | HF AutoTokenizer + conversation module | Multimodal models (LLaVA, NVLM, etc.) with image token support |
| `null-multimodal` | `MegatronNullMultimodalTokenizer` | Space-split integers | Testing multimodal pipelines without a real tokenizer |

---

## Command-Line API

### Primary Flags

```
--tokenizer-library {huggingface,sentencepiece,tiktoken,megatron,byte-level,null}
--tokenizer-mode {text,sft,multimodal}     # default: text
--tokenizer-model PATH                      # path to tokenizer file or directory
--tokenizer-prompt-format FORMAT            # required for sft and multimodal modes
```

### Library-Specific Flags

**SentencePiece:**
```
--tokenizer-sentencepiece-legacy            # restore legacy SentencePiece behavior
```

**HuggingFace / Megatron:**
```
--tokenizer-hf-no-use-fast                  # disable fast tokenizer
--tokenizer-hf-no-include-special-tokens    # exclude special tokens from encoding
```

**TikToken:**
```
--tiktoken-pattern PATTERN                  # regex pattern for TikToken
--tiktoken-num-special-tokens N             # default: 1000
```

**Common:**
```
--tokenizer-special-tokens TOK1 TOK2 ...   # additional special token strings
--tokenizer-metadata PATH                   # path to metadata JSON
--vocab-file PATH                           # vocabulary file (megatron library)
--merge-file PATH                           # BPE merges file (megatron library)
--vocab-size N                              # vocabulary size (tiktoken, null, byte-level)
```

### Usage Examples

```bash
# SentencePiece (e.g. LLaMA)
--tokenizer-library sentencepiece \
    --tokenizer-model /path/to/tokenizer.model

# HuggingFace (e.g. any HF model)
--tokenizer-library huggingface \
    --tokenizer-model /path/to/hf-tokenizer-dir

# TikToken
--tokenizer-library tiktoken \
    --tokenizer-model /path/to/tiktoken.vocab.json \
    --tiktoken-pattern v2

# Megatron-native BPE (GPT-2)
--tokenizer-library megatron \
    --tokenizer-model GPT2BPETokenizer \
    --vocab-file /path/to/gpt2-vocab.json \
    --merge-file /path/to/gpt2-merges.txt

# Megatron-native WordPiece (BERT / T5)
--tokenizer-library megatron \
    --tokenizer-model BertWordPieceCase \
    --vocab-file /path/to/bert-large-cased-vocab.txt

# SFT mode (supervised fine-tuning with any text library)
--tokenizer-library huggingface \
    --tokenizer-mode sft \
    --tokenizer-model /path/to/hf-tokenizer \
    --tokenizer-prompt-format nemotron-h-aligned

# Multimodal
--tokenizer-library huggingface \
    --tokenizer-mode multimodal \
    --tokenizer-model /path/to/tokenizer \
    --tokenizer-prompt-format qwen2p0

# Null (testing)
--tokenizer-library null \
    --vocab-size 131072
```

---

## Programmatic API

### Loading a Tokenizer

Use `MegatronTokenizer.from_pretrained()` (not the constructor):

```python
from megatron.core.tokenizers import MegatronTokenizer

# With metadata file (tokenizer_metadata.json auto-detected in tokenizer directory)
tokenizer = MegatronTokenizer.from_pretrained("/path/to/tokenizer")

# With inline metadata dict
tokenizer = MegatronTokenizer.from_pretrained(
    tokenizer_path="/path/to/tokenizer.model",
    metadata_path={"library": "sentencepiece"},
)

# With SFT prompt format
tokenizer = MegatronTokenizer.from_pretrained(
    tokenizer_path="/path/to/hf-tokenizer",
    metadata_path={"library": "huggingface"},
    prompt_format="nemotron-h-aligned",
)

# With chat template
tokenizer = MegatronTokenizer.from_pretrained(
    tokenizer_path="/path/to/tokenizer.model",
    metadata_path={"library": "sentencepiece"},
    chat_template="{% for message in messages %}...",
)
```

### Saving Metadata

```python
MegatronTokenizer.write_metadata(
    tokenizer_path="/path/to/tokenizer",
    tokenizer_library="huggingface",
    chat_template="{% for message in messages %}...",  # optional
    overwrite=True,
)
```

This creates a `tokenizer_metadata.json` file in the tokenizer directory:

```json
{
    "library": "huggingface",
    "class_name": null,
    "class_path": null,
    "model_type": "default-text",
    "chat_template": "{% for message in messages %}..."
}
```

### Building from CLI Arguments

Inside Megatron training scripts, use `build_tokenizer(args)`:

```python
from megatron.core.tokenizers.utils.build_tokenizer import build_tokenizer

tokenizer = build_tokenizer(args)
```

This reads `--tokenizer-library`, `--tokenizer-mode`, `--tokenizer-model`, and all library-specific flags from `args`, then calls `MegatronTokenizer.from_pretrained()` internally.

---

## Core Methods

All tokenizers (text and vision) expose the same high-level interface:

### Encoding & Decoding

```python
# Text to token IDs
ids = tokenizer.tokenize("Hello, world!")

# Token IDs to text
text = tokenizer.detokenize([15496, 11, 995, 0])
```

### Chat Templates

Apply a Jinja2 chat template to a conversation:

```python
conversation = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"},
]

# Returns token IDs (default: tokenize=True)
ids = tokenizer.apply_chat_template(conversation, chat_template=template)

# Returns formatted string
text = tokenizer.apply_chat_template(
    conversation, chat_template=template, tokenize=False
)
```

The chat template resolution order is:
1. Explicit `chat_template` argument
2. `chat_template` from tokenizer metadata
3. `chat_template` from the underlying tokenizer (e.g. HF tokenizer config)

### Special Token Properties

| Property | Aliases | Description |
|---|---|---|
| `vocab_size` | | Total vocabulary size |
| `pad` | `pad_id` | Padding token ID (uses `PromptConfig.pad_token_id` in SFT mode) |
| `eod` | | End-of-document token ID (falls back to `eos_id`) |
| `eos_id` | `eos` | End-of-sentence token ID |
| `bos` | `bos_id` | Beginning-of-sentence token ID |
| `unk` | `unk_id` | Unknown token ID |
| `mask` | `mask_id` | Mask token ID (for MLM) |
| `cls` | `cls_id` | Classification token ID (for BERT-like models) |
| `sep` | `sep_id` | Separator token ID |

---

## SFT & Conversation Tokenization

SFT (Supervised Fine-Tuning) conversation tokenization is a **capability** on any text tokenizer, activated by providing a `prompt_format`. It is not a separate tokenizer library.

### How It Works

When `--tokenizer-mode sft` and `--tokenizer-prompt-format <format>` are set, the text tokenizer gains a `tokenize_conversation()` method that:

1. Formats a conversation using a chat template (Jinja2)
2. Tokenizes the formatted text
3. Optionally produces **target tokens** with non-assistant content masked (`IGNORE_INDEX = -100`) for loss computation

```python
conversation = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "4"},
]

# Get token IDs only
tokens = tokenizer.tokenize_conversation(
    conversation, return_target=False, add_generation_prompt=False
)

# Get token IDs and target mask for training
tokens, target = tokenizer.tokenize_conversation(
    conversation, return_target=True, add_generation_prompt=False
)
# target has IGNORE_INDEX (-100) for system/user tokens
```

### Target Masking

When `return_target=True`, the function produces a target array where:
- **System** and **user** tokens are set to `IGNORE_INDEX` (-100)
- **Tool** tokens are masked if the prompt config enables `allow_tool_role`
- **Assistant prefix tokens** (e.g. `<|assistant|>\n`) are masked based on `assistant_prefix_len`
- **Assistant content tokens** are kept as-is (training signal)

### PromptConfig

Each prompt format defines a `PromptConfig` dataclass that controls tokenization behavior:

```python
@dataclass
class PromptConfig:
    assistant_prefix_len: int        # tokens in assistant header to mask
    pad_token_id: int                # padding token ID for this format
    custom_chat_template: str | None # Jinja2 template override
    has_bos: bool                    # whether tokenizer adds BOS
    has_system_role: bool            # whether format supports system role
    force_system_message: bool       # force a default system message
    system_default: dict | None      # default system message dict
    validate_no_image_in_assistant: bool  # reject <image> in assistant
    capitalize_roles: bool           # capitalize role names (e.g. "User")
    skip_masking: bool               # skip all target masking
    allow_tool_role: bool            # include "tool" in masked roles
```

### Available Prompt Formats

**SFT formats** (used with `--tokenizer-mode sft`):

| Format | Description |
|---|---|
| `nemotron-h-aligned` | Nemotron-H aligned format with `<SPECIAL_10>`, `<SPECIAL_11>` tokens |
| `nemotron-nano-v2` | Nemotron Nano v2 format |
| `identity` | Identity template (concatenates content without role markers) |
| `default` | Uses the tokenizer's native chat template; skips all masking |

**Multimodal formats** (used with `--tokenizer-mode multimodal`):

| Format | Description |
|---|---|
| `mistral` | Mistral format; no system role |
| `llama3` | LLaMA 3 format; has BOS and system role |
| `llama3p1` | LLaMA 3.1 with full chat template |
| `llama3p2` | Same as llama3p1 |
| `nvlm-yi-34b` | Yi 34B NVLM format |
| `chatml` | ChatML (OpenAI-style) format |
| `nemotron5` | Nemotron 5 with special tokens |
| `nemotron5-aligned` | Nemotron 5 aligned with capitalized roles |
| `qwen2p0` | Qwen 2.0 format |
| `qwen2p5` | Qwen 2.5 format |

### Library-Agnostic Design

The conversation tokenization pipeline is library-agnostic. It only requires the underlying tokenizer to implement:
- `apply_chat_template(conversation, tokenize=True, chat_template=..., add_generation_prompt=...)`
- `text_to_ids(text)` (used internally by `apply_chat_template`)

This means SFT works with **any** text tokenizer library:

```bash
# SFT with SentencePiece
--tokenizer-library sentencepiece --tokenizer-mode sft \
    --tokenizer-model /path/to/model.sp \
    --tokenizer-prompt-format nemotron-h-aligned

# SFT with TikToken
--tokenizer-library tiktoken --tokenizer-mode sft \
    --tokenizer-model /path/to/vocab.json \
    --tokenizer-prompt-format default

# SFT with HuggingFace (default for SFT mode)
--tokenizer-library huggingface --tokenizer-mode sft \
    --tokenizer-model /path \
    --tokenizer-prompt-format nemotron-h-aligned
```

---

## Multimodal Tokenization

Vision/multimodal tokenizers (`--tokenizer-mode multimodal`) handle image tokens within conversations.

### Image Tags

The multimodal tokenizer wraps image tokens (e.g. `<image>`) with configurable tags:

| `image_tag_type` | Open tag | Close tag | Result |
|---|---|---|---|
| `nvlm` | `<Image>` | `</Image>` | `<Image><image></Image>` |
| `internvl` | `<img>` | `</img>` | `<img><image></img>` |
| `""` (empty) | — | — | `<image>` (unchanged) |

### Conversation Tokenization

Multimodal tokenizers use the same `tokenize_conversation()` function as SFT, with an additional `apply_image_tag_fn` callback:

```python
tokens = tokenizer.tokenize_conversation(
    conversation, return_target=False, add_generation_prompt=True
)

tokens, target = tokenizer.tokenize_conversation(
    conversation, return_target=True, add_generation_prompt=False
)
```

### Required Arguments

```bash
--tokenizer-library huggingface \
    --tokenizer-mode multimodal \
    --tokenizer-model /path/to/tokenizer \
    --tokenizer-prompt-format qwen2p0 \
    --special-tokens "<image>" \
    --image-tag-type nvlm
```

---

## Deprecation: `--tokenizer-type`

The legacy `--tokenizer-type` flag is deprecated but still supported via automatic mapping in argument validation. It is converted to `--tokenizer-library` + `--tokenizer-mode` internally.

| Old `--tokenizer-type` | New `--tokenizer-library` | New `--tokenizer-mode` |
|---|---|---|
| `GPTSentencePieceTokenizer` | `sentencepiece` | `text` |
| `SentencePieceTokenizer` | `sentencepiece` | `text` |
| `Llama2Tokenizer` | `sentencepiece` | `text` |
| `HuggingFaceTokenizer` | `huggingface` | `text` |
| `TikTokenizer` | `tiktoken` | `text` |
| `GPT2BPETokenizer` | `megatron` | `text` |
| `BertWordPieceCase` | `megatron` | `text` |
| `BertWordPieceLowerCase` | `megatron` | `text` |
| `NullTokenizer` | `null` | `text` |
| `SFTTokenizer` | `huggingface` | `sft` |
| `MultimodalTokenizer` | `huggingface` | `multimodal` |
| `NullMultimodalTokenizer` | `null` | `multimodal` |

When `--tokenizer-type` is used, a deprecation warning is emitted and the value is mapped to the new flags automatically.

---

## Custom Tokenizer Classes

You can register a custom tokenizer class in metadata:

```python
MegatronTokenizer.write_metadata(
    tokenizer_path="/path/to/tokenizer",
    tokenizer_library="huggingface",
    tokenizer_class=MyCustomTokenizerClass,
)
```

This stores `class_name` and `class_path` in the metadata. On load, the factory imports and instantiates the custom class instead of the default `MegatronTokenizerText` / `MegatronTokenizerVision`.

Custom classes must inherit from `MegatronTokenizerBase`.
