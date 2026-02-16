# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch
from packaging import version

from megatron.core.tokenizers import MegatronTokenizer

try:
    from megatron.core.tokenizers.text.libraries.huggingface_tokenizer import (
        HAVE_TRANSFORMERS,
        HuggingFaceTokenizer,
    )
except Exception:
    HAVE_TRANSFORMERS = False
    HuggingFaceTokenizer = None


def get_conversation():
    return [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {
            "role": "user",
            "content": "Hi, can you help me understand how transformers work in machine learning?",
        },
        {
            "role": "assistant",
            "content": "Sure! Transformers are a type of deep learning model introduced in the paper \"Attention Is All You Need\". They rely heavily on self-attention mechanisms to process sequences of data in parallel, unlike RNNs which process data sequentially.",
        },
        {"role": "user", "content": "What is self-attention?"},
        {
            "role": "assistant",
            "content": "Self-attention is a mechanism that allows the model to weigh the importance of different words in a sentence when encoding each word. It helps the model capture relationships between words regardless of their distance in the sequence.",
        },
        {"role": "user", "content": "Thanks, that's really helpful!"},
        {"role": "assistant", "content": "You're welcome! Let me know if you have more questions."},
    ]


def get_chat_template():
    return """{% for message in messages %}
                    {% if message['role'] == 'system' %}
                <|system|>
                {{ message['content'].strip() }}
                    {% elif message['role'] == 'user' %}
                <|user|>
                {{ message['content'].strip() }}
                    {% elif message['role'] == 'assistant' %}
                <|assistant|>
                {{ message['content'].strip() }}
                    {% endif %}
                {% endfor %}
                {% if add_generation_prompt %}
                <|assistant|>
                {% endif %}"""


def test_sp_tokenizer():
    # Load SP tokenizer
    tokenizer = MegatronTokenizer.from_pretrained(
        "/opt/data/tokenizers/sentencepiece/tokenizer.model"
    )

    # Load SP tokenizer with custom metadata
    metadata = {"library": "sentencepiece", "model_type": "gpt"}

    chat_template = get_chat_template()
    tokenizer = MegatronTokenizer.from_pretrained(
        tokenizer_path="/opt/data/tokenizers/sentencepiece/tokenizer.model",
        metadata_path=metadata,
        chat_template=chat_template,
    )

    # Test chat template
    tokenizer.apply_chat_template(conversation=get_conversation(), chat_template=chat_template)

    # Test tokenization
    ids = tokenizer.tokenize("hi how are you?")
    assert ids == [
        7251,
        920,
        526,
        366,
        29973,
    ], f"[7251, 920, 526, 366, 29973] are expeted ids but got {ids}."

    # Test detokenization
    text = tokenizer.detokenize([306, 29915, 29885, 2691, 3969, 29889])
    assert text == "I'm fine thanks.", f"'I'm fine thanks.' is expeted output but got {text}."

    assert tokenizer.vocab_size == 32000
    assert tokenizer.eos_id == 2
    assert tokenizer.eod == 2
    assert tokenizer.pad == -1
    assert tokenizer.bos == 1


def test_hf_tokenizer():
    # Load HF tokenizer with custom metadata
    metadata = {"library": "huggingface"}
    chat_template = "test chat template"

    tokenizer = MegatronTokenizer.from_pretrained(
        "/opt/data/tokenizers/huggingface", metadata_path=metadata
    )

    # Load HF tokenizer with adding special tokens
    special_tokens = {"bos_token": "<TEST_BOS>", "eos_token": "<TEST_EOS>"}

    tokenizer = MegatronTokenizer.from_pretrained(
        "/opt/data/tokenizers/huggingface",
        metadata_path=metadata,
        chat_template=chat_template,
        **special_tokens,
    )

    assert tokenizer.chat_template == chat_template
    assert tokenizer.tokenize("<TEST_BOS><TEST_EOS>") == [128257, 128256]
    assert tokenizer.detokenize([3, 4, 5]) == "$%&"
    assert tokenizer.vocab_size == 128258


# HuggingFaceTokenizer.ids_to_text and include_special_tokens (--tokenizer-hf-include-special-tokens).
# Uses same local path as test_hf_tokenizer; tests EOS stripping vs keeping in detokenized output (e.g. RL).
LOCAL_HF_TOKENIZER_PATH = "/opt/data/tokenizers/huggingface"


def _eos_in_text(text: str, eos_token: str) -> bool:
    return eos_token in text or text.endswith(eos_token.strip())


@pytest.mark.skipif(not HAVE_TRANSFORMERS, reason="transformers not installed")
@pytest.mark.parametrize("include_special_tokens", [True, False])
@pytest.mark.parametrize("remove_special_tokens", [True, False])
def test_hf_ids_to_text_eos_with_include_and_remove_special_tokens(
    include_special_tokens, remove_special_tokens
):
    """ids_to_text EOS presence: parametrized on include_special_tokens and remove_special_tokens.
    When remove_special_tokens=True, EOS is stripped; when False, EOS is kept (explicit overrides default).
    """
    try:
        tok = HuggingFaceTokenizer(
            LOCAL_HF_TOKENIZER_PATH, include_special_tokens=include_special_tokens
        )
    except Exception:
        pytest.skip("Could not load local HuggingFace tokenizer (path not available)")
    eos_id = tok.eos_id
    ids = tok.text_to_ids("hello") + [eos_id]
    text = tok.ids_to_text(ids, remove_special_tokens=remove_special_tokens)
    eos_expected = not remove_special_tokens
    if eos_expected:
        assert _eos_in_text(text, tok.tokenizer.eos_token), (
            f"Expected EOS in output for include_special_tokens={include_special_tokens}, "
            f"remove_special_tokens={remove_special_tokens}. Got: {text!r}"
        )
    else:
        assert tok.tokenizer.eos_token not in text, (
            f"Expected EOS stripped for include_special_tokens={include_special_tokens}, "
            f"remove_special_tokens={remove_special_tokens}. Got: {text!r}"
        )


def test_megatron_tokenizer():
    # Load tokenizer with additional special tokens
    special_tokens = {}
    special_tokens['additional_special_tokens'] = [f'<extra_id_{i}>' for i in range(100)]

    metadata = {"library": "megatron", "model_type": "gpt"}
    vocab_file = "/opt/data/tokenizers/megatron/gpt2-vocab.json"
    merges_file = "/opt/data/tokenizers/megatron/gpt2-vocab.json"
    tokenizer = MegatronTokenizer.from_pretrained(
        tokenizer_path="GPT2BPETokenizer",
        metadata_path=metadata,
        vocab_file=vocab_file,
        merges_file=merges_file,
        **special_tokens,
    )

    # Test tokenization
    ids = tokenizer.tokenize("hi how are you?")
    assert ids == [
        5303,
        703,
        389,
        345,
        30,
    ], f"[5303, 703, 389, 345, 30] are expeted ids but got {ids}."

    # Test detokenization
    text = tokenizer.detokenize([40, 1101, 3734, 5176, 13])
    assert text == "I'm fine thanks.", f"'I'm fine thanks.' is expeted output but got {text}."

    assert tokenizer.vocab_size == 50357
    assert tokenizer.eos_id == 50256
    assert tokenizer.eod == 50256
    assert tokenizer.model_type == "gpt"

    assert tokenizer.vocab_file == vocab_file
    assert tokenizer.merges_file == merges_file


@pytest.mark.skipif(
    version.parse(torch.__version__) < version.parse('2.3.0'), reason="Not supported for LTS"
)
def test_tiktoken_tokenizer():
    # Load tiktoken tokenizer
    chat_template = get_chat_template()
    tokenizer = MegatronTokenizer.from_pretrained(
        tokenizer_path="/opt/data/tokenizers/tiktoken/tiktoken.vocab.json",
        chat_template=chat_template,
        vocab_size=131072,
    )

    # Test tokenization
    ids = tokenizer.tokenize("hi how are you?")
    assert ids == [
        8101,
        2606,
        1584,
        1636,
        1063,
    ], f"[8101, 2606, 1584, 1636, 1063] are expeted ids but got {ids}."

    # Test detokenization
    text = tokenizer.detokenize([1073, 4525, 7771, 14899, 1046])
    assert text == "I'm fine thanks.", f"'I'm fine thanks.' is expeted output but got {text}."

    text = tokenizer.detokenize([0, 1073, 2, 5])
    assert text == "<unk>I</s><cls>"

    ids = tokenizer.tokenize("<unk>I</s><mask>")
    assert ids == [0, 1073, 2, 3]

    # Test methods
    assert tokenizer.vocab_size == 131072
    assert tokenizer.eos_id == 2
    assert tokenizer.eod == 2
    assert tokenizer.unk == 0
    assert tokenizer.mask == 3
    assert tokenizer.cls == 5

    # Test chat template
    tokenizer.apply_chat_template(conversation=get_conversation(), chat_template=chat_template)


def test_null_tokenizer():
    metadata = {"library": "null-text"}
    tokenizer = MegatronTokenizer.from_pretrained(metadata_path=metadata, vocab_size=131072)

    ids = tokenizer.tokenize("11 325 97")

    assert ids == [11, 325, 97]
    assert tokenizer.vocab_size == 131073


def test_bytelevel_tokenizer():
    metadata = {"library": "byte-level"}
    vocab_size = 1024
    special_tokens = ["<TEST1>", "<TEST2>"]
    tokenizer = MegatronTokenizer.from_pretrained(
        metadata_path=metadata, vocab_size=vocab_size, _bos_id=3, special_tokens=special_tokens
    )

    assert tokenizer.vocab_size == (vocab_size + len(special_tokens))
    assert tokenizer.tokenize("Hello") == [72, 101, 108, 108, 111]
    assert tokenizer.detokenize([72, 101, 108, 108, 111]) == "Hello"


def test_write_metadata():
    tokenizer_path = "/opt/data/tokenizers/huggingface"
    chat_template = "test chat template"
    tokenizer_library = "huggingface"
    MegatronTokenizer.write_metadata(
        tokenizer_path=tokenizer_path,
        tokenizer_library=tokenizer_library,
        chat_template=chat_template,
        overwrite=True,
    )

    # When metadata already exists
    with pytest.raises(ValueError):
        MegatronTokenizer.write_metadata(
            tokenizer_path=tokenizer_path, tokenizer_library=tokenizer_library
        )

    # Overwrite metadata
    class CustomTokenizerClass:
        pass

    MegatronTokenizer.write_metadata(
        tokenizer_path=tokenizer_path,
        tokenizer_library=tokenizer_library,
        tokenizer_class=CustomTokenizerClass,
        overwrite=True,
    )

    # Save metadata to specific path
    metadata_path = f"{tokenizer_path}/test_metadata.json"
    MegatronTokenizer.write_metadata(
        tokenizer_path=tokenizer_path,
        metadata_path=metadata_path,
        tokenizer_library=tokenizer_library,
        model_type="gpt",
        overwrite=True,
    )


def test_multimodal_tokenizer():
    """Test MegatronMultimodalTokenizer."""
    prompt_format = "qwen2p0"
    special_tokens = ["<image>"]
    image_tag_type = "nvlm"
    tokenizer = MegatronTokenizer.from_pretrained(
        tokenizer_path="/opt/data/tokenizers/multimodal",
        metadata_path={"library": "multimodal"},
        prompt_format=prompt_format,
        special_tokens=special_tokens,
        image_tag_type=image_tag_type,
    )
    # Simple encode - decode roundtrip.
    assert (
        tokenizer.detokenize(tokenizer.tokenize("abc")) == "abc"
    ), "encode-decode roundtrip failed"

    conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello! Can you summarize this image for me?"},
        {"role": "user", "content": "<image>"},
        {"role": "assistant", "content": "Sure! The image shows a sunset over a mountain range."},
        {"role": "user", "content": "Thanks! Can you also give a short poem about it?"},
    ]

    conv_tokens = tokenizer.tokenize_conversation(
        conversation, return_target=False, add_generation_prompt=False
    )
    assert len(conv_tokens) > 0, "failed to tokenize conversation"

    conv_tokens, target_tokens = tokenizer.tokenize_conversation(
        conversation, return_target=True, add_generation_prompt=False
    )
    assert len(conv_tokens) > 0 and len(conv_tokens) == len(
        target_tokens
    ), "failed to tokenize conversation and return target tokens"

    # Try converting tokens to ids.
    assert tokenizer.convert_tokens_to_ids("a"), "failed to convert tokens to ids."

    assert tokenizer._tokenizer._apply_image_tag("<image>hello") == "<Image><image></Image>hello"
    assert tokenizer._tokenizer._apply_image_tag([{"role": "user", "content": "<image>hello"}]) == [
        {"role": "user", "content": "<Image><image></Image>hello"}
    ]


def test_null_multimodal_tokenizer():
    """Test MegatronNullMultimodalTokenizer."""
    vocab_size = 10000
    tokenizer = MegatronTokenizer.from_pretrained(
        metadata_path={"library": "null-multimodal"}, vocab_size=vocab_size
    )

    assert tokenizer.vocab_size == (vocab_size + 1), f"expected vocab size is {vocab_size + 1}."

    assert tokenizer.tokenize("1 22 333") == [1, 22, 333], "tokenization is failed."

    assert tokenizer.detokenize([1, 22, 333]) == "1 22 333", "detokenization is failed."


def test_sft_tokenizer():
    """Test SFTTokenizer."""
    prompt_format = "nemotron-nano-v2"
    tokenizer = MegatronTokenizer.from_pretrained(
        tokenizer_path="/opt/data/tokenizers/multimodal",
        metadata_path={"library": "sft"},
        prompt_format=prompt_format,
    )

    # Simple encode - decode roundtrip.
    assert (
        tokenizer.detokenize(tokenizer.tokenize("abc")) == "abc"
    ), "encode-decode roundtrip failed"

    conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello! Can you summarize this image for me?"},
        {"role": "user", "content": "<image>"},
        {"role": "assistant", "content": "Sure! The image shows a sunset over a mountain range."},
        {"role": "user", "content": "Thanks! Can you also give a short poem about it?"},
    ]

    conv_tokens = tokenizer.tokenize_conversation(
        conversation, return_target=False, add_generation_prompt=False
    )
    assert len(conv_tokens) > 0, "failed to tokenize conversation"

    conv_tokens, target_tokens = tokenizer.tokenize_conversation(
        conversation, return_target=True, add_generation_prompt=False
    )
    assert len(conv_tokens) > 0 and len(conv_tokens) == len(
        target_tokens
    ), "failed to tokenize conversation and return target tokens"
