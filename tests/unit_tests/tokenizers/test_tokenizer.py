import json
import sys

import pytest
import torch
from packaging import version

from megatron.core.tokenizers import MegatronTokenizer


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
    metadata = {"library": "null"}
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
