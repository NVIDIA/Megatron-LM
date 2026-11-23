# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import numpy as np
import pytest
from tokenizers import Regex, Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from megatron.core.tokenizers.vision.libraries import multimodal_tokenizer


def _tokenizer(monkeypatch, prompt_format):
    """Small deterministic header vocabulary; no external model downloads.

    Model tokenizers are additionally exercised by integration checks. The fixture
    keeps the Llama header lengths (6/10) and Nemotron-H boundary ID (11) explicit
    so changes to the rendered templates cannot silently shift the loss mask.
    """
    pieces = [
        "<unk>",
        "<|begin_of_text|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|eot_id|>",
        "<|end_of_text|>",
        "<|finetune_right_pad_id|>",
        "<SPECIAL_0>",
        "<SPECIAL_10>",
        "<pad>",
        "<reserved>",
        "<SPECIAL_11>",
        "system",
        "user",
        "assistant",
        "System",
        "User",
        "Assistant",
        "\n",
        "\n\n",
        "det",
        "ailed",
        " thinking",
        " off",
        "QuestionA",
        "QuestionB",
        "AnswerA",
        "AnswerB",
    ]
    backend = Tokenizer(
        models.WordLevel({piece: i for i, piece in enumerate(pieces)}, unk_token="<unk>")
    )
    backend.pre_tokenizer = pre_tokenizers.Split(
        Regex(r"<\|[^>]+\|>|<SPECIAL_\d+>|\n\n|\n|det|ailed| ?\w+|[^\w\s]"), "isolated"
    )
    hf = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token="<|begin_of_text|>",
        unk_token="<unk>",
        pad_token="<pad>",
    )
    monkeypatch.setattr(
        multimodal_tokenizer.transformers.AutoTokenizer, "from_pretrained", lambda **kwargs: hf
    )
    return multimodal_tokenizer.MegatronMultimodalTokenizer("unused", prompt_format, [], "")


@pytest.mark.parametrize(
    "prompt_format",
    [
        "llama_nemotron_8b",
        "nemotron-h-reasoning",
        "llama-nemotron-super",
        "llama-nemotron-super-1p5",
    ],
)
@pytest.mark.parametrize("with_system", [False, True])
def test_prompt_format_masks_headers_and_user_turns(monkeypatch, prompt_format, with_system):
    tokenizer = _tokenizer(monkeypatch, prompt_format)
    conversation = [
        {"role": "system", "content": "QuestionA"},
        {"role": "user", "content": "QuestionA"},
        {"role": "assistant", "content": "AnswerA"},
        {"role": "user", "content": "QuestionB"},
        {"role": "assistant", "content": "AnswerB"},
    ]
    if not with_system:
        # Exercise the templates' automatic default system headers too.
        conversation = conversation[1:]
    tokens, targets = tokenizer.tokenize_conversation(conversation, True, False)
    hf = tokenizer._hf_tokenizer
    for text in ["QuestionA", "QuestionB", "user", "assistant", "User", "Assistant"]:
        positions = tokens == hf.convert_tokens_to_ids(text)
        assert np.all(targets[positions] == -100)
    for text in ["AnswerA", "AnswerB"]:
        [position] = np.flatnonzero(tokens == hf.convert_tokens_to_ids(text))
        assert targets[position] == tokens[position]
    if prompt_format != "nemotron-h-reasoning":
        assert np.count_nonzero(tokens == hf.bos_token_id) == 1
        assert tokens[0] == hf.bos_token_id
        assert targets[0] == -100
    prompt = tokenizer.tokenize_conversation(conversation[:-1], False, True)
    rendered = tokenizer._apply_chat_template_to_text(conversation[:-1], True)
    np.testing.assert_array_equal(prompt, hf.encode(rendered, add_special_tokens=False))


def test_super15_preserves_published_tool_instruction(monkeypatch):
    """Preserve the literal braces found in the published model template."""
    tokenizer = _tokenizer(monkeypatch, "llama-nemotron-super-1p5")
    rendered = tokenizer._apply_chat_template_to_text(
        [{"role": "user", "content": "QuestionA"}],
        True,
        tools=[{"type": "function", "function": {"name": "example", "parameters": {}}}],
    )
    # Published nvidia/Llama-3_3-Nemotron-Super-49B-v1_5 tokenizer_config.json,
    # revision 420ba7d28211abf116b8b103ab700d92619daf98, uses these literal braces.
    assert '<TOOLCALL>[{{"name": "tool_name1", "arguments": "tool_args1"}}' in rendered
    assert '<TOOL_RESPONSE>[{{"response": "tool_response1"}}' in rendered
