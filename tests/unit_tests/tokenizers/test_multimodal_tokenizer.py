# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import numpy as np

from megatron.core.models.multimodal.llava_model import IGNORE_INDEX
from megatron.core.tokenizers.text.libraries.sft_tokenizer import PromptConfig
from megatron.core.tokenizers.vision.libraries.multimodal_tokenizer import (
    MegatronMultimodalTokenizer,
)


class _FakeNemotron6Tokenizer:
    """Return a fixed Nemotron 6 MoE conversation and record template options."""

    tokens = np.asarray(
        (10, 3263, 1010, 42, 11, 1010)
        + (10, 1503, 19464, 1010, 98, 11, 1010)
        + (10, 3263, 1010, 43, 11, 1010)
        + (10, 1503, 19464, 1010, 99, 100, 11, 1010),
        dtype=np.int64,
    )

    def apply_chat_template(self, *_args, **kwargs):
        """Return one tokenized conversation."""
        self.template_kwargs = kwargs
        return self.tokens[np.newaxis, :]


def test_nemotron6_moe_masks_non_assistant_tokens_and_keeps_history_thinking():
    """Only assistant content should contribute to loss and history thinking should be retained."""
    tokenizer = object.__new__(MegatronMultimodalTokenizer)
    tokenizer.tokenizer = _FakeNemotron6Tokenizer()
    tokenizer._prompt_config = PromptConfig(
        assistant_prefix_len=None,
        pad_token_id=0,
        custom_chat_template=None,
        has_bos=False,
        has_system_role=True,
    )
    tokenizer._prompt_format = "nemotron6-moe"
    tokenizer._image_tag = None
    tokenizer._keep_history_thinking = True

    tokens, target = tokenizer.tokenize_conversation(
        [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "<think>first</think>answer"},
            {"role": "user", "content": "follow-up"},
            {"role": "assistant", "content": "<think>second</think>answer"},
        ],
        return_target=True,
        add_generation_prompt=False,
    )

    expected_target = np.full_like(tokens, IGNORE_INDEX)
    expected_target[10:12] = tokens[10:12]
    expected_target[23:26] = tokens[23:26]
    np.testing.assert_array_equal(target, expected_target)
    assert tokenizer.tokenizer.template_kwargs["truncate_history_thinking"] is False
