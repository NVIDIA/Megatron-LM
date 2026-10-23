# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from copy import deepcopy

import numpy as np
import pytest

from megatron.core.tokenizers import MegatronTokenizer
from megatron.core.tokenizers.vision.libraries.multimodal_tokenizer import IMAGE_TAGS


@pytest.fixture(params=[False, True], ids=["huggingface", "gigatoken"])
def tokenizer(request):
    return MegatronTokenizer.from_pretrained(
        tokenizer_path="/opt/data/tokenizers/multimodal",
        metadata_path={"library": "multimodal"},
        prompt_format="qwen2p0",
        special_tokens=["<image>"],
        image_tag_type="nvlm",
        use_gigatoken=request.param,
    )


@pytest.mark.parametrize("image_tag_type", ["", "nvlm", "internvl"])
def test_legacy_string_image_contract(tokenizer, image_tag_type):
    """Old task encoders must still find vocabulary image IDs, not -200."""
    implementation = tokenizer._tokenizer
    implementation._image_tag = IMAGE_TAGS[image_tag_type]
    conversation = [
        {"role": "user", "content": "Describe <image> and <image>."},
        {"role": "assistant", "content": "Two images."},
    ]
    original = deepcopy(conversation)
    tags = IMAGE_TAGS[image_tag_type] or ("", "")
    rendered = deepcopy(conversation)
    rendered[0]["content"] = rendered[0]["content"].replace("<image>", f"{tags[0]}<image>{tags[1]}")
    expected_text = implementation._hf_tokenizer.apply_chat_template(
        rendered,
        tokenize=False,
        add_generation_prompt=False,
        chat_template=implementation._prompt_config.custom_chat_template,
    )
    expected = implementation._hf_tokenizer.encode(expected_text, add_special_tokens=False)
    for _ in range(2):
        tokens, targets = tokenizer.tokenize_conversation(conversation, True, False)
        np.testing.assert_array_equal(tokens, expected)
        image_id = tokenizer.convert_tokens_to_ids("<image>")
        assert image_id >= 0
        assert np.count_nonzero(tokens == image_id) == 2
        assert not np.any(tokens == tokenizer.image_token_index)
        assert np.all(targets[tokens == image_id] == -100)
        assert np.any(targets != -100)
        assert conversation == original

    expected_plain = implementation._hf_tokenizer.encode(
        f"{tags[0]}<image>{tags[1]} hello", add_special_tokens=False
    )
    assert tokenizer.tokenize("<image> hello", add_special_tokens=False) == expected_plain


def test_structured_image_and_literal_text_remain_distinct(tokenizer):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Literal <image>; actual image: "},
                {"type": "image"},
            ],
        },
        {"role": "assistant", "content": "An image."},
    ]
    original = deepcopy(conversation)
    tokens, targets = tokenizer.tokenize_conversation(conversation, True, False)
    [image_index] = np.flatnonzero(tokens == tokenizer.image_token_index)
    assert np.count_nonzero(tokens == tokenizer.convert_tokens_to_ids("<image>")) == 1
    prefix = tokenizer.detokenize(tokens[:image_index])
    assert "Literal <image>;" in prefix
    assert "<Image><image>" not in prefix
    assert prefix.endswith("<Image>")
    assert tokenizer.detokenize(tokens[image_index + 1 :]).startswith("</Image>")
    assert targets[image_index] == -100
    assert conversation == original


@pytest.mark.parametrize("content", ["An <image>", [{"type": "image"}]])
def test_training_rejects_images_in_assistant_content(tokenizer, content):
    conversation = [
        {"role": "user", "content": "Describe this."},
        {"role": "assistant", "content": content},
    ]
    with pytest.raises(RuntimeError, match="not allowed in assistant content"):
        tokenizer.tokenize_conversation(conversation, True, False)
