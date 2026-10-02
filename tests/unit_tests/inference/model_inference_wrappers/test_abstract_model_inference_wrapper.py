# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from megatron.core.inference.config import MediaPromptSpec, MultimodalPromptConfig
from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import (
    GPTInferenceWrapper,
)


def _wrapper(prompt_config=None):
    wrapper = object.__new__(GPTInferenceWrapper)
    wrapper.multimodal_prompt_config = prompt_config
    return wrapper


def test_wrapper_rejects_shortcut_moe_models():
    """Every engine builds one of these wrappers, so shortcut MoE is refused for all of them."""
    model = SimpleNamespace(config=SimpleNamespace(moe_shortcut_connection=True))

    with pytest.raises(NotImplementedError, match="moe_shortcut_connection"):
        GPTInferenceWrapper(model, inference_context=None)


@pytest.mark.parametrize("enable_mhc,pp_size", [(True, 2), (True, 1), (False, 2)])
def test_wrapper_rejects_mhc_pipeline_inference(monkeypatch, enable_mhc, pp_size):
    """Use the inference PP group, which can differ from the training configuration."""
    pg_collection = SimpleNamespace(tp=object(), pp=object())
    model = SimpleNamespace(
        config=SimpleNamespace(
            enable_mhc_connections=enable_mhc,
            pipeline_model_parallel_size=3 - pp_size,
            fp32_residual_connection=False,
            params_dtype=torch.float32,
            sequence_parallel=False,
            fp8=None,
        )
    )
    context = SimpleNamespace(config=SimpleNamespace(pg_collection=pg_collection))
    monkeypatch.setattr(
        torch.distributed,
        "get_world_size",
        lambda group: pp_size if group is pg_collection.pp else 1,
    )

    if enable_mhc and pp_size > 1:
        with pytest.raises(NotImplementedError, match="mHC inference with pipeline parallelism"):
            GPTInferenceWrapper(model, context)
    else:
        wrapper = GPTInferenceWrapper(model, context)
        assert wrapper.pp_group is pg_collection.pp


def test_validate_input_modalities_accepts_declared_capabilities_and_rejects_others():
    wrapper = _wrapper()
    wrapper.supports_image = True

    wrapper.validate_input_modalities("text", "image")

    with pytest.raises(ValueError, match="does not support video inputs"):
        wrapper.validate_input_modalities("video")
    with pytest.raises(ValueError, match="Unknown input modality"):
        wrapper.validate_input_modalities("depth")


def test_resolve_media_token_id_uses_nested_tokenizer_and_list_conversion_fallback():
    class _Tokenizer:
        unk_token_id = 0

        def convert_tokens_to_ids(self, tokens):
            if isinstance(tokens, str):
                raise TypeError("this tokenizer requires a list")
            return [42]

    wrapper = _wrapper(MultimodalPromptConfig(image_spec=MediaPromptSpec(model_token="<image>")))
    tokenizer = SimpleNamespace(_tokenizer=SimpleNamespace(tokenizer=_Tokenizer()))

    assert wrapper.resolve_media_token_id(tokenizer, "image") == 42


def test_resolve_media_token_id_ignores_unknown_id_and_uses_tensor_encode_fallback():
    tokenizer = SimpleNamespace(
        unk_token_id=0,
        convert_tokens_to_ids=lambda _token: 0,
        encode=lambda _token, add_special_tokens=False: torch.tensor([57]),
    )
    wrapper = _wrapper(MultimodalPromptConfig(video_spec=MediaPromptSpec(model_token="<video>")))

    assert wrapper.resolve_media_token_id(tokenizer, "video") == 57


@pytest.mark.parametrize(
    ("prompt_config", "tokenizer", "error"),
    [
        (None, SimpleNamespace(), "does not define a multimodal prompt contract"),
        (
            MultimodalPromptConfig(image_spec=MediaPromptSpec(model_token="")),
            SimpleNamespace(),
            "does not define a model token",
        ),
        (
            MultimodalPromptConfig(image_spec=MediaPromptSpec(model_token="<image>")),
            SimpleNamespace(
                encode=lambda _token, add_special_tokens=False: [1, 2],
                tokenize=lambda _token: [1, 2],
            ),
            "does not define '<image>' as one nonnegative token",
        ),
    ],
)
def test_resolve_media_token_id_rejects_missing_or_ambiguous_contracts(
    prompt_config, tokenizer, error
):
    with pytest.raises(ValueError, match=error):
        _wrapper(prompt_config).resolve_media_token_id(tokenizer, "image")


def test_build_preexpanded_media_token_mask_numbers_only_media_positions():
    wrapper = _wrapper()
    wrapper.get_preexpanded_media_token_id = lambda modality: -200
    prompt_tokens = torch.tensor([10, -200, 20, -200, -200], dtype=torch.int32)

    mask = wrapper.build_preexpanded_media_token_mask(prompt_tokens, "image")

    assert mask.dtype == torch.int64
    assert mask.tolist() == [-1, 0, -1, 1, 2]


def test_build_preexpanded_media_token_mask_handles_text_only_prompt():
    wrapper = _wrapper()
    wrapper.get_preexpanded_media_token_id = lambda modality: -200

    mask = wrapper.build_preexpanded_media_token_mask(torch.tensor([10, 20, 30]), "video")

    assert mask.tolist() == [-1, -1, -1]


def test_base_preexpanded_media_token_id_requires_model_specific_implementation():
    wrapper = _wrapper()

    with pytest.raises(NotImplementedError, match="pre-expanded video token id"):
        wrapper.get_preexpanded_media_token_id("video")
