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


@pytest.mark.parametrize("context_chunk_size", [128, 256, 64])
def test_wrapper_checks_the_dynamic_context_chunking(monkeypatch, context_chunk_size):
    """The wrapper is where model and context first meet, so it compares the context's own
    prefill chunking with the model's SSM layers before any controller or engine exists."""
    from megatron.core.inference.contexts import DynamicInferenceContext
    from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols

    pg_collection = SimpleNamespace(tp=object(), pp=object())
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 1)
    layers = [SimpleNamespace(mixer=SimpleNamespace(chunk_size=128, ssm_inference_chunk_size=128))]
    model = SimpleNamespace(
        config=SimpleNamespace(
            pipeline_model_parallel_size=1,
            fp32_residual_connection=False,
            params_dtype=torch.float32,
            sequence_parallel=False,
            fp8=None,
        ),
        decoder=SimpleNamespace(
            layer_type_list=[Symbols.MAMBA], physical_layers=lambda: tuple(layers)
        ),
    )
    context = object.__new__(DynamicInferenceContext)
    context.config = SimpleNamespace(pg_collection=pg_collection)
    context.is_hybrid_model = True
    context.mamba_chunk_size = context.ssm_chunk_alignment = context_chunk_size
    context.gdp_num_householder = 0

    if context_chunk_size == 128:
        assert GPTInferenceWrapper(model, context).inference_context is context
    else:
        with pytest.raises(ValueError, match="mamba_chunk_size is .* but the Mamba mixers chunk"):
            GPTInferenceWrapper(model, context)
