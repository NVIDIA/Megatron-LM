# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from megatron.core.inference.utils import Counter, detokenize_tokens, get_language_model
from megatron.core.models.common.language_module.language_module import LanguageModule


class _Wrapper(torch.nn.Module):
    """Stands in for a multimodal wrapper such as LLaVAModel."""

    def __init__(self, language_model):
        super().__init__()
        self.language_model = language_model


def _tokenizer(generation_config=None):
    """A tokenizer whose `</s>` is id 2, detokenizing ids to space-separated text."""
    return SimpleNamespace(
        eod=2,
        generation_config=generation_config,
        detokenize=lambda tokens: " ".join(str(token) for token in tokens),
    )


class TestInferenceUtils:

    def test_counter(self):
        counter = Counter()
        r = next(counter)
        assert r == 0, f'Counter return value should be 0 but it is {r}'
        assert counter.counter == 1, f'Counter should be 1 but it is {counter.counter}'
        counter.reset()
        assert counter.counter == 0, f'Counter should be 0 but it is {counter.counter}'

    def test_get_language_model_returns_a_plain_model_itself(self):
        model = torch.nn.Linear(2, 2)
        assert get_language_model(model) is model

    def test_get_language_model_resolves_a_multimodal_wrapper(self):
        language_model = torch.nn.Linear(2, 2)
        assert get_language_model(_Wrapper(language_model)) is language_model

    def test_get_language_model_keeps_a_wrapper_without_a_language_model(self):
        # e.g. a pipeline stage that builds no decoder.
        wrapper = _Wrapper(None)
        assert get_language_model(wrapper) is wrapper

    def test_detokenize_strips_every_generation_config_eos(self):
        """A chat model can stop on `<|im_end|>` (11) as well as `</s>` (2)."""
        tokenizer = _tokenizer({"eos_token_id": [2, 11]})
        assert detokenize_tokens(tokenizer, [5, 6, 11]) == "5 6"
        assert detokenize_tokens(tokenizer, [5, 6, 2]) == "5 6"
        assert detokenize_tokens(tokenizer, [11]) == ""

    def test_detokenize_keeps_eos_inside_the_text_and_when_asked(self):
        tokenizer = _tokenizer({"eos_token_id": [2, 11]})
        assert detokenize_tokens(tokenizer, [5, 11, 6]) == "5 11 6"
        assert detokenize_tokens(tokenizer, [5, 6, 11], remove_EOD=False) == "5 6 11"

    def test_detokenize_without_generation_config_strips_only_eod(self):
        tokenizer = _tokenizer()
        assert detokenize_tokens(tokenizer, [5, 11, 2]) == "5 11"

    @staticmethod
    def _head_capabilities(monkeypatch, mxfp8, general_gemm):
        monkeypatch.setattr(
            "megatron.core.fp8_utils.is_mxfp8_output_proj_active", lambda config: mxfp8
        )
        monkeypatch.setattr(
            "megatron.core.extensions.transformer_engine.te_general_gemm",
            object() if general_gemm else None,
        )

    @pytest.mark.parametrize(
        "params_dtype, mxfp8, general_gemm, expected",
        [
            (torch.bfloat16, False, True, torch.float32),  # general_gemm emits fp32 from bf16
            (torch.float32, False, False, torch.float32),  # an fp32 head needs nothing
            (torch.bfloat16, False, False, None),  # no general_gemm: the head keeps bf16
            (torch.bfloat16, True, True, None),  # the MXFP8 output projection emits its input dtype
        ],
    )
    def test_language_module_inference_logits_dtype(
        self, monkeypatch, params_dtype, mxfp8, general_gemm, expected
    ):
        """Under an inference context the head asks its output layer for fp32 exactly when it can
        emit it (`lm_head_can_emit`); outside inference it asks for nothing."""
        monkeypatch.setattr(
            "megatron.core.fp8_utils.is_mxfp8_output_proj_active", lambda config: mxfp8
        )
        monkeypatch.setattr(
            "megatron.core.extensions.transformer_engine.te_general_gemm",
            object() if general_gemm else None,
        )
        head = object.__new__(LanguageModule)
        object.__setattr__(head, "config", SimpleNamespace(params_dtype=params_dtype))
        assert head.inference_logits_dtype(SimpleNamespace()) is expected
        assert head.inference_logits_dtype(None) is None
