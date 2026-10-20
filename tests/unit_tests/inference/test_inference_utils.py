# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import warnings
from types import SimpleNamespace

import pytest
import torch

from megatron.core.inference.utils import (
    Counter,
    detokenize_tokens,
    get_language_model,
    resolve_logits_dtype,
)
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

    @pytest.mark.parametrize("multimodal", [False, True])
    @pytest.mark.parametrize(
        "logit_dtype, requested, mxfp8, general_gemm, expected, warns",
        [
            (None, None, False, True, torch.bfloat16, False),  # None: the head's own dtype ...
            (torch.float32, None, False, True, torch.float32, False),  # ... set by `logit_dtype`
            (None, torch.float32, False, True, torch.float32, False),  # honored via general_gemm
            (torch.float32, torch.bfloat16, False, True, torch.bfloat16, False),  # input dtype
            (None, torch.float32, False, False, torch.bfloat16, True),  # no general_gemm
            (None, torch.float32, True, True, torch.bfloat16, True),  # MXFP8 output projection
        ],
    )
    def test_resolve_logits_dtype(
        self, monkeypatch, multimodal, logit_dtype, requested, mxfp8, general_gemm, expected, warns
    ):
        """`requested` wins when the head can emit it: always for the GEMM input dtype, through
        TE general_gemm otherwise, never from the MXFP8 output projection. The fallback is the
        language model's `logit_dtype`, else `params_dtype`, read through a multimodal wrapper."""
        language_model = torch.nn.Linear(2, 2)
        language_model.logit_dtype = logit_dtype
        model = _Wrapper(language_model) if multimodal else language_model
        model_config = SimpleNamespace(params_dtype=torch.bfloat16)
        self._head_capabilities(monkeypatch, mxfp8, general_gemm)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert resolve_logits_dtype(model, model_config, requested) is expected
        assert len([w for w in caught if "keeps the logits" in str(w.message)]) == int(warns)

    @pytest.mark.parametrize(
        "requested, general_gemm, expected",
        [
            (torch.float32, True, torch.float32),  # asked and able
            (torch.bfloat16, False, torch.bfloat16),  # the input dtype needs nothing
            (torch.float32, False, None),  # unable: the head keeps its own dtype
            (None, True, None),  # nothing asked
        ],
    )
    def test_language_module_inference_logits_dtype(
        self, monkeypatch, requested, general_gemm, expected
    ):
        """The model-side half of the contract: the head asks its output layer for the context's
        `logits_dtype` exactly when `resolve_logits_dtype` would honor it, and for nothing
        without a context."""
        self._head_capabilities(monkeypatch, mxfp8=False, general_gemm=general_gemm)
        head = SimpleNamespace(config=SimpleNamespace(params_dtype=torch.bfloat16))
        context = SimpleNamespace(config=SimpleNamespace(logits_dtype=requested))
        assert LanguageModule.inference_logits_dtype(head, context) is expected
        assert LanguageModule.inference_logits_dtype(head, None) is None
