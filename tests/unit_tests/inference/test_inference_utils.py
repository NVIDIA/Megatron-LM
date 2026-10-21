# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from megatron.core.inference.utils import (
    Counter,
    detokenize_tokens,
    get_language_model,
    lm_head_can_emit,
)
from megatron.core.tensor_parallel.layers import ColumnParallelLinear


class _Wrapper(torch.nn.Module):
    """Stands in for a multimodal wrapper such as LLaVAModel."""

    def __init__(self, language_model):
        super().__init__()
        self.language_model = language_model


class _MXFP8Head(torch.nn.Module):
    """Stands in for `TELMHeadColumnParallelLinear`, which emits only its input dtype."""


def _lm_head(head_cls, params_dtype):
    """An LM head of `head_cls` reduced to the config the predicate reads."""
    head = object.__new__(head_cls)
    head.config = SimpleNamespace(params_dtype=params_dtype)
    return head


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
    def _head_capabilities(monkeypatch, general_gemm, batch_invariant=False):
        monkeypatch.setattr(
            "megatron.core.extensions.transformer_engine.te_general_gemm",
            object() if general_gemm else None,
        )
        monkeypatch.setattr(
            "megatron.core.transformer.custom_layers.batch_invariant_kernels."
            "is_batch_invariant_mode_enabled",
            lambda: batch_invariant,
        )

    @pytest.mark.parametrize(
        "params_dtype, head_cls, general_gemm, batch_invariant, expected",
        [
            (torch.bfloat16, ColumnParallelLinear, True, False, True),  # general_gemm emits fp32
            (torch.float32, ColumnParallelLinear, False, False, True),  # an fp32 head needs nothing
            (torch.bfloat16, ColumnParallelLinear, False, False, False),  # no general_gemm
            (torch.bfloat16, _MXFP8Head, True, False, False),  # MXFP8 head emits its input dtype
            (torch.bfloat16, ColumnParallelLinear, True, True, False),  # BI keeps training's bf16
            (torch.float32, ColumnParallelLinear, False, True, True),  # fp32 short-circuit under BI
        ],
    )
    def test_lm_head_can_emit_fp32(
        self, monkeypatch, params_dtype, head_cls, general_gemm, batch_invariant, expected
    ):
        """The GEMM input dtype is always available; fp32 from a half-precision head needs TE
        general_gemm, which only the plain `ColumnParallelLinear` head routes through, and is
        skipped under batch-invariant mode (whose guarantee is bitwise parity with the training
        forward)."""
        self._head_capabilities(monkeypatch, general_gemm, batch_invariant)
        head = _lm_head(head_cls, params_dtype)
        assert lm_head_can_emit(head, torch.float32) is expected
        assert lm_head_can_emit(head, params_dtype) is True
