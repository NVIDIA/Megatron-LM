# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from collections import OrderedDict
from unittest.mock import MagicMock

import pytest
import torch

from megatron.core.inference.inference_request import InferenceRequest, VLMInferenceRequest
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.text_generation_controllers.encoder_decoder_text_generation_controller import (  # noqa: E501
    EncoderDecoderTextGenerationController,
)
from megatron.core.inference.text_generation_controllers.vlm_text_generation_controller import (
    VLMTextGenerationController,
)


def _make_controller(controller_cls, prep_return=None):
    """Build a controller instance via __new__ + manual attribute injection."""
    ctrl = controller_cls.__new__(controller_cls)
    ctrl.inference_wrapped_model = MagicMock()
    ctrl.inference_wrapped_model.prep_inference_input.return_value = prep_return or {}
    ctrl.tokenizer = MagicMock()
    return ctrl


class TestEncoderDecoderTextGenerationController:

    def test_prep_inference_input_collects_encoder_prompts(self):
        """prep_inference_input forwards the encoder_prompt of each active request to the wrapper."""
        ctrl = _make_controller(EncoderDecoderTextGenerationController, prep_return={"foo": 1})
        sp = SamplingParams(num_tokens_to_generate=1)
        active = OrderedDict()
        active["a"] = InferenceRequest(
            request_id=1, prompt="p1", sampling_params=sp, encoder_prompt="enc1"
        )
        active["b"] = InferenceRequest(
            request_id=2, prompt="p2", sampling_params=sp, encoder_prompt="enc2"
        )
        prompts_tokens = torch.tensor([[1, 2, 3], [4, 5, 6]])
        out = ctrl.prep_inference_input(prompts_tokens, active)
        assert out == {"foo": 1}
        # Wrapper must receive the encoder prompts and tokenizer.
        kwargs = ctrl.inference_wrapped_model.prep_inference_input.call_args
        passed_tokens = kwargs.args[0]
        encoder_prompts = kwargs.args[1]
        assert torch.equal(passed_tokens, prompts_tokens)
        assert encoder_prompts == ["enc1", "enc2"]
        assert kwargs.kwargs["tokenizer"] is ctrl.tokenizer

    def test_prep_inference_input_adds_attention_mask_when_requested(self):
        """use_attention_mask=True populates 'attention_mask' if the wrapper didn't provide one."""
        ctrl = _make_controller(EncoderDecoderTextGenerationController, prep_return={})
        sp = SamplingParams(num_tokens_to_generate=1)
        active = OrderedDict()
        active["a"] = InferenceRequest(
            request_id=1, prompt="p", sampling_params=sp, encoder_prompt="e"
        )
        prompts_tokens = torch.zeros(1, 5, dtype=torch.long)
        out = ctrl.prep_inference_input(prompts_tokens, active, use_attention_mask=True)
        assert "attention_mask" in out

    def test_prep_inference_input_skips_attention_mask_when_disabled(self):
        """use_attention_mask=False leaves the wrapper output untouched."""
        ctrl = _make_controller(EncoderDecoderTextGenerationController, prep_return={"x": 1})
        sp = SamplingParams(num_tokens_to_generate=1)
        active = OrderedDict()
        active["a"] = InferenceRequest(
            request_id=1, prompt="p", sampling_params=sp, encoder_prompt="e"
        )
        prompts_tokens = torch.zeros(1, 4, dtype=torch.long)
        out = ctrl.prep_inference_input(prompts_tokens, active, use_attention_mask=False)
        assert "attention_mask" not in out


class TestVLMTextGenerationController:

    def _make_vlm_request(self):
        sp = SamplingParams(num_tokens_to_generate=1)
        return VLMInferenceRequest(
            request_id=1,
            prompt="describe the picture",
            sampling_params=sp,
            num_img_embeddings_per_tile=4,
            imgs=torch.zeros(1, 3, 8, 8),
            num_tiles=torch.tensor([1]),
            decoder_seq_length=16,
        )

    def test_prep_inference_input_passes_image_fields_to_wrapper(self):
        """VLMTextGenerationController forwards img-specific fields from the request to the wrapper."""
        ctrl = _make_controller(VLMTextGenerationController, prep_return={"out": 1})
        active = OrderedDict()
        active["a"] = self._make_vlm_request()
        prompts_tokens = torch.zeros(1, 4, dtype=torch.long)
        out = ctrl.prep_inference_input(prompts_tokens, active)
        assert out == {"out": 1}
        args = ctrl.inference_wrapped_model.prep_inference_input.call_args.args
        # args = (prompts_tokens, num_img_embeddings_per_tile, imgs, num_tiles, decoder_seq_length)
        assert args[1] == 4
        assert args[4] == 16

    def test_prep_inference_input_asserts_batch_size_one(self):
        """Multiple active requests trigger an assertion (only batch=1 is supported)."""
        ctrl = _make_controller(VLMTextGenerationController)
        active = OrderedDict()
        active["a"] = self._make_vlm_request()
        active["b"] = self._make_vlm_request()
        prompts_tokens = torch.zeros(2, 4, dtype=torch.long)
        with pytest.raises(AssertionError, match="batch size 1"):
            ctrl.prep_inference_input(prompts_tokens, active)

    def test_prep_inference_input_asserts_request_type(self):
        """A non-VLM request triggers an AssertionError."""
        ctrl = _make_controller(VLMTextGenerationController)
        sp = SamplingParams(num_tokens_to_generate=1)
        active = OrderedDict()
        active["a"] = InferenceRequest(request_id=1, prompt="text-only", sampling_params=sp)
        prompts_tokens = torch.zeros(1, 4, dtype=torch.long)
        with pytest.raises(AssertionError, match="VLMInferenceRequest"):
            ctrl.prep_inference_input(prompts_tokens, active)

    def test_prep_inference_input_adds_attention_mask_when_requested(self):
        """use_attention_mask=True populates 'attention_mask' when the wrapper didn't."""
        ctrl = _make_controller(VLMTextGenerationController, prep_return={})
        active = OrderedDict()
        active["a"] = self._make_vlm_request()
        prompts_tokens = torch.zeros(1, 5, dtype=torch.long)
        out = ctrl.prep_inference_input(prompts_tokens, active, use_attention_mask=True)
        assert "attention_mask" in out
