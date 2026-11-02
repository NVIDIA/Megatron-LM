# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Dynamic inference with FlashInfer's fused RoPE rotates queries and keys as the unfused path.

With the context's `use_flashinfer_fused_rope`, attention applies RoPE from a cos/sin cache the
model builds (`rotary_pos_cos_sin`) instead of from `rotary_pos_emb`, for the RoPE variants
that cache expresses exactly (RoPE and YaRN); layers in `no_rope_freq` take no RoPE on either
path. Each test runs a prefill and a decode step of one model with fused RoPE off and on, and
compares the queries and keys every attention layer hands to the attention kernel.
"""

from unittest.mock import patch

import pytest
import torch

from megatron.core.inference.config import InferenceConfig, MambaInferenceStateConfig
from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
from megatron.core.inference.inference_request import DynamicInferenceRequest
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.utils import InferenceMode
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_mtp_block_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.attention import Attention
from megatron.core.transformer.module import Float16Module
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

VOCAB_SIZE = 128
MAX_SEQUENCE_LENGTH = 64
PROMPT_LENGTHS = (12, 7)


def _config(**kwargs):
    # Head dim 64 is one FlashInfer rotates exactly.
    return TransformerConfig(
        num_layers=2,
        hidden_size=128,
        num_attention_heads=2,
        use_cpu_initialization=True,
        params_dtype=torch.bfloat16,
        bf16=True,
        **kwargs,
    )


def _with_yarn(config):
    """Set YaRN's attributes, which TransformerConfig does not declare. Scaling factor 8 makes
    YaRN's frequencies and its concentration factor (1.21) differ clearly from plain RoPE's."""
    config.yarn_rotary_scaling_factor = 8.0
    config.yarn_original_max_position_embeddings = 32
    config.yarn_beta_fast = 32.0
    config.yarn_beta_slow = 1.0
    config.yarn_mscale = 1.0
    config.yarn_mscale_all_dim = 0.0
    config.yarn_correction_range_round_to_int = True
    return config


def _rope_inputs_reaching_attention(model, use_flashinfer_fused_rope):
    """Run a prefill and a decode step; return the rotated queries and keys of every attention
    layer, keyed by (step, "query" or "key", layer number), and the number of fused RoPE calls.
    """
    context = DynamicInferenceContext(
        model_config=model.config,
        inference_config=InferenceConfig(
            max_sequence_length=MAX_SEQUENCE_LENGTH,
            buffer_size_gb=0.05,
            block_size_tokens=256,
            unified_memory_level=0,
            mamba_inference_state_config=MambaInferenceStateConfig.from_model(model.module),
            use_flashinfer_fused_rope=use_flashinfer_fused_rope,
        ),
    )
    generator = torch.Generator().manual_seed(7)
    for request_id, length in enumerate(PROMPT_LENGTHS):
        context.add_request(
            DynamicInferenceRequest(
                request_id=request_id,
                prompt_tokens=torch.randint(0, VOCAB_SIZE, (length,), generator=generator),
                sampling_params=SamplingParams(num_tokens_to_generate=2, termination_id=-1),
            )
        )

    step = 0
    captured = {}
    fused_rope_calls = []
    flash_decode_and_prefill = Attention.flash_decode_and_prefill
    apply_fused_qk_rotary_emb = context.apply_fused_qk_rotary_emb
    append_key_value_cache = context.append_key_value_cache

    def record_query(self, q, *args, **kwargs):
        captured[step, "query", self.layer_number] = q[: context.active_token_count].clone()
        return flash_decode_and_prefill(self, q, *args, **kwargs)

    def record_key(layer_number, key, value):
        captured[step, "key", layer_number] = key[: context.active_token_count].clone()
        return append_key_value_cache(layer_number, key, value)

    def record_fused_rope(*args, **kwargs):
        fused_rope_calls.append(step)
        return apply_fused_qk_rotary_emb(*args, **kwargs)

    context.append_key_value_cache = record_key
    context.apply_fused_qk_rotary_emb = record_fused_rope
    with (
        patch.object(Attention, "flash_decode_and_prefill", record_query),
        torch.no_grad(),
        InferenceMode.active(),
    ):
        for step in range(2):
            if step > 0:
                new_tokens = torch.randint(
                    0, VOCAB_SIZE, (len(PROMPT_LENGTHS),), generator=generator
                )
                context.update_requests(
                    active_requests_mask=torch.ones(len(PROMPT_LENGTHS), dtype=torch.int32),
                    new_tokens=new_tokens,
                )
            context.initialize_attention_state()
            input_ids, position_ids = context.current_input_and_position_ids()
            model(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=None,
                inference_context=context,
                runtime_gather_output=True,
            )
    return captured, len(fused_rope_calls)


def _assert_fused_rope_matches_unfused(model, rope_layers):
    """Fused RoPE rotates the queries and keys of `rope_layers` attention layers per step as the
    unfused path does, up to the rounding of the two implementations."""
    reference, reference_fused_calls = _rope_inputs_reaching_attention(model, False)
    fused, fused_calls = _rope_inputs_reaching_attention(model, True)

    assert reference_fused_calls == 0
    assert fused_calls == 2 * rope_layers, f"{fused_calls} fused RoPE calls in two steps"
    assert fused.keys() == reference.keys()
    errors = []
    for name, expected in reference.items():
        error = ((fused[name].float() - expected.float()).norm() / expected.float().norm()).item()
        # bf16 rounding of the two RoPE paths differs by about 0.3%; a missing or extra rotation
        # moves these tensors by tens of percent.
        if error > 0.02:
            errors.append(f"step {name[0]} layer {name[2]} {name[1]}: rel {error:.3g}")
    assert not errors, "fused RoPE differs from the unfused path:\n" + "\n".join(errors)


class TestGPTModelFusedRope:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize(
        "position_embedding_type,no_rope_freq",
        [("rope", None), ("yarn", None), ("rope", [1, 0])],
        ids=["rope", "yarn", "no-rope-layer"],
    )
    def test_fused_rope_matches_unfused(self, position_embedding_type, no_rope_freq):
        config = _config(no_rope_freq=no_rope_freq)
        if position_embedding_type == "yarn":
            _with_yarn(config)
        model = GPTModel(
            config=config,
            transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
            vocab_size=VOCAB_SIZE,
            max_sequence_length=MAX_SEQUENCE_LENGTH,
            position_embedding_type=position_embedding_type,
        )
        model = Float16Module(config, model).cuda().eval()
        _assert_fused_rope_matches_unfused(
            model, rope_layers=config.num_layers - sum(no_rope_freq or [])
        )

    def test_mtp_model_takes_unfused_rope(self):
        """FlashInfer's fused RoPE is not validated with MTP, so a YaRN model with MTP layers
        runs the unfused RoPE with the context's fused RoPE on, instead of failing."""
        config = _with_yarn(_config(mtp_num_layers=1))
        layer_spec = get_gpt_layer_with_transformer_engine_spec()
        model = GPTModel(
            config=config,
            transformer_layer_spec=layer_spec,
            vocab_size=VOCAB_SIZE,
            max_sequence_length=MAX_SEQUENCE_LENGTH,
            position_embedding_type="yarn",
            mtp_block_spec=get_gpt_mtp_block_spec(
                config=config, spec=layer_spec, use_transformer_engine=True
            ),
        )
        model = Float16Module(config, model).cuda().eval()
        _assert_fused_rope_matches_unfused(model, rope_layers=0)


class TestHybridModelFusedRope:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize(
        "position_embedding_type,rotary_interleaved",
        [("rope", False), ("rope", True), ("yarn", False)],
        ids=["rope", "rope-interleaved", "yarn"],
    )
    def test_fused_rope_matches_unfused(self, position_embedding_type, rotary_interleaved):
        config = _config(rotary_interleaved=rotary_interleaved)
        if position_embedding_type == "yarn":
            _with_yarn(config)
        model = HybridModel(
            config=config,
            hybrid_stack_spec=hybrid_stack_spec,
            vocab_size=VOCAB_SIZE,
            max_sequence_length=MAX_SEQUENCE_LENGTH,
            hybrid_layer_pattern="*-",
            position_embedding_type=position_embedding_type,
        )
        model = Float16Module(config, model).cuda().eval()
        _assert_fused_rope_matches_unfused(model, rope_layers=1)

    def test_yarn_rejects_interleaved_layout(self):
        """YaRN does not implement rotary_interleaved; HybridModel rejects it when built, as
        GPTModel does, instead of applying a non-rotation on the unfused path."""
        with pytest.raises(AssertionError, match="Yarn RoPE does not support interleaved"):
            HybridModel(
                config=_with_yarn(_config(rotary_interleaved=True)),
                hybrid_stack_spec=hybrid_stack_spec,
                vocab_size=VOCAB_SIZE,
                max_sequence_length=MAX_SEQUENCE_LENGTH,
                hybrid_layer_pattern="*-",
                position_embedding_type="yarn",
            )
