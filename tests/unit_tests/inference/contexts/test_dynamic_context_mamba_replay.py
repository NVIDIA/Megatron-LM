# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""ReplaySSM bookkeeping on `DynamicInferenceContext`.

  * Startup validation: `mamba_replay_ssm` is accepted only when the model's recurrent layers
    are Mamba-2 (no Gated DeltaNet, no Gated Delta Product, and at least one Mamba layer).
  * Ring-cursor resets: every path that binds an exact SSM state to a Mamba slot (local
    zero/restore, disaggregated decode admission) must leave that slot with an empty ring,
    even when the slot is recycled from a request that left its cursors mid-ring.
"""

import pytest
import torch

from megatron.core.inference.config import InferenceConfig, MambaInferenceStateConfig
from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
from megatron.core.inference.disaggregation.decode_admission import (
    additional_decode_blocks,
    admit_prefilled_decode,
)
from megatron.core.inference.inference_request import DynamicInferenceRequest
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

BLOCK_SIZE_TOKENS = 8
NUM_SPECULATIVE_TOKENS = 2
MAMBA_HYBRID = [Symbols.MAMBA, Symbols.MLP, Symbols.ATTENTION, Symbols.MLP]
GDN_HYBRID = [Symbols.GDN, Symbols.MLP, Symbols.ATTENTION, Symbols.MLP]


def _make_context(
    layer_type_list=MAMBA_HYBRID, hybrid=True, gdp_num_householder=0, mamba_replay_ssm=True
) -> DynamicInferenceContext:
    """Build a real context; conv/SSM shapes describe 8 heads x 64 dims, 1 group, d_state 16."""
    mamba_inference_state_config = None
    if hybrid:
        mamba_inference_state_config = MambaInferenceStateConfig(
            layer_type_list=list(layer_type_list),
            conv_states_shape=(544, 4),
            ssm_states_shape=(8, 64, 16),
            conv_states_dtype=torch.bfloat16,
            ssm_states_dtype=torch.bfloat16,
            gdp_num_householder=gdp_num_householder,
        )
    return DynamicInferenceContext(
        model_config=TransformerConfig(
            params_dtype=torch.bfloat16, num_layers=4, kv_channels=16, num_attention_heads=4
        ),
        inference_config=InferenceConfig(
            max_sequence_length=64,
            buffer_size_gb=0.1,
            paused_buffer_size_gb=0.02,
            block_size_tokens=BLOCK_SIZE_TOKENS,
            max_tokens=256,
            max_requests=16,
            num_speculative_tokens=NUM_SPECULATIVE_TOKENS,
            mamba_inference_state_config=mamba_inference_state_config,
            mamba_replay_ssm=mamba_replay_ssm,
            unified_memory_level=0,  # unit tests currently broken with UVM
        ),
    )


def _dirty_cursors(context, slots):
    """Leave a previous occupant's non-empty ring (mid-ring, flush pending) on each slot."""
    for slot in slots:
        context.mamba_replay_write_pos[slot] = 5
        context.mamba_replay_origin[slot] = 3
        context.mamba_replay_is_flush[slot] = 1


def _cursors(context, slot):
    return (
        int(context.mamba_replay_write_pos[slot]),
        int(context.mamba_replay_origin[slot]),
        int(context.mamba_replay_is_flush[slot]),
    )


class TestMambaReplayStartupValidation:
    """`mamba_replay_ssm` must fail fast on models whose recurrent layers are not Mamba-2."""

    @classmethod
    def setup_class(cls):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        model_parallel_cuda_manual_seed(123)

    @classmethod
    def teardown_class(cls):
        Utils.destroy_model_parallel()

    def test_mamba2_hybrid_is_accepted(self):
        context = _make_context()
        assert context.mamba_replay_ssm
        assert context.mamba_replay_write_pos.shape == (context.max_requests,)

    @pytest.mark.parametrize(
        "kwargs,reason",
        [
            (dict(layer_type_list=GDN_HYBRID), "Gated DeltaNet"),
            # Gated Delta Product mixers sit in Mamba ("M") layers; only the Householder count
            # tells them apart from Mamba-2.
            (dict(gdp_num_householder=2), "Gated Delta Product"),
            (dict(hybrid=False), "no Mamba layers"),
        ],
        ids=["gdn", "gdp", "transformer"],
    )
    def test_non_mamba2_models_are_rejected(self, kwargs, reason):
        with pytest.raises(ValueError, match=reason):
            _make_context(**kwargs)

    def test_validation_is_skipped_when_disabled(self):
        _make_context(layer_type_list=GDN_HYBRID, mamba_replay_ssm=False)


class TestMambaReplayCursorResets:
    """Binding an exact state to a recycled slot must empty that slot's ring, and only it."""

    @classmethod
    def setup_class(cls):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        model_parallel_cuda_manual_seed(123)

    @classmethod
    def teardown_class(cls):
        Utils.destroy_model_parallel()

    def test_pending_zero_resets_cursors(self):
        context = _make_context()
        slot, other = (int(context.mamba_metadata.allocate_slot()) for _ in range(2))
        _dirty_cursors(context, [slot, other])

        context._pending_mamba_zeros.append(slot)
        context._execute_pending_mamba_ops()

        assert _cursors(context, slot) == (0, 0, 0)
        assert _cursors(context, other) == (5, 3, 1)

    def test_disaggregated_admission_resets_cursors(self):
        """An imported SSM state is exact (handed off right after prefill), so the decode
        engine's recycled slot must not replay the previous occupant's ring on top of it."""
        context = _make_context()
        slot, other = (int(context.mamba_metadata.allocate_slot()) for _ in range(2))
        _dirty_cursors(context, [slot, other])

        prompt_length = BLOCK_SIZE_TOKENS + 3
        input_tokens = list(range(NUM_SPECULATIVE_TOKENS + 1))
        num_prompt_blocks = -(-prompt_length // BLOCK_SIZE_TOKENS)
        num_decode_blocks = additional_decode_blocks(
            prompt_length,
            len(input_tokens),
            BLOCK_SIZE_TOKENS,
            mtp_kv_cache=context.enable_mtp_kv_cache,
        )
        blocks = context.kv_block_allocator.allocate_memory_blocks(
            num_prompt_blocks + num_decode_blocks
        ).tolist()
        request = DynamicInferenceRequest(
            request_id=1,
            prompt_tokens=torch.arange(prompt_length, device=torch.cuda.current_device()),
            sampling_params=SamplingParams(num_tokens_to_generate=10),
            block_size_tokens=BLOCK_SIZE_TOKENS,
            enable_prefix_caching=False,
        )

        admit_prefilled_decode(
            context,
            request,
            prompt_block_ids=blocks[:num_prompt_blocks],
            continuation_block_ids=blocks[num_prompt_blocks:],
            input_tokens=input_tokens,
            ssm_state_idx=slot,
        )

        assert int(context.mamba_metadata.request_to_mamba_state_idx[0]) == slot
        assert _cursors(context, slot) == (0, 0, 0)
        assert _cursors(context, other) == (5, 3, 1)
