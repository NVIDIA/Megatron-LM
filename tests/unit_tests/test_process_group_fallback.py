# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Deprecated global process-group fallbacks keep working and warn once per owner."""

import warnings

import pytest
import torch

from megatron.core import parallel_state, process_groups_config
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.models.multimodal.context_parallel import split_to_context_parallel_ranks
from megatron.core.process_groups_config import warn_global_process_group_fallback
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.multi_token_prediction import mtp_on_this_rank
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


@pytest.fixture(autouse=True)
def fresh_warning_registry(monkeypatch):
    """Each test observes the first fallback warning of every owner."""
    monkeypatch.setattr(process_groups_config, "_warned_global_process_group_fallbacks", set())


def test_fallback_warning_is_emitted_once_per_owner():
    with pytest.warns(DeprecationWarning, match="Owner was called without `cp_group`"):
        warn_global_process_group_fallback("Owner", "cp_group")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warn_global_process_group_fallback("Owner", "cp_group")
    with pytest.warns(DeprecationWarning, match="OtherOwner was called without `pg_collection`"):
        warn_global_process_group_fallback("OtherOwner")


class TestGlobalProcessGroupFallbacks:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_rotary_embedding_without_cp_group_uses_global_group(self):
        with pytest.warns(
            DeprecationWarning, match="RotaryEmbedding was called without `cp_group`"
        ):
            rope = RotaryEmbedding(kv_channels=8, rotary_percent=1.0, use_cpu_initialization=True)
        assert rope.cp_group is parallel_state.get_context_parallel_group()

    def test_dot_product_attention_without_collection_uses_global_groups(self):
        config = TransformerConfig(num_layers=1, hidden_size=16, num_attention_heads=4)
        with pytest.warns(
            DeprecationWarning, match="DotProductAttention was called without `pg_collection`"
        ):
            attention = DotProductAttention(
                config, layer_number=1, attn_mask_type=AttnMaskType.causal, attention_type="self"
            )
        assert attention.tp_group is parallel_state.get_tensor_model_parallel_group()

    def test_mtp_placement_without_pp_group_uses_global_group(self):
        with pytest.warns(
            DeprecationWarning, match="mtp_on_this_rank was called without `pp_group`"
        ):
            on_this_rank = mtp_on_this_rank(mtp_num_layers=1, ignore_virtual=True)
        assert on_this_rank == mtp_on_this_rank(
            mtp_num_layers=1,
            ignore_virtual=True,
            pp_group=parallel_state.get_pipeline_model_parallel_group(),
        )

    def test_context_parallel_split_without_cp_group_uses_global_group(self):
        global_t = torch.arange(12, dtype=torch.float32).reshape(4, 3)
        with pytest.warns(
            DeprecationWarning,
            match="split_to_context_parallel_ranks was called without `cp_group`",
        ):
            local_t, global_pad = split_to_context_parallel_ranks(global_t)
        expected_t, expected_pad = split_to_context_parallel_ranks(
            global_t, cp_group=parallel_state.get_context_parallel_group()
        )
        assert torch.equal(local_t, expected_t)
        assert global_pad == expected_pad
