# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Deprecated global process-group fallbacks keep working and warn once per owner."""

import warnings

import pytest
import torch

from megatron.core import parallel_state, process_groups_config
from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_submodules,
)
from megatron.core.models.multimodal.context_parallel import (
    gather_from_context_parallel_ranks,
    split_to_context_parallel_ranks,
)
from megatron.core.process_groups_config import (
    ProcessGroupCollection,
    warn_global_process_group_fallback,
)
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.multi_token_prediction import mtp_on_this_rank
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_te_min_version
from tests.unit_tests.test_utilities import Utils

if HAVE_TE:
    from megatron.core.extensions.transformer_engine import TEDotProductAttention

requires_te_hierarchical_cp = pytest.mark.skipif(
    not HAVE_TE or not is_te_min_version("1.12.0"),
    reason="hierarchical (a2a+p2p) context parallelism needs Transformer Engine >= 1.12",
)


@pytest.fixture(autouse=True)
def fresh_warning_registry(monkeypatch):
    """Each test observes the first fallback warning of every owner."""
    monkeypatch.setattr(process_groups_config, "_warned_global_process_group_fallbacks", set())


def _fallback_warning(record, owner, argument="pg_collection"):
    """Return the single fallback warning for ``owner`` and check it names this file."""
    matches = [
        w
        for w in record
        if issubclass(w.category, FutureWarning)
        and f"{owner} was called without `{argument}`" in str(w.message)
    ]
    assert len(matches) == 1
    # The warning points at the code that omitted the argument, not at Megatron Core or torch.
    assert matches[0].filename == __file__
    return matches[0]


def test_fallback_warning_is_emitted_once_per_owner():
    with pytest.warns(FutureWarning) as record:
        warn_global_process_group_fallback("Owner", "cp_group")
    message = str(_fallback_warning(record, "Owner", "cp_group").message)
    assert f"since Megatron Core {process_groups_config._FALLBACK_DEPRECATED_IN}" in message
    assert f"removed in {process_groups_config._FALLBACK_REMOVED_IN}" in message
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warn_global_process_group_fallback("Owner", "cp_group")
    with pytest.warns(FutureWarning, match="OtherOwner was called without `pg_collection`"):
        warn_global_process_group_fallback("OtherOwner")


def test_mtp_placement_without_mtp_layers_needs_no_pp_group():
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        assert mtp_on_this_rank(mtp_num_layers=None, ignore_virtual=True) is False


class TestGlobalProcessGroupFallbacks:
    """Each fallback resolves the same groups as the global grid, which has size > 1 here."""

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_rotary_embedding_without_cp_group_uses_global_group(self):
        Utils.initialize_model_parallel(context_parallel_size=2)
        with pytest.warns(FutureWarning) as record:
            rope = RotaryEmbedding(kv_channels=8, rotary_percent=1.0, use_cpu_initialization=True)
        _fallback_warning(record, "RotaryEmbedding", "cp_group")
        assert rope.cp_group is parallel_state.get_context_parallel_group()

    def test_dot_product_attention_without_collection_uses_global_groups(self):
        Utils.initialize_model_parallel(tensor_model_parallel_size=2)
        config = TransformerConfig(
            num_layers=1, hidden_size=16, num_attention_heads=4, tensor_model_parallel_size=2
        )
        with pytest.warns(FutureWarning) as record:
            attention = DotProductAttention(
                config, layer_number=1, attn_mask_type=AttnMaskType.causal, attention_type="self"
            )
        _fallback_warning(record, "DotProductAttention")
        assert attention.tp_group is parallel_state.get_tensor_model_parallel_group()

    @pytest.mark.skipif(not HAVE_TE, reason="the GPT attention spec needs Transformer Engine")
    @pytest.mark.parametrize("test_mode", [False, True])
    def test_self_attention_without_collection_uses_global_groups(self, monkeypatch, test_mode):
        Utils.initialize_model_parallel(tensor_model_parallel_size=2, context_parallel_size=2)
        global_dp_group = parallel_state.get_data_parallel_group()
        if not test_mode:
            # Only run_realtime_tests reads dp, so building must not need an initialized DP group.
            def forbid_dp_group(*args, **kwargs):
                raise AssertionError("SelfAttention resolved the DP group outside test_mode")

            monkeypatch.setattr(parallel_state, "get_data_parallel_group", forbid_dp_group)
        config = TransformerConfig(
            num_layers=1,
            hidden_size=64,
            num_attention_heads=4,
            tensor_model_parallel_size=2,
            context_parallel_size=2,
            use_cpu_initialization=True,
            test_mode=test_mode,
        )
        with pytest.warns(FutureWarning) as record:
            attention = SelfAttention(
                config,
                get_gpt_layer_with_transformer_engine_submodules().self_attention.submodules,
                layer_number=1,
                attn_mask_type=AttnMaskType.causal,
            )
        # The fallback runs in Attention.__init__, reached through SelfAttention's super().
        _fallback_warning(record, "SelfAttention")
        groups = vars(attention.pg_collection)
        assert groups["tp"] is parallel_state.get_tensor_model_parallel_group()
        assert groups["cp"] is parallel_state.get_context_parallel_group()
        assert "hcp" in groups
        if test_mode:
            assert groups["dp"] is global_dp_group
        else:
            assert "dp" not in groups

    @requires_te_hierarchical_cp
    def test_self_attention_without_collection_supports_hierarchical_cp(self):
        Utils.initialize_model_parallel(
            context_parallel_size=2, hierarchical_context_parallel_sizes=[2, 1]
        )
        config = TransformerConfig(
            num_layers=1,
            hidden_size=64,
            num_attention_heads=4,
            context_parallel_size=2,
            hierarchical_context_parallel_sizes=[2, 1],
            cp_comm_type="a2a+p2p",
            use_cpu_initialization=True,
        )
        with pytest.warns(FutureWarning) as record:
            attention = SelfAttention(
                config,
                get_gpt_layer_with_transformer_engine_submodules().self_attention.submodules,
                layer_number=1,
                attn_mask_type=AttnMaskType.causal,
                cp_comm_type="a2a+p2p",
            )
        _fallback_warning(record, "SelfAttention")
        assert (
            attention.pg_collection.hcp is parallel_state.get_hierarchical_context_parallel_groups()
        )

    @requires_te_hierarchical_cp
    @pytest.mark.parametrize("context_parallel_size", [1, 2])
    def test_te_attention_requires_hcp_only_for_hierarchical_cp(self, context_parallel_size):
        Utils.initialize_model_parallel(context_parallel_size=context_parallel_size)
        config = TransformerConfig(
            num_layers=1,
            hidden_size=64,
            num_attention_heads=4,
            context_parallel_size=context_parallel_size,
        )
        # A collection built for another grid carries no hcp unless the caller sets it.
        pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])

        def build():
            return TEDotProductAttention(
                config,
                layer_number=1,
                attn_mask_type=AttnMaskType.causal,
                attention_type="self",
                cp_comm_type="a2a+p2p",
                pg_collection=pg_collection,
            )

        if context_parallel_size == 1:
            build()
        else:
            with pytest.raises(ValueError, match="requires pg_collection.hcp"):
                build()

    def test_mtp_placement_without_pp_group_uses_global_group(self):
        Utils.initialize_model_parallel(pipeline_model_parallel_size=2)
        with pytest.warns(FutureWarning) as record:
            on_this_rank = mtp_on_this_rank(mtp_num_layers=1, ignore_virtual=True)
        _fallback_warning(record, "mtp_on_this_rank", "pp_group")
        # Without a layout, MTP sits on the last of the two pipeline stages only.
        assert on_this_rank == (parallel_state.get_pipeline_model_parallel_rank() == 1)

    def test_context_parallel_helpers_without_cp_group_use_global_group(self):
        Utils.initialize_model_parallel(context_parallel_size=2)
        cp_rank = parallel_state.get_context_parallel_rank()

        global_t = torch.arange(12, dtype=torch.float32, device="cuda").reshape(4, 3)
        with pytest.warns(FutureWarning) as record:
            local_t, global_pad = split_to_context_parallel_ranks(global_t)
        _fallback_warning(record, "split_to_context_parallel_ranks", "cp_group")
        assert torch.equal(local_t, global_t[cp_rank * 2 : (cp_rank + 1) * 2])
        assert global_pad == 0

        global_s = torch.arange(8, dtype=torch.float32, device="cuda").reshape(1, 4, 2)
        local_s = global_s[:, cp_rank * 2 : (cp_rank + 1) * 2].contiguous()
        with pytest.warns(FutureWarning) as record:
            gathered = gather_from_context_parallel_ranks(local_s, 0)
        # The fallback runs inside an autograd Function, reached through torch's apply().
        _fallback_warning(record, "GatherFromContextParallelRanks", "cp_group")
        assert torch.equal(gathered, global_s)
