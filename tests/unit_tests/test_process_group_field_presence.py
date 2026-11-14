# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Builders reject a process-group field they read but the caller never set.

ProcessGroupCollection returns None for a declared field that was never set, so
``hasattr(pg_collection, name)`` is always true and ``getattr(pg_collection, name, None)`` cannot
tell an unset field from one set to None. The builders below check the fields they read with
``vars()`` semantics: an unset field raises, and a field set to None keeps meaning that its axis
is off.
"""

from types import SimpleNamespace

import pytest
import torch

from megatron.core import parallel_state, process_groups_config
from megatron.core.extensions.transformer_engine import (
    HAVE_TE,
    TEDotProductAttention,
    TEFusedMLP,
    TELayerNormColumnParallelLinear,
    TERowParallelLinear,
)
from megatron.core.process_groups_config import ProcessGroupCollection, ProcessGroupFallbackWarning
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant.dsa import (
    DSAttention,
    DSAttentionSubmodules,
)
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import (
    Utils,
    forbid_global_process_groups,
    new_group_with_same_ranks,
)


def _fail_if_built(*args, **kwargs):
    raise AssertionError("the MLP must not be built from a collection without tp")


class TestDSAttentionGroups:
    """DSAttention takes its tensor- and context-parallel layout from its collection."""

    @staticmethod
    def _build(pg_collection):
        # Layer 2 shares the top-k indices of layer 1, so it builds no indexer.
        config = SimpleNamespace(
            dsa_indexer_topk=8,
            dsa_indexer_topk_freq=4,
            dsa_indexer_skip_topk_offset=1,
            kv_channels=16,
        )
        return DSAttention(
            config=config,
            submodules=DSAttentionSubmodules(indexer=object()),
            layer_number=2,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
            softmax_scale=1.0,
            pg_collection=pg_collection,
        )

    @pytest.mark.parametrize("field", ["tp", "cp"])
    def test_rejects_an_unset_group(self, field):
        # An unset cp or tp would read as None in the forward pass, which turns that axis off.
        groups = {"tp": None, "cp": None}
        del groups[field]
        with pytest.raises(ValueError, match=f"DSAttention requires pg_collection to set {field}"):
            self._build(ProcessGroupCollection(**groups))

    def test_none_groups_mean_that_the_axes_are_off(self):
        attention = self._build(ProcessGroupCollection(tp=None, cp=None))
        assert attention.pg_collection.tp is None
        assert attention.pg_collection.cp is None


_MLP_CLASSES = [
    pytest.param(MLP, id="MLP"),
    pytest.param(
        TEFusedMLP,
        marks=pytest.mark.skipif(TEFusedMLP is None, reason="TE operation-based MLP unavailable"),
        id="TEFusedMLP",
    ),
]


@pytest.mark.parametrize("mlp_class", _MLP_CLASSES)
def test_mlp_builder_rejects_a_collection_without_tp(mlp_class):
    # Without tp, the MLP would shard its weights over the global tensor-parallel group.
    config = TransformerConfig(num_layers=1, hidden_size=8, num_attention_heads=2)
    with pytest.raises(ValueError, match="as_mlp_submodule requires pg_collection to set tp"):
        mlp_class.as_mlp_submodule(
            submodules=MLPSubmodules(linear_fc1=_fail_if_built, linear_fc2=_fail_if_built),
            config=config,
            pg_collection=ProcessGroupCollection(cp=None),
            is_mtp_layer=False,
        )


@pytest.mark.skipif(not HAVE_TE, reason="needs Transformer Engine")
class TestTensorParallelGroupOfTheCollection:
    """With the collection's groups passed, the builders read no global process group."""

    @pytest.fixture(autouse=True)
    def fresh_warning_registry(self, monkeypatch):
        monkeypatch.setattr(process_groups_config, "_warned_global_process_group_fallbacks", set())

    def setup_method(self, method):
        if Utils.world_size < 2 or Utils.world_size % 2:
            pytest.skip("needs an even number of ranks")
        Utils.initialize_model_parallel(tensor_model_parallel_size=2)
        model_parallel_cuda_manual_seed(123)
        # Same ranks as the global TP group, but another communicator.
        self.tp_group = new_group_with_same_ranks(parallel_state.get_tensor_model_parallel_group())

    def teardown_method(self, method):
        torch.distributed.destroy_process_group(self.tp_group)
        Utils.destroy_model_parallel()

    @staticmethod
    def _config():
        return TransformerConfig(
            num_layers=1,
            hidden_size=64,
            ffn_hidden_size=128,
            num_attention_heads=4,
            tensor_model_parallel_size=2,
        )

    def _build_attention(self, pg_collection):
        return TEDotProductAttention(
            self._config(),
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
            pg_collection=pg_collection,
        )

    def test_attention_uses_the_tp_group_of_the_collection(self):
        # Without context parallelism the attention reads no cp group, so cp may stay unset.
        with forbid_global_process_groups():
            attention = self._build_attention(ProcessGroupCollection(tp=self.tp_group))
        assert attention._tp_group is self.tp_group

    def test_attention_rejects_a_collection_without_tp(self):
        with forbid_global_process_groups():
            with pytest.raises(ValueError, match="TEDotProductAttention requires pg_collection"):
                self._build_attention(ProcessGroupCollection(cp=None))

    def test_attention_without_a_collection_warns_and_uses_the_global_tp_group(self):
        with pytest.warns(ProcessGroupFallbackWarning, match="TEDotProductAttention"):
            attention = self._build_attention(None)
        assert attention._tp_group is parallel_state.get_tensor_model_parallel_group()

    def test_mlp_builder_uses_the_tp_group_of_the_collection(self):
        # gtp_remat=None keeps the linear layers from looking up GTP groups in the global grid.
        pg_collection = ProcessGroupCollection(tp=self.tp_group, gtp_remat=None)
        with forbid_global_process_groups():
            mlp = MLP.as_mlp_submodule(
                submodules=MLPSubmodules(
                    linear_fc1=TELayerNormColumnParallelLinear, linear_fc2=TERowParallelLinear
                ),
                config=self._config(),
                pg_collection=pg_collection,
                is_mtp_layer=False,
                ffn_hidden_size=128,
            )
        assert mlp.tp_group is self.tp_group
