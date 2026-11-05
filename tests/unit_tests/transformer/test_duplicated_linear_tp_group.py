# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Duplicated-mode TELinear takes its owner's TP group instead of the global one.

A duplicated linear holds the full weight on every rank of its owner's TP group. Transformer Engine
runs it without TP communication, and the group only sets the weight's checkpoint replica id. The
owners (MLA, absorbed MLA, the DSA, simplified DSA and CSA indexers, and the KDA two-stage gates)
pass their collection's group, so on a grid other than the global one exactly one rank saves each
duplicated weight.
"""

import sys
from functools import partial
from operator import attrgetter

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.dist_checkpointing.mapping import is_main_replica
from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.post_training.modelopt.layers import Linear
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.gated_delta_net import (
    HAVE_FLA_KDA,
    KimiDeltaAttention,
    KimiDeltaAttentionSubmodules,
)
from megatron.core.ssm.kda_layer_config import KDALayerConfig
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant.absorbed_mla import (
    AbsorbedMLASelfAttention,
    AbsorbedMLASelfAttentionSubmodules,
)
from megatron.core.transformer.experimental_attention_variant.csa import (
    Compressor,
    CompressorSubmodules,
    CSAIndexer,
    CSAIndexerSubmodules,
)
from megatron.core.transformer.experimental_attention_variant.dsa import (
    DSAIndexer,
    DSAIndexerSubmodules,
)
from megatron.core.transformer.experimental_attention_variant.dsa_gqa import (
    SimplifiedDSGQAIndexer,
    SimplifiedDSGQAIndexerSubmodules,
)
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.multi_latent_attention import (
    FusedMLASelfAttention,
    MLASelfAttention,
    MLASelfAttentionSubmodules,
)
from megatron.core.transformer.transformer_config import MLATransformerConfig, TransformerConfig
from tests.unit_tests.test_utilities import Utils

if HAVE_TE:
    from megatron.core.extensions.transformer_engine import TELinear, TENorm

_GLOBAL_TP_PREFIXES = ("get_tensor_model_parallel", "get_expert_tensor_parallel")


def _forbid_global_tp_group(patch):
    """Make the global TP accessors raise, including by-name imports of them."""
    for name in dir(parallel_state):
        if not name.startswith(_GLOBAL_TP_PREFIXES):
            continue
        original = getattr(parallel_state, name)

        def forbid(*args, _name=name, **kwargs):
            raise AssertionError(f"read of the global TP group: parallel_state.{_name}")

        for module in list(sys.modules.values()):
            if getattr(module, "__name__", "").startswith("megatron.") and (
                getattr(module, "__dict__", {}).get(name) is original
            ):
                patch.setattr(module, name, forbid)


class _CoreAttention(torch.nn.Module):
    """Stands in for the core attention, which these tests do not run."""

    def __init__(self, *args, **kwargs):
        super().__init__()


def _mla_config(**overrides):
    config = dict(
        num_layers=1,
        hidden_size=256,
        num_attention_heads=8,
        q_lora_rank=128,
        kv_lora_rank=64,
        qk_head_dim=32,
        qk_pos_emb_head_dim=32,
        v_head_dim=32,
        rope_type="rope",
        apply_rope_fusion=False,
        add_bias_linear=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        tensor_model_parallel_size=2,
        sequence_parallel=True,
    )
    config.update(overrides)
    return MLATransformerConfig(**config)


def _mla_submodules(submodules_cls, **down_projections):
    backend = TESpecProvider()
    return submodules_cls(
        linear_q_up_proj=backend.column_parallel_linear(),
        linear_kv_up_proj=backend.column_parallel_linear(),
        core_attention=_CoreAttention,
        linear_proj=backend.row_parallel_linear(),
        q_layernorm=IdentityOp,
        kv_layernorm=IdentityOp,
        **down_projections,
    )


def _build_mla(pg_collection):
    module = MLASelfAttention(
        config=_mla_config(),
        submodules=_mla_submodules(
            MLASelfAttentionSubmodules, linear_q_down_proj=TELinear, linear_kv_down_proj=TELinear
        ),
        layer_number=1,
        attn_mask_type=AttnMaskType.causal,
        pg_collection=pg_collection,
    )
    return module, ["linear_q_down_proj", "linear_kv_down_proj"]


def _build_fused_mla(pg_collection):
    module = FusedMLASelfAttention(
        config=_mla_config(),
        submodules=_mla_submodules(MLASelfAttentionSubmodules, linear_qkv_down_proj=TELinear),
        layer_number=1,
        attn_mask_type=AttnMaskType.causal,
        pg_collection=pg_collection,
    )
    return module, ["linear_qkv_down_proj"]


def _build_absorbed_mla(pg_collection):
    module = AbsorbedMLASelfAttention(
        config=_mla_config(),
        submodules=_mla_submodules(
            AbsorbedMLASelfAttentionSubmodules,
            linear_q_down_proj=TELinear,
            linear_kv_down_proj=TELinear,
        ),
        layer_number=1,
        attn_mask_type=AttnMaskType.causal,
        pg_collection=pg_collection,
    )
    return module, ["linear_q_down_proj", "linear_kv_down_proj"]


def _dsa_indexer_config():
    return _mla_config(
        sequence_parallel=False,
        use_cpu_initialization=True,
        dsa_indexer_n_heads=8,
        dsa_indexer_head_dim=64,
        dsa_indexer_topk=32,
        dsa_indexer_loss_coeff=1.0,
    )


def _build_dsa_indexer(pg_collection):
    submodules = DSAIndexerSubmodules(
        linear_wq_b=TELinear, linear_wk=TELinear, k_norm=TENorm, linear_weights_proj=TELinear
    )
    module = DSAIndexer(_dsa_indexer_config(), submodules, pg_collection)
    return module, ["linear_wq_b", "linear_wk", "linear_weights_proj"]


def _build_simplified_dsa_indexer(pg_collection):
    config = TransformerConfig(
        num_layers=1,
        hidden_size=32,
        num_attention_heads=4,
        num_query_groups=1,
        kv_channels=8,
        experimental_attention_variant="dsa",
        add_bias_linear=False,
        dsa_indexer_mode="simplified",
        dsa_indexer_topk=4,
        dsa_kernel_backend="min-memory-torch",
        dsa_indexer_loss_coeff=0.1,
        dsa_indexer_use_sparse_loss=True,
    )
    # The indexer reads its RoPE settings as dynamic attributes that megatron/training sets.
    config.rotary_percent = 0.0
    submodules = SimplifiedDSGQAIndexerSubmodules(linear_q=TELinear, linear_k=TELinear)
    module = SimplifiedDSGQAIndexer(config, submodules, pg_collection=pg_collection)
    return module, ["linear_q", "linear_k"]


def _build_csa_indexer(pg_collection):
    # CSA runs without tensor parallelism; its owner's TP group has one rank.
    config = _mla_config(
        q_lora_rank=64,
        kv_lora_rank=64,
        qk_head_dim=32,
        v_head_dim=64,
        tensor_model_parallel_size=1,
        sequence_parallel=False,
        use_cpu_initialization=True,
        multi_latent_attention=True,
        csa_compress_ratios=[4],
        csa_window_size=8,
        dsa_indexer_n_heads=8,
        dsa_indexer_head_dim=64,
        dsa_indexer_topk=8,
    )
    compressor_submodules = CompressorSubmodules(
        linear_wkv=TELinear, linear_wgate=TELinear, norm=TENorm
    )
    submodules = CSAIndexerSubmodules(
        linear_wq_b=TELinear,
        linear_weights_proj=TELinear,
        compressor=partial(Compressor, submodules=compressor_submodules),
    )
    module = CSAIndexer(config, submodules, compress_ratio=4, pg_collection=pg_collection)
    return module, [
        "linear_wq_b",
        "linear_weights_proj",
        "compressor.linear_wkv",
        "compressor.linear_wgate",
    ]


def _build_kda(pg_collection):
    if not HAVE_FLA_KDA:
        pytest.skip("FLA with KDA support is not installed")
    # Two-stage gates build the duplicated f_a_proj and g_a_proj.
    config = KDALayerConfig(
        num_layers=1,
        hidden_size=128,
        num_attention_heads=4,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        kda_two_stage_gates=True,
        normalization="RMSNorm",
        add_bias_linear=False,
        use_cpu_initialization=True,
        tensor_model_parallel_size=2,
    )
    backend = TESpecProvider()
    submodules = KimiDeltaAttentionSubmodules(
        in_proj=backend.column_parallel_linear(),
        beta_proj=backend.column_parallel_linear(),
        f_a_proj=TELinear,
        f_b_proj=backend.column_parallel_linear(),
        g_a_proj=TELinear,
        g_b_proj=backend.column_parallel_linear(),
        out_norm=TENorm,
        out_proj=backend.row_parallel_linear(),
    )
    module = KimiDeltaAttention(config, submodules, layer_number=1, pg_collection=pg_collection)
    return module, ["f_a_proj", "g_a_proj"]


_OWNERS = {
    "mla": (_build_mla, 2),
    "fused_mla": (_build_fused_mla, 2),
    "absorbed_mla": (_build_absorbed_mla, 2),
    "dsa_indexer": (_build_dsa_indexer, 2),
    "simplified_dsa_indexer": (_build_simplified_dsa_indexer, 2),
    "csa_indexer": (_build_csa_indexer, 1),
    "kda": (_build_kda, 2),
}


@pytest.mark.skipif(not HAVE_TE, reason="Transformer Engine is not installed")
@pytest.mark.skipif(
    Utils.world_size < 4 or Utils.world_size % 2 != 0,
    reason="needs an even number of ranks, at least 4, so the owner grid differs from the global",
)
class TestDuplicatedLinearUsesOwnerTPGroup:
    """The global grid has TP=2 over adjacent ranks; the owner's grid lays ranks out differently."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(tensor_model_parallel_size=2)
        model_parallel_cuda_manual_seed(123)
        self.grids = []

    def teardown_method(self, method):
        for grid in self.grids:
            grid.destroy()
        Utils.destroy_model_parallel()

    def _owner_groups(self, tp_size):
        """Return a collection on a grid whose DP dimension varies fastest, and its DP x CP group.

        With TP=2 the owner's TP groups pair rank r with rank r + world_size / 2, while the global
        TP groups pair adjacent ranks.
        """
        world_size = torch.distributed.get_world_size()
        grid = HyperCommGrid([world_size // tp_size, 1, 1, tp_size], ["dp", "cp", "pp", "tp"])
        self.grids.append(grid)
        pg_collection = ProcessGroupCollection(
            tp=grid.create_pg("tp"), cp=grid.create_pg("cp"), pp=grid.create_pg("pp")
        )
        dp_cp_group = grid.create_pg(["cp", "dp"])
        global_tp_group = parallel_state.get_tensor_model_parallel_group()
        assert pg_collection.tp is not global_tp_group
        if tp_size == 2:
            assert torch.distributed.get_process_group_ranks(
                pg_collection.tp
            ) != torch.distributed.get_process_group_ranks(global_tp_group)
        return pg_collection, dp_cp_group

    def test_duplicated_linear_checkpoints_on_given_groups(self, monkeypatch):
        pg_collection, dp_cp_group = self._owner_groups(tp_size=2)
        config = TransformerConfig(
            num_layers=1, hidden_size=64, num_attention_heads=4, tensor_model_parallel_size=2
        )

        with monkeypatch.context() as patch:
            _forbid_global_tp_group(patch)
            layer = TELinear(
                64,
                32,
                parallel_mode="duplicated",
                config=config,
                init_method=config.init_method,
                bias=False,
                skip_bias_add=False,
                skip_weight_param_allocation=False,
                tp_group=pg_collection.tp,
            )
            sharded_state_dict = layer.sharded_state_dict(
                prefix="layer.", metadata={"dp_cp_group": dp_cp_group}
            )

        # Transformer Engine sees a plain, non-parallel linear.
        assert layer.parallel_mode is None
        assert layer.tp_size == 1
        assert layer.weight.shape == (32, 64)
        assert layer._tp_group is pg_collection.tp

        replica_id = sharded_state_dict["layer.weight"].replica_id
        assert replica_id == (0, pg_collection.tp.rank(), dp_cp_group.rank())
        replica_ids = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(replica_ids, replica_id)
        assert sum(is_main_replica(replica) for replica in replica_ids) == 1

    def test_standard_grid_selects_the_global_tp_group(self):
        config = TransformerConfig(
            num_layers=1, hidden_size=64, num_attention_heads=4, tensor_model_parallel_size=2
        )
        global_tp_group = parallel_state.get_tensor_model_parallel_group()
        dp_cp_group = parallel_state.get_data_parallel_group(with_context_parallel=True)
        layers = [
            TELinear(
                64,
                32,
                parallel_mode="duplicated",
                config=config,
                init_method=config.init_method,
                bias=False,
                skip_bias_add=False,
                skip_weight_param_allocation=False,
                tp_group=tp_group,
            )
            for tp_group in (global_tp_group, None)
        ]

        replica_ids = [
            layer.sharded_state_dict(prefix="layer.", metadata={"dp_cp_group": dp_cp_group})[
                "layer.weight"
            ].replica_id
            for layer in layers
        ]
        assert all(layer._tp_group is global_tp_group for layer in layers)
        assert replica_ids[0] == replica_ids[1] == (0, global_tp_group.rank(), dp_cp_group.rank())

    @pytest.mark.parametrize("owner", list(_OWNERS))
    def test_owner_passes_its_tp_group(self, monkeypatch, owner):
        build, tp_size = _OWNERS[owner]
        pg_collection, _ = self._owner_groups(tp_size)

        with monkeypatch.context() as patch:
            _forbid_global_tp_group(patch)
            module, duplicated_linear_names = build(pg_collection)

        for name in duplicated_linear_names:
            linear = attrgetter(name)(module)
            assert type(linear) is TELinear, name
            assert linear.parallel_mode is None, name
            assert linear._tp_group is pg_collection.tp, name

    def test_dsa_indexer_passes_its_tp_group_to_local_linears(self, monkeypatch):
        """The ModelOpt spec builds the DSA indexer from local (non-TE) duplicated linears."""
        pg_collection, _ = self._owner_groups(tp_size=2)
        submodules = DSAIndexerSubmodules(
            linear_wq_b=Linear, linear_wk=Linear, k_norm=TENorm, linear_weights_proj=Linear
        )

        with monkeypatch.context() as patch:
            _forbid_global_tp_group(patch)
            module = DSAIndexer(_dsa_indexer_config(), submodules, pg_collection)

        for linear in (module.linear_wq_b, module.linear_wk, module.linear_weights_proj):
            assert type(linear) is Linear
            assert linear.parallel_mode == "duplicated"
            assert linear.tp_group is pg_collection.tp
