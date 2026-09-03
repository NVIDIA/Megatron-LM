# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for megatron.elastification.router.flex_budget_utils."""

import pytest

from megatron.core.models.hybrid import MTPSplit, PipelineSplit
from megatron.core.ssm.mamba_layer_config import MambaLayerConfig
from megatron.core.ssm.mlp_layer_config import MLPLayerConfig
from megatron.core.transformer.attention_layer_config import AttentionLayerConfig
from megatron.core.transformer.moe.moe_layer_config import MoELayerConfig
from megatron.elastification.router.flex_budget_utils import get_num_parameters

# Reference dimensions used by most tests. Small enough to compute by hand.
_DIMS = dict(
    mamba_num_heads=4,
    mamba_d_head=2,
    mamba_d_state=2,
    num_attention_heads=2,
    num_query_groups=1,
    ffn_hidden_size=8,
    hidden_size=4,
    kv_channels=2,
    vocab_size=10,
    num_experts=2,
    shared_expert_intermediate_size=0,
    moe_router_topk=1,
)

_EMBED_PLUS_LN = (_DIMS["vocab_size"] * _DIMS["hidden_size"]) + _DIMS["hidden_size"]
_OUTPUT_LAYER = _DIMS["vocab_size"] * _DIMS["hidden_size"]


def _layers(*layer_config_types):
    return tuple(
        layer_config_type(
            num_layers=1,
            hidden_size=_DIMS["hidden_size"],
            num_attention_heads=_DIMS["num_attention_heads"],
            num_query_groups=_DIMS["num_query_groups"],
            ffn_hidden_size=_DIMS["ffn_hidden_size"],
            kv_channels=_DIMS["kv_channels"],
            num_moe_experts=_DIMS["num_experts"],
            moe_router_topk=_DIMS["moe_router_topk"],
            moe_router_pre_softmax=True,
            mamba_num_heads=_DIMS["mamba_num_heads"],
            mamba_head_dim=_DIMS["mamba_d_head"],
            mamba_state_dim=_DIMS["mamba_d_state"],
        )
        for layer_config_type in layer_config_types
    )


def _att_cost():
    h, k, q = (_DIMS["hidden_size"], _DIMS["kv_channels"], _DIMS["num_query_groups"])
    n_heads = _DIMS["num_attention_heads"]
    input_ln = h
    linear_proj = n_heads * k * h
    linear_qkv = (n_heads + 2 * q) * k * h
    return input_ln + linear_proj + linear_qkv


def _moe_cost_all():
    pre_mlp_ln = _DIMS["hidden_size"]
    n_experts = _DIMS["num_experts"]
    ffn = _DIMS["ffn_hidden_size"]
    shared = _DIMS["shared_expert_intermediate_size"]
    h = _DIMS["hidden_size"]
    linear_fc1 = ffn * (h * n_experts + shared)
    linear_fc2 = ffn * (h * n_experts + shared)
    return pre_mlp_ln + linear_fc1 + linear_fc2


def _moe_cost_active():
    pre_mlp_ln = _DIMS["hidden_size"]
    topk = _DIMS["moe_router_topk"]
    ffn = _DIMS["ffn_hidden_size"]
    shared = _DIMS["shared_expert_intermediate_size"]
    h = _DIMS["hidden_size"]
    linear_fc1 = ffn * (h * topk + shared)
    linear_fc2 = ffn * (h * topk + shared)
    return pre_mlp_ln + linear_fc1 + linear_fc2


def _mamba_cost(num_groups=8):
    h = _DIMS["hidden_size"]
    nheads = _DIMS["mamba_num_heads"]
    d_head = _DIMS["mamba_d_head"]
    d_state = _DIMS["mamba_d_state"]
    d_inner = nheads * d_head
    cdim = d_inner + 2 * num_groups * d_state
    mamba_conv = cdim + cdim * 1 * 4  # bias + weight, kernel=4, stride=1
    mamba_input_ln = h
    mamba_in_proj = h * (d_inner * 2 + 2 * num_groups * d_state + nheads)
    mamba_norm = d_inner
    mamba_out_proj = d_inner * h
    scalars = nheads + nheads + nheads  # dt_bias + A_log + D
    return scalars + mamba_input_ln + mamba_in_proj + mamba_conv + mamba_norm + mamba_out_proj


class TestGetNumParameters:
    def test_single_moe_layer_matches_manual(self):
        total, active = get_num_parameters(
            layer_config_list=_layers(MoELayerConfig), tied_vocab=False, **_DIMS
        )
        expected_total = _EMBED_PLUS_LN + _OUTPUT_LAYER + _moe_cost_all()
        expected_active = _EMBED_PLUS_LN + _OUTPUT_LAYER + _moe_cost_active()
        assert total == expected_total
        assert active == expected_active

    def test_single_attention_layer(self):
        total, active = get_num_parameters(
            layer_config_list=_layers(AttentionLayerConfig), tied_vocab=False, **_DIMS
        )
        expected = _EMBED_PLUS_LN + _OUTPUT_LAYER + _att_cost()
        assert total == expected
        # Attention has no active/total split.
        assert active == expected

    def test_single_mamba_layer(self):
        total, active = get_num_parameters(
            layer_config_list=_layers(MambaLayerConfig), tied_vocab=False, **_DIMS
        )
        expected = _EMBED_PLUS_LN + _OUTPUT_LAYER + _mamba_cost()
        assert total == expected
        assert active == expected

    def test_mamba_layer_uses_typed_config_group_count(self):
        layer_config = _layers(MambaLayerConfig)[0]
        layer_config.mamba_num_groups = 2

        total, active = get_num_parameters(
            layer_config_list=(layer_config,), tied_vocab=False, **_DIMS
        )
        expected = _EMBED_PLUS_LN + _OUTPUT_LAYER + _mamba_cost(num_groups=2)
        assert total == expected
        assert active == expected

    def test_layer_config_list_is_sum_of_per_layer_costs(self):
        layer_config_list = _layers(
            MambaLayerConfig, MoELayerConfig, MambaLayerConfig, AttentionLayerConfig, MoELayerConfig
        )
        total, active = get_num_parameters(
            layer_config_list=layer_config_list, tied_vocab=False, **_DIMS
        )
        expected_total = (
            _EMBED_PLUS_LN + _OUTPUT_LAYER + 2 * _mamba_cost() + 2 * _moe_cost_all() + _att_cost()
        )
        expected_active = (
            _EMBED_PLUS_LN
            + _OUTPUT_LAYER
            + 2 * _mamba_cost()
            + 2 * _moe_cost_active()
            + _att_cost()
        )
        assert total == expected_total
        assert active == expected_active

    def test_tied_vocab_zeros_output_layer(self):
        total_tied, _ = get_num_parameters(
            layer_config_list=_layers(MambaLayerConfig), tied_vocab=True, **_DIMS
        )
        total_untied, _ = get_num_parameters(
            layer_config_list=_layers(MambaLayerConfig), tied_vocab=False, **_DIMS
        )
        # Untied adds one more vocab*hidden block.
        assert total_untied - total_tied == _DIMS["vocab_size"] * _DIMS["hidden_size"]

    @pytest.mark.parametrize("marker", [PipelineSplit, MTPSplit])
    def test_split_markers_are_rejected(self, marker):
        with pytest.raises(NotImplementedError, match="does not support"):
            get_num_parameters(
                layer_config_list=(*_layers(MambaLayerConfig), marker), tied_vocab=False, **_DIMS
            )

    def test_unsupported_layer_config_is_rejected(self):
        with pytest.raises(NotImplementedError, match="supports only"):
            get_num_parameters(layer_config_list=_layers(MLPLayerConfig), tied_vocab=False, **_DIMS)

    def test_moe_active_less_than_or_equal_total(self):
        # topk < num_experts, so active < total; topk == num_experts, active == total.
        total_tk1, active_tk1 = get_num_parameters(
            layer_config_list=_layers(MoELayerConfig),
            tied_vocab=False,
            **{**_DIMS, "moe_router_topk": 1},
        )
        total_tkN, active_tkN = get_num_parameters(
            layer_config_list=_layers(MoELayerConfig),
            tied_vocab=False,
            **{**_DIMS, "moe_router_topk": _DIMS["num_experts"]},
        )
        assert active_tk1 < total_tk1
        assert active_tkN == total_tkN

    def test_topk_zero_active_excludes_experts(self):
        # With topk=0 the active cost per expert's linear_fc1/fc2 contribution
        # collapses to 0 (shared_expert_intermediate_size=0 in our fixture).
        _, active = get_num_parameters(
            layer_config_list=_layers(MoELayerConfig),
            tied_vocab=False,
            **{**_DIMS, "moe_router_topk": 0},
        )
        # active == embed + output + pre_mlp_ln (no fc1/fc2 contribution)
        assert active == _EMBED_PLUS_LN + _OUTPUT_LAYER + _DIMS["hidden_size"]

    def test_empty_list_only_embeddings_and_final_norm(self):
        total, active = get_num_parameters(layer_config_list=(), tied_vocab=False, **_DIMS)
        assert total == _EMBED_PLUS_LN + _OUTPUT_LAYER
        assert active == _EMBED_PLUS_LN + _OUTPUT_LAYER
