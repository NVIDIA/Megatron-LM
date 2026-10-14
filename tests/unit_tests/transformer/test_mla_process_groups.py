# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""MLA attention variants communicate and checkpoint on the groups of their own collection."""

import sys

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.dist_checkpointing.mapping import ShardedObject, ShardedTensor
from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant.absorbed_mla import (
    AbsorbedMLASelfAttention,
    AbsorbedMLASelfAttentionSubmodules,
)
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.multi_latent_attention import (
    FusedMLASelfAttention,
    MLASelfAttention,
    MLASelfAttentionSubmodules,
)
from megatron.core.transformer.transformer_config import MLATransformerConfig
from tests.unit_tests.test_utilities import Utils

_GLOBAL_GRID_PREFIXES = ("get_tensor_model_parallel", "get_data_parallel", "get_context_parallel")


def _forbid_global_grid(patch):
    """Make the global TP, DP and CP accessors raise, including by-name imports of them."""
    for name in dir(parallel_state):
        if not name.startswith(_GLOBAL_GRID_PREFIXES):
            continue
        original = getattr(parallel_state, name)

        def forbid(*args, _name=name, **kwargs):
            raise AssertionError(f"read of the global grid: parallel_state.{_name}")

        for module in list(sys.modules.values()):
            if getattr(module, "__name__", "").startswith("megatron.") and (
                getattr(module, "__dict__", {}).get(name) is original
            ):
                patch.setattr(module, name, forbid)


class _CoreAttention(torch.nn.Module):
    """Unmasked softmax attention for the standard and the absorbed (single latent KV) layout."""

    def __init__(self, *args, softmax_scale=None, v_channels=None, **kwargs):
        super().__init__()
        self.softmax_scale = softmax_scale
        self.v_channels = v_channels

    def forward(self, query, key, value=None, *args, **kwargs):
        """query/key/value: [s, b, n, d]; absorbed MLA passes one KV head and no value."""
        if value is None:
            value = key[..., : self.v_channels]
        num_heads = query.size(2)
        key = key.expand(-1, -1, num_heads, -1).float()
        value = value.expand(-1, -1, num_heads, -1).float()
        scores = torch.einsum("sbnd,tbnd->bnst", query.float(), key) * self.softmax_scale
        output = torch.einsum("bnst,tbnd->sbnd", scores.softmax(dim=-1), value)
        return output.to(query.dtype).flatten(2)


def _build_attention(variant, pg_collection=None):
    """Build an MLA variant whose down projections are sharded over TP."""
    backend = TESpecProvider()
    config = MLATransformerConfig(
        num_layers=1,
        hidden_size=256,
        num_attention_heads=8,
        q_lora_rank=None if variant.endswith("no_q_lora") else 128,
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
    shared = dict(
        linear_q_proj=backend.column_parallel_linear(),
        linear_q_up_proj=backend.column_parallel_linear(),
        linear_kv_up_proj=backend.column_parallel_linear(),
        core_attention=_CoreAttention,
        linear_proj=backend.row_parallel_linear(),
        q_layernorm=IdentityOp,
        kv_layernorm=IdentityOp,
    )
    if variant.startswith("absorbed"):
        attention_cls = AbsorbedMLASelfAttention
        submodules = AbsorbedMLASelfAttentionSubmodules(
            linear_q_down_proj=backend.column_parallel_linear(),
            linear_kv_down_proj=backend.column_parallel_linear(),
            **shared,
        )
    elif variant == "fused":
        attention_cls = FusedMLASelfAttention
        submodules = MLASelfAttentionSubmodules(
            linear_qkv_down_proj=backend.column_parallel_layer_norm_linear(), **shared
        )
    else:
        attention_cls = MLASelfAttention
        submodules = MLASelfAttentionSubmodules(
            linear_q_down_proj=backend.column_parallel_linear(),
            linear_kv_down_proj=backend.column_parallel_linear(),
            **shared,
        )
    return attention_cls(
        config=config,
        submodules=submodules,
        layer_number=1,
        attn_mask_type=AttnMaskType.causal,
        pg_collection=pg_collection,
    ).cuda()


def _sharding_metadata(sharded_state_dict):
    """The parts of each sharded entry that place it in the global checkpoint."""
    return {
        key: (type(value), value.global_shape, value.global_offset, value.replica_id)
        for key, value in sharded_state_dict.items()
        if isinstance(value, (ShardedTensor, ShardedObject))
    }


@pytest.mark.skipif(
    Utils.world_size < 2 or Utils.world_size % 2 != 0, reason="needs an even number of ranks"
)
class TestMLAWithOwnProcessGroups:
    """MLA built with its own collection matches the global-grid module without reading it.

    The collection has the global grid's layout (TP=2) but new communicators, and the global TP,
    DP and CP accessors raise while the module is built, run and checkpointed.
    """

    def setup_method(self, method):
        Utils.initialize_model_parallel(tensor_model_parallel_size=2)
        model_parallel_cuda_manual_seed(123)
        self.grid = HyperCommGrid([2, 1, 1, Utils.world_size // 2], ["tp", "cp", "pp", "dp"])
        self.pg_collection = ProcessGroupCollection(
            tp=self.grid.create_pg("tp"), cp=self.grid.create_pg("cp"), pp=self.grid.create_pg("pp")
        )
        self.dp_cp_group = self.grid.create_pg(["cp", "dp"])
        assert self.pg_collection.tp is not parallel_state.get_tensor_model_parallel_group()
        assert torch.distributed.get_process_group_ranks(
            self.pg_collection.tp
        ) == torch.distributed.get_process_group_ranks(
            parallel_state.get_tensor_model_parallel_group()
        )

    def teardown_method(self, method):
        self.grid.destroy()
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize(
        "variant", ["mla", "mla_no_q_lora", "absorbed", "absorbed_no_q_lora", "fused"]
    )
    def test_forward_backward_matches_global_grid(self, monkeypatch, variant):
        reference = _build_attention(variant)
        generator = torch.Generator(device="cuda").manual_seed(torch.distributed.get_rank())
        # Sequence parallel: each rank holds [s / TP, b, h].
        hidden_states = torch.randn(
            (8, 2, 256), dtype=torch.bfloat16, device="cuda", generator=generator
        )
        output_grad = torch.randn(
            (8, 2, 256), dtype=torch.bfloat16, device="cuda", generator=generator
        )
        reference_input = hidden_states.clone().requires_grad_(True)
        reference_output, _ = reference(reference_input, attention_mask=None)
        reference_output.backward(output_grad)

        with monkeypatch.context() as patch:
            _forbid_global_grid(patch)
            attention = _build_attention(variant, self.pg_collection)
            attention.load_state_dict(reference.state_dict())
            attention_input = hidden_states.clone().requires_grad_(True)
            output, _ = attention(attention_input, attention_mask=None)
            output.backward(output_grad)

        torch.testing.assert_close(output, reference_output)
        torch.testing.assert_close(attention_input.grad, reference_input.grad)
        reference_params = dict(reference.named_parameters())
        for name, param in attention.named_parameters():
            torch.testing.assert_close(param.grad, reference_params[name].grad, msg=name)

    def test_fused_sharded_state_dict_matches_global_grid(self, monkeypatch):
        reference = _build_attention("fused")
        expected = _sharding_metadata(
            reference.sharded_state_dict(
                metadata={
                    "dp_cp_group": parallel_state.get_data_parallel_group(
                        with_context_parallel=True
                    )
                }
            )
        )

        with monkeypatch.context() as patch:
            _forbid_global_grid(patch)
            attention = _build_attention("fused", self.pg_collection)
            sharded_state_dict = attention.sharded_state_dict(
                metadata={"dp_cp_group": self.dp_cp_group}
            )

        # The split q/kv down-projection weights are the entries this module adds itself.
        assert "linear_q_down_proj.weight" in sharded_state_dict
        assert "linear_kv_down_proj.weight" in sharded_state_dict
        assert _sharding_metadata(sharded_state_dict) == expected
