# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.models.T5.t5_spec import (
    decoder_model_with_local_spec,
    decoder_model_with_transformer_engine_default_spec,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.attention import CrossAttention
from megatron.core.transformer.enums import AttnMaskType
from tests.unit_tests.test_utilities import Utils

TP_SIZE = 2


def _cross_attention_submodules(backend):
    if backend == "te":
        pytest.importorskip("transformer_engine")
        layer_spec = decoder_model_with_transformer_engine_default_spec()
    else:
        layer_spec = decoder_model_with_local_spec()
    return layer_spec.submodules.cross_attention.submodules


class TestCrossAttentionWithCustomProcessGroups:

    def setup_method(self, method):
        Utils.destroy_model_parallel()
        self._allow_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False

    def teardown_method(self, method):
        torch.backends.cuda.matmul.allow_tf32 = self._allow_tf32
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("backend", ["local", "te"])
    def test_sequence_parallel_matches_unsharded_reference(self, backend):
        """Q and KV projections must shard and gather over ``pg_collection.tp``.

        The explicit TP groups stride across the global MPU TP groups, and each explicit TP
        group holds a different batch. A sequence-parallel all-gather over the MPU group would
        stitch together sequence shards from different batches.
        """
        if Utils.world_size < 2 * TP_SIZE or Utils.world_size % TP_SIZE != 0:
            pytest.skip(f"Needs a multiple of {TP_SIZE} ranks and at least {2 * TP_SIZE}")
        submodules = _cross_attention_submodules(backend)

        Utils.initialize_model_parallel(tensor_model_parallel_size=TP_SIZE)
        model_parallel_cuda_manual_seed(123)

        # dp is the innermost grid dimension, so TP groups are strided, e.g. [0, 4] on 8 ranks,
        # while the MPU TP groups are contiguous, e.g. [0, 1].
        grid = HyperCommGrid([Utils.world_size // TP_SIZE, TP_SIZE, 1], ["dp", "tp", "cp"])
        pg_collection = ProcessGroupCollection(tp=grid.create_pg("tp"), cp=grid.create_pg("cp"))
        tp_group = pg_collection.tp
        tp_ranks = torch.distributed.get_process_group_ranks(tp_group)
        assert tp_ranks != torch.distributed.get_process_group_ranks(
            parallel_state.get_tensor_model_parallel_group()
        )
        tp_rank = torch.distributed.get_rank(tp_group)

        # Unsharded reference on single-rank groups.
        ref_grid = HyperCommGrid([1, 1, Utils.world_size], ["tp", "cp", "dp"])
        ref_pg_collection = ProcessGroupCollection(
            tp=ref_grid.create_pg("tp"), cp=ref_grid.create_pg("cp")
        )

        config_kwargs = dict(
            num_layers=1,
            hidden_size=64,
            num_attention_heads=4,
            add_bias_linear=False,
            attention_dropout=0.0,
            hidden_dropout=0.0,
        )
        ref_attention = CrossAttention(
            TransformerConfig(**config_kwargs),
            submodules,
            layer_number=1,
            attn_mask_type=AttnMaskType.no_mask,
            pg_collection=ref_pg_collection,
        ).cuda()
        attention = CrossAttention(
            TransformerConfig(
                **config_kwargs, tensor_model_parallel_size=TP_SIZE, sequence_parallel=True
            ),
            submodules,
            layer_number=1,
            attn_mask_type=AttnMaskType.no_mask,
            pg_collection=pg_collection,
        ).cuda()

        for linear in (attention.linear_q, attention.linear_kv, attention.linear_proj):
            assert torch.distributed.get_process_group_ranks(linear.tp_group) == tp_ranks

        # (name, TP shard dim): Q/KV are column parallel, the output projection is row parallel.
        sharded_linears = (("linear_q", 0), ("linear_kv", 0), ("linear_proj", 1))
        generator = torch.Generator().manual_seed(0)
        with torch.no_grad():
            for name, dim in sharded_linears:
                weight = getattr(ref_attention, name).weight
                fan_in = weight.shape[1]
                weight.copy_(torch.randn(weight.shape, generator=generator) / fan_in**0.5)
                getattr(attention, name).weight.copy_(weight.chunk(TP_SIZE, dim=dim)[tp_rank])

        # Replicated within an explicit TP group, distinct across groups.
        generator = torch.Generator().manual_seed(1000 + min(tp_ranks))

        def randn(*shape):
            return torch.randn(*shape, generator=generator).cuda()

        hidden_size = config_kwargs["hidden_size"]
        query_input = randn(8, 2, hidden_size)  # [sq, b, h]
        context_input = randn(12, 2, hidden_size)  # [sk, b, h]
        output_grad = randn(8, 2, hidden_size)

        def shard(tensor):
            return tensor.chunk(TP_SIZE, dim=0)[tp_rank].contiguous()

        ref_query = query_input.clone().requires_grad_()
        ref_context = context_input.clone().requires_grad_()
        ref_output, _ = ref_attention(ref_query, attention_mask=None, key_value_states=ref_context)
        ref_output.backward(output_grad)

        query = shard(query_input).requires_grad_()
        context = shard(context_input).requires_grad_()
        output, _ = attention(query, attention_mask=None, key_value_states=context)
        output.backward(shard(output_grad))

        # Transformer Engine runs FP32 GEMMs in TF32.
        tols = dict(rtol=1e-2, atol=1e-2) if backend == "te" else dict(rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(output, shard(ref_output), **tols)
        torch.testing.assert_close(query.grad, shard(ref_query.grad), **tols)
        torch.testing.assert_close(context.grad, shard(ref_context.grad), **tols)
        for name, dim in sharded_linears:
            torch.testing.assert_close(
                getattr(attention, name).weight.grad,
                getattr(ref_attention, name).weight.grad.chunk(TP_SIZE, dim=dim)[tp_rank],
                **tols,
                msg=lambda m, name=name: f"{name}.weight.grad: {m}",
            )
