# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""DCP graph discovery and replay must operate on the mHC wrapper, not its inner layer."""

from types import SimpleNamespace

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.models.hybrid.hybrid_block import HybridStack, HyperConnectionHybridLayer
from megatron.core.models.hybrid.hybrid_layer_allocation import validate_segment_layers
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.num_microbatches_calculator import (
    destroy_num_microbatches_calculator,
    init_num_microbatches_calculator,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import (
    initialize_rng_tracker,
    model_parallel_cuda_manual_seed,
)
from megatron.core.transformer.cuda_graphs import (
    TECudaGraphHelper,
    _DynamicCPCaptureCallable,
    _layer_is_graphable,
)
from megatron.core.transformer.enums import CudaGraphModule
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.module import GraphableMegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer
from tests.unit_tests.test_utilities import Utils


def _wrapper(config):
    layer = HyperConnectionHybridLayer.__new__(HyperConnectionHybridLayer)
    torch.nn.Module.__init__(layer)
    layer.config = config
    inner = TransformerLayer.__new__(TransformerLayer)
    torch.nn.Module.__init__(inner)
    inner.self_attention = torch.nn.Identity()
    inner.cross_attention = IdentityOp()
    inner.mlp = IdentityOp()
    layer.inner_layer = inner
    return layer


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("scope", [[], [CudaGraphModule.attn]])
def test_wrapper_discovery_preserves_dynamic_cp_support(dynamic, scope):
    config = SimpleNamespace(dynamic_context_parallel=dynamic, cuda_graph_modules=scope)
    layer = _wrapper(config)
    assert _layer_is_graphable(layer, config)
    if dynamic:
        # Inheriting the bank-selection helper does not confer THD/DCP kernel support.
        layer.inner_layer = GraphableMegatronModule.__new__(GraphableMegatronModule)
        torch.nn.Module.__init__(layer.inner_layer)
        assert not _layer_is_graphable(layer, config)


@pytest.mark.parametrize("cp_size", [1, 2, 4, 8])
def test_wrapper_reconstructs_capture_variant(cp_size):
    group = object()
    layer = _wrapper(
        SimpleNamespace(
            dynamic_context_parallel=True,
            context_parallel_size=8,
            max_seqlen_per_dp_cp_rank=64,
            cp_partition_mode="zigzag",
            _cuda_graph_capture_dynamic_cp=(cp_size, group),
        )
    )
    kwargs = {
        key: torch.tensor([0, 64 * cp_size], dtype=torch.int32)
        for key in ("cu_seqlens_q", "cu_seqlens_kv", "cu_seqlens_q_padded", "cu_seqlens_kv_padded")
    }
    layer._reconstruct_packed_seq_params_from_kwargs(kwargs)
    params = kwargs["packed_seq_params"]
    assert params.local_cp_size == cp_size
    assert params.cp_group is group
    assert params.max_seqlen_q == params.max_seqlen_kv == 64 * cp_size
    assert params.pad_between_seqs


def test_wrapper_replay_selects_own_bank_before_decomposing_metadata():
    layer = _wrapper(SimpleNamespace(fine_grained_activation_offloading=False))
    groups = {size: object() for size in (1, 2)}
    replayed = []

    def graph(size):
        def replay(hidden_states, **kwargs):
            replayed.append(size)
            assert "packed_seq_params" not in kwargs
            return (hidden_states,)

        return replay

    layer.cuda_graphs_by_dynamic_cp_size = {size: [graph(size)] for size in groups}
    layer.cuda_graph_cp_groups_by_dynamic_cp_size = groups
    layer.cuda_graphs = layer.cuda_graphs_by_dynamic_cp_size[2]
    layer.cuda_graph_manual_hooks = []
    layer._te_cuda_graph_static_hidden_inputs_by_dynamic_cp_size = {}
    for index, size in enumerate((1, 2, 1, 2)):
        layer.current_microbatch = index  # W=1: replay wraps on every microbatch.
        layer._te_cuda_graph_replay(
            torch.ones(1),
            packed_seq_params=PackedSeqParams(
                qkv_format="thd", local_cp_size=size, cp_group=groups[size]
            ),
        )
    assert replayed == [1, 2, 1, 2]
    with pytest.raises(RuntimeError, match="does not match"):
        layer._te_cuda_graph_replay(
            torch.ones(1), packed_seq_params=PackedSeqParams(local_cp_size=1, cp_group=object())
        )
    assert len(replayed) == 4


class _GraphTestChunk(torch.nn.Module):
    def zero_grad_buffer(self):
        self.zero_grad(set_to_none=True)


def test_dynamic_capture_exposes_grouped_prefix_state_without_changing_model():
    layer = _wrapper(SimpleNamespace())
    prefix = torch.nn.Linear(2, 2)
    prefix.register_buffer("routing_table", torch.arange(4))
    object.__setattr__(layer, '_get_submodules_under_cudagraphs', lambda: [layer, prefix])
    before = tuple(layer.state_dict())
    capture = _DynamicCPCaptureCallable(layer, lambda context: None, (1, object()))
    assert tuple(layer.state_dict()) == before
    assert prefix in set(capture.modules())
    assert {id(p) for p in prefix.parameters()} <= {id(p) for p in capture.parameters()}
    assert any(buffer is prefix.routing_table for buffer in capture.buffers())


def test_hybrid_rope_bound_rejects_real_truncation():
    model = SimpleNamespace(config=SimpleNamespace(_cuda_graph_thd_rotary_seq_lens={2: 128}))
    cu = torch.tensor([0, 64, 128], dtype=torch.int32)
    params = PackedSeqParams(qkv_format="thd", local_cp_size=2, cu_seqlens_q=cu, cu_seqlens_kv=cu)
    assert HybridModel._bound_thd_rotary_seq_len(model, 256, params) == 128
    params.cu_seqlens_kv = torch.tensor([0, 256], dtype=torch.int32)
    with pytest.raises(ValueError, match="exceeds the captured RoPE"):
        HybridModel._bound_thd_rotary_seq_len(model, 256, params)


@pytest.mark.internal
@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("hash_routing", [False, True])
def test_hybrid_multi_cp_capture_replay_matches_eager(grouped, hash_routing, monkeypatch):
    """Real mHC + attention + MoE F/B, multiple CP banks and repeated slot wrap.

    This is a layer integration test, not a PP/VPP training accuracy benchmark.
    The alltoall expert dispatcher stays eager; attention and routing are graphed.
    """
    Utils.initialize_model_parallel(context_parallel_size=1, dynamic_context_parallel=True)
    helper = None
    try:
        config = TransformerConfig(
            num_layers=2,
            hidden_size=256,
            num_attention_heads=8,
            ffn_hidden_size=512,
            num_moe_experts=4,
            moe_ffn_hidden_size=128,
            moe_router_topk=2,
            moe_grouped_gemm=True,
            moe_token_dispatcher_type="alltoall",
            moe_router_load_balancing_type="aux_loss",
            moe_aux_loss_coeff=0.01,
            moe_n_hash_layers=2 if hash_routing else 0,
            actual_vocab_size=128,
            bf16=True,
            params_dtype=torch.bfloat16,
            use_cpu_initialization=True,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            gradient_accumulation_fusion=False,
            add_bias_linear=False,
            enable_hyper_connections=True,
            num_residual_streams=4,
            is_hybrid_model=True,
            dynamic_context_parallel=True,
            sequence_packing_scheduler="default_dynamic_cp",
            max_seqlen_per_dp_cp_rank=64,
            pad_packed_seq_alignment=64,
            thd_max_packed_sequences=2,
            cuda_graph_impl="transformer_engine",
            cuda_graph_dynamic_microbatches=True,
            cuda_graph_modules=[CudaGraphModule.attn, CudaGraphModule.moe_router],
        )
        initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
        torch.manual_seed(123)
        model_parallel_cuda_manual_seed(123)
        groups = ProcessGroupCollection.use_mpu_process_groups()
        destroy_num_microbatches_calculator()
        init_num_microbatches_calculator(
            rank=torch.distributed.get_rank(),
            global_batch_size=groups.dp.size(),
            micro_batch_size=1,
            data_parallel_size=groups.dp.size(),
        )
        block = HybridStack(
            config,
            hybrid_stack_spec.submodules,
            layer_type_list=validate_segment_layers("*E"),
            pp_layer_offset=0,
            pg_collection=groups,
        ).cuda()
        if not grouped:
            monkeypatch.setattr(
                HyperConnectionHybridLayer,
                '_can_group_te_cuda_graph_with',
                lambda self, next_layer: False,
            )
        chunk = _GraphTestChunk()
        chunk.config, chunk.decoder = config, block
        chunk.position_embedding_type = "rope"
        chunk.rotary_pos_emb = RotaryEmbedding(
            kv_channels=32, rotary_percent=1.0, cp_group=groups.cp
        )
        getter = parallel_state.get_dynamic_data_context_parallel_groups
        helper = TECudaGraphHelper(
            [chunk],
            config,
            seq_length=64 * groups.dp_cp.size(),
            micro_batch_size=1,
            pg_collection=groups,
            dynamic_cp_group_getter=getter,
        )
        assert len(helper.flattened_callables) == (1 if grouped else 2)
        helper.create_cudagraphs()
        assert helper.graphs_created()
        sizes = tuple(sorted(helper.flattened_callables[0].cuda_graphs_by_dynamic_cp_size))
        assert len(sizes) >= 2
        assert helper.num_microbatches == 1

        for index, size in enumerate((*sizes, *reversed(sizes), *sizes)):
            inputs = torch.randn(64, 1, 1024, dtype=torch.bfloat16, device="cuda")
            cu = torch.tensor([0, 64 * size, 64 * size], dtype=torch.int32, device="cuda")
            params = PackedSeqParams(
                qkv_format="thd",
                cu_seqlens_q=cu,
                cu_seqlens_kv=cu,
                cu_seqlens_q_padded=cu,
                cu_seqlens_kv_padded=cu,
                max_seqlen_q=64 * size,
                max_seqlen_kv=64 * size,
                local_cp_size=size,
                cp_group=getter(group_size=size),
                pad_between_seqs=True,
            )
            kwargs = dict(
                attention_mask=None,
                packed_seq_params=params,
                padding_mask=torch.zeros(1, 64, dtype=torch.bool, device="cuda"),
            )
            kwargs["rotary_pos_emb"] = chunk.rotary_pos_emb(
                64 * size, packed_seq=True, cp_group=params.cp_group
            )
            if hash_routing:
                kwargs["input_ids"] = torch.arange(64, device="cuda").view(1, 64)
            values = []
            for eager in (True, False):
                chunk.zero_grad_buffer()
                hidden = inputs.clone().requires_grad_()
                if eager:
                    output = hidden
                    for layer in block.layers:
                        output, _ = layer.forward(output, **kwargs)
                else:
                    output = hidden
                    for layer in helper.flattened_callables:
                        layer.current_microbatch = index
                        output, _ = layer(output, **kwargs)
                output.float().square().mean().backward()
                values.append(
                    (
                        output.detach().clone(),
                        hidden.grad.clone(),
                        {
                            name: parameter.grad.clone()
                            for name, parameter in chunk.named_parameters()
                            if parameter.grad is not None
                        },
                    )
                )
            torch.testing.assert_close(values[1][0], values[0][0], atol=2e-3, rtol=1e-2)
            torch.testing.assert_close(values[1][1], values[0][1], atol=1e-6, rtol=1e-2)
            assert values[0][2].keys() == values[1][2].keys()
            for name in values[0][2]:
                torch.testing.assert_close(
                    values[1][2][name], values[0][2][name], atol=1e-6, rtol=1e-2, msg=name
                )
        print(
            f"HYBRID_DCP_REPLAY_PASS grouped={grouped} hash={hash_routing} "
            f"cp_sizes={sizes} physical_slots=1 replay_microbatches={index + 1}"
        )
    finally:
        if helper is not None:
            helper.delete_cuda_graphs()
        destroy_num_microbatches_calculator()
        Utils.destroy_model_parallel()
