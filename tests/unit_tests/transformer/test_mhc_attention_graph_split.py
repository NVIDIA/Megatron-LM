# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import sys
from copy import deepcopy
from dataclasses import replace
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from megatron.core.models.hybrid.hybrid_block import HyperConnectionHybridLayer
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.enums import CudaGraphModule
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.module import GraphableMegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import (
    HyperConnectionTransformerLayer,
    TransformerLayer,
)


@pytest.fixture
def hybrid_split_wrapper():
    """A CPU wrapper with only the inner compute and graph transport replaced."""
    config = TransformerConfig(
        num_layers=2,
        hidden_size=4,
        num_attention_heads=1,
        is_hybrid_model=True,
        enable_hyper_connections=True,
        num_residual_streams=2,
        recompute_granularity="selective",
        recompute_modules=["mhc"],
        cuda_graph_impl="transformer_engine",
        cuda_graph_modules=[CudaGraphModule.attn],
        mhc_recompute_attn_cuda_graph_split=True,
        hidden_dropout=0.0,
    )
    inner = TransformerLayer.__new__(TransformerLayer)
    torch.nn.Module.__init__(inner)
    inner.config = config
    inner.layer_number = 1
    inner.hidden_dropout = config.hidden_dropout
    inner.is_moe_layer = False
    inner.input_layernorm = torch.nn.Linear(4, 4)
    inner.self_attention = torch.nn.Linear(4, 4)
    inner.cross_attention = IdentityOp()
    inner.mlp = IdentityOp()

    wrapper = HyperConnectionHybridLayer.__new__(HyperConnectionHybridLayer)
    torch.nn.Module.__init__(wrapper)
    wrapper.config = config
    wrapper.inner_layer = inner
    wrapper.layer_number = 1
    wrapper.hyper_connection = Mock()
    return wrapper


class TestHybridMHCAttentionGraphSplit:
    @pytest.mark.parametrize("split", [False, True])
    def test_static_input_matches_capture_boundary(self, hybrid_split_wrapper, split):
        wrapper = hybrid_split_wrapper
        wrapper.config.mhc_recompute_attn_cuda_graph_split = split
        mask = torch.zeros(1, 1, 3, 3, dtype=torch.bool)
        wrapper.inner_layer.get_layer_static_inputs = Mock(
            return_value={"hidden_states": torch.ones(3, 1, 4), "attention_mask": mask}
        )
        inputs = wrapper.get_layer_static_inputs(3, 1)
        assert inputs["hidden_states"].shape == (3, 1, 4 if split else 8)
        assert inputs["attention_mask"] is mask

    def test_mlp_wrapper_stays_eager(self, hybrid_split_wrapper):
        from megatron.core.transformer.cuda_graphs import _layer_is_graphable

        wrapper = hybrid_split_wrapper
        wrapper.inner_layer.self_attention = IdentityOp()
        wrapper.inner_layer.mlp = torch.nn.Linear(4, 4)
        assert not wrapper._uses_mhc_recompute_attn_cuda_graph_split()
        assert not _layer_is_graphable(wrapper, wrapper.config)

    @pytest.mark.parametrize("mla_recompute", [False, True])
    def test_split_keeps_static_input_liveness_validation(
        self, hybrid_split_wrapper, mla_recompute
    ):
        from megatron.core.transformer.cuda_graphs import TECudaGraphHelper

        helper = object.__new__(TECudaGraphHelper)
        helper.config = replace(
            hybrid_split_wrapper.config,
            multi_latent_attention=True,
            recompute_modules=["mhc", "mla_up_proj"] if mla_recompute else ["mhc"],
        )
        shared = torch.ones(3, 1, 4)
        samples = [(shared,), (shared.detach(),)]
        helper._mhc_sample_order_intervals = {0: [0, 1], 1: [2, 3]}
        helper._validate_mhc_static_hidden_inputs(samples)
        helper._mhc_sample_order_intervals = {0: [0, 2], 1: [1, 3]}
        with pytest.raises(RuntimeError, match="windows overlap"):
            helper._validate_mhc_static_hidden_inputs(samples)

    @pytest.mark.parametrize("branch", ["mlp", "cross_attention"])
    def test_split_rejects_compound_inner_layer(self, hybrid_split_wrapper, branch):
        wrapper = hybrid_split_wrapper
        setattr(wrapper.inner_layer, branch, torch.nn.Linear(4, 4))
        with pytest.raises(ValueError, match="attention-only"):
            HyperConnectionHybridLayer(wrapper.config, wrapper.inner_layer)

    def test_only_inner_attention_owns_graph_hooks(self, hybrid_split_wrapper):
        wrapper = hybrid_split_wrapper
        assert wrapper._get_submodules_under_cudagraphs() == [
            wrapper.inner_layer.input_layernorm,
            wrapper.inner_layer.self_attention,
        ]

    @pytest.mark.parametrize("with_bias", [False, True])
    def test_capture_returns_raw_branch_without_mhc(self, hybrid_split_wrapper, with_bias):
        wrapper = hybrid_split_wrapper
        hidden = torch.randn(3, 1, 4)
        output = torch.randn_like(hidden)
        bias = torch.randn(4) if with_bias else None
        rotary = torch.randn(3, 1, 1, 4)
        wrapper.inner_layer._forward_self_attention_output_with_bias = Mock(
            return_value=((output, bias), None, hidden)
        )
        result = wrapper._te_cuda_graph_capture_impl(hidden, rotary_pos_emb=rotary)
        assert result[0] is output
        assert len(result) == (2 if with_bias else 1)
        if with_bias:
            assert result[1] is bias
        wrapper.hyper_connection.assert_not_called()
        call = wrapper.inner_layer._forward_self_attention_output_with_bias.call_args.kwargs
        assert call["hidden_states"] is hidden
        assert call["rotary_pos_emb"] is rotary
        assert "mhc_recompute_manager" not in call

    def test_replay_uses_inner_attention_argument_normalization(self, hybrid_split_wrapper):
        wrapper = hybrid_split_wrapper
        expected = ((torch.ones(3, 1, 4),), {"attention_mask": torch.zeros(1)})
        wrapper.inner_layer._get_te_cuda_graph_replay_args = Mock(return_value=expected)
        result = wrapper._get_te_cuda_graph_replay_args(torch.ones(3, 1, 4), attention_mask=None)
        assert result is expected

    @pytest.mark.parametrize("with_manager", [False, True])
    def test_replay_direct_writes_each_microbatch_slot(
        self, hybrid_split_wrapper, monkeypatch, with_manager
    ):
        wrapper = hybrid_split_wrapper
        slots = [torch.empty(3, 1, 4) for _ in range(2)]
        graph_reads = []

        def producer(hidden, *, mhc_recompute_manager, output_slot):
            aggregate = hidden[..., :4].clone()
            if with_manager:
                assert mhc_recompute_manager is wrapper._mhc_recompute_manager
                aggregate = output_slot.writer.copy_(aggregate)
            else:
                assert output_slot is None
            return aggregate, "h_res", "h_post", hidden

        def replay(_self, aggregate, **kwargs):
            if with_manager:
                assert aggregate.data_ptr() == slots[wrapper.current_microbatch].data_ptr()
            assert kwargs["attention_mask"] is None
            torch.testing.assert_close(kwargs["padding_mask"], torch.zeros(1, 3))
            assert set(kwargs) == {"attention_mask", "padding_mask"}
            graph_reads.append(aggregate.clone())
            return (aggregate * 2,)

        wrapper.hyper_connection.side_effect = producer
        wrapper.hyper_connection.fused_h_res_h_post_bda.side_effect = (
            lambda h_res, residual, h_post, output, **kwargs: residual + output[0].repeat(1, 1, 2)
        )
        monkeypatch.setattr(GraphableMegatronModule, "_te_cuda_graph_replay", replay)
        for microbatch in range(2):
            wrapper.current_microbatch = microbatch
            slot = SimpleNamespace(writer=slots[microbatch])
            manager = SimpleNamespace(
                mhc_arena=SimpleNamespace(bind_external_slot=Mock(return_value=slot)),
                is_last_layer_in_recompute_block=False,
            )
            wrapper._mhc_recompute_manager = manager if with_manager else None
            wrapper.get_te_cuda_graph_static_hidden_input = Mock(return_value=slots[microbatch])
            hidden = torch.full((3, 1, 8), float(microbatch + 1))
            output, context = wrapper._te_cuda_graph_replay(
                hidden_states=hidden, attention_mask=None, padding_mask=torch.zeros(1, 3)
            )
            torch.testing.assert_close(output, hidden * 3)
            assert context is None
            if with_manager:
                manager.mhc_arena.bind_external_slot.assert_called_once_with(
                    ("attention", 1, "aggregate", 0), slots[microbatch]
                )
            else:
                wrapper.get_te_cuda_graph_static_hidden_input.assert_not_called()
        if with_manager:
            for slot, expected in zip(slots, graph_reads):
                torch.testing.assert_close(slot, expected)

    @pytest.mark.parametrize("is_group_end", [False, True])
    def test_post_preserves_group_boundary_and_residual_dtype(
        self, hybrid_split_wrapper, is_group_end
    ):
        wrapper = hybrid_split_wrapper
        wrapper.config.fp32_residual_connection = True
        wrapper.config.params_dtype = torch.bfloat16
        wrapper.hyper_connection.fused_h_res_h_post_bda.return_value = torch.ones(3, 1, 8)
        aggregate = torch.ones(3, 1, 4)
        manager = SimpleNamespace(is_last_layer_in_recompute_block=is_group_end)
        result = wrapper._forward_mhc_post(
            aggregate,
            "h_res",
            "h_post",
            torch.ones(3, 1, 8),
            (aggregate, None),
            0.0,
            False,
            manager,
        )
        assert result.dtype == torch.bfloat16
        call = wrapper.hyper_connection.fused_h_res_h_post_bda.call_args.kwargs
        assert call["manager"] is (None if is_group_end else manager)


@pytest.fixture(params=["gpt", "hybrid"])
def packed_split_layer(hybrid_split_wrapper, request, monkeypatch):
    """Real packed graph boundaries with CPU-only branch compute and transport."""
    import megatron.core.transformer.transformer_config as config_module

    monkeypatch.setattr(config_module, "is_te_min_version", lambda _version: True)
    config = replace(
        hybrid_split_wrapper.config,
        is_hybrid_model=request.param == "hybrid",
        sequence_packing_scheduler="dp_balanced",
        max_seqlen_per_dp_cp_rank=8,
        thd_max_packed_sequences=2,
        pad_packed_seq_alignment="max",
    )
    aggregate = torch.ones(8, 1, 4)
    residual = torch.ones(8, 1, 8)
    producer = Mock(return_value=(aggregate, "h_res", "h_post", residual))
    if request.param == "hybrid":
        layer = hybrid_split_wrapper
        layer.inner_layer.config = config
        attention = Mock(return_value=((aggregate * 2, None), None, aggregate))
        layer.inner_layer._forward_self_attention_output_with_bias = attention
        layer.hyper_connection = producer
        layer._forward_mhc_post = Mock(return_value=residual)
        layer._offload_module_in_cuda_graph_cached = False
    else:
        layer = HyperConnectionTransformerLayer.__new__(HyperConnectionTransformerLayer)
        torch.nn.Module.__init__(layer)
        layer.layer_number = 1
        layer.is_moe_layer = False
        layer.input_layernorm = torch.nn.Identity()
        attention = Mock(return_value=(aggregate * 2, None))
        layer.self_attention = attention
        layer.self_attention_hyper_connection = producer
        layer._forward_mhc_attention_post_cuda_graph = Mock(return_value=(residual, None))
        layer._forward_mlp = Mock(return_value=residual)
    layer.config = config
    layer.off_interface = Mock()
    return layer, producer, attention


def _cpu_packed_metadata(boundary=3):
    cu = torch.tensor([0, boundary, 8], dtype=torch.int32)
    return PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu.clone(),
        cu_seqlens_q_padded=cu.clone(),
        cu_seqlens_kv_padded=cu.clone(),
        max_seqlen_q=max(boundary, 8 - boundary),
        max_seqlen_kv=max(boundary, 8 - boundary),
        tokens_per_sample=4,
    )


class TestMHCAttentionGraphSplitTHD:
    def test_static_input_keeps_thd_metadata_with_one_stream(self, packed_split_layer, monkeypatch):
        layer, _, _ = packed_split_layer
        monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
        inputs = layer.get_layer_static_inputs(8, 1)
        assert inputs["hidden_states"].shape == (8, 1, 4)
        assert inputs["padding_mask"].shape == (1, 8)
        assert "attention_mask" not in inputs
        for name in (
            "cu_seqlens_q",
            "cu_seqlens_kv",
            "cu_seqlens_q_padded",
            "cu_seqlens_kv_padded",
        ):
            assert inputs[name].shape == (3,)
            assert inputs[name].dtype == torch.int32

    def test_replay_does_not_allocate_a_dense_thd_attention_mask(
        self, packed_split_layer, monkeypatch
    ):
        import megatron.core.transformer.transformer_layer as layer_module

        layer, _, _ = packed_split_layer
        te = ModuleType("transformer_engine")
        te.pytorch = ModuleType("transformer_engine.pytorch")
        monkeypatch.setitem(sys.modules, "transformer_engine", te)
        monkeypatch.setitem(sys.modules, "transformer_engine.pytorch", te.pytorch)
        monkeypatch.setattr(layer_module, "is_te_min_version", lambda _version: True)
        monkeypatch.setattr(
            torch, "zeros", lambda *args, **kwargs: pytest.fail("THD replay allocated a dense mask")
        )
        hidden = torch.ones(8, 1, 4)
        cu = _cpu_packed_metadata().cu_seqlens_q
        args, kwargs = layer._get_te_cuda_graph_replay_args(
            hidden, attention_mask=None, cu_seqlens_q=cu
        )
        assert args[0] is hidden
        assert "attention_mask" not in kwargs
        assert kwargs["cu_seqlens_q"] is cu

    def test_capture_reconstructs_packed_attention_metadata(self, packed_split_layer):
        layer, producer, attention = packed_split_layer
        packed = _cpu_packed_metadata()
        kwargs = {"packed_seq_params": packed, "padding_mask": torch.zeros(1, 8, dtype=torch.bool)}
        layer._decompose_packed_seq_params_to_kwargs(kwargs)
        result = layer._te_cuda_graph_capture(torch.ones(8, 1, 4), **kwargs)
        assert len(result) == 1
        captured = attention.call_args.kwargs["packed_seq_params"]
        assert captured.qkv_format == "thd"
        assert captured.pad_between_seqs is True
        assert captured.max_seqlen_q == captured.max_seqlen_kv == 8
        for name in (
            "cu_seqlens_q",
            "cu_seqlens_kv",
            "cu_seqlens_q_padded",
            "cu_seqlens_kv_padded",
        ):
            assert getattr(captured, name) is getattr(packed, name)
        producer.assert_not_called()

    def test_replay_forwards_packed_kwargs_and_preserves_eager_metadata(
        self, packed_split_layer, monkeypatch
    ):
        layer, producer, _ = packed_split_layer
        graph_reads = []

        def replay(_self, aggregate, **kwargs):
            assert "packed_seq_params" not in kwargs
            assert all(isinstance(value, torch.Tensor) for value in kwargs.values())
            assert set(kwargs) == set(expected)
            for name, value in expected.items():
                assert kwargs[name] is value
            graph_reads.append(kwargs["cu_seqlens_q"].clone())
            return (aggregate * 2,)

        monkeypatch.setattr(GraphableMegatronModule, "_te_cuda_graph_replay", replay)
        for boundary in (3, 5):
            packed = _cpu_packed_metadata(boundary)
            mask = torch.tensor([[False] * 7 + [True]])
            kwargs = {"packed_seq_params": packed, "padding_mask": mask}
            expected = {"padding_mask": mask}
            for name in (
                "cu_seqlens_q",
                "cu_seqlens_kv",
                "cu_seqlens_q_padded",
                "cu_seqlens_kv_padded",
            ):
                expected[name] = getattr(packed, name)
            if isinstance(layer, HyperConnectionTransformerLayer):
                kwargs["input_ids"] = torch.arange(8).reshape(1, 8)
                expected["input_ids"] = kwargs["input_ids"]
            _, context = layer._te_cuda_graph_replay(torch.ones(8, 1, 8), **kwargs)
            assert context is None
            assert kwargs["packed_seq_params"] is packed
            if isinstance(layer, HyperConnectionTransformerLayer):
                eager = layer._forward_mlp.call_args.kwargs
                assert eager["packed_seq_params"] is packed
                assert eager["packed_seq_params"].tokens_per_sample == 4
                assert eager["padding_mask"] is mask
                assert eager["input_ids"] is kwargs["input_ids"]
        assert producer.call_count == 2
        assert graph_reads[0].tolist() == [0, 3, 8]
        assert graph_reads[1].tolist() == [0, 5, 8]

    def test_packed_route_slot_is_resolved_before_mhc_direct_write(
        self, packed_split_layer, monkeypatch
    ):
        layer, producer, _ = packed_split_layer
        slots = (torch.empty(8, 1, 4), torch.empty(8, 1, 4))
        layer.cuda_graphs = [object(), object()]
        layer._te_cuda_graph_static_hidden_inputs = slots
        layer._te_cuda_graph_static_hidden_input_ptrs = tuple(slot.data_ptr() for slot in slots)
        layer.current_microbatch = 1
        arena_slot = SimpleNamespace(writer=slots[0])
        layer._mhc_recompute_manager = SimpleNamespace(
            mhc_arena=SimpleNamespace(bind_external_slot=Mock(return_value=arena_slot))
        )
        decompose = layer._decompose_packed_seq_params_to_kwargs

        def decompose_retained_invocation(kwargs):
            decompose(kwargs)
            layer._te_cuda_graph_route_replay_state = (0, 0)

        def produce(hidden, *, mhc_recompute_manager, output_slot):
            assert output_slot is arena_slot
            assert layer._te_cuda_graph_route_replay_state == (0, 0)
            return output_slot.writer.copy_(hidden[..., :4]), "h_res", "h_post", hidden

        def replay(_self, aggregate, **kwargs):
            assert aggregate.data_ptr() == slots[0].data_ptr()
            assert layer._te_cuda_graph_route_replay_state == (0, 0)
            return (aggregate * 2,)

        producer.side_effect = produce
        layer._decompose_packed_seq_params_to_kwargs = decompose_retained_invocation
        monkeypatch.setattr(GraphableMegatronModule, "_te_cuda_graph_replay", replay)
        layer._te_cuda_graph_replay(
            torch.ones(8, 1, 8),
            packed_seq_params=_cpu_packed_metadata(),
            padding_mask=torch.zeros(1, 8, dtype=torch.bool),
        )
        layer._mhc_recompute_manager.mhc_arena.bind_external_slot.assert_called_once_with(
            ("attention", 1, "aggregate", 0), slots[0]
        )
        assert layer._te_cuda_graph_route_replay_state is None
        assert layer.get_te_cuda_graph_static_hidden_input() is slots[1]
        layer._te_cuda_graph_route_replay_state = (0, 0)
        assert layer.get_te_cuda_graph_static_hidden_input() is slots[0]
        assert layer.get_te_cuda_graph_static_hidden_input(microbatch_idx=1) is slots[1]

    @pytest.mark.parametrize("fail", [False, True])
    def test_gpt_split_preserves_offload_lifecycle(self, hybrid_split_wrapper, monkeypatch, fail):
        layer = HyperConnectionTransformerLayer.__new__(HyperConnectionTransformerLayer)
        torch.nn.Module.__init__(layer)
        layer.config = hybrid_split_wrapper.config
        layer.config.delay_offload_until_cuda_graph = True
        layer.off_interface = Mock()
        packed = _cpu_packed_metadata()

        def replay_impl(args, kwargs, context):
            assert kwargs["packed_seq_params"] is packed
            if fail:
                raise RuntimeError("replay failed")
            return args[0], context

        monkeypatch.setattr(layer, "_te_cuda_graph_replay_impl", replay_impl)
        if fail:
            with pytest.raises(RuntimeError, match="replay failed"):
                layer._te_cuda_graph_replay(torch.ones(8, 1, 8), packed_seq_params=packed)
        else:
            layer._te_cuda_graph_replay(torch.ones(8, 1, 8), packed_seq_params=packed)
        layer.off_interface.enter_replay.assert_called_once_with()
        layer.off_interface.exit_replay.assert_called_once_with()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph regression requires a GPU")
@pytest.mark.parametrize("hybrid", [False, True])
@pytest.mark.parametrize("mla_recompute", [False, True])
@pytest.mark.parametrize("packed", [False, True])
def test_gpu_mhc_split_replay_matches_eager(hybrid, mla_recompute, packed):
    """Exercise real MLA graphs with two in-flight microbatches and fresh replay inputs."""
    te_graph = pytest.importorskip("transformer_engine.pytorch.graph")
    from megatron.core.models.gpt.gpt_layer_specs import (
        get_gpt_layer_with_transformer_engine_submodules,
    )
    from megatron.core.tensor_parallel.random import (
        MHCCheckpointManager,
        initialize_rng_tracker,
        model_parallel_cuda_manual_seed,
    )
    from megatron.core.transformer.cuda_graphs import _set_capture_end, _set_capture_start
    from megatron.core.transformer.enums import AttnBackend
    from megatron.core.transformer.transformer_config import MLATransformerConfig
    from megatron.core.utils import is_te_min_version
    from tests.unit_tests.test_utilities import Utils

    if not is_te_min_version("2.10.0"):
        pytest.skip("Partial CUDA graph regression requires Transformer Engine >= 2.10.0")
    Utils.initialize_model_parallel()
    initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
    model_parallel_cuda_manual_seed(123)
    config = MLATransformerConfig(
        num_layers=4 if hybrid else 2,
        hidden_size=64,
        num_attention_heads=4,
        q_lora_rank=16,
        kv_lora_rank=16,
        qk_head_dim=32,
        qk_pos_emb_head_dim=16,
        v_head_dim=32,
        multi_latent_attention=True,
        is_hybrid_model=hybrid,
        enable_hyper_connections=True,
        num_residual_streams=2,
        recompute_granularity="selective",
        recompute_modules=["mhc", "mla_up_proj"] if mla_recompute else ["mhc"],
        cuda_graph_impl="transformer_engine",
        cuda_graph_modules=[CudaGraphModule.attn],
        mhc_recompute_attn_cuda_graph_split=True,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        attention_backend=AttnBackend.fused if packed else AttnBackend.unfused,
        bf16=packed,
        params_dtype=torch.bfloat16 if packed else torch.float32,
        create_attention_mask_in_dataloader=False,
        gradient_accumulation_fusion=False,
        use_cpu_initialization=True,
        use_te_rng_tracker=True,
        bias_dropout_fusion=False,
        sequence_packing_scheduler="dp_balanced" if packed else None,
        max_seqlen_per_dp_cp_rank=64 if packed else None,
        thd_max_packed_sequences=2 if packed else None,
        pad_packed_seq_alignment="max" if packed else None,
    )

    def build_layers(layer_config):
        submodules = get_gpt_layer_with_transformer_engine_submodules(
            multi_latent_attention=True, enable_hyper_connection=not hybrid
        )
        layers = torch.nn.ModuleList()
        for index in range(2):
            if hybrid:
                attention_submodules = deepcopy(submodules)
                attention_submodules.mlp = IdentityOp
                attention_submodules.pre_mlp_layernorm = IdentityOp
                mlp_submodules = deepcopy(submodules)
                mlp_submodules.self_attention = IdentityOp
                mlp_submodules.input_layernorm = IdentityOp
                for offset, spec in enumerate((attention_submodules, mlp_submodules)):
                    inner = TransformerLayer(
                        layer_config, spec, layer_number=2 * index + offset + 1
                    )
                    layers.append(HyperConnectionHybridLayer(layer_config, inner))
            else:
                layers.append(
                    HyperConnectionTransformerLayer(
                        layer_config, submodules, layer_number=index + 1
                    )
                )
        return layers.cuda()

    try:
        reference = build_layers(
            replace(
                config,
                cuda_graph_impl="none",
                cuda_graph_modules=[],
                mhc_recompute_attn_cuda_graph_split=False,
            )
        )
        graphed = build_layers(config)
        graphed.load_state_dict(reference.state_dict())
        callables = list(graphed[::2]) if hybrid else list(graphed)
        seq_length = 64 if packed else 8
        sample_kwargs = [
            layer.get_layer_static_inputs(seq_length, 1) for _ in range(2) for layer in callables
        ]
        sample_args = [(kwargs.pop("hidden_states"),) for kwargs in sample_kwargs]
        _set_capture_start()
        try:
            graphs = te_graph.make_graphed_callables(
                tuple(callables),
                sample_args,
                sample_kwargs=sample_kwargs,
                num_warmup_iters=3,
                allow_unused_input=True,
                _order=[1, 1, -1, -1],
                _num_layers_per_chunk=[len(callables)],
            )
        finally:
            _set_capture_end()
        for layer_index, layer in enumerate(callables):
            indices = [slot * len(callables) + layer_index for slot in range(2)]
            layer.cuda_graphs = [graphs[index] for index in indices]
            layer.set_te_cuda_graph_static_hidden_inputs(
                [sample_args[index][0] for index in indices]
            )

        def run(layers, data, grad_outputs, packed_inputs):
            layers.zero_grad(set_to_none=True)
            inputs = [value.detach().clone().requires_grad_() for value in data]
            outputs = []
            for microbatch, hidden in enumerate(inputs):
                manager = MHCCheckpointManager()
                for layer_index, layer in enumerate(layers):
                    layer.current_microbatch = microbatch
                    manager.is_last_layer_in_recompute_block = layer_index == len(layers) - 1
                    hidden, _ = layer(
                        hidden, mhc_recompute_manager=manager, **packed_inputs[microbatch]
                    )
                manager.discard_all_outputs_and_register_unified_recompute(hidden)
                outputs.append(hidden)
            for output, grad in zip(outputs, grad_outputs):
                output.backward(grad)
            return outputs, inputs

        tolerance = dict(atol=1e-2, rtol=1e-2) if packed else dict(atol=1e-5, rtol=1e-4)
        for replay_index in range(3):
            data = [
                torch.randn(seq_length, 1, 128, device="cuda", dtype=config.params_dtype)
                for _ in range(2)
            ]
            grads = [torch.randn_like(value) for value in data]
            packed_inputs = [{}, {}]
            if packed:
                for microbatch in range(2):
                    boundary = 16 * (1 + (replay_index + microbatch) % 3)
                    cu = torch.tensor([0, boundary, 64], dtype=torch.int32, device="cuda")
                    packed_inputs[microbatch] = dict(
                        packed_seq_params=PackedSeqParams(
                            qkv_format="thd",
                            cu_seqlens_q=cu,
                            cu_seqlens_kv=cu.clone(),
                            cu_seqlens_q_padded=cu.clone(),
                            cu_seqlens_kv_padded=cu.clone(),
                            max_seqlen_q=max(boundary, 64 - boundary),
                            max_seqlen_kv=max(boundary, 64 - boundary),
                            pad_between_seqs=True,
                        ),
                        padding_mask=torch.zeros(1, 64, dtype=torch.bool, device="cuda"),
                    )
            expected, expected_inputs = run(reference, data, grads, packed_inputs)
            actual, actual_inputs = run(graphed, data, grads, packed_inputs)
            for got, want in zip(actual, expected):
                torch.testing.assert_close(got, want, **tolerance)
            for got, want in zip(actual_inputs, expected_inputs):
                torch.testing.assert_close(got.grad, want.grad, **tolerance)
            for (name, got), (_, want) in zip(
                graphed.named_parameters(), reference.named_parameters()
            ):
                assert (got.grad is None) == (want.grad is None), name
                if want.grad is not None:
                    torch.testing.assert_close(got.grad, want.grad, msg=name, **tolerance)
    finally:
        Utils.destroy_model_parallel()
