# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from unittest.mock import Mock

import pytest
import torch

import megatron.core.transformer.moe.moe_layer as moe_layer_module
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_layer_local_submodules,
    get_gpt_layer_with_transformer_engine_submodules,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.cuda_graphs import _set_capture_end, _set_capture_start
from megatron.core.transformer.enums import CudaGraphModule
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.moe.moe_utils import drop_padding_routes
from megatron.core.transformer.spec_utils import get_submodules
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from megatron.core.utils import is_te_min_version
from megatron.training.initialize import _set_random_seed
from tests.unit_tests.test_utilities import Utils


class _RecordingMegakernel(torch.nn.Module):
    """Megakernel backend stand-in that records the routes it computes: padding tokens dropped."""

    def __init__(self):
        super().__init__()
        self.routes = []

    def forward(self, hidden_states, probs, routing_map, padding_mask=None):
        if padding_mask is not None:
            probs, routing_map = drop_padding_routes(probs, routing_map, padding_mask)
        self.routes.append((probs.detach().clone(), routing_map.clone()))
        return hidden_states * 2


def _megakernel_moe_layer(monkeypatch, backend, **config_overrides):
    """Build a two-expert MoE layer that runs ``backend`` as its megakernel."""
    config = TransformerConfig(
        num_layers=1,
        hidden_size=128,
        num_attention_heads=4,
        num_moe_experts=2,
        moe_ffn_hidden_size=128,
        moe_shared_expert_intermediate_size=128,
        moe_router_topk=1,
        moe_router_pre_softmax=True,
        moe_grouped_gemm=True,
        gated_linear_unit=True,
        activation_func=torch.nn.functional.silu,
        gradient_accumulation_fusion=True,
        moe_megakernel_backend="mok",
        add_bias_linear=False,
        bf16=True,
        use_cpu_initialization=True,
        **config_overrides,
    )
    submodules = get_submodules(
        get_gpt_layer_with_transformer_engine_submodules(num_experts=2, moe_grouped_gemm=True).mlp
    )
    monkeypatch.setattr(moe_layer_module, "build_megakernel_backend", lambda **kwargs: backend)
    return MoELayer(config, submodules)


class _CapacityCheckingMegakernel(_RecordingMegakernel):
    """Megakernel stand-in with a static per-expert capacity, like MOK's schedule capacity: running
    (not capturing) more routes on one expert than it holds fails, where MOK traps on the GPU."""

    def __init__(self, capacity):
        super().__init__()
        self.capacity = capacity
        self.routing_maps = []

    def forward(self, hidden_states, probs, routing_map, padding_mask=None):
        self.routing_maps.append(routing_map.clone())
        output = super().forward(hidden_states, probs, routing_map, padding_mask)
        if not torch.cuda.is_current_stream_capturing():
            routes_per_expert = self.routes[-1][1].sum(dim=0)
            if int(routes_per_expert.max()) > self.capacity:
                raise RuntimeError(
                    f"schedule capacity {self.capacity} exceeded: {routes_per_expert.tolist()}"
                )
        return output


def _moe_transformer_layer(monkeypatch, backend=None, num_experts=4, **config_overrides):
    """Build a CUDA TransformerLayer whose MLP is a top-1 MoE, without attention (as the MoE layers
    of hybrid models): ``backend`` runs as its megakernel, or the native MoE path when None."""
    if backend is not None:
        config_overrides["moe_megakernel_backend"] = "mok"
        monkeypatch.setattr(moe_layer_module, "build_megakernel_backend", lambda **kwargs: backend)
    config = TransformerConfig(
        num_layers=1,
        hidden_size=128,
        num_attention_heads=4,
        num_moe_experts=num_experts,
        moe_ffn_hidden_size=128,
        moe_shared_expert_intermediate_size=128,
        moe_router_topk=1,
        moe_router_pre_softmax=True,
        moe_grouped_gemm=True,
        gated_linear_unit=True,
        activation_func=torch.nn.functional.silu,
        gradient_accumulation_fusion=True,
        normalization="RMSNorm",
        add_bias_linear=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        use_cpu_initialization=True,
        **config_overrides,
    )
    layer_submodules = get_gpt_layer_with_transformer_engine_submodules(
        num_experts=num_experts, moe_grouped_gemm=True
    )
    submodules = TransformerLayerSubmodules(
        pre_mlp_layernorm=layer_submodules.pre_mlp_layernorm,
        mlp=layer_submodules.mlp,
        mlp_bda=layer_submodules.mlp_bda,
    )
    return TransformerLayer(config, submodules).cuda()


class TestMoELayerInit:

    @pytest.mark.skipif(
        not is_te_min_version("1.7.0.dev0"),
        reason="Expert with TE Linear is only supported in TE 1.7.0 and later.",
    )
    def test_megakernel_skips_native_token_dispatcher_construction(self, monkeypatch):
        Utils.initialize_model_parallel(1, 1)
        try:
            config = TransformerConfig(
                num_layers=1,
                hidden_size=128,
                num_attention_heads=4,
                num_moe_experts=1,
                moe_ffn_hidden_size=128,
                moe_shared_expert_intermediate_size=128,
                moe_router_topk=1,
                moe_router_pre_softmax=True,
                moe_grouped_gemm=True,
                gated_linear_unit=True,
                activation_func=torch.nn.functional.silu,
                gradient_accumulation_fusion=True,
                moe_megakernel_backend="mok",
                add_bias_linear=False,
                bf16=True,
                use_cpu_initialization=True,
            )
            submodules = get_submodules(
                get_gpt_layer_with_transformer_engine_submodules(
                    num_experts=1, moe_grouped_gemm=True
                ).mlp
            )
            backend = torch.nn.Identity()

            def fail_if_constructed(*args, **kwargs):
                del args, kwargs
                pytest.fail("megakernel path must not construct a native token dispatcher")

            monkeypatch.setattr(
                moe_layer_module, "MoEAllGatherTokenDispatcher", fail_if_constructed
            )
            monkeypatch.setattr(
                moe_layer_module, "build_megakernel_backend", lambda **kwargs: backend
            )

            layer = MoELayer(config, submodules)

            assert layer.token_dispatcher is None
            assert layer.megakernel_experts is backend

            hidden = torch.zeros((3, 2, config.hidden_size), dtype=torch.bfloat16)
            padding_mask = torch.tensor([[0, 0, 1], [0, 1, 1]])
            input_ids = torch.arange(6).view(2, 3)
            probs = torch.ones((6, 1))
            routing_map = torch.ones((6, 1), dtype=torch.bool)
            router_forward = Mock(return_value=(probs, routing_map))
            backend_forward = Mock(return_value=hidden)
            monkeypatch.setattr(layer.router, "forward", router_forward)
            monkeypatch.setattr(backend, "forward", backend_forward)

            output, bias = layer(hidden, padding_mask=padding_mask, input_ids=input_ids)

            assert output is hidden and bias is None
            router_forward.assert_called_once()
            assert router_forward.call_args.args[0] is hidden
            torch.testing.assert_close(router_forward.call_args.args[1], padding_mask.T.bool())
            assert router_forward.call_args.kwargs["input_ids"] is input_ids
            # The backend drops the padding routes itself, from the batch-first mask.
            backend_forward.assert_called_once_with(hidden, probs, routing_map, padding_mask)
        finally:
            Utils.destroy_model_parallel()

    @pytest.mark.skipif(
        not is_te_min_version("1.7.0.dev0"),
        reason="Expert with TE Linear is only supported in TE 1.7.0 and later.",
    )
    @pytest.mark.parametrize("path", ["eager", "recompute", "moe_router_graph_replay"])
    def test_megakernel_receives_no_padding_routes(self, monkeypatch, path):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        try:
            config_overrides = {
                "eager": {},
                "recompute": {"recompute_granularity": "selective", "recompute_modules": ["moe"]},
                "moe_router_graph_replay": {
                    "cuda_graph_impl": "transformer_engine",
                    "cuda_graph_modules": [CudaGraphModule.attn, CudaGraphModule.moe_router],
                },
            }[path]
            backend = _RecordingMegakernel()
            layer = _megakernel_moe_layer(monkeypatch, backend, **config_overrides)

            hidden = torch.randn((3, 2, 128), dtype=torch.bfloat16, requires_grad=True)
            padding_mask = torch.tensor([[False, False, True], [False, True, True]])
            probs = torch.rand((6, 2))
            routing_map = torch.tensor([[True, False], [False, True]] * 3)
            router_forward = Mock(return_value=(probs, routing_map))
            monkeypatch.setattr(layer.router, "forward", router_forward)
            if path == "moe_router_graph_replay":
                # Replay hands the layer the router outputs captured in its graph, unmasked.
                layer.cudagraph_tensor_store.set(
                    hidden_states=hidden, probs=probs, routing_map=routing_map
                )

            output, bias = layer(hidden, padding_mask=padding_mask)
            output.sum().backward()
            layer.cudagraph_tensor_store.clear()

            assert bias is None
            # Recompute reruns the router and the megakernel in backward; replay skips the router.
            router_calls, backend_calls = {
                "eager": (1, 1),
                "recompute": (2, 2),
                "moe_router_graph_replay": (0, 1),
            }[path]
            assert router_forward.call_count == router_calls
            assert len(backend.routes) == backend_calls
            # Tokens flatten sequence-first, so the padded tokens are rows 3, 4 and 5.
            valid = torch.tensor([[True], [True], [True], [False], [False], [False]])
            for backend_probs, backend_routing_map in backend.routes:
                torch.testing.assert_close(backend_probs, probs * valid)
                torch.testing.assert_close(backend_routing_map, routing_map & valid)
        finally:
            Utils.destroy_model_parallel()

    @pytest.mark.skipif(
        not is_te_min_version("1.7.0.dev0"),
        reason="Expert with TE Linear is only supported in TE 1.7.0 and later.",
    )
    def test_megakernel_moe_router_graph_capture_outputs_unmasked_routes(self, monkeypatch):
        Utils.initialize_model_parallel(1, 1)
        try:
            backend = _RecordingMegakernel()
            layer = _megakernel_moe_layer(
                monkeypatch,
                backend,
                cuda_graph_impl="transformer_engine",
                cuda_graph_modules=[CudaGraphModule.attn, CudaGraphModule.moe_router],
            )
            hidden = torch.zeros((3, 2, 128), dtype=torch.bfloat16)
            padding_mask = torch.tensor([[False, False, True], [False, True, True]])
            probs = torch.rand((6, 2))
            routing_map = torch.tensor([[True, False], [False, True]] * 3)
            monkeypatch.setattr(layer.router, "forward", Mock(return_value=(probs, routing_map)))

            _set_capture_start()
            try:
                graph_outputs = layer(hidden, padding_mask=padding_mask)
            finally:
                _set_capture_end()

            # The graph ends after the router, so the mask is applied at replay, not captured.
            assert graph_outputs[0] is hidden
            assert graph_outputs[1] is probs and graph_outputs[2] is routing_map
            assert backend.routes == []
        finally:
            Utils.destroy_model_parallel()

    @pytest.mark.skipif(
        not is_te_min_version("2.19.0"),
        reason="The megakernel moe CUDA graph scope requires TE 2.19 and later.",
    )
    def test_megakernel_moe_graph_capture_covers_router_and_megakernel(self, monkeypatch):
        Utils.initialize_model_parallel(1, 1)
        try:
            backend = _RecordingMegakernel()
            layer = _megakernel_moe_layer(
                monkeypatch,
                backend,
                cuda_graph_impl="transformer_engine",
                cuda_graph_modules=[CudaGraphModule.attn, CudaGraphModule.moe],
            )
            hidden = torch.randn((3, 2, 128), dtype=torch.bfloat16)
            padding_mask = torch.tensor([[False, False, True], [False, True, True]])
            probs = torch.rand((6, 2))
            routing_map = torch.tensor([[True, False], [False, True]] * 3)
            router_forward = Mock(return_value=(probs, routing_map))
            monkeypatch.setattr(layer.router, "forward", router_forward)

            _set_capture_start()
            try:
                output, bias = layer(hidden, padding_mask=padding_mask)
            finally:
                _set_capture_end()

            # No early return: the graph runs the router, drops the padding routes and runs the
            # megakernel, so its output is the MoE output.
            router_forward.assert_called_once()
            assert bias is None
            torch.testing.assert_close(output, hidden * 2, rtol=0, atol=0)
            valid = torch.tensor([[True], [True], [True], [False], [False], [False]])
            assert len(backend.routes) == 1
            torch.testing.assert_close(backend.routes[0][0], probs * valid)
            torch.testing.assert_close(backend.routes[0][1], routing_map & valid)
        finally:
            Utils.destroy_model_parallel()

    @pytest.mark.skipif(
        not is_te_min_version("2.19.0"),
        reason="The megakernel moe CUDA graph scope requires TE 2.19 and later.",
    )
    def test_megakernel_moe_graph_warmup_on_static_inputs_routes_no_tokens(self, monkeypatch):
        """TE warms the graphed layer up eagerly on get_layer_static_inputs before capturing it.
        Their all-ones hidden states give every token the same router logits, so real routing sends
        all of them to one expert, beyond a megakernel's static capacity (MOK traps). The static
        padding mask of a graphed megakernel marks every token as padding: no routes."""
        Utils.initialize_model_parallel(1, 1)
        torch.manual_seed(123)
        model_parallel_cuda_manual_seed(123)
        try:
            seq_length, micro_batch_size, num_experts = 8, 2, 4
            # Three times the balanced top-1 load: real routing fits, a single-expert batch not.
            backend = _CapacityCheckingMegakernel(
                capacity=3 * seq_length * micro_batch_size // num_experts
            )
            layer = _moe_transformer_layer(
                monkeypatch,
                backend,
                num_experts,
                cuda_graph_impl="transformer_engine",
                cuda_graph_modules=[CudaGraphModule.attn, CudaGraphModule.moe],
            )
            padding_mask = torch.zeros((micro_batch_size, seq_length), dtype=torch.bool).cuda()
            padding_mask[0, -3:] = True
            hidden = torch.randn((seq_length, micro_batch_size, 128), dtype=torch.bfloat16).cuda()

            # Eager cuda_graph_warmup_steps: real routing, and the layer records the mask shape.
            layer(hidden_states=hidden, padding_mask=padding_mask)
            assert backend.routes[-1][1].any()

            # TE's warmup of the capture runs the layer on its static inputs (an all-False mask
            # here would exceed the backend's capacity and raise, as MOK traps).
            static_inputs = layer.get_layer_static_inputs(seq_length, micro_batch_size)
            static_mask = static_inputs["padding_mask"]
            _set_capture_start()
            try:
                layer(static_inputs.pop("hidden_states"), **static_inputs)
            finally:
                _set_capture_end()

            # The router sent every warmup token to one expert, more than the backend holds...
            warmup_routes = backend.routing_maps[-1]
            assert int(warmup_routes.any(dim=0).sum()) == 1
            assert int(warmup_routes.sum(dim=0).max()) > backend.capacity
            # ...but the all-padding static mask left the megakernel no routes to run.
            assert static_mask.shape == padding_mask.shape and bool(static_mask.all())
            assert not backend.routes[-1][1].any()
            # Replays pass the batch's real mask, which TE copies into the static mask buffer.
            assert layer._te_cuda_graph_padding_mask(padding_mask, hidden.device) is padding_mask
        finally:
            Utils.destroy_model_parallel()

    @pytest.mark.skipif(
        not is_te_min_version("2.19.0"),
        reason="The megakernel moe CUDA graph scope requires TE 2.19 and later.",
    )
    @pytest.mark.parametrize(
        "megakernel, cuda_graph_modules, config_overrides, all_padding",
        [
            (True, [CudaGraphModule.attn, CudaGraphModule.moe], {}, True),
            (True, [CudaGraphModule.moe], {}, True),
            # A router-only graph ends before the megakernel, which runs eagerly on the real mask.
            (True, [CudaGraphModule.attn, CudaGraphModule.moe_router], {}, False),
            # Native MoE layers keep the all-False static mask.
            (False, [CudaGraphModule.attn, CudaGraphModule.moe_router], {}, False),
            (
                False,
                [CudaGraphModule.attn, CudaGraphModule.moe],
                {
                    "moe_token_dispatcher_type": "alltoall",
                    "moe_expert_capacity_factor": 1.0,
                    "moe_pad_expert_input_to_capacity": True,
                },
                False,
            ),
        ],
    )
    def test_static_padding_mask_is_all_padding_only_for_graphed_megakernels(
        self, monkeypatch, megakernel, cuda_graph_modules, config_overrides, all_padding
    ):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        try:
            layer = _moe_transformer_layer(
                monkeypatch,
                _RecordingMegakernel() if megakernel else None,
                cuda_graph_impl="transformer_engine",
                cuda_graph_modules=cuda_graph_modules,
                **config_overrides,
            )
            # What the eager cuda_graph_warmup_steps record from masked batches.
            layer._padding_mask_shape_seen = (2, 8)

            static_mask = layer.get_layer_static_inputs(8, 2)["padding_mask"]

            assert static_mask.dtype == torch.bool and tuple(static_mask.shape) == (2, 8)
            assert bool(static_mask.all()) if all_padding else not bool(static_mask.any())
            assert layer._cuda_graph_padding_mask_shape == (2, 8)
        finally:
            Utils.destroy_model_parallel()

    def setup_method(self, method):
        pass

    @pytest.mark.skipif(
        not is_te_min_version("1.7.0.dev0"),
        reason="Expert with TE Linear is only supported in TE 1.7.0 and later.",
    )
    @pytest.mark.parametrize("moe_token_dispatcher_type", ["allgather", "alltoall"])
    @pytest.mark.parametrize("num_moe_experts", [1, 2])
    @pytest.mark.parametrize("grouped_gemm", [True, False])
    def test_te_moe_layer(self, num_moe_experts, moe_token_dispatcher_type, grouped_gemm):
        Utils.initialize_model_parallel(1, 1)
        _set_random_seed(seed_=123, data_parallel_random_init=False)
        self.transformer_config = TransformerConfig(
            num_layers=1,
            hidden_size=12,
            num_attention_heads=4,
            num_moe_experts=num_moe_experts,
            use_cpu_initialization=True,
            moe_token_dispatcher_type=moe_token_dispatcher_type,
            moe_router_topk=2,
            moe_aux_loss_coeff=0.01,
            moe_grouped_gemm=grouped_gemm,
            moe_ffn_hidden_size=128,
            add_bias_linear=False,
        )
        submodules = get_submodules(
            get_gpt_layer_with_transformer_engine_submodules(
                num_experts=num_moe_experts, moe_grouped_gemm=grouped_gemm
            ).mlp
        )
        assert isinstance(submodules, MoESubmodules)
        moe_layer = MoELayer(self.transformer_config, submodules)
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("moe_token_dispatcher_type", ["allgather", "alltoall"])
    @pytest.mark.parametrize("num_moe_experts", [1, 2])
    @pytest.mark.parametrize("grouped_gemm", [True, False])
    def test_legacy_moe_layer(self, num_moe_experts, moe_token_dispatcher_type, grouped_gemm):
        Utils.initialize_model_parallel(1, 1)
        _set_random_seed(seed_=123, data_parallel_random_init=False)
        num_moe_experts = 4
        self.transformer_config = TransformerConfig(
            num_layers=1,
            hidden_size=12,
            num_attention_heads=4,
            num_moe_experts=num_moe_experts,
            use_cpu_initialization=True,
            moe_token_dispatcher_type=moe_token_dispatcher_type,
            moe_router_load_balancing_type="aux_loss",
            moe_router_topk=2,
            moe_aux_loss_coeff=0.01,
            moe_grouped_gemm=grouped_gemm,
            add_bias_linear=False,
        )
        submodules = get_submodules(
            get_gpt_layer_local_submodules(
                num_experts=num_moe_experts, moe_grouped_gemm=grouped_gemm
            ).mlp
        )
        assert isinstance(submodules, MoESubmodules)
        moe_layer = MoELayer(self.transformer_config, submodules)
        Utils.destroy_model_parallel()

    @pytest.mark.skip(
        "Late init of parallel_state was broken after parallel states refactor MR2988."
    )
    @pytest.mark.parametrize("moe_token_dispatcher_type", ["alltoall", "allgather"])
    @pytest.mark.parametrize("grouped_gemm", [True, False])
    @pytest.mark.parametrize("tp_size,ep_size", [(1, 1), (2, 2)])
    def test_moe_with_late_initialize(
        self, moe_token_dispatcher_type, grouped_gemm, tp_size, ep_size
    ):
        num_moe_experts = 4
        hidden_size = 12
        transformer_config = TransformerConfig(
            num_layers=1,
            hidden_size=hidden_size,
            num_attention_heads=4,
            num_moe_experts=num_moe_experts,
            use_cpu_initialization=True,
            moe_router_load_balancing_type="aux_loss",
            moe_router_topk=2,
            moe_aux_loss_coeff=0.01,
            add_bias_linear=False,
            moe_grouped_gemm=grouped_gemm,
            moe_token_dispatcher_type=moe_token_dispatcher_type,
            tensor_model_parallel_size=tp_size,
            expert_model_parallel_size=ep_size,
            sequence_parallel=tp_size > 1,
            bf16=True,
            params_dtype=torch.bfloat16,
        )
        submodules = get_submodules(
            get_gpt_layer_with_transformer_engine_submodules(
                num_experts=num_moe_experts, moe_grouped_gemm=grouped_gemm
            ).mlp
        )
        assert isinstance(submodules, MoESubmodules)

        # Fake initialization as NeMo does
        Utils.fake_initialize_model_parallel(
            tensor_model_parallel_size=tp_size, expert_model_parallel_size=ep_size
        )
        moe_layer = MoELayer(transformer_config, submodules).cuda()

        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp_size, expert_model_parallel_size=ep_size
        )
        _set_random_seed(seed_=123, data_parallel_random_init=False)

        input_data = torch.randn(
            16, 4, hidden_size, device=torch.cuda.current_device(), dtype=torch.bfloat16
        )
        output = moe_layer(input_data)

        Utils.destroy_model_parallel()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()


class TestInterleaveTransformerBlock:

    @pytest.mark.parametrize("moe_layer_freq", [2, eval("[0,1,1,1]"), eval("[0]*2+[1]*2")])
    def test_interleave_transformer_block(self, moe_layer_freq):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        self.transformer_config = TransformerConfig(
            num_layers=4,
            hidden_size=64,
            num_attention_heads=4,
            moe_layer_freq=moe_layer_freq,
            moe_ffn_hidden_size=256,
            use_cpu_initialization=True,
            num_moe_experts=2,
            add_bias_linear=False,
        )
        self.parallel_transformer_block = TransformerBlock(
            self.transformer_config, get_gpt_decoder_block_spec(self.transformer_config, False)
        )

        # Check if the moe layer is interleaved correctly
        if isinstance(self.transformer_config.moe_layer_freq, int):
            moe_layer_pattern = [
                1 if (i % self.transformer_config.moe_layer_freq == 0) else 0
                for i in range(self.transformer_config.num_layers)
            ]
        else:
            moe_layer_pattern = self.transformer_config.moe_layer_freq

        for i, layer in enumerate(self.parallel_transformer_block.layers):
            is_moe_layer = isinstance(layer.mlp, MoELayer)
            assert is_moe_layer == moe_layer_pattern[i]

        # Test forward pass
        parallel_transformer_block = self.parallel_transformer_block
        config: TransformerConfig = parallel_transformer_block.config
        sequence_length = 32
        micro_batch_size = 2
        parallel_transformer_block.cuda()

        # [sequence length, batch size, hidden size]
        hidden_states = torch.ones((sequence_length, micro_batch_size, config.hidden_size))
        hidden_states = hidden_states.cuda()

        attention_mask = torch.ones((1, 1, sequence_length, sequence_length), dtype=bool).cuda()
        hidden_states = parallel_transformer_block(
            hidden_states=hidden_states, attention_mask=attention_mask
        )
        assert hidden_states.shape[0] == sequence_length
        assert hidden_states.shape[1] == micro_batch_size
        assert hidden_states.shape[2] == config.hidden_size

    def teardown_method(self, method):
        Utils.destroy_model_parallel()


class TestMoELayerFP16:
    """Test MoE layer with FP16 precision."""

    def setup_method(self, method):
        pass

    @pytest.mark.parametrize("moe_token_dispatcher_type", ["allgather", "alltoall"])
    @pytest.mark.parametrize("num_moe_experts", [2, 4])
    @pytest.mark.parametrize("tp_size,ep_size", [(1, 1), (2, 2), (4, 2)])
    def test_moe_layer_fp16_forward_backward(
        self, num_moe_experts, moe_token_dispatcher_type, tp_size, ep_size
    ):
        """Test MoE layer forward and backward pass with fp16 params and inputs."""
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp_size, expert_model_parallel_size=ep_size
        )
        _set_random_seed(seed_=123, data_parallel_random_init=False)

        hidden_size = 64
        sequence_length = 32
        micro_batch_size = 2

        transformer_config = TransformerConfig(
            num_layers=1,
            hidden_size=hidden_size,
            num_attention_heads=4,
            num_moe_experts=num_moe_experts,
            use_cpu_initialization=False,
            moe_token_dispatcher_type=moe_token_dispatcher_type,
            moe_router_load_balancing_type="aux_loss",
            moe_router_topk=2,
            moe_aux_loss_coeff=0.01,
            moe_grouped_gemm=False,  # Use SequentialMLP for fp16 test
            moe_ffn_hidden_size=256,
            add_bias_linear=False,
            tensor_model_parallel_size=tp_size,
            expert_model_parallel_size=ep_size,
            sequence_parallel=tp_size > 1,
            fp16=True,
            params_dtype=torch.float16,
        )

        submodules = get_submodules(
            get_gpt_layer_local_submodules(num_experts=num_moe_experts, moe_grouped_gemm=False).mlp
        )
        assert isinstance(submodules, MoESubmodules)

        moe_layer = MoELayer(transformer_config, submodules).cuda()

        hidden_states = torch.randn(
            sequence_length,
            micro_batch_size,
            hidden_size,
            device=torch.cuda.current_device(),
            dtype=torch.float16,
            requires_grad=True,
        )

        # Forward pass
        output, _ = moe_layer(hidden_states)

        assert output.dtype == torch.float16, f"Expected fp16 output, got {output.dtype}"
        assert output.shape == hidden_states.shape, f"Output shape mismatch"

        # Backward pass
        loss = output.sum()
        loss.backward()

        assert hidden_states.grad is not None, "Input gradients should exist"
        assert (
            hidden_states.grad.dtype == torch.float16
        ), f"Expected fp16 gradients, got {hidden_states.grad.dtype}"

        for name, param in moe_layer.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Gradient for {name} should exist"

        Utils.destroy_model_parallel()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()


class TestMoELayerRecompute:
    """Test MoE layer with recompute enabled (activation checkpointing).

    Tests both code paths:
    - fp8=False: uses tensor_parallel.checkpoint
    - fp8=True: uses te_checkpoint (requires TE >= 1.7.0)
    """

    def setup_method(self, method):
        pass

    @pytest.mark.parametrize("moe_token_dispatcher_type", ["allgather", "alltoall"])
    @pytest.mark.parametrize("num_moe_experts", [2, 4])
    @pytest.mark.parametrize("with_padding_mask", [True, False])
    @pytest.mark.parametrize("tp_size,ep_size", [(1, 1), (4, 2)])
    @pytest.mark.parametrize("fp8", [False, True])
    def test_moe_layer_recompute_forward_backward(
        self, num_moe_experts, moe_token_dispatcher_type, with_padding_mask, tp_size, ep_size, fp8
    ):
        """Test MoE layer forward and backward pass with recompute enabled.

        When fp8=False, uses tensor_parallel.checkpoint.
        When fp8=True, uses te_checkpoint (requires TE >= 1.7.0).
        """
        # Skip fp8 tests if TE version is not sufficient
        if fp8 and not is_te_min_version("1.7.0.dev0"):
            pytest.skip("FP8 MoE recompute requires TE 1.7.0 and later.")

        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp_size, expert_model_parallel_size=ep_size
        )
        _set_random_seed(seed_=123, data_parallel_random_init=False)

        hidden_size = 64
        sequence_length = 32
        micro_batch_size = 2

        transformer_config = TransformerConfig(
            num_layers=1,
            hidden_size=hidden_size,
            num_attention_heads=4,
            num_moe_experts=num_moe_experts,
            use_cpu_initialization=False,
            moe_token_dispatcher_type=moe_token_dispatcher_type,
            moe_router_load_balancing_type="aux_loss",
            moe_router_topk=2,
            moe_aux_loss_coeff=0.01,
            moe_grouped_gemm=False,
            moe_ffn_hidden_size=256,
            add_bias_linear=False,
            # Enable recompute for MoE layer
            recompute_granularity="selective",
            recompute_modules=["moe"],
            tensor_model_parallel_size=tp_size,
            expert_model_parallel_size=ep_size,
            sequence_parallel=tp_size > 1,
            fp8=fp8,
            bf16=True,
            params_dtype=torch.bfloat16,
        )

        # Use TE spec for fp8, local spec otherwise
        if fp8:
            submodules = get_submodules(
                get_gpt_layer_with_transformer_engine_submodules(
                    num_experts=num_moe_experts, moe_grouped_gemm=False
                ).mlp
            )
        else:
            submodules = get_submodules(
                get_gpt_layer_local_submodules(
                    num_experts=num_moe_experts, moe_grouped_gemm=False
                ).mlp
            )
        assert isinstance(submodules, MoESubmodules)

        moe_layer = MoELayer(transformer_config, submodules).cuda()

        hidden_states = torch.randn(
            sequence_length,
            micro_batch_size,
            hidden_size,
            device=torch.cuda.current_device(),
            dtype=torch.bfloat16,
            requires_grad=True,
        )

        # Create padding mask if needed: shape [batch_size, sequence_length]
        # Convention: True = padding (exclude), False = valid (include)
        padding_mask = None
        if with_padding_mask:
            padding_mask = torch.zeros(
                micro_batch_size,
                sequence_length,
                device=torch.cuda.current_device(),
                dtype=torch.bool,
            )
            # Mark last 4 tokens as padding for each batch
            padding_mask[:, -4:] = True

        output, _ = moe_layer(hidden_states, padding_mask=padding_mask)

        assert output.dtype == torch.bfloat16, f"Expected bf16 output, got {output.dtype}"
        assert output.shape == hidden_states.shape, f"Output shape mismatch"

        # Backward pass - this is where recompute/checkpoint is actually used
        loss = output.sum()
        loss.backward()

        assert hidden_states.grad is not None, "Input gradients should exist"
        assert (
            hidden_states.grad.dtype == torch.bfloat16
        ), f"Expected bf16 gradients, got {hidden_states.grad.dtype}"

        for name, param in moe_layer.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Gradient for {name} should exist"

        Utils.destroy_model_parallel()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()
