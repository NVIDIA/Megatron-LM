# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F

import megatron.core.transformer.moe.moe_layer as moe_layer_module
from megatron.core import parallel_state
from megatron.core.distributed import (
    DistributedDataParallel,
    DistributedDataParallelConfig,
    finalize_model_grads,
)
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_layer_local_submodules,
    get_gpt_layer_with_transformer_engine_submodules,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.spec_utils import get_submodules
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_te_min_version
from megatron.training.initialize import _set_random_seed
from tests.unit_tests.test_utilities import Utils


def _shared_composition_reference(x, fc1, fc2):
    gate, up = F.linear(x, fc1).chunk(2, dim=-1)
    return F.linear(F.silu(gate) * up, fc2)


def _composition_reference(x, parameters, config):
    probs = F.linear(x, parameters["router.weight"]).softmax(dim=-1)
    hidden = F.linear(x, parameters["fc1_latent_proj.weight"]) if config.moe_latent_size else x
    output = hidden.tanh() * (probs[..., :1] + 2 * probs[..., 1:])
    if config.moe_latent_size:
        output = output * torch.rsqrt(
            output.square().mean(-1, keepdim=True) + config.layernorm_epsilon
        )
        output = output * parameters["fc2_norm.weight"]
        output = F.linear(output, parameters["fc2_latent_proj.weight"])
    if config.moe_shared_expert_intermediate_size:
        output = output + _shared_composition_reference(
            x,
            parameters["shared_experts.linear_fc1.weight"],
            parameters["shared_experts.linear_fc2.weight"],
        )
    return output


def _mok_composition_layer(
    monkeypatch, *, tp=1, latent=None, shared=64, mode="auto", recompute=False
):
    """Real MoELayer/TE composition; only the external routed backend is substituted."""
    seen = {"router": [], "shared": [], "backend": []}

    def forbidden(*args, **kwargs):
        pytest.fail("MOK composition must not construct or call a native dispatcher")

    class NoDispatcher:
        __getattr__ = forbidden

    class Routed(torch.nn.Module):
        def __init__(self, shared_module) -> None:
            super().__init__()
            self.shared = shared_module

        def forward(self, hidden, probs, routes):
            seen["backend"].append(hidden.shape)
            assert routes.all()
            scale = (probs[:, :1] + 2 * probs[:, 1:]).view(*hidden.shape[:-1], 1)
            output = hidden.tanh() * scale
            if self.shared is not None:
                output = output + _shared_composition_reference(
                    hidden, self.shared.linear_fc1.weight, self.shared.linear_fc2.weight
                )
            return output

    for name in (
        "MoEAllGatherTokenDispatcher",
        "MoEAlltoAllTokenDispatcher",
        "MoEFlexTokenDispatcher",
    ):
        monkeypatch.setattr(moe_layer_module, name, forbidden)
    monkeypatch.setattr(
        moe_layer_module,
        "build_megakernel_backend",
        lambda **kwargs: Routed(kwargs["shared_experts"]),
    )
    config = TransformerConfig(
        num_layers=1,
        hidden_size=128,
        num_attention_heads=4,
        ffn_hidden_size=64,
        num_moe_experts=2,
        moe_router_topk=2,
        moe_router_pre_softmax=True,
        moe_router_load_balancing_type="none",
        moe_aux_loss_coeff=0.0,
        moe_router_dtype="fp32",
        moe_ffn_hidden_size=64,
        moe_shared_expert_intermediate_size=shared,
        moe_latent_size=latent,
        moe_use_norm_before_up_proj=latent is not None,
        normalization="RMSNorm",
        tensor_model_parallel_size=tp,
        sequence_parallel=tp > 1,
        expert_tensor_parallel_size=1,
        moe_grouped_gemm=True,
        gated_linear_unit=True,
        activation_func=F.silu,
        add_bias_linear=False,
        bias_activation_fusion=False,
        bf16=True,
        params_dtype=torch.float32,
        gradient_accumulation_fusion=True,
        use_cpu_initialization=True,
        moe_megakernel_backend="mok",
        moe_megakernel_backend_config={"shared_expert_mode": mode},
        recompute_granularity="selective" if recompute else None,
        recompute_modules=["moe", "shared_experts"] if recompute else [],
    )
    spec = get_submodules(
        get_gpt_layer_with_transformer_engine_submodules(num_experts=2, moe_grouped_gemm=True).mlp
    )
    layer = MoELayer(config, spec).cuda()
    assert layer.token_dispatcher is None
    assert isinstance(layer.megakernel_experts, Routed)
    assert layer.megakernel_experts.shared is (
        layer.shared_experts if layer.megakernel_fuses_shared_expert else None
    )
    layer.token_dispatcher = NoDispatcher()
    for name in ("router", "shared"):
        module = layer.router if name == "router" else layer.shared_experts
        if module is not None:
            module.register_forward_pre_hook(
                lambda module, args, key=name: seen[key].append(args[0].detach().clone())
            )
    with torch.no_grad():
        for index, (name, param) in enumerate(layer.named_parameters()):
            values = torch.arange(param.numel(), device=param.device).reshape(param.shape)
            param.copy_(
                torch.ones_like(param)
                if "layer_norm_weight" in name or name == "fc2_norm.weight"
                else 0.04 * torch.sin(values * 0.37 + index + 1)
            )
    parameters = {
        name: param
        for name, param in layer.named_parameters()
        if name.startswith(
            ("router.", "shared_experts.", "fc1_latent_proj.", "fc2_norm.", "fc2_latent_proj.")
        )
    }
    reference = {
        name: param.detach().clone().requires_grad_() for name, param in parameters.items()
    }
    model = DistributedDataParallel(
        config, DistributedDataParallelConfig(grad_reduce_in_fp32=True), layer
    )
    return model, parameters, reference, seen


class TestMoELayerInit:

    @pytest.mark.parametrize("already_sharded", [False, True])
    def test_mok_route_aligns_sequence_parallel_metadata(self, already_sharded):
        Utils.initialize_model_parallel(2, 1)
        try:
            layer = object.__new__(MoELayer)
            torch.nn.Module.__init__(layer)
            layer.config = SimpleNamespace(
                moe_megakernel_backend="mok", sequence_parallel=True, cuda_graph_impl="none"
            )
            layer.tp_group = parallel_state.get_tensor_model_parallel_group()
            layer.token_dispatcher = None
            layer.router = torch.nn.Module()
            layer.router.forward = Mock(return_value=(None, None))
            hidden = torch.zeros(3, 2, 8, device="cuda")
            token_ids = torch.arange(12, device="cuda").view(2, 6)
            mask = token_ids % 3 == 0
            start = layer.tp_group.rank() * 3
            expected_ids = token_ids[:, start : start + 3]
            expected_mask = mask[:, start : start + 3]
            if already_sharded:
                token_ids, mask = expected_ids, expected_mask

            layer.route(hidden, mask, token_ids)

            args, kwargs = layer.router.forward.call_args
            assert args[0] is hidden
            torch.testing.assert_close(args[1], expected_mask.T)
            torch.testing.assert_close(kwargs["input_ids"], expected_ids)
        finally:
            Utils.destroy_model_parallel()

    @pytest.mark.skipif(
        not is_te_min_version("1.7.0.dev0"),
        reason="Expert with TE Linear is only supported in TE 1.7.0 and later.",
    )
    @pytest.mark.parametrize("latent,shared_size", [(None, 128), (64, 256), (None, None)])
    def test_megakernel_skips_native_token_dispatcher_construction(
        self, monkeypatch, latent, shared_size
    ):
        Utils.initialize_model_parallel(1, 1)
        try:
            config = TransformerConfig(
                num_layers=1,
                hidden_size=128,
                num_attention_heads=4,
                num_moe_experts=1,
                moe_ffn_hidden_size=128,
                moe_shared_expert_intermediate_size=shared_size,
                moe_latent_size=latent,
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

            def build_backend(**kwargs):
                assert (kwargs["shared_experts"] is not None) == (
                    latent is None and shared_size == 128
                )
                return backend

            monkeypatch.setattr(moe_layer_module, "build_megakernel_backend", build_backend)

            layer = MoELayer(config, submodules)

            assert layer.token_dispatcher is None
            assert layer.megakernel_experts is backend
            if latent is None and shared_size == 128:
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
                backend_forward.assert_called_once_with(hidden, probs, routing_map)

            if latent is None:
                hidden = torch.zeros(2, 1, config.hidden_size)
                monkeypatch.setattr(
                    layer.router, "forward", lambda *args, **kwargs: (hidden, hidden)
                )
                probs, routing_map = layer.route(hidden)
                prepared, _ = layer.preprocess(hidden, probs, routing_map)
                assert prepared is hidden
                assert layer.postprocess(hidden, None) is hidden
        finally:
            Utils.destroy_model_parallel()

    @pytest.mark.skipif(not is_te_min_version("1.7.0.dev0"), reason="Composition test requires TE")
    @pytest.mark.parametrize(
        "latent,shared,mode",
        [(64, 128, "auto"), (None, None, "auto"), (None, 64, "native"), (None, 64, "fused")],
    )
    def test_mok_forward_composition(self, monkeypatch, latent, shared, mode):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        previous = None
        for recompute in ((False, True) if latent else (False,)):
            model, params, reference, seen = _mok_composition_layer(
                monkeypatch, latent=latent, shared=shared, mode=mode, recompute=recompute
            )
            x = (
                torch.linspace(-0.7, 0.9, 8 * 2 * 128, device="cuda")
                .reshape(8, 2, 128)
                .requires_grad_()
            )
            ref_x = x.detach().clone().requires_grad_()
            dy = torch.linspace(0.2, 1.1, x.numel(), device="cuda").reshape_as(x)
            actual, bias = model(x)
            expected = _composition_reference(ref_x, reference, model.config)
            assert bias is None and actual.shape == x.shape
            torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-6)
            actual.backward(dy)
            expected.backward(dy)
            torch.testing.assert_close(x.grad, ref_x.grad, rtol=2e-4, atol=2e-6)
            for name, param in params.items():
                assert reference[name].grad is not None and reference[name].grad.abs().max() > 0
                assert param.main_grad.abs().max() > 0, name
                torch.testing.assert_close(
                    param.main_grad, reference[name].grad, rtol=3e-4, atol=2e-5
                )
            assert seen["router"]
            for observed in seen["router"] + seen["shared"]:
                torch.testing.assert_close(observed, x.detach(), rtol=0, atol=0)
            assert seen["backend"] and all(
                shape[-1] == (latent or 128) for shape in seen["backend"]
            )
            assert bool(seen["shared"]) == (shared is not None and mode != "fused")
            values = [
                actual.detach(),
                x.grad,
                *(p.main_grad.detach().clone() for p in params.values()),
            ]
            if previous is not None:
                for value, baseline in zip(values, previous):
                    torch.testing.assert_close(value, baseline, rtol=3e-4, atol=2e-5)
            previous = values

    @pytest.mark.skipif(
        not is_te_min_version("1.7.0.dev0"), reason="TP gradient reference requires TE"
    )
    def test_mok_latent_tp_gradient_reference(self, monkeypatch):
        if Utils.world_size < 2 or Utils.world_size % 2:
            pytest.skip("TP2 gradient reference requires an even distributed world size")
        Utils.initialize_model_parallel(tensor_model_parallel_size=2, expert_tensor_parallel_size=1)
        model_parallel_cuda_manual_seed(123)
        model, params, reference, _ = _mok_composition_layer(
            monkeypatch, tp=2, latent=64, shared=None
        )
        for param in params.values():
            peers = [torch.empty_like(param) for _ in range(2)]
            torch.distributed.all_gather(peers, param.detach().contiguous(), group=model.tp_group)
            assert all(torch.equal(param, peer) for peer in peers)
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        # DP replicas use the same data; TP ranks own explicitly different halves.
        full = torch.linspace(-0.8, 1.0, 16 * 2 * 128, device="cuda").reshape(16, 2, 128)
        full_dy = torch.linspace(0.1, 0.7, full.numel(), device="cuda").reshape_as(full)
        x = full.chunk(2)[tp_rank].clone().requires_grad_()
        ref_x = full.clone().requires_grad_()
        actual, _ = model(x)
        expected = _composition_reference(ref_x, reference, model.config)
        assert actual.shape == x.shape
        torch.testing.assert_close(actual, expected.chunk(2)[tp_rank], rtol=2e-4, atol=2e-6)
        # Sum loss; DP averaging is neutral because each DP replica has identical TP shards.
        actual.backward(full_dy.chunk(2)[tp_rank].contiguous())
        expected.backward(full_dy)
        finalize_model_grads([model])
        torch.testing.assert_close(x.grad, ref_x.grad.chunk(2)[tp_rank], rtol=2e-4, atol=2e-6)
        for name, param in params.items():
            if name.startswith(("fc1_latent_proj.", "fc2_norm.", "fc2_latent_proj.")):
                assert param.sequence_parallel and reference[name].grad.abs().max() > 0
                assert param.main_grad.abs().max() > 0, name
                torch.testing.assert_close(
                    param.main_grad, reference[name].grad, rtol=3e-4, atol=2e-5
                )

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
        assert moe_layer.megakernel_fuses_shared_expert is False
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
