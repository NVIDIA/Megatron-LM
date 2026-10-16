# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import gc
from contextlib import nullcontext

import pytest
import torch

from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_submodules,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.moe import fused_a2a
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.spec_utils import get_submodules
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_te_min_version
from tests.unit_tests.test_utilities import Utils


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not is_te_min_version("2.3.0"), reason="Requires TE >= 2.3.0")
class TestEarlyExpertWgrad:
    """Exercise the shared/routed fork with real experts and communication."""

    @pytest.fixture(autouse=True, params=["deepep", "deepepv2", "hybridep", "alltoall"])
    def backend(self, request):
        available = {
            "deepep": fused_a2a.HAVE_DEEP_EP,
            "deepepv2": fused_a2a.HAVE_DEEP_EP_V2,
            "hybridep": fused_a2a.HAVE_HYBRIDEP,
            "alltoall": True,
        }
        if not available[request.param]:
            pytest.skip(f"Requires {request.param}")
        self.dispatcher_backend = request.param
        self.dispatcher_config = {
            "moe_token_dispatcher_type": "alltoall" if request.param == "alltoall" else "flex",
            "moe_flex_dispatcher_backend": (
                "deepep" if request.param == "alltoall" else request.param
            ),
        }

    @pytest.fixture(autouse=True)
    def parallel_groups(self, request, backend, monkeypatch):
        expert_dp_size = getattr(request, "param", 1)
        if Utils.world_size % expert_dp_size:
            pytest.skip("World size must be divisible by expert data parallel size")
        self.expert_parallel_size = Utils.world_size // expert_dp_size
        nccl_groups = []
        original_new_group = torch.distributed.new_group

        def new_group(*args, **kwargs):
            group = original_new_group(*args, **kwargs)
            if (
                isinstance(group, torch.distributed.ProcessGroup)
                and torch.distributed.get_backend(group) == "nccl"
            ):
                nccl_groups.append(group)
            return group

        monkeypatch.setattr(torch.distributed, "new_group", new_group)
        Utils.initialize_model_parallel(expert_model_parallel_size=self.expert_parallel_size)
        yield expert_dp_size
        torch.cuda.synchronize()
        # Drop instrumented bound methods before collecting models/communication
        # buffers, which may still borrow these groups' NCCL communicators.
        monkeypatch.undo()
        fused_a2a._buffer = None
        fused_a2a._elastic_buffer = None
        if self.dispatcher_backend == "hybridep":
            fused_a2a.reset_hybrid_ep_buffer()
        gc.collect()
        Utils.destroy_model_parallel()
        # Utils clears NCCL references but only destroys Gloo groups. Explicitly
        # release this test's NCCL groups so the backend matrix does not retain
        # thousands of communicator threads in torch.distributed's registry.
        for group in reversed(nccl_groups):
            torch.distributed.destroy_process_group(group)
        torch.cuda.empty_cache()

    def make_layer(self, overlap=True, recompute=False, fp8=None, fusion=False):
        if fp8 is not None:
            from transformer_engine.pytorch.fp8 import (  # type: ignore[import-untyped]
                check_fp8_block_scaling_support,
            )

            fp8_available, reason = check_fp8_block_scaling_support()
            if not fp8_available:
                pytest.skip(reason)
        torch.manual_seed(123)
        model_parallel_cuda_manual_seed(123)
        config = TransformerConfig(
            num_layers=1,
            hidden_size=256,
            num_attention_heads=4,
            num_moe_experts=2 * Utils.world_size,
            expert_model_parallel_size=self.expert_parallel_size,
            moe_ffn_hidden_size=256,
            moe_shared_expert_intermediate_size=256,
            moe_router_topk=2,
            moe_router_dtype="fp32",
            moe_aux_loss_coeff=0.0,
            moe_shared_expert_overlap=True,
            overlap_dispatch_backward_with_experts_wgrad=overlap,
            moe_grouped_gemm=True,
            moe_permute_fusion=True,
            gated_linear_unit=True,
            activation_func=torch.nn.functional.silu,
            add_bias_linear=False,
            bf16=True,
            params_dtype=torch.bfloat16,
            gradient_accumulation_fusion=fusion,
            use_cpu_initialization=False,
            recompute_granularity="selective" if recompute else None,
            recompute_modules=["moe"] if recompute else [],
            fp8=fp8,
            fp8_recipe="blockwise",
            **self.dispatcher_config,
        )
        submodules = get_submodules(
            get_gpt_layer_with_transformer_engine_submodules(
                num_experts=config.num_moe_experts, moe_grouped_gemm=True
            ).mlp
        )
        layer = MoELayer(config, submodules).cuda()
        layer.set_layer_number(1)
        # DeepEP reads this group's NCCL handle, so initialize the actual TPxEP group.
        torch.distributed.barrier(
            group=layer.token_dispatcher.tp_ep_group, device_ids=[torch.cuda.current_device()]
        )
        return layer

    @pytest.mark.parametrize("recompute", [False, True])
    @pytest.mark.parametrize("fp8", [None, "e4m3"])
    def test_launch_precedes_shared_input_gradient_merge(self, monkeypatch, recompute, fp8):
        layer = self.make_layer(recompute=recompute, fp8=fp8)
        order = []
        dispatcher = layer.token_dispatcher
        dispatch_method = "token_dispatch"
        if self.dispatcher_backend != "alltoall":
            # Register the observation before Flex installs its wgrad post-hook.
            dispatcher = dispatcher._comm_manager
            dispatch_method = "dispatch"
        original_dispatch = getattr(dispatcher, dispatch_method)
        original_preprocess = layer.token_dispatcher.dispatch_preprocess
        original_shared = layer.shared_experts.pre_forward_comm
        original_wgrad = layer.backward_dw

        def dispatch(*args, **kwargs):
            output = original_dispatch(*args, **kwargs)
            hidden_states = output[0] if self.dispatcher_backend == "alltoall" else output
            if hidden_states.grad_fn is not None:
                hidden_states.grad_fn.register_hook(lambda *unused: order.append("dispatch"))
            return output

        def shared(hidden_states, *args, **kwargs):
            if (
                torch.is_grad_enabled()
                and hidden_states.requires_grad
                and self.dispatcher_backend != "alltoall"
            ):
                hidden_states.register_hook(lambda grad: order.append("merge"))
            return original_shared(hidden_states, *args, **kwargs)

        def preprocess(hidden_states, *args, **kwargs):
            if (
                torch.is_grad_enabled()
                and hidden_states.requires_grad
                and self.dispatcher_backend == "alltoall"
            ):
                # All-to-all creates the shared/routed fork during preprocessing.
                hidden_states.register_hook(lambda grad: order.append("merge"))
            return original_preprocess(hidden_states, *args, **kwargs)

        def wgrad(*args, **kwargs):
            order.append("wgrad")
            return original_wgrad(*args, **kwargs)

        monkeypatch.setattr(dispatcher, dispatch_method, dispatch)
        monkeypatch.setattr(layer.token_dispatcher, "dispatch_preprocess", preprocess)
        monkeypatch.setattr(layer.shared_experts, "pre_forward_comm", shared)
        monkeypatch.setattr(layer, "backward_dw", wgrad)
        if self.dispatcher_backend != "alltoall":
            # Observe the real FC1 backward without inserting autograd nodes that
            # could change the scheduling priority being tested.
            def observe_shared_fc1(module, inputs, output):
                if output[0].grad_fn is not None:
                    output[0].grad_fn.register_prehook(lambda *unused: order.append("shared_fc1"))

            layer.shared_experts.linear_fc1.register_forward_hook(observe_shared_fc1)
        # This is also the completion point used by deferred gradient processing hooks.
        parameter = next(layer.experts.parameters())
        parameter.post_wgrad_grad_acc_hook = lambda: order.append("complete")
        for _ in range(2):
            order.clear()
            layer.zero_grad(set_to_none=True)
            x = torch.randn(128, 1, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            with get_fp8_context(layer.config):
                output = layer(x)[0]
            output.float().square().mean().backward()
            torch.cuda.synchronize()
            expected = ["dispatch", "wgrad", "shared_fc1", "merge", "complete"]
            if self.dispatcher_backend == "alltoall":
                expected = ["dispatch", "wgrad", "complete", "merge"]
            assert order == expected, order
            assert x.grad is not None and torch.isfinite(x.grad).all()

    @pytest.mark.parametrize("recompute", [False, True])
    @pytest.mark.parametrize("fp8", [None, "e4m3"])
    def test_gradients(self, recompute, fp8):
        reference = self.make_layer(overlap=False, recompute=recompute, fp8=fp8)
        candidate = self.make_layer(recompute=recompute, fp8=fp8)
        candidate.load_state_dict(reference.state_dict())
        generator = torch.Generator(device="cuda").manual_seed(321 + Utils.rank)
        for step in range(2):
            for microbatch in range(2):
                # TE's non-fused backward_dw assigns .grad rather than accumulating it.
                # Accumulation is exercised through fused main_grad buffers below.
                reference.zero_grad(set_to_none=True)
                candidate.zero_grad(set_to_none=True)
                value = (
                    torch.randn(
                        128, 1, 256, generator=generator, device="cuda", dtype=torch.bfloat16
                    )
                    / 8
                )
                grad = (
                    torch.randn(
                        value.shape, generator=generator, device="cuda", dtype=torch.bfloat16
                    )
                    / 8
                )
                inputs, outputs = [], []
                for layer in (reference, candidate):
                    x = value.detach().clone().requires_grad_()
                    with get_fp8_context(layer.config):
                        y = layer(x)[0]
                    y.backward(grad)
                    torch.cuda.synchronize()
                    inputs.append(x)
                    outputs.append(y)
                torch.testing.assert_close(outputs[1], outputs[0], rtol=0.02, atol=2e-4)
                torch.testing.assert_close(inputs[1].grad, inputs[0].grad, rtol=0.02, atol=2e-4)
                for (name, expected), (_, actual) in zip(
                    reference.named_parameters(), candidate.named_parameters()
                ):
                    assert expected.grad is not None and actual.grad is not None, name
                    torch.testing.assert_close(
                        actual.grad,
                        expected.grad,
                        rtol=0.02,
                        atol=2e-4,
                        msg=lambda message: f"step={step}, microbatch={microbatch}, {name}: {message}",
                    )

    def test_multiple_pending_forwards(self):
        """Each outstanding autograd graph must launch and join its own wgrad once."""
        reference = self.make_layer(overlap=False)
        candidate = self.make_layer()
        candidate.load_state_dict(reference.state_dict())
        generator = torch.Generator(device="cuda").manual_seed(641 + Utils.rank)
        values = [
            torch.randn(128, 1, 256, generator=generator, device="cuda", dtype=torch.bfloat16) / 8
            for _ in range(2)
        ]
        pending = []
        completed = []
        next(candidate.experts.parameters()).post_wgrad_grad_acc_hook = lambda: completed.append(1)
        for model in (reference, candidate):
            # A no-grad forward must not leave launch state for a later backward.
            with torch.no_grad():
                model(values[0])
            inputs = [x.detach().clone().requires_grad_() for x in values]
            outputs = [model(x)[0] for x in inputs]
            pending.append((inputs, outputs))
        for index in (1, 0):
            for model, (inputs, outputs) in zip((reference, candidate), pending):
                model.zero_grad(set_to_none=True)
                outputs[index].backward(torch.ones_like(outputs[index]) / 128)
            torch.cuda.synchronize()
            torch.testing.assert_close(
                pending[1][0][index].grad, pending[0][0][index].grad, rtol=0.02, atol=2e-4
            )
            for expected, actual in zip(reference.parameters(), candidate.parameters()):
                torch.testing.assert_close(actual.grad, expected.grad, rtol=0.02, atol=2e-4)
        assert len(completed) == 2

    @pytest.mark.parametrize("recompute", [False, True])
    @pytest.mark.parametrize("fp8", [None, "e4m3"])
    @pytest.mark.parametrize("overlap_grad_reduce", [False, True])
    @pytest.mark.parametrize(
        "parallel_groups", [1, 2], indirect=True, ids=["expert_dp1", "expert_dp2"]
    )
    def test_fused_accumulation_and_updates(
        self, recompute, fp8, overlap_grad_reduce, parallel_groups
    ):
        layers = [
            self.make_layer(overlap=w, recompute=recompute, fp8=fp8, fusion=True)
            for w in (False, True)
        ]
        layers[1].load_state_dict(layers[0].state_dict())
        models = [
            DistributedDataParallel(
                layer.config,
                DistributedDataParallelConfig(
                    overlap_grad_reduce=overlap_grad_reduce,
                    # Explicitly select the deferred accumulation/reduction hook path.
                    delay_wgrad_compute=True,
                ),
                layer,
            )
            for layer in layers
        ]
        for model in models:
            assert model.expt_dp_group.size() == parallel_groups
            assert model.intra_expt_dp_group.size() == parallel_groups
        generator = torch.Generator(device="cuda").manual_seed(321 + Utils.rank)
        masters = [[p.detach().float().clone() for p in m.parameters()] for m in models]
        for step in range(2):
            for model in models:
                model.zero_grad_buffer()
            values = [
                torch.randn(128, 1, 256, generator=generator, device="cuda", dtype=torch.bfloat16)
                / 8
                for _ in range(2)
            ]
            grads = [
                torch.randn(v.shape, generator=generator, device="cuda", dtype=torch.bfloat16) / 8
                for v in values
            ]
            observations = []
            for model, layer in zip(models, layers):
                observed = []
                for microbatch, (value, grad) in enumerate(zip(values, grads)):
                    x = value.detach().clone().requires_grad_()
                    with model.no_sync() if microbatch == 0 else nullcontext():
                        with get_fp8_context(layer.config):
                            y = model(x)[0]
                        y.backward(grad)
                    observed.extend([y.detach().clone(), x.grad.detach().clone()])
                model.finish_grad_sync()
                torch.cuda.synchronize()
                observations.append(observed)
            for expected, actual in zip(*observations):
                torch.testing.assert_close(actual, expected, rtol=0.02, atol=2e-4)
            for (name, expected), (_, actual), master_ref, master_new in zip(
                models[0].named_parameters(), models[1].named_parameters(), *masters
            ):
                torch.testing.assert_close(
                    actual.main_grad,
                    expected.main_grad,
                    rtol=0.02,
                    atol=2e-4,
                    msg=lambda m: name + ": " + m,
                )
                if parallel_groups > 1 and not getattr(actual, "allreduce", True):
                    # Rank-specific inputs make a missing expert reduction observable.
                    replica_grads = [
                        torch.empty_like(actual.main_grad) for _ in range(parallel_groups)
                    ]
                    torch.distributed.all_gather(
                        replica_grads, actual.main_grad.contiguous(), group=models[1].expt_dp_group
                    )
                    for replica_grad in replica_grads:
                        torch.testing.assert_close(
                            actual.main_grad, replica_grad, rtol=0, atol=0, msg=name
                        )
                with torch.no_grad():
                    master_ref.add_(expected.main_grad.float(), alpha=-0.01)
                    master_new.add_(actual.main_grad.float(), alpha=-0.01)
                    expected.copy_(master_ref)
                    actual.copy_(master_new)
                torch.testing.assert_close(master_new, master_ref, rtol=0.002, atol=2e-5)
