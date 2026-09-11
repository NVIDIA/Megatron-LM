# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""End-to-end gradient parity for HybridEP virtual-expert load balancing.

Run on one four-GPU NVLink node::

    uv run python -m torch.distributed.run --nproc-per-node 4 -m pytest -q \
      tests/unit_tests/transformer/moe/test_virtual_expert_hybridep.py

Each case builds the same MoE layer twice -- once on a reference dispatcher and
once with virtual-expert load balancing -- loads identical weights, and compares the output
and every training gradient. Planner and bridge internals are covered
process-locally in ``test_virtual_expert_planner.py``; the transport kernels in
``test_virtual_expert_triton.py``.

A bare ``MoELayer`` carries no DDP-time GTP wrapper. These cases do not cover the
bridge's GTP weight-gather and gradient reduce-scatter integration.
"""

import os

import pytest
import torch
import torch.nn.functional as F
from transformer_engine.pytorch.quantization import FP8GlobalStateManager

from megatron.core.activations import squared_relu
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.transformer.moe import fused_a2a
from megatron.core.transformer.moe.virtual_expert_load_balancer import VirtualExpertLoadBalancer

MXFP8_COMPONENTS = (
    "_rowwise_data",
    "_rowwise_scale_inv",
    "_columnwise_data",
    "_columnwise_scale_inv",
)

# The GB200 CI bucket launches marked files with four ranks, which these tests need.
pytestmark = pytest.mark.launch_on_gb200

requires_four_ranks = pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "1")) != 4
    or not torch.cuda.is_available()
    or not fused_a2a.HAVE_HYBRIDEP,
    reason="virtual-expert parity requires a 4-rank torchrun launch with HybridEP",
)


# Config keys both parity cases share. Each case overrides only what it varies.
BASE_CONFIG = {
    "num_layers": 1,
    "num_attention_heads": 8,
    "num_moe_experts": 4,
    "expert_tensor_parallel_size": 1,
    "moe_router_topk": 2,
    "moe_router_load_balancing_type": "none",
    "moe_router_dtype": "fp32",
    "moe_grouped_gemm": True,
    "moe_single_grouped_weight": False,
    "use_transformer_engine_op_fuser": True,
    "gradient_accumulation_fusion": True,
    "add_bias_linear": False,
    "bf16": True,
    "params_dtype": torch.bfloat16,
    "use_cpu_initialization": False,
}


def _set_main_grad(parameter, dtype=torch.float32):
    parameter.main_grad = torch.zeros(parameter.shape, dtype=dtype, device=parameter.device)
    parameter.grad_added_to_main_grad = False
    parameter.overwrite_main_grad = True


def _dense_linears(layer):
    """The layer's ordinary TE linears: latent FC layers and shared experts, if present."""
    linears = []
    if layer.config.moe_latent_size is not None:
        linears += [layer.fc1_latent_proj, layer.fc2_latent_proj]
    if layer.use_shared_expert:
        linears += [layer.shared_experts.linear_fc1, layer.shared_experts.linear_fc2]
    return linears


def _set_main_grads(layer, dtype):
    for linear in (layer.experts.linear_fc1, layer.experts.linear_fc2):
        for index in range(linear.num_gemms):
            _set_main_grad(linear.get_parameter(f"weight{index}"), dtype)
    for linear in _dense_linears(layer):
        _set_main_grad(linear.weight, dtype)


def _stack_linear_main_grad(linear):
    return torch.stack(
        tuple(
            linear.get_parameter(f"weight{i}").main_grad.detach() for i in range(linear.num_gemms)
        )
    )


def _weight_storage_ptrs(weight):
    if hasattr(weight, "_rowwise_data"):
        return tuple(getattr(weight, name).data_ptr() for name in MXFP8_COMPONENTS)
    return (weight.data_ptr(),)


def _assert_numerical_parity(actual, expected, tolerance, name):
    """Bound relative L2 and absolute error against each tensor/expert's own signal scale."""
    reference = expected.float()
    assert torch.isfinite(reference).all() and reference.norm() > 0, f"{name}: invalid reference"
    relative_l2 = (actual.float() - reference).norm() / reference.norm()
    assert relative_l2 <= tolerance, f"{name}: relative L2 error {relative_l2} > {tolerance}"
    torch.testing.assert_close(
        actual,
        expected,
        rtol=0,
        atol=tolerance * reference.abs().max().item(),
        msg=lambda msg: f"{name}: {msg}",
    )


def _assert_mxfp8_prefetch_exact(manager, plan, orientation):
    """Check every active virtual MXFP8 component byte-for-byte against its owning rank."""
    components = MXFP8_COMPONENTS[:2] if orientation == "rowwise" else MXFP8_COMPONENTS[2:]
    errors = []
    for index, parameters in enumerate(manager.virtual_experts.parameters):
        for component in components:
            local = torch.stack(tuple(getattr(source, component) for source in parameters))
            gathered = [torch.empty_like(local) for _ in range(manager.ep_size)]
            torch.distributed.all_gather(gathered, local, group=manager.group)
            semantic = torch.cat(gathered)
            for expert, weight in enumerate(semantic):
                if any(torch.equal(weight, other) for other in semantic[:expert]):
                    errors.append(f"FC{index + 1} {component}: indistinguishable expert {expert}")
            for slot, expert in enumerate(
                plan.experts_to_copy[manager.virtual_experts.rank].tolist()
            ):
                if expert < 0:
                    continue
                owner, owned = divmod(expert, manager.num_owned_experts)
                if not torch.equal(
                    getattr(manager.virtual_experts.slot_weights[index][slot], component),
                    gathered[owner][owned],
                ):
                    errors.append(f"fc_layer={index} {component} slot={slot} expert={expert}")
    any_error = torch.tensor(int(bool(errors)), dtype=torch.int32, device=manager.device)
    torch.distributed.all_reduce(any_error, op=torch.distributed.ReduceOp.MAX, group=manager.group)
    assert (
        not any_error.item()
    ), f"rank {manager.virtual_experts.rank} {orientation} MXFP8 prefetch mismatch: " + (
        ", ".join(errors) if errors else "reported by another rank"
    )


def _assert_runtime_layout(manager, *, grad_dtype, mxfp8):
    """Check that the runtime weights and grads TE executes against alias the shared arenas."""
    assert manager.virtual_experts.grad_arena.dtype == grad_dtype
    for fc_layer, parameters in enumerate(manager.virtual_experts.parameters):
        runtime_weights = manager.runtime_weights(fc_layer)
        assert len(runtime_weights) == manager.num_runtime_experts
        virtual_parameters = manager.virtual_experts.slot_weights[fc_layer]
        assert all(slot.main_grad.dtype == grad_dtype for slot in virtual_parameters)
        for index, runtime_weight in enumerate(runtime_weights):
            if index < manager.num_owned_experts:
                # A bare MoELayer has no DDP-time GTP wrapper, so natives alias the
                # optimizer parameters directly.
                assert manager.virtual_experts.gtp_leaders[fc_layer] is None
                expected_weight = parameters[index]
                expected_grad = manager.virtual_experts.native_grads[fc_layer][index]
            else:
                slot = index - manager.num_owned_experts
                expected_weight = virtual_parameters[slot]
                expected_grad = virtual_parameters[slot].main_grad
                if mxfp8:
                    # One arena serves both orientations; only one is live at a time.
                    assert (
                        expected_weight._rowwise_data.data_ptr()
                        == expected_weight._columnwise_data.data_ptr()
                    )
            assert _weight_storage_ptrs(runtime_weight) == _weight_storage_ptrs(expected_weight)
            assert runtime_weight.main_grad.data_ptr() == expected_grad.data_ptr()
            assert runtime_weight.overwrite_main_grad


def _run_full_layer_parity(
    monkeypatch,
    *,
    activation="swiglu",
    moe_latent_size=None,
    shared_expert_size=None,
    mxfp8=False,
    gtp_topology=False,
    grad_dtype=torch.float32,
    reference_dispatcher="alltoall",
    bitwise=False,
):
    """Compare one MoE layer's output and every gradient against a reference dispatcher."""
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
    from megatron.core.transformer.moe.moe_layer import MoELayer
    from megatron.core.transformer.spec_utils import get_submodules
    from megatron.core.transformer.transformer_config import TransformerConfig
    from tests.unit_tests.test_utilities import Utils

    monkeypatch.setenv("NVTE_CUTEDSL_FUSED_GROUPED_MLP", "1")
    monkeypatch.setenv("NVTE_GROUPED_LINEAR_SINGLE_PARAM", "0")
    expert_model_parallel_size = 2 if gtp_topology else 4
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1,
        expert_model_parallel_size=expert_model_parallel_size,
        expert_tensor_parallel_size=1,
        expert_gtp_remat_size=2 if gtp_topology else 1,
    )
    torch.manual_seed(1234)

    common = {
        **BASE_CONFIG,
        "hidden_size": 1024,
        "ffn_hidden_size": 1024,
        "moe_ffn_hidden_size": 1024,
        "expert_model_parallel_size": expert_model_parallel_size,
        "expert_tensor_parallel_num_weight_shards": 2 if gtp_topology else 1,
        "activation_func": F.silu if activation == "swiglu" else squared_relu,
        "gated_linear_unit": activation == "swiglu",
        "use_fused_weighted_squared_relu": activation != "swiglu",
        "moe_latent_size": moe_latent_size,
        "moe_shared_expert_intermediate_size": shared_expert_size,
        # Cover both fused (dense output recovered as compact routes) and unfused routing.
        "moe_router_fusion": reference_dispatcher == "hybridep",
    }
    if mxfp8:
        common.update(
            fp8="e4m3", fp8_recipe="mxfp8", fp8_param=True, moe_router_padding_for_quantization=True
        )
    reference_config = TransformerConfig(
        # With fewer than 256 input rows, all-to-all must pad inside the experts:
        # padding a routing mask cannot create the missing rows.
        **{
            **common,
            "moe_router_padding_for_quantization": mxfp8 and reference_dispatcher != "alltoall",
        },
        **(
            {"moe_token_dispatcher_type": "alltoall"}
            if reference_dispatcher == "alltoall"
            else {"moe_token_dispatcher_type": "flex", "moe_flex_dispatcher_backend": "hybridep"}
        ),
    )
    virtual_expert_config = TransformerConfig(
        **common,
        moe_token_dispatcher_type="flex",
        moe_flex_dispatcher_backend="hybridep",
        moe_virtual_expert_load_balance=True,
    )
    mlp_spec = get_gpt_layer_with_transformer_engine_spec(
        num_experts=4, moe_grouped_gemm=True
    ).submodules.mlp
    submodules = get_submodules(mlp_spec)

    try:

        def build(config):
            with get_fp8_context(config, is_init=True):
                return MoELayer(config, submodules).cuda()

        ref_layer = build(reference_config)
        virtual_expert_layer = build(virtual_expert_config)
        with torch.no_grad():
            for fc, linear in enumerate(
                (ref_layer.experts.linear_fc1, ref_layer.experts.linear_fc2)
            ):
                for local, expert in enumerate(ref_layer.local_expert_indices):
                    weight = linear.get_parameter(f"weight{local}")
                    generator = torch.Generator(device=weight.device).manual_seed(
                        1000 + 17 * expert + fc
                    )
                    values = torch.randn(
                        weight.shape, dtype=weight.dtype, device=weight.device, generator=generator
                    )
                    # Distinct semantic experts, including quantized data AND block scales.
                    weight.copy_(values * (0.0075 * (expert + 1)))
        for layer in (ref_layer, virtual_expert_layer):
            assert not layer.experts.linear_fc1.single_grouped_weight
            assert not layer.experts.linear_fc2.single_grouped_weight
        if mxfp8:
            # In production DDP exposes an MXFP8 parameter's main-grad buffer
            # through its distributed-weight wrapper. This focused MoELayer test
            # has no DDP wrapper, so let the ordinary latent and shared-expert
            # linears return wgrads through autograd; expert wgrads stay fused and
            # exercise the virtual-expert reduction.
            for layer in (ref_layer, virtual_expert_layer):
                for linear in _dense_linears(layer):
                    linear.fuse_wgrad_accumulation = False
        virtual_expert_layer.load_state_dict(ref_layer.state_dict())
        assert virtual_expert_layer.state_dict().keys() == ref_layer.state_dict().keys()
        _set_main_grads(ref_layer, grad_dtype)
        _set_main_grads(virtual_expert_layer, grad_dtype)

        generator = torch.Generator(device="cuda").manual_seed(1234 + torch.distributed.get_rank())
        # Dense MXFP8 linears require 32 rows; expert dispatch remains padding-heavy.
        test_input = torch.randn(
            8, 4, 1024, device="cuda", dtype=torch.bfloat16, generator=generator
        )
        upstream = torch.randn(
            test_input.shape, device="cuda", dtype=torch.bfloat16, generator=generator
        )
        manager = virtual_expert_layer.token_dispatcher._comm_manager
        manager._runtime_init(test_input)
        _assert_runtime_layout(manager, grad_dtype=grad_dtype, mxfp8=mxfp8)
        # The plan is dropped from the manager at the layer output; keep each forward's for the
        # checks below.
        plans = []
        plan_dispatch = manager.plan_dispatch

        def record_plan(*routes):
            plan_dispatch(*routes)
            plans.append(manager._plan)

        manager.plan_dispatch = record_plan
        native_only = {}
        start_grad_reduce = manager._start_grad_reduce

        def record_native_grad(fc_layer):
            # The actual GEMM partial before any peer's virtual-expert gradient is added.
            native_only[fc_layer] = manager.virtual_experts.native_grads[fc_layer].clone()
            start_grad_reduce(fc_layer)

        manager._start_grad_reduce = record_native_grad
        if mxfp8:
            # A state_dict load does not carry the quantized component storage,
            # so mirror it explicitly before comparing the two layers.
            for linear, parameters in zip(
                (ref_layer.experts.linear_fc1, ref_layer.experts.linear_fc2),
                manager.virtual_experts.parameters,
            ):
                for index, destination in enumerate(parameters):
                    source = linear.get_parameter(f"weight{index}")
                    for component in MXFP8_COMPONENTS:
                        getattr(destination, component).copy_(getattr(source, component))

        def run(layer, *, virtual_experts=None):
            hidden = test_input.detach().clone().requires_grad_(True)
            with get_fp8_context(layer.config):
                assert FP8GlobalStateManager.is_fp8_enabled() == mxfp8
                output, _ = layer(hidden)
            if virtual_experts is not None and mxfp8:
                _assert_mxfp8_prefetch_exact(virtual_experts, plans[-1], "rowwise")
            (output.float() * upstream).sum().backward()
            if virtual_experts is not None:
                for fc_layer, parameters in enumerate(virtual_experts.virtual_experts.parameters):
                    for parameter in parameters:
                        if virtual_experts.virtual_experts.gtp_leaders[fc_layer] is None:
                            # The bridge hands the reduced wgrad to the optimizer
                            # parameter through autograd's main-grad protocol.
                            assert parameter.grad is not None
                            assert parameter.grad_added_to_main_grad
                        parameter.grad = None
                assert all(
                    runtime_parameter.grad is None
                    for weights in virtual_experts.virtual_experts.runtime_weights
                    for runtime_parameter in weights
                )
                if mxfp8:
                    _assert_mxfp8_prefetch_exact(virtual_experts, plans[-1], "columnwise")
            values = [
                output.detach(),
                hidden.grad.detach(),
                layer.router.weight.grad.detach().clone(),
                _stack_linear_main_grad(layer.experts.linear_fc1),
                _stack_linear_main_grad(layer.experts.linear_fc2),
            ]
            for linear in _dense_linears(layer):
                gradient = (
                    linear.weight.main_grad
                    if linear.fuse_wgrad_accumulation
                    else linear.weight.grad
                )
                values.append(gradient.detach().clone())
            return values

        ref_values = run(ref_layer)
        if reference_dispatcher == "hybridep":
            # HybridEP owns one process-global buffer for a fixed local-expert
            # count. Baseline and virtual-expert layouts use N and 2N respectively, so
            # reinitialize it between the two sequential comparisons.
            torch.cuda.synchronize()
            torch.distributed.barrier()
            fused_a2a.reset_hybrid_ep_buffer()
            torch.distributed.barrier()
        virtual_expert_values = run(virtual_expert_layer, virtual_experts=manager)

        assert manager.moe_expert_rank_capacity_factor == 1.0
        assert not manager.over_budget.item()
        if mxfp8:
            # This input has far fewer than 256 routes per runtime expert, so
            # the comparison below runs a padding-heavy dispatch. Matching input,
            # router and expert-weight gradients prove the padding is neutral.
            assert torch.all(manager.tokens_per_expert % 256 == 0)
            num_routes = test_input.shape[0] * test_input.shape[1] * 2
            num_dispatched = manager.tokens_per_expert.sum().item()
            assert num_dispatched > num_routes
            dispatched_probs = manager.dispatched_probs[:num_dispatched]
            assert torch.count_nonzero(dispatched_probs).item() == num_routes
        # A run in which no expert was materialized would compare nothing.
        assert len(plans) == 1
        active_virtual_expert = torch.any(plans[0].experts_to_copy >= 0).to(torch.int32)
        torch.distributed.all_reduce(active_virtual_expert, op=torch.distributed.ReduceOp.MAX)
        assert active_virtual_expert.item(), "parity must exercise an active virtual-expert"

        # Gather before asserting so a failing rank cannot strand peers in a later collective.
        wrong_experts = []
        for expected in ref_values[3:5]:
            gathered = [torch.empty_like(expected) for _ in range(manager.ep_size)]
            torch.distributed.all_gather(gathered, expected, group=manager.group)
            wrong_experts.append(torch.cat(gathered))
        remote_experts = {
            expert
            for rank, row in enumerate(plans[0].experts_to_copy.tolist())
            if rank != manager.virtual_experts.rank
            for expert in row
            if expert >= 0
        }
        names = ["output", "input grad", "router grad", "FC1 main_grad", "FC2 main_grad"]
        if moe_latent_size is not None:
            names += ["latent FC1 main_grad", "latent FC2 main_grad"]
        if shared_expert_size is not None:
            names += ["shared FC1 main_grad", "shared FC2 main_grad"]
        for name, actual, expected in zip(names, virtual_expert_values, ref_values):
            # Fixed-fixture peak error / reference peak: below 4e-6 for MXFP8 FP32
            # wgrads, below 0.008 when summing separately rounded BF16 partials.
            tolerance = 0.01 if grad_dtype == torch.bfloat16 else 1e-5
            if bitwise:
                tolerance = 2e-6 if "main_grad" in name else 0
            fc = ("FC1 main_grad", "FC2 main_grad").index(name) if name in names[3:5] else None
            pairs = zip(actual, expected) if fc is not None else [(actual, expected)]
            for local, (value, reference) in enumerate(pairs):
                label = (
                    f"{name} expert {ref_layer.local_expert_indices[local]}"
                    if fc is not None
                    else name
                )
                _assert_numerical_parity(value, reference, tolerance, label)
                corruptions = {"zeroed": torch.zeros_like(value)}
                if fc is not None:
                    expert = ref_layer.local_expert_indices[local]
                    corruptions["wrong expert"] = wrong_experts[fc][(expert + 1) % 4]
                    if expert in remote_experts:
                        assert (
                            native_only[fc][local].norm() > 0
                        ), f"{label}: missing native contribution"
                        corruptions["omitted remote"] = native_only[fc][local]
                for corruption, broken in corruptions.items():
                    with pytest.raises(AssertionError):
                        _assert_numerical_parity(
                            broken, reference, tolerance, f"{corruption} {label}"
                        )
    finally:
        # Release the arenas while their communicator is alive, then destroy the
        # process-global HybridEP buffer in lockstep across ranks.
        FP8GlobalStateManager.reset()
        VirtualExpertLoadBalancer.finalize()
        Utils.destroy_model_parallel()
        torch.cuda.synchronize()
        torch.distributed.barrier()
        fused_a2a.reset_hybrid_ep_buffer()
        torch.cuda.synchronize()
        torch.distributed.barrier()


def _run_repeated_mtp_parity(monkeypatch):
    """Compare a two-depth shared MTP block against ordinary HybridEP end to end."""
    from megatron.core.models.gpt import GPTModel
    from megatron.core.models.gpt.gpt_layer_specs import (
        get_gpt_layer_with_transformer_engine_spec,
        get_gpt_mtp_block_spec,
    )
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.enums import AttnBackend
    from megatron.core.transformer.moe.moe_layer import MoELayer
    from megatron.core.transformer.transformer_config import TransformerConfig
    from tests.unit_tests.test_utilities import Utils

    monkeypatch.setenv("NVTE_CUTEDSL_FUSED_GROUPED_MLP", "1")
    monkeypatch.setenv("NVTE_GROUPED_LINEAR_SINGLE_PARAM", "0")
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1, expert_model_parallel_size=4, expert_tensor_parallel_size=1
    )
    model_parallel_cuda_manual_seed(1234)
    torch.manual_seed(1234)

    common = {
        **BASE_CONFIG,
        "hidden_size": 128,
        "ffn_hidden_size": 256,
        "moe_ffn_hidden_size": 256,
        "kv_channels": 16,
        "expert_model_parallel_size": 4,
        "activation_func": F.silu,
        "gated_linear_unit": True,
        "hidden_dropout": 0.0,
        "attention_dropout": 0.0,
        "attention_backend": AttnBackend.unfused,
        "mtp_num_layers": 2,
        "mtp_use_repeated_layer": True,
        "mtp_loss_scaling_factor": 0.1,
        "calculate_per_token_loss": True,
    }
    layer_spec = get_gpt_layer_with_transformer_engine_spec(num_experts=4, moe_grouped_gemm=True)

    def build(virtual_expert_load_balance):
        config = TransformerConfig(
            **common,
            moe_token_dispatcher_type="flex",
            moe_flex_dispatcher_backend="hybridep",
            moe_virtual_expert_load_balance=virtual_expert_load_balance,
        )
        return GPTModel(
            config=config,
            transformer_layer_spec=layer_spec,
            vocab_size=128,
            max_sequence_length=8,
            pre_process=True,
            post_process=True,
            share_embeddings_and_output_weights=False,
            mtp_block_spec=get_gpt_mtp_block_spec(config, layer_spec, use_transformer_engine=True),
        ).cuda()

    def initialize_main_grads(model):
        for parameter in model.parameters():
            _set_main_grad(parameter)
            # Ordinary Megatron DDP zeroes persistent main-grad buffers and TE
            # accumulates every tied-layer use into them. ``overwrite`` suits
            # only a single-use synthetic forward: with two outstanding autograd
            # contexts both would otherwise overwrite the same buffer.
            del parameter.overwrite_main_grad

    def snapshot(model):
        return {
            name: (
                (
                    parameter.grad.detach().clone()
                    if parameter.grad is not None
                    and not getattr(parameter, "grad_added_to_main_grad", False)
                    else None
                ),
                parameter.main_grad.detach().clone(),
                bool(getattr(parameter, "grad_added_to_main_grad", False)),
            )
            for name, parameter in model.named_parameters()
        }

    generator = torch.Generator(device="cuda").manual_seed(5678)
    batch, sequence = 2, 8
    input_ids = torch.randint(0, 128, (batch, sequence), generator=generator, device="cuda")
    labels = torch.randint(0, 128, (batch, sequence), generator=generator, device="cuda")
    position_ids = torch.arange(sequence, device="cuda").unsqueeze(0).expand(batch, -1)
    loss_mask = torch.ones((batch, sequence), device="cuda")
    reference_model = virtual_expert_model = None

    def forward(model):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return model(
                input_ids, position_ids, attention_mask=None, labels=labels, loss_mask=loss_mask
            )

    try:
        reference_model = build(False)
        virtual_expert_model = build(True)
        virtual_expert_model.load_state_dict(reference_model.state_dict())
        assert virtual_expert_model.state_dict().keys() == reference_model.state_dict().keys()
        initialize_main_grads(reference_model)
        initialize_main_grads(virtual_expert_model)

        reference_loss = forward(reference_model)
        reference_loss.sum().backward()
        reference_gradients = snapshot(reference_model)

        # Baseline and virtual-expert layouts use N and 2N local runtime experts, so
        # reinitialize HybridEP's process-global transport buffer between them.
        torch.cuda.synchronize()
        torch.distributed.barrier()
        fused_a2a.reset_hybrid_ep_buffer()
        torch.distributed.barrier()

        moe_layers = [
            module
            for module in virtual_expert_model.mtp.layers[0].mtp_model_layer.modules()
            if isinstance(module, MoELayer)
        ]
        assert len(moe_layers) == 1, "repeated MTP must exercise one shared MoE layer"
        manager = moe_layers[0].token_dispatcher._comm_manager
        # Two depths share one layer, so the second plan must not clobber the first:
        # each backward reads its own experts_to_copy.
        plans = []
        plan_dispatch = manager.plan_dispatch

        def record_plan(*routes):
            plan_dispatch(*routes)
            plans.append(manager._plan)

        manager.plan_dispatch = record_plan
        virtual_expert_loss = forward(virtual_expert_model)
        assert len(plans) == 2
        assert plans[0].experts_to_copy.data_ptr() != plans[1].experts_to_copy.data_ptr()
        active_virtual_expert = torch.stack(
            [torch.any(plan.experts_to_copy >= 0) for plan in plans]
        ).any()
        torch.distributed.all_reduce(active_virtual_expert, op=torch.distributed.ReduceOp.MAX)
        assert (
            active_virtual_expert.item()
        ), "repeated MTP parity must exercise an active virtual-expert"

        virtual_expert_loss.sum().backward()
        virtual_expert_gradients = snapshot(virtual_expert_model)
        assert manager._plan is None

        torch.testing.assert_close(
            virtual_expert_loss,
            reference_loss,
            rtol=0,
            atol=0,
            msg=lambda msg: f"repeated MTP loss must be bitwise equal: {msg}",
        )
        assert virtual_expert_gradients.keys() == reference_gradients.keys()
        for name in virtual_expert_gradients:
            virtual_expert_grad, virtual_expert_main_grad, virtual_expert_fused = (
                virtual_expert_gradients[name]
            )
            reference_grad, reference_main_grad, reference_fused = reference_gradients[name]
            assert virtual_expert_fused == reference_fused, name
            assert (virtual_expert_grad is None) == (reference_grad is None), name
            if virtual_expert_grad is not None:
                torch.testing.assert_close(
                    virtual_expert_grad,
                    reference_grad,
                    rtol=0,
                    atol=0,
                    msg=lambda msg, name=name: f"{name} autograd gradient: {msg}",
                )
            # Virtual-expert expert wgrads sum independently rounded FP32 partials in a
            # different order while retaining the same mathematical result.
            rtol, atol = (2e-7, 2e-6) if ".experts." in name else (0, 0)
            assert torch.isfinite(virtual_expert_main_grad).all(), name
            torch.testing.assert_close(
                virtual_expert_main_grad,
                reference_main_grad,
                rtol=rtol,
                atol=atol,
                msg=lambda msg, name=name: f"{name} main_grad: {msg}",
            )
    finally:
        # The losses keep the autograd graphs, whose TE contexts hold the runtime parameters
        # aliasing the symmetric arenas; drop them so the arenas are released with the group.
        reference_loss = virtual_expert_loss = None
        del reference_model, virtual_expert_model
        VirtualExpertLoadBalancer.finalize()
        Utils.destroy_model_parallel()
        torch.cuda.synchronize()
        torch.distributed.barrier()
        fused_a2a.reset_hybrid_ep_buffer()
        torch.cuda.synchronize()
        torch.distributed.barrier()


@pytest.mark.internal
@requires_four_ranks
def test_virtual_expert_hybridep_production_recipe_matches_alltoall(monkeypatch):
    """Cover MXFP8 compute/weights, the EP2/GTP2 topology, BF16 grads, and latent MoE
    with shared experts (which must see the full-width layer input, not the latent one)."""
    try:
        from transformer_engine.pytorch.ops import ScaledSReLU  # noqa: F401
    except ImportError:
        pytest.skip("Transformer Engine ScaledSReLU is required")
    _run_full_layer_parity(
        monkeypatch,
        activation="squared_relu",
        moe_latent_size=640,
        shared_expert_size=1024,
        mxfp8=True,
        gtp_topology=True,
        grad_dtype=torch.bfloat16,
    )


@pytest.mark.internal
@requires_four_ranks
def test_virtual_expert_hybridep_bf16_semantics_match_hybridep(monkeypatch):
    """Require bitwise HybridEP semantics and tightly bounded expert wgrad reduction noise."""
    _run_full_layer_parity(monkeypatch, reference_dispatcher="hybridep", bitwise=True)


@pytest.mark.internal
@requires_four_ranks
def test_virtual_expert_hybridep_mxfp8_matches_hybridep(monkeypatch):
    """Bound MX execution noise while requiring byte-exact MXFP8 weight transport."""
    _run_full_layer_parity(monkeypatch, mxfp8=True, reference_dispatcher="hybridep")


@pytest.mark.internal
@requires_four_ranks
def test_virtual_expert_hybridep_repeated_mtp_semantics_match_hybridep(monkeypatch):
    """Preserve two-depth tied-layer MTP loss and every model gradient."""
    _run_repeated_mtp_parity(monkeypatch)
