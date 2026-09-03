# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""End-to-end gradient parity for the replica_hybridep flex dispatcher.

Run on one four-GPU NVLink node::

    uv run python -m torch.distributed.run --nproc-per-node 4 -m pytest -q \
      tests/unit_tests/transformer/moe/test_replica_hybridep.py

Each case builds the same MoE layer twice -- once on a reference dispatcher and
once on replica_hybridep -- loads identical weights, and compares the output and
every training gradient. Planner and bridge internals are covered
process-locally in ``test_replica_planner.py``; the transport kernels in
``test_replica_weight_triton.py``.

A bare ``MoELayer`` carries no DDP-time GTP wrapper, so the bridge's GTP gather and
reduce-scatter path is exercised only by the training recipe with
``--expert-tensor-parallel-num-weight-shards > 1``, not here.
"""

import os

import pytest
import torch
import torch.nn.functional as F

from megatron.core.activations import squared_relu
from megatron.core.transformer.moe import fused_a2a

MXFP8_COMPONENTS = (
    "_rowwise_data",
    "_rowwise_scale_inv",
    "_columnwise_data",
    "_columnwise_scale_inv",
)

requires_four_ranks = pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "1")) != 4
    or not torch.cuda.is_available()
    or not fused_a2a.HAVE_HYBRIDEP,
    reason="replica_hybridep parity requires a 4-rank torchrun launch with HybridEP",
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


def _set_main_grads(layer, dtype):
    for linear in (layer.experts.linear_fc1, layer.experts.linear_fc2):
        for index in range(linear.num_gemms):
            _set_main_grad(linear.get_parameter(f"weight{index}"), dtype)
    if layer.config.moe_latent_size is not None:
        _set_main_grad(layer.fc1_latent_proj.weight, dtype)
        _set_main_grad(layer.fc2_latent_proj.weight, dtype)


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


def _assert_mxfp8_prefetch_exact(bridge, orientation):
    """Check every active virtual MXFP8 component byte-for-byte against its owning rank."""
    components = MXFP8_COMPONENTS[:2] if orientation == "rowwise" else MXFP8_COMPONENTS[2:]
    errors = []
    for index, projection in enumerate(bridge.projections):
        for component in components:
            local = torch.stack(
                tuple(getattr(source, component) for source in projection.parameters)
            )
            gathered = [torch.empty_like(local) for _ in range(bridge.world_size)]
            torch.distributed.all_gather(gathered, local, group=bridge.group)
            for slot, expert in enumerate(bridge.last_plan.experts_to_copy[bridge.rank].tolist()):
                if expert < 0:
                    continue
                owner, owned = divmod(expert, bridge.num_local_experts)
                if not torch.equal(
                    getattr(projection.virtual_weights[slot], component), gathered[owner][owned]
                ):
                    errors.append(f"projection={index} {component} slot={slot} expert={expert}")
    any_error = torch.tensor(int(bool(errors)), dtype=torch.int32, device=bridge.device)
    torch.distributed.all_reduce(any_error, op=torch.distributed.ReduceOp.MAX, group=bridge.group)
    assert not any_error.item(), f"rank {bridge.rank} {orientation} MXFP8 prefetch mismatch: " + (
        ", ".join(errors) if errors else "reported by another rank"
    )


def _assert_bridge_layout(bridge, *, grad_dtype, mxfp8):
    """Check that the runtime weights and grads TE executes against alias bridge storage."""
    assert bridge.workspace.grad_arena.dtype == grad_dtype
    for projection, runtime_weights in zip(
        bridge.projections, (bridge.runtime_fc1_weights, bridge.runtime_fc2_weights)
    ):
        assert len(runtime_weights) == bridge.num_runtime_experts
        assert projection.virtual_grad.dtype == grad_dtype
        for index, runtime_weight in enumerate(runtime_weights):
            if index < bridge.num_local_experts:
                # A bare MoELayer has no DDP-time GTP wrapper, so natives alias the
                # optimizer parameters directly.
                assert projection.gtp_leader is None
                expected_weight = projection.parameters[index]
                expected_grad = projection.native_grad[index]
            else:
                slot = index - bridge.num_local_experts
                expected_weight = projection.virtual_weights[slot]
                expected_grad = projection.virtual_grad[slot]
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
    mxfp8=False,
    gtp=False,
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
    expert_model_parallel_size = 2 if gtp else 4
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1,
        expert_model_parallel_size=expert_model_parallel_size,
        expert_tensor_parallel_size=1,
        expert_gtp_remat_size=2 if gtp else 1,
    )
    if gtp:
        from megatron.core.tensor_parallel.generalized_tensor_parallelism import update_gtp_config

        # This test isolates the bridge's explicit materialize-before-exchange
        # dependency; the production script covers linked async GTP chains.
        update_gtp_config(
            weight_prefetch=False,
            async_reduction=False,
            reduce_scatter_with_fp32_accumulation=(grad_dtype == torch.bfloat16),
        )
    torch.manual_seed(1234)

    bf16_grads = grad_dtype == torch.bfloat16
    common = {
        **BASE_CONFIG,
        "hidden_size": 1024,
        "ffn_hidden_size": 1024,
        "moe_ffn_hidden_size": 1024,
        "expert_model_parallel_size": expert_model_parallel_size,
        "expert_tensor_parallel_num_weight_shards": 2 if gtp else 1,
        "activation_func": F.silu if activation == "swiglu" else squared_relu,
        "gated_linear_unit": activation == "swiglu",
        "use_fused_weighted_squared_relu": activation != "swiglu",
        "moe_latent_size": moe_latent_size,
    }
    if mxfp8:
        common.update(
            fp8="e4m3", fp8_recipe="mxfp8", fp8_param=True, moe_router_padding_for_quantization=True
        )
    reference_config = TransformerConfig(
        **common,
        **(
            {"moe_token_dispatcher_type": "alltoall"}
            if reference_dispatcher == "alltoall"
            else {"moe_token_dispatcher_type": "flex", "moe_flex_dispatcher_backend": "hybridep"}
        ),
    )
    replica_config = TransformerConfig(
        **common,
        moe_token_dispatcher_type="flex",
        moe_flex_dispatcher_backend="replica_hybridep",
        grad_reduce_in_bf16=bf16_grads,
        ddp_reduce_scatter_with_fp32_accumulation=bf16_grads,
        gtp_remat_reduce_scatter_with_fp32_accumulation=bf16_grads and gtp,
    )
    mlp_spec = get_gpt_layer_with_transformer_engine_spec(
        num_experts=4, moe_grouped_gemm=True
    ).submodules.mlp
    submodules = get_submodules(mlp_spec)

    try:
        if mxfp8:
            from transformer_engine.common.recipe import MXFP8BlockScaling
            from transformer_engine.pytorch import fp8_model_init

            def build(config):
                with fp8_model_init(enabled=True, recipe=MXFP8BlockScaling()):
                    return MoELayer(config, submodules).cuda()

        else:

            def build(config):
                return MoELayer(config, submodules).cuda()

        ref_layer = build(reference_config)
        replica_layer = build(replica_config)
        for layer in (ref_layer, replica_layer):
            assert not layer.experts.linear_fc1.single_grouped_weight
            assert not layer.experts.linear_fc2.single_grouped_weight
        if mxfp8 and moe_latent_size is not None:
            # In production DDP exposes an MXFP8 parameter's main-grad buffer
            # through its distributed-weight wrapper. This focused MoELayer test
            # has no DDP wrapper, so let the two ordinary latent linears return
            # wgrads through autograd; expert wgrads stay fused and exercise the
            # replica reduction.
            for layer in (ref_layer, replica_layer):
                layer.fc1_latent_proj.fuse_wgrad_accumulation = False
                layer.fc2_latent_proj.fuse_wgrad_accumulation = False
        replica_layer.load_state_dict(ref_layer.state_dict())
        assert replica_layer.state_dict().keys() == ref_layer.state_dict().keys()
        _set_main_grads(ref_layer, grad_dtype)
        _set_main_grads(replica_layer, grad_dtype)

        bridge = replica_layer.token_dispatcher._comm_manager._bridge
        _assert_bridge_layout(bridge, grad_dtype=grad_dtype, mxfp8=mxfp8)
        if mxfp8:
            # A state_dict load does not carry the quantized component storage,
            # so mirror it explicitly before comparing the two layers.
            for linear, replica in zip(
                (ref_layer.experts.linear_fc1, ref_layer.experts.linear_fc2), bridge.projections
            ):
                for index, destination in enumerate(replica.parameters):
                    source = linear.get_parameter(f"weight{index}")
                    for component in MXFP8_COMPONENTS:
                        getattr(destination, component).copy_(getattr(source, component))

        torch.manual_seed(1234)
        test_input = torch.randn(2, 4, 1024, device="cuda", dtype=torch.bfloat16)

        def run(layer, *, replica_bridge=None):
            hidden = test_input.detach().clone().requires_grad_(True)
            output, _ = layer(hidden)
            if replica_bridge is not None and mxfp8:
                _assert_mxfp8_prefetch_exact(replica_bridge, "rowwise")
            output.float().sum().backward()
            if replica_bridge is not None:
                for projection in replica_bridge.projections:
                    for parameter in projection.parameters:
                        if projection.gtp_leader is None:
                            # The bridge hands the reduced wgrad to the optimizer
                            # parameter through autograd's main-grad protocol.
                            assert parameter.grad is not None
                            assert parameter.grad_added_to_main_grad
                        parameter.grad = None
                assert all(
                    runtime_parameter.grad is None
                    for projection in replica_bridge.projections
                    for runtime_parameter in projection.runtime_parameters
                )
                if mxfp8:
                    _assert_mxfp8_prefetch_exact(replica_bridge, "columnwise")
            values = [
                output.detach(),
                hidden.grad.detach(),
                layer.router.weight.grad.detach().clone(),
                _stack_linear_main_grad(layer.experts.linear_fc1),
                _stack_linear_main_grad(layer.experts.linear_fc2),
            ]
            if moe_latent_size is not None:
                for projection in (layer.fc1_latent_proj, layer.fc2_latent_proj):
                    gradient = (
                        projection.weight.main_grad
                        if projection.fuse_wgrad_accumulation
                        else projection.weight.grad
                    )
                    values.append(gradient.detach().clone())
            return values

        ref_values = run(ref_layer)
        if reference_dispatcher == "hybridep":
            # HybridEP owns one process-global buffer for a fixed local-expert
            # count. Baseline and replica layouts use N and 2N respectively, so
            # reinitialize it between the two sequential comparisons.
            torch.cuda.synchronize()
            torch.distributed.barrier()
            fused_a2a.reset_hybrid_ep_buffer()
            torch.distributed.barrier()
        replica_values = run(replica_layer, replica_bridge=bridge)

        manager = replica_layer.token_dispatcher._comm_manager
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
        # A run in which no expert was replicated would compare nothing.
        active_replica = torch.any(bridge.last_plan.experts_to_copy >= 0).to(torch.int32)
        torch.distributed.all_reduce(active_replica, op=torch.distributed.ReduceOp.MAX)
        assert active_replica.item(), "parity must exercise an active replica"

        names = ["output", "input grad", "router grad", "FC1 main_grad", "FC2 main_grad"]
        if moe_latent_size is not None:
            names += ["latent FC1 main_grad", "latent FC2 main_grad"]
        for name, actual, expected in zip(names, replica_values, ref_values):
            if bitwise and "main_grad" not in name:
                tolerance = dict(rtol=0, atol=0)
            elif bitwise:
                # A replicated expert's wgrad sums independently rounded FP32
                # partials. That changes addition order, not the gradient.
                tolerance = dict(rtol=2e-7, atol=2e-6)
            elif mxfp8:
                # Replica placement changes the token population of each MX
                # quantization block, so per-element absolute tolerances are not
                # stable across those block boundaries. Bound execution noise
                # while the raw weights and scales stay byte-exact above; these
                # limits sit far below the corruption a wrong expert or scale
                # mapping produces.
                atol = 32.0 if "main_grad" in name else 16.0 if name == "router grad" else 0.75
                tolerance = dict(rtol=0.2, atol=atol)
            else:
                # Different dispatchers may reorder BF16 reductions even when
                # replica planning leaves the mathematical result unchanged.
                tolerance = dict(rtol=2e-2, atol=2e-2)
            torch.testing.assert_close(
                actual, expected, **tolerance, msg=lambda msg: f"{name}: {msg}"
            )
    finally:
        # Replica bridges own CUDA work that can reference the HybridEP
        # execution context. Finalize them first, then destroy the
        # process-global buffer in lockstep across ranks.
        Utils.destroy_model_parallel()
        torch.cuda.synchronize()
        torch.distributed.barrier()
        fused_a2a.reset_hybrid_ep_buffer()
        torch.cuda.synchronize()
        torch.distributed.barrier()
        if gtp:
            update_gtp_config(
                weight_prefetch=True,
                async_reduction=True,
                reduce_scatter_with_fp32_accumulation=False,
            )


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

    def build(backend):
        config = TransformerConfig(
            **common, moe_token_dispatcher_type="flex", moe_flex_dispatcher_backend=backend
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
    reference_model = replica_model = None

    def forward(model):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return model(
                input_ids, position_ids, attention_mask=None, labels=labels, loss_mask=loss_mask
            )

    try:
        reference_model = build("hybridep")
        replica_model = build("replica_hybridep")
        replica_model.load_state_dict(reference_model.state_dict())
        assert replica_model.state_dict().keys() == reference_model.state_dict().keys()
        initialize_main_grads(reference_model)
        initialize_main_grads(replica_model)

        reference_loss = forward(reference_model)
        reference_loss.sum().backward()
        reference_gradients = snapshot(reference_model)

        # Baseline and replica layouts use N and 2N local runtime experts, so
        # reinitialize HybridEP's process-global transport buffer between them.
        torch.cuda.synchronize()
        torch.distributed.barrier()
        fused_a2a.reset_hybrid_ep_buffer()
        torch.distributed.barrier()

        moe_layers = [
            module
            for module in replica_model.mtp.layers[0].mtp_model_layer.modules()
            if isinstance(module, MoELayer)
        ]
        assert len(moe_layers) == 1, "repeated MTP must exercise one shared MoE layer"
        manager = moe_layers[0].token_dispatcher._comm_manager
        # Two depths share one layer, so the second plan must not clobber the first:
        # each backward reads its own experts_to_copy.
        plans = []
        plan_dispatch = manager.plan_dispatch

        def record_plan(hidden_states):
            plan_dispatch(hidden_states)
            plans.append(manager._plan)

        manager.plan_dispatch = record_plan
        replica_loss = forward(replica_model)
        assert len(plans) == 2
        assert plans[0].experts_to_copy.data_ptr() != plans[1].experts_to_copy.data_ptr()
        active_replica = torch.stack([torch.any(plan.experts_to_copy >= 0) for plan in plans]).any()
        torch.distributed.all_reduce(active_replica, op=torch.distributed.ReduceOp.MAX)
        assert active_replica.item(), "repeated MTP parity must exercise an active replica"

        replica_loss.sum().backward()
        replica_gradients = snapshot(replica_model)
        assert manager._plan is None and manager._context is None

        torch.testing.assert_close(
            replica_loss,
            reference_loss,
            rtol=0,
            atol=0,
            msg=lambda msg: f"repeated MTP loss must be bitwise equal: {msg}",
        )
        assert replica_gradients.keys() == reference_gradients.keys()
        for name in replica_gradients:
            replica_grad, replica_main_grad, replica_fused = replica_gradients[name]
            reference_grad, reference_main_grad, reference_fused = reference_gradients[name]
            assert replica_fused == reference_fused, name
            assert (replica_grad is None) == (reference_grad is None), name
            if replica_grad is not None:
                torch.testing.assert_close(
                    replica_grad,
                    reference_grad,
                    rtol=0,
                    atol=0,
                    msg=lambda msg, name=name: f"{name} autograd gradient: {msg}",
                )
            # Replica expert wgrads sum independently rounded FP32 partials in a
            # different order while retaining the same mathematical result.
            rtol, atol = (2e-7, 2e-6) if ".experts." in name else (0, 0)
            assert torch.isfinite(replica_main_grad).all(), name
            torch.testing.assert_close(
                replica_main_grad,
                reference_main_grad,
                rtol=rtol,
                atol=atol,
                msg=lambda msg, name=name: f"{name} main_grad: {msg}",
            )
    finally:
        del reference_model, replica_model
        Utils.destroy_model_parallel()
        torch.cuda.synchronize()
        torch.distributed.barrier()
        fused_a2a.reset_hybrid_ep_buffer()
        torch.cuda.synchronize()
        torch.distributed.barrier()


@pytest.mark.internal
@requires_four_ranks
def test_replica_hybridep_production_recipe_matches_alltoall(monkeypatch):
    """Cover the production combination: MXFP8 weights, GTP experts, BF16 grads, latent MoE."""
    try:
        from transformer_engine.pytorch.ops import ScaledSReLU  # noqa: F401
    except ImportError:
        pytest.skip("Transformer Engine ScaledSReLU is required")
    _run_full_layer_parity(
        monkeypatch,
        activation="squared_relu",
        moe_latent_size=640,
        mxfp8=True,
        gtp=True,
        grad_dtype=torch.bfloat16,
    )


@pytest.mark.internal
@requires_four_ranks
def test_replica_hybridep_bf16_semantics_match_hybridep(monkeypatch):
    """Require bitwise HybridEP semantics and tightly bounded expert wgrad reduction noise."""
    _run_full_layer_parity(monkeypatch, reference_dispatcher="hybridep", bitwise=True)


@pytest.mark.internal
@requires_four_ranks
def test_replica_hybridep_mxfp8_matches_hybridep(monkeypatch):
    """Bound MX execution noise while requiring byte-exact MXFP8 weight transport."""
    _run_full_layer_parity(monkeypatch, mxfp8=True, reference_dispatcher="hybridep")


@pytest.mark.internal
@requires_four_ranks
def test_replica_hybridep_repeated_mtp_semantics_match_hybridep(monkeypatch):
    """Preserve two-depth tied-layer MTP loss and every model gradient."""
    _run_repeated_mtp_parity(monkeypatch)
