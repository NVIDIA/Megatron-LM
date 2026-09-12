# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Four-rank virtual-expert/GTP storage and training parity (also runs on GB200)."""

import gc
import os

import pytest
import torch
import torch.nn.functional as F
from transformer_engine.pytorch.quantization import FP8GlobalStateManager

from megatron.core import parallel_state as ps
from megatron.core.activations import squared_relu
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.fp8_utils import get_fp8_context, is_mxfp8tensor
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel import generalized_tensor_parallelism as gtp
from megatron.core.tensor_parallel.random import (
    initialize_rng_tracker,
    model_parallel_cuda_manual_seed,
)
from megatron.core.transformer.identity_op import IdentityFuncOp, IdentityOp
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe import fused_a2a
from megatron.core.transformer.moe.virtual_expert_load_balancer import (
    VirtualExpertLoadBalancer,
    _VirtualExperts,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import MoETransformerLayer
from tests.unit_tests.test_utilities import Utils

pytestmark = [
    pytest.mark.internal,
    pytest.mark.launch_on_gb200,
    pytest.mark.skipif(
        int(os.environ.get("WORLD_SIZE", "1")) != 4 or not fused_a2a.HAVE_HYBRIDEP,
        reason="requires four ranks and HybridEP",
    ),
]


class _ExpertStack(MegatronModule):
    def __init__(self, config, pg_collection, repeat_last=False):
        super().__init__(config)
        self.repeat_last = repeat_last
        spec = get_gpt_layer_with_transformer_engine_spec(
            num_experts=config.num_moe_experts, moe_grouped_gemm=True
        )
        spec.submodules.input_layernorm = IdentityOp
        spec.submodules.self_attention = IdentityOp
        spec.submodules.self_attn_bda = IdentityFuncOp
        self.layers = torch.nn.ModuleList(
            [
                MoETransformerLayer(
                    config,
                    spec.submodules,
                    layer_number=i + 1,
                    pg_collection=pg_collection,
                    name=f"layers.{i}",
                )
                for i in range(config.num_layers)
            ]
        )

    def forward(self, hidden_states):
        for layer in self.layers:
            hidden_states, _ = layer(hidden_states, attention_mask=None)
        if self.repeat_last:
            hidden_states, _ = self.layers[-1](hidden_states, attention_mask=None)
        return hidden_states

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Use each layer's EP/GTP checkpoint mapping across the ordinary ModuleList."""
        return {
            key: value
            for index, layer in enumerate(self.layers)
            for key, value in layer.sharded_state_dict(
                f"{prefix}layers.{index}.", sharded_offsets, metadata
            ).items()
        }


def _expert_config(mxfp8, **overrides):
    """Share the small expert-stack configuration across storage and training parity tests."""
    config = dict(
        num_layers=3,
        hidden_size=512,
        ffn_hidden_size=512,
        num_attention_heads=8,
        num_moe_experts=8,
        moe_ffn_hidden_size=512,
        moe_router_topk=2,
        expert_model_parallel_size=2,
        expert_gtp_weight_remat_size=2,
        gtp_weight_remat_size=2 if mxfp8 else 1,
        gtp_remat_opt_in_modules=["moe_latent_proj"] if mxfp8 else [],
        moe_router_load_balancing_type="none",
        moe_router_dtype="fp32",
        moe_grouped_gemm=True,
        moe_single_grouped_weight=False,
        moe_token_dispatcher_type="flex",
        moe_flex_dispatcher_backend="hybridep",
        moe_virtual_expert_load_balance=True,
        use_transformer_engine_op_fuser=True,
        gradient_accumulation_fusion=True,
        add_bias_linear=False,
        gated_linear_unit=not mxfp8,
        activation_func=squared_relu if mxfp8 else F.silu,
        use_fused_weighted_squared_relu=mxfp8,
        moe_latent_size=512 if mxfp8 else None,
        moe_shared_expert_intermediate_size=512 if mxfp8 else None,
        bf16=True,
        params_dtype=torch.bfloat16,
        use_cpu_initialization=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        bias_dropout_fusion=False,
        fp8="e4m3" if mxfp8 else None,
        fp8_recipe="mxfp8",
        fp8_param=mxfp8,
        moe_router_padding_for_quantization=mxfp8,
    )
    return TransformerConfig(**(config | overrides))


@pytest.mark.parametrize("mxfp8", [False, True], ids=["bf16", "mxfp8"])
def test_virtual_expert_gtp_persistent_wgrads_match_recycled_scratch(monkeypatch, mxfp8):
    """Real DDP/GTP gradients and outputs agree with recycled scratch across changed inputs."""
    if mxfp8 and torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("MXFP8 requires Blackwell")
    monkeypatch.setenv("NVTE_CUTEDSL_FUSED_GROUPED_MLP", "1")
    Utils.initialize_distributed()
    ps.destroy_model_parallel()
    gtp._GTP_PARAMS.clear()
    ps.initialize_model_parallel(
        expert_model_parallel_size=2, expert_gtp_remat_size=2, gtp_remat_size=2 if mxfp8 else 1
    )
    initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
    model_parallel_cuda_manual_seed(1234)
    config = _expert_config(mxfp8)
    pg = ProcessGroupCollection.use_mpu_process_groups()
    with get_fp8_context(config, is_init=True):
        module = _ExpertStack(config, pg, repeat_last=mxfp8).cuda()
    model = DDP(
        config=config,
        ddp_config=DistributedDataParallelConfig(
            grad_reduce_in_fp32=False,
            overlap_grad_reduce=True,
            check_for_nan_in_grad=False,
            average_in_collective=False,
        ),
        module=module,
        pg_collection=pg,
    )
    gtp.classify_gtp_remat_chains(
        model, cuda_graph_modules=config.cuda_graph_modules, cuda_graph_impl="none"
    )
    monkeypatch.setattr(gtp.GTP_CONFIG, "async_reduction", mxfp8)
    monkeypatch.setattr(gtp.GTP_CONFIG, "weight_prefetch", True)
    params = list(model.parameters())
    experts = [p for p in params if getattr(p, "is_routed_expert", False)]
    assert experts and all(p.gtp_remat_size == 2 for p in experts)
    assert all("ungraphed" in p.chain_id for p in experts)
    assert all(is_mxfp8tensor(p) == mxfp8 and p.main_grad.dtype == torch.bfloat16 for p in experts)

    def run(seed):
        model.zero_grad_buffer()
        rng = torch.Generator(device="cuda").manual_seed(seed + ps.get_expert_model_parallel_rank())
        x = torch.randn(
            32,
            1,
            config.hidden_size,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
            generator=rng,
        )
        dy = torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=rng)
        model.set_is_first_microbatch()
        with get_fp8_context(config):
            y = model(x)
        saved_output = y.detach().clone()
        saved_input = x.detach().clone()
        # Distinct upstream gradients exercise every output component and router gradient.
        (y.float() * dy).sum().backward()
        gtp.wait_for_gtp_grad_reduction_on_current_stream()
        model.finish_grad_sync()
        torch.cuda.synchronize()
        return saved_output, x.grad.clone(), [p.main_grad.clone() for p in params], saved_input

    get_weight_table = _VirtualExperts.get_weight_table
    reference_layers = set()

    def recycled_reference_table(owner, fc_layer, key, sources):
        # The legacy control uses moving scratch; explicitly rebuild only its gradient tables.
        reference_layers.add(owner)
        if key == "grad":
            owner._tables[fc_layer].pop(key, None)
        return get_weight_table(owner, fc_layer, key, sources)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(_VirtualExperts, "get_weight_table", recycled_reference_table)
            for param_type in {type(p) for p in experts}:
                get_wgrad_tensor = param_type.get_wgrad_tensor
                patch.setattr(
                    param_type,
                    "get_wgrad_tensor",
                    lambda p, method=get_wgrad_tensor, **kwargs: method(p),
                )
            refs = [run(seed) for seed in (21, 22)]
            assert not any(hasattr(p, "_gtp_wgrad_ring_slot") for p in experts)
        while reference_layers:
            for tables in reference_layers.pop()._tables:
                tables.pop("grad", None)
        for result in refs:
            for tensor in (result[0], result[1], *result[2]):
                assert torch.isfinite(tensor).all() and tensor.float().norm() > 0

        slots = None
        for seed in (21, 22, 21, 22):
            result = run(seed)
            reference = refs[seed - 21]
            torch.testing.assert_close(result[3], reference[3], atol=0, rtol=0)
            for name, actual, expected in zip(
                ("output", "input gradient", *(name for name, _ in model.named_parameters())),
                (result[0], result[1], *result[2]),
                (reference[0], reference[1], *reference[2]),
            ):
                rel_error = (actual.float() - expected.float()).norm() / expected.float().norm()
                assert rel_error < 0.02, f"{name}: relative L2 error {rel_error}"
                torch.testing.assert_close(
                    actual, expected, atol=0.01, rtol=0.02, msg=lambda msg: f"{name}: {msg}"
                )
            pointers = [p._gtp_wgrad_ring_slot.tensor.data_ptr() for p in experts]
            if slots is None:
                slots = pointers
                # Three layers share two slots per FC role/expert/shape domain.
                assert len(set(slots)) < len(slots)

                def check_static_table(owner, fc_layer, key, sources):
                    tables = tuple(owner._tables[fc_layer].values())
                    result = get_weight_table(owner, fc_layer, key, sources)
                    assert any(result is tensor for tensor, _ in tables)
                    return result

                monkeypatch.setattr(_VirtualExperts, "get_weight_table", check_static_table)
            else:
                assert pointers == slots
    finally:
        torch.cuda.synchronize()
        del model, module, params, experts
        reference_layers.clear()
        FP8GlobalStateManager.reset()
        gc.collect()
        VirtualExpertLoadBalancer.finalize()
        gtp.reset_gtp_state()
        gtp._GTP_PARAMS.clear()
        ps.destroy_model_parallel()
        fused_a2a.reset_hybrid_ep_buffer()


@pytest.mark.parametrize("expert_gtp", [2, 1], ids=["egtp2", "edp2"])
@pytest.mark.parametrize("mxfp8", [False, True], ids=["bf16", "mxfp8"])
def test_virtual_expert_gtp_training_matches_hybridep(
    monkeypatch, tmp_path_dist_ckpt, mxfp8, expert_gtp
):
    """Compare complete DDP/GTP training, Adam updates and resumed training with HybridEP."""
    from contextlib import nullcontext

    from megatron.core.dist_checkpointing import load, save
    from megatron.core.distributed.finalize_model_grads import finalize_model_grads
    from megatron.core.distributed.param_and_grad_buffer import _ParamAndGradBucketGroup
    from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
    from megatron.core.transformer.moe.moe_logging import (
        destroy_moe_metrics_tracker,
        get_moe_metrics_tracker,
    )
    from megatron.core.transformer.moe.moe_utils import MoEAuxLossAutoScaler
    from tests.unit_tests.dist_checkpointing import TempNamedDir
    from tests.unit_tests.transformer.moe.test_virtual_expert_hybridep import (
        _assert_numerical_parity,
    )

    if mxfp8 and torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("MXFP8 requires Blackwell")
    Utils.initialize_distributed()
    monkeypatch.setenv("NVTE_CUTEDSL_FUSED_GROUPED_MLP", "1")
    monkeypatch.setenv("NVTE_GROUPED_LINEAR_SINGLE_PARAM", "0")
    monkeypatch.setattr(gtp.GTP_CONFIG, "async_reduction", True)
    monkeypatch.setattr(gtp.GTP_CONFIG, "weight_prefetch", True)
    monkeypatch.setattr(gtp.GTP_CONFIG, "calculate_per_token_loss", True)
    monkeypatch.setattr(gtp.GTP_CONFIG, "reduce_scatter_with_fp32_accumulation", True)
    monkeypatch.setattr(MoEAuxLossAutoScaler, "main_loss_backward_scale", None)
    MoEAuxLossAutoScaler.set_loss_scale(torch.tensor(0.37, device="cuda"))

    ps.destroy_model_parallel()
    ps.initialize_model_parallel(
        expert_model_parallel_size=2,
        expert_gtp_remat_size=expert_gtp,
        gtp_remat_size=2 if mxfp8 else 1,
    )

    errors = []

    def train(virtual, checkpoint, resume=False):
        gtp.reset_gtp_state()
        gtp._GTP_PARAMS.clear()
        initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
        model_parallel_cuda_manual_seed(1234)
        torch.manual_seed(1234)
        config = _expert_config(
            mxfp8,
            moe_virtual_expert_load_balance=virtual,
            expert_gtp_weight_remat_size=expert_gtp,
            # Reusing a layer before backward must accumulate both invocations. TE's
            # first-microbatch overwrite optimization assumes a single use per forward.
            disable_parameter_transpose_cache=True,
            calculate_per_token_loss=True,
            moe_router_score_function="sigmoid",
            moe_router_load_balancing_type="seq_aux_loss",
            moe_aux_loss_coeff=1e-3,
            moe_router_enable_expert_bias=True,
            moe_router_bias_update_rate=1e-3,
            moe_router_topk_scaling_factor=2.5,
            moe_router_fusion=True,
            moe_latent_size=512,
            moe_shared_expert_intermediate_size=512,
        )
        pg = ProcessGroupCollection.use_mpu_process_groups()
        module = model = optimizer = None
        history = []
        try:
            with get_fp8_context(config, is_init=True):
                module = _ExpertStack(config, pg, repeat_last=True).cuda()
            # Give the token prototypes below clear top-k margins. Tiny rounding differences
            # between independent Adam trajectories must not turn a near tie into a new route.
            with torch.no_grad():
                for layer in module.layers:
                    weight = layer.mlp.router.weight
                    weight[:, : config.num_moe_experts].add_(
                        0.5
                        * torch.eye(
                            config.num_moe_experts, device=weight.device, dtype=weight.dtype
                        )
                    )
            parameter_names = tuple(name for name, _ in module.named_parameters())
            model = DDP(
                config=config,
                ddp_config=DistributedDataParallelConfig(
                    use_distributed_optimizer=True,
                    fp8_param_gather=mxfp8,
                    reduce_scatter_with_fp32_accumulation=True,
                    reuse_grad_buf_for_mxfp8_param_ag=mxfp8,
                    overlap_param_gather=mxfp8,
                    grad_reduce_in_fp32=False,
                    overlap_grad_reduce=True,
                    check_for_nan_in_grad=False,
                    average_in_collective=False,
                ),
                module=module,
                pg_collection=pg,
            )
            gtp.tag_gtp_params_with_names(model)
            gtp.classify_gtp_remat_chains(
                model, cuda_graph_modules=config.cuda_graph_modules, cuda_graph_impl="none"
            )
            optimizer = get_megatron_optimizer(
                OptimizerConfig(
                    optimizer="adam",
                    lr=1e-5,
                    bf16=True,
                    use_distributed_optimizer=True,
                    fp8_recipe="mxfp8" if mxfp8 else None,
                    reuse_grad_buf_for_mxfp8_param_ag=mxfp8,
                    overlap_param_gather=mxfp8,
                    clip_grad=0,
                    weight_decay=0,
                ),
                [model],
                use_gloo_process_groups=False,
                pg_collection=pg,
            )
            experts = [p for p in model.parameters() if not getattr(p, "allreduce", True)]
            assert experts and all(getattr(p, "gtp_remat_size", 1) == expert_gtp for p in experts)
            assert pg.expt_dp.size() == 2 // expert_gtp
            main_grad_ptrs = tuple(p.main_grad.data_ptr() for p in experts)
            assert all(is_mxfp8tensor(p) == mxfp8 for p in experts)
            assert all(p.main_grad.dtype == torch.bfloat16 for p in experts)
            assert len(optimizer.chained_optimizers) >= 2
            zero_initialized = {
                i for i, p in enumerate(optimizer.get_parameters()) if not p.detach().any()
            }
            routers = [layer.mlp.router for layer in module.layers]
            semantic_keys = set(module.state_dict())
            metadata = {
                "distrib_optim_sharding_type": "dp_reshardable",
                "dp_cp_group": pg.dp_cp_gtp_remat,
            }
            checkpoint_keys = set(module.sharded_state_dict(metadata=metadata))

            def checkpoint_state(is_loading=False):
                model_state = module.sharded_state_dict(metadata=metadata)
                assert set(model_state) == checkpoint_keys
                return {
                    "model": model_state,
                    "optimizer": optimizer.sharded_state_dict(
                        model_state, is_loading=is_loading, metadata=metadata
                    ),
                }

            if resume:
                state = load(checkpoint_state(is_loading=True), checkpoint)
                with gtp.gtp_native_fp8_load_context(module):
                    module.load_state_dict(state["model"])
                optimizer.load_state_dict(state["optimizer"])
                optimizer.prepare_model_params_for_param_sync()
                model.start_param_sync(force_sync=True)

            routes = []
            active_plans = []
            if virtual:
                for layer in module.layers:
                    manager = layer.mlp.token_dispatcher._comm_manager
                    plan_dispatch = manager.plan_dispatch

                    def record_plan(*args, manager=manager, plan_dispatch=plan_dispatch):
                        plan_dispatch(*args)
                        active_plans.append((manager._plan.experts_to_copy >= 0).any())

                    manager.plan_dispatch = record_plan

            def record_routes(router, inputs, output):
                probs, ids = output
                if ids.dtype == torch.bool:
                    ids = ids.to(torch.int8).topk(router.topk, dim=1).indices
                    probs = probs.gather(1, ids)
                ids, order = ids.sort(dim=-1)
                routes.append((ids.detach().clone(), probs.gather(1, order).detach().clone()))

            for router in routers:
                router.register_forward_hook(record_routes)
            tables = {}
            get_weight_table = _VirtualExperts.get_weight_table

            def check_tables(owner, fc, key, sources):
                table = get_weight_table(owner, fc, key, sources)
                identity = (id(owner), fc, key)
                if step > first_step:
                    assert table is tables[identity], "optimizer step rebuilt a pointer table"
                tables[identity] = table
                return table

            local_grads = {}
            start_grad_sync = _ParamAndGradBucketGroup.start_grad_sync

            def capture_expert_grads(bucket_group, *args, **kwargs):
                if bucket_group in model.expert_parallel_bucket_groups:
                    for bucket in bucket_group.buckets:
                        if bucket not in local_grads:
                            assert bucket.gradient_scaling_factor == 1
                            local_grads[bucket] = bucket.grad_data.clone()
                return start_grad_sync(bucket_group, *args, **kwargs)

            first_step = 2 if resume else 0
            with monkeypatch.context() as patch:
                patch.setattr(_VirtualExperts, "get_weight_table", check_tables)
                if expert_gtp == 1:
                    patch.setattr(_ParamAndGradBucketGroup, "start_grad_sync", capture_expert_grads)
                for step in range(first_step, 3):
                    local_grads.clear()
                    optimizer.zero_grad()
                    model.zero_grad_buffer()
                    if mxfp8:
                        # Match training.py: stage masters after zeroing the reused grad buffer;
                        # DDP's forward hooks gather and quantize each bucket before use.
                        for child in optimizer.chained_optimizers:
                            child._copy_main_params_to_param_buffer()
                    model.set_is_first_microbatch()
                    destroy_moe_metrics_tracker()
                    routes.clear()
                    values = {}
                    before = [p.detach().clone() for p in optimizer.get_parameters()]
                    for microbatch in range(2):
                        rng = torch.Generator(device="cuda").manual_seed(
                            2100 + 100 * step + 10 * microbatch + torch.distributed.get_rank()
                        )
                        x = torch.randn(
                            32,
                            1,
                            config.hidden_size,
                            device="cuda",
                            dtype=torch.bfloat16,
                            generator=rng,
                            requires_grad=True,
                        )
                        rows = torch.arange(x.shape[0], device=x.device)
                        hot = (
                            torch.distributed.get_rank() + step + microbatch
                        ) % config.num_moe_experts
                        preferred = (rows + hot) % config.num_moe_experts
                        preferred[:16] = hot  # Skew half the tokens; the rest visit every expert.
                        with torch.no_grad():
                            x[rows, 0, preferred] += 10
                            x[rows, 0, (preferred + 1) % config.num_moe_experts] += 8
                        dy = torch.randn(x.shape, device="cuda", dtype=x.dtype, generator=rng)
                        with model.no_sync() if microbatch == 0 else nullcontext():
                            with get_fp8_context(config):
                                y = model(x)
                            (0.37 * (y.float() * dy).sum()).backward()
                        values[f"output {microbatch}"] = y.detach().float().clone()
                        values[f"input gradient {microbatch}"] = x.grad.float().clone()
                    finalize_model_grads(
                        [model],
                        num_tokens=torch.tensor(64, dtype=torch.int64, device="cuda"),
                        pg_collection=pg,
                    )
                    if expert_gtp == 1:
                        assert len(local_grads) == sum(
                            len(g.buckets) for g in model.expert_parallel_bucket_groups
                        )
                        replica = torch.distributed.get_rank(pg.expt_dp)
                        for bucket, local in local_grads.items():
                            expected = local.float()
                            torch.distributed.all_reduce(expected, group=pg.expt_dp)
                            local = local.chunk(2)[replica].float()
                            expected = expected.chunk(2)[replica]
                            try:
                                peer = expected - local
                                assert peer.norm() > 0 and not torch.equal(
                                    peer, local
                                ), "replicas must contribute distinct gradients"
                                expected = expected.bfloat16() / (64 * pg.dp_cp_gtp_remat.size())
                                torch.testing.assert_close(
                                    bucket.grad_data.chunk(2)[replica], expected, rtol=0, atol=0
                                )
                            except AssertionError as exc:
                                errors.append(
                                    f"virtual={virtual} resume={resume} step={step} expert DP reduction: {exc}"
                                )
                    values["auxiliary loss"] = (
                        get_moe_metrics_tracker().metrics["seq_load_balancing_loss"].values.cpu()
                    )
                    values["router bias"] = torch.stack([r.expert_bias for r in routers]).cpu()
                    assert all(r.weight.main_grad.float().norm() > 0 for r in routers)
                    for name, parameter in module.named_parameters():
                        values[f"model weight {name}"] = parameter.detach().float().cpu()
                    assert optimizer.step()[0], "Adam skipped an update"
                    for index, (parameter, initial) in enumerate(
                        zip(optimizer.get_parameters(), before)
                    ):
                        values[f"gradient {index}"] = parameter.grad.detach().cpu()
                        values[f"update {index}"] = (parameter.detach() - initial).cpu()
                        # A zero-initialized master is the sum of Adam updates: use the same
                        # peak/sign-flip allowance and tight aggregate bound as individual updates.
                        kind = "update total" if index in zero_initialized else "master weight"
                        values[f"{kind} {index}"] = parameter.detach().cpu()
                    for index, child in enumerate(optimizer.chained_optimizers):
                        for local, parameter in enumerate(child.get_parameters()):
                            for name in ("exp_avg", "exp_avg_sq"):
                                values[f"Adam {index}/{local} {name}"] = (
                                    child.optimizer.state[parameter][name].detach().cpu()
                                )
                    assert tuple(name for name, _ in module.named_parameters()) == parameter_names
                    assert set(module.state_dict()) == semantic_keys
                    assert tuple(p.main_grad.data_ptr() for p in experts) == main_grad_ptrs
                    if virtual:
                        assert tables
                        if expert_gtp == 2:
                            assert all(p._gtp_wgrad_ring_slot is not None for p in experts)
                        else:
                            for layer in module.layers:
                                owner = layer.mlp.token_dispatcher._comm_manager.virtual_experts
                                assert owner.gtp_leaders == (None, None)
                                for fc, grads in enumerate(owner.native_grads):
                                    owner.get_weight_table(fc, "grad", grads)
                                    assert all(
                                        p.main_grad.data_ptr() == g.data_ptr()
                                        for p, g in zip(owner.runtime_weights[fc], grads)
                                    )
                    history.append(
                        (
                            {name: value.cpu() for name, value in values.items()},
                            [(ids.cpu(), probs.cpu()) for ids, probs in routes],
                        )
                    )
                    if virtual and not resume and step == 1:
                        optimizer.prepare_model_params_for_param_sync()
                        model.start_param_sync(force_sync=True)
                        save(checkpoint_state(), checkpoint)
            if virtual:
                active = torch.stack(active_plans).any().to(torch.int32)
                torch.distributed.all_reduce(active)
                assert active.item(), "training must materialize a virtual expert"
            return history
        finally:
            torch.cuda.synchronize()
            del optimizer, model, module
            FP8GlobalStateManager.reset()
            destroy_moe_metrics_tracker()
            VirtualExpertLoadBalancer.finalize()
            gtp.reset_gtp_state()
            gtp._GTP_PARAMS.clear()
            gc.collect()

    try:
        with TempNamedDir(
            tmp_path_dist_ckpt / f"virtual_expert_training_{mxfp8}_{expert_gtp}"
        ) as checkpoint:
            reference = train(False, checkpoint)
            candidate = train(True, checkpoint)
            resumed = train(True, checkpoint, resume=True)
    finally:
        ps.destroy_model_parallel()
        fused_a2a.reset_hybrid_ep_buffer()
    try:
        assert len(reference) == len(candidate) == 3 and len(resumed) == 1
        # Compare after model collectives, then report failures on every rank before the next test.
        for label, actual_steps, expected_steps in (
            ("HybridEP", candidate, reference),
            ("resume", resumed, candidate[2:]),
        ):
            for step, ((actual, routes), (expected, expected_routes)) in enumerate(
                zip(actual_steps, expected_steps)
            ):
                assert actual.keys() == expected.keys()
                for name in actual:
                    # LayerNorm biases start at zero; gradients and updates must remain nonzero.
                    if name.startswith("model weight ") and not expected[name].any():
                        torch.testing.assert_close(actual[name], expected[name], atol=0, rtol=0)
                        continue
                    update = name.startswith("update ")
                    tolerance = (0.1 if mxfp8 else 0.03) if label == "HybridEP" else 1e-5
                    if label == "HybridEP":
                        if update:
                            # Near-zero gradient sign flips can reverse isolated Adam updates.
                            # Allow those peaks, but bound aggregate update error tightly below.
                            tolerance = 2.1
                        elif name.startswith(("output ", "input gradient ")):
                            tolerance = 0.02
                        elif name == "auxiliary loss":
                            tolerance = 1e-5
                    if name == "router bias":
                        tolerance = 0
                    _assert_numerical_parity(
                        actual[name], expected[name], tolerance, f"{label} step {step} {name}"
                    )
                    if label == "HybridEP" and (mxfp8 or update):
                        error = actual[name].float() - expected[name].float()
                        assert error.norm() <= 0.06 * expected[name].float().norm(), name
                assert (
                    len(routes) == len(expected_routes) == 2 * 4
                )  # 3 layers + repeated last, twice.
                assert any(not torch.equal(a[0], b[0]) for a, b in zip(routes[:4], routes[4:]))
                for (ids, probs), (expected_ids, expected_probs) in zip(routes, expected_routes):
                    torch.testing.assert_close(ids, expected_ids, atol=0, rtol=0)
                    _assert_numerical_parity(
                        probs,
                        expected_probs,
                        1e-3 if label == "HybridEP" else 1e-5,
                        f"{label} router probabilities",
                    )
    except AssertionError as exc:
        errors.append(str(exc))
    gathered_errors = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(gathered_errors, errors)
    assert not any(gathered_errors), gathered_errors
