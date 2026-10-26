# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Shared production-path BF16/MXFP8 parity diagnostic (no GTP)."""

import gc

import torch
import torch.distributed as dist
import torch.nn.functional as F

from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_submodules,
)
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.spec_utils import get_submodules
from megatron.core.transformer.transformer_config import TransformerConfig


def config_for(
    precision, planner=None, backend="nccl", storage="shared", layout="reuse", routing="mock"
):
    """Keep every non-scheduler field equal between the two comparison legs."""
    lowp = precision == "mxfp8"
    return TransformerConfig(
        num_layers=1,
        hidden_size=256,
        num_attention_heads=4,
        num_moe_experts=8,
        moe_ffn_hidden_size=512,
        expert_model_parallel_size=4,
        expert_tensor_parallel_size=1,
        moe_router_topk=2,
        moe_router_dtype="fp32",
        moe_router_mock_maxvio=2 if routing == "mock" else None,
        moe_router_fusion=routing == "learned",
        moe_router_load_balancing_type="none",
        moe_aux_loss_coeff=0,
        moe_router_force_load_balancing=False,
        moe_token_dispatcher_type="flex",
        moe_flex_dispatcher_backend="hybridep",
        moe_hybridep_num_sms=4,
        moe_grouped_gemm=True,
        moe_permute_fusion=True,
        moe_router_padding_for_quantization=lowp,
        use_transformer_engine_op_fuser=True,
        gradient_accumulation_fusion=True,
        moe_enable_scheduler=planner is not None,
        moe_scheduler_planner_type=planner or "eplb",
        moe_scheduler_expert_dispatcher_type="replica_" + backend,
        moe_scheduler_num_idle_experts=4 if planner == "echo" else 8,
        moe_scheduler_home_update_interval=0,
        moe_expert_rank_capacity_factor=None,
        moe_scheduler_replica_storage=storage,
        moe_scheduler_replica_mxfp8_storage=layout,
        gated_linear_unit=True,
        activation_func=F.silu,
        add_bias_linear=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        hidden_dropout=0,
        attention_dropout=0,
        fp8="e4m3" if lowp else None,
        fp8_recipe="mxfp8",
        fp8_param=lowp,
    )


def snapshot(tensor):
    """Materialize a detached plain CPU tensor, including native FP8 parameters."""
    tensor = tensor.detach()
    if callable(getattr(tensor, "to_local", None)):
        tensor = tensor.to_local()
    return tensor.float().cpu().clone()


def run_leg(
    precision, pg, inputs, upstream, case=None, routing="mock", trace_routing=False, fsdp=False
):
    """Exercise real TE kernels, distributed gradients, and distributed Adam."""
    config = config_for(precision, *(case or ()), routing=routing)
    lowp = precision == "mxfp8"
    torch.manual_seed(1234)
    model_parallel_cuda_manual_seed(1234)
    submodules = get_submodules(
        get_gpt_layer_with_transformer_engine_submodules(
            num_experts=8, moe_grouped_gemm=True, use_te_op_fuser=True
        ).mlp
    )
    with get_fp8_context(config, is_init=True):
        layer = MoELayer(config, submodules, layer_number=1, pg_collection=pg).cuda()
    parameters = dict(layer.named_parameters())
    observed = {"initial/" + name: snapshot(p) for name, p in parameters.items()}
    from megatron.core.distributed.fsdp.mcore_fsdp_adapter import FullyShardedDataParallel

    wrapper = FullyShardedDataParallel if fsdp else DistributedDataParallel
    ddp = wrapper(
        config=config,
        ddp_config=DistributedDataParallelConfig(
            use_distributed_optimizer=True,
            overlap_grad_reduce=False,
            overlap_param_gather=False,
            grad_reduce_in_fp32=True,
            megatron_fsdp_main_grads_dtype=torch.float32,
            megatron_fsdp_grad_comm_dtype=torch.float32,
            use_megatron_fsdp=fsdp,
            data_parallel_sharding_strategy="optim_grads_params" if fsdp else "no_shard",
            fp8_param_gather=lowp,
            reuse_grad_buf_for_mxfp8_param_ag=lowp and not fsdp,
        ),
        module=layer,
        pg_collection=pg,
        **({"fsdp_unit_modules": [MoELayer]} if fsdp else {}),
    )
    optimizer = get_megatron_optimizer(
        OptimizerConfig(
            optimizer="adam",
            lr=1e-4,
            weight_decay=0,
            bf16=True,
            params_dtype=torch.bfloat16,
            use_distributed_optimizer=True,
            overlap_param_gather=False,
            # A single FSDP optimizer reports 0 without computing the norm
            # when clipping is disabled. Exercise real norm computation and
            # clipping on both FSDP comparison legs before checking gradients.
            clip_grad=1.0 if fsdp else 0,
            fp8_recipe="mxfp8" if lowp else None,
            reuse_grad_buf_for_mxfp8_param_ag=lowp and not fsdp,
        ),
        [ddp],
        pg_collection=pg,
        use_gloo_process_groups=False,
    )
    masters = [p for group in optimizer.param_groups for p in group["params"]]
    assert all(p.dtype == torch.float32 for p in masters)
    for i, p in enumerate(masters):
        observed[f"initial_master/{i}"] = snapshot(p)
    diagnostics = []
    route_capture = {}
    if trace_routing:

        def capture_routes(module, args, output):
            probabilities, routes = output
            # Router probabilities retain the logical [tokens, experts] shape
            # even when the fused routing map is emitted as dense top-k IDs.
            logical = probabilities.detach()
            assert logical.shape == (routes.shape[0], config.num_moe_experts)
            route_capture["routing_probs"] = snapshot(logical)
            route_capture["routing_map"] = snapshot(logical != 0)

        layer.router.register_forward_hook(capture_routes)
    try:
        for step, (xs, dys) in enumerate(zip(inputs, upstream)):
            ddp.zero_grad_buffer()
            optimizer.zero_grad()
            before = [p.detach().clone() for p in masters]
            for micro, (x, dy) in enumerate(zip(xs, dys)):
                x = x.detach().clone().requires_grad_(True)
                with get_fp8_context(config):
                    y, bias = ddp(x)
                assert bias is None
                assert y.detach().float().norm().item() > 0, "Degenerate zero forward output"
                y.backward(dy / len(xs))
                assert x.grad.float().norm().item() > 0, "Degenerate zero input gradient"
                prefix = f"step{step}/mb{micro}"
                observed[f"{prefix}/forward"] = snapshot(y)
                observed[f"{prefix}/linear_objective"] = snapshot((y.float() * dy.float()).sum())
                observed[f"{prefix}/dgrad"] = snapshot(x.grad)
                for name, tensor in route_capture.items():
                    observed[f"{prefix}/{name}"] = tensor
                if not fsdp:
                    for name, p in parameters.items():
                        observed[f"{prefix}/wgrad_accum/{name}"] = snapshot(p.main_grad)
                overflow = bool(layer.token_dispatcher.check_over_budget().item())
                assert not overflow, "Dispatch overflow invalidates the numerical comparison"
            ddp.finish_grad_sync()
            if not fsdp:
                for name, p in parameters.items():
                    observed[f"step{step}/wgrad/{name}"] = snapshot(p.main_grad)
            success, grad_norm, _ = optimizer.step()
            assert success, "Optimizer skipped an update"
            assert grad_norm is not None and float(grad_norm) > 0, "Missing training gradients"
            # Non-overlapped DistributedOptimizer.step() already gathers and
            # post-processes parameters. A second gather would read the MXFP8
            # staging buffer after it has been zeroed for gradient reuse.
            for i, (p, old) in enumerate(zip(masters, before)):
                observed[f"step{step}/master_delta/{i}"] = snapshot(p.detach() - old)
            if not fsdp:
                for name, p in parameters.items():
                    observed[f"step{step}/updated_weight/{name}"] = snapshot(p)
            runtime = layer.moe_scheduler.expert_dispatch.runtime if layer.moe_scheduler else None
            cache = runtime.weight_cache if runtime else None
            diagnostics.append(
                {
                    "step": step,
                    "grad_norm": float(grad_norm) if grad_norm is not None else None,
                    "cache_hits": cache.hits if cache else 0,
                    "cache_misses": cache.misses if cache else 0,
                    "planner_bound": (
                        layer.moe_scheduler.max_dispatch_routes_per_rank
                        if layer.moe_scheduler
                        else None
                    ),
                }
            )
        torch.cuda.synchronize()
        return observed, diagnostics
    finally:
        if layer.moe_scheduler:
            layer.moe_scheduler.expert_dispatch.runtime.destroy()
        del optimizer, ddp, layer
        gc.collect()
        torch.cuda.empty_cache()
        dist.barrier()
