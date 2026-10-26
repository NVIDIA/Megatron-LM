# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Paired production-path diagnostic for the scheduler-disabled loss mismatch.

The user-selected reference is scheduler-disabled MoELayer at 231f7f367d22,
not a mathematical oracle. Both legs use real HybridEP, TE fused SwiGLU and
MCore DDP/distributed Adam. Shapes match the failing storage-training proxy.
No FP8, dropout, home exchange, or generated-data differences are involved.
"""

import gc
import json
import os
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_submodules,
)
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.spec_utils import get_submodules
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.moe.test_replica_nccl_transport import distributed_world

pytestmark = [pytest.mark.launch_on_gb200, pytest.mark.experimental]


def _config(backend, capacity=None, **overrides):
    return TransformerConfig(
        num_layers=1,
        hidden_size=256,
        num_attention_heads=4,
        num_moe_experts=8,
        moe_ffn_hidden_size=512,
        expert_model_parallel_size=4,
        moe_router_topk=2,
        moe_router_dtype="fp32",
        moe_router_mock_maxvio=2,
        moe_router_load_balancing_type="none",
        moe_aux_loss_coeff=0,
        moe_router_force_load_balancing=False,
        moe_token_dispatcher_type="flex",
        moe_flex_dispatcher_backend="hybridep",
        moe_hybridep_num_sms=4,
        moe_grouped_gemm=True,
        moe_permute_fusion=True,
        moe_router_padding_for_quantization=False,
        use_transformer_engine_op_fuser=True,
        gradient_accumulation_fusion=True,
        moe_enable_scheduler=backend is not None,
        moe_scheduler_planner_type="eplb",
        moe_scheduler_expert_dispatcher_type=backend or "replica_nccl",
        moe_scheduler_num_idle_experts=8,
        moe_scheduler_home_update_interval=0,
        moe_expert_rank_capacity_factor=capacity,
        gated_linear_unit=True,
        activation_func=F.silu,
        add_bias_linear=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        hidden_dropout=0,
        attention_dropout=0,
        **overrides,
    )


def _run(backend, pg, initial, inputs, upstream, capacity=None):
    config = _config(backend, capacity)
    model_parallel_cuda_manual_seed(1234)
    spec = get_gpt_layer_with_transformer_engine_submodules(
        num_experts=8, moe_grouped_gemm=True
    ).mlp
    layer = MoELayer(config, get_submodules(spec), layer_number=1, pg_collection=pg).cuda()
    # Capture optimizer-owned parameters before lazy runtime/TE wrappers appear.
    parameters = dict(layer.named_parameters())
    if initial is None:
        initial = {n: p.detach().clone() for n, p in parameters.items()}
    else:
        assert initial.keys() == parameters.keys()
        with torch.no_grad():
            for n, p in parameters.items():
                p.copy_(initial[n])
                assert torch.equal(p, initial[n])
    ddp = DistributedDataParallel(
        config=config,
        ddp_config=DistributedDataParallelConfig(
            use_distributed_optimizer=True,
            overlap_grad_reduce=False,
            overlap_param_gather=False,
            grad_reduce_in_fp32=True,
        ),
        module=layer,
        pg_collection=pg,
    )
    optimizer = get_megatron_optimizer(
        OptimizerConfig(
            optimizer="adam",
            lr=1e-4,
            weight_decay=0,
            bf16=True,
            use_distributed_optimizer=True,
            overlap_param_gather=False,
            clip_grad=0,
        ),
        [ddp],
        pg_collection=pg,
        use_gloo_process_groups=False,
    )
    masters = [p for g in optimizer.param_groups for p in g["params"]]
    assert all(p.dtype == torch.float32 for p in masters)
    initial_masters = [p.detach().clone() for p in masters]
    observed = {}
    dispatch = []
    try:
        ddp.zero_grad_buffer()
        optimizer.zero_grad()
        for i, (x, dy) in enumerate(zip(inputs, upstream)):
            x = x.detach().clone().requires_grad_(True)
            y, bias = ddp(x)
            assert bias is None
            manager = layer.token_dispatcher._comm_manager
            counts = manager.routing_map.sum(dim=0).to(torch.int64)
            dist.all_reduce(counts, group=pg.ep)
            # BF16 has no quantization alignment requirement (None).
            pad = manager.pad_multiple or 1
            padded = ((counts + pad - 1) // pad) * pad
            dispatch.append(
                {
                    "capacity_factor": config.moe_expert_rank_capacity_factor,
                    "logical_assignments_per_rank": counts.reshape(4, -1).sum(1).tolist(),
                    "padded_assignments_per_rank": padded.reshape(4, -1).sum(1).tolist(),
                    "padding_multiple": pad,
                    "overflow": bool(layer.token_dispatcher.check_over_budget().item()),
                }
            )
            y.backward(dy / len(inputs))
            observed[f"forward/{i}"] = y.detach().cpu()
            observed[f"dgrad/{i}"] = x.grad.detach().cpu()
            for n, p in parameters.items():
                observed[f"wgrad_accum_{i}/{n}"] = p.main_grad.detach().cpu().clone()
        ddp.finish_grad_sync()
        for n, p in parameters.items():
            observed[f"wgrad/{n}"] = p.main_grad.detach().cpu().clone()
        success, _, _ = optimizer.step()
        assert success
        ddp.start_param_sync(force_sync=True)
        for i, (p, before) in enumerate(zip(masters, initial_masters)):
            observed[f"master_delta/{i}"] = (p.detach() - before).cpu()
        for n, p in parameters.items():
            observed[f"updated_weight/{n}"] = p.detach().cpu().clone()
            observed[f"weight_delta/{n}"] = (p.detach().float() - initial[n].float()).cpu()
        torch.cuda.synchronize()
        return initial, observed, dispatch
    finally:
        if layer.moe_scheduler is not None:
            layer.moe_scheduler.expert_dispatch.runtime.destroy()
        del optimizer, ddp, layer
        gc.collect()


def _metrics(reference, actual):
    assert reference.shape == actual.shape
    a, b = reference.double().flatten(), actual.double().flatten()
    na, nb = a.norm().item(), b.norm().item()
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    denom = (a.square().sum() + b.square().sum()).item()
    dot = (a * b).sum().item()
    return {
        "max_abs": (a - b).abs().max().item() if a.numel() else 0,
        "relative_l2": (a - b).norm().item() / max(na, 1e-30),
        "reference_norm": na,
        "actual_norm": nb,
        "cosine": dot / (na * nb) if na * nb else float(na == nb),
        "tensor_similarity": 2 * dot / denom if denom else 1,
        "finite": finite,
    }


@pytest.mark.parametrize("backend", ["replica_nccl", "replica_peer_tma"])
@pytest.mark.parametrize("microbatches", [1, 4])
def test_bf16_scheduler_layer_forward_backward_adam(
    distributed_world, monkeypatch, backend, microbatches
):
    _compare(distributed_world, monkeypatch, backend, microbatches)


def test_explicit_static_budget_reproduces_dispatch_overflow(distributed_world, monkeypatch):
    # An explicit insufficient static budget requires the training overflow
    # retry runner. This diagnostic deliberately calls the bare layer and
    # checks the overflow flag; it does not declare numerical parity passing.
    _compare(distributed_world, monkeypatch, "replica_nccl", 1, capacity=1.0)


def test_scheduler_capacity_default_and_capture_validation():
    assert _config("replica_peer_tma").moe_expert_rank_capacity_factor is None
    assert _config("replica_peer_tma", 1.25).moe_expert_rank_capacity_factor == 1.25
    with pytest.raises(ValueError, match="requires an explicit"):
        _config(
            "replica_peer_tma", cuda_graph_impl="transformer_engine", cuda_graph_modules=["moe"]
        )


def _compare(distributed_world, monkeypatch, backend, microbatches, capacity=None):
    assert dist.get_world_size() == 4, "This parity recipe requires four GPUs."
    monkeypatch.setenv("NVTE_CUTEDSL_FUSED_GROUPED_MLP", "1")
    # Match scheduler's wgrad setting in the baseline as well.
    monkeypatch.setenv("NVTE_DISABLE_CUTEDSL_WGRAD_FUSED_GROUPED_MLP", "1")
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, expert_model_parallel_size=4)
    pg = ProcessGroupCollection.use_mpu_process_groups()
    generator = torch.Generator(device="cuda").manual_seed(7000 + dist.get_rank())
    inputs = [
        torch.randn(256, 1, 256, device="cuda", dtype=torch.bfloat16, generator=generator)
        for _ in range(microbatches)
    ]
    upstream = [
        torch.randn(x.shape, device="cuda", dtype=torch.bfloat16, generator=generator) * 0.01
        for x in inputs
    ]
    try:
        initial, reference, reference_dispatch = _run(None, pg, None, inputs, upstream)
        _, actual, actual_dispatch = _run(backend, pg, initial, inputs, upstream, capacity)
        assert reference.keys() == actual.keys()
        metrics = {name: _metrics(reference[name], actual[name]) for name in reference}
        reports = [None] * dist.get_world_size()
        dist.all_gather_object(reports, metrics)
        dispatch_reports = [None] * dist.get_world_size()
        dist.all_gather_object(
            dispatch_reports, {"baseline": reference_dispatch, "scheduler": actual_dispatch}
        )
        # Record every stage before asserting, so one failure cannot hide later
        # gradient or optimizer evidence. Every rank reaches the same assertion.
        failures = []
        for rank, report in enumerate(reports):
            for name, values in report.items():
                if (
                    not values["finite"]
                    or values["cosine"] < 0.999
                    or values["tensor_similarity"] < 0.999
                ):
                    failures.append(f"rank={rank} {name}: {values}")
        if dist.get_rank() == 0:
            path = Path(os.getenv("MOE_PARITY_REPORT_DIR", "/results/bf16-parity"))
            path.mkdir(parents=True, exist_ok=True)
            suffix = "dynamic" if capacity is None else "static-overflow-control"
            (path / f"{backend}-mb{microbatches}-{suffix}.json").write_text(
                json.dumps(
                    {
                        "backend": backend,
                        "microbatches": microbatches,
                        "ranks": reports,
                        "failures": failures,
                        "dispatch": dispatch_reports,
                        "capacity": capacity,
                    },
                    indent=2,
                )
                + "\n"
            )
            print(
                f"BF16_PARITY {backend} mb={microbatches}: {len(failures)} discrepancies",
                flush=True,
            )
        overflow = any(d["overflow"] for r in dispatch_reports for d in r["scheduler"])
        if capacity is None:
            assert not overflow, dispatch_reports
            assert not failures, "\n".join(failures[:16])
        else:
            assert overflow, "Static-budget reproducer unexpectedly did not overflow."
            assert failures, "Overflow should be visible in the paired numerical comparison."
    finally:
        Utils.destroy_model_parallel()
