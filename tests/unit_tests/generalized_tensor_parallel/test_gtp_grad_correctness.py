# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Numeric repro: GTP_remat gradient correctness through the REAL
DDP + distributed-optimizer + finalize path, with replicate (DP) > 1.

The validated loss-trajectory test uses DP=1 (replicate=1) and manual
SGD on main_grad, so it cannot catch a gradient-reduction error that only shows
up when the dist-opt shards over a replicate group of size > 1 (the new-at-64-GPU
condition: DP2 x GTP16). This test reproduces that condition at small scale
(world=4 = GTP2 x DP2) and checks the gradient end-to-end against a trusted
no-GTP_remat DP=4 baseline.

Decisive choices:
  * SGD lr=1.0 (NOT Adam): the step is scale-SENSITIVE, so a gtp_remat x gradient
    under-scale shows up directly as a gtp_remat x smaller weight delta. Adam would
    normalize a uniform scale error away and mask the bug.
  * Distinct input per rank (seed=rank): each data-parallel position sees a
    different batch (the HSDP guarantee), so the correct reduced grad is the
    MEAN over all 4 positions. Baseline (DP4) and GTP_remat (GTP2xDP2) both
    span the same 4 positions, so their reduced grads -- and thus post-step
    weights and grad-norm -- must match.
"""

import contextlib

import pytest
import torch
import torch.distributed as dist

from megatron.core.tensor_parallel.gtp_api import HAVE_GTP

if not HAVE_GTP:
    pytest.skip("GTP requires TransformerEngine >= 2.19", allow_module_level=True)

from megatron.core.tensor_parallel.generalized_tensor_parallelism import GTPShardedParam
from tests.unit_tests.generalized_tensor_parallel.gtp_test_utils import (  # noqa: F401
    _run_distributed,
    _torchrun_dist_init,
    reset_fp8_state,
    reset_gtp_globals,
)

HIDDEN = 256
NUM_HEADS = 8
FFN_HIDDEN = 512
NUM_LAYERS = 1
SEQ = 16
BATCH = 1
LR = 1.0  # scale-sensitive SGD step
dtype = torch.bfloat16


def _make_config(calculate_per_token_loss=False):
    from megatron.core.transformer.transformer_config import TransformerConfig

    return TransformerConfig(
        num_attention_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        hidden_size=HIDDEN,
        ffn_hidden_size=FFN_HIDDEN,
        add_bias_linear=False,
        params_dtype=dtype,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        bias_dropout_fusion=False,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        calculate_per_token_loss=calculate_per_token_loss,
    )


def _make_stack(config, pg_collection):
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec

    spec = get_gpt_layer_with_transformer_engine_spec()
    return torch.nn.ModuleList(
        [
            spec.module(config, spec.submodules, layer_number=i + 1, pg_collection=pg_collection)
            for i in range(NUM_LAYERS)
        ]
    )


def _build_ddp(stack, calculate_per_token_loss=False):
    """Wrap the stack in a NON-distributed-optimizer DDP so main_grad holds the
    full all-reduced gradient (no optimizer needed; no Adam scale-invariance to
    mask a scaling error)."""
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig

    config = _make_config(calculate_per_token_loss=calculate_per_token_loss)
    ddp_config = DistributedDataParallelConfig(
        use_distributed_optimizer=False, overlap_grad_reduce=False
    )
    module = torch.nn.Sequential()
    for i, layer in enumerate(stack):
        module.add_module(str(i), layer)
    return DistributedDataParallel(config, ddp_config, module)


def _run_one_backward(ddp_model, rank, calculate_per_token_loss=False):
    ddp_model.zero_grad_buffer()
    # Distinct input per rank => the correct reduced grad is the MEAN over ranks.
    torch.manual_seed(1000 + rank)
    x = torch.randn(SEQ, BATCH, HIDDEN, dtype=dtype, device='cuda')
    out = x
    for layer in ddp_model.module.children():
        out, _ = layer(out, attention_mask=None)
    loss = out.float().mean()
    loss.backward()
    # Sync ONCE: finish_grad_sync() triggers the (single) grad reduction for
    # overlap_grad_reduce=False. Do NOT also call start_grad_sync() — that double-
    # reduces, which is idempotent at full-DP size but halves at replicate size.
    ddp_model.finish_grad_sync()
    from megatron.core import parallel_state as ps
    from megatron.core.distributed.finalize_model_grads import (
        _allreduce_replicated_grads_over_gtp_remat_group,
    )

    _allreduce_replicated_grads_over_gtp_remat_group(
        [ddp_model],
        ps.get_gtp_weight_remat_group(check_initialized=False),
        ps.get_expert_gtp_weight_remat_group(check_initialized=False),
        calculate_per_token_loss=calculate_per_token_loss,
    )
    return float(loss.item())


def _full_main_grads(stack):
    """Reconstruct full (unsharded) reduced gradients keyed by param name.

    GTPShardedParam.main_grad is the local shard -> all-gather over its own axis (expert params
    shard over egtp_remat, dense over gtp_remat). Non-GTP_remat params are replicated -> take
    the local (already gtp_remat-summed) copy.
    """
    from megatron.core import parallel_state as ps

    out = {}
    for layer in stack:
        for name, p in layer.named_parameters():
            g_attr = 'main_grad' if hasattr(p, 'main_grad') else 'grad'
            mg = getattr(p, g_attr)
            if isinstance(p, GTPShardedParam):
                g = (
                    ps.get_expert_gtp_weight_remat_group()
                    if _is_expert_param(name, p)
                    else ps.get_gtp_weight_remat_group()
                )
                shards = [torch.empty_like(mg) for _ in range(g.size())]
                dist.all_gather(shards, mg.contiguous(), group=g)
                out[name] = torch.cat(shards, dim=0).float().cpu()
            else:
                out[name] = mg.detach().float().cpu()
    return out


def _load_gtp_shards(stack, saved, moe=False):
    """Copy each param's own shard of the unsharded `saved` weights into `stack`."""
    from megatron.core import parallel_state as ps

    gtp_rank = ps.get_gtp_weight_remat_group().rank()
    egtp_rank = ps.get_expert_gtp_weight_remat_group().rank() if moe else 0
    for name, p in stack.named_parameters():
        full = saved[name]
        if isinstance(p, GTPShardedParam):
            r = egtp_rank if _is_expert_param(name, p) else gtp_rank
            ss = p.shape[0]
            p.data.copy_(full[r * ss : (r + 1) * ss])
        else:
            p.data.copy_(full)


def _worker(rank, world_size, port, calculate_per_token_loss=False):
    from megatron.core import parallel_state as ps
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    # ---------- Phase A: baseline, GTP_remat=1 DP=4 (trusted standard path) ----------
    ps.destroy_model_parallel()
    ps.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, gtp_remat_size=1
    )
    model_parallel_cuda_manual_seed(42)
    pgc = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp', 'gtp_remat'])
    base_stack = _make_stack(_make_config(calculate_per_token_loss=calculate_per_token_loss), pgc)
    for layer in base_stack:
        layer.cuda()
    for p in base_stack.parameters():
        dist.broadcast(p.data, src=0)
    saved = {n: p.data.clone() for n, p in base_stack.named_parameters()}

    base_ddp = _build_ddp(base_stack, calculate_per_token_loss=calculate_per_token_loss)
    _run_one_backward(base_ddp, rank, calculate_per_token_loss=calculate_per_token_loss)
    base_grads = _full_main_grads(base_stack)

    ps.destroy_model_parallel()
    GTPShardedParam._chain_state = {}

    # ---------- Phase B: GTP_remat=2 DP=2 (replicate>1!) ----------
    ps.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, gtp_remat_size=2
    )
    model_parallel_cuda_manual_seed(42)
    pgc = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp', 'gtp_remat'])
    gtp_stack = _make_stack(_make_config(calculate_per_token_loss=calculate_per_token_loss), pgc)
    for layer in gtp_stack:
        layer.cuda()

    g = ps.get_gtp_weight_remat_group()
    gtp_rank = g.rank()
    assert g.size() == 2, f"expected gtp_remat shard group size 2, got {g.size()}"

    # Load the SAME init weights as baseline: GTP_remat params get their gtp_remat shard.
    for name, p in gtp_stack.named_parameters():
        full = saved[name]
        if isinstance(p, GTPShardedParam):
            ss = p.shape[0]
            p.data.copy_(full[gtp_rank * ss : (gtp_rank + 1) * ss])
        else:
            p.data.copy_(full)

    gtp_ddp = _build_ddp(gtp_stack, calculate_per_token_loss=calculate_per_token_loss)
    _run_one_backward(gtp_ddp, rank, calculate_per_token_loss=calculate_per_token_loss)
    gtp_grads = _full_main_grads(gtp_stack)

    ps.destroy_model_parallel()
    GTPShardedParam._chain_state = {}

    # ---------- Compare reduced gradients on rank 0 ----------
    if rank == 0:
        max_err = 0.0
        worst = None
        for name in base_grads:
            bg, gg = base_grads[name], gtp_grads[name]
            assert bg.shape == gg.shape, f"{name}: {bg.shape} vs {gg.shape}"
            err = (bg - gg).abs().max().item()
            denom = bg.abs().max().item() + 1e-8
            rel = err / denom
            ratio = (gg.norm() / (bg.norm() + 1e-12)).item()
            print(
                f"[grad] {name:55s} rel_max_err={rel:.3e}  norm_ratio(orth/base)={ratio:.4f}",
                flush=True,
            )
            if rel > max_err:
                max_err, worst = rel, name
        print(
            f"[summary] max relative grad error GTP_remat-vs-DP4-baseline = {max_err:.3e} "
            f"(worst: {worst})",
            flush=True,
        )
        assert max_err < 2e-2, (
            f"GTP_remat2xDP2 reduced gradient does not match the no-GTP_remat DP4 baseline "
            f"(max rel err {max_err:.3e} on {worst}) -> gtp_remat-axis grad reduce/scaling error."
        )


# ---------------------------------------------------------------------------
# Distributed-optimizer + grad-norm path (the production 64-GPU path)
# ---------------------------------------------------------------------------


def _build_ddp_distopt_and_optim(
    stack,
    overlap_grad_reduce=False,
    bucket_size=None,
    reduce_scatter_with_fp32_accumulation=False,
    grad_reduce_in_fp32=True,
):
    """Real distributed-optimizer setup (Adam), matching the 64-GPU production path."""
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
    from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer

    config = _make_config()
    ddp_config = DistributedDataParallelConfig(
        use_distributed_optimizer=True,
        overlap_grad_reduce=overlap_grad_reduce,
        bucket_size=bucket_size,
        grad_reduce_in_fp32=grad_reduce_in_fp32,
        reduce_scatter_with_fp32_accumulation=reduce_scatter_with_fp32_accumulation,
    )
    module = torch.nn.Sequential()
    for i, layer in enumerate(stack):
        module.add_module(str(i), layer)
    ddp_model = DistributedDataParallel(config, ddp_config, module)
    opt_config = OptimizerConfig(
        optimizer='adam',
        lr=0.01,
        bf16=True,
        use_distributed_optimizer=True,
        use_precision_aware_optimizer=False,
        main_params_dtype=torch.float32,
        main_grads_dtype=torch.float32,
        exp_avg_dtype=torch.float32,
        exp_avg_sq_dtype=torch.float32,
        clip_grad=1.0,  # reported grad-norm is computed pre-clip, so this is just for the step
    )
    optim = get_megatron_optimizer(opt_config, [ddp_model])
    return ddp_model, optim


def _run_step_distopt(ddp_model, optim, rank):
    """Mirror production finalize order: finish_grad_sync -> gtp_remat-finalize -> optim.step().
    Returns the optimizer-reported grad-norm (computed pre-clip from the reduced grads)."""
    optim.zero_grad()
    ddp_model.zero_grad_buffer()
    torch.manual_seed(1000 + rank)
    x = torch.randn(SEQ, BATCH, HIDDEN, dtype=dtype, device='cuda')
    out = x
    for layer in ddp_model.module.children():
        out, _ = layer(out, attention_mask=None)
    loss = out.float().mean()
    loss.backward()
    # Production order (finalize_model_grads): fence GTP's wgrad reduce-scatters (line 634),
    # reduce across DP (line 640), THEN the gtp_remat all-reduce of replicated grads. The fence
    # matters once wgrad_finalize_depth > 1: collectives still queued behind the depth have not
    # been accumulated into main_grad yet, and finish_grad_sync would reduce without them.
    from megatron.core.tensor_parallel.generalized_tensor_parallelism import (
        wait_for_gtp_grad_reduction_on_current_stream,
    )

    wait_for_gtp_grad_reduction_on_current_stream()
    ddp_model.finish_grad_sync()
    from megatron.core import parallel_state as ps
    from megatron.core.distributed.finalize_model_grads import (
        _allreduce_replicated_grads_over_gtp_remat_group,
    )

    _allreduce_replicated_grads_over_gtp_remat_group(
        [ddp_model],
        ps.get_gtp_weight_remat_group(check_initialized=False),
        ps.get_expert_gtp_weight_remat_group(check_initialized=False),
    )
    _, grad_norm, _ = optim.step()
    return float(grad_norm)


def _worker_distopt(rank, world_size, port):
    from megatron.core import parallel_state as ps
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    # ---------- Phase A: baseline, GTP_remat=1 DP=4, dist-opt + Adam ----------
    ps.destroy_model_parallel()
    ps.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, gtp_remat_size=1
    )
    model_parallel_cuda_manual_seed(42)
    pgc = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp', 'gtp_remat'])
    base_stack = _make_stack(_make_config(), pgc)
    for layer in base_stack:
        layer.cuda()
    for p in base_stack.parameters():
        dist.broadcast(p.data, src=0)
    saved = {n: p.data.clone() for n, p in base_stack.named_parameters()}
    base_ddp, base_optim = _build_ddp_distopt_and_optim(base_stack)
    base_gn = _run_step_distopt(base_ddp, base_optim, rank)

    ps.destroy_model_parallel()
    GTPShardedParam._chain_state = {}

    # ---------- Phase B: GTP_remat=2 DP=2, dist-opt + Adam ----------
    ps.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, gtp_remat_size=2
    )
    model_parallel_cuda_manual_seed(42)
    pgc = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp', 'gtp_remat'])
    gtp_stack = _make_stack(_make_config(), pgc)
    for layer in gtp_stack:
        layer.cuda()
    g = ps.get_gtp_weight_remat_group()
    gtp_rank = g.rank()
    for name, p in gtp_stack.named_parameters():
        full = saved[name]
        if isinstance(p, GTPShardedParam):
            ss = p.shape[0]
            p.data.copy_(full[gtp_rank * ss : (gtp_rank + 1) * ss])
        else:
            p.data.copy_(full)
    gtp_ddp, gtp_optim = _build_ddp_distopt_and_optim(gtp_stack)
    gtp_gn = _run_step_distopt(gtp_ddp, gtp_optim, rank)

    ps.destroy_model_parallel()
    GTPShardedParam._chain_state = {}

    if rank == 0:
        ratio = gtp_gn / max(base_gn, 1e-12)
        print(
            f"\n[distopt grad-norm] baseline={base_gn:.6f}  GTP_remat={gtp_gn:.6f}  "
            f"ratio={ratio:.4f}",
            flush=True,
        )
        # Same model, same data, gradients proven equal -> grad-norm must match.
        torch.testing.assert_close(torch.tensor(gtp_gn), torch.tensor(base_gn), atol=0, rtol=3e-2)


# ---------------------------------------------------------------------------
# MoE + EGTP_remat dist-opt grad-norm path (EGTP_remat shards expert weights)
# ---------------------------------------------------------------------------

NUM_EXPERTS = 4
MOE_FFN = 256


def _make_moe_config():
    from megatron.core.transformer.transformer_config import TransformerConfig

    return TransformerConfig(
        num_attention_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        hidden_size=HIDDEN,
        ffn_hidden_size=FFN_HIDDEN,
        num_moe_experts=NUM_EXPERTS,
        moe_router_topk=2,
        moe_ffn_hidden_size=MOE_FFN,
        moe_grouped_gemm=True,
        moe_token_dispatcher_type="alltoall",
        moe_aux_loss_coeff=0.0,
        add_bias_linear=False,
        params_dtype=dtype,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        bias_dropout_fusion=False,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
    )


def _make_moe_stack(config, pg_collection):
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec

    spec = get_gpt_layer_with_transformer_engine_spec(
        num_experts=NUM_EXPERTS, moe_grouped_gemm=True
    )
    return torch.nn.ModuleList(
        [
            spec.module(config, spec.submodules, layer_number=i + 1, pg_collection=pg_collection)
            for i in range(NUM_LAYERS)
        ]
    )


def _is_expert_param(name, p):
    return ('experts' in name) or (not getattr(p, 'allreduce', True))


def _worker_depth_grad_invariance(rank, world_size, port, moe=False):
    """main_grad must be BITWISE identical at every wgrad_finalize_depth, through DDP.

    The plain-chain invariance test in test_gtp_basics cannot see the hazard this targets:
    DDP dispatches a bucket's reduce-scatter from inside grad-ready, and it reduces the WHOLE
    bucket. So the invariant is not "each param's main_grad ends up right" but "every param in
    the bucket has its add landed before the bucket is dispatched". Deferring the finalize moves
    a param's add and its grad-ready together, but moves both relative to its bucket peers -- and
    with several microbatches a queued entry can carry its grad-ready into the next microbatch,
    where register_grad_ready counts it again.

    Runs several microbatches (no_sync on all but the last, as the training loop does), reduces,
    and compares. No optimizer step, so every depth starts from identical weights.
    """
    from contextlib import nullcontext

    import megatron.core.tensor_parallel.generalized_tensor_parallelism as gtp_module
    from megatron.core import parallel_state as ps
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.tensor_parallel.generalized_tensor_parallelism import (
        wait_for_gtp_grad_reduction_on_current_stream,
    )
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    num_microbatches = 4

    ps.destroy_model_parallel()
    # Every worker in this file resets the chain-linking anchor before building a stack: a stale
    # _chain_state["last_weight"] gives this stack's first weight a non-None prev_w, so it issues
    # an async RS instead of the sync one and the chain has no head -- and the state left behind
    # leaks into the next worker.
    GTPShardedParam._chain_state = {}
    if moe:
        # MoE/EGTP: grouped experts add their own chains and their own (chain, group) queues,
        # and the expert weights sit in different DDP buckets from the dense ones. None =>
        # pull every group from the MPU globals (the token dispatcher needs tp_ep).
        ps.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=2,
            gtp_remat_size=2,
            expert_gtp_remat_size=2,
        )
        model_parallel_cuda_manual_seed(42)
        pgc = ProcessGroupCollection.use_mpu_process_groups(required_pgs=None)
        stack = _make_moe_stack(_make_moe_config(), pgc)
    else:
        ps.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1, gtp_remat_size=2
        )
        model_parallel_cuda_manual_seed(42)
        pgc = ProcessGroupCollection.use_mpu_process_groups(
            required_pgs=['tp', 'cp', 'gtp_remat']
        )
        stack = _make_stack(_make_config(), pgc)
    for layer in stack:
        layer.cuda()
    for n, p in stack.named_parameters():
        # Expert weights are EP-local and must not be broadcast (see _worker_moe_distopt).
        if not _is_expert_param(n, p):
            dist.broadcast(p.data, src=0)

    # Small buckets so there is more than one bucket group: a single bucket cannot expose an
    # early dispatch, because there is no second bucket whose params are still accumulating.
    ddp_model, optim = _build_ddp_distopt_and_optim(
        stack, overlap_grad_reduce=True, bucket_size=10_000
    )

    torch.manual_seed(1000 + rank)
    inputs = [
        torch.randn(SEQ, BATCH, HIDDEN, dtype=dtype, device='cuda')
        for _ in range(num_microbatches)
    ]

    def _grads_at_depth(depth):
        saved_depth = gtp_module.GTP_CONFIG.wgrad_finalize_depth
        gtp_module.GTP_CONFIG.wgrad_finalize_depth = depth
        try:
            optim.zero_grad()
            ddp_model.zero_grad_buffer()
            for mb, x in enumerate(inputs):
                last = mb == len(inputs) - 1
                ctx = nullcontext() if last else ddp_model.no_sync()
                with ctx:
                    out = x
                    for layer in ddp_model.module.children():
                        out, _ = layer(out, attention_mask=None)
                    out.float().mean().backward()
                # No manual drain: the autograd end-of-backward callback settles the queue.
                # Assert that it actually did, for every queue including the MoE/EGTP chains.
                leaked = {
                    key: [w._debug_name for w in q]
                    for key, q in gtp_module._PENDING_WGRAD_RS.items()
                    if q
                }
                assert not leaked, (
                    f"depth={depth}: reduce-scatters still queued at the end of microbatch "
                    f"{mb} (of {num_microbatches}): {leaked}"
                )
            # Production order (finalize_model_grads): fence GTP, then reduce across DP.
            wait_for_gtp_grad_reduction_on_current_stream()
            ddp_model.finish_grad_sync()
            torch.cuda.synchronize()
            return {
                n: p.main_grad.detach().clone()
                for n, p in ddp_model.module.named_parameters()
                if hasattr(p, 'main_grad')
            }
        finally:
            gtp_module.GTP_CONFIG.wgrad_finalize_depth = saved_depth

    def _worst_delta(a, b):
        worst_name, worst = None, 0.0
        for name, ref in a.items():
            delta = (ref.float() - b[name].float()).abs().max().item()
            if delta > worst:
                worst_name, worst = name, delta
        return worst_name, worst

    reference = _grads_at_depth(1)
    assert reference, "no main_grad captured"
    assert any(g.abs().max() > 0 for g in reference.values()), "reference grads are all zero"

    # CONTROL first: repeat the SAME depth. MoE dispatch/permutation can use atomics, so this
    # config is not guaranteed bitwise reproducible; without measuring that, a between-depth
    # difference cannot be attributed to depth. The control sets the bar the depth runs must meet.
    control = _grads_at_depth(1)
    ctl_name, ctl_delta = _worst_delta(reference, control)
    print(f"[rank {rank}] same-depth control: worst max|delta|={ctl_delta:.6g} on {ctl_name}")

    tolerance = ctl_delta * 4 if ctl_delta > 0 else 0.0
    for depth in (2, 3):
        got = _grads_at_depth(depth)
        name, delta = _worst_delta(reference, got)
        print(f"[rank {rank}] depth {depth}: worst max|delta|={delta:.6g} on {name}")
        assert delta <= tolerance, (
            f"wgrad_finalize_depth={depth} changed main_grad through DDP over "
            f"{num_microbatches} microbatches: worst max|delta|={delta:.6g} on {name}, "
            f"vs same-depth control {ctl_delta:.6g} (tolerance {tolerance:.6g}). "
            f"Depth must be a scheduling knob with no numerical footprint."
        )

    for queue in gtp_module._PENDING_WGRAD_RS.values():
        assert not queue, "work left queued after the end-of-backward flush"

    ps.destroy_model_parallel()
    GTPShardedParam._chain_state = {}


def _worker_moe_distopt(rank, world_size, port):
    from megatron.core import parallel_state as ps
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    # None => pull every group from the MPU globals. An explicit list is a trap here: the MoE
    # token dispatcher reads pg_collection.tp_ep, and an omitted group comes back as None rather
    # than raising, so the dispatcher silently gathers over a size-1 group and blows up in
    # preprocess() with "shape '[ep,tp,num_experts]' is invalid".
    pgs = None

    # ---------- Phase A: baseline GTP1/EGTP1, EP2 (DP2 dense / expert_dp2) ----------
    ps.destroy_model_parallel()
    ps.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=2,
        gtp_remat_size=1,
        expert_gtp_remat_size=1,
    )
    model_parallel_cuda_manual_seed(42)
    pgc = ProcessGroupCollection.use_mpu_process_groups(required_pgs=pgs)
    base_stack = _make_moe_stack(_make_moe_config(), pgc)
    for layer in base_stack:
        layer.cuda()
    # Broadcast only NON-expert (dense) params; expert weights are EP-local and must
    # stay rank-distinct. Save all params per-rank for the GTP_remat phase to mirror.
    for name, p in base_stack.named_parameters():
        if not _is_expert_param(name, p):
            dist.broadcast(p.data, src=0)
    saved = {n: p.data.clone() for n, p in base_stack.named_parameters()}
    base_ddp, base_optim = _build_ddp_distopt_and_optim(base_stack)
    base_gn = _run_step_distopt(base_ddp, base_optim, rank)

    ps.destroy_model_parallel()
    GTPShardedParam._chain_state = {}

    # ---------- Phase B: GTP2/EGTP2, EP2 (EGTP_remat actually shards experts) ----------
    ps.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=2,
        gtp_remat_size=2,
        expert_gtp_remat_size=2,
    )
    model_parallel_cuda_manual_seed(42)
    pgc = ProcessGroupCollection.use_mpu_process_groups(required_pgs=pgs)
    moe_stack = _make_moe_stack(_make_moe_config(), pgc)
    for layer in moe_stack:
        layer.cuda()
    g = ps.get_gtp_weight_remat_group()
    eg = ps.get_expert_gtp_weight_remat_group()
    gtp_rank, egtp_rank = g.rank(), eg.rank()
    n_egtp_sharded = 0
    for name, p in moe_stack.named_parameters():
        full = saved[name]  # EP2 layout identical to baseline -> rank-local match
        if isinstance(p, GTPShardedParam):
            # dense GTP_remat shards over gtp_remat group; expert (EGTP_remat) over egtp_remat.
            is_expert = _is_expert_param(name, p)
            r = egtp_rank if is_expert else gtp_rank
            ss = p.shape[0]
            p.data.copy_(full[r * ss : (r + 1) * ss])
            if is_expert:
                n_egtp_sharded += 1
        else:
            p.data.copy_(full)
    if rank == 0:
        print(
            f"[moe-egtp] egtp-sharded expert params = {n_egtp_sharded} (must be >0 to be a "
            f"faithful EGTP_remat test)",
            flush=True,
        )
    moe_ddp, moe_optim = _build_ddp_distopt_and_optim(moe_stack)
    moe_gn = _run_step_distopt(moe_ddp, moe_optim, rank)

    ps.destroy_model_parallel()
    GTPShardedParam._chain_state = {}

    if rank == 0:
        ratio = moe_gn / max(base_gn, 1e-12)
        print(
            f"\n[moe distopt grad-norm] baseline={base_gn:.6f}  GTP_remat={moe_gn:.6f}  "
            f"ratio={ratio:.4f}",
            flush=True,
        )
        torch.testing.assert_close(torch.tensor(moe_gn), torch.tensor(base_gn), atol=0, rtol=3e-2)


# ---------------------------------------------------------------------------
# GTP_remat + reduce_scatter_with_fp32_accumulation
# ---------------------------------------------------------------------------

# Small enough to split the stack into several bucket groups: with one big bucket every wgrad
# has landed before the single dispatch, hiding a grad-ready ordering bug.
FP32ACCUM_BUCKET_SIZE = 20000
FP32ACCUM_STEPS = 3


def _reset_dist_reduce_scatter_func():
    """Undo param_and_grad_buffer's sticky module-level ``dist_reduce_scatter_func``.

    It is set (never reset) by the first fp32-accum DDP, so without this a later plain-RS phase
    keeps using fp32-accum and the comparison passes vacuously.
    """
    import megatron.core.distributed.param_and_grad_buffer as pgb

    pgb.dist_reduce_scatter_func = torch.distributed._reduce_scatter_base


def _run_gtp2_phase(rank, saved, fp32_accum):
    """GTP_remat=2 dist-opt run with overlapped grad reduce; returns per-step grad-norms."""
    from megatron.core import parallel_state as ps
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    ps.destroy_model_parallel()
    ps.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, gtp_remat_size=2
    )
    model_parallel_cuda_manual_seed(42)
    pgc = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp', 'gtp_remat'])
    stack = _make_stack(_make_config(), pgc)
    for layer in stack:
        layer.cuda()
    gtp_rank = ps.get_gtp_weight_remat_group().rank()
    for name, p in stack.named_parameters():
        full = saved[name]
        if isinstance(p, GTPShardedParam):
            ss = p.shape[0]
            p.data.copy_(full[gtp_rank * ss : (gtp_rank + 1) * ss])
        else:
            p.data.copy_(full)

    _reset_dist_reduce_scatter_func()
    ddp_model, optim = _build_ddp_distopt_and_optim(
        stack,
        overlap_grad_reduce=True,
        bucket_size=FP32ACCUM_BUCKET_SIZE,
        reduce_scatter_with_fp32_accumulation=fp32_accum,
        grad_reduce_in_fp32=False,  # bf16 on the wire: the regime fp32-accum exists for
    )
    assert len(ddp_model.bucket_groups) > 1, (
        f"expected >1 bucket group at bucket_size={FP32ACCUM_BUCKET_SIZE}, "
        f"got {len(ddp_model.bucket_groups)} -- test would not cover the dispatch-ordering hazard"
    )
    grad_norms = [
        _run_step_distopt(ddp_model, optim, rank + 17 * it) for it in range(FP32ACCUM_STEPS)
    ]

    ps.destroy_model_parallel()
    GTPShardedParam._chain_state = {}
    return grad_norms


def _worker_fp32accum(rank, world_size, port):
    """GTP_remat=2 grad-norms must be identical with and without fp32-accumulation RS.

    fp32-accum reads grad_data through an all-to-all at dispatch time, while GTP defers its
    wgrad ``main_grad.add_`` to a later backward node. If grad-ready fires from autograd instead
    of GTP's manual hook, the all-to-all reads grad_data before the add lands. Both phases share
    weights and data, so any difference is that staleness.
    """
    from megatron.core import parallel_state as ps
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    # Common (unsharded) init weights from a GTP_remat=1 build.
    ps.destroy_model_parallel()
    ps.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, gtp_remat_size=1
    )
    model_parallel_cuda_manual_seed(42)
    pgc = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp', 'gtp_remat'])
    ref_stack = _make_stack(_make_config(), pgc)
    for layer in ref_stack:
        layer.cuda()
    for p in ref_stack.parameters():
        dist.broadcast(p.data, src=0)
    saved = {n: p.data.clone() for n, p in ref_stack.named_parameters()}
    del ref_stack
    ps.destroy_model_parallel()
    GTPShardedParam._chain_state = {}

    plain_gns = _run_gtp2_phase(rank, saved, fp32_accum=False)
    fp32_gns = _run_gtp2_phase(rank, saved, fp32_accum=True)
    _reset_dist_reduce_scatter_func()  # don't leak fp32-accum into later tests

    if rank == 0:
        for step, (pg, fg) in enumerate(zip(plain_gns, fp32_gns)):
            print(
                f"[fp32accum] step {step}: grad_norm plainRS={pg:.6f} fp32accum={fg:.6f}",
                flush=True,
            )
        torch.testing.assert_close(
            torch.tensor(fp32_gns), torch.tensor(plain_gns), atol=0, rtol=1e-3
        )


# ---------------------------------------------------------------------------
# GTP_remat wgrad reduce-scatter via FP32 accumulation
# (--gtp-remat-reduce-scatter-with-fp32-accumulation)
# ---------------------------------------------------------------------------


def _reset_gtp_global_state():
    """Drop GTP process-globals sized against the current layout.

    The weight cache and wgrad pool are keyed by shape and outlive `destroy_model_parallel()`,
    so a phase that changes the gtp_remat degree would be handed a stale-sized buffer.
    """
    import megatron.core.tensor_parallel.generalized_tensor_parallelism as gtp_module

    GTPShardedParam._chain_state = {}
    gtp_module.get_global_GTP_cache().clear()
    gtp_module._wgrad_buf_pool.clear()
    gtp_module._inflight_comm_params.clear()


def _gtp_rs_phase(rank, saved, fp32_accum, moe=False, gtp_size=4):
    """One GTP_remat backward; returns the full (gathered) reduced gradients per param.

    `saved` may be None when only the dispatch counts matter. gtp_size defaults to 4 because
    size <= 2 is bypassed -- which the bypass test asserts and which would otherwise make the
    gradient comparison vacuous.
    """
    import megatron.core.tensor_parallel.generalized_tensor_parallelism as gtp_module
    from megatron.core import parallel_state as ps
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    ps.destroy_model_parallel()
    init_kwargs = dict(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, gtp_remat_size=gtp_size
    )
    if moe:
        # EP=1 (not 2) so all 4 ranks form ONE expert-GTP group: EP=2 would cap EGTP_remat at 2
        # on a 4-GPU run, which the size-<=2 bypass turns back into a plain reduce-scatter.
        init_kwargs.update(expert_model_parallel_size=1, expert_gtp_remat_size=gtp_size)
    ps.initialize_model_parallel(**init_kwargs)
    model_parallel_cuda_manual_seed(42)
    # None => every group; the MoE dispatcher needs tp_ep (see _worker_moe_distopt).
    pgc = ProcessGroupCollection.use_mpu_process_groups()
    stack = (_make_moe_stack if moe else _make_stack)(
        _make_moe_config() if moe else _make_config(), pgc
    )
    for layer in stack:
        layer.cuda()
    assert ps.get_gtp_weight_remat_group().size() == gtp_size
    if saved is not None:
        _load_gtp_shards(stack, saved, moe)

    gtp_module.update_gtp_config(reduce_scatter_with_fp32_accumulation=fp32_accum)
    try:
        # Non-dist-opt DDP so main_grad holds the full reduced gradient and can be compared
        # directly (the dist-opt buffer only has this rank's reduce-scattered slice).
        _run_one_backward(_build_ddp(stack), rank)
        grads = _full_main_grads(stack)
    finally:
        gtp_module.update_gtp_config(reduce_scatter_with_fp32_accumulation=False)

    ps.destroy_model_parallel()
    _reset_gtp_global_state()
    return grads


@contextlib.contextmanager
def _fp32accum_probe():
    """Count fp32-accum dispatches and coalesced batched composites; yields the counter dict.

    Both counters guard against a vacuous pass -- a flag that never switched the collective, or
    a batched path that regressed to one launch per weight. Patching the module attribute works
    because ``_reduce_scatter_fp32_accum`` imports the primitive at call time.
    """
    import megatron.core.distributed.reduce_scatter_with_fp32_accumulation as rs_fp32_module
    import megatron.core.tensor_parallel.generalized_tensor_parallelism as gtp_module

    counts = {"dispatches": 0, "coalesced_composites": 0}
    orig_rs = rs_fp32_module.reduce_scatter_with_fp32_accumulation
    orig_init = gtp_module._GTPCompositeWorkHandle.__init__

    def _counting_rs(*args, **kwargs):
        counts["dispatches"] += 1
        return orig_rs(*args, **kwargs)

    def _counting_init(self, handles):
        # The batched path builds _GTPCompositeWorkHandle([cm, *sum_handles]): a coalescing
        # manager first, then the deferred FP32 sums.
        if handles and isinstance(handles[0], dist.distributed_c10d._CoalescingManager):
            counts["coalesced_composites"] += 1
        orig_init(self, handles)

    rs_fp32_module.reduce_scatter_with_fp32_accumulation = _counting_rs
    gtp_module._GTPCompositeWorkHandle.__init__ = _counting_init
    try:
        yield counts
    finally:
        rs_fp32_module.reduce_scatter_with_fp32_accumulation = orig_rs
        gtp_module._GTPCompositeWorkHandle.__init__ = orig_init


def _saved_unsharded_weights(moe=False):
    """Unsharded init weights from a GTP_remat=1 build, so every phase starts identically.

    EP=1, so expert weights are DP replicas rather than EP-local and every param can be
    broadcast.
    """
    from megatron.core import parallel_state as ps
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    ps.destroy_model_parallel()
    init_kwargs = dict(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, gtp_remat_size=1
    )
    if moe:
        init_kwargs.update(expert_model_parallel_size=1, expert_gtp_remat_size=1)
    ps.initialize_model_parallel(**init_kwargs)
    model_parallel_cuda_manual_seed(42)
    pgc = ProcessGroupCollection.use_mpu_process_groups()
    ref_stack = (_make_moe_stack if moe else _make_stack)(
        _make_moe_config() if moe else _make_config(), pgc
    )
    for layer in ref_stack:
        layer.cuda()
    for _name, p in ref_stack.named_parameters():
        dist.broadcast(p.data, src=0)
    saved = {n: p.data.clone() for n, p in ref_stack.named_parameters()}
    del ref_stack
    ps.destroy_model_parallel()
    _reset_gtp_global_state()
    return saved


def _max_rel_grad_diff(reference, other):
    """Largest per-param relative gradient difference -> (value, param name).

    Relative, not absolute: max|grad| is ~1e-4 on this micro-model, so an absolute bound would
    read benign rounding as a huge error. All-zero params are skipped.
    """
    worst, worst_name = 0.0, None
    for name, ref_grad in reference.items():
        denom = ref_grad.abs().max().item()
        if denom < 1e-30:
            continue
        rel = (ref_grad - other[name]).abs().max().item() / denom
        if rel > worst:
            worst, worst_name = rel, name
    return worst, worst_name


def _worker_gtp_rs_fp32accum(rank, world_size, port, moe):
    """FP32 accumulation must not change the reduced gradients.

    It alters only the summation precision of the gtp_remat reduce-scatter, never its math.
    Guards the integration: the pooled all-to-all scratch, the deferred FP32 sums, and
    (moe=True) the batched grouped path.
    """
    saved = _saved_unsharded_weights(moe)

    with _fp32accum_probe() as counts:
        plain = _gtp_rs_phase(rank, saved, fp32_accum=False, moe=moe)
        n_plain = counts["dispatches"]
        fp32 = _gtp_rs_phase(rank, saved, fp32_accum=True, moe=moe)
        n_fp32 = counts["dispatches"] - n_plain
        n_coalesced = counts["coalesced_composites"]

    assert n_plain == 0, f"FP32-accum RS ran with the flag off ({n_plain} dispatches)"
    assert n_fp32 > 0, "FP32-accum RS never ran with the flag on -- test would be vacuous"
    if moe:
        # Only grouped expert chains carry several wgrads per RS, so only they build a composite.
        assert n_coalesced > 0, (
            "batched grouped RS did not coalesce its all-to-alls -- regressed to one "
            "ncclGroupStart/End per weight"
        )

    if rank == 0:
        worst, worst_name = _max_rel_grad_diff(plain, fp32)
        print(
            f"[gtp-rs-fp32accum moe={moe}] dispatches={n_fp32} "
            f"max rel grad diff={worst:.3e} ({worst_name})",
            flush=True,
        )
        # Tolerance, not equality: the paths legitimately differ by the rounding fp32-accum
        # removes. Structural breakage (dropped mean, unwaited handle) lands near 1.0.
        assert worst < 2e-2, (
            f"GTP FP32-accumulation reduce-scatter changed the reduced gradient "
            f"(max rel diff {worst:.3e} on {worst_name})"
        )


def _worker_gtp_rs_fp32accum_bypassed_at_size_2(rank, world_size, port):
    """A gtp_remat axis of size 2 must fall back to the plain reduce-scatter, flag or not.

    One addition rounds the same either way, so fp32-accum cannot change the result at size 2
    while still costing the all-to-all scratch. Enabling the flag there must issue zero
    fp32-accum collectives.
    """
    with _fp32accum_probe() as counts:
        # saved=None: only the dispatch count matters here, not the gradient values.
        _gtp_rs_phase(rank, saved=None, fp32_accum=True, gtp_size=2)

    assert counts["dispatches"] == 0, (
        f"gtp_remat=2 issued {counts['dispatches']} FP32-accumulation reduce-scatters; the "
        "size-<=2 bypass is not firing, so the run pays the extra all-to-all buffer for no "
        "precision gain"
    )


def _worker_idog_span(rank, world_size, port):
    """Dist-opt grad-stats group (intra_dist_opt) must span the FULL world for both
    dense-only and MoE(EP2/EGTP2) configs. A naive build collapses the MoE case to a sub-world
    group (egtp factored out of expert_data_parallel_size), under-counting the grad-norm."""
    from megatron.core import parallel_state as ps

    # MoE EP2 EGTP2 GTP2 expert config.
    ps.destroy_model_parallel()
    ps.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=2,
        gtp_remat_size=2,
        expert_gtp_remat_size=2,
    )
    moe_idog = ps.get_intra_distributed_optimizer_instance_group().size()
    ps.destroy_model_parallel()
    # Dense-only GTP2 (must remain world too).
    ps.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, gtp_remat_size=2
    )
    dense_idog = ps.get_intra_distributed_optimizer_instance_group().size()
    ps.destroy_model_parallel()
    if rank == 0:
        print(
            f"[idog] MoE intra_dist_opt.size={moe_idog} dense.size={dense_idog} "
            f"(world={world_size})",
            flush=True,
        )
        assert moe_idog == world_size, (
            f"MoE grad-stats group = {moe_idog}, expected world {world_size} "
            f"-> grad-norm would under-count gtp_remat/egtp_remat-sharded params"
        )
        assert dense_idog == world_size, f"dense grad-stats group = {dense_idog}"


class TestGTPGradCorrectness:
    def test_distopt_gradstats_group_spans_world(self):
        """intra_dist_opt_group (grad-stats) must span the full world."""
        if torch.cuda.device_count() < 4:
            pytest.skip("Requires 4 CUDA devices")
        _run_distributed(_worker_idog_span, 4)

    @pytest.mark.parametrize("per_token_loss", [False, True])
    def test_gtp2_dp2_grad_matches_dp4_baseline(self, per_token_loss):
        """GTP2xDP2 reduced grad must match no-GTP_remat DP4 (non-dist-opt main_grad).

        per_token_loss=True disables DDP's 1/dp pre-scaling and normalizes by
        1/total_global_tokens, so the gtp_remat axis must be SUM-reduced (plain reduce-scatter +
        SUM finalize), NOT the 1/gtp MEAN used otherwise. A regression to an unconditional mean
        shrinks every gtp grad by 1/gtp and the per_token_loss case catches it (GTP2xDP2 sum-grad
        must still match the DP4 sum-grad). GTP_CONFIG is a process-global, so set it for the run
        and always reset it.
        """
        if torch.cuda.device_count() < 4:
            pytest.skip("Requires 4 CUDA devices")
        from megatron.core.tensor_parallel.generalized_tensor_parallelism import update_gtp_config

        update_gtp_config(calculate_per_token_loss=per_token_loss)
        try:
            _run_distributed(_worker, 4, per_token_loss)
        finally:
            update_gtp_config(calculate_per_token_loss=False)

    @pytest.mark.parametrize("moe", [False, True])
    def test_gtp_remat_rs_fp32_accumulation_preserves_grads(self, moe):
        """--gtp-remat-reduce-scatter-with-fp32-accumulation must not change the gradients.

        moe=False covers the single-tensor RS; moe=True covers the batched grouped/expert RS,
        whose coalesced all-to-alls and deferred FP32 sums are joined under one composite
        handle -- asserted, not just exercised, so a regression to per-weight launches fails.
        """
        if torch.cuda.device_count() < 4:
            pytest.skip("Requires 4 CUDA devices")
        _run_distributed(_worker_gtp_rs_fp32accum, 4, moe)

    def test_gtp_remat_rs_fp32_accumulation_bypassed_at_size_2(self):
        """A size-2 gtp_remat axis must ignore the flag (no gain, real memory cost)."""
        if torch.cuda.device_count() < 4:
            pytest.skip("Requires 4 CUDA devices")
        _run_distributed(_worker_gtp_rs_fp32accum_bypassed_at_size_2, 4)

    def test_gtp2_fp32accum_rs_grad_norm_matches_plain_rs(self):
        """GTP_remat + --ddp-reduce-scatter-with-fp32-accumulation must match plain RS.

        Regression guard for the dispatch-ordering hazard: the fp32-accum all-to-all reads
        grad_data at dispatch, so DDP grad-ready has to fire from GTP's manual post-add hook,
        not from autograd. Needs >1 bucket group (small bucket_size) to be sensitive.
        """
        if torch.cuda.device_count() < 4:
            pytest.skip("Requires 4 CUDA devices")
        _run_distributed(_worker_fp32accum, 4)

    def test_gtp2_dp2_distopt_grad_norm_matches_dp4_baseline(self):
        """GTP2xDP2 dist-opt grad-norm must match no-GTP_remat DP4 (the 64-GPU path)."""
        if torch.cuda.device_count() < 4:
            pytest.skip("Requires 4 CUDA devices")
        _run_distributed(_worker_distopt, 4)

    def test_depth_grad_invariance_through_ddp(self):
        """wgrad_finalize_depth must not change main_grad, through DDP + microbatches."""
        if torch.cuda.device_count() < 4:
            pytest.skip("Requires 4 CUDA devices")
        _run_distributed(_worker_depth_grad_invariance, 4)

    def test_depth_grad_invariance_through_ddp_moe(self):
        """Same invariant on the MoE/EGTP path: more chains, more queues, more buckets."""
        if torch.cuda.device_count() < 4:
            pytest.skip("Requires 4 CUDA devices")
        _run_distributed(_worker_depth_grad_invariance, 4, True)

    def test_moe_egtp_distopt_grad_norm_matches_baseline(self):
        """GTP2/EGTP2 MoE dist-opt grad-norm must match GTP1/EGTP1 baseline (EP=2 both)."""
        if torch.cuda.device_count() < 4:
            pytest.skip("Requires 4 CUDA devices")
        _run_distributed(_worker_moe_distopt, 4)
