# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""GTP_remat gradient correctness through the real DDP + finalize path, with replicate (DP) > 1.

Each test builds the same model twice on 4 ranks -- an unsharded reference and a GTP/EGTP-sharded
one -- loads identical weights, feeds each data-parallel position a distinct batch (seed=rank) and
compares the reduced gradients per parameter. Both layouts span the same 4 positions, so the
correctly reduced gradient is the same mean (or, under calculate_per_token_loss, the same sum).
A gtp-axis scaling error shows up as a uniform factor on every sharded parameter.
"""

import contextlib

import pytest
import torch
import torch.distributed as dist

from megatron.core.tensor_parallel.gtp_api import HAVE_GTP

if not HAVE_GTP:
    pytest.skip("GTP requires TransformerEngine >= 2.19", allow_module_level=True)

import megatron.core.tensor_parallel.generalized_tensor_parallelism as gtp_module
from megatron.core import parallel_state as ps
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.distributed.finalize_model_grads import (
    _allreduce_replicated_grads_over_gtp_remat_group,
)
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.generalized_tensor_parallelism import (
    GTPShardedParam,
    reset_gtp_state,
    update_gtp_config,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.generalized_tensor_parallel.gtp_test_utils import (
    _make_gtp_linear,
    cached_group,
)

HIDDEN = 256
NUM_HEADS = 8
FFN_HIDDEN = 512
SEQ = 16
dtype = torch.bfloat16
NUM_WIDE_STREAMS = 3
NUM_EXPERTS = 4
# BF16 reduction-order noise is ~5e-3 of max|grad| here; a dropped mean or a missing shard is ~1.
GRAD_REL_TOL = 2e-2


@pytest.fixture(autouse=True)
def _requires_four_ranks():
    if dist.get_world_size() != 4:
        pytest.skip("Requires 4 ranks")


def _make_config(gtp=1, moe=False, egtp=1, ep=2, per_token=False):
    """Dense or MoE single-layer config; the GTP/EGTP/EP sizes must match the MPU's."""
    moe_kwargs = (
        dict(
            num_moe_experts=NUM_EXPERTS,
            moe_router_topk=2,
            moe_ffn_hidden_size=HIDDEN,
            moe_grouped_gemm=True,
            moe_token_dispatcher_type="alltoall",
            moe_aux_loss_coeff=0.0,
            expert_model_parallel_size=ep,
            expert_tensor_parallel_num_weight_shards=egtp,
        )
        if moe
        else {}
    )
    return TransformerConfig(
        num_attention_heads=NUM_HEADS,
        num_layers=1,
        hidden_size=HIDDEN,
        ffn_hidden_size=FFN_HIDDEN,
        add_bias_linear=False,
        params_dtype=dtype,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        bias_dropout_fusion=False,
        tensor_parallel_num_weight_shards=gtp,
        calculate_per_token_loss=per_token,
        **moe_kwargs,
    )


def _is_expert_param(name, p):
    return ('experts' in name) or (not getattr(p, 'allreduce', True))


def _reset_gtp_global_state():
    """Drop GTP process-globals sized against the previous phase's layout.

    The weight cache and wgrad pool are keyed by shape and outlive `destroy_model_parallel()`,
    so a phase that changes the gtp_remat degree would be handed a stale-sized buffer.
    """
    reset_gtp_state()
    gtp_module.get_global_GTP_cache().clear()
    gtp_module._wgrad_buf_pool.clear()
    gtp_module._inflight_comm_params.clear()


def _build_stack(gtp=1, moe=False, egtp=1, ep=2, per_token=False):
    """Initialize the MPU for this phase and build one GPT layer on it."""
    ps.destroy_model_parallel()
    _reset_gtp_global_state()
    kwargs = dict(gtp_remat_size=gtp)
    if moe:
        kwargs.update(expert_model_parallel_size=ep, expert_gtp_remat_size=egtp)
    ps.initialize_model_parallel(**kwargs)
    model_parallel_cuda_manual_seed(42)
    config = _make_config(gtp, moe, egtp, ep, per_token)
    spec = get_gpt_layer_with_transformer_engine_spec(
        num_experts=NUM_EXPERTS if moe else None, moe_grouped_gemm=moe
    )
    # Every MPU group: the MoE token dispatcher reads pg_collection.tp_ep.
    pgc = ProcessGroupCollection.use_mpu_process_groups()
    layer = spec.module(config, spec.submodules, layer_number=1, pg_collection=pgc)
    return torch.nn.ModuleList([layer]).cuda()


def _unsharded_weights(stack, moe=False):
    """Make every non-EP-local weight identical on all ranks and return a copy."""
    for name, p in stack.named_parameters():
        if not (
            moe and _is_expert_param(name, p) and ps.get_expert_model_parallel_world_size() > 1
        ):
            dist.broadcast(p.data, src=0)
    return {n: p.data.clone() for n, p in stack.named_parameters()}


def _load_gtp_shards(stack, saved, moe=False):
    """Copy each param's own shard of the unsharded `saved` weights into `stack`."""
    gtp_rank = ps.get_gtp_weight_remat_group().rank()
    egtp_rank = ps.get_expert_gtp_weight_remat_group().rank() if moe else 0
    n_sharded = 0
    for name, p in stack.named_parameters():
        full = saved[name]
        if isinstance(p, GTPShardedParam):
            r = egtp_rank if _is_expert_param(name, p) else gtp_rank
            ss = p.shape[0]
            p.data.copy_(full[r * ss : (r + 1) * ss])
            n_sharded += 1
        else:
            p.data.copy_(full)
    assert n_sharded > 0, "no GTP/EGTP-sharded param -- the comparison would be vacuous"


def _build_ddp(stack, use_distributed_optimizer=False, **ddp_kwargs):
    """DDP over the stack with the stack's own config (it carries the GTP/EGTP/EP sizes that
    DDP's expert gradient scaling reads). Without a distributed optimizer, main_grad holds the
    full reduced gradient."""
    module = torch.nn.Sequential()
    for i, layer in enumerate(stack):
        module.add_module(str(i), layer)
    ddp_config = DistributedDataParallelConfig(
        use_distributed_optimizer=use_distributed_optimizer, **ddp_kwargs
    )
    return DistributedDataParallel(stack[0].config, ddp_config, module)


def _backward_and_reduce(ddp_model, rank, x_shape=(SEQ, 1, HIDDEN), x_dtype=dtype, replay=None):
    """Run fwd+bwd on a per-rank batch, then the production grad reduction -> (output, input).

    ``replay`` is a residual-stream recompute context (wide-residual layers only).
    """
    ddp_model.zero_grad_buffer()
    torch.manual_seed(1000 + rank)
    x = torch.randn(*x_shape, dtype=x_dtype, device='cuda', requires_grad=True)
    out = x
    for layer in ddp_model.module.children():
        kw = {} if replay is None else dict(residual_stream_recompute_context=replay)
        out, _ = layer(out, attention_mask=None, **kw)
        if replay is not None:
            replay.finalize(out)
    out.float().square().mean().backward()
    # Production order (finalize_model_grads): reduce across DP first, then the gtp_remat
    # finalize. finish_grad_sync() alone issues the reduction when overlap_grad_reduce=False.
    ddp_model.finish_grad_sync()
    _allreduce_replicated_grads_over_gtp_remat_group(
        [ddp_model],
        ps.get_gtp_weight_remat_group(check_initialized=False),
        ps.get_expert_gtp_weight_remat_group(check_initialized=False),
        calculate_per_token_loss=ddp_model.module[0].config.calculate_per_token_loss,
    )
    return out.detach().float(), x


def _full_main_grads(stack):
    """Full (unsharded) reduced gradient per param name.

    A GTPShardedParam's main_grad is the local shard -> all-gather over its own axis (expert
    params shard over egtp_remat, dense over gtp_remat). Other params are already full.
    """
    out = {}
    for name, p in stack.named_parameters():
        mg = p.main_grad
        if isinstance(p, GTPShardedParam):
            g = (
                ps.get_expert_gtp_weight_remat_group()
                if _is_expert_param(name, p)
                else ps.get_gtp_weight_remat_group()
            )
            shards = [torch.empty_like(mg) for _ in range(g.size())]
            dist.all_gather(shards, mg.contiguous(), group=g)
            out[name] = torch.cat(shards, dim=0).float()
        else:
            out[name] = mg.detach().float()
    return out


def _assert_grads_match(reference, other, label, rel_tol=GRAD_REL_TOL):
    """Every gradient must be finite and within ``rel_tol`` of max|reference grad|.

    Relative to the param's own scale: max|grad| is ~1e-4 on this micro-model, so an absolute
    bound would read benign rounding as a huge error.
    """
    assert reference.keys() == other.keys(), f"{label}: param sets differ"
    worst, worst_name = 0.0, None
    for name, ref in reference.items():
        got = other[name]
        assert got.shape == ref.shape, f"{label} {name}: {got.shape} vs {ref.shape}"
        assert torch.isfinite(ref).all() and torch.isfinite(got).all(), f"{label} {name}: NaN/inf"
        scale = ref.abs().max().item()
        err = (got - ref).abs().max().item()
        rel = err / scale if scale > 0 else err
        if rel >= worst:
            worst, worst_name = rel, name
    print(f"[{label}] max relative grad error {worst:.3e} ({worst_name})", flush=True)
    assert worst < rel_tol, f"{label}: gradient differs by {worst:.3e} on {worst_name}"


# ---------------------------------------------------------------------------
# Per-parameter gradients, dense GTP and MoE EGTP
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "moe,gtp,egtp,per_token",
    [
        pytest.param(False, 2, 1, False, id="dense-gtp2"),
        # Per-token loss: DDP does no 1/dp prescale, so the gtp axis must SUM, not average.
        pytest.param(False, 2, 1, True, id="dense-gtp2-per_token"),
        pytest.param(True, 2, 2, False, id="moe-gtp2-egtp2"),
        # Mismatched sizes exercise DDP's egtp/gtp expert gradient correction both ways.
        pytest.param(True, 4, 1, False, id="moe-gtp4-egtp1"),
        pytest.param(True, 1, 2, False, id="moe-gtp1-egtp2"),
        pytest.param(True, 2, 2, True, id="moe-gtp2-egtp2-per_token"),
    ],
)
def test_gtp_grads_match_baseline(moe, gtp, egtp, per_token):
    """GTP/EGTP-sharded reduced grads equal the unsharded (GTP1/EGTP1) reference per param.

    MoE runs EP=2 in both phases: only EP>1 tags expert weights allreduce=False and builds DDP's
    expert buffers, so the expert DP size (and hence its gradient scaling) differs from dense.
    """
    rank = dist.get_rank()
    update_gtp_config(calculate_per_token_loss=per_token)

    base_stack = _build_stack(moe=moe, per_token=per_token)
    saved = _unsharded_weights(base_stack, moe)
    _backward_and_reduce(_build_ddp(base_stack), rank)
    base_grads = _full_main_grads(base_stack)

    stack = _build_stack(gtp, moe, egtp, per_token=per_token)
    _load_gtp_shards(stack, saved, moe)
    ddp = _build_ddp(stack)
    assert not moe or ddp.expert_parallel_buffers, "no expert buffer -- test setup bug"
    _backward_and_reduce(ddp, rank)
    # Every rank checks: expert grads are EP-local.
    _assert_grads_match(base_grads, _full_main_grads(stack), f"gtp{gtp} egtp{egtp}")


@pytest.mark.parametrize("residual_stream_recompute", [False, True], ids=["plain", "replay"])
def test_wide_residual_gtp2_dp2_matches_dp4_baseline(residual_stream_recompute):
    """Wide-residual outputs and grads match when branch GEMMs are GTP-sharded while the
    wide-residual controllers stay replicated (and through selective residual replay)."""
    from megatron.core.transformer.residual_recompute import build_residual_stream_recompute_plan
    from megatron.core.transformer.wide_residual_config import WideResidualConfig
    from megatron.core.transformer.wide_residual_transformer_layer import (
        WideResidualTransformerLayer,
    )

    rank = dist.get_rank()

    def build(gtp):
        ps.destroy_model_parallel()
        _reset_gtp_global_state()
        ps.initialize_model_parallel(gtp_remat_size=gtp)
        model_parallel_cuda_manual_seed(42)
        recompute = residual_stream_recompute and gtp > 1
        config = TransformerConfig(
            num_attention_heads=NUM_HEADS,
            num_layers=1,
            hidden_size=HIDDEN,
            ffn_hidden_size=FFN_HIDDEN,
            add_bias_linear=False,
            params_dtype=dtype,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            bias_dropout_fusion=False,
            tensor_parallel_num_weight_shards=gtp,
            fp32_residual_connection=True,
            recompute_granularity="selective" if recompute else None,
            recompute_modules=["residual_stream"] if recompute else None,
            residual_stream_recompute_num_layers=1 if recompute else None,
            wide_residual=WideResidualConfig(
                num_streams=NUM_WIDE_STREAMS,
                streamwise_sigmoid_init_scale=0.01,
                learned_retention=True,
                retention_init=0.999,
                retention_max_forget=0.10,
            ),
        )
        pgc = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp', 'gtp_remat'])
        spec = get_gpt_layer_with_transformer_engine_spec()
        layer = WideResidualTransformerLayer(
            config, spec.submodules, layer_number=1, pg_collection=pgc
        )
        return torch.nn.ModuleList([layer]).cuda()

    def run(stack):
        ddp = _build_ddp(stack)
        replay = None
        if "residual_stream" in (stack[0].config.recompute_modules or []):
            (replay,) = build_residual_stream_recompute_plan(1, 1)
        out, x = _backward_and_reduce(
            ddp,
            rank,
            x_shape=(SEQ, 1, NUM_WIDE_STREAMS * HIDDEN),
            x_dtype=torch.float32,
            replay=replay,
        )
        return out, x.grad.float(), _full_main_grads(stack)

    base_stack = build(1)
    saved = _unsharded_weights(base_stack)
    base_out, base_dx, base_grads = run(base_stack)

    stack = build(2)
    _load_gtp_shards(stack, saved)
    controllers = [
        n for n, _ in stack.named_parameters() if 'residual_read_' in n or 'residual_write_' in n
    ]
    assert controllers, "no wide-residual controller parameter was found"
    for name, p in stack.named_parameters():
        assert not (name in controllers and isinstance(p, GTPShardedParam)), name
    out, dx, grads = run(stack)

    torch.testing.assert_close(out, base_out, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(dx, base_dx, atol=2e-6, rtol=2e-2)
    _assert_grads_match(base_grads, grads, "wide residual")
    for name in controllers:
        assert torch.count_nonzero(base_grads[name][:NUM_WIDE_STREAMS]) > 0, name
        torch.testing.assert_close(grads[name], base_grads[name], atol=2e-6, rtol=2e-2, msg=name)


# ---------------------------------------------------------------------------
# Distributed optimizer: grad-norm over GTP/EGTP-sharded params (the production path)
# ---------------------------------------------------------------------------


def _build_ddp_distopt_and_optim(stack, **ddp_kwargs):
    """Real distributed-optimizer setup (Adam), matching the production path."""
    ddp_kwargs.setdefault("overlap_grad_reduce", False)
    ddp_kwargs.setdefault("grad_reduce_in_fp32", True)
    ddp_model = _build_ddp(stack, use_distributed_optimizer=True, **ddp_kwargs)
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
        clip_grad=1.0,  # the reported grad-norm is computed pre-clip
    )
    return ddp_model, get_megatron_optimizer(opt_config, [ddp_model])


def _run_step_distopt(ddp_model, optim, seed):
    """finish_grad_sync -> gtp_remat finalize -> optim.step(); returns the pre-clip grad-norm."""
    optim.zero_grad()
    _backward_and_reduce(ddp_model, seed)
    _, grad_norm, _ = optim.step()
    return float(grad_norm)


@pytest.mark.parametrize("moe", [False, True], ids=["dense-gtp2", "moe-gtp2-egtp2"])
def test_distopt_grad_norm_matches_baseline(moe):
    """The dist-opt grad-norm over GTP/EGTP shards matches the unsharded reference.

    The norm is reduced over the dist-opt grad-stats group, which must cover every gtp/egtp
    shard exactly once (see test_distopt_gradstats_group_spans_world).
    """
    rank = dist.get_rank()
    base_stack = _build_stack(moe=moe)
    saved = _unsharded_weights(base_stack, moe)
    base_gn = _run_step_distopt(*_build_ddp_distopt_and_optim(base_stack), rank)

    stack = _build_stack(2, moe, 2)
    _load_gtp_shards(stack, saved, moe)
    gtp_gn = _run_step_distopt(*_build_ddp_distopt_and_optim(stack), rank)
    print(f"[distopt grad-norm moe={moe}] baseline={base_gn:.6f} gtp={gtp_gn:.6f}", flush=True)
    # bf16 noise floor is ~6e-5 here; a missing or double-counted shard is O(1).
    torch.testing.assert_close(torch.tensor(gtp_gn), torch.tensor(base_gn), atol=0, rtol=1e-4)


def test_distopt_gradstats_group_spans_world():
    """The dist-opt grad-stats group (intra_dist_opt) spans the full world, dense and MoE.

    A naive build collapses the MoE case to a sub-world group (egtp factored out of
    expert_data_parallel_size), under-counting the grad-norm.
    """
    world_size = dist.get_world_size()
    for kwargs in (dict(expert_model_parallel_size=2, expert_gtp_remat_size=2), {}):
        ps.destroy_model_parallel()
        ps.initialize_model_parallel(gtp_remat_size=2, **kwargs)
        size = ps.get_intra_distributed_optimizer_instance_group().size()
        assert size == world_size, f"{kwargs}: grad-stats group {size}, expected {world_size}"


# ---------------------------------------------------------------------------
# DDP --ddp-reduce-scatter-with-fp32-accumulation under GTP_remat
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


def test_gtp2_fp32accum_rs_grad_norm_matches_plain_rs():
    """GTP_remat + --ddp-reduce-scatter-with-fp32-accumulation must match plain RS.

    fp32-accum reads grad_data through an all-to-all at dispatch time, while GTP defers its
    wgrad ``main_grad.add_`` to a later backward node. If grad-ready fires from autograd instead
    of GTP's manual post-add hook, the all-to-all reads grad_data before the add lands. Both
    phases share weights and data, so any difference is that staleness. Needs >1 bucket group.
    """
    rank = dist.get_rank()
    saved = _unsharded_weights(_build_stack())
    grad_norms = {}
    try:
        for fp32_accum in (False, True):
            stack = _build_stack(2)
            _load_gtp_shards(stack, saved)
            _reset_dist_reduce_scatter_func()
            ddp_model, optim = _build_ddp_distopt_and_optim(
                stack,
                overlap_grad_reduce=True,
                bucket_size=FP32ACCUM_BUCKET_SIZE,
                reduce_scatter_with_fp32_accumulation=fp32_accum,
                grad_reduce_in_fp32=False,  # bf16 on the wire: the regime fp32-accum exists for
            )
            assert len(ddp_model.bucket_groups) > 1, "single bucket group hides the hazard"
            grad_norms[fp32_accum] = [
                _run_step_distopt(ddp_model, optim, rank + 17 * it) for it in range(FP32ACCUM_STEPS)
            ]
    finally:
        _reset_dist_reduce_scatter_func()  # don't leak fp32-accum into later tests
    print(f"[fp32accum] plain={grad_norms[False]} fp32accum={grad_norms[True]}", flush=True)
    torch.testing.assert_close(
        torch.tensor(grad_norms[True]), torch.tensor(grad_norms[False]), atol=0, rtol=1e-3
    )


# ---------------------------------------------------------------------------
# GTP_remat wgrad reduce-scatter via FP32 accumulation
# (--gtp-remat-reduce-scatter-with-fp32-accumulation)
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _fp32accum_probe():
    """Count fp32-accum dispatches and coalesced batched composites; yields the counter dict.

    Both counters guard against a vacuous pass -- a flag that never switched the collective, or
    a batched path that regressed to one launch per weight. Patching the module attribute works
    because ``_reduce_scatter_fp32_accum`` imports the primitive at call time.
    """
    import megatron.core.distributed.reduce_scatter_with_fp32_accumulation as rs_fp32_module

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


@pytest.mark.parametrize("moe", [False, True])
def test_gtp_remat_rs_fp32_accumulation_preserves_grads(moe):
    """FP32 accumulation alters only the summation precision of the gtp_remat reduce-scatter.

    moe=False covers the single-tensor RS; moe=True the batched grouped/expert RS, whose
    coalesced all-to-alls and deferred FP32 sums are joined under one composite handle --
    asserted, so a regression to per-weight launches fails. GTP=4 (EP=1, EGTP=4) because a
    size <= 2 axis bypasses fp32-accum.
    """
    rank = dist.get_rank()
    saved = _unsharded_weights(_build_stack(moe=moe, ep=1), moe)
    grads, dispatches = {}, {}
    with _fp32accum_probe() as counts:
        for fp32_accum in (False, True):
            stack = _build_stack(4, moe, 4, ep=1)
            _load_gtp_shards(stack, saved, moe)
            update_gtp_config(reduce_scatter_with_fp32_accumulation=fp32_accum)
            before = counts["dispatches"]
            _backward_and_reduce(_build_ddp(stack), rank)
            dispatches[fp32_accum] = counts["dispatches"] - before
            grads[fp32_accum] = _full_main_grads(stack)

    assert dispatches[False] == 0, f"FP32-accum RS ran with the flag off: {dispatches}"
    assert dispatches[True] > 0, "FP32-accum RS never ran with the flag on -- vacuous"
    if moe:
        # Only grouped expert chains carry several wgrads per RS, so only they build a composite.
        assert counts["coalesced_composites"] > 0, "batched grouped RS did not coalesce"
    # The paths legitimately differ by the rounding fp32-accum removes.
    _assert_grads_match(grads[False], grads[True], f"gtp-rs-fp32accum moe={moe}")


def test_gtp_remat_rs_fp32_accumulation_bypassed_at_size_2():
    """A size-2 gtp_remat axis reduce-scatters plainly even with the flag on.

    One addition rounds the same either way, so fp32-accum cannot change the result at size 2
    while still costing the all-to-all scratch.
    """
    rank = dist.get_rank()
    groups = [cached_group(ranks) for ranks in ((0, 1), (2, 3))]
    layer = _make_gtp_linear(64, 128, groups[rank // 2])
    update_gtp_config(reduce_scatter_with_fp32_accumulation=True)
    wgrad = torch.full((128, 64), rank + 1.0, dtype=dtype, device='cuda')
    with _fp32accum_probe() as counts:
        (out,), _, _ = layer.weight._reduce_scatter([wgrad], async_op=False)
    assert counts["dispatches"] == 0, f"{counts['dispatches']} fp32-accum RS at gtp_remat=2"
    # The gtp mean of the pair's values (2r+1, 2r+2) for pair r.
    expected = 2 * (rank // 2) + 1.5
    torch.testing.assert_close(out, torch.full_like(out, expected), atol=0, rtol=0)
