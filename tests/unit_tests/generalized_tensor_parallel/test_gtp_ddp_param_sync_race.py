# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""GTP ahead-of-consume weight prefetch vs. DDP's asynchronous parameter all-gather.

With ``overlap_param_gather=True`` DDP publishes a bucket group only from the forward pre-hook
of a module owning one of its parameters. GTP prefetches ahead of that module, and its gather
input is ``next_w.data`` -- a view into the bucket's ``param_data``. So when the consumed weight
and the prefetch target sit in different bucket groups, GTP reads a buffer whose all-gather is
still in flight or not yet dispatched, and whose post-gather quantize has not run: it gathers
last iteration's weights.

This is the base prefetch, not the grouped-expert optimization. Default chains look one weight
ahead, grouped-expert chains one MoE block; reading further ahead just widens the window.

Tests use the real DDP + distributed-optimizer path and none depend on timing:

- ``..._prefetch_target_bucket_is_not_synced``  -- was the target's bucket published when read?
- ``..._prefetch_reads_stale_weight_values``    -- did the read actually see stale values?
- ``..._grouped_expert_one_block_ahead_prefetch`` -- same, on the widest (grouped) window.
- ``..._recompute_forward_does_not_request_publication`` -- recompute must not ask DDP to publish.
  (The TE flag that gate depends on is pinned in ``test_gtp_basics.py``.)
"""

import contextlib

import pytest

from megatron.core.tensor_parallel.gtp_api import HAVE_GTP

if not HAVE_GTP:
    pytest.skip("GTP requires TransformerEngine >= 2.19", allow_module_level=True)

import torch

from megatron.core import parallel_state as ps
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.distributed.param_and_grad_buffer import shard_buffer
from megatron.core.tensor_parallel import generalized_tensor_parallelism as gtp_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import PARAM_READY_CALLBACK_ATTR
from tests.unit_tests.generalized_tensor_parallel.gtp_test_utils import (  # noqa: F401
    _make_gtp_linear,
    _make_gtp_remat_grouped_linear,
    _requires_multi_gpu,
    _run_distributed,
    _torchrun_dist_init,
    reset_fp8_state,
    reset_gtp_globals,
)

WORLD = 4
GTP = 2
HIDDEN = 128
SEQ = 8
BATCH = 2
DTYPE = torch.bfloat16

# Each GTP shard is (HIDDEN / GTP) x HIDDEN = 8192 elements, so any bucket size below that
# forces one bucket group per weight -- which is what puts the prefetch target outside the
# bucket the consuming module's pre-hook drains.
BUCKET_SIZE = 4096
NUM_EXPERTS = 2


def _make_config():
    return TransformerConfig(
        num_layers=1, hidden_size=HIDDEN, num_attention_heads=8, bf16=True, params_dtype=DTYPE
    )


def _build_ddp_two_gtp_layers():
    """Two GTP-sharded ``te.Linear`` layers in separate DDP bucket groups.

    Returns ``(ddp_model, [w0, w1])`` where ``w0``/``w1`` are the ``GTPShardedParam``s.
    """
    gtp_group = ps.get_gtp_weight_remat_group()
    fc0 = _make_gtp_linear(HIDDEN, HIDDEN, gtp_group, dtype=DTYPE)
    fc1 = _make_gtp_linear(HIDDEN, HIDDEN, gtp_group, dtype=DTYPE)

    module = torch.nn.Sequential()
    module.add_module("fc0", fc0)
    module.add_module("fc1", fc1)

    ddp_config = DistributedDataParallelConfig(
        use_distributed_optimizer=True,
        overlap_param_gather=True,
        overlap_grad_reduce=True,
        bucket_size=BUCKET_SIZE,
    )
    ddp_model = DistributedDataParallel(_make_config(), ddp_config, module)
    weights = [fc0.weight, fc1.weight]
    # Covers DDP's attach loop against a real DistributedDataParallel. The CPU protocol tests
    # build the callback by hand, so without this nothing would catch the registration silently
    # not running -- which would make every readiness call a no-op and the fix ineffective.
    for w in weights:
        assert hasattr(w, PARAM_READY_CALLBACK_ATTR), (
            f"DDP did not attach {PARAM_READY_CALLBACK_ATTR} to a bucketed parameter; the "
            "readiness protocol is not wired up"
        )
    return ddp_model, weights


def _forward(ddp_model, rank):
    torch.manual_seed(1234 + rank)
    x = torch.randn(SEQ * BATCH, HIDDEN, dtype=DTYPE, device="cuda")
    out = x
    for layer in ddp_model.module.children():
        out = layer(out)
    return out


def _param_names(ddp_model):
    """``id(param) -> name``; ``GTPShardedParam._debug_name`` is empty on this path."""
    return {id(p): name for name, p in ddp_model.module.named_parameters()}


def _record_prefetches(ddp_model, targets):
    """Log DDP bucket state at the exact instant GTP reads a prefetched weight's storage.

    The measurement point is ``get_padded_shard`` -- the call that produces the all-gather
    input (``generalized_tensor_parallelism.py``, "build gather inputs") -- not the entry to
    ``_all_gather_weight``. Anything the gather does to publish the shard beforehand therefore
    counts, which keeps the test measuring the read rather than a particular fix's placement.
    Scoped to asynchronous forward gathers, i.e. one-block-ahead prefetches.

    Covers the BF16 path only: with ``--fp8-param-gather`` the gather input is ``w.quantized``,
    not ``get_padded_shard()``, so this probe would record nothing there. The fix itself is
    path-independent (it runs before the input is built either way), but the native-FP8
    configuration is not regression-guarded by this test.

    Returns ``(records, restore)``.
    """
    cls = gtp_module.GTPShardedParam
    original_gather = cls._all_gather_weight
    original_shard = cls.get_padded_shard
    records = []
    watched = {id(p) for p in targets}
    names = _param_names(ddp_model)
    inside_prefetch = [False]

    def patched_gather(self, *args, **kwargs):
        async_op = kwargs.get("async_op", args[0] if args else False)
        fwd = kwargs.get("fwd", args[1] if len(args) > 1 else True)
        previous = inside_prefetch[0]
        inside_prefetch[0] = bool(async_op and fwd)
        try:
            return original_gather(self, *args, **kwargs)
        finally:
            inside_prefetch[0] = previous

    def patched_shard(self):
        if inside_prefetch[0] and id(self) in watched:
            bucket_group = ddp_model.param_to_bucket_group.get(self)
            records.append(
                {
                    "name": names.get(id(self), "<unnamed>"),
                    "dispatched": bucket_group.param_gather_dispatched,
                    "in_flight": bucket_group.param_gather_handle is not None,
                    # The all-gather input, snapshotted on the caller's stream.
                    "read_value": self.data.detach().clone(),
                }
            )
        return original_shard(self)

    cls._all_gather_weight = patched_gather
    cls.get_padded_shard = patched_shard

    def restore():
        cls._all_gather_weight = original_gather
        cls.get_padded_shard = original_shard

    return records, restore


@contextlib.contextmanager
def _gtp_env():
    """MPU at ``gtp_remat_size=GTP`` with padding off; restores global GTP state on exit."""
    ps.destroy_model_parallel()
    ps.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, gtp_remat_size=GTP
    )
    orig_pad = gtp_module.GTP_CONFIG.pad_for_alignment
    gtp_module.GTP_CONFIG.pad_for_alignment = 0
    try:
        yield
    finally:
        gtp_module.GTP_CONFIG.pad_for_alignment = orig_pad
        ps.destroy_model_parallel()
        gtp_module.GTPShardedParam._chain_state = {}
        gtp_module.get_global_GTP_cache().clear()
        ps.initialize_model_parallel()


@contextlib.contextmanager
def _patched(module, *names):
    """Restore ``module.<name>`` for each name on exit, so tests can monkeypatch freely."""
    saved = {n: getattr(module, n) for n in names}
    try:
        yield
    finally:
        for n, v in saved.items():
            setattr(module, n, v)


def _assert_published(records, rank, what):
    """Every recorded prefetch read must have hit an already-published DDP bucket."""
    assert records, f"no prefetch recorded for {what} -- the chain did not link, test is vacuous"
    unsynced = [r for r in records if (not r["dispatched"]) or r["in_flight"]]
    detail = "\n".join(
        f"  {r['name']}: dispatched={r['dispatched']} in_flight={r['in_flight']}" for r in records
    )
    assert not unsynced, (
        f"[rank {rank}] {what}: GTP prefetched {len(unsynced)}/{len(records)} weight(s) whose "
        f"DDP parameter all-gather had not completed:\n{detail}"
    )


def _worker_bucket_not_synced(rank, world_size, _port):
    with _gtp_env():
        ddp_model, weights = _build_ddp_two_gtp_layers()
        groups = [ddp_model.param_to_bucket_group[w] for w in weights]
        assert groups[0] is not groups[1], (
            "test precondition: the two GTP weights must land in different DDP bucket groups; "
            f"lower BUCKET_SIZE (currently {BUCKET_SIZE})"
        )

        # The first forward links the chain lazily; prefetch only fires from the second onwards.
        ddp_model.start_param_sync(force_sync=True)
        _forward(ddp_model, rank)
        ddp_model.reset_param_sync_dispatch_state()

        records, restore = _record_prefetches(ddp_model, weights)
        try:
            _forward(ddp_model, rank)
        finally:
            restore()
        _assert_published(records, rank, "one-weight-ahead prefetch")


def _worker_stale_values(rank, world_size, _port):
    with _gtp_env():
        ddp_model, weights = _build_ddp_two_gtp_layers()
        assert (
            ddp_model.param_to_bucket_group[weights[0]]
            is not ddp_model.param_to_bucket_group[weights[1]]
        ), "test precondition: the two GTP weights must land in different DDP bucket groups"

        ddp_model.start_param_sync(force_sync=True)
        _forward(ddp_model, rank)
        ddp_model.reset_param_sync_dispatch_state()

        # Suppress the pre-hook's "dispatch the *next* bucket group" step so the prefetch
        # target's bucket is provably un-dispatched rather than merely in flight. This is the
        # production `--overlap-param-gather-with-optimizer-step` configuration and makes the
        # stale read deterministic instead of timing-dependent.
        ddp_model.overlap_param_gather_with_optimizer_step = True

        # STALE = what the buffer holds from the previous iteration.
        # FRESH = what this rank owns and what the all-gather is about to publish.
        STALE, FRESH = -1.0, float(rank + 1)
        for bucket_group in ddp_model.bucket_groups:
            for bucket in bucket_group.buckets:
                bucket.param_data.fill_(STALE)
                local_view = shard_buffer(
                    bucket.param_data, bucket_group.intra_distributed_optimizer_instance_size
                )[bucket_group.intra_distributed_optimizer_instance_rank]
                local_view.fill_(FRESH)

        # The whole test rests on param.data being a VIEW into bucket.param_data -- that aliasing
        # is what makes the fill above visible to GTP's gather. Assert it: if a future change (or
        # the native-FP8 path, which gathers w.quantized instead) breaks the aliasing, the fill
        # would silently miss and every read_value would look fresh, turning this into a green
        # test that no longer guards anything.
        for w in weights:
            assert w.data.eq(STALE).any() or w.data.eq(FRESH).all(), (
                f"[rank {rank}] {type(w).__name__}.data does not alias bucket.param_data, so the "
                "stale-value fill never reached it -- this test would pass vacuously"
            )

        records, restore = _record_prefetches(ddp_model, weights)
        try:
            _forward(ddp_model, rank)
        finally:
            restore()
            ddp_model.overlap_param_gather_with_optimizer_step = False

        assert records, "no GTP prefetch was issued -- the chain did not link, test is vacuous"

        stale = []
        for r in records:
            n_stale = int((r["read_value"] == STALE).sum().item())
            if n_stale:
                stale.append((r["name"], n_stale, r["read_value"].numel()))
        detail = "\n".join(
            f"  {name}: {n}/{total} elements still stale" for name, n, total in stale
        )
        assert not stale, (
            f"[rank {rank}] GTP all-gathered weight storage that DDP had not yet refreshed, so "
            f"the gathered weight carries the previous iteration's values:\n{detail}"
        )


def _worker_recompute_skips_readiness(rank, world_size, _port):
    """Activation-recompute must NOT ask DDP to publish parameters.

    Recompute-forward is a forward replayed inside backward, so the shards were already
    published by the real forward. Asking again is not merely wasted work: DDP has moved on to
    gradient reduction by then, and publishing could dispatch a parameter all-gather mid-backward
    into a buffer that aliases the gradient buffer under --reuse-grad-buf-for-mxfp8-param-ag.
    """
    patched = _patched(gtp_module, "ensure_params_ready", "in_activation_recompute_phase")
    with _gtp_env(), patched:
        ddp_model, weights = _build_ddp_two_gtp_layers()
        ddp_model.start_param_sync(force_sync=True)

        calls = []
        gtp_module.ensure_params_ready = lambda params: calls.append(tuple(params))

        # Baseline: a genuine forward gather must consult readiness, otherwise this test could
        # pass simply because the call site was deleted.
        gtp_module.in_activation_recompute_phase = lambda: False
        weights[0]._all_gather_weight(async_op=False, fwd=True)
        assert calls, (
            f"[rank {rank}] a real forward gather did not consult param readiness -- the call "
            "site is gone, so the recompute assertion below would be vacuous"
        )

        # The actual guarantee: inside recompute, readiness must not be consulted at all.
        calls.clear()
        gtp_module.in_activation_recompute_phase = lambda: True
        weights[0]._all_gather_weight(async_op=False, fwd=True)
        assert not calls, (
            f"[rank {rank}] activation-recompute asked DDP to publish parameters "
            f"({len(calls)} call(s)); recompute runs inside backward, where the shards are "
            "already published and a dispatch could gather into the gradient buffer"
        )


class _Experts(torch.nn.Module):
    """``.mlp.experts.linear_fc1/2`` -- the names ``_classify_param_chain`` keys on."""

    def __init__(self, gtp_group):
        super().__init__()
        self.linear_fc1 = _make_gtp_remat_grouped_linear(
            NUM_EXPERTS, HIDDEN, HIDDEN, gtp_group, dtype=DTYPE
        )
        self.linear_fc2 = _make_gtp_remat_grouped_linear(
            NUM_EXPERTS, HIDDEN, HIDDEN, gtp_group, dtype=DTYPE
        )


class _MoEBlock(torch.nn.Module):
    def __init__(self, gtp_group):
        super().__init__()
        self.mlp = torch.nn.Module()
        self.mlp.experts = _Experts(gtp_group)


def _worker_grouped_one_block_ahead(rank, world_size, _port):
    """Grouped-expert chains prefetch one MoE BLOCK ahead -- the widest window.

    ``next_w`` links the same weight role across consecutive blocks, so consuming block 0's fc1
    gathers block 1's fc1 while an entire block of compute still separates it from that module's
    DDP pre-hook. Also the only route where ``ensure_params_ready`` receives several parameters
    at once (one per expert), exercising the dedup path through the real GTP call.
    """
    with _gtp_env():
        gtp_group = ps.get_gtp_weight_remat_group()
        model = torch.nn.Module()
        model.layers = torch.nn.ModuleList([_MoEBlock(gtp_group) for _ in range(2)])
        model = model.cuda()

        # Stamp chain_id from the parameter NAMES; without this the grouped chains are never
        # selected and this would silently retest the default one-weight-ahead chain.
        gtp_module.classify_gtp_chains(model)
        anchors = [b.mlp.experts.linear_fc1.weight0 for b in model.layers]
        chain_ids = {a.chain_id for a in anchors}
        assert all("grouped" in c for c in chain_ids), (
            f"expected grouped-expert chains, got {chain_ids} -- the one-block-ahead path is "
            "not under test"
        )

        ddp_config = DistributedDataParallelConfig(
            use_distributed_optimizer=True,
            overlap_param_gather=True,
            overlap_grad_reduce=True,
            bucket_size=BUCKET_SIZE,
        )
        ddp_model = DistributedDataParallel(_make_config(), ddp_config, model)
        assert (
            ddp_model.param_to_bucket_group[anchors[0]]
            is not ddp_model.param_to_bucket_group[anchors[1]]
        ), "test precondition: the two blocks' fc1 weights must land in different bucket groups"

        # First pass links block0.fc1 -> block1.fc1 in the grouped chain; prefetch fires after.
        ddp_model.start_param_sync(force_sync=True)
        for a in anchors:
            a.all_gather_and_prefetch(fwd=True)
        ddp_model.reset_param_sync_dispatch_state()

        targets = anchors[1]._weights  # every expert weight of block 1's fc1
        records, restore = _record_prefetches(ddp_model, targets)
        try:
            ddp_model.start_param_sync(force_sync=True)
            anchors[0].all_gather_and_prefetch(fwd=True)
        finally:
            restore()

        _assert_published(records, rank, "one-block-ahead grouped-expert prefetch")


class TestGTPDDPParamSyncRace:
    """GTP prefetch must not read a parameter before DDP has published it."""

    def test_prefetch_target_bucket_is_not_synced(self):
        """Structural: every prefetch target's DDP bucket group must already be finished."""
        _requires_multi_gpu(WORLD)
        _run_distributed(_worker_bucket_not_synced, WORLD)

    def test_prefetch_reads_stale_weight_values(self):
        """Numerical: the storage GTP feeds to NCCL must not hold last iteration's values."""
        _requires_multi_gpu(WORLD)
        _run_distributed(_worker_stale_values, WORLD)

    def test_recompute_forward_does_not_request_publication(self):
        """Scope: only a real forward publishes; recompute runs inside backward."""
        _requires_multi_gpu(WORLD)
        _run_distributed(_worker_recompute_skips_readiness, WORLD)

    def test_grouped_expert_one_block_ahead_prefetch(self):
        """The widest prefetch window: grouped-expert chains reach a whole MoE block ahead."""
        _requires_multi_gpu(WORLD)
        _run_distributed(_worker_grouped_one_block_ahead, WORLD)
