# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Layer-sharded Muon: layer sharding for Newton-Schulz over the GTP x TP domain.

Instead of per-weight all-gather + redundant full-matrix NS on every rank, each weight
is assigned one NS home rank in the (GTP x TP) domain. Two all_to_all stages route the
momentum shards so the home holds the complete (P, Q) matrix, Newton-Schulz runs there
with zero communication and zero redundancy — the exact same full-matrix NS as
duplicated mode — and two reverse all_to_all stages scatter the result back to the
original shards. All collectives use the existing gtp / tp process groups.
"""

import contextlib
import logging
from typing import Any, Callable, Literal, Optional

import torch
from torch.optim.optimizer import ParamsT

from megatron.core.optimizer.emerging_optimizers import TensorParallelMuon
from megatron.core.optimizer.layer_sharded_a2a import (
    layer_sharded_all_to_all_bwd,
    layer_sharded_all_to_all_fwd,
    layer_sharded_fused_bwd,
    layer_sharded_fused_fwd,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.utils import is_emerging_optimizers_min_version, log_single_rank

try:
    from emerging_optimizers import triton_kernels
    from emerging_optimizers.orthogonalized_optimizers.muon import MuonScaleT, get_muon_scale_factor
    from emerging_optimizers.orthogonalized_optimizers.muon_utils import NSCoeffT, newton_schulz
    from emerging_optimizers.utils import FP32MatmulPrecT, fp32_matmul_precision

    HAVE_EMERGING_OPTIMIZERS = True
except ImportError:
    HAVE_EMERGING_OPTIMIZERS = False

__all__ = ["LayerShardedMuon"]

logger = logging.getLogger(__name__)


def _resolve_use_syrk(use_syrk: bool) -> bool:
    """Validate SYRK hardware availability, downgrading to False (with an error
    log) if unmet.

    This guard lives in emerging-optimizers' ``Muon.__init__`` — which this
    class no longer inherits from — and NOT in ``TensorParallelMuon``, whose
    constructor only version-gates ``use_syrk``. Without it, Triton < 3.4.0
    asserts on the first optimizer step instead of downgrading at construction,
    and unvalidated SM architectures silently run a kernel emerging-optimizers
    does not vouch for. Resolved *before* the parent constructor so the parent's
    closure captures the resolved value (fallback and layer-sharded paths agree)
    and so unfit hardware downgrades before the parent's EO-version raise.
    """
    if not use_syrk:
        return False
    if torch.cuda.is_available():
        sm_version = torch.cuda.get_device_capability()
    else:
        sm_version = (0, 0)
    if not triton_kernels.HAS_TRITON_340:  # type: ignore[attr-defined]
        logger.error("Triton 3.4.0 or higher is required for use_syrk to be True.")
        return False
    if sm_version not in ((8, 0), (9, 0), (10, 0), (10, 3)):
        logger.error(
            f"Correctness of Triton kernel on SM {sm_version} cannot be guaranteed. "
            "Setting use_syrk to False."
        )
        return False
    return True


def _has_batched_syrk() -> bool:
    """Whether the installed emerging-optimizers has the batched (3-D) SYRK kernel.

    ``batched_tsyrk_ex`` landed on emerging-optimizers main with PR #276 (>= 0.5.0a0);
    on such installs ``newton_schulz`` dispatches 3-D inputs to the batched SYRK step,
    so batched chunks no longer need to fall back to baddbmm. Detected by symbol
    rather than version so pre-release mains qualify.
    """
    return hasattr(triton_kernels, 'batched_tsyrk_ex')


# Phase-level NVTX ranges. Kernel-name classification cannot separate the forward
# from the reverse all_to_all, nor the momentum update from the weight update, so
# the step is annotated explicitly. Phase granularity (a handful of pushes per
# step, not per param) keeps the cost negligible.
_NVTX_ENABLED = torch.cuda.is_available()


@contextlib.contextmanager
def _phase(name: str):
    if not _NVTX_ENABLED:
        yield
        return
    torch.cuda.nvtx.range_push(f"lsmuon/{name}")
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


class LayerShardedMuon(TensorParallelMuon):
    """Muon with layer sharding over the GTP x TP domain.

    Sharding model per 2D weight of full shape ``(P, Q)``:

    - TP shards along ``param.partition_dim`` (0 = column-parallel, 1 = row-parallel,
      None / -1 = not TP-sharded).
    - GTP shards dim 0 of the TP-local shard, for params tagged
      ``param.is_gtp_weight_remat`` (Megatron's marker; absent means unsharded).
    - A param sharded by neither is whole on every rank of the domain (e.g. the MoE
      router and latent projections): it skips both exchanges and every rank runs the
      same deterministic NS on its own copy.
    - GTP alignment padding (``param.pad_length`` trailing zero rows on the
      gtp-gathered, TP-local dim 0) is stripped before Newton-Schulz — so the scale
      factor sees the true dims, matching the parent's duplicated path bitwise — and
      restored before the reverse gtp exchange. On the fused exchange this is
      supported for single-axis domains only (tp_size == 1); a 2-D fused domain with
      padded params raises NotImplementedError (use the two-stage path there).

    ``step()`` runs, per param group:

    1. Momentum update on the local shard (elementwise, identical to base Muon).
    2. Stage-1 all_to_all over ``gtp_group`` (dim 0): each param's GTP extent is
       assembled on its assigned ``g_home`` column.
    3. Stage-2 all_to_all over ``tp_group`` (along ``partition_dim``): the full
       matrix is assembled on the ``(g_home, t_home)`` NS home. Params that are not
       TP-sharded skip this stage — every TP peer of the column already holds the
       full matrix and runs the same (deterministic) NS so each column can scatter
       its own updates.
    4. Full-matrix Newton-Schulz on the home — bit-identical to duplicated mode.
    5. Reverse stage-2 / stage-1 all_to_all scatter the scaled NS result back to
       every rank's original shard, which applies ``p -= lr * update``.

    Args:
        params: Parameters to optimize. Every rank in the domain must pass the same
            params in the same order (they hold different shards of the same weights).
        gtp_group: GTP weight-shard process group (dim-0 sharding of the TP-local shard).
        tp_group: TP process group, or None when TP is not used.
        fused_group: Optional flattened process group over the whole (GTP x TP)
            domain, sized ``gtp_size * tp_size`` with group rank ``g * tp_size + t``
            (TP innermost). When provided, one all_to_all per direction replaces the
            two-stage GTP-then-TP exchange — same blocks, same assembly order, so the
            NS input is bit-identical. None (default) keeps the two-stage path.
        use_syrk: Use the Triton SYRK kernel for the two symmetric-output NS GEMMs
            (``A = X Xᵀ`` and ``B = bA + cA²``), computing one triangle only —
            roughly a third off total NS FLOPs for near-square matrices. Applies to
            unbatched chunks always; batched (3-D) chunks additionally require an
            emerging-optimizers with the batched SYRK kernel (>= 0.5.0a0, PR #276)
            and otherwise fall back to baddbmm. Only takes effect with
            ``fp32_matmul_prec="medium"`` and 8-aligned dims; auto-disabled when
            Triton/SM requirements are unmet. Same math, different kernel — results
            differ from the GEMM path by kernel-level rounding.
        ns_batch_size: Maximum number of same-shape matrices fused into a single
            batched Newton-Schulz on a home. MoE homes own hundreds of identically
            shaped expert weights, where the per-matrix loop is kernel-launch bound;
            batching trades a transient ``ns_batch_size x matrix`` stack for far
            fewer launches. Defaults to 1 (no batching): a batch of more than one
            runs through ``baddbmm`` rather than ``addmm``, so results differ from
            the unbatched path by kernel-level floating point rounding and bitwise
            parity with duplicated mode is lost. Raise (e.g. to 32) to trade that
            parity for launch-bound MoE-home throughput.
        concurrent_groups: Run each param group's pipeline on its own CUDA stream
            instead of serializing them. Groups own disjoint params and, under MoE,
            disjoint process groups, so nothing orders them against each other; on a
            single stream one group's all_to_all stall blocks the other group's
            Newton-Schulz even though the GPU is idle. The ops and their order within
            a group are unchanged, so results are unaffected, but the transient
            buffers of all groups are live at once -- lower ``ns_batch_size`` or set
            this to False if that pushes peak memory too high. No effect with fewer
            than two param groups or without CUDA.
        All other args: same as :class:`TensorParallelMuon`. In particular
            ``split_qkv`` / ``is_qkv_fn`` / ``qkv_split_shapes``, ``tp_mode`` and
            ``pg_collection`` only take effect on the paths that delegate to the
            parent (the empty-``param_ns_homes`` fallback and the degenerate
            single-rank domain, both of which run the parent's TP-aware
            full-matrix Newton-Schulz).

    Note:
        ``None`` for either process group means "no group / size 1", **not** torch's
        "the default group" — a missing expert group must not silently become the
        whole world.

    Usage::

        optimizer = LayerShardedMuon(params, lr=3e-4, gtp_group=gtp, tp_group=tp)
        optimizer.set_param_ns_homes({id(p): (g_home, t_home) for ...})
        optimizer.step()
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 3e-4,
        momentum: float = 0.95,
        weight_decay: float = 0.01,
        *,
        nesterov: bool = False,
        fp32_matmul_prec: FP32MatmulPrecT = "medium",
        coefficient_type: NSCoeffT = "quintic",
        num_ns_steps: int = 5,
        scale_mode: MuonScaleT = "spectral",
        extra_scale_factor: float = 1.0,
        gtp_group: "torch.distributed.ProcessGroup | None",
        tp_group: "torch.distributed.ProcessGroup | None" = None,
        fused_group: "torch.distributed.ProcessGroup | None" = None,
        ns_batch_size: int = 1,
        use_syrk: bool = False,
        concurrent_groups: bool = True,
        use_decoupled_weight_decay: bool = True,
        split_qkv: bool = False,
        is_qkv_fn: Callable[[torch.Tensor], bool] | None = None,
        qkv_split_shapes: list[int] | None = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        tp_mode: Literal["blockwise", "duplicated", "distributed"] = "duplicated",
    ) -> None:
        if split_qkv:
            # The layer-sharded exchange routes whole matrices to their NS homes
            # and never goes through TensorParallelMuon.orthogonalize, where
            # split-QKV is implemented — only the fallback and degenerate paths
            # would split. Accepting split_qkv=True would make the update rule
            # depend on whether param_ns_homes happens to be set; reject it at
            # the class level rather than relying on validate_args alone.
            raise ValueError(
                "LayerShardedMuon does not implement split-QKV Newton-Schulz on the "
                "layer-sharded path; pass split_qkv=False (--muon-no-split-qkv)."
            )
        # Hardware validation first: the parent only version-gates use_syrk, and
        # unfit hardware should downgrade rather than hit the EO-version raise.
        use_syrk = _resolve_use_syrk(use_syrk)
        if ns_batch_size > 1 and not is_emerging_optimizers_min_version("0.3.0"):
            # Only the batched (3-D) Newton-Schulz path needs emerging-optimizers
            # >= 0.3.0 (older releases fail inside torch.addmm with "mat1 must be a
            # matrix, got 3-D tensor"). The per-matrix baseline (ns_batch_size=1,
            # the default) uses the 2-D newton_schulz API that 0.2.0 already ships,
            # so it must not raise on older installs.
            raise ImportError(
                'LayerShardedMuon with ns_batch_size > 1 requires emerging-optimizers '
                '>= 0.3.0 (batched Newton-Schulz).'
            )
        # The parent validates num_ns_steps and hard-raises when use_syrk is set on
        # an emerging-optimizers older than the newton_schulz_tp use_syrk forwarding
        # (>= 0.4.0.dev0). That gate is inherited deliberately: after this refactor
        # the fallback and degenerate paths DO go through newton_schulz_tp, so on
        # older installs those paths genuinely cannot do SYRK and failing loudly
        # beats a partial silent downgrade.
        # Explicit class call, matching the convention used by
        # TensorParallelAdaptiveMuon (see the comment in TensorParallelMuon.__init__).
        TensorParallelMuon.__init__(
            self,
            params,
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
            use_decoupled_weight_decay=use_decoupled_weight_decay,
            split_qkv=split_qkv,
            is_qkv_fn=is_qkv_fn,
            qkv_split_shapes=qkv_split_shapes,
            fp32_matmul_prec=fp32_matmul_prec,
            coefficient_type=coefficient_type,
            num_ns_steps=num_ns_steps,
            scale_mode=scale_mode,
            extra_scale_factor=extra_scale_factor,
            pg_collection=pg_collection,
            tp_mode=tp_mode,
            use_syrk=use_syrk,
        )
        self.gtp_group = gtp_group
        self.tp_group = tp_group
        self.fused_group = fused_group
        self.ns_batch_size = max(1, ns_batch_size)
        # TensorParallelMuon does not set these on self -- it only captures them in
        # its scaled_orthogonalize_fn closure. _run_ns reads them off self, so assign
        # them as plain attributes here (safe: the parent defines no properties).
        self.coefficient_type = coefficient_type
        self.num_ns_steps = num_ns_steps
        self.scale_mode = scale_mode
        self.extra_scale_factor = extra_scale_factor
        self.use_syrk = use_syrk
        # Batched (3-D) chunks can use SYRK only when the installed emerging-optimizers
        # ships the batched kernel; otherwise they fall back to baddbmm as before.
        self._batched_syrk = self.use_syrk and _has_batched_syrk()
        if self.use_syrk and self.ns_batch_size > 1 and not self._batched_syrk:
            log_single_rank(
                logger,
                logging.WARNING,
                'use_syrk is set but this emerging-optimizers has no batched SYRK '
                'kernel (needs >= 0.5.0a0, PR #276): batched chunks fall back to '
                'baddbmm; only unbatched chunks get SYRK.',
            )
        self.concurrent_groups = concurrent_groups
        self._group_streams: "list | None" = None
        # id(param) -> (g_home, t_home). Set via set_param_ns_homes().
        self._param_ns_homes: dict[int, tuple[int, int]] = {}
        # param_group index -> (gtp_group, tp_group), overriding the constructor
        # defaults. Set via set_group_process_groups().
        self._group_process_groups: dict[int, tuple] = {}

    def set_param_ns_homes(self, param_ns_homes: dict[int, tuple[int, int]]) -> None:
        """Set the NS home for each param (by id).

        Args:
            param_ns_homes: Maps ``id(param)`` -> ``(g_home, t_home)``: the rank in
                ``gtp_group`` and in ``tp_group`` that runs NS for it. ``t_home`` is
                ignored for params that are not TP-sharded and when ``tp_group`` is None.
        """
        self._param_ns_homes = param_ns_homes
        self._warned_missing_homes = False

    def set_group_process_groups(self, group_process_groups: dict[int, tuple]) -> None:
        """Override the (GTP, TP) process groups per param group.

        Different param groups can be sharded over different domains — e.g. under
        MoE, expert weights are sharded over the *expert* GTP/TP groups while dense
        weights use the dense ones. Groups absent from the mapping fall back to the
        ``gtp_group`` / ``tp_group`` passed to the constructor.

        Args:
            group_process_groups: Maps the index of a ``self.param_groups`` entry to
                ``(gtp_group, tp_group)`` or ``(gtp_group, tp_group, fused_group)``.
                Any entry may be None (treated as size 1 / not available).
                A 2-tuple selects the two-stage GTP-then-TP path for that group and
                does **not** inherit the constructor's ``fused_group`` — pass an
                explicit 3-tuple to enable the fused path.  Using the constructor's
                fused group as an implicit default would be incorrect when a group
                uses a different (e.g. expert) domain whose flat communicator differs.
        """
        self._group_process_groups = group_process_groups

    def _apply_update(self, p: torch.Tensor, update: torch.Tensor, lr: float) -> None:
        """Apply one weight update through the base-class hook points.

        ``OrthogonalizedOptimizer.step()`` brackets every ``p.add_`` with
        ``pre_weight_update_fn_inplace`` / ``post_weight_update_fn_inplace``;
        this helper keeps layer sharding's overridden ``step()`` honouring them
        too, and keeps the four update sites (replicated, fused, two-stage,
        degenerate domain) from diverging. No dtype cast on purpose: the base
        class's ``p.add_(orth_grad, alpha=-lr)`` — the fifth path, taken by the
        empty-homes fallback — computes the fused multiply-add in the promoted
        precision and downcasts once on store, so casting here first would give
        bf16 params different rounding on the layer-sharded paths than on the
        fallback and than TensorParallelMuon's duplicated mode. TODO: forward
        the ``weight_update_hook`` constructor parameter once the
        emerging-optimizers pin moves past EO #224.
        """
        self.pre_weight_update_fn_inplace(p, update)
        p.add_(update, alpha=-lr)
        self.post_weight_update_fn_inplace(p)

    def _run_ns(self, full_by_k: dict) -> dict:
        """Full-matrix Newton-Schulz per home-owned matrix, batched by shape.

        Same-shape matrices are stacked into one batched NS: under MoE a home owns
        hundreds of identically shaped expert weights, and the per-matrix loop is
        dominated by kernel-launch overhead. Batches are capped at ``ns_batch_size``
        to bound the transient stack memory. A batch of one stays 2-D, so the
        unbatched numerics are preserved exactly whenever nothing is actually batched.
        """
        ns_by_k: dict = {}
        by_shape: dict = {}
        for k, full in full_by_k.items():
            by_shape.setdefault(tuple(full.shape), []).append(k)

        for ks in by_shape.values():
            for start in range(0, len(ks), self.ns_batch_size):
                chunk = ks[start : start + self.ns_batch_size]
                batched = len(chunk) > 1
                x = torch.stack([full_by_k[k] for k in chunk]) if batched else full_by_k[chunk[0]]
                # SYRK halves the two symmetric-output NS GEMMs. Unbatched (2-D)
                # chunks — the big dense matrices — always qualify; batched (3-D)
                # chunks additionally need the batched SYRK kernel (emerging-
                # optimizers >= 0.5.0a0, PR #276), else they fall back to baddbmm.
                orth = newton_schulz(
                    x,
                    steps=self.num_ns_steps,
                    coefficient_type=self.coefficient_type,
                    use_syrk=self._batched_syrk if batched else self.use_syrk,
                )
                scale = get_muon_scale_factor(orth.size(-2), orth.size(-1), mode=self.scale_mode)
                # Two sequential multiplies, NOT a pre-combined scalar: matches
                # TensorParallelMuon's `orth * scale * extra` rounding exactly, so
                # duplicated-mode parity holds bitwise (elementwise ops commute with
                # unbind, so batching does not perturb this).
                orth = orth * scale * self.extra_scale_factor
                if batched:
                    for k, o in zip(chunk, orth.unbind(0)):
                        ns_by_k[k] = o
                else:
                    ns_by_k[chunk[0]] = orth
        return ns_by_k

    def _param_group_streams(self) -> "list | None":
        """Per-group CUDA streams, or None when the groups must stay serialized."""
        if not self.concurrent_groups or len(self.param_groups) < 2:
            return None
        if not torch.cuda.is_available():
            return None
        if self._group_streams is None or len(self._group_streams) != len(self.param_groups):
            self._group_streams = [torch.cuda.Stream() for _ in self.param_groups]
        return self._group_streams

    @torch.no_grad()
    def step(self, closure: Any = None) -> None:
        """Run one optimizer step: momentum update, shard exchange to the NS homes,
        full-matrix Newton-Schulz there, reverse exchange, weight update (see the
        class docstring for the per-stage breakdown)."""
        if closure is not None:
            raise ValueError("closure is not supported")

        # Fall back to TensorParallelMuon when no assignment is set: all-gather +
        # TP-aware full-matrix Newton-Schulz per param. Mathematically correct
        # (unlike the pre-refactor base-Muon fallback, which silently degraded to
        # local-shard NS), just redundant — every rank recomputes every matrix.
        if not self._param_ns_homes:
            if not getattr(self, '_warned_no_homes', False):
                self._warned_no_homes = True
                log_single_rank(
                    logger,
                    logging.WARNING,
                    "LayerShardedMuon: param_ns_homes is empty — falling back to "
                    "TensorParallelMuon (per-param all-gather + full-matrix "
                    "Newton-Schulz on every rank; correct but redundant). Call "
                    "set_param_ns_homes() before step() to enable layer sharding.",
                )
            return super().step(closure)

        streams = self._param_group_streams()
        if streams is None:
            return self._step_groups(None, None)

        # Groups touch disjoint params and, under MoE, disjoint process groups, so
        # nothing orders them against each other. Left on one stream the dense group's
        # reverse all_to_all -- which is mostly the GPU idling inside the NCCL kernel --
        # blocks the expert group's Newton-Schulz, which is pure compute. Giving each
        # group its own stream lets one fill the other's stall.
        default_stream = torch.cuda.current_stream()
        ready = torch.cuda.Event()
        ready.record(default_stream)
        try:
            self._step_groups(streams, ready)
        finally:
            torch.cuda.set_stream(default_stream)
            for s in streams:
                default_stream.wait_stream(s)

    def _step_groups(self, streams: "list | None", ready: "torch.cuda.Event | None") -> None:
        for group_index, group in enumerate(self.param_groups):
            if streams is not None:
                # Wait for the backward that produced these grads, then run this
                # group's whole pipeline on its own stream.
                streams[group_index].wait_event(ready)
                torch.cuda.set_stream(streams[group_index])
            # Each param group may live in its own domain (dense vs expert).
            pgs = self._group_process_groups.get(
                group_index, (self.gtp_group, self.tp_group, self.fused_group)
            )
            gtp_group, tp_group = pgs[0], pgs[1]
            fused_group = pgs[2] if len(pgs) > 2 else None
            gtp_size = torch.distributed.get_world_size(gtp_group) if gtp_group is not None else 1
            tp_size = torch.distributed.get_world_size(tp_group) if tp_group is not None else 1

            self._init_group(group)
            lr = group["lr"]
            beta = group["momentum"]

            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue

            # 1. Momentum update on the local shard.
            # NOTE: with nesterov=False, ``moms[i]`` aliases the momentum buffer
            # (``.float()`` is a no-op on fp32) — everything below treats it as read-only.
            moms: list[torch.Tensor] = []
            with _phase("momentum"):
                for p in params:
                    grad = p.grad
                    state = self.state[p]
                    self._apply_weight_decay_inplace(p, grad, lr, group["weight_decay"])
                    state["momentum_buffer"].lerp_(grad, 1 - beta)
                    if self.nesterov:
                        m = grad.lerp(state["momentum_buffer"], beta)
                    else:
                        m = state["momentum_buffer"]
                    moms.append(m.float())

            # Degenerate domain: this rank's shard is the whole matrix it owns, so
            # run plain local NS exactly as base Muon would.
            if gtp_size * tp_size <= 1:
                group_kwargs = {k: v for k, v in group.items() if k != "params"}
                with fp32_matmul_precision(self.fp32_matmul_prec):
                    for p, m in zip(params, moms):
                        self._apply_update(p, self.orthogonalize(p, m, **group_kwargs), lr)
                continue

            gtp_rank = torch.distributed.get_rank(gtp_group) if gtp_size > 1 else 0
            tp_rank = torch.distributed.get_rank(tp_group) if tp_size > 1 else 0

            def _partition_dim(p: torch.Tensor, _tp_size: int = tp_size) -> "int | None":
                if _tp_size <= 1:
                    return None
                pd = getattr(p, "partition_dim", None)
                return None if pd is None or pd == -1 else pd

            def _gtp_sharded(p: torch.Tensor, _gtp_size: int = gtp_size) -> bool:
                return _gtp_size > 1 and bool(getattr(p, "is_gtp_weight_remat", False))

            # Params sharded by neither GTP nor TP already hold the whole matrix on
            # every rank of the domain (TE leaves the MoE router and the latent
            # projections unsharded). They join neither exchange: every rank runs the
            # same deterministic NS on its own copy, which is both correct and cheaper
            # than electing a home and broadcasting the result back.
            replicated, routed = [], []
            for i, p in enumerate(params):
                # A TP-sharded param that is not GTP-sharded is REPLICATED across the
                # GTP group; stage-1 would concatenate the G identical copies as if
                # they were dim-0 shards and hand Newton-Schulz a (G*rows, cols)
                # matrix. The reverse-path shape asserts come out numerically
                # consistent, so this corrupts silently — reject it loudly instead.
                if gtp_size > 1 and _partition_dim(p) is not None and not _gtp_sharded(p):
                    raise ValueError(
                        f"LayerShardedMuon: param of shape {tuple(p.shape)} is TP-sharded "
                        f"(partition_dim={getattr(p, 'partition_dim', None)}) but not GTP-sharded "
                        f"(is_gtp_weight_remat absent/False) while gtp_size={gtp_size} > 1. "
                        "The GTP exchange would concatenate replicated copies as shards and "
                        "silently corrupt the update. Tag the param with is_gtp_weight_remat "
                        "or run it in a domain without a GTP axis."
                    )
                target = (
                    routed if (_gtp_sharded(p) or _partition_dim(p) is not None) else replicated
                )
                target.append(i)

            if replicated:
                # Batched like the routed path: these are few but identically shaped
                # per layer (one router and two latent projections each), and the
                # redundant NS is launch-bound, not compute-bound -- measured at 336
                # kernels but only 0.74 ms of GPU time per step on a 12-layer MoE.
                with _phase("ns_replicated"), fp32_matmul_precision(self.fp32_matmul_prec):
                    for i, upd in self._run_ns({i: moms[i] for i in replicated}).items():
                        self._apply_update(params[i], upd, lr)
                if not routed:
                    continue
                params = [params[i] for i in routed]
                moms = [moms[i] for i in routed]

            n_missing = sum(1 for p in params if id(p) not in self._param_ns_homes)
            if n_missing and not getattr(self, '_warned_missing_homes', False):
                # Any home is mathematically valid (assignment only affects load
                # balance), but a miss usually means homes were wired against stale
                # param objects (e.g. before an fp32 main-param swap) — surface it.
                self._warned_missing_homes = True
                log_single_rank(
                    logger,
                    logging.WARNING,
                    f"LayerShardedMuon: {n_missing}/{len(params)} params missing from "
                    "param_ns_homes; falling back to round-robin (g=i%G, t=0). Load "
                    "balancing (LPT) is NOT in effect for these params.",
                )
            homes = [
                self._param_ns_homes.get(id(p), (routed[i] % gtp_size, 0))
                for i, p in enumerate(params)
            ]
            g_home = {i: homes[i][0] for i in range(len(params))}

            # GTP alignment padding: ``p.pad_length`` trailing zero rows on the
            # gtp-gathered, TP-LOCAL dim 0 — the same attribute the parent strips
            # in its duplicated path. Stripped before Newton-Schulz so the scale
            # factor sees the true dims (and parity with duplicated mode holds),
            # restored before the reverse gtp exchange, whose split sizes derive
            # from the padded momentum shards.
            pads = [getattr(p, 'pad_length', 0) for p in params]

            # --- Fused path: one all_to_all over the flattened (GTP x TP) domain
            # replaces the two stages in each direction. Moves the exact same shard
            # blocks and assembles them in the exact same order, so the NS input is
            # bit-identical to the two-stage path.
            if fused_group is not None:
                # The fused exchange assumes flat rank g * tp_size + t (TP innermost).
                # A group built with any other rank order scatters blocks to the wrong
                # coordinates — silently, since all split sizes still line up.
                fused_rank = torch.distributed.get_rank(fused_group)
                assert fused_rank == gtp_rank * tp_size + tp_rank, (
                    f"LayerShardedMuon: fused_group rank {fused_rank} != "
                    f"gtp_rank({gtp_rank}) * tp_size({tp_size}) + tp_rank({tp_rank}); the "
                    "flattened (GTP x TP) group must be built with TP innermost."
                )
                pdims = [_partition_dim(p) for p in params]
                if tp_size > 1 and any(pads):
                    # After 2-D assembly the pad is embedded at the tail of every
                    # TP block, not the matrix tail, so a single strip is wrong.
                    raise NotImplementedError(
                        "LayerShardedMuon: GTP alignment padding on the fused (2-D) "
                        "exchange is not implemented. Use the two-stage path "
                        "(fused_group=None) for padded params when tp_size > 1."
                    )
                with fp32_matmul_precision(self.fp32_matmul_prec):
                    with _phase("a2a_fwd"):
                        fulls, my_idx = layer_sharded_fused_fwd(
                            moms, homes, pdims, gtp_rank, tp_rank, gtp_size, tp_size, fused_group
                        )
                        # tp_size == 1 here whenever pads are present: the
                        # home-assembled matrix has a contiguous dim-0 pad tail.
                        fulls = [
                            self._strip_pad(t, pads[my_idx[k]]) for k, t in enumerate(fulls)
                        ]
                    with _phase("ns"):
                        ns_by_k = self._run_ns(dict(enumerate(fulls)))
                with _phase("a2a_bwd"):
                    update_shards = layer_sharded_fused_bwd(
                        [
                            self._restore_pad(ns_by_k[k], pads[my_idx[k]])
                            for k in range(len(fulls))
                        ],
                        my_idx,
                        moms,
                        homes,
                        pdims,
                        gtp_rank,
                        tp_rank,
                        gtp_size,
                        tp_size,
                        fused_group,
                    )
                with _phase("update"):
                    for p, shard in zip(params, update_shards):
                        if shard is not None:
                            self._apply_update(p, shard, lr)
                continue

            with fp32_matmul_precision(self.fp32_matmul_prec):
                with _phase("a2a_fwd"):
                    # 2. Stage-1 all_to_all over GTP (dim 0).
                    stage1, my_g = layer_sharded_all_to_all_fwd(
                        moms, g_home, gtp_rank, gtp_size, gtp_group, 0
                    )
                    # Strip the GTP alignment padding at the stage-1 seam: the
                    # stage-1 output is the gtp-gathered TP-LOCAL tensor, where
                    # the pad is a contiguous dim-0 tail for every partition_dim
                    # (after TP assembly it would be embedded per TP block) —
                    # the same strip point the parent's duplicated path uses.
                    stage1 = [self._strip_pad(t, pads[my_g[k]]) for k, t in enumerate(stage1)]

                    # Split this column's params by TP partition dim. Keys 0/1 go
                    # through stage 2; None params are already complete on every TP peer.
                    sub_pos: dict = {0: [], 1: [], None: []}
                    for k, i in enumerate(my_g):
                        sub_pos[_partition_dim(params[i])].append(k)

                    # 3. Stage-2 all_to_all over TP per partition dim.
                    full_by_k: dict[int, torch.Tensor] = {}
                    stage2_ctx: dict[int, tuple] = {}
                    for pd in (0, 1):
                        pos = sub_pos[pd]
                        if not pos:
                            continue
                        templates = [stage1[k] for k in pos]
                        t_home = {n: homes[my_g[pos[n]]][1] for n in range(len(pos))}
                        fulls, my_sel = layer_sharded_all_to_all_fwd(
                            templates, t_home, tp_rank, tp_size, tp_group, pd
                        )
                        stage2_ctx[pd] = (pos, templates, t_home, my_sel)
                        for n_sel, full in zip(my_sel, fulls):
                            full_by_k[pos[n_sel]] = full
                    for k in sub_pos[None]:
                        full_by_k[k] = stage1[k]

                # 4. Full-matrix Newton-Schulz on the home (identical to duplicated
                #    mode), batched by shape — see _run_ns.
                with _phase("ns"):
                    ns_by_k = self._run_ns(full_by_k)

            with _phase("a2a_bwd"):
                # 5. Reverse stage-2 all_to_all: scatter NS results back to TP parts.
                col_updates: list = [None] * len(my_g)
                for pd, (pos, templates, t_home, my_sel) in stage2_ctx.items():
                    ns_sub = [ns_by_k[pos[n]] for n in my_sel]
                    parts = layer_sharded_all_to_all_bwd(
                        ns_sub, my_sel, templates, t_home, tp_rank, tp_size, tp_group, pd
                    )
                    for n, part in enumerate(parts):
                        col_updates[pos[n]] = part
                for k in sub_pos[None]:
                    col_updates[k] = ns_by_k[k]

                # Restore the padding (zero rows) before the reverse gtp
                # exchange: its split sizes derive from the padded momentum
                # shards, and every rank's shard slice must line up again.
                col_updates = [
                    None if t is None else self._restore_pad(t, pads[my_g[k]])
                    for k, t in enumerate(col_updates)
                ]

                # 6. Reverse stage-1 all_to_all: scatter column updates back to GTP shards.
                update_shards = layer_sharded_all_to_all_bwd(
                    col_updates, my_g, moms, g_home, gtp_rank, gtp_size, gtp_group, 0
                )

            # 7. Weight update on the local shard.
            with _phase("update"):
                for p, shard in zip(params, update_shards):
                    if shard is not None:
                        self._apply_update(p, shard, lr)
