# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Layer-sharded Muon: layer sharding for Newton-Schulz over the GTP_remat x TP domain.

Instead of per-weight all-gather + redundant full-matrix NS on every rank, each weight
is assigned one NS home rank in the (GTP_remat x TP) domain. Two all_to_all stages route the
momentum shards so the home holds the complete (P, Q) matrix, Newton-Schulz runs there
with zero communication and zero redundancy — the exact same full-matrix NS as
duplicated mode — and two reverse all_to_all stages scatter the result back to the
original shards. All collectives use the existing gtp_remat / tp process groups.
"""

from __future__ import annotations

import contextlib
import dataclasses
import enum
import logging
import math
from typing import Any, Callable, Literal

import torch
from torch.optim.optimizer import ParamsT

from megatron.core.optimizer.emerging_optimizers import TensorParallelMuon
from megatron.core.optimizer.layer_sharded_a2a import (
    params_by_home,
    route_from_ns_home,
    route_to_ns_home,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.utils import (
    get_emerging_optimizers_version,
    get_pg_rank,
    get_pg_size,
    is_emerging_optimizers_min_version,
    log_single_rank,
    nvtx_range_pop,
    nvtx_range_push,
)

try:
    from emerging_optimizers import triton_kernels
    from emerging_optimizers.orthogonalized_optimizers.muon import MuonScaleT, get_muon_scale_factor
    from emerging_optimizers.orthogonalized_optimizers.muon_utils import NSCoeffT, newton_schulz
    from emerging_optimizers.utils import FP32MatmulPrecT, fp32_matmul_precision

    HAVE_EMERGING_OPTIMIZERS = True
except ImportError:
    HAVE_EMERGING_OPTIMIZERS = False

try:
    from megatron.core.tensor_parallel.gtp_api import is_gtp_param
except ImportError:  # GTP module unavailable (TransformerEngine too old): same one-line tag test

    def is_gtp_param(param) -> bool:
        """True if ``param`` carries the GTP weight-remat shard tag."""
        return getattr(param, "is_gtp_weight_remat", False)


__all__ = ["LayerShardedMuon", "ParamShardSpec", "ParamSharding", "tp_partition_dim"]

logger = logging.getLogger(__name__)

# newton_schulz() accepts a batched (3-D) input from this release on (PR #170); 0.2.0 fails
# inside torch.addmm ("mat1 must be a matrix") on the first step. The per-matrix path
# (ns_batch_size=1) uses the 2-D API every release ships.
_BATCHED_NS_MIN_EO_VERSION = "0.3.0"
# newton_schulz() dispatches 3-D inputs to the batched SYRK step (batched_tsyrk_ex, PR #276)
# from this release on; older releases raise TypeError on a 3-D input with use_syrk.
_BATCHED_SYRK_MIN_EO_VERSION = "0.5.0a0"
# SM architectures emerging-optimizers validated the SYRK kernel on (its Muon.__init__).
_SYRK_VALIDATED_SMS = ((8, 0), (9, 0), (10, 0), (10, 3))


def _validate_ns_config(use_syrk: bool, ns_batch_size: int) -> None:
    """Reject a Newton-Schulz configuration the installed stack cannot run.

    One place, one exception type (ValueError, like the parent's own gates) for every
    "this install cannot do that" condition: batched Newton-Schulz needs an
    emerging-optimizers that accepts 3-D input, and SYRK needs Triton >= 3.4.0, an SM
    emerging-optimizers validated the kernel on, and for batched chunks the batched SYRK
    kernel. The Triton / SM conditions mirror the guard in emerging-optimizers'
    ``Muon.__init__`` (which this class does not inherit from) but raise instead of
    downgrading: a run must not silently switch kernels, and with them numerics, with the
    hardware or the installed version. The parent's emerging-optimizers version gate for
    ``use_syrk`` itself still applies. Follow-up: generalize the SYRK conditions to every
    Muon mode in TensorParallelMuon.
    """
    if ns_batch_size < 1:
        raise ValueError(f"ns_batch_size must be at least 1, got {ns_batch_size}")
    if ns_batch_size > 1 and not is_emerging_optimizers_min_version(_BATCHED_NS_MIN_EO_VERSION):
        raise ValueError(
            "ns_batch_size > 1 (batched Newton-Schulz) requires emerging-optimizers >= "
            f"{_BATCHED_NS_MIN_EO_VERSION}, but {get_emerging_optimizers_version()} is "
            "installed; upgrade it or set ns_batch_size=1 (--muon-ns-batch-size 1)."
        )
    if not use_syrk:
        return
    if not torch.cuda.is_available():
        raise ValueError("use_syrk needs a CUDA device: the SYRK kernel is a Triton GPU kernel.")
    if not triton_kernels.HAS_TRITON_340:
        raise ValueError(
            "use_syrk requires Triton >= 3.4.0; upgrade Triton or drop use_syrk (--muon-use-syrk)."
        )
    sm_version = torch.cuda.get_device_capability()
    if sm_version not in _SYRK_VALIDATED_SMS:
        raise ValueError(
            "use_syrk: emerging-optimizers validates the SYRK kernel only on SM "
            f"{_SYRK_VALIDATED_SMS}, this device is SM {sm_version}; drop use_syrk "
            "(--muon-use-syrk)."
        )
    if ns_batch_size > 1 and not is_emerging_optimizers_min_version(_BATCHED_SYRK_MIN_EO_VERSION):
        raise ValueError(
            "use_syrk with ns_batch_size > 1 requires emerging-optimizers >= "
            f"{_BATCHED_SYRK_MIN_EO_VERSION} (batched SYRK kernel), but "
            f"{get_emerging_optimizers_version()} is installed; upgrade it, set "
            "ns_batch_size=1 (--muon-ns-batch-size 1), or drop use_syrk (--muon-use-syrk)."
        )


@contextlib.contextmanager
def _phase(name: str):
    """Phase-level NVTX range (active only under ``--profile`` with ``--nvtx-ranges``).

    Kernel-name classification cannot separate the forward from the reverse all_to_all,
    nor the momentum update from the weight update, so the step is annotated explicitly;
    a handful of ranges per step, not per param.
    """
    msg = f"lsmuon/{name}"
    nvtx_range_push(msg)
    try:
        yield
    finally:
        nvtx_range_pop(msg)


class ParamSharding(enum.Enum):
    """Which axes of a param group's (gtp_remat, tp) domain shard a 2-D weight.

    Derived from the model's sharding attributes and the domain sizes
    (:meth:`ParamShardSpec.from_param`); the exchanges follow from it: stage 1 runs for a
    gtp_remat axis, stage 2 for a TP axis.
    """

    REPLICATED = "replicated"
    """Whole on every rank of the domain (MoE router, latent projections, or any param in
    a single-rank domain): no exchange, every rank runs the same local Newton-Schulz."""

    GTP_REMAT = "gtp_remat"
    """dim 0 split over gtp_remat: stage-1 exchange, then every TP peer of the home column
    holds the full matrix and runs the same Newton-Schulz."""

    TP = "tp"
    """Split over TP only (the domain has no gtp_remat axis): stage-2 exchange only."""

    GTP_REMAT_AND_TP = "gtp_remat_and_tp"
    """gtp_remat shards of a TP shard: stage 1 assembles the TP-local matrix on the home
    column, stage 2 assembles the full matrix on the ``(g_home, t_home)`` rank."""


def tp_partition_dim(p: torch.Tensor) -> int | None:
    """The dim TP splits ``p`` along, or None when ``p`` is replicated across TP.

    ``tensor_model_parallel`` is the sharded/replicated flag and ``partition_dim`` is only
    meaningful when it is set, the same convention ``param_is_not_tensor_parallel_duplicate``
    uses. Megatron marks duplicated-mode TE weights ``tensor_model_parallel=False`` while TE
    still stamps ``partition_dim=0`` on them, so ``partition_dim`` alone misclassifies them.
    """
    if not getattr(p, "tensor_model_parallel", False):
        return None
    pd = getattr(p, "partition_dim", None)
    if pd is not None and pd >= 0:
        return int(pd)
    return None


@dataclasses.dataclass(frozen=True)
class ParamShardSpec:
    """How the model sharded one parameter over a (gtp_remat, tp) domain.

    Holds only model-imposed facts, fixed by the forward/backward parallelism: the
    optimizer reads them once (:meth:`from_param`) and never recomputes them in ``step()``.
    Everything the optimizer derives from them is a property.

    - ``tp_dim``: the dim TP splits (0 column-parallel, 1 row-parallel) or None
      (:func:`tp_partition_dim`, ignored when the domain has no TP axis).
    - ``gtp_sharded``: dim 0 of the TP-local shard is split over gtp_remat
      (``is_gtp_param``) and the domain has a gtp_remat axis.
    - ``pad_length``: GTP alignment padding, trailing zero rows on the gtp-gathered
      TP-local dim 0 (0 when not GTP-sharded).
    - ``full_shape``: shape of the matrix Newton-Schulz runs on (pad stripped).
    """

    tp_dim: int | None
    gtp_sharded: bool
    pad_length: int
    full_shape: tuple[int, ...]

    @classmethod
    def from_param(cls, p: torch.Tensor, gtp_remat_size: int, tp_size: int) -> "ParamShardSpec":
        """Read ``p``'s sharding for a (gtp_remat, tp) domain of the given sizes.

        An axis of size 1 is absent from the domain, so its tag is ignored: with
        ``tp_size == 1`` every param is TP-replicated, and in a single-rank domain
        everything is REPLICATED (plain local Newton-Schulz).

        Raises:
            ValueError: ``p`` is TP-sharded but not GTP-sharded while the domain has a
                gtp_remat axis. Such a param is replicated across gtp_remat, and the
                stage-1 exchange would concatenate its copies as dim-0 shards and
                silently corrupt the update.
        """
        tp_dim = tp_partition_dim(p) if tp_size > 1 else None
        gtp_tagged = bool(is_gtp_param(p))
        if gtp_remat_size > 1 and tp_dim is not None and not gtp_tagged:
            raise ValueError(
                f"LayerShardedMuon: param of shape {tuple(p.shape)} is TP-sharded "
                f"(partition_dim={tp_dim}) but not GTP-sharded (is_gtp_weight_remat "
                f"absent/False) while gtp_remat_size={gtp_remat_size} > 1. The GTP_remat "
                "exchange would concatenate replicated copies as shards and silently corrupt "
                "the update. Tag the param with is_gtp_weight_remat or run it in a domain "
                "without a GTP_remat axis."
            )
        gtp_sharded = gtp_tagged and gtp_remat_size > 1
        pad_length = int(getattr(p, "pad_length", 0) or 0) if gtp_sharded else 0
        if p.dim() != 2:
            full_shape = tuple(p.shape)
        else:
            rows, cols = p.shape
            if gtp_sharded:
                rows = rows * gtp_remat_size - pad_length
            if tp_dim == 0:
                rows *= tp_size
            elif tp_dim == 1:
                cols *= tp_size
            full_shape = (rows, cols)
        return cls(tp_dim, gtp_sharded, pad_length, full_shape)

    @property
    def sharding(self) -> ParamSharding:
        """Which domain axes shard the param; decides the exchanges it joins."""
        if self.tp_dim is not None:
            return ParamSharding.GTP_REMAT_AND_TP if self.gtp_sharded else ParamSharding.TP
        return ParamSharding.GTP_REMAT if self.gtp_sharded else ParamSharding.REPLICATED

    @property
    def ns_cost(self) -> int:
        """Newton-Schulz cost estimate on ``full_shape``, ~ max(M, N) * min(M, N)^2, the
        weight NS-home balancing uses; non-2-D shapes count their elements."""
        if len(self.full_shape) != 2:
            return math.prod(self.full_shape)
        m, n = self.full_shape
        return m * n * min(m, n)


@dataclasses.dataclass
class _GroupExchangePlan:
    """Everything ``step()`` needs for one param group that is constant across steps.

    Built once per param identity tuple by :meth:`LayerShardedMuon._build_plan` and
    invalidated by both setters; ``step()`` only consumes it. Index spaces: ``i`` indexes
    the group's grad-bearing params, ``n`` the routed sub-list, ``k`` the subset of the
    routed list that stage 1 delivers to this rank (``stage1_routed_indices``).
    """

    param_ids: tuple[int, ...]
    specs: list[ParamShardSpec]  # per i
    replicated: list[int]  # i: whole on every rank, local NS
    routed: list[int]  # i: goes through the exchanges
    ns_homes: list[tuple[int, int]]  # per n: (g_home, t_home)
    g_home: dict[int, int]  # n -> g_home, the stage-1 routing table
    pad_lengths: list[int]  # per n
    stage1_routed_indices: list[int]  # k -> n: routed params whose g_home is this rank
    tp_exchanges: dict[int, tuple[list[int], dict[int, int]]]  # tp_dim -> (k's, j -> t_home)
    tp_complete: list[int]  # k: complete after stage 1 (no TP axis), skips stage 2
    # a2a routing metadata per stage ("s1f", "s1b", ("s2f", pd), ("s2b", pd)), filled by
    # the route_* helpers on the first step.
    route_plans: dict = dataclasses.field(default_factory=dict)


class LayerShardedMuon(TensorParallelMuon):
    """Muon with layer sharding over the GTP_remat x TP domain.

    Sharding model per 2D weight of full shape ``(P, Q)``:

    - TP shards along ``param.partition_dim`` (0 = column-parallel, 1 = row-parallel) when
      ``param.tensor_model_parallel`` is set; otherwise the param is TP-replicated.
    - GTP_remat shards dim 0 of the TP-local shard, for params tagged
      ``param.is_gtp_weight_remat`` (``is_gtp_param``; absent means unsharded).
    - A param sharded by neither is whole on every rank of the domain (e.g. the MoE
      router and latent projections): it skips both exchanges and every rank runs the
      same deterministic NS on its own copy.
    - GTP alignment padding (``param.pad_length`` trailing zero rows on the
      gtp-gathered, TP-local dim 0) is stripped before Newton-Schulz — so the scale
      factor sees the true dims, matching the parent's duplicated path bitwise — and
      restored before the reverse gtp_remat exchange.

    Each param's sharding is read once from these attributes
    (:meth:`ParamShardSpec.from_param`), the per-group exchange plan (homes, routing
    tables) is cached, and ``step()`` only consumes it.

    ``step()`` runs, per param group:

    1. Momentum update on the local shard (elementwise, identical to base Muon).
    2. Stage-1 all_to_all over ``gtp_remat_group`` (dim 0): each param's GTP_remat extent is
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
        gtp_remat_group: GTP_remat weight-shard process group (dim-0 sharding of the
            TP-local shard).
        tp_group: TP process group, or None when TP is not used.
        use_syrk: Use the Triton SYRK kernel for the two symmetric-output NS GEMMs
            (``A = X Xᵀ`` and ``B = bA + cA²``), computing one triangle only —
            roughly a third off total NS FLOPs for near-square matrices. Needs
            Triton >= 3.4.0 and a validated SM (8.0/9.0/10.0/10.3); with
            ``ns_batch_size > 1`` also an emerging-optimizers with the batched SYRK
            kernel (>= 0.5.0a0, PR #276). Unmet requirements raise at construction;
            nothing downgrades silently. Only takes effect with
            ``fp32_matmul_prec="medium"`` and 8-aligned dims. Same math, different
            kernel — results differ from the GEMM path by kernel-level rounding.
        ns_batch_size: Maximum number of same-shape matrices fused into one batched
            Newton-Schulz on a home (see ``OptimizerConfig.muon_ns_batch_size``). Defaults
            to 1, the bit-exact per-matrix path; batches of more than one use ``baddbmm``
            and lose bitwise parity with duplicated mode.
        concurrent_groups: Run each param group's pipeline on its own CUDA stream
            instead of serializing them. Groups own disjoint params and, under MoE,
            disjoint process groups, so nothing orders them against each other; on a
            single stream one group's all_to_all stall blocks the other group's
            Newton-Schulz even though the GPU is idle. The ops and their order within
            a group are unchanged, so results are unaffected, but the transient
            buffers of all groups are live at once -- lower ``ns_batch_size`` or set
            this to False if that pushes peak memory too high. No effect with fewer
            than two param groups or without CUDA. Requires the groups' domains to
            be disjoint; groups sharing a (gtp_remat, tp) domain are automatically
            serialized (NCCL forbids concurrent collectives on one communicator).
        All other args: same as :class:`TensorParallelMuon`. In particular
            ``split_qkv`` / ``is_qkv_fn`` / ``qkv_split_shapes``, ``tp_mode`` and
            ``pg_collection`` only take effect on the paths that delegate to the
            parent (the empty-``param_ns_homes`` fallback and the degenerate
            single-rank domain, both of which run the parent's TP-aware
            full-matrix Newton-Schulz). "Degenerate" means the LAYER-SHARDING
            domain (gtp_remat_size * tp_size) is trivial, not that the step is
            collective-free: with a non-trivial ``pg_collection.tp`` and
            partition_dim-tagged params (direct API only), the parent path
            still issues TP collectives.

    Note:
        ``None`` for either process group means "no group / size 1", **not** torch's
        "the default group" — a missing expert group must not silently become the
        whole world.

    Usage::

        optimizer = LayerShardedMuon(params, lr=3e-4, gtp_remat_group=gtp, tp_group=tp)
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
        nesterov: bool = True,
        fp32_matmul_prec: FP32MatmulPrecT = "medium",
        coefficient_type: NSCoeffT = "quintic",
        num_ns_steps: int = 5,
        scale_mode: MuonScaleT = "spectral",
        extra_scale_factor: float = 1.0,
        gtp_remat_group: torch.distributed.ProcessGroup | None,
        tp_group: torch.distributed.ProcessGroup | None = None,
        ns_batch_size: int = 1,
        use_syrk: bool = False,
        concurrent_groups: bool = True,
        use_decoupled_weight_decay: bool = True,
        split_qkv: bool = False,
        is_qkv_fn: Callable[[torch.Tensor], bool] | None = None,
        qkv_split_shapes: list[int] | None = None,
        pg_collection: ProcessGroupCollection | None = None,
        tp_mode: Literal["blockwise", "duplicated", "distributed", "auto"] = "duplicated",
    ) -> None:
        if not HAVE_EMERGING_OPTIMIZERS:
            raise ImportError(
                "LayerShardedMuon requires the emerging-optimizers package "
                "(https://github.com/NVIDIA-NeMo/Emerging-Optimizers), which is not installed."
            )
        if tp_mode == "layer_sharded":
            raise ValueError(
                "LayerShardedMuon: 'layer_sharded' is a registry-level selector for "
                "--muon-tp-mode, not a class mode. The layer-sharded exchange is always "
                "active in this class; tp_mode selects the mode for the delegated "
                "(fallback/degenerate) paths only."
            )
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
        # Fail loudly on a Newton-Schulz configuration this stack cannot run (batched NS
        # floor, Triton/SM, batched SYRK kernel) before the parent's emerging-optimizers
        # version gate for use_syrk itself: no version upgrade fixes unfit hardware, so
        # that message must not be masked.
        _validate_ns_config(use_syrk, ns_batch_size)
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
        self.gtp_remat_group = gtp_remat_group
        self.tp_group = tp_group
        self.ns_batch_size = ns_batch_size
        # The parent stores num_ns_steps and use_syrk (validated by _validate_ns_config) but
        # only captures these three in its scaled_orthogonalize_fn closure; _run_ns reads
        # them off self.
        self.coefficient_type = coefficient_type
        self.scale_mode = scale_mode
        self.extra_scale_factor = extra_scale_factor
        self.concurrent_groups = concurrent_groups
        self._group_streams: list | None = None
        # id(param) -> (g_home, t_home). Set via set_param_ns_homes(); until then step()
        # delegates to the parent.
        self._param_ns_homes: dict[int, tuple[int, int]] = {}
        self._homes_set = False
        # Warn-once flags for the log messages below.
        self._warned_no_homes = False
        self._warned_missing_homes = False
        self._warned_shared_domain = False
        # param_group index -> (gtp_remat_group, tp_group), overriding the constructor
        # defaults. Set via set_group_process_groups().
        self._group_process_groups: dict[int, tuple] = {}
        # Per-group exchange plan (param specs, homes, a2a routing metadata): a pure
        # function of the group's params, their sharding attributes and the domain sizes,
        # all static across steps; only the tensor packing is per-step. Keyed by the
        # grad-filtered param identities so the wired path (persistent grad buffers,
        # always fully populated) hits every step, while direct-API callers that drop a
        # grad on some step safely trigger a rebuild. Metadata only, never buffers:
        # persistent exchange buffers would raise steady-state memory between steps.
        self._plans: dict[int, _GroupExchangePlan] = {}

    def set_param_ns_homes(self, param_ns_homes: dict[int, tuple[int, int]]) -> None:
        """Set the NS home for each param (by id).

        Args:
            param_ns_homes: Maps ``id(param)`` -> ``(g_home, t_home)``: the rank in
                ``gtp_remat_group`` and in ``tp_group`` that runs NS for it. ``t_home`` is
                ignored for params that are not TP-sharded and when ``tp_group`` is None.
                An empty mapping is valid: params in single-rank domains run local
                Newton-Schulz, any other routed param falls back to round-robin homes.
        """
        self._param_ns_homes = param_ns_homes
        self._homes_set = True
        self._warned_missing_homes = False
        self._plans.clear()

    def set_group_process_groups(self, group_process_groups: dict[int, tuple]) -> None:
        """Override the (GTP, TP) process groups per param group.

        Different param groups can be sharded over different domains — e.g. under
        MoE, expert weights are sharded over the *expert* GTP/TP groups while dense
        weights use the dense ones. Groups absent from the mapping fall back to the
        ``gtp_remat_group`` / ``tp_group`` passed to the constructor.

        Args:
            group_process_groups: Maps the index of a ``self.param_groups`` entry to
                ``(gtp_remat_group, tp_group)``. Either entry may be None (treated as
                size 1 / not available).
        """
        self._group_process_groups = group_process_groups
        self._warned_shared_domain = False
        self._plans.clear()

    def _apply_update(self, p: torch.Tensor, update: torch.Tensor, lr: float) -> None:
        """Apply one weight update through the base-class hook points.

        ``OrthogonalizedOptimizer.step()`` brackets every ``p.add_`` with
        ``pre_weight_update_fn_inplace`` / ``post_weight_update_fn_inplace``;
        this helper keeps layer sharding's overridden ``step()`` honouring them
        too, and keeps the two update sites (replicated, routed) from diverging. No
        dtype cast on purpose: the base class's ``p.add_(orth_grad, alpha=-lr)``
        (the third path, taken by the no-homes fallback) computes the fused
        multiply-add in the promoted
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
                # SYRK halves the two symmetric-output NS GEMMs; for batched (3-D)
                # chunks newton_schulz dispatches to the batched SYRK kernel, whose
                # availability _validate_ns_config checked at construction.
                orth = newton_schulz(
                    x,
                    steps=self.num_ns_steps,
                    coefficient_type=self.coefficient_type,
                    use_syrk=self.use_syrk,
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

    def _param_group_streams(self) -> list | None:
        """Per-group CUDA streams, or None when the groups must stay serialized.

        Concurrency requires the groups' communication domains to be disjoint:
        NCCL serializes collectives per communicator, so two param groups
        sharing a (gtp_remat, tp) domain (e.g. two dense groups produced by a
        per-layer lr override) issuing collectives from different streams can
        interleave and deadlock. Such configurations fall back to serialized
        execution (bitwise-neutral, see the concurrent_groups docstring).
        """
        if not self.concurrent_groups or len(self.param_groups) < 2:
            return None
        if not torch.cuda.is_available():
            return None
        domain_keys = []
        for group_index in range(len(self.param_groups)):
            pgs = self._group_process_groups.get(group_index, (self.gtp_remat_group, self.tp_group))
            domain_keys.append((id(pgs[0]), id(pgs[1])))
        if len(set(domain_keys)) != len(domain_keys):
            if not self._warned_shared_domain:
                self._warned_shared_domain = True
                log_single_rank(
                    logger,
                    logging.INFO,
                    "LayerShardedMuon: param groups share a (gtp_remat, tp) "
                    "communication domain; serializing groups instead of running "
                    "them on concurrent streams (NCCL requires per-communicator "
                    "serialization). Results are unaffected.",
                )
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

        # Fall back to TensorParallelMuon until homes are set: all-gather + TP-aware
        # full-matrix Newton-Schulz per param (with the groups from pg_collection).
        # Mathematically correct, just redundant: every rank recomputes every matrix.
        if not self._homes_set:
            if not self._warned_no_homes:
                self._warned_no_homes = True
                log_single_rank(
                    logger,
                    logging.WARNING,
                    "LayerShardedMuon: set_param_ns_homes() was never called; falling back "
                    "to TensorParallelMuon (per-param all-gather + full-matrix "
                    "Newton-Schulz on every rank; correct but redundant). Call it before "
                    "step() to enable layer sharding.",
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

    def _plan_for(
        self, group_index: int, params: list[torch.Tensor], gtp_remat_group, tp_group
    ) -> _GroupExchangePlan:
        """The cached exchange plan for this group's grad-bearing params.

        Rebuilt only when the param set changes (a direct-API caller dropping a grad);
        the wired path hits the cache every step.
        """
        param_ids = tuple(id(p) for p in params)
        plan = self._plans.get(group_index)
        if plan is None or plan.param_ids != param_ids:
            plan = self._build_plan(params, gtp_remat_group, tp_group)
            self._plans[group_index] = plan
        return plan

    def _build_plan(
        self, params: list[torch.Tensor], gtp_remat_group, tp_group
    ) -> _GroupExchangePlan:
        """Read every param's sharding once and assemble the routing tables."""
        gtp_remat_size = get_pg_size(gtp_remat_group)
        tp_size = get_pg_size(tp_group)
        specs = [ParamShardSpec.from_param(p, gtp_remat_size, tp_size) for p in params]
        replicated = [i for i, s in enumerate(specs) if s.sharding is ParamSharding.REPLICATED]
        routed = [i for i, s in enumerate(specs) if s.sharding is not ParamSharding.REPLICATED]

        n_missing = sum(1 for i in routed if id(params[i]) not in self._param_ns_homes)
        if n_missing and not self._warned_missing_homes:
            # Any home is mathematically valid (assignment only affects load balance),
            # but a miss usually means homes were wired against stale param objects
            # (e.g. before an fp32 main-param swap): surface it.
            self._warned_missing_homes = True
            log_single_rank(
                logger,
                logging.WARNING,
                f"LayerShardedMuon: {n_missing}/{len(routed)} params missing from "
                "param_ns_homes; falling back to round-robin (g=i%G, t=0). Load "
                "balancing (LPT) is NOT in effect for these params.",
            )
        ns_homes = [
            self._param_ns_homes.get(id(params[i]), (i % gtp_remat_size, 0)) for i in routed
        ]
        g_home = {n: home[0] for n, home in enumerate(ns_homes)}

        # This rank's share of the stage-1 exchange, by the rule the router itself applies
        # (a trivial gtp_remat group homes everything locally).
        if gtp_remat_size <= 1:
            stage1_routed_indices = list(range(len(routed)))
        else:
            stage1_routed_indices = params_by_home(len(routed), g_home, gtp_remat_size)[
                get_pg_rank(gtp_remat_group)
            ]
        by_tp_dim: dict[int | None, list[int]] = {0: [], 1: [], None: []}
        for k, n in enumerate(stage1_routed_indices):
            by_tp_dim[specs[routed[n]].tp_dim].append(k)
        tp_exchanges: dict[int, tuple[list[int], dict[int, int]]] = {}
        for pd in (0, 1):
            positions = by_tp_dim[pd]
            if positions:
                t_home = {
                    j: ns_homes[stage1_routed_indices[positions[j]]][1]
                    for j in range(len(positions))
                }
                tp_exchanges[pd] = (positions, t_home)

        return _GroupExchangePlan(
            param_ids=tuple(id(p) for p in params),
            specs=specs,
            replicated=replicated,
            routed=routed,
            ns_homes=ns_homes,
            g_home=g_home,
            pad_lengths=[specs[i].pad_length for i in routed],
            stage1_routed_indices=stage1_routed_indices,
            tp_exchanges=tp_exchanges,
            tp_complete=by_tp_dim[None],
        )

    def _step_groups(self, streams: list | None, ready: torch.cuda.Event | None) -> None:
        for group_index, group in enumerate(self.param_groups):
            if streams is not None:
                # Wait for the backward that produced these grads, then run this
                # group's whole pipeline on its own stream.
                streams[group_index].wait_event(ready)
                torch.cuda.set_stream(streams[group_index])
            # Each param group may live in its own domain (dense vs expert).
            gtp_remat_group, tp_group = self._group_process_groups.get(
                group_index, (self.gtp_remat_group, self.tp_group)
            )

            self._init_group(group)
            lr = group["lr"]
            beta = group["momentum"]

            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue
            plan = self._plan_for(group_index, params, gtp_remat_group, tp_group)

            # 1. Momentum update on the local shard. Each entry is this rank's local
            #    Newton-Schulz input: the Nesterov-corrected direction, or the momentum
            #    buffer itself.
            # NOTE: with nesterov=False, ``local_ns_inputs[i]`` aliases the momentum buffer
            # (``.float()`` is a no-op on fp32); everything below treats it as read-only.
            local_ns_inputs: list[torch.Tensor] = []
            with _phase("momentum"):
                for p in params:
                    grad = p.grad
                    state = self.state[p]
                    self._apply_weight_decay_inplace(p, grad, lr, group["weight_decay"])
                    state["momentum_buffer"].lerp_(grad, 1 - beta)
                    if self.nesterov:
                        local_ns_input = grad.lerp(state["momentum_buffer"], beta)
                    else:
                        local_ns_input = state["momentum_buffer"]
                    local_ns_inputs.append(local_ns_input.float())

            # 2. Replicated params (whole on every rank of the domain, which is every param
            #    of a single-rank domain): local Newton-Schulz on each rank's own copy,
            #    batched by shape. Correct and cheaper than electing a home and
            #    broadcasting the result back.
            if plan.replicated:
                with _phase("ns_replicated"), fp32_matmul_precision(self.fp32_matmul_prec):
                    replicated_ns_outputs = self._run_ns(
                        {i: local_ns_inputs[i] for i in plan.replicated}
                    )
                    for i, ns_output in replicated_ns_outputs.items():
                        self._apply_update(params[i], ns_output, lr)
            if not plan.routed:
                continue
            # The rank-local shards that enter the GTP_remat / TP routing path.
            local_param_shards = [params[i] for i in plan.routed]
            local_ns_input_shards = [local_ns_inputs[i] for i in plan.routed]
            stage1_routed_indices = plan.stage1_routed_indices
            route_plans = plan.route_plans

            with fp32_matmul_precision(self.fp32_matmul_prec):
                with _phase("a2a_fwd"):
                    # 3. Stage-1 all_to_all over gtp_remat (dim 0): each param's gtp_remat
                    #    extent is assembled on its g_home column. The router delivers the
                    #    same params to this rank as ``plan.stage1_routed_indices`` (both
                    #    use params_by_home).
                    stage1_ns_inputs, _ = route_to_ns_home(
                        local_ns_input_shards,
                        plan.g_home,
                        gtp_remat_group,
                        0,
                        plan=route_plans.setdefault("s1f", {}),
                    )
                    # Strip the GTP alignment padding at the stage-1 seam: the stage-1
                    # output is the gtp-gathered TP-LOCAL tensor, where the pad is a
                    # contiguous dim-0 tail for every partition dim (after TP assembly it
                    # would be embedded per TP block), the same strip point the parent's
                    # duplicated path uses.
                    stage1_ns_inputs = [
                        self._strip_pad(t, plan.pad_lengths[stage1_routed_indices[k]])
                        for k, t in enumerate(stage1_ns_inputs)
                    ]

                    # 4. Stage-2 all_to_all over TP, one exchange per partition dim.
                    #    GTP_REMAT-only params skip it: every TP peer of the column already
                    #    holds their full matrix. Both dicts below are keyed by ``k``, the
                    #    stage-1 index.
                    full_ns_inputs_by_stage1_index: dict[int, torch.Tensor] = {}
                    # Forward TP-routing state the reverse exchange needs, per partition dim:
                    # the TP-local shards sent and the positions this rank's home received.
                    tp_reverse_context_by_partition_dim: dict[
                        int, tuple[list[torch.Tensor], list[int]]
                    ] = {}
                    for pd, (positions, t_home) in plan.tp_exchanges.items():
                        tp_local_ns_input_shards = [stage1_ns_inputs[k] for k in positions]
                        local_full_ns_inputs, tp_input_indices_for_local_home = route_to_ns_home(
                            tp_local_ns_input_shards,
                            t_home,
                            tp_group,
                            pd,
                            plan=route_plans.setdefault(("s2f", pd), {}),
                        )
                        tp_reverse_context_by_partition_dim[pd] = (
                            tp_local_ns_input_shards,
                            tp_input_indices_for_local_home,
                        )
                        for tp_index, full_ns_input in zip(
                            tp_input_indices_for_local_home, local_full_ns_inputs
                        ):
                            full_ns_inputs_by_stage1_index[positions[tp_index]] = full_ns_input
                    for k in plan.tp_complete:
                        full_ns_inputs_by_stage1_index[k] = stage1_ns_inputs[k]

                # 5. Full-matrix Newton-Schulz on the home (identical to duplicated mode),
                #    batched by shape; see _run_ns. The TP-complete (GTP_REMAT-only)
                #    subset runs through its OWN _run_ns call: every TP column
                #    orthogonalizes those matrices independently and scatters its own
                #    result over gtp_remat alone, so their batch chunking must not depend
                #    on the column's TP-sharded params (a different set per column).
                #    Pooled chunking would put the same replicated matrix in a baddbmm
                #    chunk on one column and addmm on another, and the TP replicas of its
                #    weight would drift apart and compound every step.
                tp_complete_indices = set(plan.tp_complete)
                with _phase("ns"):
                    ns_outputs_by_stage1_index = self._run_ns(
                        {
                            k: v
                            for k, v in full_ns_inputs_by_stage1_index.items()
                            if k not in tp_complete_indices
                        }
                    )
                    ns_outputs_by_stage1_index.update(
                        self._run_ns(
                            {k: full_ns_inputs_by_stage1_index[k] for k in plan.tp_complete}
                        )
                    )

            with _phase("a2a_bwd"):
                # 6. Reverse stage-2 all_to_all: scatter NS results back to TP parts. The
                #    result, per stage-1 index, is the update for the TP-local shard.
                stage1_update_shards: list = [None] * len(stage1_routed_indices)
                for pd, (positions, t_home) in plan.tp_exchanges.items():
                    tp_local_ns_input_shards, tp_input_indices_for_local_home = (
                        tp_reverse_context_by_partition_dim[pd]
                    )
                    tp_ns_outputs = [
                        ns_outputs_by_stage1_index[positions[tp_index]]
                        for tp_index in tp_input_indices_for_local_home
                    ]
                    tp_update_shards = route_from_ns_home(
                        tp_ns_outputs,
                        tp_input_indices_for_local_home,
                        tp_local_ns_input_shards,
                        t_home,
                        tp_group,
                        pd,
                        plan=route_plans.setdefault(("s2b", pd), {}),
                    )
                    for tp_index, tp_update_shard in enumerate(tp_update_shards):
                        stage1_update_shards[positions[tp_index]] = tp_update_shard
                for k in plan.tp_complete:
                    stage1_update_shards[k] = ns_outputs_by_stage1_index[k]

                # Restore the padding (zero rows) before the reverse gtp_remat exchange:
                # its split sizes derive from the padded momentum shards, and every
                # rank's shard slice must line up again.
                stage1_update_shards = [
                    (
                        None
                        if t is None
                        else self._restore_pad(t, plan.pad_lengths[stage1_routed_indices[k]])
                    )
                    for k, t in enumerate(stage1_update_shards)
                ]

                # 7. Reverse stage-1 all_to_all: scatter the TP-local updates back to the
                #    gtp_remat shards.
                update_shards = route_from_ns_home(
                    stage1_update_shards,
                    stage1_routed_indices,
                    local_ns_input_shards,
                    plan.g_home,
                    gtp_remat_group,
                    0,
                    plan=route_plans.setdefault("s1b", {}),
                )

            # 8. Weight update on the local shard.
            with _phase("update"):
                for p, shard in zip(local_param_shards, update_shards):
                    if shard is not None:
                        self._apply_update(p, shard, lr)
