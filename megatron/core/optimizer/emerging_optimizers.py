# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Emerging optimizer registry.

To add a new emerging optimizer:
  1. Define its optimizer class (or import it).
  2. Write its ``_<name>_init_state_fn`` and ``_<name>_config_to_kwargs``.
  3. Add an ``EmergingOptimizerEntry`` to ``_EMERGING_OPTIMIZERS`` at the bottom.
"""

import inspect
import itertools
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generator, Literal, NamedTuple, Optional, Sequence, get_args

import torch
from torch.optim.optimizer import ParamsT

from megatron.core.process_groups_config import ProcessGroupCollection, resolve_gtp_remat_group
from megatron.core.utils import (
    get_emerging_optimizers_version,
    get_pg_rank,
    get_pg_size,
    is_emerging_optimizers_min_version,
    log_single_rank,
)

from .optimizer_config import ParamKey, ParamPredicate

try:
    from emerging_optimizers import registry
    from emerging_optimizers.orthogonalized_optimizers import (
        AdaptiveMuon,
        OrthogonalizedOptimizer,
        get_muon_scale_factor,
    )
    from emerging_optimizers.orthogonalized_optimizers.muon_utils import NSCoeffT, newton_schulz_tp

    # It is necessary to import optimizers for the registry to work.
    from emerging_optimizers.scalar_optimizers import Lion  # pylint: disable=unused-import
    from emerging_optimizers.soap import SOAP  # pylint: disable=unused-import

    HAVE_EMERGING_OPTIMIZERS = True
except ImportError:
    HAVE_EMERGING_OPTIMIZERS = False
    OrthogonalizedOptimizer = object
    AdaptiveMuon = object


logger = logging.getLogger(__name__)

# newton_schulz_tp() gained the use_syrk kwarg in emerging_optimizers 0.4.0. Earlier releases
# expose use_syrk on the non-TP newton_schulz() only, so 0.3.x still rejects it here. Spelled
# ".dev0" so pre-release builds of that line are accepted too, matching how the TE minimums
# elsewhere in the tree are written.
_SYRK_MIN_EO_VERSION = "0.4.0.dev0"


def get_supported_coefficient_types() -> tuple[str, ...]:
    """Return the coefficient types supported by the installed emerging_optimizers.

    Reads the members of the ``NSCoeffT`` Literal type so that new types
    added upstream are automatically available without code changes here.
    """
    assert (
        HAVE_EMERGING_OPTIMIZERS
    ), "emerging_optimizers >= 0.2 is required for NSCoeffT. Please install or upgrade it."
    return get_args(NSCoeffT)


def validate_coefficient_type(coefficient_type: str) -> None:
    """Raise ``ValueError`` if *coefficient_type* is not supported."""
    supported = get_supported_coefficient_types()
    if coefficient_type not in supported:
        raise ValueError(
            f"Unsupported muon coefficient type '{coefficient_type}'. "
            f"Supported types: {supported}"
        )


# ===========================================================================
# Registry dataclass and public API
# ===========================================================================


def _eopt_init_state_fn(opt, config=None):
    """Initialize emerging optimizer state for torch_dist checkpoint format."""
    for group in opt.param_groups:
        # Checkpoint init needs state for all parameters, including those without grads yet.
        opt._init_group(group, skip_non_grad_params=False)


def _default_param_overrides_factory() -> Dict[ParamKey, Dict[str, Any]]:
    """Default param overrides: route non-linear/embedding params to Adam."""
    return {
        ParamKey(
            predicate=ParamPredicate(name="nonlinear_or_embedding", fn=_is_nonlinear_or_embedding)
        ): {'optimizer': 'adam'}
    }


@dataclass
class EmergingOptimizerEntry:
    """Everything needed to create and configure an emerging optimizer.

    Attributes:
        optimizer_cls: The torch optimizer class (default when config_to_cls is None).
        init_state_fn: Lazily initialises optimizer state (needed for checkpoint formats).
        config_to_kwargs: ``(config, model_chunks, pg_collection) -> dict`` of constructor kwargs.
        config_to_cls: Optional ``(config) -> type`` callable. When provided, overrides
            ``optimizer_cls`` so the instantiated class can vary based on config (e.g.
            selecting ``LayerShardedMuon`` when ``muon_tp_mode == 'layer_sharded'``).
        default_param_overrides: Per-parameter config overrides applied automatically
            (e.g. route non-linear params to Adam).
    """

    optimizer_cls: type
    init_state_fn: Callable = _eopt_init_state_fn
    config_to_kwargs: Callable | None = None
    config_to_cls: Callable | None = None
    default_param_overrides: Dict[ParamKey, Dict[str, Any]] = field(
        default_factory=_default_param_overrides_factory
    )


def _create_emerging_optimizer(config, param_groups, eopt_name, model_chunks, pg_collection):
    """Instantiate an emerging optimizer and return it with its init_state_fn."""
    entry = _EMERGING_OPTIMIZERS[eopt_name]
    if entry.config_to_kwargs is not None:
        eopt_kwargs = entry.config_to_kwargs(config, model_chunks, pg_collection)
    else:
        eopt_kwargs = _default_adam_based_eopt_config_to_kwargs(
            eopt_name, config, model_chunks, pg_collection
        )
    optimizer_cls = (
        entry.config_to_cls(config) if entry.config_to_cls is not None else entry.optimizer_cls
    )
    optimizer = optimizer_cls(param_groups, **eopt_kwargs)
    return optimizer, entry.init_state_fn


# ===========================================================================
# Shared helpers
# ===========================================================================


def _is_nonlinear_or_embedding(param):
    """True for parameters that should NOT use the emerging optimizer."""
    return getattr(param, 'is_embedding_or_output_parameter', False) or len(param.shape) != 2


def _is_muon_excluded(param):
    """True for parameters that should use the scalar optimizer instead of Muon."""
    return not getattr(param, 'use_muon', True) or _is_nonlinear_or_embedding(param)


def _get_qkv_split_shapes(model_cfg) -> list[int]:
    """Compute QKV split shapes from model config."""
    query_projection_size = (
        model_cfg.num_attention_heads // model_cfg.num_query_groups * model_cfg.kv_channels
    )
    if getattr(model_cfg, 'attention_output_gate', False):
        return [
            query_projection_size,
            query_projection_size,
            model_cfg.kv_channels,
            model_cfg.kv_channels,
        ]
    return [query_projection_size, model_cfg.kv_channels, model_cfg.kv_channels]


def split_qkv_shard_rows(
    qkv_split_shapes: Sequence[int], rows: int, shard_rows: int, rank: int
) -> list[list[tuple[int, int, int]]]:
    """Where the projection pieces of a fused qkv weight sit in one GTP row shard.

    The fused matrix has ``rows`` logical rows laid out per query group as
    ``[q | (gate |) k | v]`` with the widths ``qkv_split_shapes``; GTP padding rows trail them.
    It is split into row shards of ``shard_rows`` rows, and shard ``rank`` covers fused rows
    ``[rank * shard_rows, (rank + 1) * shard_rows)``. A piece's rows are its query groups'
    slabs concatenated in group order, the layout the duplicated split orthogonalizes.

    Returns, per piece, the runs ``(local_start, local_end, piece_start)``: local rows
    ``[local_start, local_end)`` of the shard are piece rows
    ``[piece_start, piece_start + local_end - local_start)``. Runs are in piece-row order;
    padding rows belong to no piece.
    """
    group_rows = sum(qkv_split_shapes)
    if rows % group_rows != 0:
        raise RuntimeError(
            f"Muon QKV split shape mismatch: rows={rows}, split_shapes={list(qkv_split_shapes)}"
        )
    lo, hi = rank * shard_rows, (rank + 1) * shard_rows
    runs: list[list[tuple[int, int, int]]] = []
    offset = 0
    for width in qkv_split_shapes:
        piece_runs = []
        for g in range(rows // group_rows):
            start = g * group_rows + offset
            a, b = max(start, lo), min(start + width, hi)
            if a < b:
                piece_runs.append((a - lo, b - lo, g * width + a - start))
        runs.append(piece_runs)
        offset += width
    return runs


class _ChainStep(NamedTuple):
    """One step a Muon chain generator hands to its driver; the driver sends back the result.

    ``kind`` tells a driver what it may do with the step: ``"collective"`` is a
    ``torch.distributed`` collective (a step scheduler may issue it on a communication
    stream), ``"distributed_ns"`` is a distributed Newton-Schulz over ``args[1]`` (a scheduler
    may run it as its own per-step chain). :func:`run_serially` runs every step as written.
    """

    kind: Literal["collective", "distributed_ns"]
    fn: Callable
    args: tuple
    kwargs: dict


def run_serially(chain: Generator[_ChainStep, Any, Any]) -> Any:
    """Drive a chain generator to completion on the current stream: every yielded
    :class:`_ChainStep` runs synchronously and its result is sent back. Returns the
    generator's return value."""
    try:
        step = next(chain)
        while True:
            step = chain.send(step.fn(*step.args, **step.kwargs))
    except StopIteration as stop:
        return stop.value


# ===========================================================================
# Registry – populated below only when emerging_optimizers is installed.
# ===========================================================================

_EMERGING_OPTIMIZERS: Dict[str, EmergingOptimizerEntry] = {}


# ===========================================================================
# Muon
# ===========================================================================


# tp_mode="auto" selects tp_mode per weight (Dense/GTP weights only)
_AUTO_TP_MODES = ("duplicated", "distributed")


@dataclass(frozen=True)
class HardwareProfile:
    """HW spec for different GPU. Use for cost model when selecting TP mode.
    Bandwidths are UNIDIRECTIONAL."""

    bf16_peak_tflops: float  # dense, fp32_matmul_prec = "medium" for now
    bw_intra_gbps: float  # collectives staying inside one NVLink domain
    bw_inter_gbps: float  # collectives crossing domains, over the fabric


_PROFILES = {
    # Keys are matched as a substring of the reported device name ("NVIDIA GB200").
    # Both bandwidths are PER GPU. Only run on GB200 & GB300 for now.
    # TODO: May need to add other HW Spec
    "GB200": HardwareProfile(bf16_peak_tflops=2500.0, bw_intra_gbps=900.0, bw_inter_gbps=100.0),
    "GB300": HardwareProfile(bf16_peak_tflops=2500.0, bw_intra_gbps=900.0, bw_inter_gbps=100.0),
}


def _hardware_profile() -> Optional[HardwareProfile]:
    """Profile for the local GPU, or None when the hardware is not in the registry."""
    try:
        name = torch.cuda.get_device_properties(0).name
    except Exception:  # noqa: BLE001 - no CUDA device: fall back, do not fail
        return None
    return next((prof for key, prof in _PROFILES.items() if key in name), None)


def _select_tp_mode(
    m: int,
    n: int,
    group_size: int,
    steps: int,
    use_syrk: bool,
    elem_size: int,
    communication_crosses_domain: bool,
    profile: Optional[HardwareProfile] = None,
    candidates: tuple[str, ...] = _AUTO_TP_MODES,
) -> str:
    """Cost model for per weight tp_mode selection. Mirrors the op sequence in
    scaled_orthogonalize_fn_with_gtp_remat -- keep in sync.
    """
    min_dim, max_dim = min(m, n), max(m, n)
    m_partitioned = (
        m // group_size
    )  # dist orthogonalizes the [n, m/group_size] shard; transpose is forced
    gram = 1 if use_syrk else 2  # SYRK halves the two gram ops
    # Per NS step: gram X@X.T + gram A@A + GEMM B@X.
    flops = {
        "duplicated": steps
        * (gram * (min_dim * min_dim * max_dim + min_dim**3) + 2 * min_dim * min_dim * max_dim),
        "distributed": steps * (gram * (n * n * m_partitioned + n**3) + 2 * n * n * m_partitioned),
    }
    if profile is None:
        return "duplicated" if communication_crosses_domain else min(candidates, key=flops.get)

    ring_fraction = (
        group_size - 1
    ) / group_size  # ring: each rank moves (group_size-1)/group_size of the buffer
    num_bytes = {
        "duplicated": m * n * elem_size * ring_fraction,  # one all-gather
        "distributed": steps * 2 * n * n * elem_size * ring_fraction,  # gram all-reduce per step
    }
    bw = (profile.bw_inter_gbps if communication_crosses_domain else profile.bw_intra_gbps) * 1e9
    peak = profile.bf16_peak_tflops * 1e12
    cost = {mode: flops[mode] / peak + num_bytes[mode] / bw for mode in candidates}
    return min(candidates, key=cost.get)


class TensorParallelMuon(OrthogonalizedOptimizer):
    """Tensor Parallel Muon optimizer."""

    def __init__(
        self,
        params: ParamsT,
        lr: float = 3e-4,
        momentum: float = 0.95,
        nesterov: bool = True,
        weight_decay: float = 0.01,
        use_decoupled_weight_decay: bool = True,
        split_qkv: bool = False,
        is_qkv_fn: Callable[[torch.Tensor], bool] | None = None,
        qkv_split_shapes: list[int] | None = None,
        fp32_matmul_prec: str = "medium",
        coefficient_type: str = "quintic",
        num_ns_steps: int = 5,
        scale_mode: str = "spectral",
        extra_scale_factor: float = 1.0,
        pg_collection: Optional[ProcessGroupCollection] = None,
        tp_mode: Literal["blockwise", "duplicated", "distributed", "auto"] = "duplicated",
        use_syrk: bool = False,
    ) -> None:
        if num_ns_steps < 1:
            raise ValueError(f"num_ns_steps must be at least 1, got {num_ns_steps}")
        if use_syrk and not is_emerging_optimizers_min_version(_SYRK_MIN_EO_VERSION):
            raise ValueError(
                f"use_syrk requires emerging_optimizers >= {_SYRK_MIN_EO_VERSION}, but "
                f"{get_emerging_optimizers_version()} is installed. Upgrade "
                "emerging_optimizers or drop --muon-use-syrk."
            )

        def scaled_orthogonalize_fn(
            grad: torch.Tensor,
            tp_group: torch.distributed.ProcessGroup,
            partition_dim: int | None = None,
            tp_mode_this_group: str = tp_mode,
            scale_shape: tuple[int, int] | None = None,
        ) -> torch.Tensor:
            log_single_rank(
                logger,
                logging.DEBUG,
                f'Orthogonalizing grad with {num_ns_steps} steps, '
                f'{coefficient_type} coefficient, '
                f'{scale_mode} scale mode, extra_scale_factor={extra_scale_factor}',
            )
            if scale_shape is None:
                size = [grad.size(-2), grad.size(-1)]
                if partition_dim is not None:
                    size[partition_dim] *= get_pg_size(tp_group)
            else:
                # This overrides only the final Muon scalar; NS still uses grad's physical
                # shape and partition metadata for its collectives.
                size = scale_shape
            # Only forward the kwarg when enabled; older emerging_optimizers do not
            # accept it at all, and __init__ has already rejected use_syrk on those.
            ns_kwargs = {"use_syrk": True} if use_syrk else {}
            orth_grad = newton_schulz_tp(
                grad,
                steps=num_ns_steps,
                coefficient_type=coefficient_type,
                tp_group=tp_group,
                partition_dim=partition_dim,
                tp_mode="duplicated" if tp_mode_this_group == "blockwise" else tp_mode_this_group,
                **ns_kwargs,
            )
            scale_factor = get_muon_scale_factor(size[0], size[1], mode=scale_mode)
            return orth_grad * scale_factor * extra_scale_factor

        self.pg_collection = pg_collection
        self.tp_mode = tp_mode
        self.split_qkv = split_qkv
        self.is_qkv_fn = is_qkv_fn
        self.qkv_split_shapes = qkv_split_shapes
        # For the tp_mode="auto" cost model (_resolve_tp_mode / _select_tp_mode).
        self.num_ns_steps = num_ns_steps
        self.use_syrk = use_syrk
        self.elem_size = 2 if fp32_matmul_prec == "medium" else 4  # bf16 vs tf32/fp32
        self._tp_mode_cache: Dict[tuple, str] = {}
        self._hw_profile = _hardware_profile() if tp_mode == "auto" else None
        self._warned_qkv_split_disabled = False
        self._split_qkv_modes_cache: Dict[tuple, tuple[bool, ...]] = {}

        weight_decay_method = "decoupled" if use_decoupled_weight_decay else "l2"
        # Use explicit class call instead of super() so that subclasses with
        # multiple inheritance (e.g. TensorParallelAdaptiveMuon) don't route
        # through an intermediate class that doesn't accept scaled_orthogonalize_fn.
        OrthogonalizedOptimizer.__init__(
            self,
            params,
            lr,
            momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
            weight_decay_method=weight_decay_method,
            fp32_matmul_prec=fp32_matmul_prec,
            scaled_orthogonalize_fn=scaled_orthogonalize_fn,
        )

    @staticmethod
    def _all_gather_tensor(t, group, dim):
        """All-gather equal-size shards of ``t`` over ``group`` and concat along ``dim``."""
        shards = [torch.empty_like(t) for _ in range(get_pg_size(group))]
        torch.distributed.all_gather(shards, t.contiguous(), group)
        return torch.cat(shards, dim=dim)

    @staticmethod
    def _strip_pad(t, pad_length):
        """Drop the trailing ``pad_length`` rows of dim 0 (no-op if ``pad_length == 0``)."""
        return t[:-pad_length] if pad_length else t

    @staticmethod
    def _restore_pad(t, pad_length):
        """Re-append ``pad_length`` zero rows to dim 0 (no-op if ``pad_length == 0``)."""
        return torch.nn.functional.pad(t, (0, 0, 0, pad_length)) if pad_length else t

    def _resolve_tp_mode(self, m: int, n: int, group_size: int) -> str:
        """Cached per-shape mode for tp_mode="auto", dense (GTP) weights only.

        communication_crosses_domain=False always: this is only called for dense weights
        (see scaled_orthogonalize_fn_with_gtp_remat), and GTP stays inside one NVLink domain.
        """
        key = (m, n, group_size)
        if key not in self._tp_mode_cache:
            self._tp_mode_cache[key] = _select_tp_mode(
                m,
                n,
                group_size,
                self.num_ns_steps,
                self.use_syrk,
                self.elem_size,
                communication_crosses_domain=False,
                profile=self._hw_profile,
            )
            log_single_rank(
                logger,
                logging.INFO,
                f"muon tp_mode=auto (dense): ({m}, {n}) group_size={group_size} -> "
                f"{self._tp_mode_cache[key]}",
            )
        return self._tp_mode_cache[key]

    def _split_qkv_piece_modes(
        self, p, qkv_split_shapes, gtp_active, gtp_remat_size, is_expert, tp_group, partition_dim
    ) -> Optional[tuple[bool, ...]]:
        """Split-QKV under the distributed mode: per projection piece of ``p``, whether it runs
        the distributed Newton-Schulz on its rows of the GTP shards
        (:meth:`_split_qkv_distributed_chain`). Under ``tp_mode="distributed"`` every piece
        does; under ``auto`` the cost model resolves each piece on its own logical shape, like
        any other dense matrix (:meth:`_resolve_tp_mode`). None when that chain cannot serve
        ``p``: no split, ``tp_mode`` neither ``distributed`` nor ``auto``, or ``p`` is not a
        dense weight with GTP as its only sharding axis. Cached per shape.

        TODO: the chain has no test with ``use_syrk`` yet (it needs an emerging_optimizers
        build with the SYRK kernel); a rank that holds no row of a distributed piece hands the
        distributed Newton-Schulz an empty shard."""
        if not qkv_split_shapes or self.tp_mode not in ("distributed", "auto"):
            return None
        if not gtp_active or is_expert:
            return None
        if partition_dim is not None and tp_group is not None and get_pg_size(tp_group) > 1:
            return None  # GTP + TP two-step weights keep their existing behaviour.
        rows = p.shape[0] * gtp_remat_size - getattr(p, 'pad_length', 0)
        cols = p.shape[1]
        key = (rows, cols, gtp_remat_size, tuple(qkv_split_shapes))
        if key not in self._split_qkv_modes_cache:
            group_rows = sum(qkv_split_shapes)
            if rows % group_rows != 0:
                raise RuntimeError(
                    f"Muon QKV split shape mismatch: rows={rows}, "
                    f"split_shapes={list(qkv_split_shapes)}"
                )
            piece_rows = [rows // group_rows * width for width in qkv_split_shapes]
            modes = [
                (
                    "distributed"
                    if self.tp_mode == "distributed"
                    else self._resolve_tp_mode(piece, cols, gtp_remat_size)
                )
                for piece in piece_rows
            ]
            self._split_qkv_modes_cache[key] = tuple(mode == "distributed" for mode in modes)
            log_single_rank(
                logger,
                logging.INFO,
                f"muon split-QKV tp_mode={self.tp_mode}: ({rows}, {cols}) "
                f"group_size={gtp_remat_size} pieces {piece_rows} x {cols} -> {modes}",
            )
        return self._split_qkv_modes_cache[key]

    def _split_qkv_distributed_chain(
        self, grad, gtp_group, tp_group, partition_dim, pad_length, qkv_split_shapes, distributed
    ) -> Generator[_ChainStep, Any, torch.Tensor]:
        """Projection-level split-QKV of a GTP-sharded fused qkv momentum without gathering it.

        A chain generator: every collective and the pieces' distributed Newton-Schulz are
        yielded as ``_ChainStep`` requests, so the same code serves the serial step
        (:func:`run_serially`) and a step scheduler. ``grad`` is this rank's ``[R, K]`` row
        shard of the padded fused matrix. Every projection piece is its own Newton-Schulz
        problem, scaled by its logical shape, exactly the duplicated split's rule.
        ``distributed[j]`` (:meth:`_split_qkv_piece_modes`) says how piece ``j`` runs:

        - a distributed piece (Q at production sizes): the distributed chain over
          ``gtp_group`` on this rank's rows of the piece. The Gram is over the complete
          columns, so the row order inside the shard does not matter and a rank without rows
          contributes zeros.
        - the other pieces (K and V at production sizes, which the cost model keeps
          duplicated): ONE all-reduce of the zero-filled pieces assembles them on every rank
          (each row has exactly one contributor, so the sum is the exact copy the duplicated
          path all-gathers), then the duplicated split's own Newton-Schulz call on the ranks
          that hold rows of the piece: bitwise its result.

        Order: the assembled pieces' all-reduce, the distributed pieces' chains, then the
        assembled pieces' Newton-Schulz. Run before the chains, that host-launch-bound
        Newton-Schulz would hold every rank at the first chain collective until the ranks
        holding K / V rows finish it; queued behind the chains, its launches hide under their
        GPU time. Padding rows stay zero. Every rank issues the same collectives in the same
        order: the assembled pieces' Newton-Schulz is a direct call on the ranks that hold
        their rows, which is only sound because TP is not a sharding axis here (checked on
        entry), so that call involves no other rank.
        """
        if partition_dim is not None and tp_group is not None and get_pg_size(tp_group) > 1:
            # The assembled pieces' Newton-Schulz below runs only on the ranks that hold their
            # rows and is not a yielded step: with TP as a sharding axis it would all-gather
            # over the TP group on some ranks only.
            raise RuntimeError(
                "The split-aware split-QKV chain requires GTP to be the weight's only sharding "
                f"axis, got a TP group of size {get_pg_size(tp_group)} with "
                f"partition_dim={partition_dim}"
            )
        group_size, rank = get_pg_size(gtp_group), get_pg_rank(gtp_group)
        shard_rows, cols = grad.shape
        rows = shard_rows * group_size - pad_length
        runs = split_qkv_shard_rows(qkv_split_shapes, rows, shard_rows, rank)
        piece_rows = [rows // sum(qkv_split_shapes) * width for width in qkv_split_shapes]
        out = torch.zeros_like(grad)

        gathered = [j for j, is_distributed in enumerate(distributed) if not is_distributed]
        starts = list(itertools.accumulate((piece_rows[j] for j in gathered), initial=0))
        assembled = grad.new_zeros(starts[-1], cols)
        if gathered:
            for j, at in zip(gathered, starts):
                for lo, hi, piece_start in runs[j]:
                    assembled[at + piece_start : at + piece_start + hi - lo] = grad[lo:hi]
            yield _ChainStep(
                "collective", torch.distributed.all_reduce, (assembled,), {"group": gtp_group}
            )

        for j, is_distributed in enumerate(distributed):
            if not is_distributed:
                continue
            local = (
                torch.cat([grad[lo:hi] for lo, hi, _ in runs[j]])
                if runs[j]
                else grad.new_zeros(0, cols)
            )
            orth = yield _ChainStep(
                "distributed_ns",
                self.scaled_orthogonalize_fn,
                (local, gtp_group),
                {
                    "partition_dim": 0,
                    "tp_mode_this_group": "distributed",
                    "scale_shape": (piece_rows[j], cols),
                },
            )
            at = 0
            for lo, hi, _ in runs[j]:
                out[lo:hi] = orth[at : at + hi - lo]
                at += hi - lo

        for j, at in zip(gathered, starts):
            if not runs[j]:
                continue  # no row of this piece here: its result is not needed
            orth = self.scaled_orthogonalize_fn(
                assembled[at : at + piece_rows[j]],
                tp_group,
                partition_dim,
                tp_mode_this_group="duplicated",
            )
            for lo, hi, piece_start in runs[j]:
                out[lo:hi] = orth[piece_start : piece_start + hi - lo]
        return out

    def scaled_orthogonalize_fn_with_gtp_remat(
        self, p, grad, tp_group, partition_dim, qkv_split_shapes=None
    ):
        """Orthogonalize a (possibly GTP-sharded) momentum, then reshard.

        When GTP is inactive this is a plain passthrough to ``scaled_orthogonalize_fn``.
        Otherwise, ``mode`` (``self.tp_mode``, or resolved per-weight when
        ``self.tp_mode == "auto"``) controls how GTP sharding is handled:

        - **blockwise**: orthogonalize the local GTP shard independently, no collective.
        - **duplicated**: all-gather over GTP, run whole-matrix NS (TP-aware), reshard.
        - **distributed**: distribute NS over GTP via small-Gram all-reduce. When both
          GTP and TP are active, NS is distributed over the larger group to minimize
          redundant compute; the smaller group is all-gathered beforehand.

        GTP_remat may pad dim 0 for alignment (see gtp_remat_shard_dim0). blockwise and
        duplicated strip the padding before calling scaled_orthogonalize_fn and restore it
        after, since every rank holds a uniform, fully-reconstructed tensor by then.
        distributed keeps the padding needed by its collective, but applies Muon's scale
        factor using the unpadded logical matrix shape.

        ``qkv_split_shapes`` (when set) runs Newton-Schulz on q, k and v separately. Under
        GTP the duplicated path splits the gathered matrix. Under distributed and under auto a
        dense weight whose only sharding axis is GTP gets a mode per projection piece
        (:meth:`_split_qkv_piece_modes`: every piece distributed under distributed, the cost
        model's answer for each piece under auto) and runs the same split without gathering
        the matrix (:meth:`_split_qkv_distributed_chain`); where auto keeps every piece
        duplicated it runs the duplicated split. Every other blockwise / distributed case
        cannot keep the q/k/v boundaries: it falls back to whole-matrix NS and warns once.
        """
        if qkv_split_shapes is not None and not qkv_split_shapes:
            raise ValueError(
                "qkv_split_shapes must be None (no split) or non-empty; got an empty sequence"
            )

        # TODO: Clean up code that determines if parameter is a MoE layer and which TP group to use
        is_expert = getattr(p, 'expert_tp', False)
        gtp_remat_group = (
            resolve_gtp_remat_group(self.pg_collection, is_expert) if self.pg_collection else None
        )

        # Parameters with is_gtp_weight_remat=False are not sharded along the
        # GTP process group, and do not require all-gathering prior to
        # orthogonalization.
        gtp_active = (
            gtp_remat_group is not None
            and get_pg_size(gtp_remat_group) > 1
            and getattr(p, 'is_gtp_weight_remat', False)
        )
        gtp_remat_size = get_pg_size(gtp_remat_group) if gtp_active else 1

        mode = self.tp_mode
        split_modes = self._split_qkv_piece_modes(
            p, qkv_split_shapes, gtp_active, gtp_remat_size, is_expert, tp_group, partition_dim
        )
        if split_modes is not None:
            # A split qkv weight the distributed split chain can serve has one mode per
            # projection piece: any distributed piece runs the chain, none (auto only) is the
            # duplicated split.
            mode = "distributed" if any(split_modes) else "duplicated"
        elif mode == "auto":
            if qkv_split_shapes is not None:
                # Only duplicated can honor a split; the cost model below doesn't know
                # that, so pin it here instead of letting the shape decide.
                mode = "duplicated"
            else:
                # Scoped to dense (GTP) weights for now; expert weights keep today's default.
                mode = (
                    self._resolve_tp_mode(p.shape[0] * gtp_remat_size, p.shape[1], gtp_remat_size)
                    if gtp_active and not is_expert
                    else "duplicated"
                )

        def scaled_orthogonalize_fn_with_qkv_split(grad):
            """Orthogonalize `grad`, splitting [q|k|v] first when `qkv_split_shapes` is not None.

            `grad` is always GTP-unsharded -- GTP is inactive, or the caller has already
            gathered. A GTP row shard would cut q/k/v mid-boundary.
            """
            if qkv_split_shapes is None:
                return self.scaled_orthogonalize_fn(
                    grad, tp_group, partition_dim, tp_mode_this_group=mode
                )
            qkv_rows = sum(qkv_split_shapes)
            if grad.size(0) % qkv_rows != 0:
                raise RuntimeError(
                    f"Muon QKV split shape mismatch: grad_shape={tuple(grad.shape)}, "
                    f"split_shapes={qkv_split_shapes}"
                )
            num_query_groups = grad.size(0) // qkv_rows
            cols = grad.size(-1)
            qkv_grads = torch.split(
                grad.view(num_query_groups, qkv_rows, cols), qkv_split_shapes, dim=1
            )
            qkv_grads = [
                self.scaled_orthogonalize_fn(
                    g.reshape(-1, cols), tp_group, partition_dim, tp_mode_this_group=mode
                ).view(num_query_groups, -1, cols)
                for g in qkv_grads
            ]
            return torch.cat(qkv_grads, dim=1).view(grad.shape)

        if not gtp_active:
            return scaled_orthogonalize_fn_with_qkv_split(grad)

        gtp_rank = get_pg_rank(gtp_remat_group)
        pad_length = getattr(p, 'pad_length', 0)

        if split_modes is not None and mode == "distributed":
            # The decision above reads p's layout and the chain reads grad's: they must be the
            # same shard (the plain distributed path checks this for padded weights below).
            if grad.shape != p.shape:
                raise RuntimeError(
                    "Split-aware distributed split-QKV requires the momentum and parameter "
                    f"layouts to match, got grad={tuple(grad.shape)} and p={tuple(p.shape)}"
                )
            return run_serially(
                self._split_qkv_distributed_chain(
                    grad,
                    gtp_remat_group,
                    tp_group,
                    partition_dim,
                    pad_length,
                    qkv_split_shapes,
                    split_modes,
                )
            )

        if qkv_split_shapes is not None and mode != "duplicated":
            # Fall back, not raise: these configs ran before the qkv fix, so an abort
            # here would break live jobs.
            if not self._warned_qkv_split_disabled:
                self._warned_qkv_split_disabled = True
                debug_name = getattr(p, '_debug_name', '')
                log_single_rank(
                    logger,
                    logging.WARNING,
                    f"Muon: split-QKV is disabled for GTP-sharded qkv weights under "
                    f"tp_mode='{mode}'"
                    + (f" (auto-resolved from '{self.tp_mode}')" if self.tp_mode == "auto" else "")
                    + (f", first seen on {debug_name}" if debug_name else "")
                    + "; blockwise/distributed keep the rows sharded, so the q/k/v "
                    "boundaries are not available on any rank. These weights get "
                    "whole-matrix Newton-Schulz, which is NOT the update rule the same "
                    "model gets at TP1. Pass --muon-tp-mode duplicated for a "
                    "layout-invariant qkv update, or --muon-no-split-qkv to make the "
                    "whole-matrix rule explicit at every layout.",
                )
            qkv_split_shapes = None

        if mode == "blockwise":
            # Local block NS on this rank's GTP row-shard (shape [M/gtp_remat_size, K]):
            # partition_dim=None makes scaled_orthogonalize_fn run a plain Newton-Schulz on
            # the shard with no GTP/TP collective. pad_length can exceed one shard's row
            # count, so only the overlap between this rank's shard and the trailing padded
            # rows of the full tensor is this rank's own padding.
            shard_size = grad.size(0)
            ranks_from_end = gtp_remat_size - 1 - gtp_rank
            local_pad_length = min(shard_size, max(0, pad_length - ranks_from_end * shard_size))
            if local_pad_length == shard_size:
                # Entirely padding: grad is exact zero, and NS(0) = 0.
                return torch.zeros_like(grad)
            result = self.scaled_orthogonalize_fn(
                self._strip_pad(grad, local_pad_length), tp_group, None, tp_mode_this_group=mode
            )
            return self._restore_pad(result, local_pad_length)

        if mode == "duplicated":
            # All-gather over GTP (dim 0), strip/restore padding exactly (every rank now
            # holds the same padded tensor), orthogonalize the whole matrix
            # (scaled_orthogonalize_fn handles any TP sharding per tp_mode), reshard dim 0.
            gathered_grad = self._all_gather_tensor(grad, gtp_remat_group, 0)
            result = scaled_orthogonalize_fn_with_qkv_split(
                self._strip_pad(gathered_grad, pad_length)
            )
            result = self._restore_pad(result, pad_length)
            reshard_size = result.size(0) // gtp_remat_size
            return result[gtp_rank * reshard_size : (gtp_rank + 1) * reshard_size].contiguous()

        # distributed: NS via the small-Gram all-reduce (no redundant full-matrix NS).
        # QKV splitting has already fallen back to this unsplit fused layout above, so
        # pad_length describes grad's full dim-0 layout here. Keep the physical rows for
        # the collective, but exclude them from Muon's shape-based scale.
        scale_shape = None
        if pad_length:
            if grad.shape != p.shape:
                raise RuntimeError(
                    "Distributed GTP Muon padding requires the momentum and parameter "
                    f"layouts to match, got grad={tuple(grad.shape)} and p={tuple(p.shape)}"
                )
            size = [grad.size(-2) * gtp_remat_size - pad_length, grad.size(-1)]
            if partition_dim is not None:
                size[partition_dim] *= get_pg_size(tp_group)
            scale_shape = (size[0], size[1])

        # A momentum with both TP and GTP as sharding axes takes two communication steps: an
        # all-gather that eliminates one axis, then the Gram all-reduce that distributes NS over
        # the other. With GTP as the only sharding axis, the Gram all-reduce is the only
        # communication needed. partition_dim is what says whether TP is a sharding axis here,
        # the same signal scaled_orthogonalize_fn and newton_schulz_tp key off.
        needs_two_step_communication = (
            partition_dim is not None and tp_group is not None and get_pg_size(tp_group) > 1
        )

        if not needs_two_step_communication:
            # GTP is the only sharding axis: distribute NS over it on the local dim-0 row shard.
            return self.scaled_orthogonalize_fn(
                grad,
                gtp_remat_group,
                partition_dim=0,
                tp_mode_this_group=mode,
                scale_shape=scale_shape,
            )

        # GTP + TP: distributed NS can only operate over one (group, dim) at a
        # time. Distribute over the larger group so that the NS GEMMs are sharded
        # across more ranks (less redundant compute), and all-gather the smaller
        # group to eliminate its sharding beforehand.
        tp_size = get_pg_size(tp_group)
        if gtp_remat_size >= tp_size:
            smaller_group, smaller_dim = tp_group, partition_dim
            larger_group, larger_dim = gtp_remat_group, 0
        else:
            smaller_group, smaller_dim = gtp_remat_group, 0
            larger_group, larger_dim = tp_group, partition_dim

        gathered_grad = self._all_gather_tensor(grad, smaller_group, smaller_dim)
        orthogonalized_grad = self.scaled_orthogonalize_fn(
            gathered_grad,
            larger_group,
            larger_dim,
            tp_mode_this_group=mode,
            scale_shape=scale_shape,
        )
        shard_size = orthogonalized_grad.size(smaller_dim) // get_pg_size(smaller_group)
        reshard_rank = get_pg_rank(smaller_group)
        return orthogonalized_grad.narrow(
            smaller_dim, reshard_rank * shard_size, shard_size
        ).contiguous()

    def orthogonalize(self, p: torch.Tensor, grad: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Orthogonalize the momentum.

        Args:
            p: The parameter tensor. It is necessary to pass param tensor in addition to
                momentum because a lot of information is only available in the param tensor,
                attributes for example.
            grad: The momentum tensor.

        Returns:
            The orthogonalized gradient tensor.
        """
        # TODO(deyuf): switch to group
        if self.pg_collection:
            tp_group = (
                self.pg_collection.expt_tp
                if getattr(p, 'expert_tp', False)
                else self.pg_collection.tp
            )
        else:
            tp_group = None
        partition_dim = None if self.tp_mode == "blockwise" else getattr(p, "partition_dim", None)
        if partition_dim == -1:
            partition_dim = None

        # Look up the shapes only. `grad` is still this rank's row shard here, so the
        # split has to wait until the callee has gathered the whole matrix.
        qkv_split_shapes = None
        if self.split_qkv and self.is_qkv_fn(p):  # type: ignore[misc]
            qkv_split_shapes = getattr(p, "qkv_split_shapes", None)
            if qkv_split_shapes is None:
                qkv_split_shapes = self.qkv_split_shapes
            if not qkv_split_shapes:
                raise RuntimeError("Muon QKV split requested but qkv_split_shapes is not set")
        grad = self.scaled_orthogonalize_fn_with_gtp_remat(
            p, grad, tp_group, partition_dim, qkv_split_shapes=qkv_split_shapes
        )
        return grad


class TensorParallelAdaptiveMuon(TensorParallelMuon, AdaptiveMuon):
    """Tensor Parallel Adaptive Muon optimizer.

    This class extends Muon by adding AdamW-style or NorMuon-style second moment
    accumulation after orthogonalization. This idea was first explored in D.E. Carlson,
    E. Collins, Ya-Ping Hsieh, L. Carin, and V. Cevher. *Preconditioned spectral
    descent for deep learning.* In Advances in neural information processing systems 28 (2015).
    The step() method is overridden to include second moment normalization logic.

    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups.
        lr: Learning rate.
        momentum: The exponential decay rate for momentum.
        nesterov: Whether to use Nesterov momentum.
        weight_decay: Weight decay coefficient.
        use_decoupled_weight_decay: Whether to use decoupled weight decay.
        split_qkv: Whether to split QKV weights for orthogonalization.
        is_qkv_fn: Function to determine if a tensor is a QKV weight.
        qkv_split_shapes: Shapes for splitting QKV weights.
        fp32_matmul_prec: Precision for FP32 matrix multiplication.
        coefficient_type: The type of coefficient set to use for the Newton-Schulz iteration.
        num_ns_steps: The number of iteration steps to use in the Newton-Schulz iteration.
        scale_mode: The type of scale factor to use for the update.
        extra_scale_factor: The additional scale factor to use for the update.
        pg_collection: Process group collection for distributed training.
        tp_mode: Tensor parallel mode ("blockwise", "duplicated", "distributed", or "auto").
        use_syrk: Whether to use the Triton SYRK kernel for the Gram matrix in
            Newton-Schulz. Requires emerging_optimizers >= 0.4.0.
        moment2_method: Method for second moment accumulation ("adamuon" or "normuon").
        beta2: The exponential decay rate for second moment.
        eps: Small constant for numerical stability.
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 3e-4,
        momentum: float = 0.95,
        nesterov: bool = True,
        weight_decay: float = 0.01,
        use_decoupled_weight_decay: bool = True,
        split_qkv: bool = False,
        is_qkv_fn: Callable[[torch.Tensor], bool] | None = None,
        qkv_split_shapes: list[int] | None = None,
        fp32_matmul_prec: str = "medium",
        coefficient_type: str = "quintic",
        num_ns_steps: int = 5,
        scale_mode: str = "spectral",
        extra_scale_factor: float = 1.0,
        pg_collection: Optional[ProcessGroupCollection] = None,
        tp_mode: Literal["blockwise", "duplicated", "distributed", "auto"] = "duplicated",
        use_syrk: bool = False,
        moment2_method: Literal["adamuon", "normuon"] = "adamuon",
        beta2: float = 0.95,
        eps: float = 1e-8,
    ) -> None:
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
        self.scale_mode = scale_mode
        self.extra_scale_factor = extra_scale_factor
        self.moment2_method = moment2_method

        for group in self.param_groups:
            group.setdefault("beta2", beta2)
            group.setdefault("eps", eps)

    @torch.no_grad()  # type: ignore[misc]
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """Step function"""
        return AdaptiveMuon.step(self, closure)


def _kwargs_from_config(optimizer_cls: type, prefix: str, config) -> Dict[str, Any]:
    """Match ``optimizer_cls.__init__`` parameters to config attributes.

    For each init parameter, looks for ``{prefix}_{name}`` on *config* first,
    then falls back to ``{name}`` (unprefixed).  ``self`` and ``params`` are
    always skipped.
    """
    skip_params = {"self", "params"}
    sig = inspect.signature(optimizer_cls.__init__)
    kwargs: Dict[str, Any] = {}
    for name in sig.parameters:
        if name in skip_params:
            continue
        prefixed = f"{prefix}_{name}"
        if hasattr(config, prefixed):
            kwargs[name] = getattr(config, prefixed)
        elif hasattr(config, name):
            kwargs[name] = getattr(config, name)
    return kwargs


def _muon_config_to_cls(config) -> type:
    """The ``muon`` registry entry's class.

    ``LayerShardedMuon`` when ``muon_tp_mode == 'layer_sharded'`` (a registry-level class
    selector, not a TensorParallelMuon runtime mode), ``TensorParallelMuon`` otherwise.
    """
    if getattr(config, 'muon_tp_mode', 'duplicated') == 'layer_sharded':
        from megatron.core.optimizer.layer_sharded_muon import LayerShardedMuon

        return LayerShardedMuon
    return TensorParallelMuon


def _muon_config_to_kwargs(config, model_chunks, pg_collection) -> Dict[str, Any]:
    """Convert OptimizerConfig to TensorParallelMuon constructor kwargs.

    Shared by ``adaptive_muon`` (which layers its own kwargs on top), so it never
    dispatches on ``muon_tp_mode='layer_sharded'``; that lives in
    :func:`_muon_registry_config_to_kwargs`.
    """
    kwargs = _kwargs_from_config(TensorParallelMuon, "muon", config)
    kwargs["is_qkv_fn"] = lambda p: getattr(p, "is_qkv", False)
    kwargs["qkv_split_shapes"] = _get_qkv_split_shapes(model_chunks[0].config)
    kwargs["pg_collection"] = pg_collection
    return kwargs


def _muon_registry_config_to_kwargs(config, model_chunks, pg_collection) -> Dict[str, Any]:
    """``config_to_kwargs`` for the ``muon`` registry entry.

    TensorParallelMuon kwargs, plus LayerShardedMuon's own when
    :func:`_muon_config_to_cls` selects it (the same helper the entry's
    ``config_to_cls`` uses, so class and kwargs cannot disagree).
    """
    kwargs = _muon_config_to_kwargs(config, model_chunks, pg_collection)
    cls = _muon_config_to_cls(config)
    if cls is TensorParallelMuon:
        return kwargs
    # LayerShardedMuon: its own muon-prefixed kwargs (ns_batch_size, concurrent_groups, ...).
    kwargs.update(_kwargs_from_config(cls, "muon", config))
    # 'layer_sharded' selected the class; it is not a TensorParallelMuon mode, so the
    # delegated (empty-homes fallback) path runs the bitwise reference mode instead.
    kwargs["tp_mode"] = "duplicated"
    # No config attr for these: the (gtp_remat, tp) axes come from the collection.
    kwargs["gtp_remat_group"] = getattr(pg_collection, "gtp_remat", None)
    kwargs["tp_group"] = getattr(pg_collection, "tp", None)
    return kwargs


def _adaptive_muon_config_to_kwargs(config, model_chunks, pg_collection) -> Dict[str, Any]:
    """Convert OptimizerConfig to TensorParallelAdaptiveMuon constructor kwargs."""
    kwargs = _muon_config_to_kwargs(config, model_chunks, pg_collection)
    kwargs.update(_kwargs_from_config(TensorParallelAdaptiveMuon, "adaptive_muon", config))
    return kwargs


def _default_adam_based_eopt_config_to_kwargs(
    eopt_name, config, model_chunks, pg_collection
) -> Dict[str, Any]:
    """Convert OptimizerConfig to default emerging optimizer constructor kwargs."""
    kwargs = _kwargs_from_config(registry.get_optimizer_cls(eopt_name), eopt_name, config)
    kwargs["betas"] = (config.adam_beta1, config.adam_beta2)
    return kwargs


# -----------------------------------------------------------------------
# Register emerging optimizers
# -----------------------------------------------------------------------
_EMERGING_OPTIMIZERS.update(
    {
        'muon': EmergingOptimizerEntry(
            optimizer_cls=TensorParallelMuon,
            init_state_fn=_eopt_init_state_fn,
            config_to_kwargs=_muon_registry_config_to_kwargs,
            config_to_cls=_muon_config_to_cls,
            default_param_overrides={
                ParamKey(predicate=ParamPredicate(name="muon_excluded", fn=_is_muon_excluded)): {
                    'optimizer': 'adam'
                }
            },
        ),
        "adaptive_muon": EmergingOptimizerEntry(
            optimizer_cls=TensorParallelAdaptiveMuon,
            init_state_fn=_eopt_init_state_fn,
            config_to_kwargs=_adaptive_muon_config_to_kwargs,
            default_param_overrides={
                ParamKey(predicate=ParamPredicate(name="muon_excluded", fn=_is_muon_excluded)): {
                    'optimizer': 'adam'
                }
            },
        ),
    }
)

# Register soap with default config
# TODO(skyw): register all emerging optimizers.
if HAVE_EMERGING_OPTIMIZERS:
    for eopt_name in registry.get_optimizer_name_list():
        if eopt_name in _EMERGING_OPTIMIZERS:
            # skip already registered local versions, e.g. TensorParallel versions.
            continue
        _EMERGING_OPTIMIZERS[eopt_name] = EmergingOptimizerEntry(
            optimizer_cls=registry.get_optimizer_cls(eopt_name)
        )
