# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Emerging optimizer registry.

To add a new emerging optimizer:
  1. Define its optimizer class (or import it).
  2. Write its ``_<name>_init_state_fn`` and ``_<name>_config_to_kwargs``.
  3. Add an ``EmergingOptimizerEntry`` to ``_EMERGING_OPTIMIZERS`` at the bottom.
"""

import inspect
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Literal, Optional, get_args

import torch
from torch.optim.optimizer import ParamsT

from megatron.core.process_groups_config import ProcessGroupCollection
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
        optimizer_cls: The torch optimizer class.
        init_state_fn: Lazily initialises optimizer state (needed for checkpoint formats).
        config_to_kwargs: ``(config, model_chunks, pg_collection) -> dict`` of constructor kwargs.
        default_param_overrides: Per-parameter config overrides applied automatically
            (e.g. route non-linear params to Adam).
    """

    optimizer_cls: type
    init_state_fn: Callable = _eopt_init_state_fn
    config_to_kwargs: Callable | None = None
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
    optimizer = entry.optimizer_cls(param_groups, **eopt_kwargs)
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
        split_glu: bool = False,
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
        ) -> torch.Tensor:
            log_single_rank(
                logger,
                logging.DEBUG,
                f'Orthogonalizing grad with {num_ns_steps} steps, '
                f'{coefficient_type} coefficient, '
                f'{scale_mode} scale mode, extra_scale_factor={extra_scale_factor}',
            )
            size = [grad.size(-2), grad.size(-1)]
            if partition_dim is not None:
                size[partition_dim] *= get_pg_size(tp_group)
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
        self.split_glu = split_glu

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

    def _get_gtp_remat_group(self, p):
        """Return the weight-rematerialization group for ``p`` when configured."""
        if self.pg_collection is None:
            return None
        group_name = 'expt_gtp_remat' if getattr(p, 'expert_tp', False) else 'gtp_remat'
        return getattr(self.pg_collection, group_name, None)

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

    def scaled_orthogonalize_fn_with_gtp_remat(self, p, grad, tp_group, partition_dim):
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
        distributed does not: it stays row-sharded through its own collective, where
        stripping isn't safe (known limitation).
        """
        gtp_remat_group = self._get_gtp_remat_group(p)

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
        if mode == "auto":
            # Scoped to dense (GTP) weights for now; expert weights keep today's default.
            mode = (
                self._resolve_tp_mode(p.shape[0] * gtp_remat_size, p.shape[1], gtp_remat_size)
                if gtp_active and not is_expert
                else "duplicated"
            )

        if not gtp_active:
            return self.scaled_orthogonalize_fn(
                grad, tp_group, partition_dim, tp_mode_this_group=mode
            )

        gtp_rank = get_pg_rank(gtp_remat_group)
        pad_length = getattr(p, 'pad_length', 0)

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
            result = self.scaled_orthogonalize_fn(
                self._strip_pad(gathered_grad, pad_length),
                tp_group,
                partition_dim,
                tp_mode_this_group=mode,
            )
            result = self._restore_pad(result, pad_length)
            reshard_size = result.size(0) // gtp_remat_size
            return result[gtp_rank * reshard_size : (gtp_rank + 1) * reshard_size].contiguous()

        # distributed: NS via the small-Gram all-reduce (no redundant full-matrix NS).
        # A momentum with both TP and GTP as sharding axes takes two communication steps: an
        # all-gather that eliminates one axis, then the Gram all-reduce that distributes NS over
        # the other. With GTP as the only sharding axis, the Gram all-reduce is the only
        # communication needed. partition_dim is what says whether TP is a sharding axis here,
        # the same signal scaled_orthogonalize_fn and newton_schulz_tp key off.
        #
        # GTP_remat's alignment padding is not corrected for here -- see the class docstring.
        needs_two_step_communication = (
            partition_dim is not None and tp_group is not None and get_pg_size(tp_group) > 1
        )

        if not needs_two_step_communication:
            # GTP is the only sharding axis: distribute NS over it on the local dim-0 row shard.
            return self.scaled_orthogonalize_fn(
                grad, gtp_remat_group, partition_dim=0, tp_mode_this_group=mode
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
            gathered_grad, larger_group, larger_dim, tp_mode_this_group=mode
        )
        shard_size = orthogonalized_grad.size(smaller_dim) // get_pg_size(smaller_group)
        reshard_rank = get_pg_rank(smaller_group)
        return orthogonalized_grad.narrow(
            smaller_dim, reshard_rank * shard_size, shard_size
        ).contiguous()

    def _gather_glu_grad(self, p, grad):
        """Gather a GTP-local GLU shard and remove tail padding before layout transforms."""
        if not getattr(p, 'is_gtp_weight_remat', False):
            return grad, None

        gtp_remat_group = self._get_gtp_remat_group(p)
        expected_size = getattr(p, 'gtp_remat_size', 1)
        if gtp_remat_group is None:
            if expected_size > 1:
                raise RuntimeError(
                    "Muon GLU split requires the GTP-remat process group to reconstruct "
                    f"a {expected_size}-way weight shard"
                )
            gtp_remat_size = 1
            gtp_rank = 0
            gathered_grad = grad
        else:
            gtp_remat_size = get_pg_size(gtp_remat_group)
            gtp_rank = get_pg_rank(gtp_remat_group)
            if gtp_remat_size != expected_size:
                raise RuntimeError(
                    "Muon GLU split GTP size mismatch: "
                    f"parameter metadata={expected_size}, process group={gtp_remat_size}"
                )
            if gtp_remat_size > 1:
                shards = [torch.empty_like(grad) for _ in range(gtp_remat_size)]
                torch.distributed.all_gather(shards, grad, gtp_remat_group)
                gathered_grad = torch.cat(shards, dim=0)
            else:
                gathered_grad = grad

        pad_length = getattr(p, 'pad_length', 0)
        if pad_length < 0 or pad_length >= gathered_grad.shape[0]:
            raise RuntimeError(
                f"Muon GLU split has invalid GTP padding: grad_shape={tuple(gathered_grad.shape)}, "
                f"pad_length={pad_length}"
            )
        logical_grad = gathered_grad[:-pad_length] if pad_length else gathered_grad
        restore_info = (gtp_rank, gtp_remat_size, grad.shape[0], pad_length)
        return logical_grad, restore_info

    @staticmethod
    def _restore_local_glu_grad(grad, restore_info):
        """Restore GTP tail padding and return this rank's original dim-0 shard."""
        if restore_info is None:
            return grad
        gtp_rank, gtp_remat_size, shard_size, pad_length = restore_info
        if pad_length:
            grad = torch.nn.functional.pad(grad, (0, 0, 0, pad_length))
        expected_rows = gtp_remat_size * shard_size
        if grad.shape[0] != expected_rows:
            raise RuntimeError(
                f"Muon GLU GTP restore shape mismatch: grad_shape={tuple(grad.shape)}, "
                f"expected_rows={expected_rows}"
            )
        start = gtp_rank * shard_size
        return grad[start : start + shard_size].contiguous()

    @staticmethod
    def _deinterleave_glu_grad(grad, interleave_size):
        """Convert alternating gate/up blocks to contiguous gate and up halves."""
        if interleave_size is None:
            return grad
        if interleave_size <= 0 or grad.shape[0] % (2 * interleave_size) != 0:
            raise RuntimeError(
                f"Muon GLU interleave shape mismatch: grad_shape={tuple(grad.shape)}, "
                f"interleave_size={interleave_size}"
            )
        num_blocks = grad.shape[0] // (2 * interleave_size)
        return (
            grad.view(num_blocks, 2, interleave_size, grad.shape[1])
            .transpose(0, 1)
            .contiguous()
            .view_as(grad)
        )

    @staticmethod
    def _interleave_glu_grad(grad, interleave_size):
        """Restore alternating gate/up blocks after separate orthogonalization."""
        if interleave_size is None:
            return grad
        num_blocks = grad.shape[0] // (2 * interleave_size)
        return (
            grad.view(2, num_blocks, interleave_size, grad.shape[1])
            .transpose(0, 1)
            .contiguous()
            .view_as(grad)
        )

    def _orthogonalize_split_glu(self, p, grad, tp_group, partition_dim):
        """Orthogonalize gate and up matrices independently for all supported GLU layouts."""
        gathered_grad, restore_info = self._gather_glu_grad(p, grad)
        if gathered_grad.ndim != 2 or gathered_grad.shape[0] % 2 != 0:
            raise RuntimeError(
                f"Muon GLU split requires a 2D gradient with even rows, got "
                f"shape={tuple(gathered_grad.shape)}"
            )

        interleave_size = getattr(p, 'glu_interleave_size', None)
        contiguous_grad = self._deinterleave_glu_grad(gathered_grad, interleave_size)
        split_size = contiguous_grad.shape[0] // 2
        log_single_rank(
            logger,
            logging.DEBUG,
            f'glu split grad shape {tuple(contiguous_grad.shape)}, '
            f'split shapes {[split_size, split_size]}, interleave size {interleave_size}',
        )
        gate_grad, up_grad = torch.split(contiguous_grad, split_size, dim=0)
        gate_grad = self.scaled_orthogonalize_fn(gate_grad, tp_group, partition_dim)
        up_grad = self.scaled_orthogonalize_fn(up_grad, tp_group, partition_dim)
        orthogonalized_grad = torch.cat((gate_grad, up_grad), dim=0)
        orthogonalized_grad = self._interleave_glu_grad(orthogonalized_grad, interleave_size)
        return self._restore_local_glu_grad(orthogonalized_grad, restore_info)

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

        if self.split_qkv and self.is_qkv_fn(p):  # type: ignore[misc]
            grad_shape = grad.shape
            qkv_split_shapes = getattr(p, "qkv_split_shapes", None)
            if qkv_split_shapes is None:
                qkv_split_shapes = self.qkv_split_shapes
            if qkv_split_shapes is None:
                raise RuntimeError("Muon QKV split requested but qkv_split_shapes is not set")
            qkv_split_dim = sum(qkv_split_shapes)
            if grad_shape[0] % qkv_split_dim != 0:
                raise RuntimeError(
                    f"Muon QKV split shape mismatch: grad_shape={tuple(grad_shape)}, "
                    f"split_shapes={qkv_split_shapes}"
                )
            log_single_rank(
                logger,
                logging.DEBUG,
                f'qkv split grad shape {grad_shape}, split shapes {qkv_split_shapes}',
            )
            num_query_groups = grad_shape[0] // qkv_split_dim
            qkv_grads = torch.split(
                grad.view(num_query_groups, qkv_split_dim, -1), qkv_split_shapes, dim=1
            )
            qkv_grads = [g.reshape(-1, grad_shape[-1]) for g in qkv_grads]

            qkv_grads = [
                self.scaled_orthogonalize_fn_with_gtp_remat(p, g, tp_group, partition_dim).view(
                    num_query_groups, -1, grad_shape[-1]
                )
                for g in qkv_grads
            ]
            grad = torch.cat(qkv_grads, dim=1).view(grad_shape)
        elif self.split_glu and getattr(p, "is_glu", False):
            grad = self._orthogonalize_split_glu(p, grad, tp_group, partition_dim)
        else:
            grad = self.scaled_orthogonalize_fn_with_gtp_remat(p, grad, tp_group, partition_dim)
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
        split_glu: Whether to split fused GLU FC1 weights for orthogonalization.
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
        split_glu: bool = False,
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
            split_glu=split_glu,
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


def _muon_config_to_kwargs(config, model_chunks, pg_collection) -> Dict[str, Any]:
    """Convert OptimizerConfig to TensorParallelMuon constructor kwargs."""
    kwargs = _kwargs_from_config(TensorParallelMuon, "muon", config)
    kwargs["is_qkv_fn"] = lambda p: getattr(p, "is_qkv", False)
    kwargs["qkv_split_shapes"] = _get_qkv_split_shapes(model_chunks[0].config)
    kwargs["pg_collection"] = pg_collection
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
            config_to_kwargs=_muon_config_to_kwargs,
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
