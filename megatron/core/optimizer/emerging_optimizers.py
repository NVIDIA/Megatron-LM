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
from typing import Any, Callable, Dict, List, Literal, Optional, get_args

import torch
from torch.optim.optimizer import ParamsT

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.utils import get_pg_rank, get_pg_size, log_single_rank

from .optimizer_config import ParamKey, ParamPredicate

try:
    from torch.distributed.tensor import DTensor as _DTensor

    _HAVE_DTENSOR = True
except ImportError:
    _DTensor = None  # type: ignore[assignment,misc]
    _HAVE_DTENSOR = False

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


def _get_qkv_split_shapes(model_cfg) -> List[int]:
    """Compute QKV split shapes from model config."""
    return [
        model_cfg.num_attention_heads // model_cfg.num_query_groups * model_cfg.kv_channels,
        model_cfg.kv_channels,
        model_cfg.kv_channels,
    ]


# ===========================================================================
# Registry – populated below only when emerging_optimizers is installed.
# ===========================================================================

_EMERGING_OPTIMIZERS: Dict[str, EmergingOptimizerEntry] = {}


# ===========================================================================
# Megatron-FSDP integration
# ===========================================================================


def _get_mfsdp_models(model_chunks):
    """Extract list of MegatronFSDP instances from FSDP-wrapped model chunks."""
    mfsdp_models = []
    for chunk in model_chunks:
        # FullyShardedDataParallel delegates finish_grad_sync / start_param_sync
        # from its .module (MegatronFSDP). install_optimized_model_weights lives
        # directly on MegatronFSDP, so we need the inner module reference.
        if hasattr(chunk, "finish_grad_sync") and hasattr(chunk, "module"):
            mfsdp_models.append(chunk.module)
    if not mfsdp_models:
        raise RuntimeError(
            "Could not find any MegatronFSDP instances in model_chunks. "
            "Ensure the model is wrapped with FullyShardedDataParallel."
        )
    return mfsdp_models


class FSDPMuonChainedOptimizer:
    """Thin FSDP-protocol adapter wrapping a Muon-based MegatronOptimizer.

    Injects the MegatronFSDP step contract around the inner optimizer:
      1. `finish_grad_sync()` – waits for async grad sync, attaches grads
         (allreduces for `no_shard`; reduce-scatters for ZeRO-1/2/3).
      2. `inner_optimizer.step()` – Muon NS + weight update + Adam.
      3. `install_optimized_model_weights()` – copies fp32 main weights
         back into the model's bf16 (sharded for ZeRO-3) buffer.

    All other attribute accesses are delegated to the inner optimizer via
    `__getattr__`, making this class transparent to the training loop.
    """

    def __init__(self, inner: Any, mfsdp_models: list) -> None:
        # Use object.__setattr__ to avoid triggering our own __getattr__ during init.
        object.__setattr__(self, "inner", inner)
        object.__setattr__(self, "_mfsdp_models", mfsdp_models)

    @torch.no_grad()  # type: ignore[misc]
    def step(self) -> Any:
        """FSDP-aware optimizer step: sync grads -> inner step -> install weights."""
        for mfsdp in self._mfsdp_models:
            if not mfsdp.model_auto_sync:
                mfsdp.finish_grad_sync()
        result = self.inner.step()
        for mfsdp in self._mfsdp_models:
            mfsdp.install_optimized_model_weights()
        return result

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Zero optimizer gradients. FSDP grad buffer is zeroed by the training loop."""
        self.inner.zero_grad(set_to_none)

    def __getattr__(self, name: str) -> Any:
        """Delegate all other attribute accesses to the inner optimizer."""
        return getattr(object.__getattribute__(self, "inner"), name)


# ===========================================================================
# Muon
# ===========================================================================


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
        qkv_split_shapes: tuple[int, int, int] | None = None,
        fp32_matmul_prec: str = "medium",
        coefficient_type: str = "quintic",
        num_ns_steps: int = 5,
        scale_mode: str = "spectral",
        extra_scale_factor: float = 1.0,
        pg_collection: Optional[ProcessGroupCollection] = None,
        tp_mode: Literal["blockwise", "duplicated", "distributed"] = "duplicated",
    ) -> None:
        if num_ns_steps < 1:
            raise ValueError(f"num_ns_steps must be at least 1, got {num_ns_steps}")

        def scaled_orthogonalize_fn(
            grad: torch.Tensor,
            tp_group: torch.distributed.ProcessGroup,
            partition_dim: int | None = None,
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
            orth_grad = newton_schulz_tp(
                grad,
                steps=num_ns_steps,
                coefficient_type=coefficient_type,
                tp_group=tp_group,
                partition_dim=partition_dim,
                tp_mode="duplicated" if tp_mode == "blockwise" else tp_mode,
            )
            scale_factor = get_muon_scale_factor(size[0], size[1], mode=scale_mode)
            return orth_grad * scale_factor * extra_scale_factor

        self.pg_collection = pg_collection
        self.tp_mode = tp_mode
        self.split_qkv = split_qkv
        self.is_qkv_fn = is_qkv_fn
        self.qkv_split_shapes = qkv_split_shapes

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

    def orthogonalize(self, p: torch.Tensor, grad: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Orthogonalize the momentum.

        Args:
            p: The parameter tensor. i is necessary to pass param tensor in addition to
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
            log_single_rank(
                logger,
                logging.DEBUG,
                f'qkv split grad shape {grad_shape}, ' f'split shapes {self.qkv_split_shapes}',
            )
            num_query_groups = grad_shape[0] // sum(self.qkv_split_shapes)
            qkv_grads = torch.split(
                grad.view(num_query_groups, sum(self.qkv_split_shapes), -1),
                self.qkv_split_shapes,
                dim=1,
            )
            qkv_grads = [g.reshape(-1, grad_shape[-1]) for g in qkv_grads]

            qkv_grads = [
                self.scaled_orthogonalize_fn(g, tp_group, partition_dim).view(
                    num_query_groups, -1, grad_shape[-1]
                )
                for g in qkv_grads
            ]
            grad = torch.cat(qkv_grads, dim=1).view(grad_shape)
        else:
            grad = self.scaled_orthogonalize_fn(grad, tp_group, partition_dim)
        return grad


class FSDPZeROTensorParallelMuon(TensorParallelMuon):
    """`TensorParallelMuon` extended for Megatron-FSDP ZeRO-1/2/3.

    Supports all three sharded strategies:

    * `optim` (ZeRO-1): optimizer state sharded; grads reduce-scattered.
    * `optim_grads` (ZeRO-2): optimizer state + grads sharded.
    * `optim_grads_params` (ZeRO-3): optimizer state + grads + params sharded.

    For all three, `finish_grad_sync()` reduce-scatters gradients so each DP
    rank holds only a contiguous `Shard(0)` row-shard of the full TP-local 2D
    gradient. Newton-Schulz needs the full matrix, so this class:

      1. Extracts the local shard via `.to_local()` for any Shard(0) DTensor.
      2. Allgathers the DP row-shards across the DP group to reconstruct the
         TP-local, DP-full gradient matrix.
      3. Trims FSDP bucket-padding rows using the declared global shape from the
         DTensor.
      4. Delegates to `TensorParallelMuon.orthogonalize` which handles the
         remaining TP dimension via `newton_schulz_tp`.
      5. Extracts the local DP row-shard of the orthogonalized result, padding
         the last rank with zeros when FSDP bucket padding is present so the
         per-rank shard size stays uniform.
      6. Re-wraps the result as a DTensor matching the input's placements so
         `OrthogonalizedOptimizer.step` can apply the in-place update without
         placement promotion.

    Memory note for ZeRO-3: each Muon step allocates a temporary `full_grad`
    of shape `(tp_local_rows, C)` per Muon parameter for the allgather, plus
    the NS output. Peak extra memory is roughly
    `3 * (m * n) * sizeof(float32)` per linear layer; momentum stays sharded.

    For "no_shard" or single-rank DP this falls back transparently to the
    parent implementation.
    """

    def __init__(
        self,
        params: ParamsT,
        dp_group: Optional[torch.distributed.ProcessGroup] = None,
        **kwargs: Any,
    ) -> None:
        self.dp_group = dp_group
        super().__init__(params, **kwargs)

    @torch.no_grad()  # type: ignore[misc]
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """Muon step for Megatron-FSDP ZeRO-1/2/3.

        M-FSDP shards a flat grad buffer across DP, so the per-param local
        shard can be empty on some ranks (`p.grad is None`) or variable in
        size – unlike a uniform `Shard(0)` DTensor. The standard
        `OrthogonalizedOptimizer.step` skips params with `grad is None`,
        which desyncs the DP collective inside `orthogonalize` (different
        ranks process different params → allgather mismatch → hang).

        We fix that here by processing every param on every rank in lockstep.
        Each param's full gradient is reconstructed via a DP-wide all-gather
        (supports variable-size local shards), but Newton-Schulz is run on a
        single owner rank (round-robin by param index) and broadcast back,
        so NS compute is amortised across DP rather than duplicated 8×.
        """
        if closure is None:
            loss = None
        else:
            loss = closure()

        for group in self.param_groups:
            # `skip_non_grad_params=False`: we must create momentum buffers
            # for every param, because even params with empty local grad shards
            # on this rank need to participate in the DP-collective orthogonalize.
            self._init_group(group, skip_non_grad_params=False)

            for p in group["params"]:
                self._muon_step_one(p, group)

        return loss

    @torch.no_grad()  # type: ignore[misc]
    def _muon_step_one(self, p: torch.Tensor, group: Dict[str, Any]) -> None:
        """Per-param Muon update that participates in DP collectives on every rank."""
        # Fall back to the plain parent step for the single-rank / unsharded case.
        if self.dp_group is None or get_pg_size(self.dp_group) == 1:
            if p.grad is None:
                return
            return self._local_muon_update(p, p.grad, group)

        # Skip params that aren't DTensors (shouldn't happen in M-FSDP).
        if not (_HAVE_DTENSOR and isinstance(p, _DTensor)):
            if p.grad is None:
                return
            return self._local_muon_update(p, p.grad, group)

        # Every rank processes every param in lockstep (regardless of whether
        # the local grad shard is empty) so the DP collectives below stay in
        # sync. M-FSDP's flat-buffer sharding means local shard sizes vary
        # per rank and per param, so we operate on DTensors and use
        # `full_tensor()` to reconstruct the full matrix.

        # Use local views to mutate momentum and param. Momentum is kept
        # per-shard (same size as p.to_local()); the initial buffer was
        # allocated via `torch.zeros_like(p.data)` in `_init_group` which,
        # for a DTensor p, yields a Shard(0) DTensor with the correct local
        # shape on this rank.
        state = self.state[p]
        mom_buffer = state["momentum_buffer"]
        mom_local = mom_buffer.to_local() if isinstance(mom_buffer, _DTensor) else mom_buffer

        # Derive this rank's local grad. If p.grad is None (empty shard on
        # this rank), use a zero tensor of the expected local shape so the
        # local lerp_ / weight-decay math is a no-op while still letting us
        # participate in the full_tensor() allgather below.
        grad = p.grad
        if grad is None:
            local_grad = torch.zeros_like(mom_local)
        else:
            local_grad = grad.to_local() if isinstance(grad, _DTensor) else grad

        p_local = p.to_local()

        # Local weight decay on this rank's param/grad slice.
        wd = group["weight_decay"]
        lr = group["lr"]
        if wd != 0.0:
            # Decoupled weight decay: p *= (1 - lr * wd)
            p_local.mul_(1.0 - lr * wd)

        # Local momentum EMA update.
        mom_local.lerp_(local_grad, 1 - group["momentum"])

        # Assemble the pre-NS gradient (local).
        if self.nesterov:
            pre_local = local_grad.lerp(mom_local, group["momentum"])
        else:
            pre_local = mom_local

        # Gather per-rank local row counts so we can allocate a per-rank
        # destination for the allgather. M-FSDP uses a flat-buffer grad
        # sharding → local shard sizes vary per rank, which
        # `all_gather_into_tensor` (equal-size) and DTensor `full_tensor`
        # (assumes uniform Shard(0)) both handle poorly. We use
        # `all_gather(tensor_list, input)`, which supports variable sizes.
        dp_size = get_pg_size(self.dp_group)
        dp_rank = get_pg_rank(self.dp_group)
        row_counts = torch.zeros(dp_size, dtype=torch.long, device=p_local.device)
        row_counts[dp_rank] = pre_local.shape[0]
        torch.distributed.all_reduce(row_counts, group=self.dp_group)
        row_counts_cpu = row_counts.tolist()

        # Build per-rank output buffers of the right row count.
        gathered = [
            torch.empty(
                (row_counts_cpu[r], *pre_local.shape[1:]),
                device=pre_local.device,
                dtype=pre_local.dtype,
            )
            for r in range(dp_size)
        ]
        torch.distributed.all_gather(
            gathered, pre_local.contiguous(), group=self.dp_group
        )
        full_grad = torch.cat(gathered, dim=0)

        # Trim any trailing FSDP bucket padding rows so NS sees only the real
        # parameter. `p.shape[0]` is the FSDP-declared global row count
        # (= sum of per-rank shard rows, possibly including padding at the
        # tail); the true param row count is the global shape from the DTensor.
        tp_group = (
            (
                self.pg_collection.expt_tp
                if getattr(p, "expert_tp", False)
                else self.pg_collection.tp
            )
            if self.pg_collection
            else None
        )
        partition_dim = None if self.tp_mode == "blockwise" else getattr(p, "partition_dim", None)
        if partition_dim == -1:
            partition_dim = None
        tp_size_dim0 = (
            get_pg_size(tp_group) if (partition_dim == 0 and tp_group is not None) else 1
        )
        tp_local_rows = p.shape[0] // max(tp_size_dim0, 1)
        if full_grad.shape[0] > tp_local_rows:
            full_grad = full_grad[:tp_local_rows]

        # Newton-Schulz on the full matrix. We run it on every rank (duplicated
        # compute across DP) intentionally. A partition-and-broadcast scheme
        # was tried (owner does NS; others bcast) but the per-param serial
        # loop turns the broadcast into a per-param sync, negating the
        # parallelism across ranks – it measured slightly slower in practice.
        # An effective scheme would need to pipeline NS with the allgather/
        # broadcast for the *next* param; left as future work.
        from emerging_optimizers import utils

        with utils.fp32_matmul_precision(self.fp32_matmul_prec):
            group_kwargs = {k: v for k, v in group.items() if k != "params"}
            orth_full_grad = super(FSDPZeROTensorParallelMuon, self).orthogonalize(
                p, full_grad, **group_kwargs
            )

        # Slice out this rank's contribution. Exclusive prefix sum of
        # row_counts gives the offset; we saved it above.
        start = sum(row_counts_cpu[:dp_rank])
        end = start + row_counts_cpu[dp_rank]
        # Guard against the last rank holding only padding rows: clamp end.
        end = min(end, orth_full_grad.shape[0])
        orth_local = orth_full_grad[start:end].to(dtype=p_local.dtype)
        # If this rank's shard was purely padding (no real rows), orth_local
        # may be shorter than p_local; pad with zeros so the add_ lines up.
        if orth_local.shape[0] < p_local.shape[0]:
            pad = torch.zeros(
                (p_local.shape[0] - orth_local.shape[0], *p_local.shape[1:]),
                device=p_local.device,
                dtype=p_local.dtype,
            )
            orth_local = torch.cat([orth_local, pad], dim=0)

        # In-place weight update on this rank's local shard.
        p_local.add_(orth_local, alpha=-lr)

    @torch.no_grad()  # type: ignore[misc]
    def _local_muon_update(
        self, p: torch.Tensor, grad: torch.Tensor, group: Dict[str, Any]
    ) -> None:
        """Local (non-DP) Muon update – identical to OrthogonalizedOptimizer.step body."""
        from emerging_optimizers import utils

        state = self.state[p]
        self._apply_weight_decay_inplace(p, grad, group["lr"], group["weight_decay"])
        state["momentum_buffer"].lerp_(grad, 1 - group["momentum"])
        if self.nesterov:
            grad = grad.lerp(state["momentum_buffer"], group["momentum"])
        else:
            grad = state["momentum_buffer"]
        with utils.fp32_matmul_precision(self.fp32_matmul_prec):
            group_kwargs = {k: v for k, v in group.items() if k != "params"}
            orth_grad = self.orthogonalize(p, grad, **group_kwargs)
        self.pre_weight_update_fn_inplace(p, orth_grad)
        p.add_(orth_grad, alpha=-group["lr"])
        self.post_weight_update_fn_inplace(p)


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
        tp_mode: Tensor parallel mode ("blockwise", "duplicated", or "distributed").
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
        qkv_split_shapes: tuple[int, int, int] | None = None,
        fp32_matmul_prec: str = "medium",
        coefficient_type: str = "quintic",
        num_ns_steps: int = 5,
        scale_mode: str = "spectral",
        extra_scale_factor: float = 1.0,
        pg_collection: Optional[ProcessGroupCollection] = None,
        tp_mode: Literal["blockwise", "duplicated", "distributed"] = "duplicated",
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
        )
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
                ParamKey(
                    predicate=ParamPredicate(
                        name="nonlinear_or_embedding", fn=_is_nonlinear_or_embedding
                    )
                ): {'optimizer': 'adam'}
            },
        ),
        "adaptive_muon": EmergingOptimizerEntry(
            optimizer_cls=TensorParallelAdaptiveMuon,
            init_state_fn=_eopt_init_state_fn,
            config_to_kwargs=_adaptive_muon_config_to_kwargs,
            default_param_overrides={
                ParamKey(
                    predicate=ParamPredicate(
                        name="nonlinear_or_embedding", fn=_is_nonlinear_or_embedding
                    )
                ): {'optimizer': 'adam'}
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
