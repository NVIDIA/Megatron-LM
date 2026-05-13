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
from megatron.core.utils import get_pg_size, log_single_rank

from .optimizer_config import ParamKey, ParamPredicate

try:
    from torch.distributed.tensor import DTensor as _DTensor
    from torch.distributed.tensor.placement_types import Replicate, Shard, _StridedShard

    from megatron.core.distributed.fsdp.src.megatron_fsdp.uneven_dtensor import (
        _assert_chunks_cover_full_tensor,
        redistribute_uneven_dtensor_to_replicated,
        update_uneven_dtensor_chunk_metadata,
    )

    _HAVE_DTENSOR = True
except ImportError:
    _DTensor = None  # type: ignore[assignment,misc]
    Replicate = None  # type: ignore[assignment,misc]
    Shard = None  # type: ignore[assignment,misc]
    _StridedShard = None  # type: ignore[assignment,misc]
    _assert_chunks_cover_full_tensor = None  # type: ignore[assignment]
    redistribute_uneven_dtensor_to_replicated = None  # type: ignore[assignment]
    update_uneven_dtensor_chunk_metadata = None  # type: ignore[assignment]
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


class FSDPTensorParallelMuon(TensorParallelMuon):
    """TensorParallelMuon for Megatron-FSDP ZeRO-1/2/3.

    M-FSDP shards parameters unevenly across DP ranks; params split at rank
    boundaries must be gathered before Newton-Schulz orthogonalization. Fully
    local params are orthogonalized without any collective.
    """

    def __init__(
        self,
        params: ParamsT,
        dp_group: Optional[torch.distributed.ProcessGroup] = None,
        fsdp_batched_all_gather: bool = False,
        **kwargs: Any,
    ) -> None:
        assert _HAVE_DTENSOR, (
            "[Megatron-FSDP] torch.distributed.tensor.DTensor "
            f"is required to use {type(self).__name__}."
        )
        self.dp_group = dp_group
        self.fsdp_batched_all_gather = fsdp_batched_all_gather
        self._boundary_gather_indices_cache: Dict[tuple[int, ...], set[int]] = {}
        self._uneven_gather_plan_cache: Dict[int, Dict[str, Any]] = {}
        super().__init__(params, **kwargs)

    @torch.no_grad()  # type: ignore[misc]
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """Muon step for Megatron-FSDP ZeRO-1/2/3.

        Separates collective (AG) and local (NS) work into three phases so that
        no rank is blocked waiting on another rank computing NS to reach AG:
          1. Compute momentum updates locally for all params.
          2. All-gather boundary params — all collectives, no NS interleaved.
          3. Newton-Schulz + weight update locally for all params.
        """
        loss = None if closure is None else closure()

        if self.dp_group is None or get_pg_size(self.dp_group) == 1:
            for group in self.param_groups:
                self._init_group(group, skip_non_grad_params=False)
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    self._local_muon_update(p, p.grad, group)
            return loss

        # Track all parameters to update ordered by parameter group index.
        # (param, pre_ns_grad, is_gathered, lr, group_kwargs)
        all_updates: list = []

        # Phase 1: Compute momentum updates (fully local).
        for group in self.param_groups:
            self._init_group(group, skip_non_grad_params=False)
            gather_param_indices = self._get_boundary_gather_param_indices(group)
            lr = group["lr"]
            group_kwargs = {k: v for k, v in group.items() if k != "params"}

            for param_idx, p in enumerate(group["params"]):
                p_local = p.to_local()
                needs_gather = param_idx in gather_param_indices
                if p_local.numel() == 0 and not needs_gather:
                    # If this parameter is not split by Megatron-FSDP,
                    # and is empty on this DP rank, then we can skip this
                    # update for all TP ranks, as tensor parallelism uses
                    # even sharding, so empty implies that FSDP did not
                    # assign any fraction of the parameter to this DP rank.
                    continue

                state = self.state[p]
                mom_local = state["momentum_buffer"].to_local()

                grad = p.grad
                local_grad = grad.to_local() if grad is not None else torch.zeros_like(mom_local)

                self._apply_weight_decay_inplace(p_local, local_grad, lr, group["weight_decay"])
                mom_local.lerp_(local_grad, 1 - group["momentum"])
                if self.nesterov:
                    pre_ns_grad = local_grad.lerp(mom_local, group["momentum"])
                else:
                    pre_ns_grad = mom_local

                all_updates.append((p, pre_ns_grad, needs_gather, lr, group_kwargs))

        # Phase 2: AG all boundary gradients.
        boundary_update_indices = [
            i for i, (_, _, needs_gather, _, _) in enumerate(all_updates) if needs_gather
        ]
        if boundary_update_indices:
            boundary_items = [
                (all_updates[i][0], all_updates[i][1]) for i in boundary_update_indices
            ]
            if self.fsdp_batched_all_gather:
                gathered_boundary_updates = self._gather_full_uneven_local_tensors_like(
                    boundary_items
                )
            else:
                gathered_boundary_updates = [
                    self._gather_full_uneven_local_tensor_like(p, local_tensor)
                    for p, local_tensor in boundary_items
                ]
            for i, full_pre_ns_grad in zip(boundary_update_indices, gathered_boundary_updates):
                p, _, _, lr, group_kwargs = all_updates[i]
                all_updates[i] = (p, full_pre_ns_grad, True, lr, group_kwargs)

        # Phase 3: NS orthogonalization and weight update (fully local).
        from emerging_optimizers import utils

        with utils.fp32_matmul_precision(self.fp32_matmul_prec):
            for p, pre_ns_grad, is_gathered, lr, group_kwargs in all_updates:
                if is_gathered:
                    if pre_ns_grad is None:
                        continue
                    orth_update = super(FSDPTensorParallelMuon, self).orthogonalize(
                        p, pre_ns_grad, **group_kwargs
                    )
                    local_update = self._local_shard_from_full_update_like(p, orth_update)
                    self.pre_weight_update_fn_inplace(p._local_tensor, local_update)
                    p._local_tensor.add_(local_update, alpha=-lr)
                    self.post_weight_update_fn_inplace(p._local_tensor)
                else:
                    orth_update = (
                        super(FSDPTensorParallelMuon, self)
                        .orthogonalize(p, pre_ns_grad, **group_kwargs)
                        .to(dtype=p._local_tensor.dtype)
                    )
                    # Apply a Tensor step.
                    self.pre_weight_update_fn_inplace(p._local_tensor, orth_update)
                    p._local_tensor.add_(orth_update, alpha=-lr)
                    self.post_weight_update_fn_inplace(p._local_tensor)

        return loss

    def _needs_boundary_gather(self, dtensor: torch.Tensor) -> bool:
        assert isinstance(
            dtensor, _DTensor
        ), f"Detected non-DTensor during {type(self).__name__}: {dtensor}"
        local_tensor = dtensor.to_local()
        return local_tensor.numel() > 0 and tuple(dtensor.shape) != tuple(local_tensor.shape)

    def _get_mfsdp_param_layout(self, param: torch.Tensor, param_idx: int):
        """Return M-FSDP flat-buffer layout metadata for `param`.

        FSDP optimizer params keep a pointer to the original module parameter.
        M-FSDP attaches the original parameter's backing buffer and item id
        there, which lets Muon test whether the parameter's flat item interval
        crosses a shard boundary exactly.
        """
        orig_param = getattr(param, "orig_param", None)
        if orig_param is None:
            return None

        gbuf = getattr(orig_param, "_gbuf", None)
        item_id = getattr(orig_param, "_item_id", None)
        if gbuf is None or item_id is None or not hasattr(gbuf, "item_index_map"):
            raise AssertionError(
                "M-FSDP optimizer parameter is missing bucket metadata required "
                f"for Muon boundary gather detection: param_idx={param_idx}."
            )

        item_index = gbuf.item_index_map.get(item_id)
        if (
            item_index is None
            or not hasattr(item_index, "global_data_index")
            or not hasattr(item_index, "size")
        ):
            raise AssertionError(
                "M-FSDP optimizer parameter has invalid bucket item metadata required "
                f"for Muon boundary gather detection: param_idx={param_idx}, "
                f"item_id={item_id}."
            )

        bucket_index = getattr(gbuf, "bucket_index", None)
        shard_bucket_index = getattr(gbuf, "shard_bucket_index", None)
        if (
            bucket_index is None
            or shard_bucket_index is None
            or not hasattr(bucket_index, "global_data_index")
            or not hasattr(bucket_index, "size")
            or not hasattr(shard_bucket_index, "size")
        ):
            raise AssertionError(
                "M-FSDP optimizer parameter is missing bucket/shard metadata required "
                f"for Muon boundary gather detection: param_idx={param_idx}."
            )

        return gbuf, item_index, bucket_index, shard_bucket_index

    def _mfsdp_param_crosses_shard_boundary(self, param: torch.Tensor, param_idx: int) -> bool:
        """Return whether `param`'s flat M-FSDP item interval spans DP shards."""
        layout = self._get_mfsdp_param_layout(param, param_idx)
        if layout is None:
            raise AssertionError(
                "Expected M-FSDP parameter metadata while checking Muon boundary "
                f"gather requirement: param_idx={param_idx}."
            )

        gbuf, item_index, bucket_index, shard_bucket_index = layout
        if not getattr(gbuf, "is_data_distributed", True):
            return False

        item_size = int(item_index.size)
        if item_size == 0:
            return False

        bucket_start = int(bucket_index.global_data_index)
        bucket_size = int(bucket_index.size)
        shard_size = int(shard_bucket_index.size)
        if shard_size <= 0 or bucket_size % shard_size != 0:
            raise AssertionError(
                "Invalid M-FSDP shard metadata for Muon boundary gather detection: "
                f"param_idx={param_idx}, bucket_size={bucket_size}, shard_size={shard_size}."
            )

        item_start = int(item_index.global_data_index)
        item_end = item_start + item_size
        if item_start < bucket_start or item_end > bucket_start + bucket_size:
            raise AssertionError(
                "M-FSDP item interval falls outside its bucket during Muon boundary "
                f"gather detection: param_idx={param_idx}, item=({item_start}, {item_end}), "
                f"bucket=({bucket_start}, {bucket_start + bucket_size})."
            )

        first_shard = (item_start - bucket_start) // shard_size
        last_shard = (item_end - 1 - bucket_start) // shard_size
        crosses_boundary = first_shard != last_shard

        local_tensor = param.to_local()
        if local_tensor.numel() > 0:
            local_is_partial = tuple(param.shape) != tuple(local_tensor.shape)
            if local_is_partial and not crosses_boundary:
                raise AssertionError(
                    "M-FSDP DTensor local shape indicates a split parameter, but flat "
                    "bucket metadata does not cross a shard boundary: "
                    f"param_idx={param_idx}, item=({item_start}, {item_end}), "
                    f"shard_size={shard_size}, global_shape={tuple(param.shape)}, "
                    f"local_shape={tuple(local_tensor.shape)}."
                )

        return crosses_boundary

    def _get_boundary_gather_param_indices(self, group: Dict[str, Any]) -> set[int]:
        """Return globally-agreed parameters whose flat items cross FSDP shard boundaries."""
        params = group["params"]
        cache_key = tuple(id(param) for param in params)
        cached_indices = self._boundary_gather_indices_cache.get(cache_key)
        if cached_indices is not None:
            return cached_indices

        has_mfsdp_params = any(getattr(param, "orig_param", None) is not None for param in params)
        if has_mfsdp_params:
            result = set()
            for idx, param in enumerate(params):
                if getattr(param, "orig_param", None) is None:
                    raise AssertionError(
                        "Muon optimizer group mixes M-FSDP params with params lacking "
                        f"M-FSDP bucket metadata: param_idx={idx}."
                    )
                if self._mfsdp_param_crosses_shard_boundary(param, idx):
                    result.add(idx)
            self._boundary_gather_indices_cache[cache_key] = result
            return result

        local_boundary_indices = [
            idx for idx, param in enumerate(params) if self._needs_boundary_gather(param)
        ]

        if self.dp_group is None or get_pg_size(self.dp_group) == 1:
            result = set(local_boundary_indices)
            self._boundary_gather_indices_cache[cache_key] = result
            return result

        gathered_indices: list[list[int] | None] = [None] * get_pg_size(self.dp_group)
        torch.distributed.all_gather_object(
            gathered_indices, local_boundary_indices, group=self.dp_group
        )
        result = {
            idx
            for rank_indices in gathered_indices
            if rank_indices is not None
            for idx in rank_indices
        }
        self._boundary_gather_indices_cache[cache_key] = result
        return result

    def _copy_dtensor_chunk_metadata(self, dst, src) -> None:
        if hasattr(src._local_tensor, "__create_chunk_list__"):
            dst._local_tensor.__create_chunk_list__ = src._local_tensor.__create_chunk_list__
        if hasattr(src._local_tensor, "__create_write_items__"):
            dst._local_tensor.__create_write_items__ = src._local_tensor.__create_write_items__

    def _dtensor_from_local_like(self, dtensor_ref, local_tensor: torch.Tensor):
        dtensor = _DTensor.from_local(
            local_tensor=local_tensor,
            device_mesh=dtensor_ref.device_mesh,
            placements=dtensor_ref.placements,
            shape=dtensor_ref.shape,
            stride=dtensor_ref.stride(),
        )
        self._copy_dtensor_chunk_metadata(dtensor, dtensor_ref)
        return dtensor

    def _get_uneven_gather_plan(self, dtensor_ref) -> Dict[str, Any] | None:
        """Return static metadata needed to gather a Megatron-FSDP uneven DTensor.

        The parameter layout is fixed after M-FSDP construction, so the chunk
        offsets/shapes only need to be exchanged once per boundary parameter.
        """
        cache_key = id(dtensor_ref)
        cached_plan = self._uneven_gather_plan_cache.get(cache_key)
        if cached_plan is not None:
            return cached_plan

        shard_mesh_dims = []
        for mesh_dim, placement in enumerate(dtensor_ref.placements):
            if isinstance(placement, (Shard, _StridedShard)):
                shard_mesh_dims.append(mesh_dim)
            elif isinstance(placement, Replicate):
                continue
            else:
                raise ValueError(
                    f"Unexpected placement {placement} at mesh dimension {mesh_dim}. "
                    "Expected Shard, _StridedShard, or Replicate."
                )

        if len(shard_mesh_dims) != 1:
            # M-FSDP optimizer shards are expected to have one sharded mesh
            # dimension. Keep the generic helper as a conservative fallback.
            return None

        if not hasattr(dtensor_ref._local_tensor, "__create_chunk_list__"):
            update_uneven_dtensor_chunk_metadata(dtensor_ref)

        chunk_metadata_list = dtensor_ref._local_tensor.__create_chunk_list__()
        if len(chunk_metadata_list) != 1:
            raise ValueError(
                f"Expected exactly one chunk metadata per rank, got {len(chunk_metadata_list)}."
            )

        local_tensor = dtensor_ref.to_local()
        local_chunk_metadata = chunk_metadata_list[0]
        local_chunks_info = [
            {"shape": torch.Size(local_tensor.shape), "offset": tuple(local_chunk_metadata.offsets)}
        ]
        shard_group = dtensor_ref.device_mesh.get_group(shard_mesh_dims[0])
        group_chunks_info: list[list[Dict[str, Any]] | None] = [None] * shard_group.size()
        torch.distributed.all_gather_object(group_chunks_info, local_chunks_info, group=shard_group)

        chunk_infos = [
            {
                "shape": torch.Size(chunk_info["shape"]),
                "offset": tuple(chunk_info["offset"]),
                "numel": torch.Size(chunk_info["shape"]).numel(),
            }
            for chunks_info in group_chunks_info
            if chunks_info is not None
            for chunk_info in chunks_info
        ]
        plan = {"shard_group": shard_group, "chunk_infos": chunk_infos}
        self._uneven_gather_plan_cache[cache_key] = plan
        return plan

    def _gather_full_uneven_local_tensor_like(
        self, dtensor_ref, local_tensor: torch.Tensor
    ) -> torch.Tensor | None:
        """Gather a local tensor using the uneven sharding layout of `dtensor_ref`."""
        plan = self._get_uneven_gather_plan(dtensor_ref)
        if plan is None:
            if redistribute_uneven_dtensor_to_replicated is None:
                raise RuntimeError(
                    "Megatron-FSDP `redistribute_uneven_dtensor_to_replicated` is required "
                    "to gather un-evenly sharded parameters for Muon step()."
                )
            local_dtensor = self._dtensor_from_local_like(dtensor_ref, local_tensor.contiguous())
            if not hasattr(local_dtensor._local_tensor, "__create_chunk_list__"):
                update_uneven_dtensor_chunk_metadata(local_dtensor)
            full_tensor = redistribute_uneven_dtensor_to_replicated(local_dtensor).to_local()
            self._copy_dtensor_chunk_metadata(dtensor_ref, local_dtensor)
            return None if local_tensor.numel() == 0 else full_tensor

        local_buffer = local_tensor.contiguous().view(-1)
        group_tensors = [
            torch.empty(chunk_info["numel"], dtype=local_tensor.dtype, device=local_tensor.device)
            for chunk_info in plan["chunk_infos"]
        ]
        torch.distributed.all_gather(group_tensors, local_buffer, group=plan["shard_group"])

        if local_tensor.numel() == 0:
            return None

        full_tensor = torch.empty(
            dtensor_ref.shape, dtype=local_tensor.dtype, device=local_tensor.device
        )
        assigned_numel = 0
        for chunk_info, gathered_tensor in zip(plan["chunk_infos"], group_tensors):
            chunk_shape = chunk_info["shape"]
            offset = chunk_info["offset"]
            slices = tuple(slice(o, o + s) for o, s in zip(offset, chunk_shape))
            full_tensor[slices] = gathered_tensor.view(chunk_shape)
            assigned_numel += chunk_info["numel"]

        _assert_chunks_cover_full_tensor(dtensor_ref.shape, plan["chunk_infos"], assigned_numel)
        return full_tensor

    def _gather_full_uneven_local_tensors_like(
        self, items: list[tuple[torch.Tensor, torch.Tensor]]
    ) -> list[torch.Tensor | None]:
        """Batch uneven gathers for boundary tensors.

        Each boundary parameter still needs the same logical data as
        `_gather_full_uneven_local_tensor_like`. Batching concatenates local
        shards with the same dtype/device/group so the optimizer issues one
        all-gather per batch instead of one all-gather per boundary parameter.
        Ranks with empty local shards participate in the collective but do not
        reconstruct or orthogonalize the full boundary tensor.
        """
        results: list[torch.Tensor | None] = [None] * len(items)
        batches: dict[tuple[int, torch.dtype, torch.device], Dict[str, Any]] = {}

        for item_idx, (dtensor_ref, local_tensor) in enumerate(items):
            plan = self._get_uneven_gather_plan(dtensor_ref)
            if plan is None:
                results[item_idx] = self._gather_full_uneven_local_tensor_like(
                    dtensor_ref, local_tensor
                )
                continue

            key = (id(plan["shard_group"]), local_tensor.dtype, local_tensor.device)
            batch = batches.setdefault(
                key,
                {
                    "shard_group": plan["shard_group"],
                    "dtype": local_tensor.dtype,
                    "device": local_tensor.device,
                    "item_indices": [],
                    "plans": [],
                },
            )
            batch["item_indices"].append(item_idx)
            batch["plans"].append(plan)

        for batch in batches.values():
            self._gather_full_uneven_local_tensor_batch(items, results, batch)

        return results

    def _gather_full_uneven_local_tensor_batch(
        self,
        items: list[tuple[torch.Tensor, torch.Tensor]],
        results: list[torch.Tensor | None],
        batch: Dict[str, Any],
    ) -> None:
        shard_group = batch["shard_group"]
        group_size = get_pg_size(shard_group)
        group_rank = torch.distributed.get_rank(shard_group)
        item_indices = batch["item_indices"]
        plans = batch["plans"]

        local_chunks = [items[item_idx][1].contiguous().view(-1) for item_idx in item_indices]
        local_buffer = (
            torch.cat(local_chunks)
            if local_chunks and sum(chunk.numel() for chunk in local_chunks) > 0
            else torch.empty(0, dtype=batch["dtype"], device=batch["device"])
        )
        expected_local_numel = sum(plan["chunk_infos"][group_rank]["numel"] for plan in plans)
        if local_buffer.numel() != expected_local_numel:
            raise AssertionError(
                "Batched uneven DTensor gather local buffer size mismatch: "
                f"got {local_buffer.numel()}, expected {expected_local_numel}."
            )

        rank_offsets: list[list[int]] = []
        rank_total_numels: list[int] = []
        for rank in range(group_size):
            offsets = []
            total_numel = 0
            for plan in plans:
                offsets.append(total_numel)
                total_numel += plan["chunk_infos"][rank]["numel"]
            rank_offsets.append(offsets)
            rank_total_numels.append(total_numel)

        group_tensors = [
            torch.empty(total_numel, dtype=batch["dtype"], device=batch["device"])
            for total_numel in rank_total_numels
        ]
        torch.distributed.all_gather(group_tensors, local_buffer, group=shard_group)

        for batch_item_idx, item_idx in enumerate(item_indices):
            dtensor_ref, local_tensor = items[item_idx]
            if local_tensor.numel() == 0:
                results[item_idx] = None
                continue

            full_tensor = torch.empty(
                dtensor_ref.shape, dtype=local_tensor.dtype, device=local_tensor.device
            )
            assigned_numel = 0
            plan = plans[batch_item_idx]
            for rank, gathered_rank_tensor in enumerate(group_tensors):
                chunk_info = plan["chunk_infos"][rank]
                chunk_numel = chunk_info["numel"]
                if chunk_numel == 0:
                    continue
                offset = rank_offsets[rank][batch_item_idx]
                chunk_shape = chunk_info["shape"]
                chunk_tensor = gathered_rank_tensor[offset : offset + chunk_numel].view(chunk_shape)
                slices = tuple(slice(o, o + s) for o, s in zip(chunk_info["offset"], chunk_shape))
                full_tensor[slices] = chunk_tensor
                assigned_numel += chunk_numel

            _assert_chunks_cover_full_tensor(dtensor_ref.shape, plan["chunk_infos"], assigned_numel)
            results[item_idx] = full_tensor

    def _local_shard_from_full_update_like(self, dtensor_ref, full_update: torch.Tensor):
        if not hasattr(dtensor_ref._local_tensor, "__create_chunk_list__"):
            raise ValueError(
                f"{dtensor_ref} is not a Megatron-FSDP DTensor parameter "
                "with DTensor._local_tensor.__create_chunk_list__. "
                "Verify that `update_uneven_dtensor_chunk_metadata` "
                "has been called on this uneven DTensor."
            )
        shard_metadata = dtensor_ref._local_tensor.__create_chunk_list__()[0]
        slices = tuple(
            slice(offset, offset + size)
            for offset, size in zip(shard_metadata.offsets, shard_metadata.sizes)
        )
        return full_update[slices].contiguous().to(dtype=dtensor_ref._local_tensor.dtype)

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
