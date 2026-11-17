# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Adapted from https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/53c20f81cb575f19af34ff0ae1749fb13b07999b/src/megatron/bridge/models/conversion/param_mapping.py
# Source license: Apache-2.0.

import json
import re
from abc import ABC, abstractmethod
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from types import SimpleNamespace
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Literal,
    Optional,
    Tuple,
    TypeVar,
    Union,
    cast,
)

import torch
import torch.distributed
import torch.nn as nn
from torch.distributed._tensor import DTensor

from megatron.bridge.models.conversion.gtp import _get_mapping_shape, _is_gtp_param
from megatron.bridge.models.conversion.utils import (
    get_module_and_param_from_name,
    is_modelopt_dynamic_module,
    remove_non_pickleables,
)
from megatron.bridge.utils.common_utils import extract_expert_number_from_param
from megatron.core import mpu
from megatron.core.fp8_utils import FP8_TENSOR_CLASS, HAVE_TE_FP8_TENSOR_CLASS
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import get_pg_rank, get_pg_size

WeightType = TypeVar("WeightType", torch.Tensor, Dict[str, torch.Tensor])
MXFP8ShardGroup = Literal["tp", "etp", "replicated"]
LocalHFShardGroup = Literal["tp", "etp", "replicated"]

import logging

logger = logging.getLogger(__name__)


# Export policy is scoped to one conversion operation rather than stored on a
# mapping instance. Some mappings (notably AutoMapping) create their concrete
# delegate lazily while ``megatron_to_hf`` is running, so an instance property
# would not reliably reach the object that performs the PP broadcast.
_SKIP_PP_BROADCAST: ContextVar[bool] = ContextVar(
    "megatron_bridge_skip_pp_broadcast", default=False
)


@dataclass(frozen=True)
class LocalHFParamSpec:
    """A canonical HF-compatible view of one local logical Megatron parameter.

    A spec describes only how to name and, optionally, split an already-local
    logical tensor. It does not perform Bridge collectives, layout conversion,
    or destination-backend/topology validation.
    """

    name: str
    split_dim: Optional[int] = None
    split_index: int = 0
    split_count: int = 1

    def __post_init__(self) -> None:
        if self.split_count < 1:
            raise ValueError("split_count must be positive.")
        if not 0 <= self.split_index < self.split_count:
            raise ValueError("split_index must be within split_count.")
        if self.split_count > 1 and self.split_dim is None:
            raise ValueError("split_dim is required when split_count is greater than one.")

    def select(self, tensor: torch.Tensor) -> torch.Tensor:
        """Select this HF component from a logical Megatron tensor."""
        if self.split_count == 1:
            return tensor
        dim = self.split_dim % tensor.ndim
        if tensor.shape[dim] % self.split_count:
            raise ValueError(
                f"Cannot split dimension {dim} of shape {tuple(tensor.shape)} into {self.split_count} equal parts."
            )
        return torch.chunk(tensor, self.split_count, dim=dim)[self.split_index]

    def selected_shape(self, shape: torch.Size) -> torch.Size:
        """Return the shape selected from a logical Megatron tensor."""
        if self.split_count == 1:
            return torch.Size(shape)
        selected = list(shape)
        dim = self.split_dim % len(selected)
        if selected[dim] % self.split_count:
            raise ValueError(
                f"Cannot split dimension {dim} of shape {tuple(shape)} into {self.split_count} equal parts."
            )
        selected[dim] //= self.split_count
        return torch.Size(selected)


@dataclass(frozen=True)
class LocalMXFP8Param:
    """A local native MXFP8 weight and its matching E8M0 scale tensor."""

    name: str
    weight: torch.Tensor
    weight_scale: torch.Tensor
    global_weight_shape: torch.Size
    shard_group: MXFP8ShardGroup
    shard_dim: int | None


def _validate_native_mxfp8_pair(
    weight: torch.Tensor, weight_scale: torch.Tensor, global_param_name: str
) -> None:
    if weight.ndim != 2:
        raise ValueError(f"{global_param_name}: native MXFP8 weights must be two-dimensional")
    if weight.shape[-1] % 32:
        raise ValueError(f"{global_param_name}: K={weight.shape[-1]} is not divisible by 32")
    expected_scale_shape = torch.Size((*weight.shape[:-1], weight.shape[-1] // 32))
    if weight_scale.shape != expected_scale_shape:
        raise ValueError(
            f"{global_param_name}: expected native MXFP8 scale shape {tuple(expected_scale_shape)}, "
            f"got {tuple(weight_scale.shape)}"
        )


def _project_gated_native_mxfp8(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    names: tuple[str, str],
    global_param_name: str,
    shard_group: MXFP8ShardGroup,
    shard_size: int,
) -> tuple[LocalMXFP8Param, ...]:
    _validate_native_mxfp8_pair(weight, weight_scale, global_param_name)
    if weight.shape[-2] % 2:
        raise ValueError(f"{global_param_name}: fused FC1 output rows must be even")

    gate_weight, up_weight = torch.chunk(weight, 2, dim=-2)
    gate_scale, up_scale = torch.chunk(weight_scale, 2, dim=-2)
    global_weight_shape = list(gate_weight.shape)
    global_weight_shape[-2] *= shard_size
    return tuple(
        LocalMXFP8Param(
            name=name,
            weight=projected_weight,
            weight_scale=projected_scale,
            global_weight_shape=torch.Size(global_weight_shape),
            shard_group=shard_group,
            shard_dim=0,
        )
        for name, projected_weight, projected_scale in zip(
            names, (gate_weight, up_weight), (gate_scale, up_scale)
        )
    )


@dataclass(frozen=True)
class LocalHFParam:
    """A canonical HF-compatible view of one local unquantized BF16 parameter.

    The tensor remains a live view of the local Megatron parameter. Its global
    shape and sharding fields describe how an M-to-N transport can reconstruct
    the corresponding logical HF parameter without Bridge collectives.
    """

    name: str
    weight: torch.Tensor
    global_weight_shape: torch.Size
    shard_group: LocalHFShardGroup
    shard_dim: int | None


def _module_uses_fsdp(megatron_module: nn.Module) -> bool:
    """Return True if the module uses Megatron-FSDP"""
    return hasattr(megatron_module, "_parameters") and any(
        key.startswith("weight") and isinstance(value, DTensor)
        for key, value in megatron_module._parameters.items()
    )


class MegatronParamMapping(ABC, Generic[WeightType]):
    """
    Abstract base class for weight conversion between Megatron and external formats.

    This class provides the foundation for all weight mappings, handling the complex
    conversions between Megatron-Core's distributed tensor formats and standard
    (typically HuggingFace) formats. Each concrete mapping implements specific
    transformation logic while inheriting common parallel communication patterns.

    Key responsibilities:
    - Format transformation (e.g., QKV merging/splitting, gated MLP handling)
    - Tensor parallel (TP) distribution and gathering across GPUs
    - Pipeline parallel (PP) broadcasting between pipeline stages
    - Wildcard pattern resolution for layer-wise mappings

    The mapping abstraction ensures that higher-level code doesn't need to know
    about the parallel topology or format differences - it just requests a
    conversion and the mapping handles all the complexity.

    Public helper methods for subclasses:
    - broadcast_from_pp_rank: Broadcast tensors across pipeline stages
    - broadcast_obj_from_pp_rank: Broadcast Python objects across PP ranks
    - broadcast_tensor_to_tp_ranks: Broadcast within TP group
    - scatter_to_tp_ranks: Distribute tensor shards to TP ranks
    - gather_from_tp_ranks: Collect tensor shards from TP ranks

    Example:
        .. code-block:: python

            class MyCustomMapping(MegatronParamMapping[torch.Tensor]):
                def hf_to_megatron(self, hf_weights, megatron_module):
                    # Custom transformation logic
                    transformed = hf_weights.t()  # Example: transpose
                    # Use helpers for distribution
                    return self.scatter_to_tp_ranks(...)

                def megatron_to_hf(self, megatron_weights, megatron_module):
                    # Broadcast from owning PP rank
                    weight = self.broadcast_from_pp_rank(megatron_weights)
                    # Gather from TP ranks and transform
                    gathered = self.gather_from_tp_ranks(weight)
                    return {"custom_weight": gathered[0].t()}
    """

    def __init__(self, megatron_param: str, hf_param: Union[str, Dict[str, str]]):
        """Initialize the weight mapping.

        Args:
            megatron_param (str): Megatron parameter name pattern (supports *
                wildcards).
            hf_param (Union[str, Dict[str, str]]): External format name pattern(s).
        """
        self.megatron_param = megatron_param
        self.hf_param = hf_param
        self._validate_patterns()

        # Cache for metadata and tensor_spec_output
        self._broadcast_obj_cache = {}
        self._tensor_spec_output_cache = {}
        self._pg_collection = None

        if mpu.is_initialized():
            self.pp_group = mpu.get_pipeline_model_parallel_group()
            self.ep_group = mpu.get_expert_model_parallel_group()
            self._tp_group = mpu.get_tensor_model_parallel_group()
            self._etp_group = mpu.get_expert_tensor_parallel_group()
        else:
            self.pp_group = None
            self.ep_group = None
            self._tp_group = None
            self._etp_group = None

        # Set allow_hf_name_mismatch to True when the declared HF name will not be found verbatim
        # in the checkpoint's key set. That covers two cases: a name that is rewritten or
        # synthesized (see maybe_modify_loaded_hf_weight), and a weight that is legitimately
        # absent for some layers or configurations. Both bypass the hf_keys check in
        # `build_conversion_tasks`, which raises otherwise.
        self.allow_hf_name_mismatch = False

    @classmethod
    @contextmanager
    def skip_pp_broadcast(cls) -> Iterator[None]:
        """Disable PP replication for the duration of one HF export.

        In this mode ``broadcast_from_pp_rank`` and
        ``broadcast_obj_from_pp_rank`` return the value local to the current
        rank. TP/EP collectives remain enabled, so mappings still perform all
        required shard reconstruction and HF layout conversion on the owning
        pipeline stage.

        The policy is held in a :class:`ContextVar` so it also applies to
        mappings created lazily by wrappers such as :class:`AutoMapping`.
        """
        token = _SKIP_PP_BROADCAST.set(True)
        try:
            yield
        finally:
            _SKIP_PP_BROADCAST.reset(token)

    def local_hf_param_specs(
        self, global_param_name: Optional[str] = None
    ) -> tuple[LocalHFParamSpec, ...]:
        """Describe canonical local HF views for one mapped parameter.

        A non-empty result means this parameter's local logical Megatron tensor
        can be transferred as the described HF-compatible views without first
        running Bridge collectives or layout conversion. It is a per-parameter
        contract, not a declaration that the model supports general HF
        conversion or M-to-N refit.

        Mappings that require transpose, permutation, interleaving, or grouped
        export transforms must return an empty tuple unless they implement an
        explicit safe override. An empty tuple selects the normal Bridge
        conversion/packed-broadcast path; it does not mean the model is
        unsupported. Consumers must separately qualify the destination backend
        and source/destination topology.

        The contract is independent of the source and destination storage
        precision. For example, an MXFP8 transport may materialize these
        canonical logical views from quantized source storage and requantize a
        persistent MXFP8 destination in place. Direct transfer of physical
        MXFP8 data and scales requires separate backend/layout qualification.

        Args:
            global_param_name: Resolved global Megatron parameter name. Custom
                mappings may use it to derive an HF view name.

        Returns:
            Per-parameter local HF view specifications, or an empty tuple when
            the normal Bridge conversion path is required.
        """
        if not isinstance(self.hf_param, str):
            return ()
        # Grouped exports and dimension permutations need conversion work. Grouped
        # mappings must explicitly override this method when a canonical local view exists.
        if (
            getattr(self, "is_grouped_export", False)
            or getattr(self, "permute_dims", None) is not None
            or getattr(self, "transpose_on_export", False)
        ):
            return ()
        return (LocalHFParamSpec(self.hf_param),)

    def local_mxfp8_params(
        self,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        *,
        global_param_name: str,
        megatron_module: nn.Module,
    ) -> tuple[LocalMXFP8Param, ...]:
        raise ValueError(
            f"Mapping {type(self).__name__} for {global_param_name!r} does not support direct native MXFP8 export."
        )

    def _local_hf_sharding(
        self, weight: torch.Tensor, megatron_module: nn.Module
    ) -> tuple[LocalHFShardGroup, int | None, int]:
        """Describe the local tensor sharding for canonical HF views."""
        if self.tp_size == 1:
            return "replicated", None, 1
        raise ValueError(
            f"mapping {type(self).__name__} does not describe local HF sharding "
            f"when tensor parallel size is {self.tp_size}"
        )

    def local_hf_params(
        self, weight: torch.Tensor, *, global_param_name: str, megatron_module: nn.Module
    ) -> tuple[LocalHFParam, ...]:
        """Materialize canonical local HF views for an unquantized BF16 parameter.

        This method only selects views from local storage. It does not gather,
        broadcast, dequantize, cast, or emit quantization sidecars.

        Args:
            weight: Local logical Megatron parameter storage.
            global_param_name: Resolved global Megatron parameter name.
            megatron_module: Local module that owns ``weight``.

        Returns:
            Canonical local HF parameter views in mapping-defined order.

        Raises:
            ValueError: If the storage is not plain BF16 or the mapping cannot
                represent the parameter as local HF views.
        """
        if HAVE_TE_FP8_TENSOR_CLASS and isinstance(weight, FP8_TENSOR_CLASS):
            raise ValueError(
                f"{global_param_name}: local HF parameter iteration does not support quantized storage"
            )
        if weight.dtype != torch.bfloat16:
            raise ValueError(
                f"{global_param_name}: local HF parameter iteration requires unquantized BF16 storage, "
                f"got {weight.dtype}"
            )
        if isinstance(weight, DTensor) or _module_uses_fsdp(megatron_module):
            raise ValueError(
                f"{global_param_name}: local HF parameter iteration does not support DTensor/FSDP"
            )
        if _is_gtp_param(weight):
            raise ValueError(
                f"{global_param_name}: local HF parameter views cannot represent GTP sharding; "
                "use the gathered HF weight export instead"
            )

        specs = self.local_hf_param_specs(global_param_name)
        if not specs:
            raise ValueError(
                f"{global_param_name}: mapping {type(self).__name__} cannot be represented as canonical local HF views"
            )
        shard_group, shard_dim, shard_size = self._local_hf_sharding(weight, megatron_module)

        params = []
        for spec in specs:
            selected = spec.select(weight)
            global_shape = list(selected.shape)
            if shard_dim is not None:
                normalized_shard_dim = shard_dim % selected.ndim
                global_shape[normalized_shard_dim] *= shard_size
            else:
                normalized_shard_dim = None
            params.append(
                LocalHFParam(
                    name=spec.name,
                    weight=selected,
                    global_weight_shape=torch.Size(global_shape),
                    shard_group=shard_group,
                    shard_dim=normalized_shard_dim,
                )
            )
        return tuple(params)

    def set_process_groups_from_pg_collection(self, pg_collection: Any) -> None:
        """Override snapshotted Megatron-Core globals with a ``ProcessGroupCollection``.

        Used by the decentralized PG path where ``mpu`` is never initialized.
        ``__init__`` runs at registry construction time and snapshots whatever
        Megatron-Core globals exist then; in the decentralized path those are
        absent, so ``tp_size`` would default to ``world_size`` and conversions
        would compute the wrong shard sizes. ``MegatronModelBridge`` calls this
        right before running tasks to install the user-supplied groups.
        """
        if pg_collection is None:
            return
        self._pg_collection = pg_collection
        for attr, field in (
            ("pp_group", "pp"),
            ("ep_group", "ep"),
            ("_tp_group", "tp"),
            ("_etp_group", "expt_tp"),
        ):
            group = getattr(pg_collection, field, None)
            if group is not None:
                setattr(self, attr, group)
        for child in vars(self).values():
            if isinstance(child, MegatronParamMapping):
                child.set_process_groups_from_pg_collection(pg_collection)

    @property
    def tp_group(self):
        """Get the tensor model parallel group."""
        if self.is_expert:
            return self._etp_group
        return self._tp_group

    @property
    def tp_rank(self) -> int:
        """Get the tensor model parallel rank."""
        return get_pg_rank(self.tp_group)

    @property
    def tp_size(self) -> int:
        """Get the tensor model parallel size."""
        return get_pg_size(self.tp_group)

    @property
    def pp_rank(self) -> int:
        """Get the pipeline model parallel rank."""
        return get_pg_rank(self.pp_group)

    @property
    def pp_size(self) -> int:
        """Get the pipeline model parallel size."""
        return get_pg_size(self.pp_group)

    @property
    def ep_rank(self) -> int:
        """Get the expert model parallel rank."""
        return get_pg_rank(self.ep_group)

    @property
    def ep_size(self) -> int:
        """Get the expert model parallel size."""
        return get_pg_size(self.ep_group)

    @property
    def etp_rank(self) -> int:
        """Get the expert tensor parallel rank."""
        return get_pg_rank(self.etp_group)

    @property
    def etp_size(self) -> int:
        """Get the expert tensor parallel size."""
        return get_pg_size(self.etp_group)

    @property
    def is_expert(self) -> bool:
        """Check if this mapping is for an expert parameter.

        Matches both TEGroupedMLP (.experts.linear_fc) and
        SequentialMLP (.experts.local_experts.*.linear_fc) patterns.
        Uses ``.experts.`` rather than ``.mlp.experts.`` so models with an
        intermediate sub-module (e.g. ``.mlp.<pool>.experts.``) are matched too.
        """
        return (
            ".experts.linear_fc" in self.megatron_param
            or ".experts.local_experts." in self.megatron_param
        )

    def _resolve_names(self, captures: Tuple[str, ...]) -> Tuple[str, Union[str, Dict[str, str]]]:
        """Resolve wildcard patterns with captured values.

        Handles both ** (any characters) and * (digits) wildcards in order.
        ** patterns are processed before * patterns to avoid conflicts.
        """
        resolved_megatron_param = self.megatron_param
        capture_index = 0

        # First pass: resolve ** wildcards
        while "**" in resolved_megatron_param and capture_index < len(captures):
            resolved_megatron_param = resolved_megatron_param.replace(
                "**", captures[capture_index], 1
            )
            capture_index += 1

        # Second pass: resolve * wildcards
        while "*" in resolved_megatron_param and capture_index < len(captures):
            resolved_megatron_param = resolved_megatron_param.replace(
                "*", captures[capture_index], 1
            )
            capture_index += 1

        if isinstance(self.hf_param, str):
            resolved_hf_param = self.hf_param
            capture_index = 0

            # First pass: resolve ** wildcards
            while "**" in resolved_hf_param and capture_index < len(captures):
                resolved_hf_param = resolved_hf_param.replace("**", captures[capture_index], 1)
                capture_index += 1

            # Second pass: resolve * wildcards
            while "*" in resolved_hf_param and capture_index < len(captures):
                resolved_hf_param = resolved_hf_param.replace("*", captures[capture_index], 1)
                capture_index += 1
        else:
            resolved_hf_param = {}
            for k, v in self.hf_param.items():
                resolved_v = v
                capture_index = 0

                # First pass: resolve ** wildcards
                while "**" in resolved_v and capture_index < len(captures):
                    resolved_v = resolved_v.replace("**", captures[capture_index], 1)
                    capture_index += 1

                # Second pass: resolve * wildcards
                while "*" in resolved_v and capture_index < len(captures):
                    resolved_v = resolved_v.replace("*", captures[capture_index], 1)
                    capture_index += 1

                resolved_hf_param[k] = resolved_v

        return resolved_megatron_param, resolved_hf_param

    def resolve(self, captures: Tuple[str, ...]) -> "MegatronParamMapping":
        """Create a new mapping with resolved wildcards.

        This default implementation works for mappings with a
        (megatron_param, hf_param) constructor.

        Args:
            captures (Tuple[str, ...]): Captured wildcard values.

        Returns:
            MegatronParamMapping: A new mapping instance with resolved names.
        """
        resolved_megatron_param, resolved_hf_param = self._resolve_names(captures)
        return type(self)(resolved_megatron_param, resolved_hf_param)

    @abstractmethod
    def hf_to_megatron(self, hf_weights: WeightType, megatron_module: nn.Module) -> torch.Tensor:
        """Convert hf_weights TO Megatron format.

        This method handles:
        1. Format transformation (if needed)
        2. Tensor parallel distribution (if self.tp_size > 1)

        Args:
            hf_weights (WeightType): Source hf_weights in external format.
            megatron_module (nn.Module): Target Megatron module (for config
                access).

        Returns:
            torch.Tensor: Weight tensor ready for the current TP rank.
        """
        ...

    @abstractmethod
    def megatron_to_hf(
        self, megatron_weights: Optional[torch.Tensor], megatron_module: Optional[nn.Module]
    ) -> Dict[str, torch.Tensor]:
        """Convert weights FROM Megatron format.

        This method handles:
        1. Pipeline parallel broadcasting (if weight is on different PP rank)
        2. Tensor parallel gathering (if needed)
        3. Format transformation

        Args:
            megatron_weights (Optional[torch.Tensor]): Weight tensor from current
                rank (None if on different PP rank).
            megatron_module (Optional[nn.Module]): Module for config access
                (None if on different PP rank).

        Returns:
            Dict[str, torch.Tensor]: Converted weights (empty dict if not on
                TP rank 0).
        """
        ...

    def megatron_to_hf_quant(
        self,
        megatron_weights: Optional[torch.Tensor],
        megatron_module: Optional[nn.Module],
        quantization_checker: Callable[[str], bool],
        quant_fn: Callable[..., Tuple[torch.Tensor, torch.Tensor]],
        quant_block_size: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Convert weights FROM Megatron format with quantization."""
        raise NotImplementedError("megatron_to_hf_quant not implemented for this mapping")

    def broadcast_from_pp_rank(
        self, tensor: Optional[torch.Tensor], cache_key: Optional[str] = None
    ) -> Optional[torch.Tensor]:
        """Broadcast a tensor from the pipeline-parallel rank that owns it.

        Broadcasts to **all** PP ranks. This mirrors the behaviour of
        `broadcast_from_megatron_pp` in the original MMapping implementation and
        additionally keeps the tensor-parallel metadata (`tensor_model_parallel`,
        `partition_dim`) consistent on every rank.

        Args:
            tensor (Optional[torch.Tensor]): The local tensor if the current PP
                rank owns it. ``None`` otherwise.

        Returns:
            Optional[torch.Tensor]: The broadcasted tensor on every PP rank, or
                ``None`` if *no* PP rank owned the tensor (which indicates a bug
                in the calling code).
        """

        # PP-local export deliberately leaves non-owning stages with ``None``.
        # The caller filters those tasks, while the owning stage continues into
        # the normal TP/EP gathering and layout-conversion code below.
        if _SKIP_PP_BROADCAST.get():
            return tensor

        # Fast-path when we are not using pipeline parallelism.
        if self.pp_size == 1:
            return tensor

        # ------------------------------------------------------------------
        # 1.  Gather (shape, dtype, tensor_parallel flag, partition_dim) from
        #     every PP rank so that we can find the source rank.
        # ------------------------------------------------------------------
        if cache_key is not None and cache_key in self._tensor_spec_output_cache:
            tensor_spec_output = self._tensor_spec_output_cache[cache_key]
        else:
            if tensor is not None:
                shape = tensor.shape
                dtype = tensor.dtype
                tensor_parallel = getattr(tensor, "tensor_model_parallel", None)
                partition_dim = getattr(tensor, "partition_dim", None)
                tensor_spec = (shape, dtype, tensor_parallel, partition_dim)
            else:
                tensor_spec = None

            tensor_spec_output: list[Optional[tuple]] = [None] * self.pp_size
            torch.distributed.all_gather_object(
                tensor_spec_output, tensor_spec, group=self.pp_group
            )
            self._tensor_spec_output_cache[cache_key] = tensor_spec_output

        # ------------------------------------------------------------------
        # 2.  Identify the first owning rank (pick the lowest-ranked PP stage
        #     that holds the tensor).  Certain architectures (MLA / DeepSeek-V3,
        #     MTP, or models with tied embeddings) legitimately place the same
        #     weight on more than one PP rank, so we must *not* raise on
        #     duplicates – instead we deterministically pick the first owner
        #     as the broadcast source.
        # ------------------------------------------------------------------
        target_tensor_spec = None
        src_rank = None  # Rank *inside* the PP group.
        for rank, spec in enumerate(tensor_spec_output):
            if spec is not None and target_tensor_spec is None:
                target_tensor_spec = spec
                src_rank = rank

        if target_tensor_spec is None:
            # No rank had the tensor – this is an error in the caller.
            raise ValueError(
                f"Object must exist on at least one PP rank. "
                f"megatron_param={self.megatron_param}, hf_param={self.hf_param}, "
                f"cache_key={cache_key}"
            )

        # ------------------------------------------------------------------
        # 3.  Ensure every rank has an allocated tensor with the right shape
        #     and dtype before the broadcast.
        # ------------------------------------------------------------------
        if tensor is None:
            shape, dtype, tensor_parallel, partition_dim = target_tensor_spec
            # Use CPU by default, unless CUDA is available
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            tensor = torch.empty(shape, dtype=dtype, device=device)
            if tensor_parallel is not None:
                tensor.tensor_model_parallel = tensor_parallel
            if partition_dim is not None:
                tensor.partition_dim = partition_dim

        # ------------------------------------------------------------------
        # 4.  Broadcast from the source PP rank to all other PP ranks.
        # ------------------------------------------------------------------
        global_src = torch.distributed.get_global_rank(group=self.pp_group, group_rank=src_rank)
        torch.distributed.broadcast(tensor, src=global_src, group=self.pp_group)

        return tensor

    def broadcast_obj_from_pp_rank(
        self, obj: Optional[Any], cache_key: Optional[str] = None
    ) -> Any:
        """Broadcast any Python object from the PP rank that owns it.

        This method is useful for broadcasting configuration objects or
        other metadata across pipeline parallel ranks. Results are cached
        after the first call to avoid redundant broadcasts.

        Args:
            obj (Optional[Any]): Object to broadcast (None on non-owning ranks).
            cache_key (Optional[str]): Optional cache key. If not provided,
                no caching will be performed.

        Returns:
            Any: Broadcasted object on all ranks.

        Raises:
            ValueError: If object does not exist on any rank.
        """
        if _SKIP_PP_BROADCAST.get():
            return obj

        if self.pp_size == 1:
            if obj is None:
                raise ValueError(
                    f"{self.megatron_param}: Object must exist on at least one PP rank"
                )
            return obj

        # Check if we already have a cached result (only if cache_key is provided)
        if cache_key is not None and cache_key in self._broadcast_obj_cache:
            return self._broadcast_obj_cache[cache_key]

        # ------------------------------------------------------------------
        # 1. Gather presence flags from all PP ranks to find the source rank
        # ------------------------------------------------------------------
        has_obj = obj is not None
        obj_flags = [None] * self.pp_size
        torch.distributed.all_gather_object(obj_flags, has_obj, group=self.pp_group)

        # ------------------------------------------------------------------
        # 2. Identify the first owning rank (lowest PP stage with the object).
        #    Certain architectures (MLA, MTP, tied embeddings) place the same
        #    parameter on multiple PP ranks — pick the first owner.
        # ------------------------------------------------------------------
        src_rank = None  # Rank *inside* the PP group
        for rank, flag in enumerate(obj_flags):
            if flag:
                src_rank = rank
                break

        if src_rank is None:
            raise ValueError(f"{self.megatron_param}: Object must exist on at least one PP rank")

        # ------------------------------------------------------------------
        # 3. Broadcast the object from the source rank to all ranks
        # ------------------------------------------------------------------

        # Use broadcast_object_list which is more robust than all_gather_object
        obj_list = [obj]
        pp_ranks = torch.distributed.get_process_group_ranks(self.pp_group)
        global_src = pp_ranks[src_rank]
        torch.distributed.broadcast_object_list(obj_list, src=global_src, group=self.pp_group)

        result = obj_list[0]

        # Cache the result for future calls (only if cache_key is provided)
        if cache_key is not None:
            self._broadcast_obj_cache[cache_key] = result

        return result

    def clear_broadcast_cache(self):
        """Clear the broadcast object cache.

        This can be useful for testing or if the objects being broadcast
        might change during the lifetime of the mapping.
        """
        self._broadcast_obj_cache.clear()

    def clear_tensor_spec_output_cache(self):
        """Clear the tensor spec output cache.

        This can be useful for testing or if the tensor spec output
        might change during the lifetime of the mapping.
        """
        self._tensor_spec_output_cache.clear()

    def broadcast_tensor_to_tp_ranks(self, tensor: torch.Tensor, src_rank: int = 0) -> torch.Tensor:
        """Broadcast a tensor to all TP ranks.

        Args:
            tensor (torch.Tensor): The tensor to broadcast.
            src_rank (int, optional): The source rank within the TP group.
                Defaults to 0.

        Returns:
            torch.Tensor: The broadcasted tensor.
        """
        if self.tp_size == 1:
            return tensor

        global_src = torch.distributed.get_global_rank(group=self.tp_group, group_rank=src_rank)
        torch.distributed.broadcast(tensor, src=global_src, group=self.tp_group)
        return tensor

    def _get_shard_spec(
        self,
        *,
        shard_size: int,
        megatron_module: nn.Module | None = None,
        global_size: int | None = None,
        global_size_attr: str | None = None,
    ) -> tuple[int, int]:
        """Get the unique TP shard count and shard rank for the current TP rank.

        This supports layouts where TP ranks may contain replicated copies of a
        smaller set of unique shards, such as KV-head sharding when
        ``num_query_groups < tensor_model_parallel_size``.

        Args:
            shard_size: Number of elements in the local shard along the sharded axis.
            megatron_module: Optional Megatron module carrying layout metadata.
            global_size: Explicit global size along the sharded axis.
            global_size_attr: Optional module attribute that stores the global
                size along the sharded axis.

        Returns:
            Tuple of ``(shard_world_size, shard_rank)`` where ``shard_world_size``
            is the number of unique shards and ``shard_rank`` is the current
            rank's unique shard index.
        """
        resolved_global_size = shard_size * self.tp_size
        if global_size is not None:
            resolved_global_size = global_size
        elif megatron_module is not None and global_size_attr is not None:
            resolved_global_size = getattr(megatron_module, global_size_attr, resolved_global_size)

        if resolved_global_size % shard_size != 0:
            raise ValueError(
                f"Invalid sharded layout: global_size={resolved_global_size}, shard_size={shard_size}"
            )

        shard_world_size = resolved_global_size // shard_size
        if self.tp_size % shard_world_size != 0:
            raise ValueError(
                f"Invalid replicated shard layout: tp_size={self.tp_size}, shard_world_size={shard_world_size}"
            )

        replicas = self.tp_size // shard_world_size
        shard_rank = self.tp_rank // replicas
        return shard_world_size, shard_rank

    def scatter_to_tp_ranks(
        self,
        splits: Optional[List[torch.Tensor]],
        output_shape: torch.Size,
        dtype: torch.dtype,
        device: torch.device,
        src_rank: int = 0,
    ) -> torch.Tensor:
        """Scatter tensor splits to TP ranks.

        Args:
            splits (Optional[List[torch.Tensor]]): A list of tensor shards to
                scatter. Only rank `src_rank` needs this.
            output_shape (torch.Size): The shape of the output tensor on each rank.
            dtype (torch.dtype): The data type of the output tensor.
            device (torch.device): The device for the output tensor.
            src_rank (int, optional): The source rank for the scatter operation.
                Defaults to 0.

        Returns:
            torch.Tensor: The scattered tensor shard on the current rank.
        """
        if self.tp_size == 1:
            return splits[0].to(device=device, dtype=dtype) if splits else None

        output = torch.empty(output_shape, dtype=dtype, device=device)
        global_src = torch.distributed.get_global_rank(group=self.tp_group, group_rank=src_rank)

        scatter_list = None
        if self.tp_rank == src_rank and splits:
            scatter_list = [s.to(device=device, dtype=dtype).contiguous() for s in splits]

        torch.distributed.scatter(output, scatter_list, src=global_src, group=self.tp_group)
        return output

    def gather_from_tp_ranks(self, tensor: torch.Tensor) -> List[torch.Tensor]:
        """Gather tensors from all TP ranks.

        Args:
            tensor (torch.Tensor): The tensor shard to be gathered from the
                current rank.

        Returns:
            List[torch.Tensor]: A list of tensor shards from all TP ranks.
        """
        if self.tp_size == 1:
            return [tensor]

        gathered = [torch.empty_like(tensor) for _ in range(self.tp_size)]
        torch.distributed.all_gather(gathered, tensor, group=self.tp_group)
        return gathered

    @staticmethod
    def _count_wildcard_groups(pattern: str) -> int:
        """Count the number of wildcard capture groups in a pattern.

        Args:
            pattern: Pattern string with * and ** wildcards

        Returns:
            Number of capture groups that will be generated

        Note:
            ** counts as 1 group, * counts as 1 group
            ** must be counted before * to avoid double-counting
        """
        count = 0
        remaining = pattern

        # Count ** patterns first
        while "**" in remaining:
            count += 1
            remaining = remaining.replace("**", "", 1)

        # Count remaining * patterns
        count += remaining.count("*")

        return count

    def _validate_patterns(self):
        """Validate wildcard consistency between patterns.

        Skipped automatically for grouped-export mappings where the megatron
        side intentionally has more wildcards than the HF side.
        """
        if getattr(self, "is_grouped_export", False):
            return
        megatron_param_wildcards = self._count_wildcard_groups(self.megatron_param)
        if isinstance(self.hf_param, str):
            hf_param_wildcards = self._count_wildcard_groups(self.hf_param)
            if megatron_param_wildcards != hf_param_wildcards:
                raise ValueError(
                    f"Wildcard count mismatch: megatron_param='{self.megatron_param}' has "
                    f"{megatron_param_wildcards} wildcards, hf_param='{self.hf_param}' has {hf_param_wildcards}"
                )
        else:
            for key, pattern in self.hf_param.items():
                hf_param_wildcards = self._count_wildcard_groups(pattern)
                if megatron_param_wildcards != hf_param_wildcards:
                    raise ValueError(
                        f"Wildcard count mismatch: megatron_param='{self.megatron_param}' has "
                        f"{megatron_param_wildcards} wildcards, hf_param['{key}']='{pattern}' has {hf_param_wildcards}"
                    )

    def _normalize_expert_param_name(self, param_name: str) -> str:
        """Normalize expert parameter name by replacing trailing numbers with 0.
        e.g. experts.weight15 -> experts.weight0, experts.bias15 -> experts.bias0

        Args:
            param_name (str): Parameter name that may end with a number.

        Returns:
            str: Parameter name with trailing number replaced by 0.
        """
        # Use regex to replace any trailing number with 0
        return re.sub(r"\d+$", "0", param_name)

    def _get_config(self, module: nn.Module) -> Any:
        """Extract configuration from module hierarchy."""
        current = module
        while current is not None:
            if hasattr(current, "config"):
                return current.config
            # Try parent module
            if hasattr(current, "_parent"):
                current = current._parent
            else:
                # Walk up the module tree
                for parent_module in module.modules():
                    for child_name, child_module in parent_module.named_children():
                        if child_module is current:
                            current = parent_module
                            break
                    else:
                        continue
                    break
                else:
                    current = None

        raise ValueError(
            f"Could not find config in module hierarchy for {module.__class__.__name__}. "
            f"Ensure the module or its parent has a 'config' attribute."
        )

    def gather_from_ep_ranks(
        self,
        megatron_weights: Optional[torch.Tensor],
        megatron_module: Optional[MegatronModule],
        hf_param_name: Optional[str],
    ) -> Dict[str, torch.Tensor]:
        """Handle expert parallel weight gathering for MoE models.

        This method gathers expert weights across expert-parallel (EP) ranks and
        returns a mapping from HF parameter names to the corresponding tensors
        from each EP rank. Call this only for confirmed expert parameters
        (self.is_expert is True), typically after TP gathering/concatenation in
        the export path (Megatron → HF).

        Behavior and notation:
        - Let E be the total number of experts (e.g., config.num_moe_experts) and
          S be the expert-parallel size (ep_size). We assume E % S == 0.
        - Each EP rank owns E/S experts. For a given parameter name, we infer a
          local expert index L (0 ≤ L < E/S) on the current EP rank from the
          global expert id embedded in the name (works for both .weight and .bias).
        - The set of global expert ids that correspond to this local index L
          across all EP ranks is: {L + k * (E/S) | k ∈ [0, S-1]}.

        Communication and outputs:
        - We perform an all_gather over the EP group to collect the tensor from
          every EP rank into a list ordered by EP rank id.
        - For each EP rank k, we construct the HF parameter name by replacing the
          expert id in `hf_param_name` with (L + k * (E/S)), preserving the rest
          of the path, and map that name to the gathered tensor from rank k.

        Example:
        - E = 8, S = 2 → E/S = 4. Experts are distributed as:
          Rank 0: [0, 1, 2, 3], Rank 1: [4, 5, 6, 7].
          If the local index L = 0 (derived from the param name), this returns:
          {"...experts.0.weight": tensor_from_rank0, "...experts.4.weight": tensor_from_rank1}

        Args:
            megatron_weights (Optional[torch.Tensor]): The local expert weight tensor
                (after any TP handling) on this EP rank.
            megatron_module (Optional[MegatronModule]): The Megatron module containing
                configuration (used to determine E and E/S). Can be None on non-owning PP
                ranks; values will be broadcast across PP.
            hf_param_name (Optional[str]): HF parameter name template for the current
                (local) expert on this rank. The expert id within this string is replaced
                with the appropriate global expert ids for each EP rank.

        Returns:
            Dict[str, torch.Tensor]: Mapping from HF parameter names (one per EP rank)
            to the corresponding expert tensors gathered from each EP rank.
        """
        # Fast path for EP=1: no gathering needed, just return with the given name.
        if self.ep_size == 1:
            return {str(hf_param_name): megatron_weights}

        if megatron_module is None:
            num_experts_per_rank = self.broadcast_obj_from_pp_rank(None, "num_experts_per_rank")
        else:
            model_config = self._get_config(megatron_module)
            num_experts = model_config.num_moe_experts
            num_experts_per_rank = num_experts // self.ep_size
            num_experts_per_rank = self.broadcast_obj_from_pp_rank(
                num_experts_per_rank, "num_experts_per_rank"
            )

        global_expert_number = extract_expert_number_from_param(self.megatron_param)
        local_expert_number = global_expert_number % num_experts_per_rank

        # Compute global expert numbers for all EP ranks
        # use regex to replace the local expert number with the global expert number
        gathered_expert_param_names = [
            re.sub(
                r"experts\.(\d+)",
                f"experts.{int(local_expert_number) + num_experts_per_rank * i}",
                str(hf_param_name),
            )
            for i in range(self.ep_size)
        ]
        assert (
            str(hf_param_name) in gathered_expert_param_names
        ), f"hf_param_name {hf_param_name} not in gathered_expert_param_names {gathered_expert_param_names}"

        # Gather weights from all EP ranks
        gathered_weights = [torch.empty_like(megatron_weights) for _ in range(self.ep_size)]
        torch.distributed.all_gather(gathered_weights, megatron_weights, group=self.ep_group)

        # this should be in the right order because of the all-gather
        weights_dict = {}
        for i, param_name in enumerate(gathered_expert_param_names):
            if param_name in weights_dict:
                weights_dict[param_name] = torch.cat(
                    [weights_dict[param_name], gathered_weights[i].unsqueeze(0)], dim=0
                )
            else:
                weights_dict[param_name] = gathered_weights[i].unsqueeze(0)
        for param_name in weights_dict:
            weights_dict[param_name] = weights_dict[param_name].squeeze(0)
        return weights_dict

    def gather_from_ep_ranks_scale(
        self,
        megatron_weights: Optional[torch.Tensor],
        megatron_module: Optional[MegatronModule],
        hf_param_name: Optional[str],
    ) -> Dict[str, torch.Tensor]:
        """Gather expert scale tensors using the same staging path as expert weights.

        Only the leading dimension added while grouping gathered tensors is removed,
        so singleton dimensions belonging to the scale's block grid are preserved.

        Args:
            megatron_weights (Optional[torch.Tensor]): The local expert weight tensor
                (after any TP handling) on this EP rank.
            megatron_module (Optional[MegatronModule]): The Megatron module containing
                configuration (used to determine E and E/S). Can be None on non-owning PP
                ranks; values will be broadcast across PP.
            hf_param_name (Optional[str]): HF parameter name template for the current
                (local) expert on this rank. The expert id within this string is replaced
                with the appropriate global expert ids for each EP rank.

        Returns:
            Dict[str, torch.Tensor]: Mapping from HF parameter names (one per EP rank)
            to the corresponding expert tensors gathered from each EP rank.
        """
        if self.ep_size == 1:
            return {str(hf_param_name): megatron_weights}

        if megatron_module is None:
            num_experts_per_rank = self.broadcast_obj_from_pp_rank(None, "num_experts_per_rank")
        else:
            model_config = self._get_config(megatron_module)
            num_experts = model_config.num_moe_experts
            num_experts_per_rank = num_experts // self.ep_size
            num_experts_per_rank = self.broadcast_obj_from_pp_rank(
                num_experts_per_rank, "num_experts_per_rank"
            )

        global_expert_number = extract_expert_number_from_param(self.megatron_param)
        local_expert_number = global_expert_number % num_experts_per_rank

        # Compute global expert numbers for all EP ranks
        # use regex to replace the local expert number with the global expert number
        gathered_expert_param_names = [
            re.sub(
                r"experts\.(\d+)",
                f"experts.{int(local_expert_number) + num_experts_per_rank * i}",
                str(hf_param_name),
            )
            for i in range(self.ep_size)
        ]
        assert (
            str(hf_param_name) in gathered_expert_param_names
        ), f"hf_param_name {hf_param_name} not in gathered_expert_param_names {gathered_expert_param_names}"

        # Gather weights from all EP ranks
        gathered_weights = [torch.empty_like(megatron_weights) for _ in range(self.ep_size)]
        torch.distributed.all_gather(gathered_weights, megatron_weights, group=self.ep_group)

        # this should be in the right order because of the all-gather
        weights_dict = {}
        for i, param_name in enumerate(gathered_expert_param_names):
            if param_name in weights_dict:
                weights_dict[param_name] = torch.cat(
                    [weights_dict[param_name], gathered_weights[i].unsqueeze(0)], dim=0
                )
            else:
                weights_dict[param_name] = gathered_weights[i].unsqueeze(0)
        for param_name in weights_dict:
            weights_dict[param_name] = weights_dict[param_name].squeeze(0)
        return weights_dict

    def maybe_dequantize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Dequantize FP8 tensor if needed."""
        if HAVE_TE_FP8_TENSOR_CLASS and isinstance(tensor, FP8_TENSOR_CLASS):
            return tensor.dequantize(dtype=tensor.dtype)
        return tensor


class DirectMapping(MegatronParamMapping[torch.Tensor]):
    """Direct 1:1 weight mapping with no transformation or tensor parallelism."""

    def _local_hf_sharding(
        self, weight: torch.Tensor, megatron_module: nn.Module
    ) -> tuple[LocalHFShardGroup, int | None, int]:
        """Describe direct parameters as replicated local views."""
        return "replicated", None, 1

    def hf_to_megatron(self, hf_weights: torch.Tensor, megatron_module: nn.Module) -> torch.Tensor:
        """Direct copy - no transformation or distribution."""
        return hf_weights

    def megatron_to_hf(
        self, megatron_weights: Optional[torch.Tensor], megatron_module: Optional[nn.Module]
    ) -> Dict[str, torch.Tensor]:
        """Direct copy with PP broadcast."""
        # Handle cross-PP broadcast
        megatron_weights = self.broadcast_from_pp_rank(
            megatron_weights, cache_key=str(self.hf_param)
        )

        if megatron_weights is None:
            return {}

        # Dequantize if needed
        megatron_weights = self.maybe_dequantize(megatron_weights)

        return {str(self.hf_param): megatron_weights}


class ColumnParallelMapping(MegatronParamMapping[torch.Tensor]):
    """
    Mapping for column-parallel linear and embedding weights.

    Column-parallel layers in Megatron split the output dimension across tensor
    parallel ranks. This is used for layers where each rank computes a portion
    of the output features independently, such as:
    - Embedding layers (split vocabulary)
    - Linear layers producing hidden states (e.g., QKV projections, MLP up projections)

    The weight matrix is partitioned along dimension 0 (rows), so each TP rank
    holds a subset of output features while maintaining all input features.

    **Sharding pattern**
    -   Original weight: `[output_features, input_features]`
    -   Rank 0: `[output_features/tp_size, input_features]`
    -   Rank 1: `[output_features/tp_size, input_features]`
    -   ...

    **Forward path (HuggingFace → Megatron)**
    1.  Validate divisibility: output dimension must be divisible by tp_size
    2.  Split: Chunk tensor along dim 0 into tp_size equal parts
    3.  Scatter: Distribute chunks to respective TP ranks

    **Reverse path (Megatron → HuggingFace)**
    1.  Broadcast: Ensure all PP ranks have the tensor
    2.  Gather: Collect chunks from all TP ranks
    3.  Concatenate: Reassemble along dim 0 on rank 0

    Example:
        .. code-block:: python

            # For a weight of shape [4096, 1024] with tp_size=4:
            # Each rank gets [1024, 1024] after column-parallel split
            mapping = ColumnParallelMapping("linear.weight", "transformer.linear.weight")
            megatron_weights = mapping.hf_to_megatron(hf_weight, megatron_module)
            # megatron_weights.shape = [1024, 1024] on each rank

    Note:
        This mapping also handles bias terms, which are 1D tensors split
        along their only dimension following the same pattern.
    """

    def local_mxfp8_params(
        self,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        *,
        global_param_name: str,
        megatron_module: nn.Module,
    ) -> tuple[LocalMXFP8Param, ...]:
        """Project a local column-parallel native MXFP8 weight without gathering."""
        _validate_native_mxfp8_pair(weight, weight_scale, global_param_name)

        global_weight_shape = list(weight.shape)
        global_weight_shape[0] *= self.tp_size
        shard_group: MXFP8ShardGroup = "etp" if self.is_expert else "tp"
        return (
            LocalMXFP8Param(
                name=str(self.hf_param),
                weight=weight,
                weight_scale=weight_scale,
                global_weight_shape=torch.Size(global_weight_shape),
                shard_group=shard_group,
                shard_dim=0,
            ),
        )

    def _local_hf_sharding(
        self, weight: torch.Tensor, megatron_module: nn.Module
    ) -> tuple[LocalHFShardGroup, int | None, int]:
        """Describe column-parallel local views."""
        return ("etp" if self.is_expert else "tp"), 0, self.tp_size

    def hf_to_megatron(self, hf_weights: torch.Tensor, megatron_module: nn.Module) -> torch.Tensor:
        """Split weight along dim 0 and distribute to TP ranks."""
        if self.tp_size == 1:
            return hf_weights

        # Some parameters are named with global expert number, e.g. experts.weight15,
        # normalize it to experts.weight0, note we are only use the shape, dtype, device info,
        # not the actual value, so it is safe to do this.
        normalized_param = self._normalize_expert_param_name(self.megatron_param)
        _, target_param = get_module_and_param_from_name(megatron_module, normalized_param)

        # On rank 0, check for divisibility and split
        if self.tp_rank == 0:
            if hf_weights is None:
                raise ValueError("hf_weights should not be None on rank 0")

            # Dtype may differ (e.g. MambaMixer A_log is FP32 in MCore but BF16
            # in HF checkpoints). Cast to match the Megatron parameter so the
            # scatter doesn't fail on dtype mismatch.
            if hf_weights.dtype != target_param.dtype:
                if not getattr(ColumnParallelMapping, "_dtype_warned", False):
                    ColumnParallelMapping._dtype_warned = True
                    logger.warning(
                        f"Dtype mismatch: HF weights are {hf_weights.dtype} but Megatron "
                        f"module uses {target_param.dtype}. Casting all mismatched weights "
                        f"to {target_param.dtype} (further warnings suppressed)."
                    )
                hf_weights = hf_weights.to(target_param.dtype)

            actual_dim0_size = hf_weights.shape[0]
            # DTensor.shape is already the global shape across TP ranks, while
            # a regular Megatron parameter stores only its local TP shard.
            expect_dim0_size = _get_mapping_shape(target_param)[0]
            if not isinstance(target_param, DTensor):
                expect_dim0_size *= self.tp_size
            if actual_dim0_size != expect_dim0_size:
                # Only vocabulary tensors may be zero-padded, including under a wrapper
                # prefix such as a VLM's ``language_model.``.
                vocab_params = (
                    "embedding.word_embeddings.weight",
                    "output_layer.weight",
                    "output_layer.bias",
                )
                assert any(
                    self.megatron_param == name or self.megatron_param.endswith("." + name)
                    for name in vocab_params
                ), f"{hf_weights.shape=} {target_param.shape=} {self.tp_size=} {self.megatron_param=} {self.hf_param=}"
                hf_weights = _pad_right_dim0(
                    hf_weights, pad_size=expect_dim0_size - actual_dim0_size
                )

            # For bias (1D), we still split along dim 0
            # For weight (2D), we split along dim 0 (output dimension)
            full_size = hf_weights.shape[0]
            if full_size % self.tp_size != 0:
                raise ValueError(
                    f"Cannot evenly split dimension 0 size {full_size} across {self.tp_size} TP ranks"
                )
            splits = torch.chunk(hf_weights, self.tp_size, dim=0)

        else:
            splits = None

        if isinstance(target_param, DTensor):
            output_shape = target_param.orig_param.shape
        else:
            output_shape = _get_mapping_shape(target_param)
        # Scatter to all ranks. Each rank gets its sharded shape from its module.
        return self.scatter_to_tp_ranks(
            splits, output_shape, target_param.dtype, target_param.device
        )

    def megatron_to_hf(
        self, megatron_weights: Optional[torch.Tensor], megatron_module: Optional[nn.Module]
    ) -> Dict[str, torch.Tensor]:
        """Gather from all TP ranks and concatenate."""
        # Handle cross-PP broadcast
        megatron_weights = self.broadcast_from_pp_rank(
            megatron_weights, cache_key=str(self.hf_param)
        )

        if megatron_weights is None:
            return {}

        # Dequantize if needed
        megatron_weights = self.maybe_dequantize(megatron_weights)

        if self.tp_size == 1 or _module_uses_fsdp(megatron_module):
            full_weights = megatron_weights
        else:
            # Gather from all TP ranks
            gathered = self.gather_from_tp_ranks(megatron_weights)
            full_weights = torch.cat(gathered, dim=0)

        if self.is_expert:
            return self.gather_from_ep_ranks(full_weights, megatron_module, self.hf_param)

        return {str(self.hf_param): full_weights}

    def megatron_to_hf_quant(
        self,
        megatron_weights: Optional[torch.Tensor],
        megatron_module: Optional[nn.Module],
        quantization_checker: Callable[[str], bool],
        quant_fn: Callable[..., Tuple[torch.Tensor, torch.Tensor]],
        quant_block_size: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Gather from all TP ranks and concatenate with quantization before PP broadcast."""
        if not quantization_checker(str(self.megatron_param)):
            return self.megatron_to_hf(megatron_weights, megatron_module)

        q_weight, scale = None, None
        if megatron_weights is not None:
            megatron_weights = self.maybe_dequantize(megatron_weights)
            assert len(megatron_weights.shape) == 2, "Megatron weights must be 2D for quantization"
            q_weight, scale = quant_fn(megatron_weights, quant_block_size)

        q_weight = self.broadcast_from_pp_rank(q_weight, cache_key=str(self.hf_param) + "_q")
        if q_weight is None:
            return {}

        scale = self.broadcast_from_pp_rank(scale, cache_key=str(self.hf_param) + "_scale")

        if self.tp_size == 1:
            full_q_weight = q_weight
            full_scale = scale
        else:
            gathered_q = self.gather_from_tp_ranks(q_weight)
            full_q_weight = torch.cat(gathered_q, dim=0)

            gathered_s = self.gather_from_tp_ranks(scale)
            full_scale = torch.cat(gathered_s, dim=0)

        if self.is_expert:
            q_weight_dict = self.gather_from_ep_ranks(full_q_weight, megatron_module, self.hf_param)
            s_dict = self.gather_from_ep_ranks_scale(full_scale, megatron_module, self.hf_param)
            for k, v in s_dict.items():
                scale_name = k + "_scale_inv"
                q_weight_dict[scale_name] = v
            return q_weight_dict

        hf_name = str(self.hf_param)
        scale_name = hf_name + "_scale_inv"

        quant_result = {hf_name: full_q_weight, scale_name: full_scale}

        return quant_result


def _pad_right_dim0(x: torch.Tensor, pad_size: int) -> torch.Tensor:
    padder = torch.zeros((pad_size, *x.shape[1:]), dtype=x.dtype, device=x.device)
    return torch.cat([x, padder], dim=0)


class RowParallelMapping(MegatronParamMapping[torch.Tensor]):
    """Mapping for **row-parallel** linear weights.

    Megatron shards row-parallel tensors along **dimension 1** (the *input*
    dimension of a linear layer).

    **Forward path (external → Megatron)**
    1.  Rank 0 validates that the *second* dimension is divisible by `tp_size`.
    2.  Rank 0 splits the tensor with `torch.chunk(..., dim=1)` producing
        `tp_size` equally-sized shards.
    3.  The shards are **scattered** so that every TP rank receives exactly one
        shard matching the shape of its local Megatron parameter.

    **Reverse path (Megatron → external)**
    1.  The local Megatron parameter (which may live on any PP rank) is
        broadcast to all PP ranks so that the gather step can be collective.
    2.  All TP ranks **gather** their shard.
    3.  Rank 0 concatenates the gathered list along dim 1 to reconstruct the
        original unsharded weight and emits it under the external (HF) name.
    """

    def local_mxfp8_params(
        self,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        *,
        global_param_name: str,
        megatron_module: nn.Module,
    ) -> tuple[LocalMXFP8Param, ...]:
        """Project a local row-parallel native MXFP8 weight without gathering."""
        expected_scale_shape = torch.Size((*weight.shape[:-1], weight.shape[-1] // 32))
        if weight_scale.shape != expected_scale_shape:
            raise ValueError(
                f"{global_param_name}: expected native MXFP8 scale shape {tuple(expected_scale_shape)}, "
                f"got {tuple(weight_scale.shape)}"
            )

        global_weight_shape = list(weight.shape)
        global_weight_shape[1] *= self.tp_size
        shard_group: MXFP8ShardGroup = "etp" if self.is_expert else "tp"
        return (
            LocalMXFP8Param(
                name=str(self.hf_param),
                weight=weight,
                weight_scale=weight_scale,
                global_weight_shape=torch.Size(global_weight_shape),
                shard_group=shard_group,
                shard_dim=1,
            ),
        )

    def _local_hf_sharding(
        self, weight: torch.Tensor, megatron_module: nn.Module
    ) -> tuple[LocalHFShardGroup, int | None, int]:
        """Describe row-parallel local views."""
        if weight.ndim == 1:
            return "replicated", None, 1
        return ("etp" if self.is_expert else "tp"), 1, self.tp_size

    def hf_to_megatron(self, hf_weights: torch.Tensor, megatron_module: nn.Module) -> torch.Tensor:
        """Split weight along dim 1 and distribute to TP ranks."""
        # Some parameters are named with global expert number, e.g. experts.weight15,
        # normalize it to experts.weight0, note we are only use the shape, dtype, device info,
        # not the actual value, so it is safe to do this.
        normalized_param = self._normalize_expert_param_name(self.megatron_param)
        _, target_param = get_module_and_param_from_name(megatron_module, normalized_param)

        # HF fused expert weights (e.g. down_proj) may be stored in [in, out]
        # layout while Megatron expects [out, in]. Detect via the unsharded dim:
        # RowParallel TP does not split dim 0. GTP may split it further in storage,
        # so compare against the logical TP-local shape.
        target_shape = _get_mapping_shape(target_param)
        if (
            hf_weights is not None
            and hf_weights.ndim == 2
            and hf_weights.shape[0] != target_shape[0]
            and hf_weights.shape[1] == target_shape[0]
        ):
            hf_weights = hf_weights.t().contiguous()

        if self.tp_size == 1:
            return hf_weights

        # On rank 0, check for divisibility and split
        if self.tp_rank == 0:
            if hf_weights is None:
                raise ValueError("hf_weights should not be None on rank 0")

            # bias (1D) is replicated across tp ranks
            # For weight (2D), we split along dim 1
            if hf_weights.ndim == 1:
                splits = [hf_weights] * self.tp_size
            else:
                assert hf_weights.ndim == 2
                full_size = hf_weights.shape[1]
                if full_size % self.tp_size != 0:
                    raise ValueError(
                        f"Cannot evenly split dimension 0 size {full_size} across {self.tp_size} TP ranks"
                    )
                splits = torch.chunk(hf_weights, self.tp_size, dim=1)

        else:
            splits = None

        if isinstance(target_param, DTensor):
            output_shape = target_param.orig_param.shape
        else:
            output_shape = _get_mapping_shape(target_param)
        # Scatter to all ranks. Each rank gets its sharded shape from its module.
        return self.scatter_to_tp_ranks(
            splits, output_shape, target_param.dtype, target_param.device
        )

    def megatron_to_hf(
        self, megatron_weights: Optional[torch.Tensor], megatron_module: Optional[nn.Module]
    ) -> Dict[str, torch.Tensor]:
        """Gather from all TP ranks and concatenate."""
        # Handle cross-PP broadcast
        megatron_weights = self.broadcast_from_pp_rank(
            megatron_weights, cache_key=str(self.hf_param)
        )

        if megatron_weights is None:
            return {}

        # Dequantize if needed
        megatron_weights = self.maybe_dequantize(megatron_weights)

        if (
            self.tp_size == 1
            or len(megatron_weights.shape) == 1
            or _module_uses_fsdp(megatron_module)
        ):
            # bias is unsharded in row parallel, so we can just return it
            full_weights = megatron_weights
        else:
            gathered = self.gather_from_tp_ranks(megatron_weights)
            full_weights = torch.cat(gathered, dim=1)

        if self.is_expert:
            return self.gather_from_ep_ranks(full_weights, megatron_module, self.hf_param)

        return {str(self.hf_param): full_weights}

    def megatron_to_hf_quant(
        self,
        megatron_weights: Optional[torch.Tensor],
        megatron_module: Optional[nn.Module],
        quantization_checker: Callable[[str], bool],
        quant_fn: Callable[..., Tuple[torch.Tensor, torch.Tensor]],
        quant_block_size: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Gather from all TP ranks and concatenate with quantization before PP broadcast."""
        if not quantization_checker(str(self.megatron_param)):
            return self.megatron_to_hf(megatron_weights, megatron_module)

        q_weight, scale = None, None
        if megatron_weights is not None:
            megatron_weights = self.maybe_dequantize(megatron_weights)
            assert len(megatron_weights.shape) == 2, "Megatron weights must be 2D for quantization"
            q_weight, scale = quant_fn(megatron_weights, quant_block_size)

        q_weight = self.broadcast_from_pp_rank(q_weight, cache_key=str(self.hf_param) + "_q")
        if q_weight is None:
            return {}

        scale = self.broadcast_from_pp_rank(scale, cache_key=str(self.hf_param) + "_scale")

        if self.tp_size == 1 or len(q_weight.shape) == 1:
            # bias is unsharded in row parallel, so we can just return it
            full_q_weight = q_weight
            full_scale = scale
        else:
            gathered_q = self.gather_from_tp_ranks(q_weight)
            full_q_weight = torch.cat(gathered_q, dim=1)
            # scale is typically not sharded along dim 1, so we do not concatenate it.
            gathered_scale = self.gather_from_tp_ranks(scale)
            full_scale = torch.cat(gathered_scale, dim=1)

        if self.is_expert:
            q_weight_dict = self.gather_from_ep_ranks(full_q_weight, megatron_module, self.hf_param)
            s_dict = self.gather_from_ep_ranks_scale(full_scale, megatron_module, self.hf_param)
            for k, v in s_dict.items():
                scale_name = k + "_scale_inv"
                q_weight_dict[scale_name] = v
            return q_weight_dict

        hf_name = str(self.hf_param)
        scale_name = hf_name + "_scale_inv"

        quant_result = {hf_name: full_q_weight, scale_name: full_scale}

        return quant_result


class ReplicatedMapping(MegatronParamMapping[torch.Tensor]):
    """Mapping for weights that are **fully replicated** across TP ranks.

    Examples: layer-norm scales, biases, router weights in MoE, etc.

    These tensors exist in exactly the same form on *every* TP rank, so the
    mapping logic is trivial – but we still need to broadcast across TP ranks
    during *load* (HF → Megatron) and ensure we do **not** emit duplicates
    during *export* (Megatron → HF).
    """

    def _local_hf_sharding(
        self, weight: torch.Tensor, megatron_module: nn.Module
    ) -> tuple[LocalHFShardGroup, int | None, int]:
        """Describe fully replicated local views."""
        return "replicated", None, 1

    def hf_to_megatron(self, hf_weights: torch.Tensor, megatron_module: nn.Module) -> torch.Tensor:
        """Replicate weight to all TP ranks."""
        if hasattr(megatron_module, "weight"):
            target_device = megatron_module.weight.device
        else:
            # the parameter may not be called "weight"
            target_device = next(megatron_module.parameters()).device
        hf_weights = hf_weights.to(device=target_device)
        if self.tp_size == 1:
            return hf_weights

        # TODO(yuya): router.weight is on device cpu, need to check.
        if target_device.index != torch.cuda.current_device():
            hf_weights = hf_weights.to(torch.cuda.current_device())

        # All ranks need the full weight
        if self.tp_rank > 0:
            # Create empty tensor of correct shape
            hf_weights = torch.empty_like(hf_weights)

        # Broadcast from rank 0 to all TP ranks
        return self.broadcast_tensor_to_tp_ranks(hf_weights, src_rank=0)

    def megatron_to_hf(
        self, megatron_weights: Optional[torch.Tensor], megatron_module: Optional[nn.Module]
    ) -> Dict[str, torch.Tensor]:
        """Return weight only from rank 0 to avoid duplication."""
        # Handle cross-PP broadcast
        megatron_weights = self.broadcast_from_pp_rank(
            megatron_weights, cache_key=str(self.hf_param)
        )

        if megatron_weights is None:
            return {}

        # Dequantize if needed
        megatron_weights = self.maybe_dequantize(megatron_weights)

        if self.is_expert:
            return self.gather_from_ep_ranks(megatron_weights, megatron_module, self.hf_param)

        return {str(self.hf_param): megatron_weights}

    def megatron_to_hf_quant(
        self,
        megatron_weights: Optional[torch.Tensor],
        megatron_module: Optional[nn.Module],
        quantization_checker: Callable[[str], bool],
        quant_fn: Callable[..., Tuple[torch.Tensor, torch.Tensor]],
        quant_block_size: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, torch.Tensor]:
        # ReplicatedMapping doesn't need quantization, so we just call megatron_to_hf
        return self.megatron_to_hf(megatron_weights, megatron_module)


class AutoMapping(MegatronParamMapping[torch.Tensor]):
    """
    Smart mapping that automatically detects and applies the correct parallelism strategy.

    This mapping eliminates the need to manually specify whether a layer is
    column-parallel, row-parallel, or replicated. It examines the Megatron
    module at runtime and delegates to the appropriate specialized mapping.

    **Detection strategy**
    1. Check module class name against a registry of known types
    2. If unknown, examine module attributes (tensor_model_parallel, partition_dim)
    3. Delegate to appropriate mapping: ColumnParallel, RowParallel, or Replicated

    This abstraction is particularly useful for model-agnostic code where you
    don't know the parallelism type ahead of time, or when working with models
    that mix different parallelism strategies.

    **Built-in module recognition**
    -   Column-parallel: `ColumnParallelLinear`, `VocabParallelEmbedding`, etc.
    -   Row-parallel: `RowParallelLinear`, `TERowParallelLinear`
    -   Replicated: `LayerNorm`, `RMSNorm`, and other normalization layers

    **Dimension permutation**
    Supports optional tensor permutation via `permute_dims` parameter. This is useful
    for weights that need to be transposed or have their dimensions reordered during
    conversion. The same permutation is applied in both directions (HF→Megatron and
    Megatron→HF).

    Example:
        .. code-block:: python

            # Automatically handles any weight type
            mapping = AutoMapping(
                megatron_param="decoder.layers.*.mlp.linear_fc1.weight",
                hf_param="model.layers.*.mlp.gate_proj.weight"
            )

            # Works with column-parallel layers
            megatron_weights = mapping.hf_to_megatron(hf_weight, column_parallel_module)

            # Also works with normalization layers
            norm_weight = mapping.hf_to_megatron(hf_norm, layer_norm_module)

            # With dimension permutation (e.g., transpose)
            transpose_mapping = AutoMapping(
                megatron_param="vision_projection.weight",
                hf_param="multi_modal_projector.weight",
                permute_dims=(1, 0)  # Transpose dimensions
            )

            # Register custom module types
            AutoMapping.register_module_type("MyCustomLinear", "column")

    Note:
        If the parallelism type cannot be determined, the mapping will raise
        a descriptive error suggesting how to fix the issue.
    """

    # Module type registry
    _MODULE_TYPE_REGISTRY: Dict[str, set] = {
        "column": {
            "ColumnParallelLinear",
            "LinearCrossEntropyModule",
            "QuantColumnParallelLinear",
            "TEColumnParallelLinear",
            "TELayerNormColumnParallelLinear",
            "InferenceColumnParallelLinear",
            "InferenceLayerNormColumnParallelLinear",
            "TEColumnParallelGroupedLinear",
            "VocabParallelEmbedding",
            "DotProductAttention",  # for attention sink only
            "TEDotProductAttention",  # for attention sink only
        },
        "row": {
            "RowParallelLinear",
            "QuantRowParallelLinear",
            "TERowParallelLinear",
            "InferenceRowParallelLinear",
            "TERowParallelGroupedLinear",
        },
        "replicated": {
            # Normalization layers
            "TENorm",
            "FusedLayerNorm",
            "WrappedTorchNorm",
            "LayerNorm",
            "RMSNorm",
            "L2Norm",
            # Other non-parallel modules
            "InferenceTopKRouter",
            "IdentityOp",
            "LinearForLastLayer",
            "TopKRouter",
        },
    }
    _FUSED_LAYER_NORM_COLUMN_PARALLEL_SUBSTRING = "LayerNormColumnParallelLinear"

    @classmethod
    def register_module_type(cls, module_name: str, parallelism_type: str):
        """Register a new module type for automatic parallelism detection.

        Args:
            module_name (str): The name of the module class (e.g.,
                'MyColumnLinear').
            parallelism_type (str): One of 'column', 'row', or 'replicated'.
        """
        if parallelism_type not in cls._MODULE_TYPE_REGISTRY:
            raise ValueError(
                f"Invalid parallelism_type '{parallelism_type}'. "
                f"Must be one of {list(cls._MODULE_TYPE_REGISTRY.keys())}"
            )
        cls._MODULE_TYPE_REGISTRY[parallelism_type].add(module_name)

    def __init__(
        self, megatron_param: str, hf_param: str, permute_dims: Optional[Tuple[int, ...]] = None
    ):
        """Initialize TP-aware mapping.

        Args:
            megatron_param (str): Megatron parameter name pattern.
            hf_param (str): HuggingFace parameter name pattern.
            permute_dims (Optional[Tuple[int, ...]]): Dimension permutation to apply.
                If provided, the tensor will be permuted and made contiguous during conversion.
        """
        super().__init__(megatron_param, hf_param)

        # Cache for detected parallelism type and delegate mapping
        self._detected_type: Optional[str] = None
        self._mapping: Optional[MegatronParamMapping[torch.Tensor]] = None

        # Permutation settings
        self.permute_dims = permute_dims

    def _get_or_create_mapping(self, parallelism_type: str) -> MegatronParamMapping[torch.Tensor]:
        """Get or create the appropriate mapping for the given type."""
        if parallelism_type == "column":
            mapping = ColumnParallelMapping(self.megatron_param, self.hf_param)
        elif parallelism_type == "row":
            mapping = RowParallelMapping(self.megatron_param, self.hf_param)
        elif parallelism_type == "replicated":
            mapping = ReplicatedMapping(self.megatron_param, self.hf_param)
        else:
            raise ValueError(f"Unknown parallelism type: {parallelism_type}")

        # AutoMapping materializes this delegate lazily, after task construction.
        if self._pg_collection is not None:
            mapping.set_process_groups_from_pg_collection(self._pg_collection)
        return mapping

    def _detect_parallelism_type(self, module: nn.Module) -> str:
        """Detect parallelism type from module."""
        if is_modelopt_dynamic_module(module):
            module_type = module.get_original_cls_by_level(level=0).__name__
        else:
            module_type = type(module).__name__

        # Handle fused modules like TELayerNormColumnParallelLinear
        # These modules have both column-parallel weights (weight, bias)
        # and replicated layer norm weights (layer_norm_weight, layer_norm_bias)
        is_layernorm_column_parallel = (
            self._FUSED_LAYER_NORM_COLUMN_PARALLEL_SUBSTRING in module_type
        )
        if is_layernorm_column_parallel:
            # Check the actual parameter name to determine the correct parallelism type
            if self.megatron_param and (
                self.megatron_param.endswith("layer_norm_weight")
                or self.megatron_param.endswith("layer_norm_bias")
            ):
                return "replicated"
            # All other parameters (weight, bias) are column-parallel
            return "column"

        # Check registry first
        for parallelism, types in self._MODULE_TYPE_REGISTRY.items():
            if module_type in types:
                return parallelism

        # Fallback to inspecting module attributes
        if hasattr(module, "tensor_model_parallel"):
            if not module.tensor_model_parallel:
                return "replicated"

            # Check partition dimension
            partition_dim = getattr(module, "partition_dim", None)
            if partition_dim == 0:
                return "column"
            elif partition_dim == 1:
                return "row"

        # Fallback for normalization layers
        if any(norm in module_type for norm in ["Norm", "Normalization"]):
            return "replicated"

        # Check parallel_mode for TELinear
        if module_type == "TELinear":
            if module.parallel_mode == "column":
                return "column"
            elif module.parallel_mode == "row":
                return "row"
            else:
                return "replicated"

        # Cannot determine - raise informative error
        known_types = {p: sorted(list(t)) for p, t in self._MODULE_TYPE_REGISTRY.items()}

        raise ValueError(
            f"Cannot determine parallelism type for module '{module_type}' "
            f"at weight '{self.megatron_param}'.\n"
            f"Please use an explicit mapping type (e.g., ColumnParallelMapping) "
            f"or register the module type using:\n"
            f"  AutoMapping.register_module_type('{module_type}', 'column|row|replicated')\n\n"
            f"Currently known module types:\n{json.dumps(known_types, indent=2)}"
        )

    def local_mxfp8_params(
        self,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        *,
        global_param_name: str,
        megatron_module: nn.Module,
    ) -> tuple[LocalMXFP8Param, ...]:
        """Delegate direct native MXFP8 export to the detected mapping."""
        if not self.local_hf_param_specs(global_param_name):
            raise ValueError(
                f"{global_param_name}: mapping does not expose canonical local HF views"
            )
        if self.permute_dims is not None or getattr(self, "transpose_on_export", False):
            raise ValueError(
                f"{global_param_name}: native MXFP8 export does not support mapping transforms"
            )
        if megatron_module is None or _module_uses_fsdp(megatron_module):
            raise ValueError(
                f"{global_param_name}: native MXFP8 export does not support DTensor/FSDP parameters"
            )

        if self._mapping is None:
            try:
                self._detected_type = self._detect_parallelism_type(megatron_module)
                self._mapping = self._get_or_create_mapping(self._detected_type)
            except ValueError as error:
                raise ValueError(f"{global_param_name}: {error}") from error

        return self._mapping.local_mxfp8_params(
            weight,
            weight_scale,
            global_param_name=global_param_name,
            megatron_module=megatron_module,
        )

    def _local_hf_sharding(
        self, weight: torch.Tensor, megatron_module: nn.Module
    ) -> tuple[LocalHFShardGroup, int | None, int]:
        """Describe local HF sharding through the detected concrete mapping."""
        mapping = self._mapping
        if mapping is None:
            try:
                parallelism_type = self._detect_parallelism_type(megatron_module)
                mapping = self._get_or_create_mapping(parallelism_type)
            except ValueError as error:
                raise ValueError(f"{self.megatron_param}: {error}") from error
        return mapping._local_hf_sharding(weight, megatron_module)

    def hf_to_megatron(self, hf_weights: torch.Tensor, megatron_module: nn.Module) -> torch.Tensor:
        """Delegate to appropriate mapping based on module type."""
        # Apply permutation if specified (before distribution).
        # Must be applied on ALL ranks (not just tp_rank 0) because some delegate
        # mappings (e.g. ReplicatedMapping) expect hf_weights to have the correct
        # shape on every rank.
        if self.permute_dims is not None:
            hf_weights = torch.permute(hf_weights, self.permute_dims).contiguous()

        # Detect type and create delegate on first use
        if self._mapping is None:
            self._detected_type = self._detect_parallelism_type(megatron_module)
            self._mapping = self._get_or_create_mapping(self._detected_type)

        return self._mapping.hf_to_megatron(hf_weights, megatron_module)

    def megatron_to_hf(
        self, megatron_weights: Optional[torch.Tensor], megatron_module: Optional[nn.Module]
    ) -> Dict[str, torch.Tensor]:
        """Delegate to appropriate mapping based on module type."""
        # Need to determine type even if module is None (different PP rank)
        assert self.megatron_param is not None, "`megatron_param` is required for AutoMapping."

        mapping = self._mapping
        if mapping is None:
            if megatron_module is not None:
                detected_type = self._detect_parallelism_type(megatron_module)
                # Broadcast to other ranks
                detected_type = self.broadcast_obj_from_pp_rank(detected_type, "detected_type")
            else:
                # Receive from owning rank
                detected_type = self.broadcast_obj_from_pp_rank(None, "detected_type")

            # A stage-local export has nothing to receive for a non-local task.
            # Full export must instead reject missing owners in the broadcast helper.
            if _SKIP_PP_BROADCAST.get() and detected_type is None:
                return {}

            mapping = self._get_or_create_mapping(detected_type)
            # A PP-local export skips this task on non-owning stages. Do not
            # cache the lazy delegate only on the owner, or a later full export
            # could enter different PP collectives on different ranks.
            if not _SKIP_PP_BROADCAST.get():
                self._detected_type = detected_type
                self._mapping = mapping

        result = mapping.megatron_to_hf(megatron_weights, megatron_module)

        # Apply reverse permutation if specified (after gathering)
        if self.permute_dims is not None and result:
            # Get the tensor from the result dict
            key = list(result.keys())[0]
            tensor = result[key]

            # Apply reverse permutation (same permutation applied again) and make contiguous
            permuted_tensor = torch.permute(tensor, self.permute_dims).contiguous()

            # Update the result with the correct HF param name
            result = {str(self.hf_param): permuted_tensor}

        return result

    def megatron_to_hf_quant(
        self,
        megatron_weights: Optional[torch.Tensor],
        megatron_module: Optional[nn.Module],
        quantization_checker: Callable[[str], bool],
        quant_fn: Callable[..., Tuple[torch.Tensor, torch.Tensor]],
        quant_block_size: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Delegate to appropriate mapping based on module type with quantization before PP broadcast."""
        assert self.megatron_param is not None, "`megatron_param` is required for AutoMapping."

        if self._mapping is None:
            if megatron_module is not None:
                self._detected_type = self._detect_parallelism_type(megatron_module)
                self._detected_type = self.broadcast_obj_from_pp_rank(
                    self._detected_type, "detected_type"
                )
            else:
                self._detected_type = self.broadcast_obj_from_pp_rank(None, "detected_type")
            self._mapping = self._get_or_create_mapping(self._detected_type)

        result = self._mapping.megatron_to_hf_quant(
            megatron_weights, megatron_module, quantization_checker, quant_fn, quant_block_size
        )

        # Apply reverse permutation if specified (after gathering)
        if self.permute_dims is not None and result:
            # We assume result has the main tensor at self.hf_param key.
            # Scale tensors (if any) shouldn't be permuted, or handled depending on structure.
            key = str(self.hf_param)
            tensor = result[key]

            permuted_tensor = torch.permute(tensor, self.permute_dims).contiguous()
            result[key] = permuted_tensor
        return result

    def resolve(self, captures: Tuple[str, ...]) -> "MegatronParamMapping":
        """Create a new mapping with resolved wildcards, preserving permute_dims."""
        resolved_megatron_param, resolved_hf_param = self._resolve_names(captures)
        return type(self)(resolved_megatron_param, resolved_hf_param, self.permute_dims)


class QKVMapping(MegatronParamMapping[Dict[str, torch.Tensor]]):
    """
    Mapping for interleaved Query/Key/Value attention projection weights.

    This mapping handles the conversion between separate Q, K, V matrices used in
    standard transformers and Megatron's optimized interleaved format. The
    interleaving pattern groups queries with their corresponding key-value pairs
    to maximize GEMM efficiency during attention computation.

    **External format (HuggingFace)**
    -   Separate tensors: `q_proj`, `k_proj`, `v_proj`
    -   Each of shape `[hidden_size, hidden_size]` or `[hidden_size, head_dim * num_heads]`

    **Megatron format**
    -   Single interleaved tensor following grouped query attention (GQA) pattern
    -   Interleaving order: `[q1...qn, k1, v1, q1...qn, k2, v2, ...]`
    -   Where `n = num_attention_heads / num_query_groups`

    **Key features**
    1.  Format conversion: Handles merging/splitting with proper interleaving
    2.  Grouped Query Attention: Supports different numbers of Q and KV heads
    3.  Tensor parallelism: Delegates to AutoMapping for distribution

    Example:
        .. code-block:: python

            # Create mapping for attention weights
            mapping = QKVMapping(
                megatron_param="decoder.layers.*.self_attention.linear_qkv.weight",
                q="model.layers.*.self_attn.q_proj.weight",
                k="model.layers.*.self_attn.k_proj.weight",
                v="model.layers.*.self_attn.v_proj.weight"
            )

            # Convert from HuggingFace to Megatron
            qkv_weights = {"q": q_tensor, "k": k_tensor, "v": v_tensor}
            megatron_qkv = mapping.hf_to_megatron(qkv_weights, megatron_module)

            # Convert from Megatron to HuggingFace
            hf_weights = mapping.megatron_to_hf(megatron_qkv, megatron_module)
            # Returns: {"q_proj.weight": ..., "k_proj.weight": ..., "v_proj.weight": ...}

    Note:
        This mapping automatically handles both regular multi-head attention
        (same number of Q, K, V heads) and grouped query attention (fewer
        KV heads than Q heads) based on the model configuration.
    """

    def __init__(self, megatron_param: str, q: str, k: str, v: str):
        """Initialize QKV mapping.

        Args:
            megatron_param (str): Megatron QKV parameter name pattern.
            q (str): Query weight name pattern.
            k (str): Key weight name pattern.
            v (str): Value weight name pattern.
        """
        super().__init__(megatron_param, {"q": q, "k": k, "v": v})
        # Delegate all tensor-parallel logic to the smart TP-aware mapping so we
        # do not hard-code the assumption that QKV projections are column-parallel.
        # This keeps the format-handling (merge/split) concerns separate from
        # TP/PP distribution mechanics.
        self._tp_mapping = AutoMapping(megatron_param, megatron_param)

    def local_mxfp8_params(
        self,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        *,
        global_param_name: str,
        megatron_module: nn.Module,
    ) -> tuple[LocalMXFP8Param, ...]:
        """Split a local native MXFP8 QKV shard into canonical Q, K, and V parameters."""
        config = self._get_config(megatron_module)
        if config.num_attention_heads % self.tp_size:
            raise ValueError(
                f"{global_param_name}: num_attention_heads={config.num_attention_heads} "
                f"is not divisible by tp_size={self.tp_size}"
            )
        if config.num_query_groups % self.tp_size:
            raise ValueError(
                f"{global_param_name}: num_query_groups={config.num_query_groups} "
                f"is not divisible by tp_size={self.tp_size}"
            )

        local_config = cast(
            TransformerConfig,
            SimpleNamespace(
                num_attention_heads=config.num_attention_heads // self.tp_size,
                num_query_groups=config.num_query_groups // self.tp_size,
                kv_channels=config.kv_channels,
                hidden_size=config.hidden_size,
                attention_output_gate=getattr(config, "attention_output_gate", False),
            ),
        )
        qkv_weights = split_qkv_weights(local_config, weight, feature_dim=weight.shape[-1])
        qkv_scales = split_qkv_weights(
            local_config, weight_scale, feature_dim=weight_scale.shape[-1]
        )
        names = (self.hf_param["q"], self.hf_param["k"], self.hf_param["v"])

        params = []
        for name, component_weight, component_scale in zip(names, qkv_weights, qkv_scales):
            global_weight_shape = list(component_weight.shape)
            global_weight_shape[0] *= self.tp_size
            params.append(
                LocalMXFP8Param(
                    name=name,
                    weight=component_weight,
                    weight_scale=component_scale,
                    global_weight_shape=torch.Size(global_weight_shape),
                    shard_group="tp",
                    shard_dim=0,
                )
            )
        return tuple(params)

    def hf_to_megatron(
        self, hf_weights: Dict[str, torch.Tensor], megatron_module: nn.Module
    ) -> torch.Tensor:
        """Merge Q, K, V into interleaved format and distribute."""
        if self.tp_rank == 0:
            config = self._get_config(megatron_module)

            # Check if we're dealing with biases (1D tensors) or hf_weights (2D tensors)
            if hf_weights["q"].ndim == 1:
                # For biases, use the bias-specific merge function
                merged = merge_qkv_biases(config, hf_weights["q"], hf_weights["k"], hf_weights["v"])
            else:
                # For hf_weights, use the standard merge function
                merged = merge_qkv_weights(
                    config, hf_weights["q"], hf_weights["k"], hf_weights["v"]
                )
        else:
            merged = None

        # Delegate the actual sharding/broadcasting to the TP-aware mapping.
        return self._tp_mapping.hf_to_megatron(merged, megatron_module)

    def megatron_to_hf(
        self, megatron_weights: Optional[torch.Tensor], megatron_module: Optional[nn.Module]
    ) -> Dict[str, torch.Tensor]:
        """Gather QKV shards and split into Q, K, V."""
        # Dequantize if needed
        if megatron_weights is not None:
            megatron_weights = self.maybe_dequantize(megatron_weights)
        return self._megatron_to_hf(megatron_weights, megatron_module, scale_row_block_size=None)

    def megatron_to_hf_scale(
        self,
        megatron_weights: Optional[torch.Tensor],
        megatron_module: Optional[nn.Module],
        *,
        row_block_size: int,
    ) -> Dict[str, torch.Tensor]:
        """Gather and split QKV inverse scales using an explicit row block size."""
        return self._megatron_to_hf(
            megatron_weights, megatron_module, scale_row_block_size=row_block_size
        )

    def _megatron_to_hf(
        self,
        megatron_weights: Optional[torch.Tensor],
        megatron_module: Optional[nn.Module],
        *,
        scale_row_block_size: int | None,
    ) -> Dict[str, torch.Tensor]:
        # ------------------------------------------------------------------
        # Broadcast / retrieve the transformer configuration so that every PP
        # rank (also the ones that will early-return) participates in the
        # collective communication.
        # ------------------------------------------------------------------
        if megatron_module is None:
            config = self.broadcast_obj_from_pp_rank(None, "qkv_config")
        else:
            config = self._get_config(megatron_module)
            # create shallow copy and remove non-picklable objects with max depth=3
            config = remove_non_pickleables(config, max_depth=3)
            config = self.broadcast_obj_from_pp_rank(config, "qkv_config")

        # Delegate TP/PP gathering.
        packed_dict = self._tp_mapping.megatron_to_hf(megatron_weights, megatron_module)

        if not packed_dict:
            return {}

        packed_qkv = next(iter(packed_dict.values()))

        # Check if we're dealing with biases (1D) or weights (2D)
        if packed_qkv.ndim == 1:
            if scale_row_block_size is not None:
                raise ValueError("QKV inverse scales must be at least two-dimensional")
            # Split biases
            q, k, v = split_qkv_biases(config, packed_qkv)
        elif scale_row_block_size is not None:
            q, k, v = _split_qkv_weights_scale_by_row(config, packed_qkv, scale_row_block_size)
            q, k, v = q.squeeze(-1), k.squeeze(-1), v.squeeze(-1)
        else:
            # Split weights
            q, k, v = split_qkv_weights(config, packed_qkv)

        return {self.hf_param["q"]: q, self.hf_param["k"]: k, self.hf_param["v"]: v}

    def megatron_to_hf_quant(
        self,
        megatron_weights: Optional[torch.Tensor],
        megatron_module: Optional[nn.Module],
        quantization_checker: Callable[[str], bool],
        quant_fn: Callable[..., Tuple[torch.Tensor, torch.Tensor]],
        quant_block_size: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Convert weights FROM Megatron format with quantization before PP broadcast."""
        if not quantization_checker(str(self.megatron_param)):
            return self.megatron_to_hf(megatron_weights, megatron_module)

        if megatron_module is None:
            config = self.broadcast_obj_from_pp_rank(None, "qkv_config")
        else:
            config = self._get_config(megatron_module)
            import copy

            config = remove_non_pickleables(copy.copy(config), max_depth=3)
            config = self.broadcast_obj_from_pp_rank(config, "qkv_config")

        head_num = config.num_attention_heads
        head_size = config.kv_channels or (config.hidden_size // head_num)
        hidden_size = config.hidden_size
        assert quant_block_size is not None, "quant_block_size must be provided for quantization"
        assert (
            head_size % quant_block_size[0] == 0
        ), f"head_size {head_size} is not divisible by quant_block_size {quant_block_size[0]}"
        assert (
            hidden_size % quant_block_size[1] == 0
        ), f"hidden_size {hidden_size} is not divisible by quant_block_size {quant_block_size[1]}"
        # Delegate TP/PP gathering and quantization.
        packed_dict = self._tp_mapping.megatron_to_hf_quant(
            megatron_weights, megatron_module, lambda _: True, quant_fn, quant_block_size
        )

        if not packed_dict:
            return {}

        packed_qkv = packed_dict.get(self.megatron_param)
        scale_name = self.megatron_param + "_scale_inv"
        packed_scale = packed_dict.get(scale_name)

        if packed_qkv is None:
            # Fallback to finding the tensor if exact string match fails
            for k, v in packed_dict.items():
                if ".scale" in k or "_scale" in k:
                    packed_scale = v
                else:
                    packed_qkv = v

        # Check if we're dealing with biases (1D) or weights (2D)
        if packed_qkv.ndim == 1:
            # Split biases
            raise ValueError(
                "Biases are usually not quantized. Biases are not supported for quantization."
            )
        else:
            # Split weights
            q_q, k_q, v_q = split_qkv_weights(config, packed_qkv)
            q_scale, k_scale, v_scale = split_qkv_weights_scale(
                config, packed_scale, quant_block_size
            )

        quant_result = {}

        def add_to_result(result, q_tensor, scale_tensor, hf_name):
            result[hf_name] = q_tensor
            scale_name = hf_name + "_scale_inv"
            result[scale_name] = scale_tensor

        add_to_result(quant_result, q_q, q_scale, self.hf_param["q"])
        add_to_result(quant_result, k_q, k_scale, self.hf_param["k"])
        add_to_result(quant_result, v_q, v_scale, self.hf_param["v"])

        return quant_result

    def resolve(self, captures: Tuple[str, ...]) -> "MegatronParamMapping":
        """Return a new *resolved* QKVMapping instance."""
        resolved_megatron_param, resolved_hf_param = self._resolve_names(captures)

        return type(self)(
            resolved_megatron_param,
            resolved_hf_param["q"],
            resolved_hf_param["k"],
            resolved_hf_param["v"],
        )


class QKVGMapping(MegatronParamMapping[Dict[str, torch.Tensor]]):
    """QKV mapping that also fuses a per-head scalar gate (``g_proj``).

    Megatron format follows the attention module selected by the provider. For
    regular QKV modules, gate rows are appended after the standard
    GQA-interleaved QKV block. For Megatron-Core ``attention_output_gate``
    modules, the scalar gate is expanded across each head dimension and
    interleaved as the module's output-gate block.

    External (HF) format: four separate tensors ``q_proj``, ``k_proj``,
    ``v_proj``, ``g_proj``.
    """

    def __init__(self, megatron_param: str, q: str, k: str, v: str, g: str):
        super().__init__(megatron_param, {"q": q, "k": k, "v": v, "g": g})
        self._tp_mapping = AutoMapping(megatron_param, megatron_param)

    def hf_to_megatron(
        self, hf_weights: Dict[str, torch.Tensor], megatron_module: nn.Module
    ) -> torch.Tensor:
        if self.tp_rank == 0:
            config = self._get_config(megatron_module)
            if hf_weights["q"].ndim == 1:
                # Biases not supported for the fused gate variant (Step-3.5 has
                # add_qkv_bias=False and g_proj has no bias).
                raise NotImplementedError(
                    "QKVGMapping does not support bias tensors; add_qkv_bias must be False."
                )
            merged = merge_qkvg_weights(
                config, hf_weights["q"], hf_weights["k"], hf_weights["v"], hf_weights["g"]
            )
        else:
            merged = None
        return self._tp_mapping.hf_to_megatron(merged, megatron_module)

    def megatron_to_hf(
        self, megatron_weights: Optional[torch.Tensor], megatron_module: Optional[nn.Module]
    ) -> Dict[str, torch.Tensor]:
        if megatron_weights is not None:
            megatron_weights = self.maybe_dequantize(megatron_weights)

        if megatron_module is None:
            config = self.broadcast_obj_from_pp_rank(None, "qkvg_config")
        else:
            config = self._get_config(megatron_module)
            config = remove_non_pickleables(config, max_depth=3)
            config = self.broadcast_obj_from_pp_rank(config, "qkvg_config")

        packed_dict = self._tp_mapping.megatron_to_hf(megatron_weights, megatron_module)
        if not packed_dict:
            return {}

        packed_qkvg = next(iter(packed_dict.values()))
        if packed_qkvg.ndim == 1:
            raise NotImplementedError(
                "QKVGMapping does not support bias tensors; add_qkv_bias must be False."
            )
        q, k, v, g = split_qkvg_weights(config, packed_qkvg)
        return {
            self.hf_param["q"]: q,
            self.hf_param["k"]: k,
            self.hf_param["v"]: v,
            self.hf_param["g"]: g,
        }

    def resolve(self, captures: Tuple[str, ...]) -> "MegatronParamMapping":
        resolved_megatron_param, resolved_hf_param = self._resolve_names(captures)
        return type(self)(
            resolved_megatron_param,
            resolved_hf_param["q"],
            resolved_hf_param["k"],
            resolved_hf_param["v"],
            resolved_hf_param["g"],
        )


class KVMapping(MegatronParamMapping[Dict[str, torch.Tensor]]):
    """
    Mapping for interleaved Key/Value projection weights.

    This mapping converts between separate K and V tensors used in external
    checkpoints and Megatron's interleaved KV format following grouped-query
    attention semantics.

    External format (HF)
    - Separate tensors: k_proj, v_proj
    - Shapes mirror QKV mappings but without Q

    Megatron format
    - Single interleaved tensor with order: [k1, v1, k2, v2, ...]
      where index corresponds to query-group id

    Tensor-parallel distribution is delegated to AutoMapping.
    """

    def __init__(self, megatron_param: str, k: str, v: str):
        super().__init__(megatron_param, {"k": k, "v": v})
        # Delegate TP sharding/broadcasting
        self._tp_mapping = AutoMapping(megatron_param, megatron_param)

    def hf_to_megatron(
        self, hf_weights: Dict[str, torch.Tensor], megatron_module: nn.Module
    ) -> torch.Tensor:
        """Merge K and V into interleaved format and distribute across TP."""
        if self.tp_rank == 0:
            config = self._get_config(megatron_module)

            if hf_weights["k"].ndim == 1:
                merged = merge_kv_biases(config, hf_weights["k"], hf_weights["v"])
            else:
                merged = merge_kv_weights(config, hf_weights["k"], hf_weights["v"])
        else:
            merged = None

        return self._tp_mapping.hf_to_megatron(merged, megatron_module)

    def megatron_to_hf(
        self, megatron_weights: Optional[torch.Tensor], megatron_module: Optional[nn.Module]
    ) -> Dict[str, torch.Tensor]:
        """Gather KV shards and split into separate K and V tensors."""
        if megatron_weights is not None:
            megatron_weights = self.maybe_dequantize(megatron_weights)

        # Ensure all PP ranks participate in config broadcast
        if megatron_module is None:
            config = self.broadcast_obj_from_pp_rank(None, "kv_config")
        else:
            config = self._get_config(megatron_module)
            config = remove_non_pickleables(config, max_depth=2)
            config = self.broadcast_obj_from_pp_rank(config, "kv_config")

        packed_dict = self._tp_mapping.megatron_to_hf(megatron_weights, megatron_module)
        if not packed_dict:
            return {}

        packed_kv = next(iter(packed_dict.values()))

        if packed_kv.ndim == 1:
            k, v = split_kv_biases(config, packed_kv)
        else:
            k, v = split_kv_weights(config, packed_kv)

        return {self.hf_param["k"]: k, self.hf_param["v"]: v}

    def resolve(self, captures: Tuple[str, ...]) -> "MegatronParamMapping":
        resolved_megatron_param, resolved_hf_param = self._resolve_names(captures)
        return type(self)(resolved_megatron_param, resolved_hf_param["k"], resolved_hf_param["v"])


class MambaInProjMapping(MegatronParamMapping[Dict[str, torch.Tensor]]):
    """Mapping for Mamba input projection weights that handles z, x, B, C, dt components.

    Converts between HuggingFace's concatenated in_proj format and Megatron's
    tensor-parallel distributed format for Mamba SSM layers.
    """

    def __init__(self, megatron_param: str, hf_param: str):
        """Initialize Mamba input projection mapping.

        Args:
            megatron_param (str): Megatron parameter name pattern.
            hf_param (str): HuggingFace parameter name pattern.
        """
        super().__init__(megatron_param=megatron_param, hf_param=hf_param)
        self._tp_mapping = ColumnParallelMapping(megatron_param, megatron_param)

    def hf_to_megatron(
        self, hf_weights: Dict[str, torch.Tensor], megatron_module: nn.Module
    ) -> torch.Tensor:
        """Split Mamba in_proj into z, x, B, C, dt components and distribute across TP ranks."""
        if self.tp_rank == 0:
            config = self._get_config(megatron_module)
            d_inner = config.mamba_num_heads * config.mamba_head_dim
            d_tot_ssm = config.mamba_state_dim * config.mamba_num_groups

            # Define component indices in the concatenated tensor
            z_shard_idx = torch.arange(d_inner, device=hf_weights.device)
            x_shard_idx = torch.arange(d_inner, 2 * d_inner, device=hf_weights.device)
            B_shard_idx = torch.arange(
                2 * d_inner, 2 * d_inner + d_tot_ssm, device=hf_weights.device
            )
            C_shard_idx = torch.arange(
                2 * d_inner + d_tot_ssm, 2 * d_inner + 2 * d_tot_ssm, device=hf_weights.device
            )
            dt_shard_idx = torch.arange(
                2 * (d_inner + d_tot_ssm),
                2 * (d_inner + d_tot_ssm) + config.mamba_num_heads,
                device=hf_weights.device,
            )

            # Reshape for tensor parallel distribution
            target_shape = (self.tp_size, -1, config.hidden_size)
            z_shard = hf_weights[z_shard_idx].reshape(target_shape)
            x_shard = hf_weights[x_shard_idx].reshape(target_shape)
            B_shard = hf_weights[B_shard_idx].reshape(target_shape)
            C_shard = hf_weights[C_shard_idx].reshape(target_shape)
            dt_shard = hf_weights[dt_shard_idx].reshape(target_shape)

            merged = torch.cat([z_shard, x_shard, B_shard, C_shard, dt_shard], dim=1)
            merged = merged.reshape(*target_shape[1:])
        else:
            merged = None

        return self._tp_mapping.hf_to_megatron(merged, megatron_module)

    def megatron_to_hf(
        self, megatron_weights: Optional[torch.Tensor], megatron_module: Optional[nn.Module]
    ) -> Dict[str, torch.Tensor]:
        """Gather Mamba in_proj shards and merge into single HF tensor."""
        # Handle cross-PP broadcast
        megatron_weights = self.broadcast_from_pp_rank(
            megatron_weights, cache_key=str(self.hf_param)
        )

        if megatron_weights is None:
            return {}

        # Dequantize if needed
        megatron_weights = self.maybe_dequantize(megatron_weights)

        # Broadcast config to all PP ranks for collective communication
        if megatron_module is None:
            config = self.broadcast_obj_from_pp_rank(None, cache_key="config")
        else:
            config = self._get_config(megatron_module)
            # create shallow copy and remove non-picklable objects with max depth=3
            config = remove_non_pickleables(config, max_depth=3)
            config = self.broadcast_obj_from_pp_rank(config, cache_key="config")

        d_inner_local = (config.mamba_num_heads * config.mamba_head_dim) // self.tp_size
        d_tot_ssm_local = (config.mamba_state_dim * config.mamba_num_groups) // self.tp_size
        n_heads_local = config.mamba_num_heads // self.tp_size

        # Extract local components
        z_shard_idx = torch.arange(d_inner_local, device=megatron_weights.device)
        x_shard_idx = torch.arange(d_inner_local, device=megatron_weights.device) + d_inner_local
        B_shard_idx = (
            torch.arange(d_tot_ssm_local, device=megatron_weights.device) + 2 * d_inner_local
        )
        C_shard_idx = (
            torch.arange(d_tot_ssm_local, device=megatron_weights.device)
            + 2 * d_inner_local
            + d_tot_ssm_local
        )
        dt_shard_idx = torch.arange(n_heads_local, device=megatron_weights.device) + 2 * (
            d_inner_local + d_tot_ssm_local
        )

        local_components = [
            megatron_weights[z_shard_idx],
            megatron_weights[x_shard_idx],
            megatron_weights[B_shard_idx],
            megatron_weights[C_shard_idx],
            megatron_weights[dt_shard_idx],
        ]

        # Gather each component across TP ranks
        full_weights = []
        for component in local_components:
            if self.tp_size == 1:
                full_weight = component
            else:
                gathered = self.gather_from_tp_ranks(component)
                full_weight = torch.cat(gathered, dim=0)
            full_weights.append(full_weight)

        return {self.hf_param: torch.cat(full_weights, dim=0)}


class ChunkedMapping(MegatronParamMapping[Dict[str, torch.Tensor]]):
    """Abstract class to handle chunked weights mapping, e.g.,
    GDN Conv1d that handles q, k, v components, Mamba Conv1d that handles x, B, C components.
    """

    def __init__(self, megatron_param: str, hf_param: str):
        """Initialize GDN conv1d mapping.

        Args:
            megatron_param (str): Megatron parameter name pattern.
            hf_param (str): HuggingFace parameter name pattern.
        """
        super().__init__(megatron_param=megatron_param, hf_param=hf_param)
        self._tp_mapping = ColumnParallelMapping(megatron_param, megatron_param)

    @abstractmethod
    def get_shard_idx(
        self, config: TransformerConfig, local_tp: bool, device: torch.device | None = None
    ) -> List[int]:
        """Get shard indices for the given config."""
        ...

    def hf_to_megatron(
        self, hf_weights: Dict[str, torch.Tensor], megatron_module: nn.Module
    ) -> torch.Tensor:
        """Split conv1d into x, B, C components and distribute across TP ranks."""
        if self.tp_rank == 0:
            config = self._get_config(megatron_module)

            # Determine reshape based on weight vs bias
            if "weight" in self.megatron_param:
                target_shape = (self.tp_size, -1, *hf_weights.shape[-2:])
            else:
                assert (
                    "bias" in self.megatron_param
                ), "Only bias and weight are supported for conv1d"
                target_shape = (self.tp_size, -1)

            shard_idx = self.get_shard_idx(config, local_tp=False, device=hf_weights.device)

            # Extract and reshape components
            sharded_weights = [hf_weights[idx].reshape(target_shape) for idx in shard_idx]

            merged = torch.cat(sharded_weights, dim=1)
            merged = merged.reshape(*target_shape[1:])
        else:
            merged = None

        return self._tp_mapping.hf_to_megatron(merged, megatron_module)

    def megatron_to_hf(
        self, megatron_weights: Optional[torch.Tensor], megatron_module: Optional[nn.Module]
    ) -> Dict[str, torch.Tensor]:
        """Gather conv1d shards and merge into single HF tensor."""
        megatron_weights = self.broadcast_from_pp_rank(
            megatron_weights, cache_key=str(self.hf_param)
        )

        if megatron_weights is None:
            return {}

        # Dequantize if needed
        megatron_weights = self.maybe_dequantize(megatron_weights)

        # Broadcast config to all PP ranks for collective communication
        if megatron_module is None:
            config = self.broadcast_obj_from_pp_rank(None, cache_key="config")
        else:
            config = self._get_config(megatron_module)
            # create shallow copy and remove non-picklable objects with max depth=3
            config = remove_non_pickleables(config, max_depth=3)
            config = self.broadcast_obj_from_pp_rank(config, cache_key="config")

        shard_idx = self.get_shard_idx(config, local_tp=True, device=megatron_weights.device)

        local_components = [megatron_weights[idx] for idx in shard_idx]

        # Gather each component across TP ranks
        full_weights = []
        for component in local_components:
            if self.tp_size == 1:
                full_weight = component
            else:
                gathered = self.gather_from_tp_ranks(component)
                full_weight = torch.cat(gathered, dim=0)
            full_weights.append(full_weight)

        return {self.hf_param: torch.cat(full_weights, dim=0)}


class GDNConv1dMapping(ChunkedMapping):
    """Mapping for GDN 1D convolution weights that handles q, k, v components.

    Converts between HuggingFace's concatenated conv1d format and Megatron's
    tensor-parallel distributed format for GDN SSM layers.
    """

    def get_shard_idx(
        self, config: TransformerConfig, local_tp: bool, device: torch.device | None = None
    ) -> List[int]:
        """Get shard indices for the given config."""
        qk_dim = config.linear_key_head_dim * config.linear_num_key_heads
        v_dim = config.linear_value_head_dim * config.linear_num_value_heads
        if local_tp:
            qk_dim = qk_dim // self.tp_size
            v_dim = v_dim // self.tp_size

        q_shard_idx = torch.arange(qk_dim, device=device)
        k_shard_idx = torch.arange(qk_dim, device=device) + qk_dim
        v_shard_idx = torch.arange(v_dim, device=device) + qk_dim * 2

        return [q_shard_idx, k_shard_idx, v_shard_idx]


class MambaConv1dMapping(ChunkedMapping):
    """Mapping for Mamba 1D convolution weights that handles x, B, C components.

    Converts between HuggingFace's concatenated conv1d format and Megatron's
    tensor-parallel distributed format for Mamba SSM layers.
    """

    def get_shard_idx(
        self, config: TransformerConfig, local_tp: bool, device: torch.device | None = None
    ) -> List[int]:
        """Get shard indices for the given config."""
        d_inner = config.mamba_num_heads * config.mamba_head_dim
        d_tot_ssm = config.mamba_state_dim * config.mamba_num_groups
        if local_tp:
            d_inner = d_inner // self.tp_size
            d_tot_ssm = d_tot_ssm // self.tp_size

        # Extract local components
        x_shard_idx = torch.arange(d_inner, device=device)
        B_shard_idx = torch.arange(d_tot_ssm, device=device) + d_inner
        C_shard_idx = torch.arange(d_tot_ssm, device=device) + d_inner + d_tot_ssm

        return [x_shard_idx, B_shard_idx, C_shard_idx]


class GDNLinearMapping(MegatronParamMapping[Dict[str, torch.Tensor]]):
    """
    TODO: Add comments
    """

    def __init__(self, megatron_param: str, qkvz: str, ba: str):
        """Initialize GDN input linear projection mapping.
        Args:
            megatron_param (str): Megatron inut projection parameter name pattern.
            qkvz (str): QKVZ weight name pattern.
            ba (str): BA weight name pattern.
        """
        super().__init__(megatron_param, {"qkvz": qkvz, "ba": ba})
        # Delegate all tensor-parallel logic to the smart TP-aware mapping so we
        # do not hard-code the assumption that input projections are column-parallel.
        # This keeps the format-handling (merge/split) concerns separate from
        # TP/PP distribution mechanics.
        self._tp_mapping = AutoMapping(megatron_param, megatron_param)

    def hf_to_megatron(
        self, hf_weights: Dict[str, torch.Tensor], megatron_module: nn.Module
    ) -> torch.Tensor:
        """Merge QKVZ, BA."""
        if self.tp_rank == 0:
            config = self._get_config(megatron_module)
            merged = merge_gdn_linear_weights(
                config, hf_weights["qkvz"], hf_weights["ba"], tp_size=self.tp_size
            )
        else:
            merged = None

        # Delegate the actual sharding/broadcasting to the TP-aware mapping.
        return self._tp_mapping.hf_to_megatron(merged, megatron_module)

    def megatron_to_hf(
        self, megatron_weights: Optional[torch.Tensor], megatron_module: Optional[nn.Module]
    ) -> Dict[str, torch.Tensor]:
        """Gather QKVZBA shards and split into QKVZ and BA."""
        # Dequantize if needed
        if megatron_weights is not None:
            megatron_weights = self.maybe_dequantize(megatron_weights)

        # ------------------------------------------------------------------
        # Broadcast / retrieve the transformer configuration so that every PP
        # rank (also the ones that will early-return) participates in the
        # collective communication.
        # ------------------------------------------------------------------
        if megatron_module is None:
            config = self.broadcast_obj_from_pp_rank(None, cache_key="config")
        else:
            config = self._get_config(megatron_module)
            # create shallow copy and remove non-picklable objects with max depth=3
            config = remove_non_pickleables(config, max_depth=3)
            config = self.broadcast_obj_from_pp_rank(config, cache_key="config")

        # Delegate TP/PP gathering.
        packed_dict = self._tp_mapping.megatron_to_hf(megatron_weights, megatron_module)

        if not packed_dict:
            return {}

        packed_qkvzba = next(iter(packed_dict.values()))
        qkvz, ba = split_gdn_linear_weights(config, packed_qkvzba, tp_size=self.tp_size)

        return {self.hf_param["qkvz"]: qkvz, self.hf_param["ba"]: ba}

    def resolve(self, captures: Tuple[str, ...]) -> "MegatronParamMapping":
        """Return a new *resolved* GDNLinearMapping instance."""
        resolved_megatron_param, resolved_hf_param = self._resolve_names(captures)

        return type(self)(
            resolved_megatron_param, resolved_hf_param["qkvz"], resolved_hf_param["ba"]
        )


class GDNLinearMappingSeparate(MegatronParamMapping[Dict[str, torch.Tensor]]):
    """GDN input projection mapping for models with separate QKV, Z, B, A HF weights.

    Unlike :class:`GDNLinearMapping` which expects two fused tensors (``in_proj_qkvz``
    and ``in_proj_ba`` in Qwen3-Next's head-grouped layout), this mapping handles
    models that store each projection component separately:

    * ``in_proj_qkv`` - fused Q, K, V projection  (flat ``[Q; K; V]``)
    * ``in_proj_z``   - Z (gate) projection
    * ``in_proj_b``   - B (beta) projection
    * ``in_proj_a``   - A (alpha) projection

    Used by **Qwen3.5** whose GDN layers expose four distinct weight matrices.

    The class converts between the 4-tensor HF layout and Megatron's single
    ``in_proj`` tensor by first assembling the head-grouped ``qkvz`` / ``ba``
    intermediates expected by the existing :func:`merge_gdn_linear_weights` and
    :func:`split_gdn_linear_weights` helpers, keeping the TP-sharding logic
    unchanged.
    """

    def __init__(self, megatron_param: str, qkv: str, z: str, b: str, a: str):
        """Initialise GDN separate-component mapping.

        Args:
            megatron_param: Megatron ``in_proj`` parameter name pattern.
            qkv: HF weight pattern for the fused Q/K/V projection.
            z:   HF weight pattern for the Z (gate) projection.
            b:   HF weight pattern for the B (beta) projection.
            a:   HF weight pattern for the A (alpha) projection.
        """
        super().__init__(megatron_param, {"qkv": qkv, "z": z, "b": b, "a": a})
        self._tp_mapping = AutoMapping(megatron_param, megatron_param)

    # --------------------------------------------------------------------- #
    # HF → Megatron
    # --------------------------------------------------------------------- #
    def hf_to_megatron(
        self, hf_weights: Dict[str, torch.Tensor], megatron_module: nn.Module
    ) -> torch.Tensor:
        """Merge four separate HF tensors into Megatron's single ``in_proj``."""
        if self.tp_rank == 0:
            config = self._get_config(megatron_module)
            qkvz, ba = _fuse_gdn_separate_to_grouped(
                config, hf_weights["qkv"], hf_weights["z"], hf_weights["b"], hf_weights["a"]
            )
            merged = merge_gdn_linear_weights(config, qkvz, ba, tp_size=self.tp_size)
        else:
            merged = None

        return self._tp_mapping.hf_to_megatron(merged, megatron_module)

    # --------------------------------------------------------------------- #
    # Megatron → HF
    # --------------------------------------------------------------------- #
    def megatron_to_hf(
        self, megatron_weights: Optional[torch.Tensor], megatron_module: Optional[nn.Module]
    ) -> Dict[str, torch.Tensor]:
        """Gather shards and split into the four separate HF tensors."""
        if megatron_weights is not None:
            megatron_weights = self.maybe_dequantize(megatron_weights)

        # Broadcast config across PP ranks (mirrors GDNLinearMapping).
        if megatron_module is None:
            config = self.broadcast_obj_from_pp_rank(None, cache_key="config")
        else:
            config = self._get_config(megatron_module)
            config = remove_non_pickleables(config, max_depth=3)
            config = self.broadcast_obj_from_pp_rank(config, cache_key="config")

        packed_dict = self._tp_mapping.megatron_to_hf(megatron_weights, megatron_module)
        if not packed_dict:
            return {}

        packed_in_proj = next(iter(packed_dict.values()))
        qkvz, ba = split_gdn_linear_weights(config, packed_in_proj, tp_size=self.tp_size)
        qkv, z, b, a = _split_gdn_grouped_to_separate(config, qkvz, ba)

        return {
            self.hf_param["qkv"]: qkv,
            self.hf_param["z"]: z,
            self.hf_param["b"]: b,
            self.hf_param["a"]: a,
        }

    # --------------------------------------------------------------------- #
    # Pattern resolution
    # --------------------------------------------------------------------- #
    def resolve(self, captures: Tuple[str, ...]) -> "MegatronParamMapping":
        resolved_megatron_param, resolved_hf_param = self._resolve_names(captures)
        return type(self)(
            resolved_megatron_param,
            resolved_hf_param["qkv"],
            resolved_hf_param["z"],
            resolved_hf_param["b"],
            resolved_hf_param["a"],
        )


class ConcatenatedQKVMapping(MegatronParamMapping[Dict[str, torch.Tensor]]):
    """
    Mapping for interleaved Query/Key/Value attention projection weights.

    This mapping handles the conversion between Concatenated Q, K, V matrices used in
    some transformers models and Megatron's optimized interleaved format. The
    interleaving pattern groups queries with their corresponding key-value pairs
    to maximize GEMM efficiency during attention computation.

    **External format (HuggingFace)**
    -   One tensor with concatenated query, key, value: `qkv`, with shape
        `[hidden_size, head_dim * num_heads + 2 * head_dim * num_query_groups]`

    **Megatron format**
    -   Single interleaved tensor following grouped query attention (GQA) pattern
    -   Interleaving order: `[q1...qn, k1, v1, q1...qn, k2, v2, ...]`
    -   Where `n = num_attention_heads / num_query_groups`

    **Key features**
    1.  Format conversion: Handles merging/splitting with proper interleaving
    2.  Grouped Query Attention: Supports different numbers of Q and KV heads
    3.  Tensor parallelism: Delegates to AutoMapping for distribution

    Example:
        .. code-block:: python

            # Create mapping for attention weights
            mapping = ConcatenatedQKVMapping(
                megatron_param="decoder.layers.*.self_attention.linear_qkv.weight",
                qkv="model.layers.*.self_attn.qkv.weight",
            )

            # Convert from HuggingFace to Megatron
            megatron_qkv = mapping.hf_to_megatron(qkv_weights, megatron_module)

            # Convert from Megatron to HuggingFace
            hf_weights = mapping.megatron_to_hf(megatron_qkv, megatron_module)

    Note:
        This mapping automatically handles both regular multi-head attention
        (same number of Q, K, V heads) and grouped query attention (fewer
        KV heads than Q heads) based on the model configuration.
    """

    def __init__(self, megatron_param: str, hf_param: str):
        """Initialize QKV mapping.

        Args:
            megatron_param (str): Megatron interleaved QKV parameter name pattern.
            hf_param (str): HF concatenated QKV parameter name pattern.
        """
        super().__init__(megatron_param, hf_param)
        # Delegate all tensor-parallel logic to the smart TP-aware mapping so we
        # do not hard-code the assumption that QKV projections are column-parallel.
        # This keeps the format-handling (merge/split) concerns separate from
        # TP/PP distribution mechanics.
        self._tp_mapping = AutoMapping(megatron_param, megatron_param)

    def local_hf_param_specs(
        self, global_param_name: Optional[str] = None
    ) -> tuple[LocalHFParamSpec, ...]:
        """Require normal conversion to undo the Megatron QKV interleaving."""
        return ()

    def hf_to_megatron(self, hf_weights: torch.Tensor, megatron_module: nn.Module) -> torch.Tensor:
        """Merge Q, K, V into interleaved format and distribute."""
        if self.tp_rank == 0:
            config = self._get_config(megatron_module)
            head_num = config.num_attention_heads
            head_size = config.kv_channels
            num_query_groups = config.num_query_groups
            q, k, v = hf_weights.split(
                [head_num * head_size, num_query_groups * head_size, num_query_groups * head_size],
                dim=0,
            )
            # Check if we're dealing with biases (1D tensors) or hf_weights (2D tensors)
            if q.ndim == 1:
                # For biases, use the bias-specific merge function
                merged = merge_qkv_biases(config, q, k, v)
            else:
                # For hf_weights, use the standard merge function
                merged = merge_qkv_weights(config, q, k, v)
        else:
            merged = None

        # Delegate the actual sharding/broadcasting to the TP-aware mapping.
        return self._tp_mapping.hf_to_megatron(merged, megatron_module)

    def megatron_to_hf(
        self, megatron_weights: Optional[torch.Tensor], megatron_module: Optional[nn.Module]
    ) -> Dict[str, torch.Tensor]:
        """Gather QKV shards and split into Q, K, V."""
        # Dequantize if needed
        if megatron_weights is not None:
            megatron_weights = self.maybe_dequantize(megatron_weights)

        # ------------------------------------------------------------------
        # Broadcast / retrieve the transformer configuration so that every PP
        # rank (also the ones that will early-return) participates in the
        # collective communication.
        # ------------------------------------------------------------------
        if megatron_module is None:
            config = self.broadcast_obj_from_pp_rank(None, "qkv_config")
        else:
            config = self._get_config(megatron_module)
            # create shallow copy and remove non-picklable objects with max depth=2
            config = remove_non_pickleables(config, max_depth=2)
            config = self.broadcast_obj_from_pp_rank(config, "qkv_config")

        # Delegate TP/PP gathering.
        packed_dict = self._tp_mapping.megatron_to_hf(megatron_weights, megatron_module)

        if not packed_dict:
            return {}

        packed_qkv = next(iter(packed_dict.values()))

        # Check if we're dealing with biases (1D) or weights (2D)
        if packed_qkv.ndim == 1:
            # Split biases
            q, k, v = split_qkv_biases(config, packed_qkv)
        else:
            # Split weights
            q, k, v = split_qkv_weights(config, packed_qkv)

        return {str(self.hf_param): torch.cat((q, k, v), dim=0)}

    def resolve(self, captures: Tuple[str, ...]) -> "MegatronParamMapping":
        """Return a new *resolved* QKVMapping instance."""
        resolved_megatron_param, resolved_hf_param = self._resolve_names(captures)

        return type(self)(resolved_megatron_param, resolved_hf_param)


class GatedMLPMapping(MegatronParamMapping[Dict[str, torch.Tensor]]):
    r"""Mapping for **gated-MLP** projection weights (SwiGLU / GeGLU).

    Checkpoint formats expose two independent matrices:

    -   **G** – gate projection
    -   **U** – up projection

    Megatron concatenates them row-wise (`[G; U]`) so that a single GEMM can
    produce both activations.

    **Responsibilities handled by this mapping**
    1.  **Concatenate / split** – convert between `[G; U]` (Megatron) and the
        separate `{G, U}` matrices (external).
    2.  **Tensor-parallel distribution** – correctly splits gate and up
        projections separately before concatenating corresponding shards,
        ensuring each TP rank gets the proper [gate_shard; up_shard] format.

    **TP Distribution Strategy**
    For tensor parallelism, this mapping:
    - Splits gate and up matrices separately along output dimension (dim 0)
    - Concatenates corresponding shards: [gate_shard_i; up_shard_i] for rank i
    - This ensures each rank's concatenated tensor matches the expected shape
    """

    def __init__(self, megatron_param: str, gate: str, up: str):
        """Initialize gated MLP mapping.

        Args:
            megatron_param (str): Megatron MLP parameter name pattern.
            gate (str): Gate projection weight name pattern.
            up (str): Up projection weight name pattern.
        """
        super().__init__(megatron_param, {"gate": gate, "up": up})

    def local_hf_param_specs(
        self, global_param_name: Optional[str] = None
    ) -> tuple[LocalHFParamSpec, ...]:
        """Describe the gate and up views of the local fused projection."""
        if getattr(self, "is_grouped_export", False):
            return ()
        return (
            LocalHFParamSpec(self.hf_param["gate"], -2, 0, 2),
            LocalHFParamSpec(self.hf_param["up"], -2, 1, 2),
        )

    def local_mxfp8_params(
        self,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        *,
        global_param_name: str,
        megatron_module: nn.Module,
    ) -> tuple[LocalMXFP8Param, ...]:
        """Split a local native MXFP8 FC1 shard into gate and up projections."""
        specs = self.local_hf_param_specs(global_param_name)
        if not specs:
            raise ValueError(
                f"{global_param_name}: mapping does not expose canonical local HF views"
            )
        if megatron_module is None or _module_uses_fsdp(megatron_module):
            raise ValueError(
                f"{global_param_name}: native MXFP8 export does not support DTensor/FSDP parameters"
            )
        shard_group: MXFP8ShardGroup = "etp" if self.is_expert else "tp"
        return _project_gated_native_mxfp8(
            weight,
            weight_scale,
            names=(specs[0].name, specs[1].name),
            global_param_name=global_param_name,
            shard_group=shard_group,
            shard_size=self.tp_size,
        )

    def _local_hf_sharding(
        self, weight: torch.Tensor, megatron_module: nn.Module
    ) -> tuple[LocalHFShardGroup, int | None, int]:
        """Describe fused gate/up projections as column-parallel views."""
        return ("etp" if self.is_expert else "tp"), -2, self.tp_size

    def hf_to_megatron(
        self, hf_weights: Dict[str, torch.Tensor], megatron_module: nn.Module
    ) -> torch.Tensor:
        """Split gate and up separately, then concatenate corresponding shards."""
        # For single TP, just concatenate and return
        if self.tp_size == 1:
            return torch.cat([hf_weights["gate"], hf_weights["up"]], dim=0)

        # Get target parameter info from megatron module
        # Some parameters are named with global expert number, e.g. experts.weight15,
        # normalize it to experts.weight0, note we are only use the shape, dtype, device info,
        # not the actual value, so it is safe to do this.
        normalized_param = self._normalize_expert_param_name(self.megatron_param)
        _, target_param = get_module_and_param_from_name(megatron_module, normalized_param)

        # On rank 0, split gate and up separately, then concatenate corresponding pieces
        if self.tp_rank == 0:
            gate = hf_weights["gate"]
            up = hf_weights["up"]

            # Verify shapes match
            assert gate.shape == up.shape, "Gate and up weights must have the same shape"

            # Check divisibility for TP splitting
            gate_output_size = gate.shape[0]
            if gate_output_size % self.tp_size != 0:
                raise ValueError(
                    f"Cannot evenly split gate dimension 0 size {gate_output_size} across {self.tp_size} TP ranks"
                )

            # Split gate and up separately along output dimension (dim 0)
            # This works for both bias (1D) and weight (2D) tensors
            gate_splits = torch.chunk(gate, self.tp_size, dim=0)
            up_splits = torch.chunk(up, self.tp_size, dim=0)

            # Concatenate corresponding pieces: [gate_shard_i; up_shard_i] for each rank i
            splits = [torch.cat([gate_splits[i], up_splits[i]], dim=0) for i in range(self.tp_size)]
        else:
            splits = None

        if isinstance(target_param, DTensor):
            output_shape = target_param.orig_param.shape
        else:
            output_shape = _get_mapping_shape(target_param)
        # Scatter the concatenated shards to each rank
        return self.scatter_to_tp_ranks(
            splits, output_shape, target_param.dtype, target_param.device
        )

    def megatron_to_hf(
        self, megatron_weights: Optional[torch.Tensor], megatron_module: Optional[nn.Module]
    ) -> Dict[str, torch.Tensor]:
        """Gather concatenated shards and split into gate and up."""
        # Handle cross-PP broadcast first
        megatron_weights = self.broadcast_from_pp_rank(
            megatron_weights, cache_key=str(self.hf_param)
        )

        if megatron_weights is None:
            return {}

        # Dequantize if needed
        megatron_weights = self.maybe_dequantize(megatron_weights)

        # Handle TP gathering
        if self.tp_size == 1:
            # No TP, just split the concatenated tensor
            fused_mlp = megatron_weights
            gate, up = torch.chunk(fused_mlp, 2, dim=0)

        else:
            if _module_uses_fsdp(megatron_module):
                gathered_shards = torch.chunk(megatron_weights, self.tp_size, dim=0)
            else:
                # Gather shards from all TP ranks
                gathered_shards = self.gather_from_tp_ranks(megatron_weights)

            # Split each shard back into gate and up parts
            gate_parts = []
            up_parts = []
            for shard in gathered_shards:
                # Each shard is [gate_shard; up_shard] concatenated along dim 0
                # This works for both bias (1D) and weight (2D) tensors
                gate_shard, up_shard = torch.chunk(shard, 2, dim=0)
                gate_parts.append(gate_shard)
                up_parts.append(up_shard)

            # Concatenate all gate parts and all up parts separately
            gate = torch.cat(gate_parts, dim=0)
            up = torch.cat(up_parts, dim=0)

        if self.is_expert:
            gathered_gate_weights_dict = self.gather_from_ep_ranks(
                gate, megatron_module, self.hf_param["gate"]
            )
            gathered_up_weights_dict = self.gather_from_ep_ranks(
                up, megatron_module, self.hf_param["up"]
            )
            return {**gathered_gate_weights_dict, **gathered_up_weights_dict}

        return {self.hf_param["gate"]: gate, self.hf_param["up"]: up}

    def megatron_to_hf_quant(
        self,
        megatron_weights: Optional[torch.Tensor],
        megatron_module: Optional[nn.Module],
        quantization_checker: Callable[[str], bool],
        quant_fn: Callable[..., Tuple[torch.Tensor, torch.Tensor]],
        quant_block_size: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Convert weights FROM Megatron format with quantization before PP broadcast."""
        if not quantization_checker(str(self.megatron_param)):
            return self.megatron_to_hf(megatron_weights, megatron_module)

        fused_q, fused_scale = None, None

        if megatron_weights is not None:
            megatron_weights = self.maybe_dequantize(megatron_weights)

            assert (
                quant_block_size is not None
            ), "quant_block_size must be provided for quantization"

            if len(megatron_weights.shape) == 2:
                # Calculate the shape of gate and up shards without performing the actual chunk
                dim0_chunk = megatron_weights.shape[0] // 2
                dim1_chunk = megatron_weights.shape[1]

                if dim0_chunk % quant_block_size[0] != 0 or dim1_chunk % quant_block_size[1] != 0:
                    raise ValueError(
                        f"gate_shard and up_shard shape ({dim0_chunk}, {dim1_chunk}) cannot be divided by quant_block_size {quant_block_size}"
                    )

            assert len(megatron_weights.shape) == 2, "Megatron weights must be 2D for quantization"
            fused_q, fused_scale = quant_fn(megatron_weights, quant_block_size)

        fused_q = self.broadcast_from_pp_rank(
            fused_q, cache_key=str(self.hf_param["gate"]) + "_fused_q"
        )
        fused_scale = self.broadcast_from_pp_rank(
            fused_scale, cache_key=str(self.hf_param["gate"]) + "_fused_scale"
        )

        if fused_q is None:
            return {}

        if self.tp_size == 1:
            full_gate_q, full_up_q = torch.chunk(fused_q, 2, dim=0)
            if len(fused_scale.shape) > 0:
                full_gate_scale, full_up_scale = torch.chunk(fused_scale, 2, dim=0)
            else:
                raise ValueError(
                    f"fused_scale cannot be splitted, {self.megatron_param} has {fused_scale.shape=}"
                )
        else:
            gathered_fused_q = self.gather_from_tp_ranks(fused_q)
            gate_q_parts = []
            up_q_parts = []
            for shard in gathered_fused_q:
                gate_shard, up_shard = torch.chunk(shard, 2, dim=0)
                gate_q_parts.append(gate_shard)
                up_q_parts.append(up_shard)
            full_gate_q = torch.cat(gate_q_parts, dim=0)
            full_up_q = torch.cat(up_q_parts, dim=0)

            if len(fused_scale.shape) > 0:
                gathered_fused_scale = self.gather_from_tp_ranks(fused_scale)
                gate_scale_parts = []
                up_scale_parts = []
                for shard in gathered_fused_scale:
                    gate_shard, up_shard = torch.chunk(shard, 2, dim=0)
                    gate_scale_parts.append(gate_shard)
                    up_scale_parts.append(up_shard)
                full_gate_scale = torch.cat(gate_scale_parts, dim=0)
                full_up_scale = torch.cat(up_scale_parts, dim=0)
            else:
                raise ValueError(
                    f"fused_scale cannot be splitted, {self.megatron_param} has {fused_scale.shape=}"
                )
        if self.is_expert:
            ep_gate_dict = self.gather_from_ep_ranks(
                full_gate_q, megatron_module, self.hf_param["gate"]
            )
            ep_up_dict = self.gather_from_ep_ranks(full_up_q, megatron_module, self.hf_param["up"])

            quant_result = {**ep_gate_dict, **ep_up_dict}

            ep_gate_scale_dict = self.gather_from_ep_ranks_scale(
                full_gate_scale, megatron_module, self.hf_param["gate"]
            )
            for k, v in ep_gate_scale_dict.items():
                scale_name = k + "_scale_inv"
                quant_result[scale_name] = v

            ep_up_scale_dict = self.gather_from_ep_ranks_scale(
                full_up_scale, megatron_module, self.hf_param["up"]
            )
            for k, v in ep_up_scale_dict.items():
                scale_name = k + "_scale_inv"
                quant_result[scale_name] = v

            return quant_result

        quant_result = {}

        def add_to_result(result, q_tensor, scale_tensor, hf_name):
            result[hf_name] = q_tensor
            scale_name = hf_name + "_scale_inv"
            result[scale_name] = scale_tensor

        add_to_result(quant_result, full_gate_q, full_gate_scale, self.hf_param["gate"])
        add_to_result(quant_result, full_up_q, full_up_scale, self.hf_param["up"])

        return quant_result

    def resolve(self, captures: Tuple[str, ...]) -> "MegatronParamMapping":
        """Return a new *resolved* GatedMLPMapping instance."""
        resolved_megatron_param, resolved_hf_param = self._resolve_names(captures)

        return type(self)(
            resolved_megatron_param, resolved_hf_param["gate"], resolved_hf_param["up"]
        )


class RMSNorm2ZeroCenteredRMSNormMapping(AutoMapping):
    """
    Mapping for zero-centered RMSNorm to standard RMSNorm.
    """

    def local_hf_param_specs(
        self, global_param_name: Optional[str] = None
    ) -> tuple[LocalHFParamSpec, ...]:
        """Require normal conversion to restore the standard RMSNorm values."""
        return ()

    def hf_to_megatron(self, hf_weights: torch.Tensor, megatron_module: nn.Module) -> torch.Tensor:
        hf_weights = hf_weights.clone()
        hf_weights.data -= 1
        return super().hf_to_megatron(hf_weights, megatron_module)

    def megatron_to_hf(
        self, megatron_weights: torch.Tensor, megatron_module: nn.Module
    ) -> torch.Tensor:
        hf_weights = super().megatron_to_hf(megatron_weights, megatron_module)
        assert (
            isinstance(hf_weights, dict) and len(hf_weights) == 1
        ), f"Expected a dictionary with one element, got {hf_weights.keys()=}"
        key = list(hf_weights.keys())[0]
        value = hf_weights[key].clone()
        value.data += 1
        return {key: value}


def _align_expert_weight_to_shape(
    weight: torch.Tensor, target_shape: torch.Size, name: str, transpose_hint: bool | None = None
) -> torch.Tensor:
    """Align an expert weight tensor to match a Megatron target shape.

    Args:
        weight: The weight tensor to align.
        target_shape: The expected Megatron parameter shape.
        name: Name used in error messages.
        transpose_hint: If ``True``, transpose the last two dims unconditionally.
            If ``False``, return as-is (assert shape already matches).
            If ``None`` (default), auto-detect: returns the tensor directly if
            the shape matches, transposes the last two dims if the transposed
            shape matches, or raises ``ValueError`` otherwise. Auto-detection
            is ambiguous for square 2-D weights — pass an explicit
            ``transpose_hint`` in that case.
    """
    if transpose_hint is True:
        result = (
            weight.t().contiguous() if weight.ndim == 2 else weight.transpose(-1, -2).contiguous()
        )
        if tuple(result.shape) != tuple(target_shape):
            raise ValueError(
                f"Unexpected {name} shape after transpose: {tuple(result.shape)}; expected {tuple(target_shape)}."
            )
        return result
    if transpose_hint is False:
        if tuple(weight.shape) != tuple(target_shape):
            raise ValueError(
                f"Unexpected {name} shape {tuple(weight.shape)}; expected {tuple(target_shape)}."
            )
        return weight
    # Auto-detect (transpose_hint is None)
    if tuple(weight.shape) == tuple(target_shape):
        return weight
    if weight.ndim == 2 and weight.shape[0] == weight.shape[1]:
        raise ValueError(
            f"Cannot auto-detect transpose for square {name} weight {tuple(weight.shape)}; "
            f"pass an explicit transpose_hint=True/False."
        )
    if weight.ndim == 2 and tuple(weight.t().shape) == tuple(target_shape):
        return weight.t().contiguous()
    raise ValueError(
        f"Unexpected {name} shape {tuple(weight.shape)}; expected {tuple(target_shape)}."
    )


class _LooseGatedMLPMapping(GatedMLPMapping):
    """GatedMLPMapping that skips wildcard validation for fused expert mappings."""

    is_grouped_export = True


class FusedExpertMapping(AutoMapping):
    """Mapping for fused expert weights: 1 HF tensor [num_experts, ...] <-> N Megatron per-expert tensors.

    HF side: Single tensor with shape [num_experts, ...]
    Megatron side: Per-expert tensors (one param per expert)

    Import: Extracts single expert from fused HF tensor, auto-aligns shape,
            delegates to AutoMapping for TP distribution.
    Export: AutoMapping handles TP/EP gathering per expert, then the conversion
            loop merges all experts via the ``is_grouped_export`` protocol.

    Replaces per-model expert mapping classes and eliminates the need for
    ``maybe_modify_converted_hf_weight`` / ``hf_weights_cache`` on bridges.
    """

    is_grouped_export = True

    def __init__(
        self,
        megatron_param: str,
        hf_param: str,
        permute_dims: tuple[int, ...] | None = None,
        transpose_on_export: bool = False,
    ):
        super().__init__(megatron_param, hf_param, permute_dims)
        self.allow_hf_name_mismatch = True
        self.transpose_on_export = transpose_on_export

    @property
    def group_key(self) -> str:
        """Tasks sharing the same group_key are merged during export."""
        return self.hf_param

    def local_hf_param_specs(
        self, global_param_name: Optional[str] = None
    ) -> tuple[LocalHFParamSpec, ...]:
        """Describe a canonical per-expert view, not the grouped destination layout.

        Grouped transposition applies only after expert assembly. These logical
        views do not qualify direct native MXFP8 transport, which rejects it.
        """
        # Grouped destination transposition does not change the canonical expert view.
        if self.permute_dims is not None:
            return ()
        expert_idx = extract_expert_number_from_param(global_param_name or self.megatron_param)
        prefix = self.hf_param.removesuffix(".weight").removesuffix(".down_proj")
        return (LocalHFParamSpec(f"{prefix}.{expert_idx}.down_proj.weight"),)

    def local_mxfp8_params(
        self,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        *,
        global_param_name: str,
        megatron_module: nn.Module,
    ) -> tuple[LocalMXFP8Param, ...]:
        """Project one local native MXFP8 FC2 expert without gathering."""
        if self.permute_dims is not None or self.transpose_on_export:
            raise ValueError(
                f"{global_param_name}: native MXFP8 export does not support mapping transforms"
            )
        if megatron_module is None or _module_uses_fsdp(megatron_module):
            raise ValueError(
                f"{global_param_name}: native MXFP8 export does not support DTensor/FSDP parameters"
            )
        _validate_native_mxfp8_pair(weight, weight_scale, global_param_name)

        global_weight_shape = list(weight.shape)
        global_weight_shape[1] *= self.tp_size
        return (
            LocalMXFP8Param(
                name=self.local_hf_param_specs(global_param_name)[0].name,
                weight=weight,
                weight_scale=weight_scale,
                global_weight_shape=torch.Size(global_weight_shape),
                shard_group="etp",
                shard_dim=1,
            ),
        )

    def hf_to_megatron(self, hf_weights: torch.Tensor, megatron_module: nn.Module) -> torch.Tensor:
        from megatron.bridge.utils.common_utils import extract_expert_number_from_param

        expert_idx = extract_expert_number_from_param(self.megatron_param)
        expert_weight = hf_weights[expert_idx] if hf_weights.ndim >= 3 else hf_weights
        # Pass the full (unsharded) expert weight directly to AutoMapping, which handles
        # TP scatter. We must NOT align against target_param.shape here because that shape
        # is already TP-sharded and would fail for TP > 1.
        return super().hf_to_megatron(expert_weight, megatron_module)

    def resolve(self, captures: Tuple[str, ...]) -> "MegatronParamMapping":
        resolved_megatron_param, resolved_hf_param = self._resolve_names(captures)
        return type(self)(
            resolved_megatron_param, resolved_hf_param, self.permute_dims, self.transpose_on_export
        )


class FusedGatedExpertMapping(AutoMapping):
    """Mapping for fused gated expert weights (gate+up projection).

    HF side: Single tensor with shape [num_experts, 2*intermediate, hidden]
    Megatron side: Per-expert linear_fc1 tensors (with gate+up interleaved)

    Import: Extracts single expert, splits into gate+up, delegates to
            GatedMLPMapping for interleaved TP distribution.
    Export: GatedMLPMapping handles TP/EP gathering, gate+up are fused back,
            conversion loop merges all experts via the ``is_grouped_export`` protocol.
    """

    is_grouped_export = True

    def __init__(
        self,
        megatron_param: str,
        hf_param: str,
        permute_dims: tuple[int, ...] | None = None,
        transpose_on_export: bool = False,
    ):
        super().__init__(megatron_param, hf_param, permute_dims)
        self.allow_hf_name_mismatch = True
        self.transpose_on_export = transpose_on_export
        self._gated_mapping = _LooseGatedMLPMapping(
            megatron_param=self.megatron_param,
            gate=f"{self.hf_param}.gate",
            up=f"{self.hf_param}.up",
        )

    @property
    def group_key(self) -> str:
        """Tasks sharing the same group_key are merged during export."""
        return self.hf_param

    def local_hf_param_specs(
        self, global_param_name: Optional[str] = None
    ) -> tuple[LocalHFParamSpec, ...]:
        """Describe canonical gate/up views, not the grouped destination layout.

        Grouped transposition applies only after expert assembly. These logical
        views do not qualify direct native MXFP8 transport, which rejects it.
        """
        # Grouped destination transposition does not change the canonical expert views.
        if self.permute_dims is not None:
            return ()
        expert_idx = extract_expert_number_from_param(global_param_name or self.megatron_param)
        prefix = self.hf_param.removesuffix(".weight").removesuffix(".gate_up_proj")
        return (
            LocalHFParamSpec(f"{prefix}.{expert_idx}.gate_proj.weight", -2, 0, 2),
            LocalHFParamSpec(f"{prefix}.{expert_idx}.up_proj.weight", -2, 1, 2),
        )

    def local_mxfp8_params(
        self,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        *,
        global_param_name: str,
        megatron_module: nn.Module,
    ) -> tuple[LocalMXFP8Param, ...]:
        """Project one local native MXFP8 FC1 expert into gate and up views."""
        if self.permute_dims is not None or self.transpose_on_export:
            raise ValueError(
                f"{global_param_name}: native MXFP8 export does not support mapping transforms"
            )
        if megatron_module is None or _module_uses_fsdp(megatron_module):
            raise ValueError(
                f"{global_param_name}: native MXFP8 export does not support DTensor/FSDP parameters"
            )

        specs = self.local_hf_param_specs(global_param_name)
        return _project_gated_native_mxfp8(
            weight,
            weight_scale,
            names=(specs[0].name, specs[1].name),
            global_param_name=global_param_name,
            shard_group="etp",
            shard_size=self.tp_size,
        )

    def hf_to_megatron(self, hf_weights: torch.Tensor, megatron_module: nn.Module) -> torch.Tensor:
        from megatron.bridge.utils.common_utils import extract_expert_number_from_param

        expert_idx = extract_expert_number_from_param(self.megatron_param)
        expert_weight = hf_weights[expert_idx] if hf_weights.ndim >= 3 else hf_weights

        normalized_param = self._normalize_expert_param_name(self.megatron_param)
        _, target_param = get_module_and_param_from_name(megatron_module, normalized_param)
        target_shape = _get_mapping_shape(target_param)

        if target_shape[0] % 2 != 0:
            raise ValueError(
                f"Expected even fused dim for {self.megatron_param}, got {target_shape}."
            )

        if isinstance(target_param, DTensor):
            gate_full_shape = (target_shape[0] // 2, target_shape[1])
        else:
            # target_shape is the TP-sharded Megatron shape; compute the full
            # unsharded shape so raw HF weights can be validated before TP scatter.
            gate_full_shape = (target_shape[0] // 2 * self.tp_size, target_shape[1])
        gate_up_full_shape = (gate_full_shape[0] * 2, target_shape[1])

        if expert_weight.ndim == 3 and expert_weight.shape[0] == 2:
            gate = _align_expert_weight_to_shape(expert_weight[0], gate_full_shape, "gate")
            up = _align_expert_weight_to_shape(expert_weight[1], gate_full_shape, "up")
        else:
            expert_weight = _align_expert_weight_to_shape(
                expert_weight, gate_up_full_shape, "gate_up"
            )
            gate, up = torch.chunk(expert_weight, 2, dim=0)

        return self._gated_mapping.hf_to_megatron({"gate": gate, "up": up}, megatron_module)

    def megatron_to_hf(
        self, megatron_weights: Optional[torch.Tensor], megatron_module: Optional[nn.Module]
    ) -> Dict[str, torch.Tensor]:
        converted = self._gated_mapping.megatron_to_hf(megatron_weights, megatron_module)
        if not converted:
            return {}

        fused: Dict[str, torch.Tensor] = {}
        for name, tensor in converted.items():
            if not name.endswith(".gate"):
                continue
            base_name = name[: -len(".gate")]
            up_tensor = converted.get(f"{base_name}.up")
            if up_tensor is None:
                continue
            concat_dim = 0 if tensor.ndim == 2 else 1
            fused[base_name] = torch.cat([tensor, up_tensor], dim=concat_dim)
        return fused

    def resolve(self, captures: Tuple[str, ...]) -> "MegatronParamMapping":
        resolved_megatron_param, resolved_hf_param = self._resolve_names(captures)
        return type(self)(
            resolved_megatron_param, resolved_hf_param, self.permute_dims, self.transpose_on_export
        )


def merge_qkv_biases(
    config: TransformerConfig, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """Merge separate Q, K, V bias vectors into Megatron's interleaved QKV format.

    Args:
        config (TransformerConfig): Transformer configuration.
        q (torch.Tensor): Query projection biases [hidden_size].
        k (torch.Tensor): Key projection biases [kv_hidden_size].
        v (torch.Tensor): Value projection biases [kv_hidden_size].

    Returns:
        torch.Tensor: Interleaved QKV biases in Megatron format as 1D tensor.
    """
    head_num = config.num_attention_heads
    num_query_groups = config.num_query_groups
    heads_per_group = head_num // num_query_groups
    head_size = config.kv_channels or (config.hidden_size // head_num)

    # Reshape biases to expose head dimension
    if getattr(config, "attention_output_gate", False):
        q, z = torch.chunk(q.view(head_num, head_size * 2), 2, dim=-1)
    else:
        q = q.view(head_num, head_size)
    k = k.view(num_query_groups, head_size)
    v = v.view(num_query_groups, head_size)

    # Interleave in Megatron pattern: [q1...qn, k1, v1, q1...qn, k2, v2, ...]
    qkv_biases = []
    for i in range(num_query_groups):
        qkv_biases.append(q[i * heads_per_group : (i + 1) * heads_per_group, :])
        if getattr(config, "attention_output_gate", False):
            qkv_biases.append(z[i * heads_per_group : (i + 1) * heads_per_group, :])
        qkv_biases.append(k[i : i + 1, :])
        qkv_biases.append(v[i : i + 1, :])

    # Concatenate and flatten back to 1D
    qkv = torch.cat(qkv_biases)
    return qkv.flatten()


def split_qkv_biases(
    config: TransformerConfig, qkv: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split Megatron's interleaved QKV bias into separate Q, K, V biases.

    Args:
        config (TransformerConfig): Transformer configuration.
        qkv (torch.Tensor): Interleaved QKV biases in Megatron format (1D
            tensor).

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Tuple of (Q, K, V) bias vectors.
    """
    head_num = config.num_attention_heads
    num_query_groups = config.num_query_groups
    heads_per_group = head_num // num_query_groups
    head_size = config.kv_channels or (config.hidden_size // head_num)
    if getattr(config, "attention_output_gate", False):
        qkv_total_dim = 2 * head_num + 2 * num_query_groups
        total_heads_per_group = 2 * heads_per_group + 2
    else:
        qkv_total_dim = head_num + 2 * num_query_groups
        total_heads_per_group = heads_per_group + 2

    # Reshape to expose interleaved structure
    qkv = qkv.reshape(qkv_total_dim, head_size)

    # Extract Q, K, V from interleaved pattern
    q_slice = torch.cat(
        [
            torch.arange(
                total_heads_per_group * i,
                total_heads_per_group * i + heads_per_group,
                device=qkv.device,
            )
            for i in range(num_query_groups)
        ]
    )
    k_slice = torch.arange(
        total_heads_per_group - 2, qkv_total_dim, total_heads_per_group, device=qkv.device
    )
    v_slice = torch.arange(
        total_heads_per_group - 1, qkv_total_dim, total_heads_per_group, device=qkv.device
    )

    if getattr(config, "attention_output_gate", False):
        z_slice = torch.cat(
            [
                torch.arange(
                    total_heads_per_group * i + heads_per_group,
                    total_heads_per_group * i + heads_per_group * 2,
                    device=qkv.device,
                )
                for i in range(num_query_groups)
            ]
        )
        # In HF implementation, matrix Q and Z are mixed, so we need to concatenate them.
        q = torch.cat([qkv[q_slice], qkv[z_slice]], dim=1).flatten()
    else:
        q = qkv[q_slice].flatten()
    k = qkv[k_slice].flatten()
    v = qkv[v_slice].flatten()

    return q, k, v


def merge_qkv_weights(
    provider: TransformerConfig, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """Merge separate Q, K, V weight matrices into Megatron's interleaved QKV format.

    Args:
        provider (TransformerConfig): Model configuration provider.
        q (torch.Tensor): Query projection weights [hidden_size, hidden_size] or
            bias [hidden_size].
        k (torch.Tensor): Key projection weights [kv_hidden_size, hidden_size]
            or bias [kv_hidden_size].
        v (torch.Tensor): Value projection weights [kv_hidden_size,
            hidden_size] or bias [kv_hidden_size].

    Returns:
        torch.Tensor: Interleaved QKV weights in Megatron format.
    """
    head_num = provider.num_attention_heads
    num_query_groups = provider.num_query_groups
    heads_per_group = head_num // num_query_groups
    head_size = provider.kv_channels or (provider.hidden_size // head_num)
    hidden_size = provider.hidden_size
    is_bias = q.ndim == 1
    q_head_size = head_size * 2 if getattr(provider, "attention_output_gate", False) else head_size

    # Reshape to expose head dimension
    if is_bias:
        q_reshaped = q.view(head_num, q_head_size)
        k_reshaped = k.view(num_query_groups, head_size)
        v_reshaped = v.view(num_query_groups, head_size)
    else:
        q_reshaped = q.view(head_num, q_head_size, hidden_size)
        k_reshaped = k.view(num_query_groups, head_size, hidden_size)
        v_reshaped = v.view(num_query_groups, head_size, hidden_size)
    if getattr(provider, "attention_output_gate", False):
        q_reshaped, z_reshaped = torch.chunk(q_reshaped, 2, dim=1)

    # Interleave in Megatron pattern: [q1...qn, k1, v1, q1...qn, k2, v2, ...]
    qkv_weights = []
    for i in range(num_query_groups):
        q_group = q_reshaped[i * heads_per_group : (i + 1) * heads_per_group]
        k_group = k_reshaped[i : i + 1]
        v_group = v_reshaped[i : i + 1]
        if getattr(provider, "attention_output_gate", False):
            z_group = z_reshaped[i * heads_per_group : (i + 1) * heads_per_group]
            qkv_weights.extend([q_group, z_group, k_group, v_group])
        else:
            qkv_weights.extend([q_group, k_group, v_group])

    qkv = torch.cat(qkv_weights, dim=0)

    assert (
        q.numel() + k.numel() + v.numel() == qkv.numel()
    ), f"QKV weights are not correctly merged, {q.shape=}, {k.shape=}, {v.shape=}, {qkv.shape=}"

    # Final reshape
    if is_bias:
        return qkv.reshape(-1)
    else:
        return qkv.reshape([-1, hidden_size])


def merge_qkvg_weights(
    provider: TransformerConfig, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor
) -> torch.Tensor:
    """Merge Q, K, V, and per-head scalar gate into Megatron's fused linear_qkv format.

    Layout is selected by ``provider.attention_output_gate``:

    * ``False``: standard GQA-interleaved QKV block followed by ``num_heads``
      scalar gate rows.
    * ``True``: Megatron-Core gated-attention layout ``[Q, gate, K, V]`` per
      query group. The HF scalar gate row for each head is expanded across the
      head dimension so MCore can apply it elementwise after attention.

    Args:
        provider: Model configuration provider.
        q/k/v: Same as :func:`merge_qkv_weights`.
        g: Per-head gate projection weights ``[num_heads, hidden_size]`` (bias
            shape ``[num_heads]``).
    """
    if getattr(provider, "attention_output_gate", False):
        head_num = provider.num_attention_heads
        num_query_groups = provider.num_query_groups
        heads_per_group = head_num // num_query_groups
        head_size = provider.kv_channels or (provider.hidden_size // head_num)
        hidden_size = provider.hidden_size

        if g.ndim != q.ndim:
            raise ValueError(f"QKV/gate tensor rank mismatch: q.ndim={q.ndim}, g.ndim={g.ndim}")
        if g.shape[0] != head_num:
            raise ValueError(
                f"Expected scalar gate rows for {head_num} heads, got shape={tuple(g.shape)}"
            )

        q_reshaped = q.view(head_num, head_size, hidden_size)
        k_reshaped = k.view(num_query_groups, head_size, hidden_size)
        v_reshaped = v.view(num_query_groups, head_size, hidden_size)
        g_reshaped = g.view(head_num, 1, hidden_size).expand(-1, head_size, -1)

        qkvg_weights = []
        for i in range(num_query_groups):
            q_group = q_reshaped[i * heads_per_group : (i + 1) * heads_per_group]
            g_group = g_reshaped[i * heads_per_group : (i + 1) * heads_per_group]
            k_group = k_reshaped[i : i + 1]
            v_group = v_reshaped[i : i + 1]
            qkvg_weights.extend([q_group, g_group, k_group, v_group])
        return torch.cat(qkvg_weights, dim=0).reshape(-1, hidden_size)

    qkv = merge_qkv_weights(provider, q, k, v)
    if g.ndim != qkv.ndim:
        raise ValueError(f"QKV/gate tensor rank mismatch: qkv.ndim={qkv.ndim}, g.ndim={g.ndim}")
    return torch.cat([qkv, g], dim=0)


def split_qkv_weights(
    provider: TransformerConfig, qkv: torch.Tensor, feature_dim: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split Megatron's interleaved QKV tensor into separate Q, K, V matrices.

    Args:
        provider (TransformerConfig): Model configuration provider.
        qkv (torch.Tensor): Interleaved QKV weights in Megatron format.
        feature_dim: Trailing tensor dimension used for reshape/split.
            Defaults to ``provider.hidden_size``.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Tuple of (Q, K, V)
            weight matrices.
    """
    head_num = provider.num_attention_heads
    num_query_groups = provider.num_query_groups
    heads_per_group = head_num // num_query_groups
    head_size = provider.kv_channels or (provider.hidden_size // head_num)
    if getattr(provider, "attention_output_gate", False):
        qkv_total_dim = 2 * head_num + 2 * num_query_groups
        total_heads_per_group = 2 * heads_per_group + 2
    else:
        qkv_total_dim = head_num + 2 * num_query_groups
        total_heads_per_group = heads_per_group + 2
    is_bias = qkv.ndim == 1

    if is_bias:
        hidden_size = 1
        qkv_reshaped = qkv.view(qkv_total_dim, head_size)
    elif feature_dim is not None:
        # Explicit trailing tensor dimension.
        hidden_size = feature_dim
        qkv_reshaped = qkv.view(qkv_total_dim, head_size, hidden_size)
    else:
        hidden_size = provider.hidden_size
        qkv_reshaped = qkv.view(qkv_total_dim, head_size, hidden_size)

    # Extract Q, K, V from interleaved pattern
    q_slice = torch.cat(
        [
            torch.arange(
                total_heads_per_group * i,
                total_heads_per_group * i + heads_per_group,
                device=qkv.device,
            )
            for i in range(num_query_groups)
        ]
    )
    k_slice = torch.arange(
        total_heads_per_group - 2, qkv_total_dim, total_heads_per_group, device=qkv.device
    )
    v_slice = torch.arange(
        total_heads_per_group - 1, qkv_total_dim, total_heads_per_group, device=qkv.device
    )

    if getattr(provider, "attention_output_gate", False):
        z_slice = torch.cat(
            [
                torch.arange(
                    total_heads_per_group * i + heads_per_group,
                    total_heads_per_group * i + heads_per_group * 2,
                    device=qkv.device,
                )
                for i in range(num_query_groups)
            ]
        )
        # In HF implementation, matrix Q and Z are mixed, so we need to concatenate them.
        q = torch.cat([qkv_reshaped[q_slice], qkv_reshaped[z_slice]], dim=1)
    else:
        q = qkv_reshaped[q_slice]
    k = qkv_reshaped[k_slice]
    v = qkv_reshaped[v_slice]

    assert (
        q.numel() + k.numel() + v.numel() == qkv.numel()
    ), f"QKV weights are not correctly merged, {q.shape=}, {k.shape=}, {v.shape=}, {qkv.shape=}"

    if is_bias:
        q = q.reshape(-1)
        k = k.reshape(-1)
        v = v.reshape(-1)
    else:
        q = q.reshape(-1, hidden_size)
        k = k.reshape(-1, hidden_size)
        v = v.reshape(-1, hidden_size)

    return q, k, v


def split_qkv_weights_scale(
    provider: TransformerConfig, qkv: torch.Tensor, quant_block_size: Tuple[int, int]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split Megatron's interleaved QKV tensor into separate Q, K, V matrices.

    Args:
        provider (TransformerConfig): Model configuration provider.
        qkv (torch.Tensor): Interleaved QKV weights in Megatron format.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Tuple of (Q, K, V)
            weight matrices.
    """
    return _split_qkv_weights_scale_by_row(provider, qkv, quant_block_size[0])


def _split_qkv_weights_scale_by_row(
    provider: TransformerConfig, qkv: torch.Tensor, row_block_size: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    head_num = provider.num_attention_heads
    num_query_groups = provider.num_query_groups
    heads_per_group = head_num // num_query_groups
    head_size = (provider.kv_channels or (provider.hidden_size // head_num)) // row_block_size

    if getattr(provider, "attention_output_gate", False):
        qkv_total_dim = 2 * head_num + 2 * num_query_groups
        total_heads_per_group = 2 * heads_per_group + 2
    else:
        qkv_total_dim = head_num + 2 * num_query_groups
        total_heads_per_group = heads_per_group + 2
    is_bias = qkv.ndim == 1

    if is_bias:
        hidden_size = 1
        qkv_reshaped = qkv.view(qkv_total_dim, head_size)
    else:
        hidden_size = qkv.shape[1]
        qkv_reshaped = qkv.view(qkv_total_dim, head_size, hidden_size)

    # Extract Q, K, V from interleaved pattern
    q_slice = torch.cat(
        [
            torch.arange(
                total_heads_per_group * i,
                total_heads_per_group * i + heads_per_group,
                device=qkv.device,
            )
            for i in range(num_query_groups)
        ]
    )
    k_slice = torch.arange(
        total_heads_per_group - 2, qkv_total_dim, total_heads_per_group, device=qkv.device
    )
    v_slice = torch.arange(
        total_heads_per_group - 1, qkv_total_dim, total_heads_per_group, device=qkv.device
    )

    if getattr(provider, "attention_output_gate", False):
        z_slice = torch.cat(
            [
                torch.arange(
                    total_heads_per_group * i + heads_per_group,
                    total_heads_per_group * i + heads_per_group * 2,
                    device=qkv.device,
                )
                for i in range(num_query_groups)
            ]
        )
        # In HF implementation, matrix Q and Z are mixed, so we need to concatenate them.
        q = torch.cat([qkv_reshaped[q_slice], qkv_reshaped[z_slice]], dim=1)
    else:
        q = qkv_reshaped[q_slice]
    k = qkv_reshaped[k_slice]
    v = qkv_reshaped[v_slice]

    assert (
        q.numel() + k.numel() + v.numel() == qkv.numel()
    ), f"QKV weights are not correctly merged, {q.shape=}, {k.shape=}, {v.shape=}, {qkv.shape=}"

    if is_bias:
        q = q.reshape(-1)
        k = k.reshape(-1)
        v = v.reshape(-1)
    else:
        q = q.reshape(-1, hidden_size, 1)
        k = k.reshape(-1, hidden_size, 1)
        v = v.reshape(-1, hidden_size, 1)

    return q, k, v


def split_qkvg_weights(
    provider: TransformerConfig, qkvg: torch.Tensor, feature_dim: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split Megatron's fused linear_qkv tensor into (Q, K, V, gate).

    Inverse of :func:`merge_qkvg_weights`.
    """
    head_num = provider.num_attention_heads
    if getattr(provider, "attention_output_gate", False):
        num_query_groups = provider.num_query_groups
        heads_per_group = head_num // num_query_groups
        head_size = provider.kv_channels or (provider.hidden_size // head_num)
        hidden_size = feature_dim or qkvg.shape[-1]
        total_heads_per_group = 2 * heads_per_group + 2
        qkvg_total_dim = 2 * head_num + 2 * num_query_groups
        qkvg_reshaped = qkvg.view(qkvg_total_dim, head_size, hidden_size)

        q_slice = torch.cat(
            [
                torch.arange(
                    total_heads_per_group * i,
                    total_heads_per_group * i + heads_per_group,
                    device=qkvg.device,
                )
                for i in range(num_query_groups)
            ]
        )
        g_slice = torch.cat(
            [
                torch.arange(
                    total_heads_per_group * i + heads_per_group,
                    total_heads_per_group * i + heads_per_group * 2,
                    device=qkvg.device,
                )
                for i in range(num_query_groups)
            ]
        )
        k_slice = torch.arange(
            total_heads_per_group - 2, qkvg_total_dim, total_heads_per_group, device=qkvg.device
        )
        v_slice = torch.arange(
            total_heads_per_group - 1, qkvg_total_dim, total_heads_per_group, device=qkvg.device
        )

        q = qkvg_reshaped[q_slice].reshape(-1, hidden_size)
        k = qkvg_reshaped[k_slice].reshape(-1, hidden_size)
        v = qkvg_reshaped[v_slice].reshape(-1, hidden_size)
        g = qkvg_reshaped[g_slice].reshape(head_num, head_size, hidden_size)[:, 0, :]
        return q, k, v, g

    gate_rows = head_num
    if qkvg.shape[0] <= gate_rows:
        raise ValueError(
            f"fused qkvg tensor too small for gate split: shape={tuple(qkvg.shape)}, gate_rows={gate_rows}"
        )
    qkv = qkvg[:-gate_rows]
    g = qkvg[-gate_rows:]
    q, k, v = split_qkv_weights(provider, qkv, feature_dim=feature_dim)
    return q, k, v, g


def merge_gdn_linear_weights(
    provider: TransformerConfig, qkvz: torch.Tensor, ba: torch.Tensor, tp_size: int = 1
) -> torch.Tensor:
    """Merge GDN linear weights into in_proj."""

    assert tp_size >= 1, f"tp_size must be greater than 0, but got {tp_size=}"

    hidden_size = provider.hidden_size
    qk_head_dim = provider.linear_key_head_dim
    v_head_dim = provider.linear_value_head_dim
    num_qk_heads = provider.linear_num_key_heads
    num_v_heads = provider.linear_num_value_heads
    qk_dim = qk_head_dim * num_qk_heads
    v_dim = v_head_dim * num_v_heads

    # Reshape to expose head dimension
    qkvz_reshaped = qkvz.reshape(
        num_qk_heads, (qk_dim * 2 + v_dim * 2) // num_qk_heads, hidden_size
    )
    ba_reshaped = ba.reshape(num_qk_heads, 2 * num_v_heads // num_qk_heads, hidden_size)
    q, k, v, z = torch.split(
        qkvz_reshaped,
        [
            qk_head_dim,
            qk_head_dim,
            num_v_heads // num_qk_heads * v_head_dim,
            num_v_heads // num_qk_heads * v_head_dim,
        ],
        dim=1,
    )
    b, a = torch.split(
        ba_reshaped, [num_v_heads // num_qk_heads, num_v_heads // num_qk_heads], dim=1
    )

    q, k, v, z, b, a = [weight.reshape(tp_size, -1, hidden_size) for weight in [q, k, v, z, b, a]]
    in_proj = torch.cat([q, k, v, z, b, a], dim=1)
    in_proj = in_proj.reshape(-1, hidden_size)

    assert (
        in_proj.numel() == qkvz.numel() + ba.numel()
    ), f"QKVZBA weights are not correctly merged, {qkvz.numel()=}, {ba.numel()=}, {in_proj.numel()=}, {tp_size=}"

    return in_proj


def split_gdn_linear_weights(
    provider: TransformerConfig,
    in_proj: torch.Tensor,
    tp_size: int = 1,
    feature_dim: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split GDN linear weights into QKVZ and BA.

    Args:
        provider: Transformer config with GDN dimensions.
        in_proj: Packed in-proj tensor.
        tp_size: Tensor-parallel world size used for packing layout.
        feature_dim: Trailing tensor dimension used for reshape/split.
            Defaults to ``provider.hidden_size``.
    """

    assert tp_size >= 1, f"tp_size must be greater than 0, but got {tp_size=}"

    feature_dim = provider.hidden_size if feature_dim is None else feature_dim
    qk_head_dim = provider.linear_key_head_dim
    v_head_dim = provider.linear_value_head_dim
    num_qk_heads = provider.linear_num_key_heads
    num_qk_heads_local_tp = provider.linear_num_key_heads // tp_size
    num_v_heads_local_tp = provider.linear_num_value_heads // tp_size
    qk_dim_local_tp = qk_head_dim * num_qk_heads_local_tp
    v_dim_local_tp = v_head_dim * num_v_heads_local_tp

    in_proj = in_proj.reshape(tp_size, -1, feature_dim)
    q, k, v, z, b, a = torch.split(
        in_proj,
        [
            qk_dim_local_tp,
            qk_dim_local_tp,
            v_dim_local_tp,
            v_dim_local_tp,
            num_v_heads_local_tp,
            num_v_heads_local_tp,
        ],
        dim=1,
    )

    q, k, v, z, b, a = [
        weight.reshape(num_qk_heads, -1, feature_dim) for weight in [q, k, v, z, b, a]
    ]
    qkvz = torch.cat([q, k, v, z], dim=1)
    ba = torch.cat([b, a], dim=1)

    qkvz = qkvz.reshape(-1, feature_dim)
    ba = ba.reshape(-1, feature_dim)

    assert (
        qkvz.numel() + ba.numel() == in_proj.numel()
    ), f"QKVZBA weights are not correctly split, {qkvz.numel()=}, {ba.numel()=}, {in_proj.numel()=}"

    return qkvz, ba


def _fuse_gdn_separate_to_grouped(
    config: TransformerConfig, qkv: torch.Tensor, z: torch.Tensor, b: torch.Tensor, a: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert four separate (flat) GDN projection tensors into the head-grouped
    ``qkvz`` and ``ba`` format expected by :func:`merge_gdn_linear_weights`.

    Args:
        config: Transformer configuration with GDN head dimensions.
        qkv: Flat ``[Q; K; V]`` tensor of shape ``(qk_dim*2 + v_dim, hidden)``.
        z:   Z projection of shape ``(v_dim, hidden)``.
        b:   B projection of shape ``(num_v_heads, hidden)``.
        a:   A projection of shape ``(num_v_heads, hidden)``.

    Returns:
        Tuple of (qkvz, ba) in head-grouped layout that
        :func:`merge_gdn_linear_weights` can consume directly.
    """
    hidden_size = config.hidden_size
    qk_head_dim = config.linear_key_head_dim
    v_head_dim = config.linear_value_head_dim
    num_qk_heads = config.linear_num_key_heads
    num_v_heads = config.linear_num_value_heads
    qk_dim = qk_head_dim * num_qk_heads
    v_dim = v_head_dim * num_v_heads
    v_per_group = num_v_heads // num_qk_heads

    expected_qkv = (qk_dim * 2 + v_dim, hidden_size)
    expected_z = (v_dim, hidden_size)
    expected_ba = (num_v_heads, hidden_size)
    if tuple(qkv.shape) != expected_qkv:
        raise ValueError(f"qkv shape mismatch: expected {expected_qkv}, got {tuple(qkv.shape)}")
    if tuple(z.shape) != expected_z:
        raise ValueError(f"z shape mismatch: expected {expected_z}, got {tuple(z.shape)}")
    if tuple(b.shape) != expected_ba:
        raise ValueError(f"b shape mismatch: expected {expected_ba}, got {tuple(b.shape)}")
    if tuple(a.shape) != expected_ba:
        raise ValueError(f"a shape mismatch: expected {expected_ba}, got {tuple(a.shape)}")

    # --- Split flat QKV into individual components ---
    q_flat, k_flat, v_flat = torch.split(qkv, [qk_dim, qk_dim, v_dim], dim=0)

    # --- Reshape every component to (num_qk_heads, per_group_dim, hidden) ---
    q_g = q_flat.reshape(num_qk_heads, qk_head_dim, hidden_size)
    k_g = k_flat.reshape(num_qk_heads, qk_head_dim, hidden_size)
    v_g = v_flat.reshape(num_qk_heads, v_per_group * v_head_dim, hidden_size)
    z_g = z.reshape(num_qk_heads, v_per_group * v_head_dim, hidden_size)
    b_g = b.reshape(num_qk_heads, v_per_group, hidden_size)
    a_g = a.reshape(num_qk_heads, v_per_group, hidden_size)

    # --- Assemble grouped qkvz and ba ---
    qkvz = torch.cat([q_g, k_g, v_g, z_g], dim=1).reshape(-1, hidden_size)
    ba = torch.cat([b_g, a_g], dim=1).reshape(-1, hidden_size)

    return qkvz, ba


def _split_gdn_grouped_to_separate(
    config: TransformerConfig,
    qkvz: torch.Tensor,
    ba: torch.Tensor,
    feature_dim: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert head-grouped ``qkvz`` and ``ba`` tensors (as produced by
    :func:`split_gdn_linear_weights`) back into four flat tensors.

    Returns:
        Tuple of (qkv, z, b, a) where each tensor has a flat per-component layout.
    """
    feature_dim = config.hidden_size if feature_dim is None else feature_dim
    qk_head_dim = config.linear_key_head_dim
    v_head_dim = config.linear_value_head_dim
    num_qk_heads = config.linear_num_key_heads
    num_v_heads = config.linear_num_value_heads
    v_per_group = num_v_heads // num_qk_heads

    expected_qkvz_dim0 = num_qk_heads * (qk_head_dim * 2 + v_per_group * v_head_dim * 2)
    expected_ba_dim0 = num_qk_heads * v_per_group * 2
    if qkvz.ndim != 2 or qkvz.shape[0] != expected_qkvz_dim0 or qkvz.shape[1] != feature_dim:
        raise ValueError(
            f"qkvz shape mismatch: expected ({expected_qkvz_dim0}, {feature_dim}), got {tuple(qkvz.shape)}"
        )
    if ba.ndim != 2 or ba.shape[0] != expected_ba_dim0 or ba.shape[1] != feature_dim:
        raise ValueError(
            f"ba shape mismatch: expected ({expected_ba_dim0}, {feature_dim}), got {tuple(ba.shape)}"
        )

    # --- Split grouped QKVZ ---
    qkvz_g = qkvz.reshape(num_qk_heads, -1, feature_dim)
    q_g, k_g, v_g, z_g = torch.split(
        qkvz_g,
        [qk_head_dim, qk_head_dim, v_per_group * v_head_dim, v_per_group * v_head_dim],
        dim=1,
    )
    q_flat = q_g.reshape(-1, feature_dim)
    k_flat = k_g.reshape(-1, feature_dim)
    v_flat = v_g.reshape(-1, feature_dim)
    z_flat = z_g.reshape(-1, feature_dim)
    qkv = torch.cat([q_flat, k_flat, v_flat], dim=0)

    # --- Split grouped BA ---
    ba_g = ba.reshape(num_qk_heads, -1, feature_dim)
    b_g, a_g = torch.split(ba_g, [v_per_group, v_per_group], dim=1)
    b_flat = b_g.reshape(-1, feature_dim)
    a_flat = a_g.reshape(-1, feature_dim)

    return qkv, z_flat, b_flat, a_flat


def merge_kv_biases(config: TransformerConfig, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Merge separate K, V bias vectors into Megatron's interleaved KV format (1D)."""
    num_query_groups = config.num_query_groups
    head_size = config.kv_channels or (config.hidden_size // config.num_attention_heads)

    k = k.view(num_query_groups, head_size)
    v = v.view(num_query_groups, head_size)

    pieces: List[torch.Tensor] = []
    for i in range(num_query_groups):
        pieces.append(k[i : i + 1, :])
        pieces.append(v[i : i + 1, :])

    kv = torch.cat(pieces, dim=0)
    return kv.reshape(-1)


def split_kv_biases(
    config: TransformerConfig, kv: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split Megatron's interleaved KV bias (1D) into separate K and V biases."""
    num_query_groups = config.num_query_groups
    head_size = config.kv_channels or (config.hidden_size // config.num_attention_heads)
    kv_total_dim = 2 * num_query_groups

    kv_reshaped = kv.view(kv_total_dim, head_size)

    k_slice = torch.arange(0, kv_total_dim, 2, device=kv.device)
    v_slice = torch.arange(1, kv_total_dim, 2, device=kv.device)

    k = kv_reshaped[k_slice].reshape(-1)
    v = kv_reshaped[v_slice].reshape(-1)
    return k, v


def merge_kv_weights(provider: TransformerConfig, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Merge separate K, V weights into Megatron's interleaved KV format (2D)."""
    num_query_groups = provider.num_query_groups
    head_size = provider.kv_channels or (provider.hidden_size // provider.num_attention_heads)
    hidden_size = provider.hidden_size

    k_reshaped = k.view(num_query_groups, head_size, hidden_size)
    v_reshaped = v.view(num_query_groups, head_size, hidden_size)

    pieces: List[torch.Tensor] = []
    for i in range(num_query_groups):
        pieces.append(k_reshaped[i : i + 1])
        pieces.append(v_reshaped[i : i + 1])

    kv = torch.cat(pieces, dim=0)
    return kv.view(-1, hidden_size)


def split_kv_weights(
    provider: TransformerConfig, kv: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split Megatron's interleaved KV weights (2D) into separate K and V matrices."""
    num_query_groups = provider.num_query_groups
    head_size = provider.kv_channels or (provider.hidden_size // provider.num_attention_heads)
    hidden_size = kv.shape[-1]
    kv_total_dim = 2 * num_query_groups

    kv_reshaped = kv.view(kv_total_dim, head_size, hidden_size)

    k_slice = torch.arange(0, kv_total_dim, 2, device=kv.device)
    v_slice = torch.arange(1, kv_total_dim, 2, device=kv.device)

    k = kv_reshaped[k_slice].reshape(-1, hidden_size)
    v = kv_reshaped[v_slice].reshape(-1, hidden_size)
    return k, v


class HCAlphaMapping(MegatronParamMapping):
    """Map a 3-element HF hyper-connection scale tensor to/from Megatron's
    three separate ``alpha_pre`` / ``alpha_post`` / ``alpha_res`` parameters.

    The HF checkpoint stores the three scaling coefficients stacked along dim 0
    as ``<layer>.hc_{kind}_scale`` (shape ``[3]``); Megatron keeps them as three
    independent ``nn.Parameter([1])`` tensors on a single module.

    One :class:`HCAlphaMapping` is registered **per alpha parameter** (index 0, 1,
    or 2). Import extracts element ``[index:index+1]`` from the HF scale tensor.
    Only the ``index == 0`` (``alpha_pre``) mapping exports: it gathers all three
    alphas from the same module and concatenates them back into the ``[3]`` tensor.
    The ``index != 0`` mappings return ``{}`` on export (the primary already wrote
    all three), so they exist only to drive the import side and suppress export
    "no mapping found" warnings.

    This serves both models with a single wrapped ``hyper_connection`` submodule
    (e.g. GLM-5.3-Flash) and models with separate ``self_attention_hyper_connection`` /
    ``mlp_hyper_connection`` submodules (e.g. DeepSeek-V4); the caller chooses the
    ``megatron_param`` path accordingly.
    """

    def __init__(self, megatron_param: str, hf_param: str, index: int):
        super().__init__(megatron_param=megatron_param, hf_param=hf_param)
        self._index = index
        # index != 0 imports from the shared HF scale tensor but never exports
        # (the index-0 mapping writes all three alphas). Its HF key collides with
        # the index-0 key, so skip the strict hf_keys presence check.
        self.allow_hf_name_mismatch = index != 0

    def resolve(self, captures):
        resolved_mg, resolved_hf = self._resolve_names(captures)
        return HCAlphaMapping(resolved_mg, resolved_hf, self._index)

    def hf_to_megatron(self, hf_weights, megatron_module):
        attr = ("alpha_pre", "alpha_post", "alpha_res")[self._index]
        return hf_weights[self._index : self._index + 1].to(getattr(megatron_module, attr).device)

    def megatron_to_hf(self, megatron_weights, megatron_module):
        if self._index != 0:
            return {}
        parts = []
        for attr, weight in (
            ("alpha_pre", megatron_weights),
            ("alpha_post", getattr(megatron_module, "alpha_post", None)),
            ("alpha_res", getattr(megatron_module, "alpha_res", None)),
        ):
            parts.append(self.broadcast_from_pp_rank(weight, cache_key=f"{self.hf_param}_{attr}"))
        if parts[0] is None:
            return {}
        return {self.hf_param: torch.cat([self.maybe_dequantize(part).float() for part in parts])}


class InitOnlyMapping(MegatronParamMapping[torch.Tensor]):
    """Keep Megatron initialization for parameters that have no HF counterpart.

    Used for hyper-connection / MTP parameters that MCore creates but the source
    checkpoint does not store. The mapping declares a synthetic HF key ending in
    ``__init_only__`` (which never exists in any checkpoint) and returns ``None``
    on import so the random init is preserved. Export is a no-op.

    Bridges that want this behaviour should also short-circuit the HF state-dict
    lookup for the synthetic key (e.g. in ``maybe_modify_loaded_hf_weight``).
    """

    def __init__(self, megatron_param: str):
        super().__init__(megatron_param=megatron_param, hf_param=megatron_param + ".__init_only__")
        self.allow_hf_name_mismatch = True

    def resolve(self, captures):
        resolved_mg, _ = self._resolve_names(captures)
        return InitOnlyMapping(resolved_mg)

    def hf_to_megatron(self, hf_weights, megatron_module):
        # hf_weights is None when the (synthetic) HF key is absent -> keep init.
        return None

    def megatron_to_hf(self, megatron_weights, megatron_module):
        return {}
