# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Megatron-FSDP configuration and compatibility with the former DDP configuration."""

import os
import warnings
from dataclasses import dataclass, fields, replace
from typing import Optional

import torch

from ..utils import is_torch_min_version

_SHARDING_STRATEGIES = ("no_shard", "optim", "optim_grads", "optim_grads_params")

# Only the compatibility boundary knows the historical field names.
_LEGACY_NAMES = {
    "version": "megatron_fsdp_version",
    "padded_shards": "megatron_fsdp_padded_shards",
    "double_buffer": "fsdp_double_buffer",
    "db_use_persist_buf_on_alloc_fail": "fsdp_db_use_persist_buf_on_alloc_fail",
    "all_gather_in_start_param_sync": "fsdp_all_gather_in_start_param_sync",
    "manual_registration": "fsdp_manual_registration",
    "main_params_dtype": "megatron_fsdp_main_params_dtype",
    "main_grads_dtype": "megatron_fsdp_main_grads_dtype",
    "grad_comm_dtype": "megatron_fsdp_grad_comm_dtype",
    "use_decoupled_grad": "megatron_fsdp_use_decoupled_grad",
    "cuda_graph_mode": "megatron_fsdp_cuda_graph_mode",
    "enable_fine_grained_param_gather": "megatron_fsdp_enable_fine_grained_param_gather",
    "max_pool_double_buffer": "megatron_fsdp_max_pool_double_buffer",
}


@dataclass
class FullyShardedDataParallelConfig:
    """Configuration for Megatron-FSDP v1 and v2, independent of DDP.

    This describes requested settings. ``resolved()`` returns a validated copy with
    inherited expert strategies and buffer defaults applied, without mutating the
    caller's configuration. Runtime streams and process groups live in the adapter.
    """

    version: int = 1
    """Megatron-FSDP implementation version. Valid values are 1 and 2."""

    padded_shards: bool = True
    """For v2, pad parameter groups to equal shard sizes. False uses row-aligned,
    potentially unequal shards to reduce padding overhead, and is incompatible
    with nccl_ub, whose symmetric-memory collectives require equal shard sizes.
    """

    num_distributed_optimizer_instances: int = 1
    """Size of the outer data-parallel axis. One selects a single FSDP axis."""

    data_parallel_sharding_strategy: str = "no_shard"
    """Sharding strategy for FSDP. Valid values are 'no_shard', 'optim',
      'optim_grads', 'optim_grads_params'."""

    expert_data_parallel_sharding_strategy: Optional[str] = None
    """Sharding strategy applied to expert (MoE) parameters on DP-Shard. Valid values are
      'no_shard', 'optim', 'optim_grads', 'optim_grads_params'. When set,
      `data_parallel_sharding_strategy` only applies to non-expert parameters, which allows
      trading DP-Shard communication against memory separately for the two parameter classes
      (e.g. 'optim' on non-experts and 'optim_grads_params' on experts). Expert parameters are
      already sharded over a narrower DP group than non-expert parameters when expert
      parallelism is enabled, so the two classes have very different traffic-per-byte.
      None inherits `data_parallel_sharding_strategy` when the config is resolved."""

    outer_dp_sharding_strategy: str = "no_shard"
    """
    Sharding strategy for outer data parallel group in Hybrid Sharded Data Parallel (HSDP) mode.
    Valid values are 'no_shard', 'optim'. This option is only effective when Hybrid FSDP is enabled.
    """

    expert_outer_dp_sharding_strategy: Optional[str] = None
    """Sharding strategy for the outer expert data-parallel group in MFSDP v2.
    Valid values are ``'no_shard'``, ``'optim'``, ``'optim_grads'``, and
    ``'optim_grads_params'``. None is replaced with
    ``outer_dp_sharding_strategy`` when the config is resolved.
    """

    overlap_grad_reduce: bool = False
    """If true, overlap grad all-reduce / reduce-scatter with backward compute."""

    overlap_param_gather: bool = False
    """If true, overlap param all-gather with forward compute."""

    bucket_size: Optional[int] = None
    """Maximum number of parameters in each bucket. If unspecified, MCore uses a default
       value of max(40000000, 1000000 * dp_size) parameters (larger DP sizes need larger
       buckets to ensure collectives do not become latency-bound)."""

    num_buckets: Optional[int] = None
    """Number of buckets for data-parallel communication. Should only specify one of
       `bucket_size` and `num_buckets`. If `num_buckets` is specified, `bucket_size`
       will be determined at runtime."""

    check_for_nan_in_grad: bool = False
    """
    If true, check for NaNs and Infs in gradients _before_ communication collective.
    Invoked by `start_grad_sync` such as in the Megatron-LM DDP training API.
    """

    average_in_collective: bool = False
    """If true, compute average in collective directly, as opposed to dividing by the
       dp_size first and then computing sum in the collective."""

    fp8_param_gather: bool = False
    """If true, keep the compute param in fp8 (do not use any other intermediate dtype) and
       perform the param all-gather in fp8."""

    fp4_param_gather: bool = False
    """If true, keep the compute param in fp4 (do not use any other intermediate dtype) and
       perform the param all-gather in fp4."""

    gradient_reduce_div_fusion: bool = True
    """If true, perform gradient reduce and division fusion."""

    suggested_communication_unit_size: int = None
    """Specifies the number of elements to communicate at once during
      FSDP (Fully Sharded Data Parallel) operations.
      This flag also affects FSDP all-gather prefetch behavior. Setting a larger
      value increases the communication buffer size, while a smaller value
      disables prefetching and may degrade performance. Adjust this value
      based on your system's memory and performance requirements."""

    keep_fp8_transpose_cache: bool = False
    """If true, keep the fp8 transpose cache when using Megatron FSDP."""

    nccl_ub: bool = False
    """If true, allocate and register NCCL userbuffer for param and grad buffer.
      This flag enables SM efficient nccl algorithm that could improve the performance
      of FSDP and DP with comm_overlap. This flag will be much more effective when used
      together with sharp.
      The follwoing will be the expected number of SM usage for various cases.
      (Note that this is just a reference number and the number of SM usage could vary
      on message size, communication domain size and nccl version.)
      | Communication domain | use_sharp | SM usage of "AG/RS" |
      |----------------------|-----------|---------------------|
      | NVL                  | N/A       | 4 / 5               |
      | NVL+IB               | False     | 16 / 16             |
      | NVL+IB               | True      | 6 / 6               |
      | IB                   | False     | 1 / 4               |
      | IB                   | True      | 1 / 1               |
    """

    double_buffer: bool = False
    """If true, use persistently allocated double buffers for the
      temporary memory needed in the Megatron FSDP communications.
      This option will cause additional memory overhead, however, it is necessary for
      to register user buffer (nccl_ub=True) for the Megatron FSDP.
      This option will be automatically set to True when nccl_ub=True.
    """

    db_use_persist_buf_on_alloc_fail: bool = False
    """Whether to fall back to persistent buffer when a bucket does not
       fit FSDP double buffer size. If true, FSDP will use the persistently
       allocated buffer for the bucket that does not fit, it will enable NCCL
       user buffer with the cost of more memory usage. If false, FSDP will use
       Dynamic memory allocator, NCCL user buffer won't not enabled, which
       usually leads to low performance.
    """

    all_gather_in_start_param_sync: bool = True
    """
    If True, use all-gather during the initial Megatron-FSDP parameter
    synchronization step. This can increase overlap between the first
    parameter all-gather and computation, helping to better hide the
    initial communication cost. Should be deactivated when using
    full-iteration CG, or partial CG if AG/RS is launched beyond the
    CG capture scope but is waited on during the capture scope.
    """

    disable_symmetric_registration: bool = False
    """If true, disable symmetric (window) registration for NCCL userbuffer registration.
      This option will force to use conventional (local) userbuffer registration
      when nccl_ub is set.
    """

    manual_registration: bool = False
    """If true, manually register the FSDP communication buffers to NCCL user buffer.
      This option is only effective for v1 with nccl_ub enabled.
      For symmetric registration with large models, the registration itself can take
      a significant amount of time. This option minimizes the number of registration calls
      to minimize the registration time.
    """

    main_params_dtype: Optional[torch.dtype] = torch.float32
    """Data type for the main weight buffer utilized for distributed optimization
      and quantization with Megatron-FSDP. If set to None, the model compute weight
      buffer will take the role of the main weights, or when no sharding is applied,
      the native model weights become the main weights. Defaults to torch.float32.
    """

    main_grads_dtype: Optional[torch.dtype] = None
    """Data type for the main gradient buffer utilized for distributed optimization with
      Megatron-FSDP. If set to None, main gradients will match the dtype of the model
      compute parameters specified by the user model. Defaults to None.
    """

    grad_comm_dtype: Optional[torch.dtype] = None
    """Data type for gradient gather / scatter communications. Can be utilized to reduce
      communication latency, but adds overhead for type-casting and copy operations.
      If using NCCL UBR v2.27+, gradient reduction may be performed in high-precision
      depending on the network domain (NVLink or IB), and can enable mixed-precision
      communication and accumulation, e.g. setting grad_comm_dtype to `BF16` can support
      `FP32` reduction even though we have `BF16` input and output communication buffers.
      If set to None, the `main_grads_dtype` is used. If using HSDP (either DP-Replicate
      or DP-Outer in `outer_dp_sharding_strategy`), `no_shard`, `optim`, or a
      `FixedPoolAllocator` (`double_buffer`), allocating `dtype`-custom gradient
      communication buffers (per FSDP group) adds memory overhead. Defaults to None.
      No additional memory is allocated when `grad_comm_dtype == main_grads_dtype`.
    """

    use_decoupled_grad: bool = False
    """If true, Megatron-FSDP's ParamAndGradBuffer uses the precision-aware optimizer
      gradient path (e.g. `decoupled_grad` on optimizer parameters) instead of casting
      main gradients to parameter dtype for `.grad`.
    """

    cuda_graph_mode: bool = False
    """If set to True, Megatron-FSDP will practice CUDA graph-safe operations, such as
    not dereferencing `param.grad` after the optimizer step to preserve references for
    CUDA graph replay. Can affect memory utilization in some cases, such as when the
    gradient shard is not a view of the Megatron-FSDP sharded gradient buffer, so
    FusedAdam(use_decoupled_grad=True) + use_decoupled_grad=True or
    setting main_params_dtype == main_grads_dtype is
    recommended to avoid casting the gradient to the parameter precision and creating
    a casted-copy of the gradient shard that cannot be dereferenced due to replay.
    """

    enable_fine_grained_param_gather: bool = False
    """If set to True, enables fine-grained parameter gathering for Megatron-FSDP.
      This feature increases the overlap between parameter all-gather and forward computation,
      at the cost of more frequent communication calls.
      For MXFP8, this approach helps save memory during fine-grained activation
      recomputation, because MXFP8 forward and backward passes use different
      parameter representations (rowwise data for forward, colwise data for backward).
      In this mode, only the rowwise parameters of modules involved in recomputation
      will be unsharded.
    """

    max_pool_double_buffer: bool = False
    """
    Builds a double buffer maxpool that can be recycled across asymmetric / hybrid
    FSDP units, instead of the symmetrical FixedPoolAllocator that requires exact
    parity between FSDP units, when using double_buffer=True. Enables NCCL
    user buffer registration and CUDA graph replay for models with asymmetrical
    FSDP units, such as models with hybrid architectures (e.g. Mamba and MoE).
    """

    hfsdp_param_gather_overlap: bool = False
    """If true, pipeline HFSDP parameter gathers across the DP-Outer and DP-Inner
    communication domains. DP-Inner retains its size-based prefetch policy, while
    DP-Outer is prefetched one additional FSDP unit beyond the DP-Inner frontier.
    Only effective with ``outer_dp_sharding_strategy='optim'``.
    """

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate configuration values without changing them or allocating CUDA state."""
        if self.version not in (1, 2):
            raise ValueError("version must be either 1 or 2")
        if self.num_distributed_optimizer_instances < 1:
            raise ValueError("num_distributed_optimizer_instances must be positive")
        for name in (
            "data_parallel_sharding_strategy",
            "expert_data_parallel_sharding_strategy",
            "outer_dp_sharding_strategy",
            "expert_outer_dp_sharding_strategy",
        ):
            value = getattr(self, name)
            if value is None and name.startswith("expert_"):
                continue
            if value not in _SHARDING_STRATEGIES:
                raise ValueError(
                    f"{name} must be one of {list(_SHARDING_STRATEGIES)}, got {value!r}."
                )
        if self.num_buckets is not None:
            if self.bucket_size is not None:
                raise ValueError("Cannot specify both num_buckets and bucket_size")
            if self.num_buckets <= 0:
                raise ValueError("num_buckets must be greater than 0")
        if self.nccl_ub and not is_torch_min_version("2.11.0a0"):
            if "expandable_segments:True" in os.getenv("PYTORCH_CUDA_ALLOC_CONF", "").split(","):
                raise ValueError("expandable_segments:True is not supported with nccl_ub")
        if self.version == 2:
            for name in (
                "db_use_persist_buf_on_alloc_fail",
                "manual_registration",
                "use_decoupled_grad",
                "enable_fine_grained_param_gather",
                "max_pool_double_buffer",
                "fp4_param_gather",
            ):
                if getattr(self, name):
                    raise ValueError(f"MFSDP v2 does not support {name}.")
            if self.num_buckets is not None:
                raise ValueError("MFSDP v2 does not support num_buckets.")
            if self.nccl_ub and self.disable_symmetric_registration:
                raise ValueError(
                    "MFSDP v2 requires symmetric registration when nccl_ub is enabled."
                )
            if self.nccl_ub and not self.padded_shards:
                raise ValueError(
                    "MFSDP v2 padded_shards=False is incompatible with nccl_ub: "
                    "symmetric-memory collectives require equal shard sizes."
                )
            for name in ("outer_dp_sharding_strategy", "expert_outer_dp_sharding_strategy"):
                strategy = getattr(self, name) or self.outer_dp_sharding_strategy
                if strategy != "no_shard" and self.num_distributed_optimizer_instances <= 1:
                    raise ValueError(
                        f"MFSDP v2 {name}={strategy!r} requires an outer DP axis, "
                        "i.e. num_distributed_optimizer_instances > 1."
                    )

    def resolved(self) -> "FullyShardedDataParallelConfig":
        """Resolve inherited settings on a copy for model construction."""
        self.validate()
        return replace(
            self,
            expert_data_parallel_sharding_strategy=(
                self.expert_data_parallel_sharding_strategy or self.data_parallel_sharding_strategy
            ),
            expert_outer_dp_sharding_strategy=(
                self.expert_outer_dp_sharding_strategy or self.outer_dp_sharding_strategy
            ),
            double_buffer=self.double_buffer or self.max_pool_double_buffer,
        )

    @classmethod
    def from_ddp_config(cls, ddp_config) -> "FullyShardedDataParallelConfig":
        """Convert legacy FSDP settings without changing the supplied DDP config."""
        if isinstance(ddp_config, _LegacyFSDPConfigView):
            return ddp_config.fsdp_config
        from .distributed_data_parallel_config import DistributedDataParallelConfig

        if not isinstance(ddp_config, DistributedDataParallelConfig):
            raise TypeError("ddp_config must be a DistributedDataParallelConfig")
        kwargs = {
            f.name: getattr(ddp_config, _LEGACY_NAMES.get(f.name, f.name)) for f in fields(cls)
        }
        # Legacy training resolves a bucket count into bucket_size in place. Its
        # constructor already checked mutual exclusion before this derivation.
        if kwargs["num_buckets"] is not None and kwargs["bucket_size"] is not None:
            if kwargs["version"] == 1:
                kwargs["num_buckets"] = None
            else:
                # Preserve the unsupported v2 request for the adapter to reject.
                kwargs["bucket_size"] = None
        return cls(**kwargs)

    def as_legacy_config(self, ddp_config=None):
        """Expose legacy names as a view for the v1 engine and unmigrated callers."""
        return _LegacyFSDPConfigView(self, ddp_config)


_FSDP_FIELD_NAMES = {
    _LEGACY_NAMES.get(f.name, f.name): f.name for f in fields(FullyShardedDataParallelConfig)
}


class _LegacyFSDPConfigView:
    """Forward FSDP reads and writes to one authoritative config.

    DDP-only values are retained solely for old integration code. In particular,
    v1's optimizer may still read ``use_distributed_optimizer``. FSDP settings
    are never read from the retained legacy input after conversion.
    """

    def __init__(self, fsdp_config, ddp_config=None) -> None:
        from .distributed_data_parallel_config import DistributedDataParallelConfig

        if isinstance(ddp_config, _LegacyFSDPConfigView):
            ddp_config = ddp_config._legacy
        object.__setattr__(self, "fsdp_config", fsdp_config)
        object.__setattr__(
            self,
            "_legacy",
            ddp_config if ddp_config is not None else DistributedDataParallelConfig(),
        )

    def __getattr__(self, name):
        if name == "use_megatron_fsdp":
            return True
        # Pickle/deepcopy may inspect an instance before restoring its state.
        config = object.__getattribute__(self, "fsdp_config")
        if name in _FSDP_FIELD_NAMES:
            return getattr(config, _FSDP_FIELD_NAMES[name])
        return getattr(object.__getattribute__(self, "_legacy"), name)

    def __setattr__(self, name, value) -> None:
        if name in _FSDP_FIELD_NAMES:
            setattr(self.fsdp_config, _FSDP_FIELD_NAMES[name], value)
        else:
            setattr(self._legacy, name, value)

    def __eq__(self, other):
        if not isinstance(other, _LegacyFSDPConfigView):
            return NotImplemented
        return self.fsdp_config == other.fsdp_config and self._legacy == other._legacy

    def __repr__(self) -> str:
        return repr(self.fsdp_config)


def resolve_fsdp_config(*, fsdp_config=None, ddp_config=None, version=None):
    """Normalize the new or legacy wrapper input at the adapter boundary."""
    if fsdp_config is not None and ddp_config is not None:
        raise ValueError("Specify only one of fsdp_config and ddp_config")
    # Positional callers and generic legacy wrappers still use the ddp_config slot.
    if isinstance(ddp_config, FullyShardedDataParallelConfig):
        fsdp_config, ddp_config = ddp_config, None
    if fsdp_config is None:
        if ddp_config is None:
            raise ValueError("fsdp_config is required")
        if not isinstance(ddp_config, _LegacyFSDPConfigView):
            warnings.warn(
                "Passing ddp_config to Megatron-FSDP is deprecated; use "
                "fsdp_config=FullyShardedDataParallelConfig(...).",
                DeprecationWarning,
                stacklevel=3,
            )
        fsdp_config = FullyShardedDataParallelConfig.from_ddp_config(ddp_config)
    if not isinstance(fsdp_config, FullyShardedDataParallelConfig):
        raise TypeError("fsdp_config must be a FullyShardedDataParallelConfig")
    if version is not None and fsdp_config.version != version:
        if ddp_config is None:
            raise ValueError(f"This FSDP wrapper requires version={version}")
        # Explicit V1/V2 classes historically selected the implementation even
        # when the legacy DDP config retained its default version.
        fsdp_config = replace(fsdp_config, version=version)
    fsdp_config = fsdp_config.resolved()
    return fsdp_config, fsdp_config.as_legacy_config(ddp_config)
