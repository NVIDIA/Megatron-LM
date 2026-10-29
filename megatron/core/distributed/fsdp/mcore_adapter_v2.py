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

"""MCore adapter for the Megatron-FSDP v2 implementation."""

import contextlib
import logging
from contextlib import nullcontext
from typing import List, NamedTuple, Optional, Tuple, Type

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed import DeviceMesh
from torch.distributed.tensor import Partial, Replicate, Shard
from torch.distributed.tensor.placement_types import Placement

from megatron.core.config_logger import has_config_logger_enabled, log_config_to_disk
from megatron.core.distributed.data_parallel_base import _BaseDataParallel
from megatron.core.distributed.distributed_data_parallel_config import DistributedDataParallelConfig
from megatron.core.distributed.fsdp.fsdp_unit_collector import FsdpUnitCollector
from megatron.core.models.common.combined_1f1b_mfsdp_scheduler import register_combined_1f1b_hooks
from megatron.core.models.hybrid.shortcut_block import ShortcutMoEBlock
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.mamba_layer import MambaLayer
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core.utils import log_single_rank

__all__ = ["FullyShardedDataParallelV2"]

try:
    from megatron.core.distributed.fsdp.src.megatron_fsdp import MixedPrecisionPolicy
    from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
        FsdpContext,
        Placements,
        RuntimePolicy,
        SchedulePolicy,
        ShardLayout,
        fully_shard,
        fully_shard_context,
        microbatch,
    )
    from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.fully_shard import (
        current_fully_shard_context,
    )
    from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.module_utils import (
        copy_parameter_attributes,
    )
    from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.parameter_group import (
        sync_model_weights_from_main_weights,
    )
    from megatron.core.distributed.fsdp.src.megatron_fsdp.utils import get_sharding_strategy

    HAVE_MEGATRON_FSDP = True
except ImportError as import_megatron_fsdp_error:
    IMPORT_MEGATRON_FSDP_ERROR = import_megatron_fsdp_error
    HAVE_MEGATRON_FSDP = False

logger = logging.getLogger(__name__)


def _materialize_meta_module(module: nn.Module, device: torch.device | None) -> None:
    """Materialize and initialize one module's direct meta parameters."""
    # Container modules may have parameterized children but no direct parameters or
    # reset_parameters() method; their children are materialized separately.
    if not any(parameter.is_meta for parameter in module.parameters(recurse=False)):
        return

    if device is None:
        device = torch.device("cpu")

    def materialize_tensor(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.is_meta:
            return torch.empty_like(tensor, device=device)
        return tensor

    reset_parameters = getattr(module, "reset_parameters", None)
    if reset_parameters is None:
        raise ValueError(
            f"Meta parameter module {type(module).__qualname__} does not have a "
            "reset_parameters method."
        )

    # Both _apply() and TE reset_parameters() may replace Parameter objects.
    parameter_states = [
        (name, parameter, parameter.requires_grad)
        for name, parameter in module.named_parameters(recurse=False)
    ]

    module._apply(materialize_tensor, recurse=False)
    reset_parameters()

    for name, original_parameter, requires_grad in parameter_states:
        parameter = module.get_parameter(name)
        parameter.requires_grad_(requires_grad)
        copy_parameter_attributes(original_parameter, parameter)


def _materialize_owned_meta_modules(module: nn.Module, device: torch.device | None) -> None:
    """Materialize meta parameters reachable from one FSDP unit.

    PyTorch and Transformer Engine differ in reset_parameters(): PyTorch typically initializes
    existing Parameters in place, while TE may replace a Parameter via setattr(). Run resets
    before fully_shard() so FSDP sees the final Parameter objects.
    """
    for submodule in module.modules():
        _materialize_meta_module(submodule, device)


class FullyShardedDataParallelV2(_BaseDataParallel):
    """MFSDP v2 wrapper for the Megatron model.

    Supports MXFP8 compute with either high-precision or MXFP8 parameter gathers.
    For MXFP8 gathers, enable both ``config.fp8_param`` and ``ddp_config.fp8_param_gather``;
    MCore's model initialization preserves the high-precision values used for main weights.
    """

    def __init__(
        self,
        config: TransformerConfig,
        ddp_config: DistributedDataParallelConfig,
        module: torch.nn.Module,
        fsdp_unit_modules: Optional[List[Type[torch.nn.Module]]] = None,
        disable_bucketing: bool = False,
        device: Optional[torch.device] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        """Initialize the MFSDP v2 wrapper.

        Args:
            config: Transformer configuration for the model.
            ddp_config: Data-parallel and sharding configuration.
            module: Root model module to shard.
            fsdp_unit_modules: Module types to shard as child FSDP units. If
                unspecified, transformer and Mamba layers, shortcut MoE blocks, and
                TEGroupedMLP/SequentialMLP expert containers are used. An explicit list
                replaces these defaults. Expert containers use the expert-DP mesh;
                custom lists must include them when expert and dense sharding settings
                differ.
            disable_bucketing: Compatibility argument that must remain ``False`` for
                MFSDP v2.
            device: Device used to construct the data-parallel mesh and, when this wrapper
                opens its own ``fully_shard_context``, that context's device. Defaults to
                CUDA. When the caller has already opened an ambient context (multi-chunk
                wrapping), this wrapper joins it and the ambient context's device and
                ``use_symmetric_memory`` settings apply instead.
            pg_collection: Explicit process groups. The ``dp_cp`` group defines the
                data-parallel mesh.

        Raises:
            ImportError: If the Megatron FSDP implementation is unavailable.
            ValueError: If required process groups are missing or the configuration
                requests a feature unsupported by MFSDP v2.
        """
        if not HAVE_MEGATRON_FSDP:
            raise IMPORT_MEGATRON_FSDP_ERROR
        if pg_collection is None:
            raise ValueError("MFSDP v2 requires an explicit ProcessGroupCollection.")
        FullyShardedDataParallelV2._validate_config(
            config, ddp_config, module, pg_collection, disable_bucketing
        )

        if has_config_logger_enabled(config):
            log_config_to_disk(config, locals(), prefix=type(self).__name__)

        # Optimizer construction reads this attribute; retain the v1 contract for compatibility.
        self.ddp_config = ddp_config

        # Importing experts at module load creates a cycle through pipeline schedules
        # and fsdp_dtensor_checkpoint, which then incorrectly marks MFSDP unavailable.
        from megatron.core.transformer.moe.experts import SequentialMLP, TEGroupedMLP

        if fsdp_unit_modules is None:
            fsdp_unit_modules = [
                TransformerLayer,
                MambaLayer,
                ShortcutMoEBlock,
                TEGroupedMLP,
                SequentialMLP,
            ]

        log_single_rank(
            logger, logging.INFO, "Setting up FullyShardedDataParallelV2 with config %s", ddp_config
        )
        self.mp_policy = MixedPrecisionPolicy(
            main_params_dtype=ddp_config.megatron_fsdp_main_params_dtype,
            main_grads_dtype=ddp_config.megatron_fsdp_main_grads_dtype,
            grad_comm_dtype=ddp_config.megatron_fsdp_grad_comm_dtype,
        )
        log_single_rank(
            logger,
            logging.INFO,
            "Setting up Megatron-FSDP MixedPrecisionPolicy with config %s",
            self.mp_policy,
        )

        device_type = device.type if device is not None else "cuda"

        if has_outer_dp_axis := ddp_config.num_distributed_optimizer_instances > 1:
            # There is no HSDP/HFSDP special case:
            # each axis takes the placements of its own strategy, so no_shard outer over
            # ZeRO-3 inner is HSDP and ZeRO-1 outer over ZeRO-3 inner is HFSDP.
            dp_mesh = _build_hybrid_dp_mesh(
                pg_collection.inter_dist_opt, pg_collection.intra_dp_cp, device_type
            )
            outer = _DATA_PARALLEL_PLACEMENTS[ddp_config.outer_dp_sharding_strategy]
            inner = _DATA_PARALLEL_PLACEMENTS[ddp_config.data_parallel_sharding_strategy]
            dense_placements = Placements(
                dp_axes=[0, 1],
                parameter=[outer.parameter, inner.parameter],
                gradient=[outer.gradient, inner.gradient],
                optimizer=[outer.optimizer, inner.optimizer],
            )
        else:
            dp_mesh = DeviceMesh.from_group(
                pg_collection.dp_cp, device_type=device_type, mesh_dim_names=("dp",)
            )
            axis = _DATA_PARALLEL_PLACEMENTS[ddp_config.data_parallel_sharding_strategy]
            dense_placements = Placements(
                dp_axes=[0],
                parameter=[axis.parameter],
                gradient=[axis.gradient],
                optimizer=[axis.optimizer],
            )
        expert_dp_mesh, expert_placements = _build_expert_mesh_and_placements(
            config, ddp_config, pg_collection, device_type
        )
        dense_grad_scale, expert_grad_scale = _get_grad_scales(config, dp_mesh, expert_dp_mesh)

        # NCCL symmetric memory requires UB. MFSDP v2 intentionally does not support UB
        # without symmetric memory: it uses ncclCommRegister rather than the more performant
        # ncclCommWindowRegister:
        # https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/bufferreg.html#window-registration
        prefetch_size = ddp_config.suggested_communication_unit_size
        if config.pipeline_model_parallel_size > 1 or config.overlap_moe_expert_parallel_comm:
            prefetch_size = 0
        schedule_policy = SchedulePolicy(
            forward_prefetch_size=prefetch_size,
            backward_prefetch_size=prefetch_size,
            register_hooks=not config.overlap_moe_expert_parallel_comm,
        )
        common_fully_shard_kwargs = dict(
            mixed_precision_policy=self.mp_policy,
            schedule_policy=schedule_policy,
            shard_layout=(
                ShardLayout.PADDED
                if ddp_config.megatron_fsdp_padded_shards
                else ShardLayout.UNPADDED
            ),
        )
        # Join the caller's ambient context when one is active (VPP chunks); otherwise
        # open and finalize our own. MCore's finalize_model_grads() owns the wait
        # after the schedule completes, including delayed wgrad and custom 1F1B work.
        active_context = current_fully_shard_context()
        if active_context is None:
            construction_context = fully_shard_context(
                FsdpContext(
                    device=device,
                    runtime_policy=RuntimePolicy(
                        use_symmetric_memory=ddp_config.nccl_ub,
                        caller_managed_grad_sync=True,
                        overlap_outer_communication=ddp_config.overlap_outer_communication,
                    ),
                )
            )
        else:
            if not active_context.caller_managed_grad_sync:
                raise ValueError(
                    "MCore MFSDP v2 requires shared contexts to be constructed with "
                    "RuntimePolicy(caller_managed_grad_sync=True)."
                )
            construction_context = nullcontext(active_context)
        with construction_context:
            collector = FsdpUnitCollector(tuple(fsdp_unit_modules))
            for submodule in collector.collect(module):
                is_expert = isinstance(submodule, (TEGroupedMLP, SequentialMLP))
                if config.init_model_with_meta_device:
                    _materialize_owned_meta_modules(submodule, device)
                fully_shard(
                    submodule,
                    mesh=expert_dp_mesh if is_expert else dp_mesh,
                    placements=expert_placements if is_expert else dense_placements,
                    grad_scale=expert_grad_scale if is_expert else dense_grad_scale,
                    **common_fully_shard_kwargs,
                )

        super().__init__(config=config, module=module)

        if config.overlap_moe_expert_parallel_comm:
            register_combined_1f1b_hooks(self.module)

    def load_state_dict(self, state_dict, strict=True):
        """Load main weights and refresh the buffers used for forward compute."""
        super().load_state_dict(state_dict, strict=strict)
        # DCP writes into FP32 main-weight DTensors. BF16 and quantized compute
        # buffers have separate storage and must be refreshed before the first
        # resumed forward, just as they are after every optimizer step.
        sync_model_weights_from_main_weights(self.module.parameters())

    @staticmethod
    def _validate_config(
        config: TransformerConfig,
        ddp_config: DistributedDataParallelConfig,
        module: torch.nn.Module,
        pg_collection: ProcessGroupCollection,
        disable_bucketing: bool,
    ) -> None:
        """Validate that the model and configuration are supported by MFSDP v2.

        Args:
            config: Transformer configuration describing the requested model topology.
            ddp_config: Data-parallel and sharding configuration to validate.
            module: Model whose parameters are checked for expert parallelism.
            pg_collection: Materialized process groups whose topology must match the
                supported MFSDP v2 topology.
            disable_bucketing: Whether parameter bucketing is disabled.

        Raises:
            ValueError: If a required process group is missing or the model,
                topology, or data-parallel configuration uses an unsupported feature.
        """
        if config.moe_virtual_expert_load_balance:
            raise ValueError(
                "MoonEP is not yet integrated with Megatron-FSDP v2. Supporting it requires "
                "rebinding gathered expert weights after unshard, providing FSDP-managed "
                "full-gradient buffers for native and replica contributions, and completing "
                "replica-gradient accumulation before FSDP packs/reduce-scatters gradients "
                "or releases their buffers."
            )
        if disable_bucketing:
            raise ValueError("MFSDP v2 does not support disabling bucketing.")
        if not hasattr(pg_collection, 'dp_cp'):
            raise ValueError("MFSDP v2 requires an explicit dp_cp process group.")

        if "residual_stream" in (config.recompute_modules or []):
            # Stack-managed replay can run before the owning layer gathers its weights.
            raise ValueError(
                "MFSDP v2 does not support residual_stream recomputation (#7777). "
                "Remove it from recompute_modules and clear residual_stream_recompute_num_layers."
            )

        if config.mtp_detach_heads:
            # Detached MTP heads are tagged into a separate gradient-norm group, which
            # the shared helpers reduce with get_grad_norm_fp32. That path derives a
            # data-parallel group from the gradients' DTensor mesh and reduces over it
            # in addition to the grad-stats group, double-counting for MFSDP v2 whose
            # mesh already is the grad-stats group. Reject it rather than report a
            # silently inflated norm.
            raise ValueError("MFSDP v2 does not currently support mtp_detach_heads.")

        # Context parallelism is absent on purpose: the mesh is built from dp_cp, which
        # already folds CP ranks into the axis this shards and reduces gradients over.
        # Pipeline parallelism is supported with MFSDP v2 as long as a single shared
        # FsdpContext is opened across all VPP model chunks (see
        # wrap_model_chunks_with_ddp / dist_utils._ddp_wrap). The data-parallel mesh is
        # built solely from dp_cp, so DP sharding is unaffected by how layers are split
        # across pipeline stages or virtual stages.
        unsupported_parallelisms = ["tensor_model_parallel_size"]
        if any(getattr(config, parallelism) != 1 for parallelism in unsupported_parallelisms):
            raise ValueError(
                "MFSDP v2 does not currently support: "
                + ", ".join(
                    f"{parallelism}={getattr(config, parallelism)}"
                    for parallelism in unsupported_parallelisms
                )
            )

        # The config validates the requested topology, while these checks validate the
        # materialized topology supplied by the caller's process-group collection.
        for group_name in ("tp",):
            group = getattr(pg_collection, group_name, None)
            if group is not None and group.size() != 1:
                raise ValueError(
                    f"MFSDP v2 currently requires {group_name.upper()} process-group size 1, "
                    f"got {group.size()}."
                )

        if config.expert_model_parallel_size > 1:
            if (
                pg_collection.ep is None
                or pg_collection.ep.size() != config.expert_model_parallel_size
            ):
                actual_ep_size = None if pg_collection.ep is None else pg_collection.ep.size()
                raise ValueError(
                    "MFSDP v2 requires an EP process group matching "
                    f"expert_model_parallel_size={config.expert_model_parallel_size}, "
                    f"got {actual_ep_size}."
                )
            if pg_collection.expt_dp is None:
                raise ValueError("MFSDP v2 with EP requires an explicit expert-DP process group.")
            if not any(isinstance(submodule, MoELayer) for submodule in module.modules()):
                raise ValueError("MFSDP v2 with EP requires MoE transformer layers.")
        if (
            ddp_config.outer_dp_sharding_strategy != "no_shard"
            and ddp_config.num_distributed_optimizer_instances <= 1
        ):
            # Without a second instance there is no outer axis for the strategy to apply
            # to, so honouring it is impossible and ignoring it would be silent.
            raise ValueError(
                "MFSDP v2 outer_dp_sharding_strategy="
                f"{ddp_config.outer_dp_sharding_strategy!r} requires an outer DP axis, "
                "i.e. num_distributed_optimizer_instances > 1."
            )
        if (
            ddp_config.expert_outer_dp_sharding_strategy != "no_shard"
            and ddp_config.num_distributed_optimizer_instances <= 1
        ):
            raise ValueError(
                "MFSDP v2 expert_outer_dp_sharding_strategy="
                f"{ddp_config.expert_outer_dp_sharding_strategy!r} requires "
                "num_distributed_optimizer_instances > 1."
            )
        if config.gradient_accumulation_fusion:
            raise ValueError("MFSDP v2 does not currently support gradient accumulation fusion.")
        if config.calculate_per_token_loss and (
            config.pipeline_model_parallel_size != 1
            or ddp_config.num_distributed_optimizer_instances != 1
        ):
            raise ValueError(
                "MFSDP v2 per-token loss has not been tested with PP>1 "
                "or multiple optimizer instances."
            )
        if config.fp4 or ddp_config.fp4_param_gather:
            raise ValueError("MFSDP v2 does not currently support FP4.")
        if config.fp8 and config.fp8_recipe != "mxfp8":
            raise ValueError("MFSDP v2 only supports FP8 with the MXFP8 recipe.")
        if config.fp8_param != ddp_config.fp8_param_gather:
            raise ValueError("MFSDP v2 requires fp8_param and fp8_param_gather to match.")

        if ddp_config.fsdp_db_use_persist_buf_on_alloc_fail:
            raise ValueError(
                "MFSDP v2 does not support fsdp_db_use_persist_buf_on_alloc_fail: "
                "it allocates communication buffers from PyTorch memory pools."
            )
        if ddp_config.nccl_ub and ddp_config.disable_symmetric_registration:
            raise ValueError("MFSDP v2 requires symmetric registration when nccl_ub is enabled.")
        if ddp_config.nccl_ub and not ddp_config.megatron_fsdp_padded_shards:
            raise ValueError(
                "MFSDP v2 megatron_fsdp_padded_shards=False is incompatible with nccl_ub: "
                "symmetric-memory collectives require equal shard sizes."
            )
        if ddp_config.fsdp_manual_registration:
            raise ValueError("MFSDP v2 does not support fsdp_manual_registration.")
        if ddp_config.num_buckets is not None:
            raise ValueError("MFSDP v2 does not support num_buckets.")
        if ddp_config.megatron_fsdp_use_decoupled_grad:
            raise ValueError("MFSDP v2 does not support megatron_fsdp_use_decoupled_grad.")
        if ddp_config.megatron_fsdp_enable_fine_grained_param_gather:
            raise ValueError(
                "MFSDP v2 does not support megatron_fsdp_enable_fine_grained_param_gather."
            )
        if ddp_config.megatron_fsdp_max_pool_double_buffer:
            raise ValueError("MFSDP v2 does not support megatron_fsdp_max_pool_double_buffer.")

    @contextlib.contextmanager
    def no_sync(self):
        """Suppress gradient finalization for a non-final microbatch.

        HSDP/HFSDP leave the DP-outer axis Partial across microbatches and reduce it
        on the last backward of a step, so MFSDP has to be told which backward that
        is. Without it every backward finalizes that axis and marks the accumulation
        buffer stale, so the next microbatch zeroes it and only the last microbatch's
        gradient reaches the optimizer.

        MCore's schedules wrap every microbatch but the last in ``no_sync_func``.
        """
        with microbatch(self.module.context, is_last=False):
            yield

    @torch.no_grad()
    def scale_gradients(self, scaling_factor: float | torch.Tensor) -> None:
        """Scale finalized gradient shards before clipping and the optimizer step.

        Args:
            scaling_factor: Multiplier, usually the reciprocal global token count.
                Accepts a Python scalar or a scalar tensor from finalize_model_grads().
        """
        grads = []
        for parameter in self.parameters():
            if parameter.grad is not None:
                assert not any(
                    placement.is_partial() for placement in parameter.grad.placements
                ), "MFSDP v2 gradient scaling requires finalized gradients."
                grads.append(parameter.grad.to_local())
        if grads:
            torch._foreach_mul_(grads, scaling_factor)

    def start_param_sync(self, *unused, **unused_kwargs) -> None:
        """No-op: MFSDP v2 gathers parameters from its forward pre-hooks."""

    def start_grad_sync(self, *unused, **unused_kwargs) -> None:
        """MFSDP v2 reduces gradients during backward."""

    def finish_grad_sync(self, *unused, **unused_kwargs) -> None:
        """Wait on all submitted reductions, including delayed weight gradients."""
        self.module.context.finish_grad_sync()

    def synchronize_param_gather(self, *unused, **unused_kwargs) -> None:
        """MFSDP v2 parameter gathers complete inside module hooks."""

    def broadcast_params(self) -> None:
        """Reject parameter broadcast, which is unsupported by MFSDP v2."""
        raise NotImplementedError(
            "MFSDP v2 does not support parameter broadcast/data-parallel random initialization."
        )

    def stop_communication(self) -> None:
        """Wait for any pending gradient reductions."""
        self.finish_grad_sync()


# (parameter, gradient, optimizer) placements for a single mesh axis, per sharding
# strategy, following the table in src/docs/mfsdp_design.md#sharding-strategies. Each
# ZeRO level shards one more buffer than the last.
#
# Gradients are never replicated: an axis either still holds an unreduced contribution
# (Partial) or has reduce-scattered it (Shard). The reduce op has to be "avg" to match
# the partial gradient buffer MFSDP allocates, or a redistribute crosses two mesh axes
# at once and is rejected.
#
# Applying this per axis reproduces the named strategies: HSDP is no_shard outer over
# ZeRO-3 inner, HFSDP is optim outer over ZeRO-3 inner -- whose optimizer placement is
# then Shard on both axes, i.e. sharded across the flattened DP domain.
class _AxisPlacements(NamedTuple):
    """How one mesh axis places each of MFSDP's three buffers."""

    parameter: Placement
    gradient: Placement
    optimizer: Placement


_DATA_PARALLEL_PLACEMENTS = {
    "no_shard": _AxisPlacements(Replicate(), Partial("avg"), Replicate()),  # DDP
    "optim": _AxisPlacements(Replicate(), Partial("avg"), Shard(0)),  # ZeRO-1
    "optim_grads": _AxisPlacements(Replicate(), Shard(0), Shard(0)),  # ZeRO-2
    "optim_grads_params": _AxisPlacements(Shard(0), Shard(0), Shard(0)),  # ZeRO-3
}


def _get_grad_scales(
    config: TransformerConfig, dp_mesh: DeviceMesh, expert_dp_mesh: DeviceMesh | None
) -> tuple[float, float]:
    """Return the dense and expert gradient multipliers applied after MFSDP v2 averaging."""
    if config.calculate_per_token_loss:
        # With per-token normalization, backward uses each rank's sum of valid-token
        # losses, sᵢ, where nᵢ is the valid-token count.
        # Consider one microbatch on four ranks with EP = EDP = 2. The EP groups are
        # (0, 1) and (2, 3); expert A has replicas on ranks 0 and 2. EP backward routes
        # gradients from both source ranks' losses back to each expert replica:
        #     g₀ = ∂(s₀ + s₁)/∂θ_A,  g₂ = ∂(s₂ + s₃)/∂θ_A.
        # We need their sum, so cancel EDP averaging with grad_scale = EDP:
        #     ((g₀ + g₂) / EDP) * EDP = g₀ + g₂
        #                            = ∂(s₀ + s₁ + s₂ + s₃)/∂θ_A.
        # For dense parameters θ_dense, let gᵢ = ∂sᵢ/∂θ_dense. Here dp_mesh.size() = 4.
        # MFSDP averages over all four ranks, then multiplies by grad_scale = 4:
        #     ((g₀ + g₁ + g₂ + g₃) / 4) * 4 = g₀ + g₁ + g₂ + g₃
        #                                  = ∂(s₀ + s₁ + s₂ + s₃)/∂θ_dense.
        # After accumulation, finalize_model_grads() divides gradients by N, the valid-token
        # count across all ranks and microbatches. For this single-microbatch example,
        # N = n₀ + n₁ + n₂ + n₃, so the final expert gradient is:
        #     (g₀ + g₂) / N = ∂[(s₀ + s₁ + s₂ + s₃) / N]/∂θ_A.
        dense_grad_scale = float(dp_mesh.size())
        expert_grad_scale = 1.0
        if expert_dp_mesh is not None:
            expert_grad_scale = float(expert_dp_mesh.size())
    else:
        # Without per-token normalization, backward uses each microbatch's mean
        # valid-token loss, so dense gradients need the mesh's rank average.
        #
        # For experts, consider one microbatch on four ranks with EP = EDP = 2.
        # The EP groups are (0, 1) and (2, 3); expert A has replicas on ranks 0 and 2.
        # Let sᵢ be the sum of valid-token losses and nᵢ the valid-token count on
        # source rank i. Its mean loss is sᵢ / nᵢ. EP backward routes gradients back
        # to the experts, so each replica includes BOTH source ranks' mean losses:
        #     g₀ = ∂(s₀/n₀ + s₁/n₁)/∂θ_A,  g₂ = ∂(s₂/n₂ + s₃/n₃)/∂θ_A.
        # EDP averaging followed by grad_scale = 1 / EP gives:
        #     ((g₀ + g₂) / EDP) * (1 / EP) = (g₀ + g₂) / 4
        #                                  = ∂[(s₀/n₀ + s₁/n₁ + s₂/n₂ + s₃/n₃) / 4]/∂θ_A.
        # The same calculation applies to the experts on ranks 1 and 3.
        # There are EP * EDP source ranks, but EDP averaging divides by only EDP.
        # Multiply by 1 / EP as well to obtain the mean gradient over all source ranks.
        # When all nᵢ are equal, this equals:
        #     ∂[(s₀ + s₁ + s₂ + s₃) / (n₀ + n₁ + n₂ + n₃)]/∂θ_A,
        # matching the per-token branch above.
        dense_grad_scale = 1.0
        expert_grad_scale = 1.0 / config.expert_model_parallel_size
    return dense_grad_scale, expert_grad_scale


def _build_expert_mesh_and_placements(
    config: TransformerConfig,
    ddp_config: DistributedDataParallelConfig,
    pg_collection: ProcessGroupCollection,
    device_type: str,
) -> Tuple[DeviceMesh | None, Placements | None]:
    """Build expert placements for MoE models, including when EP has only one rank."""
    if config.num_moe_experts is None:
        return None, None

    inner_strategy = get_sharding_strategy(ddp_config, is_expert_param=True)

    if ddp_config.num_distributed_optimizer_instances > 1:
        # Match v1 topology: dense and expert parameters share the outer DP axis,
        # while experts use the existing expert-DP inner group. Only placements differ.
        dp_mesh = _build_hybrid_dp_mesh(
            pg_collection.inter_dist_opt, pg_collection.intra_expt_dp, device_type
        )
        inner = _DATA_PARALLEL_PLACEMENTS[inner_strategy]
        outer = _DATA_PARALLEL_PLACEMENTS[ddp_config.expert_outer_dp_sharding_strategy]
        placements = Placements(
            dp_axes=[0, 1],
            parameter=[outer.parameter, inner.parameter],
            gradient=[outer.gradient, inner.gradient],
            optimizer=[outer.optimizer, inner.optimizer],
        )
        return dp_mesh, placements

    dp_mesh = DeviceMesh.from_group(
        pg_collection.expt_dp, device_type=device_type, mesh_dim_names=("expert_dp",)
    )
    axis = _DATA_PARALLEL_PLACEMENTS[inner_strategy]
    placements = Placements(
        dp_axes=[0],
        parameter=[axis.parameter],
        gradient=[axis.gradient],
        optimizer=[axis.optimizer],
    )
    return dp_mesh, placements


def _build_hybrid_dp_mesh(outer_group, inner_group, device_type):
    """Build the ("dp_outer", "dp_shard") mesh for a hybrid data-parallel domain."""
    if outer_group is None or inner_group is None:
        raise ValueError(
            "MFSDP v2 hybrid sharding requires inter- and intra-instance process groups."
        )

    rank = dist.get_rank()
    inner_ranks = dist.get_process_group_ranks(inner_group)
    outer_ranks = dist.get_process_group_ranks(outer_group)
    outer_offsets = torch.tensor(outer_ranks) - rank
    inner_offsets = torch.tensor(inner_ranks) - rank
    layout = rank + outer_offsets[:, None] + inner_offsets[None, :]

    # Sort the mesh ranks into outer-to-inner order, then check that this rank's
    # row and column match its inner and outer process groups, respectively.
    expected_layout = layout.flatten().sort().values.reshape(layout.shape)
    expected_inner = expected_layout[outer_group.rank()].tolist()
    expected_outer = expected_layout[:, inner_group.rank()].tolist()
    if inner_ranks != expected_inner:
        raise ValueError(
            f"MFSDP v2 hybrid mesh row {expected_inner} does not match the intra "
            f"data-parallel group {inner_ranks}."
        )
    if outer_ranks != expected_outer:
        raise ValueError(
            f"MFSDP v2 hybrid mesh column {expected_outer} does not match the inter "
            f"distributed-optimizer group {outer_ranks}."
        )

    return DeviceMesh.from_group(
        [outer_group, inner_group],
        device_type=device_type,
        mesh=layout,
        mesh_dim_names=("dp_outer", "dp_shard"),
    )
