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

import logging
import random
from contextlib import contextmanager
from functools import partial
from typing import Dict, List, NamedTuple, Optional, Tuple, Type

__all__ = ["FullyShardedDataParallel"]

try:
    import einops

    HAVE_EINOPS = True
except ImportError:
    HAVE_EINOPS = False

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed import DeviceMesh
from torch.distributed.tensor import Partial, Replicate, Shard
from torch.distributed.tensor.placement_types import Placement

from megatron.core import parallel_state, tensor_parallel
from megatron.core.config_logger import has_config_logger_enabled, log_config_to_disk
from megatron.core.distributed.data_parallel_base import _BaseDataParallel
from megatron.core.distributed.distributed_data_parallel_config import DistributedDataParallelConfig
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.mamba_layer import MambaLayer
from megatron.core.transformer.moe.experts import TEGroupedMLP
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import MoETransformerLayer, TransformerLayer
from megatron.core.utils import is_te_min_version, log_single_rank

try:
    from megatron.core.distributed.fsdp.src.megatron_fsdp import (
        FSDPDistributedIndex,
        MegatronFSDP,
        MixedPrecisionPolicy,
    )
    from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
        Placements,
        fully_shard,
        fully_shard_context,
    )
    from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.module import (
        FsdpContext,
        FsdpModule,
    )

    HAVE_MEGATRON_FSDP = True
except ImportError as import_megatron_fsdp_error:
    IMPORT_MEGATRON_FSDP_ERROR = import_megatron_fsdp_error
    HAVE_MEGATRON_FSDP = False

logger = logging.getLogger(__name__)


class FullyShardedDataParallelV1(_BaseDataParallel):
    """
    Fully Sharded Data Parallel (FSDP) wrapper for the Megatron model.
    """

    # Module type registry (forked from Megatron-Bridge param_mapping utilities).
    _MODULE_TYPE_REGISTRY: Dict[str, set] = {
        "column": {
            "ColumnParallelLinear",
            "TEColumnParallelLinear",
            "TELayerNormColumnParallelLinear",
            "TEColumnParallelGroupedLinear",
            "VocabParallelEmbedding",
            "DotProductAttention",  # for attention sink only
            "TEDotProductAttention",  # for attention sink only
        },
        "row": {"RowParallelLinear", "TERowParallelLinear", "TERowParallelGroupedLinear"},
        "replicated": {
            # Normalization layers
            "TENorm",
            "FusedLayerNorm",
            "WrappedTorchNorm",
            "LayerNorm",
            "RMSNorm",
            "L2Norm",
            # Other non-parallel modules
            "IdentityOp",
            "TopKRouter",
        },
    }

    @staticmethod
    def _fine_grained_recurse_module_types(
        config: TransformerConfig, ddp_config: DistributedDataParallelConfig
    ) -> Tuple[Type[nn.Module], ...]:
        """Module classes needing ``parameters(recurse=True)`` for fine-grained hooks."""
        if (
            config.overlap_moe_expert_parallel_comm
            and ddp_config.data_parallel_sharding_strategy == "optim_grads_params"
        ):
            # Lazy import to avoid circular chain.
            from megatron.core.transformer.moe.experts import TEGroupedMLP
            from megatron.core.transformer.moe.shared_experts import SharedExpertMLP

            return (TEGroupedMLP, SharedExpertMLP)
        return ()

    def __init__(
        self,
        config: TransformerConfig,
        ddp_config: DistributedDataParallelConfig,
        module: torch.nn.Module,
        # This should be named fsdp_unit_module_types; the v1 name is retained for API
        # compatibility.
        fsdp_unit_modules: Optional[List[Type[torch.nn.Module]]] = None,
        disable_bucketing: bool = False,
        device: Optional[torch.device] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        if not HAVE_MEGATRON_FSDP:
            raise IMPORT_MEGATRON_FSDP_ERROR

        if has_config_logger_enabled(config):
            log_config_to_disk(config, locals(), prefix=type(self).__name__)

        self.num_moe_experts = getattr(config, "num_moe_experts", None)

        self.ddp_config = ddp_config
        log_single_rank(
            logger,
            logging.INFO,
            f'Setting up DistributedDataParallel with config {self.ddp_config}',
        )
        self.mp_policy = MixedPrecisionPolicy(
            main_params_dtype=ddp_config.megatron_fsdp_main_params_dtype,
            main_grads_dtype=ddp_config.megatron_fsdp_main_grads_dtype,
            grad_comm_dtype=ddp_config.megatron_fsdp_grad_comm_dtype,
        )
        log_single_rank(
            logger,
            logging.INFO,
            f'Setting up Megatron-FSDP MixedPrecisionPolicy with config {self.mp_policy}',
        )

        self.megatron_fsdp_dist_index = self._init_dist_index(pg_collection)

        if config.gradient_accumulation_fusion:
            assert is_te_min_version("2.10"), (
                "Megatron-FSDP with gradient_accumulation_fusion requires "
                "Transformer Engine version 2.10 or higher."
            )

        self.bucket_size = self.ddp_config.bucket_size
        if disable_bucketing:
            self.bucket_size = None
        self.device = device if device else torch.device(f'cuda:{torch.cuda.current_device()}')

        if fsdp_unit_modules is not None:
            self.fsdp_unit_modules = fsdp_unit_modules
        else:
            # FSDP unit modules control the granularity of FSDP communications.
            # "optim": Reduce-scatter communication groups on the final microbatch.
            # "optim_grads": Additionally, RS communication groups on all microbatches.
            # "optim_grads_params": RS & AG communication groups on all microbatches.
            if self.ddp_config.data_parallel_sharding_strategy != "no_shard":
                self.fsdp_unit_modules = [TransformerLayer, MoETransformerLayer, MambaLayer]
            else:
                self.fsdp_unit_modules = []

        self._annotate_tensor_parallelism(module)

        if config.overlap_moe_expert_parallel_comm:
            assert not ddp_config.fsdp_double_buffer, (
                "1F1B overlap with FSDP does not support double buffer. "
                "Please set fsdp_double_buffer=False in the ddp config."
            )
            assert config.cuda_graph_impl in ("none", "full_iteration"), (
                "1F1B overlap with FSDP does not support per-layer CUDA graphs "
                f"(cuda_graph_impl={config.cuda_graph_impl!r}). "
                "Use cuda_graph_impl='full_iteration' or disable CUDA graphs "
                "(cuda_graph_impl='none')."
            )

        if (
            config.overlap_moe_expert_parallel_comm
            and ddp_config.data_parallel_sharding_strategy == "optim_grads_params"
        ):
            supported_fsdp_unit_modules = [TransformerLayer, MoETransformerLayer, MambaLayer]
            assert self.fsdp_unit_modules and all(
                module in supported_fsdp_unit_modules for module in self.fsdp_unit_modules
            ), (
                "EP overlap with FSDP currently requires fsdp_unit_modules "
                "to contain only supported MCore modules "
                f"{supported_fsdp_unit_modules}, "
                f"got {self.fsdp_unit_modules}."
            )
        super().__init__(
            config=config,
            module=MegatronFSDP(
                ddp_config=ddp_config,
                mixed_precision_policy=self.mp_policy,
                module=module,
                fsdp_unit_modules=self.fsdp_unit_modules,
                disable_bucketing=disable_bucketing,
                device=self.device,
                dist_index=self.megatron_fsdp_dist_index,
                calculate_per_token_loss=config.calculate_per_token_loss,
                init_model_with_meta_device=config.init_model_with_meta_device,
                # EP overlap schedule calls sub-modules directly instead of
                # TransformerLayer.forward(), so fine-grained hooks are needed
                # to manage _training_state and all-gather each sub-module's
                # parameters individually.  This applies to all sharding
                # strategies (not only optim_grads_params) because the hooks
                # also maintain per-module training-state bookkeeping that the
                # gradient-reduction pipeline relies on.
                enable_fine_grained_param_gather_hook=(
                    (config.fp8_recipe == "mxfp8" and ddp_config.fp8_param_gather)
                    or config.overlap_moe_expert_parallel_comm
                    or self.ddp_config.megatron_fsdp_enable_fine_grained_param_gather
                ),
                enable_fine_grained_param_gather_backward_hook=(
                    config.overlap_moe_expert_parallel_comm
                    and ddp_config.data_parallel_sharding_strategy == "optim_grads_params"
                ),
                fine_grained_recurse_module_types=self._fine_grained_recurse_module_types(
                    config, ddp_config
                ),
            ),
        )
        self.param_and_grad_buffer = self.module.param_and_grad_buffer
        self.no_sync = self.module.no_sync
        self.start_param_sync = self.module.start_param_sync
        self.start_grad_sync = self.module.start_grad_sync
        self.finish_grad_sync = self.module.finish_grad_sync
        self.scale_gradients = self.module.scale_gradients
        self.zero_grad_buffer = self.module.zero_grad_buffer
        self.broadcast_params = self.module.broadcast_params
        self.synchronize_param_gather = self.module.synchronize_param_gather
        self.module.state_dict_for_save_checkpoint = self.module.state_dict
        self.state_dict_for_save_checkpoint = self.state_dict
        self.module.config = config

        self.sync_rng_states_across_tp_group()

    def load_state_dict(self, state_dict, strict=True):
        """
        Load the state dictionary into the module.
        """
        custom_state_dict = {}
        for key, value in state_dict.items():
            if self.config.fp8 and key.endswith('._extra_state'):
                # Skip extra state keys
                continue
            custom_state_dict[f"module.{key}"] = value

        if self.config.fp8 or self.config.gated_linear_unit:
            strict = False
            log_single_rank(
                logger,
                logging.WARNING,
                "Loading state_dict with strict=False due to fp8 configuration. "
                "This is expected as some keys may not match exactly.",
            )

        self.module.load_state_dict(custom_state_dict, strict=strict)

    def _detect_parallelism_type(self, param_name: str, module: nn.Module) -> Optional[str]:
        """
        Infer tensor-parallelism type for a parameter under a given module
        (forked from Megatron-Bridge).

        Returns:
            "column", "row", or "replicated" if a type can be inferred, else None.
        """
        module_type = type(module).__name__

        # Handle fused modules like TELayerNormColumnParallelLinear
        # These modules have both column-parallel weights (weight, bias)
        # and replicated layer norm weights (layer_norm_weight, layer_norm_bias)
        if module_type == "TELayerNormColumnParallelLinear":
            # Check the actual parameter name to determine the correct parallelism type
            if param_name.endswith("layer_norm_weight") or param_name.endswith("layer_norm_bias"):
                return "replicated"
            # All other parameters (weight, bias) are column-parallel
            return "column"

        # Check registry first
        for parallelism, types in self._MODULE_TYPE_REGISTRY.items():
            if module_type in types:
                if parallelism == "row" and "bias" in param_name:
                    return "replicated"
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
                if "bias" in param_name:
                    return "replicated"
                return "row"

        # Fallback for normalization layers
        if any(norm in module_type for norm in ["Norm", "Normalization"]):
            return "replicated"

        # Check parallel_mode for TELinear
        if module_type == "TELinear":
            if module.parallel_mode == "column":
                return "column"
            elif module.parallel_mode == "row":
                if "bias" in param_name:
                    return "replicated"
                return "row"
            else:
                return "replicated"

        return None

    def _annotate_tensor_parallelism(self, root_module: nn.Module) -> None:
        """Annotate parameters under root_module with inferred tensor-parallel metadata.

        Each parameter that can be classified will get a `_tensor_parallel_mode` attribute
        set to one of: "column", "row", or "replicated".
        """
        for submodule in root_module.modules():
            for name, param in submodule.named_parameters(recurse=False):
                detected_type = self._detect_parallelism_type(name, submodule)
                if detected_type is not None:
                    setattr(param, "_tensor_parallel_mode", detected_type)

    def _init_dist_index(self, pg_collection):
        """
        Initialize the distributed index for the module.
        """
        enable_hsdp = self.ddp_config.num_distributed_optimizer_instances > 1
        if pg_collection is None:
            tp_group = parallel_state.get_tensor_model_parallel_group()
            expt_tp_group = parallel_state.get_expert_tensor_parallel_group()
            if enable_hsdp:
                dp_cp_group = parallel_state.get_data_parallel_group(
                    with_context_parallel=True, partial_data_parallel=True
                )
                outer_fsdp_group = parallel_state.get_inter_distributed_optimizer_instance_group()
                hybrid_fsdp_group = parallel_state.get_data_parallel_group(
                    with_context_parallel=True, partial_data_parallel=False
                )
                expt_dp_group = parallel_state.get_expert_data_parallel_group(
                    partial_expert_data_parallel=True
                )
                hybrid_fsdp_expt_group = parallel_state.get_expert_data_parallel_group(
                    partial_expert_data_parallel=False
                )
                ep_group = parallel_state.get_expert_model_parallel_group()
            else:
                dp_cp_group = parallel_state.get_data_parallel_group(
                    with_context_parallel=True, partial_data_parallel=False
                )
                outer_fsdp_group = None
                hybrid_fsdp_group = None
                expt_dp_group = parallel_state.get_expert_data_parallel_group()
                ep_group = parallel_state.get_expert_model_parallel_group()
        else:
            tp_group = getattr(pg_collection, 'tp', None)
            expt_tp_group = getattr(pg_collection, 'expt_tp', None)
            if enable_hsdp:
                dp_cp_group = pg_collection.intra_dp_cp
                outer_fsdp_group = pg_collection.inter_dist_opt
                hybrid_fsdp_group = pg_collection.dp_cp
                # This has not been tested yet.
                expt_dp_group = getattr(pg_collection, 'intra_expt_dp', None)
                hybrid_fsdp_expt_group = getattr(pg_collection, 'expt_dp', None)
                ep_group = getattr(pg_collection, 'ep', None)
            else:
                dp_cp_group = pg_collection.dp_cp
                outer_fsdp_group = None
                hybrid_fsdp_group = None
                expt_dp_group = getattr(pg_collection, 'expt_dp', None)
                ep_group = getattr(pg_collection, 'ep', None)

        if tp_group is None:
            single_rank_group = dist.new_group(ranks=[dist.get_rank()])
            tp_group = single_rank_group

        if expt_tp_group is None:
            single_rank_group = dist.new_group(ranks=[dist.get_rank()])
            expt_tp_group = single_rank_group

        # Extract AG groups from pg_collection for explicit passing
        dp_cp_ag = getattr(pg_collection, 'dp_cp_ag', None) if pg_collection is not None else None
        expt_dp_ag = (
            getattr(pg_collection, 'expt_dp_ag', None) if pg_collection is not None else None
        )

        if enable_hsdp:
            if self.num_moe_experts is not None:
                expt_mesh = _get_hsdp_tp_mesh(
                    outer_fsdp_group, expt_dp_group, expt_tp_group, ep_size=ep_group.size()
                )
                expt_device_mesh = DeviceMesh.from_group(
                    [outer_fsdp_group, expt_dp_group, expt_tp_group],
                    device_type="cuda",
                    mesh=expt_mesh.tolist(),
                    mesh_dim_names=["outer_fsdp_dp", "dp_cp", "tp"],
                )
            else:
                expt_device_mesh = None
            mesh = _get_hsdp_tp_mesh(outer_fsdp_group, dp_cp_group, tp_group)
            dist_index = FSDPDistributedIndex(
                hsdp_outer_dp_shard=self.ddp_config.outer_dp_sharding_strategy != "no_shard",
                device_mesh=DeviceMesh.from_group(
                    [outer_fsdp_group, dp_cp_group, tp_group],
                    device_type="cuda",
                    mesh=mesh.tolist(),
                    mesh_dim_names=["outer_fsdp_dp", "dp_cp", "tp"],
                ),
                dp_outer_dim="outer_fsdp_dp",  # Use Hybrid FSDP!
                dp_shard_dim="dp_cp",
                tp_dim="tp",
                hybrid_fsdp_group=hybrid_fsdp_group,
                hybrid_fsdp_expt_group=hybrid_fsdp_expt_group,
                expt_device_mesh=expt_device_mesh,
                fsdp_group_ag=dp_cp_ag,
                expt_fsdp_group_ag=expt_dp_ag,
            )
        else:
            if self.num_moe_experts is not None:
                expt_mesh = _get_dp_tp_mesh(expt_dp_group, expt_tp_group, ep_size=ep_group.size())
                expt_device_mesh = DeviceMesh.from_group(
                    [expt_dp_group, expt_tp_group],
                    device_type="cuda",
                    mesh=expt_mesh.tolist(),
                    mesh_dim_names=["dp_cp", "tp"],
                )
            else:
                expt_device_mesh = None

            mesh = _get_dp_tp_mesh(dp_cp_group, tp_group)
            dist_index = FSDPDistributedIndex(
                device_mesh=DeviceMesh.from_group(
                    [dp_cp_group, tp_group],
                    device_type="cuda",
                    mesh=mesh.tolist(),
                    mesh_dim_names=["dp_cp", "tp"],
                ),
                dp_shard_dim="dp_cp",
                tp_dim="tp",
                expt_device_mesh=expt_device_mesh,
                fsdp_group_ag=dp_cp_ag,
                expt_fsdp_group_ag=expt_dp_ag,
            )

        self.tp_group = tp_group

        return dist_index

    def stop_communication(self):
        """
        Stop communication for the module.
        """
        self.module.synchronize_gradient_reduce()
        self.module.synchronize_param_gather()

    def sync_rng_states_across_tp_group(self):
        """
        Synchronize the tensor parallel random number generator states.
        """
        if self.tp_group.size() <= 1:
            return

        if self.tp_group.rank() == 0:
            broadcast_list = [_get_rng_state_dict()]
        else:
            broadcast_list = [None]
        torch.distributed.broadcast_object_list(broadcast_list, group=self.tp_group, group_src=0)
        _load_rng_state_dict(broadcast_list[0])


class FullyShardedDataParallelV2(_BaseDataParallel):
    """MFSDP v2 wrapper for the Megatron model."""

    @property
    def context(self) -> "FsdpContext":
        """Return the runtime context shared by this model chunk."""
        return self.module.context

    @staticmethod
    def _configure_te_grouped_mlp_wgrad_fusion(
        module: torch.nn.Module, enabled: bool
    ) -> None:
        """Restrict fused wgrad accumulation to routed TE grouped experts.

        ``gradient_accumulation_fusion`` is consumed while the model is built, so
        every compatible linear starts with fusion enabled. MFSDP's parameter-group
        setting must match those linear-module flags: first disable fusion throughout
        the constructed model, then opt only ``TEGroupedMLP``'s grouped FC1/FC2 back in.
        """
        if not enabled:
            return

        for submodule in module.modules():
            if hasattr(submodule, "fuse_wgrad_accumulation"):
                submodule.fuse_wgrad_accumulation = False
            if hasattr(submodule, "gradient_accumulation_fusion"):
                submodule.gradient_accumulation_fusion = False

        for submodule in module.modules():
            if isinstance(submodule, TEGroupedMLP):
                submodule.linear_fc1.fuse_wgrad_accumulation = True
                submodule.linear_fc2.fuse_wgrad_accumulation = True

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
                unspecified, transformer, MoE transformer, and Mamba layers are used.
            disable_bucketing: Compatibility argument accepted from the common data-parallel
                wrapping path. MFSDP v2 manages parameter groups independently and ignores it.
            device: Device whose type is used to construct the data-parallel mesh.
                Defaults to CUDA.
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
        FullyShardedDataParallelV2._validate_config(config, ddp_config, module, pg_collection)

        if has_config_logger_enabled(config):
            log_config_to_disk(config, locals(), prefix=type(self).__name__)

        # Optimizer construction reads this attribute; retain the v1 contract for compatibility.
        self.ddp_config = ddp_config

        if fsdp_unit_modules is None:
            fsdp_unit_modules = [TransformerLayer, MoETransformerLayer, MambaLayer]
        self._configure_te_grouped_mlp_wgrad_fusion(
            module, enabled=config.gradient_accumulation_fusion
        )

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

        # Expert parameters use a single mesh over the whole expert-DP domain and never
        # take an outer axis. Dense parameters are the ones that go hybrid below.
        expert_dp_mesh = None
        if config.expert_model_parallel_size > 1:
            expert_dp_mesh = DeviceMesh.from_group(
                pg_collection.expt_dp, device_type=device_type, mesh_dim_names=("expert_dp",)
            )
        expert_axis = _DATA_PARALLEL_PLACEMENTS[
            ddp_config.expert_data_parallel_sharding_strategy
            or ddp_config.data_parallel_sharding_strategy
        ]
        expert_placements = Placements(
            dp_axes=[0],
            parameter=[expert_axis.parameter],
            gradient=[expert_axis.gradient],
            optimizer=[expert_axis.optimizer],
        )
        if has_outer_dp_axis := ddp_config.num_distributed_optimizer_instances > 1:
            # Dense parameters get an outer DP axis. There is no HSDP/HFSDP special case:
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
        self.mesh = dp_mesh
        self.moe_mesh = expert_dp_mesh
        # NCCL symmetric memory requires UB. MFSDP v2 intentionally does not support UB
        # without symmetric memory: it uses ncclCommRegister rather than the more performant
        # ncclCommWindowRegister:
        # https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/bufferreg.html#window-registration
        fine_grained = config.overlap_moe_expert_parallel_comm
        skip_backward_cb = fine_grained and ddp_config.delay_wgrad_compute
        # Join an ambient multi-chunk construction scope when VPP wrapping
        # opens one; otherwise this adapter owns and finalizes its context. The
        # combined schedule uses trace-replay because VPP occurrence order does
        # not follow the static construction order.
        with fully_shard_context(
            device=device,
            reuse_existing=True,
            use_trace_replay=fine_grained,
            use_symmetric_memory=ddp_config.nccl_ub,
            enable_trace_pool=ddp_config.fsdp_trace_pool,
        ):
            if expert_dp_mesh is not None:
                # Expert parameters are replicated over expert-DP, not the full DP group.
                # Their gradients need the EP divisor because the same expert receives
                # contributions after dispatch from every EP rank.
                for submodule in module.modules():
                    if isinstance(submodule, MoELayer):
                        fully_shard(
                            submodule.experts,
                            mesh=expert_dp_mesh,
                            placements=expert_placements,
                            mixed_precision_policy=self.mp_policy,
                            fine_grained=fine_grained,
                            skip_backward_callback=skip_backward_cb,
                            fuse_wgrad_accumulation=(
                                config.gradient_accumulation_fusion
                                and isinstance(submodule.experts, TEGroupedMLP)
                            ),
                            grad_divisor=config.expert_model_parallel_size,
                        )
            for submodule in reversed(list(module.modules())):
                if submodule is module:
                    # The root is always sharded after selected child units so it is not
                    # wrapped twice when its type also appears in fsdp_unit_modules.
                    continue
                if any(isinstance(submodule, module_type) for module_type in fsdp_unit_modules):
                    fully_shard(
                        submodule,
                        mesh=dp_mesh,
                        placements=dense_placements,
                        mixed_precision_policy=self.mp_policy,
                        fine_grained=fine_grained,
                        skip_backward_callback=skip_backward_cb,
                        fuse_wgrad_accumulation=(
                            config.gradient_accumulation_fusion
                            and isinstance(submodule, TEGroupedMLP)
                        ),
                    )
                elif isinstance(submodule, TEGroupedMLP) and not isinstance(submodule, FsdpModule):
                    # Real MoE layers are sharded through their MoELayer owner above. Keep
                    # this fallback for standalone TEGroupedMLP modules without wrapping an
                    # already-owned expert module a second time.
                    if self.moe_mesh is None:
                        if pg_collection.expt_dp is None:
                            raise ValueError(
                                "MFSDP v2 MoE models require an explicit expt_dp group."
                            )
                        self.moe_mesh = DeviceMesh.from_group(
                            pg_collection.expt_dp, device_type=device_type, mesh_dim_names=("edp",)
                        )
                    fully_shard(
                        submodule,
                        mesh=self.moe_mesh,
                        placements=expert_placements,
                        mixed_precision_policy=self.mp_policy,
                        fine_grained=fine_grained,
                        skip_backward_callback=skip_backward_cb,
                        fuse_wgrad_accumulation=(
                            config.gradient_accumulation_fusion
                            and isinstance(submodule, TEGroupedMLP)
                        ),
                    )
            fully_shard(
                module,
                mesh=dp_mesh,
                placements=dense_placements,
                mixed_precision_policy=self.mp_policy,
                fine_grained=fine_grained,
                skip_backward_callback=skip_backward_cb,
                fuse_wgrad_accumulation=(
                    config.gradient_accumulation_fusion and isinstance(module, TEGroupedMLP)
                ),
            )
        super().__init__(config=config, module=module)
        if config.init_model_with_meta_device:
            self._reset_parameters_for_meta_device_init()
        if fine_grained:
            self._setup_1f1b_overlap_interface()

    def _reset_parameters_for_meta_device_init(self) -> None:
        """Reset model parameters that were initialized on the meta device.

        Meta-device init leaves parameters without values; ``fully_shard`` then
        materializes them as empty tensors. Reset each leaf module's weights on
        the full (unsharded) parameters, copy the aligned values back into the
        sharded optimizer/compute buffers, and return to the sharded resting state.
        """
        root = self.module
        fsdp_modules = [m for m in root.modules() if isinstance(m, FsdpModule)]

        # Unshard every FSDP unit so reset_parameters() writes the full weight.
        for m in fsdp_modules:
            m._unshard_parameter_groups()
        context = root.context
        context.current_stream().wait_stream(context.allgather_stream)

        # Reset the original (non-FsdpModule) leaf modules.
        for m in root.modules():
            if isinstance(m, FsdpModule):
                continue
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
            elif hasattr(m, "_reset_parameters"):
                m._reset_parameters()

        # Copy the reset full weights back into the sharded buffers, aligned
        # across DP/EDP ranks, then return to the sharded resting state.
        for m in fsdp_modules:
            for group in m._parameter_groups:
                group.sync_model_weight_from_unsharded_weight()
            m._reshard_parameter_groups(record_execution=False)

    def _setup_1f1b_overlap_interface(self) -> None:
        """Expose the parameter lifecycle callbacks used by combined 1F1B.

        All callbacks live on the adapter rather than on ``FsdpModule`` so the
        experimental module API stays schedule-agnostic; the schedule-facing
        surface is assembled here.
        """

        def _require_fsdp_module(module: torch.nn.Module) -> FsdpModule:
            if not isinstance(module, FsdpModule):
                raise TypeError(
                    "MFSDP v2 combined 1F1B callbacks require an experimental FsdpModule, "
                    f"got {type(module).__name__}."
                )
            return module

        def unshard_parameters(module: torch.nn.Module) -> None:
            """All-gather full parameter storage for compute (idempotent)."""
            module = _require_fsdp_module(module)
            module._unshard_and_prefetch("rowwise")

        def reshard_parameters(module: torch.nn.Module) -> None:
            """Release all-gathered storage and install DTensor parameters."""
            _require_fsdp_module(module)._reshard_parameter_groups()

        def reduce_grad(module: torch.nn.Module) -> None:
            """Pack gradients and launch their reduce-scatters."""
            _require_fsdp_module(module)._reduce_gradient_groups()

        def release_module(module: torch.nn.Module, *, reduce_grad: bool) -> None:
            if reduce_grad:
                _require_fsdp_module(module).post_backward()
            else:
                reshard_parameters(module)

        def _replace_param_with_raw_if_needed() -> None:
            """Initialize the root context before a fine-grained schedule runs.

            The experimental API stores raw tensors backed by DBuffer at all
            times, so no parameter swap is needed, but finalizing the context
            here ensures a child FSDP unit cannot mistake itself for the root
            when it executes first.
            """
            self.module.context.ensure_finalized()

        # The 1F1B schedule finds the FSDP wrapper via find_megatron_fsdp(),
        # which may return the bare FsdpModule (no ddp_config). Expose the
        # adapter's ddp_config on the module so the schedule can read the
        # data-parallel sharding strategy without special-casing the v2 path.
        self.module.ddp_config = self.ddp_config

        self.unshard_parameters = unshard_parameters
        self.reshard_parameters = reshard_parameters
        self.reduce_grad = reduce_grad
        self._replace_param_with_raw_if_needed = _replace_param_with_raw_if_needed
        self.post_forward_release_module = partial(release_module, reduce_grad=False)
        self.post_backward_release_module = partial(release_module, reduce_grad=True)
        self.pre_backward = partial(self.module.pre_backward, register_final_callback=False)
        self.post_backward = self.module.post_backward

    @staticmethod
    def _validate_config(
        config: TransformerConfig,
        ddp_config: DistributedDataParallelConfig,
        module: torch.nn.Module,
        pg_collection: ProcessGroupCollection,
    ) -> None:
        """Validate that the model and configuration are supported by MFSDP v2.

        Args:
            config: Transformer configuration describing the requested model topology.
            ddp_config: Data-parallel and sharding configuration to validate.
            module: Model whose parameters are checked for expert parallelism.
            pg_collection: Materialized process groups whose topology must match the
                supported MFSDP v2 topology.

        Raises:
            ValueError: If a required process group is missing or the model,
                topology, or data-parallel configuration uses an unsupported feature.
        """
        if pg_collection.dp_cp is None:
            raise ValueError("MFSDP v2 requires an explicit dp_cp process group.")

        if config.mtp_detach_heads:
            # Detached MTP heads are tagged into a separate gradient-norm group, which
            # the shared helpers reduce with get_grad_norm_fp32. That path derives a
            # data-parallel group from the gradients' DTensor mesh and reduces over it
            # in addition to the grad-stats group, double-counting for MFSDP v2 whose
            # mesh already is the grad-stats group. Reject it rather than report a
            # silently inflated norm.
            raise ValueError("MFSDP v2 does not currently support mtp_detach_heads.")

        unsupported_parallelisms = [
            "tensor_model_parallel_size",
            "context_parallel_size",
        ]
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
        for group_name in ("tp", "cp"):
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
        if ddp_config.data_parallel_sharding_strategy != "optim_grads_params":
            raise ValueError(
                "MFSDP v2 requires data_parallel_sharding_strategy='optim_grads_params'."
            )
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
        if config.gradient_accumulation_fusion:
            if not is_te_min_version("2.10"):
                raise ValueError(
                    "MFSDP v2 gradient accumulation fusion requires Transformer Engine 2.10+."
                )
            unsupported_fused_wgrad_features = []
            if config.use_transformer_engine_op_fuser:
                unsupported_fused_wgrad_features.append("Transformer Engine op fuser")
            if ddp_config.nccl_ub:
                unsupported_fused_wgrad_features.append("symmetric-memory NCCL-UB")
            if unsupported_fused_wgrad_features:
                raise ValueError(
                    "MFSDP v2 gradient accumulation fusion does not yet support: "
                    + ", ".join(unsupported_fused_wgrad_features)
                    + "."
                )
        if config.calculate_per_token_loss:
            raise ValueError("MFSDP v2 does not currently support per-token loss normalization.")
        if config.fp4 or ddp_config.fp4_param_gather:
            raise ValueError("MFSDP v2 does not currently support FP4.")
        if ddp_config.fp8_param_gather and config.fp8_recipe != "mxfp8":
            raise ValueError(
                "MFSDP v2 currently supports fp8_param_gather only with --fp8-recipe mxfp8."
            )
        # fp8 primary weights (fp8_param_gather) require fp8 mode (--fp8), whose
        # autocast context the Fp8ParameterGroup path is validated for; any
        # other fp8/fp4 usage stays rejected.
        if config.fp8 and not (ddp_config.fp8_param_gather and config.fp8_recipe == "mxfp8"):
            raise ValueError(
                "MFSDP v2 fp8 autocast is only supported together with "
                "--fp8-param-gather and --fp8-recipe mxfp8."
            )
        if config.cuda_graph_impl != "none" or ddp_config.megatron_fsdp_cuda_graph_mode:
            raise ValueError("MFSDP v2 does not currently support CUDA graphs.")

        if ddp_config.fsdp_db_use_persist_buf_on_alloc_fail:
            raise ValueError(
                "MFSDP v2 does not support fsdp_db_use_persist_buf_on_alloc_fail: "
                "it allocates communication buffers from PyTorch memory pools."
            )
        if ddp_config.nccl_ub and ddp_config.disable_symmetric_registration:
            raise ValueError("MFSDP v2 requires symmetric registration when nccl_ub is enabled.")
        if ddp_config.fsdp_trace_pool and ddp_config.nccl_ub:
            raise ValueError("MFSDP v2 trace-pool is incompatible with NCCL user buffers.")
        if ddp_config.fsdp_trace_pool and ddp_config.fsdp_double_buffer:
            raise ValueError("MFSDP v2 trace-pool is incompatible with FSDP double buffering.")
        if ddp_config.fsdp_manual_registration:
            raise ValueError("MFSDP v2 does not support fsdp_manual_registration.")
        if ddp_config.suggested_communication_unit_size is not None:
            raise ValueError("MFSDP v2 does not support suggested_communication_unit_size.")
        if ddp_config.num_buckets is not None:
            raise ValueError("MFSDP v2 does not support num_buckets.")
        if ddp_config.megatron_fsdp_use_decoupled_grad:
            raise ValueError("MFSDP v2 does not support megatron_fsdp_use_decoupled_grad.")
        if ddp_config.megatron_fsdp_max_pool_double_buffer:
            raise ValueError("MFSDP v2 does not support megatron_fsdp_max_pool_double_buffer.")

    @contextmanager
    def no_sync(self):
        """Suppress gradient finalization for non-final microbatches.

        Toggles ``is_last_microbatch`` on all root ``FsdpContext`` instances
        so gradient reduce-scatters accumulate between microbatches rather
        than finalizing on every backward.  Called by the training loop via
        ``config.no_sync_func`` and the 1F1B overlap schedule.
        """
        self.module.context.ensure_finalized()
        context = self.module.context
        previous_state = context.is_last_microbatch
        context.is_last_microbatch = False
        try:
            yield
        finally:
            context.is_last_microbatch = previous_state

    def start_param_sync(self, *unused, **unused_kwargs) -> None:
        """No-op: MFSDP v2 gathers parameters from its forward pre-hooks."""

    def start_grad_sync(self, *unused, **unused_kwargs) -> None:
        """MFSDP v2 reduces gradients during backward."""

    def finish_grad_sync(self, *unused, **unused_kwargs) -> None:
        """Fence optimizer-side work against asynchronous gradient reductions.

        Ordinary autograd backward installs an engine-final callback that creates
        this stream dependency. The combined 1F1B schedule finalizes FSDP units
        manually and deliberately skips that callback, so its reduce-scatters may
        still be in flight when ``finalize_model_grads`` reaches this method.
        """
        context = self.module.context
        context.current_stream().wait_stream(context.reduce_scatter_stream)

    def synchronize_param_gather(self, *unused, **unused_kwargs) -> None:
        """MFSDP v2 parameter gathers complete inside module hooks."""

    def broadcast_params(self) -> None:
        """Reject parameter broadcast, which is unsupported by MFSDP v2."""
        raise NotImplementedError(
            "MFSDP v2 does not support parameter broadcast/data-parallel random initialization."
        )

    def stop_communication(self) -> None:
        """MFSDP v2 communication is complete when backward returns."""


def FullyShardedDataParallel(
    config: TransformerConfig,
    ddp_config: DistributedDataParallelConfig,
    module: torch.nn.Module,
    fsdp_unit_modules: Optional[List[Type[torch.nn.Module]]] = None,
    disable_bucketing: bool = False,
    device: Optional[torch.device] = None,
    pg_collection: Optional[ProcessGroupCollection] = None,
) -> _BaseDataParallel:
    """Construct the configured Megatron-FSDP implementation.

    This is a factory function, not a wrapper type. Use the explicit V1 or V2
    implementation classes for type checks.
    """
    fsdp_class = (
        FullyShardedDataParallelV2
        if ddp_config.megatron_fsdp_version == 2
        else FullyShardedDataParallelV1
    )
    return fsdp_class(
        config, ddp_config, module, fsdp_unit_modules, disable_bucketing, device, pg_collection
    )


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


def _build_hybrid_dp_mesh(outer_group, inner_group, device_type):
    """Build the ("dp_outer", "dp_shard") mesh for a hybrid data-parallel domain.

    DeviceMesh.from_group requires an explicit rank table when given more than one group,
    since no single argument spans the mesh. parallel_state cuts the data-parallel domain
    into num_distributed_optimizer_instances contiguous chunks, so the table is world
    ranks reshaped to (outer, inner).

    The assumption is checked rather than trusted, because the position of a rank in the
    table is its mesh coordinate: a table with the right members in the wrong order would
    keep reducing over valid groups while assigning every shard index to the wrong rank.
    """
    if outer_group is None or inner_group is None:
        raise ValueError(
            "MFSDP v2 with num_distributed_optimizer_instances > 1 requires both the "
            "inter- and intra-distributed-optimizer process groups."
        )

    inner_size = inner_group.size()
    layout = torch.arange(dist.get_world_size()).reshape(outer_group.size(), inner_size).tolist()

    outer_index, inner_index = divmod(dist.get_rank(), inner_size)
    expected_inner = layout[outer_index]
    expected_outer = [row[inner_index] for row in layout]
    actual_inner = dist.get_process_group_ranks(inner_group)
    actual_outer = dist.get_process_group_ranks(outer_group)
    if actual_inner != expected_inner:
        raise ValueError(
            f"MFSDP v2 hybrid mesh row {expected_inner} does not match the intra "
            f"data-parallel group {actual_inner}."
        )
    if actual_outer != expected_outer:
        raise ValueError(
            f"MFSDP v2 hybrid mesh column {expected_outer} does not match the inter "
            f"distributed-optimizer group {actual_outer}."
        )

    return DeviceMesh.from_group(
        [outer_group, inner_group],
        device_type=device_type,
        mesh=layout,
        mesh_dim_names=("dp_outer", "dp_shard"),
    )


def _get_hsdp_tp_mesh(outer_fsdp_dp_group, dp_cp_group, tp_group, ep_size=1):
    assert HAVE_EINOPS, "einops is not installed. Please install it with `pip install einops`."
    world_size = dist.get_world_size()

    mesh = einops.rearrange(
        torch.arange(world_size),
        "(outer_fsdp_dp fsdp ep tp) -> ep outer_fsdp_dp fsdp tp",
        outer_fsdp_dp=outer_fsdp_dp_group.size(),
        tp=tp_group.size(),
        ep=ep_size,
    )

    mesh_fsdp_ranks = einops.rearrange(
        mesh,
        'ep outer_fsdp_dp fsdp tp -> (outer_fsdp_dp ep tp) fsdp',
        tp=tp_group.size(),
        fsdp=dp_cp_group.size(),
        ep=ep_size,
    )
    fsdp_group_ranks = dist.get_process_group_ranks(dp_cp_group)
    assert _check_mesh_ranks_and_group_ranks_are_consistent(mesh_fsdp_ranks, fsdp_group_ranks), (
        f"[Megatron-FSDP] FSDP ranks in the mesh {mesh_fsdp_ranks} "
        f"do not match the ranks in the FSDP group {fsdp_group_ranks}."
    )

    mesh_tp_ranks = einops.rearrange(
        mesh,
        'ep outer_fsdp_dp fsdp tp -> (outer_fsdp_dp fsdp ep) tp',
        tp=tp_group.size(),
        fsdp=dp_cp_group.size(),
    )
    tp_group_ranks = dist.get_process_group_ranks(tp_group)
    assert _check_mesh_ranks_and_group_ranks_are_consistent(mesh_tp_ranks, tp_group_ranks), (
        f"[Megatron-FSDP] Tensor Parallel ranks in the mesh {mesh_tp_ranks} "
        f"do not match the ranks in the TP group {tp_group_ranks}."
    )

    mesh_outer_fsdp_dp_ranks = einops.rearrange(
        mesh,
        'ep outer_fsdp_dp fsdp tp -> (fsdp ep tp) outer_fsdp_dp',
        tp=tp_group.size(),
        fsdp=dp_cp_group.size(),
        ep=ep_size,
    )
    outer_fsdp_dp_group_ranks = dist.get_process_group_ranks(outer_fsdp_dp_group)
    assert _check_mesh_ranks_and_group_ranks_are_consistent(
        mesh_outer_fsdp_dp_ranks, outer_fsdp_dp_group_ranks
    ), (
        f"[Megatron-FSDP] Outer FSDP Data Parallel ranks in the mesh {mesh_outer_fsdp_dp_ranks} "
        f"do not match the ranks in the Outer FSDP DP group {outer_fsdp_dp_group_ranks}."
    )

    # Exclude the expert parallel dimension
    rank = dist.get_rank()
    dp_tp_meshes = [per_ep_mesh for per_ep_mesh in mesh if rank in per_ep_mesh.reshape(-1).tolist()]
    assert (
        len(dp_tp_meshes) == 1
    ), f"[Megatron-FSDP] Current rank {rank} is not unique in the mesh ranks {mesh.tolist()}."
    assert (
        len(dp_tp_meshes[0].reshape(-1).tolist())
        == outer_fsdp_dp_group.size() * dp_cp_group.size() * tp_group.size()
    ), (
        f"[Megatron-FSDP] DP-TP mesh size {len(dp_tp_meshes[0].reshape(-1).tolist())} "
        f"does not match the expected size"
        f"{outer_fsdp_dp_group.size() * dp_cp_group.size() * tp_group.size()}."
    )
    return dp_tp_meshes[0]


def _get_dp_tp_mesh(dp_cp_group, tp_group, ep_size=1):
    assert HAVE_EINOPS, "einops is not installed. Please install it with `pip install einops`."
    world_size = dist.get_world_size()

    tp_size = dist.get_world_size(tp_group) if tp_group is not None else 1
    # TODO: Supports configurable (dp, cp, ep, tp) order.
    mesh = einops.rearrange(
        torch.arange(world_size),
        "(dp_cp ep tp) -> ep dp_cp tp",
        dp_cp=dp_cp_group.size(),
        tp=tp_size,
        ep=ep_size,
    )

    mesh_dp_ranks = einops.rearrange(mesh, 'ep dp_cp tp -> (ep tp) dp_cp', dp_cp=dp_cp_group.size())
    dp_cp_group_ranks = dist.get_process_group_ranks(dp_cp_group)
    assert _check_mesh_ranks_and_group_ranks_are_consistent(mesh_dp_ranks, dp_cp_group_ranks), (
        f"[Megatron-FSDP] Data Parallel ranks in the mesh {mesh_dp_ranks} "
        f"do not match the ranks in the DP group {dp_cp_group_ranks}."
    )

    mesh_tp_ranks = einops.rearrange(mesh, 'ep dp_cp tp -> (dp_cp ep) tp', tp=tp_size)
    tp_group_ranks = dist.get_process_group_ranks(tp_group)
    assert _check_mesh_ranks_and_group_ranks_are_consistent(mesh_tp_ranks, tp_group_ranks), (
        f"[Megatron-FSDP] Tensor Parallel ranks in the mesh {mesh_tp_ranks} "
        f"do not match the ranks in the TP group {tp_group_ranks}."
    )

    # Exclude the expert parallel dimension
    rank = dist.get_rank()
    dp_tp_meshes = [per_ep_mesh for per_ep_mesh in mesh if rank in per_ep_mesh.reshape(-1).tolist()]
    assert (
        len(dp_tp_meshes) == 1
    ), f"[Megatron-FSDP] Current rank {rank} is not unique in the mesh ranks {mesh.tolist()}."
    assert len(dp_tp_meshes[0].reshape(-1).tolist()) == dp_cp_group.size() * tp_group.size(), (
        f"[Megatron-FSDP] DP-TP mesh size {len(dp_tp_meshes[0].reshape(-1).tolist())} "
        f"does not match expected size {dp_cp_group.size() * tp_group.size()}."
    )

    return dp_tp_meshes[0]


def _check_mesh_ranks_and_group_ranks_are_consistent(mesh_ranks, group_ranks):
    current_rank = dist.get_rank()
    current_ranks = list(filter(lambda ranks: current_rank in ranks, mesh_ranks.tolist()))
    assert len(current_ranks) == 1, (
        f"[Megatron-FSDP] Current rank {current_rank} is not unique in "
        f"the mesh ranks {mesh_ranks.tolist()}."
    )
    assert sorted(current_ranks[0]) == sorted(group_ranks), (
        f"[Megatron-FSDP] Current rank {current_rank} in the mesh ranks "
        f"{mesh_ranks.tolist()} does not match the group ranks {group_ranks}."
    )
    return sorted(current_ranks[0]) == sorted(group_ranks)


def _get_rng_state_dict():
    rng_state_dict = {
        'random_rng_state': random.getstate(),
        'np_rng_state': np.random.get_state(),
        'torch_rng_state': torch.get_rng_state(),
        'cuda_rng_state': torch.cuda.get_rng_state(),
        'rng_tracker_states': tensor_parallel.get_cuda_rng_tracker().get_states(),
    }
    return rng_state_dict


def _load_rng_state_dict(rng_state_dict):
    random.setstate(rng_state_dict['random_rng_state'])
    np.random.set_state(rng_state_dict['np_rng_state'])
    torch.set_rng_state(rng_state_dict['torch_rng_state'])
    torch.cuda.set_rng_state(rng_state_dict['cuda_rng_state'])
    tensor_parallel.get_cuda_rng_tracker().set_states(rng_state_dict['rng_tracker_states'])
