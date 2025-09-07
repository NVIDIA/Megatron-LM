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
from typing import List, Optional

try:
    import einops

    HAVE_EINOPS = True
except ImportError:
    HAVE_EINOPS = False

import torch
import torch.distributed as dist

try:
    from torch.distributed import DeviceMesh

    HAVE_DTENSOR = True
except ImportError:
    HAVE_DTENSOR = False

from megatron.core import parallel_state
from megatron.core.config_logger import has_config_logger_enabled, log_config_to_disk
from megatron.core.distributed.data_parallel_base import _BaseDataParallel
from megatron.core.distributed.distributed_data_parallel_config import DistributedDataParallelConfig
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core.utils import log_single_rank

try:
    from megatron.core.distributed.fsdp.src.megatron_fsdp import FSDPDistributedIndex, MegatronFSDP

    HAVE_MEGATRON_FSDP = True
except ImportError as import_megatron_fsdp_error:
    IMPORT_MEGATRON_FSDP_ERROR = import_megatron_fsdp_error
    HAVE_MEGATRON_FSDP = False

logger = logging.getLogger(__name__)


class FullyShardedDataParallel(_BaseDataParallel):
    """
    Fully Sharded Data Parallel (FSDP) wrapper for the Megatron model.
    """

    def __init__(
        self,
        config: TransformerConfig,
        ddp_config: DistributedDataParallelConfig,
        module: torch.nn.Module,
        fsdp_unit_modules: Optional[List[torch.nn.Module]] = None,
        disable_bucketing: bool = False,
        device: Optional[torch.device] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        if not HAVE_MEGATRON_FSDP:
            raise IMPORT_MEGATRON_FSDP_ERROR

        if has_config_logger_enabled(config):
            log_config_to_disk(config, locals(), prefix=type(self).__name__)

        self.ddp_config = ddp_config
        log_single_rank(
            logger,
            logging.INFO,
            f'Setting up DistributedDataParallel with config {self.ddp_config}',
        )

        self.megatron_fsdp_dist_index = self._init_dist_index(pg_collection)

        self.bucket_size = self.ddp_config.bucket_size
        if disable_bucketing:
            self.bucket_size = None
        self.device = device if device else torch.device(f'cuda:{torch.cuda.current_device()}')

        if fsdp_unit_modules is not None:
            self.fsdp_unit_modules = fsdp_unit_modules
        else:
            if self.ddp_config.data_parallel_sharding_strategy == "optim_grads_params":
                self.fsdp_unit_modules = [TransformerLayer]
            else:
                self.fsdp_unit_modules = []

        super().__init__(
            config=config,
            module=MegatronFSDP(
                ddp_config=ddp_config,
                module=module,
                fsdp_unit_modules=self.fsdp_unit_modules,
                disable_bucketing=disable_bucketing,
                device=self.device,
                dist_index=self.megatron_fsdp_dist_index,
                calculate_per_token_loss=config.calculate_per_token_loss,
                init_model_with_meta_device=config.init_model_with_meta_device,
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
        self.module.state_dict_for_save_checkpoint = self.module.state_dict
        self.state_dict_for_save_checkpoint = self.state_dict

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

    def _init_dist_index(self, pg_collection):
        """
        Initialize the distributed index for the module.
        """
        if not HAVE_DTENSOR:
            raise ImportError(
                "This module requires PyTorch with DTensor support. "
                "Please install a compatible version of PyTorch."
            )

        enable_hsdp = self.ddp_config.num_distributed_optimizer_instances > 1
        if pg_collection is None:
            tp_group = parallel_state.get_tensor_model_parallel_group()
            if enable_hsdp:
                dp_cp_group = parallel_state.get_data_parallel_group(
                    with_context_parallel=True, partial_data_parallel=True
                )
                inter_fsdp_group = parallel_state.get_inter_distributed_optimizer_instance_group()
                hybrid_fsdp_group = parallel_state.get_data_parallel_group(
                    with_context_parallel=True, partial_data_parallel=False
                )
            else:
                dp_cp_group = parallel_state.get_data_parallel_group(
                    with_context_parallel=True, partial_data_parallel=False
                )
                inter_fsdp_group = None
                hybrid_fsdp_group = None
        else:
            tp_group = getattr(pg_collection, 'tp', None)
            if enable_hsdp:
                dp_cp_group = pg_collection.intra_dp_cp
                inter_fsdp_group = pg_collection.inter_dist_opt
                hybrid_fsdp_group = pg_collection.dp_cp
            else:
                dp_cp_group = pg_collection.dp_cp
                inter_fsdp_group = None
                hybrid_fsdp_group = None

        if tp_group is None:
            single_rank_group = dist.new_group(ranks=[dist.get_rank()])
            tp_group = single_rank_group

        if enable_hsdp:
            mesh = _get_hsdp_tp_mesh(inter_fsdp_group, dp_cp_group, tp_group)
            dist_index = FSDPDistributedIndex(
                use_hybrid_fsdp=True,
                hsdp_outer_dp_shard=self.ddp_config.outer_dp_sharding_strategy != "no_shard",
                device_mesh=DeviceMesh.from_group(
                    [inter_fsdp_group, dp_cp_group, tp_group],
                    device_type="cuda",
                    mesh=mesh.tolist(),
                    mesh_dim_names=["inter_fsdp_dp", "dp_cp", "tp"],
                ),
                dp_inter_dim="inter_fsdp_dp",
                dp_shard_dim="dp_cp",
                tp_dim="tp",
                hybrid_fsdp_group=hybrid_fsdp_group,
            )
        else:
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
            )

        return dist_index

    def stop_communication(self):
        """
        Stop communication for the module.
        """
        self.module.synchronize_gradient_reduce()
        self.module.synchronize_param_gather()


def _get_hsdp_tp_mesh(inter_fsdp_dp_group, dp_cp_group, tp_group):
    assert HAVE_EINOPS, "einops is not installed. Please install it with `pip install einops`."
    world_size = dist.get_world_size()

    mesh = einops.rearrange(
        torch.arange(world_size),
        "(inter_fsdp_dp fsdp tp) -> inter_fsdp_dp fsdp tp",
        inter_fsdp_dp=inter_fsdp_dp_group.size(),
        tp=tp_group.size(),
    )

    mesh_fsdp_ranks = einops.rearrange(
        mesh,
        'inter_fsdp_dp fsdp tp -> (inter_fsdp_dp tp) fsdp',
        tp=tp_group.size(),
        fsdp=dp_cp_group.size(),
    )
    fsdp_group_ranks = dist.get_process_group_ranks(dp_cp_group)
    assert _check_mesh_ranks_and_group_ranks_are_consistent(mesh_fsdp_ranks, fsdp_group_ranks), (
        f"[Megatron-FSDP] FSDP ranks in the mesh {mesh_fsdp_ranks} "
        f"do not match the ranks in the FSDP group {fsdp_group_ranks}."
    )

    mesh_tp_ranks = einops.rearrange(
        mesh,
        'inter_fsdp_dp fsdp tp -> (inter_fsdp_dp fsdp) tp',
        tp=tp_group.size(),
        fsdp=dp_cp_group.size(),
    )
    tp_group_ranks = dist.get_process_group_ranks(tp_group)
    assert _check_mesh_ranks_and_group_ranks_are_consistent(mesh_tp_ranks, tp_group_ranks), (
        f"[Megatron-FSDP] Tensor Parallel ranks in the mesh {mesh_tp_ranks} "
        f"do not match the ranks in the TP group {tp_group_ranks}."
    )

    mesh_inter_fsdp_dp_ranks = einops.rearrange(
        mesh,
        'inter_fsdp_dp fsdp tp -> (fsdp tp) inter_fsdp_dp',
        tp=tp_group.size(),
        fsdp=dp_cp_group.size(),
    )
    inter_fsdp_dp_group_ranks = dist.get_process_group_ranks(inter_fsdp_dp_group)
    assert _check_mesh_ranks_and_group_ranks_are_consistent(
        mesh_inter_fsdp_dp_ranks, inter_fsdp_dp_group_ranks
    ), (
        f"[Megatron-FSDP] Inter FSDP Data Parallel ranks in the mesh {mesh_inter_fsdp_dp_ranks} "
        f"do not match the ranks in the Inter FSDP DP group {inter_fsdp_dp_group_ranks}."
    )

    return mesh


def _get_dp_tp_mesh(dp_cp_group, tp_group):
    assert HAVE_EINOPS, "einops is not installed. Please install it with `pip install einops`."
    world_size = dist.get_world_size()

    tp_size = dist.get_world_size(tp_group) if tp_group is not None else 1
    # TODO: Supports configurable (dp, cp, tp) order.
    mesh = einops.rearrange(torch.arange(world_size), "(dp_cp tp) -> dp_cp tp", tp=tp_size)

    mesh_dp_ranks = einops.rearrange(mesh, 'dp_cp tp -> tp dp_cp', tp=tp_size)
    dp_cp_group_ranks = dist.get_process_group_ranks(dp_cp_group)
    assert _check_mesh_ranks_and_group_ranks_are_consistent(mesh_dp_ranks, dp_cp_group_ranks), (
        f"[Megatron-FSDP] Data Parallel ranks in the mesh {mesh_dp_ranks} "
        f"do not match the ranks in the DP group {dp_cp_group_ranks}."
    )

    mesh_tp_ranks = einops.rearrange(mesh, 'dp_cp tp -> (dp_cp) tp', tp=tp_size)
    tp_group_ranks = dist.get_process_group_ranks(tp_group)
    assert _check_mesh_ranks_and_group_ranks_are_consistent(mesh_tp_ranks, tp_group_ranks), (
        f"[Megatron-FSDP] Tensor Parallel ranks in the mesh {mesh_tp_ranks} "
        f"do not match the ranks in the TP group {tp_group_ranks}."
    )

    return mesh


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
