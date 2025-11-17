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
from typing import List, Optional

import numpy as np
import torch
import torch.distributed as dist

try:
    from torch.distributed import DeviceMesh

    HAVE_DTENSOR = True
except ImportError:
    HAVE_DTENSOR = False

from megatron.core import parallel_state, tensor_parallel
from megatron.core.config_logger import has_config_logger_enabled, log_config_to_disk
from megatron.core.distributed.data_parallel_base import _BaseDataParallel
from megatron.core.distributed.distributed_data_parallel_config import DistributedDataParallelConfig
from megatron.core.extensions.transformer_engine import TELinear
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

        self._fix_tensor_parallel_attributes(module)

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

    def _fix_tensor_parallel_attributes(self, module):
        is_expert_param = lambda n, p: ".experts." in n
        is_router_param = lambda n, p: ".router.weight" in n

        if parallel_state.get_tensor_model_parallel_group():
            tp_size = parallel_state.get_tensor_model_parallel_group().size()
        else:
            tp_size = 1

        if parallel_state.get_expert_tensor_parallel_group():
            expt_tp_size = parallel_state.get_expert_tensor_parallel_group().size()
        else:
            expt_tp_size = 1

        param_to_direct_module = {}
        for name, m in module.named_modules():
            for p in m.parameters(recurse=False):
                param_to_direct_module[p] = (name, m)

        for name, param in module.named_parameters():
            if is_expert_param(name, param) and expt_tp_size > 1:
                setattr(param, "_mcore_tp", True)
                if "linear_fc1.weight" in name:
                    setattr(param, "_tp_partition_dim", 0)
                elif "linear_fc2.weight" in name:
                    setattr(param, "_tp_partition_dim", 1)

            if not is_expert_param(name, param) and tp_size > 1:
                m_name, direct_module = param_to_direct_module[param]
                if isinstance(direct_module, (TELinear,)):
                    parallel_mode = getattr(direct_module, "parallel_mode", None)
                    if parallel_mode is None:
                        setattr(param, "_mcore_tp", True)
                        setattr(param, "_tp_duplicated", True)
                elif is_router_param(name, param):
                    setattr(param, "_mcore_tp", True)
                    setattr(param, "_tp_duplicated", True)

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
            expt_tp_group = parallel_state.get_expert_tensor_parallel_group()
            if enable_hsdp:
                dp_cp_group = parallel_state.get_data_parallel_group(
                    with_context_parallel=True, partial_data_parallel=True
                )
                outer_fsdp_group = parallel_state.get_inter_distributed_optimizer_instance_group()
                hybrid_fsdp_group = parallel_state.get_data_parallel_group(
                    with_context_parallel=True, partial_data_parallel=False
                )
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

        if enable_hsdp:
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
            )
        else:
            if ep_group is not None:
                expt_mesh = _get_dp_tp_mesh(expt_dp_group, expt_tp_group)
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


def _get_hsdp_tp_mesh(outer_fsdp_dp_group, dp_cp_group, tp_group):
    if tp_group is None:
        tp_ranks = [dist.get_rank()]
    else:
        tp_ranks = dist.get_process_group_ranks(tp_group)

    if dp_cp_group is None:
        dp_cp_tp_ranks = [tp_ranks]
    else:
        dp_size = dist.get_world_size(dp_cp_group)
        dp_cp_tp_ranks = [None for _ in range(dp_size)]
        dist.all_gather_object(dp_cp_tp_ranks, tp_ranks, group=dp_cp_group)

    if outer_fsdp_dp_group is None:
        outer_fsdp_dp_dp_cp_tp_ranks = [dp_cp_tp_ranks]
    else:
        outer_fsdp_dp_size = dist.get_world_size(outer_fsdp_dp_group)
        outer_fsdp_dp_dp_cp_tp_ranks = [None for _ in range(outer_fsdp_dp_size)]
        dist.all_gather_object(
            outer_fsdp_dp_dp_cp_tp_ranks, dp_cp_tp_ranks, group=outer_fsdp_dp_group
        )

    return torch.tensor(outer_fsdp_dp_dp_cp_tp_ranks)


def _get_dp_tp_mesh(dp_cp_group, tp_group):
    if tp_group is None:
        tp_ranks = [dist.get_rank()]
    else:
        tp_ranks = dist.get_process_group_ranks(tp_group)

    if dp_cp_group is None:
        return torch.tensor([tp_ranks])

    dp_size = dist.get_world_size(dp_cp_group)
    dp_cp_tp_ranks = [None for _ in range(dp_size)]
    dist.all_gather_object(dp_cp_tp_ranks, tp_ranks, group=dp_cp_group)

    return torch.tensor(dp_cp_tp_ranks)


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
