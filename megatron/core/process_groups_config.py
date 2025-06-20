# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Dataclasses for organizing model parallelism and gradient communication process groups."""

from dataclasses import dataclass, field, fields
from typing import List, Optional

import torch

from megatron.core import parallel_state


@dataclass
class ModelCommProcessGroups:
    """Process groups for transformer model parallelism.

    Fields use init=False and must be set after instance creation.

    Args:
        tp: Tensor parallel process group
        pp: Pipeline parallel process group
        mp: Model parallel group (tensor + pipeline)
        embd: Embedding process group
        pos_embd: Position embedding process group
        cp: Context parallel process group
        tp_cp: Tensor and context parallel group
        hcp: Hierarchical context parallel groups
        ep: Expert model parallel group
        expt_tp: Expert tensor parallel group
        tp_ep: Tensor and expert parallel group
        tp_ep_pp: Tensor, expert, and pipeline parallel group
        expt_dp: Expert data parallel group
    Example:
        # Create instance and set needed process groups
        model_pgs = ModelCommProcessGroups()
        model_pgs.tp = tp_group
        model_pgs.pp = pp_group

        # Pass to model components
        model = TransformerModel(..., process_groups=model_pgs)
    """

    # _TENSOR_MODEL_PARALLEL_GROUP
    tp: torch.distributed.ProcessGroup = field(init=False)

    # _PIPELINE_MODEL_PARALLEL_GROUP
    pp: torch.distributed.ProcessGroup = field(init=False)

    # _MODEL_PARALLEL_GROUP
    mp: torch.distributed.ProcessGroup = field(init=False)

    # _EMBEDDING_GROUP
    embd: torch.distributed.ProcessGroup = field(init=False)

    # _POSITION_EMBEDDING_GROUP
    pos_embd: torch.distributed.ProcessGroup = field(init=False)

    # _CONTEXT_PARALLEL_GROUP
    cp: torch.distributed.ProcessGroup = field(init=False)

    # _TENSOR_AND_CONTEXT_PARALLEL_GROUP
    tp_cp: torch.distributed.ProcessGroup = field(init=False)

    # _HIERARCHICAL_CONTEXT_PARALLEL_GROUPS
    hcp: List[torch.distributed.ProcessGroup] = field(init=False)

    # _EXPERT_MODEL_PARALLEL_GROUP
    ep: torch.distributed.ProcessGroup = field(init=False)

    # _EXPERT_TENSOR_PARALLEL_GROUP
    expt_tp: torch.distributed.ProcessGroup = field(init=False)

    # _EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP
    tp_ep: torch.distributed.ProcessGroup = field(init=False)

    # _EXPERT_TENSOR_MODEL_PIPELINE_PARALLEL_GROUP
    tp_ep_pp: torch.distributed.ProcessGroup = field(init=False)

    # MoE layers need expt_dp group for sharded state dict
    # we need this workaround until distributed checkpoint is refactored
    # to have sharded_state_dict can take the PG and pass it down
    # TODO (Hepteract): remove this once distributed checkpoint is refactored
    # _EXPERT_DATA_PARALLEL_GROUP
    expt_dp: torch.distributed.ProcessGroup = field(init=False)

    def __init__(self, **kwargs):
        for key in kwargs:
            if key in [field.name for field in fields(self)]:
                setattr(self, key, kwargs[key])
            else:
                raise ValueError(f"Unknown attribute: {key}")

    @classmethod
    def use_mpu_process_groups(cls, required_pgs: Optional[List[str]] = None):
        """
        Use the default process groups from parallel_state.

        Args:
            required_pgs (List[str], optional): List of process group names to initialize.
                If None, pull all default process groups. Each string should correspond to
                one of the dataclass process group attributes.
        """
        # Get all available process groups
        all_pgs = {field.name for field in fields(cls)}

        # If no specific process groups requested, use all
        if required_pgs is None:
            required_pgs = list(all_pgs)

        # Validate requested process groups
        invalid_pgs = [pg for pg in required_pgs if pg not in all_pgs]
        if invalid_pgs:
            raise ValueError(f"Invalid process groups requested: {invalid_pgs}")

        # Mapping of attribute names to their initialization functions
        pg_to_func = {
            'tp': parallel_state.get_tensor_model_parallel_group,
            'pp': parallel_state.get_pipeline_model_parallel_group,
            'mp': parallel_state.get_model_parallel_group,
            'cp': parallel_state.get_context_parallel_group,
            'tp_cp': parallel_state.get_tensor_and_context_parallel_group,
            'hcp': parallel_state.get_hierarchical_context_parallel_groups,
            'ep': parallel_state.get_expert_model_parallel_group,
            'expt_tp': parallel_state.get_expert_tensor_parallel_group,
            'tp_ep': parallel_state.get_expert_tensor_and_model_parallel_group,
            'tp_ep_pp': parallel_state.get_expert_tensor_model_pipeline_parallel_group,
            'embd': parallel_state.get_embedding_group,
            'pos_embd': parallel_state.get_position_embedding_group,
            # TODO (Hepteract): remove this once distributed checkpoint is refactored
            'expt_dp': parallel_state.get_expert_data_parallel_group,
        }

        # Build initialization dict by calling appropriate parallel_state get_foo_group
        init_dict = {pg: pg_to_func[pg](False) for pg in required_pgs}

        return cls(**init_dict)


@dataclass
class GradCommProcessGroups:
    """Process groups for gradient communication in distributed training.

    Fields use init=False and must be set after instance creation.

    Args:
        dp: Data parallel process group
        dp_cp: Data and context parallel group
        expt_dp: Expert data parallel group
        intra_dp_cp: Intra partial data parallel group
        intra_expt_dp: Intra partial expert data parallel group
        inter_dist_opt: Inter distributed optimizer instance group

    Example:
        # Create instance and set needed process groups
        grad_pgs = GradCommProcessGroups()
        grad_pgs.dp = dp_group

        # Pass to distributed data parallel wrapper
        ddp_model = DistributedDataParallel(..., process_groups=grad_pgs)
    """

    # _DATA_PARALLEL_GROUP
    dp: torch.distributed.ProcessGroup = field(init=False)

    # _DATA_PARALLEL_GROUP_WITH_CP
    dp_cp: torch.distributed.ProcessGroup = field(init=False)

    # _EXPERT_DATA_PARALLEL_GROUP
    expt_dp: torch.distributed.ProcessGroup = field(init=False)

    # _INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP
    intra_dp_cp: torch.distributed.ProcessGroup = field(init=False)

    # _INTRA_EXPERT_DATA_PARALLEL_GROUP
    intra_expt_dp: torch.distributed.ProcessGroup = field(init=False)

    # _INTER_DISTRIBUTED_OPTIMIZER_INSTANCE_GROUP
    inter_dist_opt: torch.distributed.ProcessGroup = field(init=False)
