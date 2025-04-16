# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Dataclasses for organizing model parallelism and gradient communication process groups."""

from dataclasses import dataclass, field
from typing import List

import torch

from megatron.core.wrapped_process_group import WrappedProcessGroup

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

    Example:
        # Create instance and set needed process groups
        model_pgs = ModelCommProcessGroups()
        model_pgs.tp = tp_group
        model_pgs.pp = pp_group

        # Pass to model components
        model = TransformerModel(..., process_groups=model_pgs)
    """

    # _TENSOR_MODEL_PARALLEL_GROUP
    tp: WrappedProcessGroup = field(init=False)

    # _PIPELINE_MODEL_PARALLEL_GROUP
    pp: WrappedProcessGroup = field(init=False)

    # _MODEL_PARALLEL_GROUP
    mp: WrappedProcessGroup = field(init=False)

    # _EMBEDDING_GROUP
    embd: WrappedProcessGroup = field(init=False)

    # _POSITION_EMBEDDING_GROUP
    pos_embd: WrappedProcessGroup = field(init=False)

    # _CONTEXT_PARALLEL_GROUP
    cp: WrappedProcessGroup = field(init=False)

    # _TENSOR_AND_CONTEXT_PARALLEL_GROUP
    tp_cp: WrappedProcessGroup = field(init=False)

    # _HIERARCHICAL_CONTEXT_PARALLEL_GROUPS
    hcp: List[WrappedProcessGroup] = field(init=False)

    # _EXPERT_MODEL_PARALLEL_GROUP
    ep: WrappedProcessGroup = field(init=False)

    # _EXPERT_TENSOR_PARALLEL_GROUP
    expt_tp: WrappedProcessGroup = field(init=False)

    # _EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP
    tp_ep: WrappedProcessGroup = field(init=False)

    # _EXPERT_TENSOR_MODEL_PIPELINE_PARALLEL_GROUP
    tp_ep_pp: WrappedProcessGroup = field(init=False)


@dataclass
class GradCommProcessGroups:
    """Process groups for gradient communication in distributed training.

    Fields use init=False and must be set after instance creation.

    Args:
        dp: Data parallel process group
        dp_cp: Data and context parallel group
        expt_dp: Expert data parallel group
        intra_dp_cp: Intra partial data parallel group
        inter_dp_cp: Inter partial data parallel group

    Example:
        # Create instance and set needed process groups
        grad_pgs = GradCommProcessGroups()
        grad_pgs.dp = dp_group

        # Pass to distributed data parallel wrapper
        ddp_model = DistributedDataParallel(..., process_groups=grad_pgs)
    """

    # _DATA_PARALLEL_GROUP
    dp: WrappedProcessGroup = field(init=False)

    # _DATA_PARALLEL_GROUP_WITH_CP
    dp_cp: WrappedProcessGroup = field(init=False)

    # _EXPERT_DATA_PARALLEL_GROUP
    expt_dp: WrappedProcessGroup = field(init=False)

    # _INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP
    intra_dp_cp: WrappedProcessGroup = field(init=False)

    # _INTER_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP
    inter_dp_cp: WrappedProcessGroup = field(init=False)
