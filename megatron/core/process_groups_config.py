# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Dataclasses for organizing model parallelism and gradient communication process groups."""

from dataclasses import dataclass, field, fields
from functools import partial
from typing import List, Optional

import torch

from megatron.core import parallel_state


class ProcessGroupHelperMeta(type):
    """Metaclass to protect virtual_pipeline_model_parallel_size from direct assignment."""

    def __setattr__(cls, name, value):
        if name == 'virtual_pipeline_model_parallel_size':
            raise AttributeError(
                f"Cannot set '{name}' directly. Use set_virtual_pipeline_model_parallel_size() "
                f"method instead."
            )
        super().__setattr__(name, value)


@dataclass
class ProcessGroupCollection:
    """Unified process group collection for transformer model parallelism, gradient communication,
     and finalization.

    Fields use init=False and must be set after instance creation.

    Args:
        # Model Parallelism Groups
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

        # Data Parallelism Groups
        dp: Data parallel process group
        dp_cp: Data and context parallel group
        expt_dp: Expert data parallel group
        intra_dp_cp: Intra partial data parallel group
        intra_expt_dp: Intra partial expert data parallel group
        inter_dist_opt: Inter distributed optimizer instance group

    Example:
        # Create instance and set needed process groups
        pgs = ProcessGroupCollection()
        pgs.tp = tp_group
        pgs.pp = pp_group
        pgs.dp = dp_group

        # Pass to model components
        model = TransformerModel(..., pg_collection=pgs)
        ddp_model = DistributedDataParallel(..., pg_collection=pgs)
        finalize_model_grads(..., pg_collection=pgs)
    """

    # Model Parallelism Process Groups
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

    # Expert Parallelism Process Groups
    # _EXPERT_MODEL_PARALLEL_GROUP
    ep: torch.distributed.ProcessGroup = field(init=False)

    # _EXPERT_TENSOR_PARALLEL_GROUP
    expt_tp: torch.distributed.ProcessGroup = field(init=False)

    # _EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP
    tp_ep: torch.distributed.ProcessGroup = field(init=False)

    # _EXPERT_TENSOR_MODEL_PIPELINE_PARALLEL_GROUP
    tp_ep_pp: torch.distributed.ProcessGroup = field(init=False)

    # _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP
    tp_dp_cp: torch.distributed.ProcessGroup = field(init=False)

    # Data Parallelism Process Groups
    # _DATA_PARALLEL_GROUP
    dp: torch.distributed.ProcessGroup = field(init=False)

    # _DATA_PARALLEL_GROUP_WITH_CP
    dp_cp: torch.distributed.ProcessGroup = field(init=False)

    # MoE layers need expt_dp group for sharded state dict
    # we need this workaround until distributed checkpoint is refactored
    # to have sharded_state_dict can take the PG and pass it down
    # TODO (Hepteract): remove this once distributed checkpoint is refactored
    # _EXPERT_DATA_PARALLEL_GROUP
    expt_dp: torch.distributed.ProcessGroup = field(init=False)

    # _INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP
    intra_dp_cp: torch.distributed.ProcessGroup = field(init=False)

    # _INTRA_EXPERT_DATA_PARALLEL_GROUP
    intra_expt_dp: torch.distributed.ProcessGroup = field(init=False)

    # _INTER_DISTRIBUTED_OPTIMIZER_INSTANCE_GROUP
    inter_dist_opt: torch.distributed.ProcessGroup = field(init=False)

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
            'tp_dp_cp': partial(
                parallel_state.get_tensor_and_data_parallel_group, with_context_parallel=True
            ),
        }

        # Build initialization dict by calling appropriate parallel_state get_foo_group
        init_dict = {
            pg: pg_to_func[pg](check_initialized=False) for pg in required_pgs if pg in pg_to_func
        }

        return cls(**init_dict)

    @staticmethod
    def setup_process_groups_for_optimizer(
        pg_collection: Optional['ProcessGroupCollection'],
        model_chunks: List,
        use_gloo_process_groups: bool = True,
    ):
        """
        Helper method to set up process groups for optimizer and DDP with proper validation
        and fallbacks.

        Args:
            pg_collection: Optional process group collection. If None, uses parallel_state groups.
            model_chunks: List of model chunks to extract configuration from.
            use_gloo_process_groups: Whether to set up gloo process groups.

        Returns:
            Dictionary containing all required process groups:
                - dp_group: Data parallel group
                - dp_cp_group: Data parallel with context parallel group
                - intra_dp_cp_group: Intra data parallel with context parallel group
                - expt_dp_group: Expert data parallel group
                - intra_expt_dp_group: Intra expert data parallel group
                - mp_group: Model parallel group
                - expt_tp_pp_group: Expert tensor-model-pipeline parallel group
                - inter_dist_opt_group: Inter distributed optimizer group (may be None)
                - intra_dp_cp_group_gloo: Gloo version of intra_dp_cp_group (may be None)
                - intra_expt_dp_group_gloo: Gloo version of intra_expt_dp_group (may be None)
        """
        from megatron.core import parallel_state
        from megatron.core.utils import get_model_config

        if pg_collection is None:
            # Use parallel_state groups
            dp_group = parallel_state.get_data_parallel_group(
                with_context_parallel=False, partial_data_parallel=False
            )
            dp_cp_group = parallel_state.get_data_parallel_group(
                with_context_parallel=True, partial_data_parallel=False
            )
            intra_dp_cp_group = parallel_state.get_data_parallel_group(
                with_context_parallel=True, partial_data_parallel=True
            )
            expt_dp_group = parallel_state.get_expert_data_parallel_group()
            intra_expt_dp_group = parallel_state.get_expert_data_parallel_group(
                partial_expert_data_parallel=True
            )

            # Gloo groups
            if use_gloo_process_groups:
                intra_dp_cp_group_gloo = parallel_state.get_data_parallel_group_gloo(
                    with_context_parallel=True, partial_data_parallel=True
                )
                intra_expt_dp_group_gloo = parallel_state.get_expert_data_parallel_group_gloo(
                    partial_expert_data_parallel=True
                )
            else:
                intra_dp_cp_group_gloo = None
                intra_expt_dp_group_gloo = None

            # Model communication groups
            mp_group = parallel_state.get_model_parallel_group()
            expt_tp_pp_group = parallel_state.get_expert_tensor_model_pipeline_parallel_group()

            # Inter distributed optimizer group
            if hasattr(model_chunks[0], 'ddp_config'):
                ddp_config = model_chunks[0].ddp_config
                if ddp_config.num_distributed_optimizer_instances > 1:
                    inter_dist_opt_group = (
                        parallel_state.get_inter_distributed_optimizer_instance_group()
                    )
                else:
                    inter_dist_opt_group = None
            else:
                inter_dist_opt_group = None

        else:
            # Use provided process group collection with validation and fallbacks

            # 1. dp group - this is always required
            if not hasattr(pg_collection, 'dp'):
                raise ValueError("dp process group is required but not provided in pg_collection")
            dp_group = pg_collection.dp

            # 2. dp_cp group: fallback logic based on context_parallel_size
            if hasattr(pg_collection, 'dp_cp'):
                dp_cp_group = pg_collection.dp_cp
            else:
                model_config = get_model_config(model_chunks[0])
                cp_size = getattr(model_config, 'context_parallel_size', 1)
                if cp_size == 1:
                    # If no context parallelism, dp_cp is same as dp
                    dp_cp_group = dp_group
                else:
                    raise ValueError(
                        "dp_cp process group is required when context_parallel_size > 1 "
                        "but not provided in pg_collection"
                    )

            # 3. Handle expert data parallel group
            if not hasattr(pg_collection, 'expt_dp'):
                raise ValueError(
                    "expt_dp process group is required but not provided in pg_collection. "
                    "Please explicitly set it to None if you don't need it."
                )
            expt_dp_group = pg_collection.expt_dp

            # 4. Handle intra_dp_cp, intra_expt_dp, and inter_dist_opt based on optimizer instances
            if hasattr(model_chunks[0], 'ddp_config'):
                ddp_config = model_chunks[0].ddp_config
                if ddp_config.num_distributed_optimizer_instances == 1:
                    # With a single optimizer instance:
                    # - intra_dp_cp is same as dp_cp
                    # - intra_expt_dp is same as expt_dp
                    # - inter_dist_opt is not needed (set to None)
                    intra_dp_cp_group = dp_cp_group
                    intra_expt_dp_group = expt_dp_group
                    inter_dist_opt_group = None
                else:
                    # With multiple optimizer instances, both groups must be provided
                    if not (
                        hasattr(pg_collection, 'intra_dp_cp')
                        and hasattr(pg_collection, 'intra_expt_dp')
                        and hasattr(pg_collection, 'inter_dist_opt')
                    ):
                        raise ValueError(
                            "intra_dp_cp, intra_expt_dp, and inter_dist_opt "
                            "process groups are required when using multiple optimizer "
                            "instances (>1) but not provided in pg_collection"
                        )
                    intra_dp_cp_group = pg_collection.intra_dp_cp
                    intra_expt_dp_group = pg_collection.intra_expt_dp
                    inter_dist_opt_group = pg_collection.inter_dist_opt
            else:
                # No ddp_config available - use simple fallback
                intra_dp_cp_group = dp_cp_group
                intra_expt_dp_group = expt_dp_group
                inter_dist_opt_group = None

            # 5. Model communication groups
            if not hasattr(pg_collection, 'mp'):
                raise ValueError(
                    "mp process group is required but not provided in pg_collection. "
                    "Please explicitly set it to None if you don't need it."
                )
            mp_group = pg_collection.mp

            # Expert tensor-model-pipeline group for MoE
            if not hasattr(pg_collection, 'tp_ep_pp'):
                raise ValueError(
                    "tp_ep_pp process group is required but not provided in pg_collection. "
                    "Please explicitly set it to None if you don't need it."
                )
            expt_tp_pp_group = pg_collection.tp_ep_pp

            # Gloo groups - not supported when pg_collection is provided
            if use_gloo_process_groups:
                raise ValueError(
                    "Gloo process groups are not supported when pg_collection is "
                    "provided. Please set use_gloo_process_groups to False."
                )
            intra_dp_cp_group_gloo = None
            intra_expt_dp_group_gloo = None

        return {
            'dp_group': dp_group,
            'dp_cp_group': dp_cp_group,
            'intra_dp_cp_group': intra_dp_cp_group,
            'expt_dp_group': expt_dp_group,
            'intra_expt_dp_group': intra_expt_dp_group,
            'mp_group': mp_group,
            'expt_tp_pp_group': expt_tp_pp_group,
            'inter_dist_opt_group': inter_dist_opt_group,
            'intra_dp_cp_group_gloo': intra_dp_cp_group_gloo,
            'intra_expt_dp_group_gloo': intra_expt_dp_group_gloo,
        }

    @staticmethod
    def setup_process_groups_for_ddp(
        pg_collection: Optional['ProcessGroupCollection'], config, ddp_config
    ):
        """
        Helper method to set up process groups for DDP with proper validation and fallbacks.

        Args:
            pg_collection: Optional process group collection. If None, uses parallel_state groups.
            config: Model config to extract context_parallel_size from.
            ddp_config: DDP config to extract num_distributed_optimizer_instances from.

        Returns:
            Dictionary containing all required process groups for DDP.
        """
        import logging

        import torch

        from megatron.core import parallel_state
        from megatron.core.utils import log_single_rank

        logger = logging.getLogger(__name__)

        if pg_collection is None:
            # Use parallel_state groups
            return {
                'dp_group': parallel_state.get_data_parallel_group(
                    with_context_parallel=False, partial_data_parallel=False
                ),
                'dp_cp_group': parallel_state.get_data_parallel_group(
                    with_context_parallel=True, partial_data_parallel=False
                ),
                'intra_dp_cp_group': parallel_state.get_data_parallel_group(
                    with_context_parallel=True, partial_data_parallel=True
                ),
                'expt_dp_group': parallel_state.get_expert_data_parallel_group(),
                'intra_expt_dp_group': parallel_state.get_expert_data_parallel_group(
                    partial_expert_data_parallel=True
                ),
                'tp_group': parallel_state.get_tensor_model_parallel_group(),
                'pp_group': parallel_state.get_pipeline_model_parallel_group(),
                'ep_group': parallel_state.get_expert_model_parallel_group(),
                'inter_dist_opt_group': (
                    parallel_state.get_inter_distributed_optimizer_instance_group()
                    if ddp_config.num_distributed_optimizer_instances > 1
                    else None
                ),
            }
        else:
            # Use provided process group collection with validation and fallbacks
            result = {}

            # 1. dp group - this is always required
            if not hasattr(pg_collection, 'dp'):
                raise ValueError("dp process group is required but not provided in pg_collection")
            result['dp_group'] = pg_collection.dp

            # 2. dp_cp group: fallback logic based on context_parallel_size
            if hasattr(pg_collection, 'dp_cp'):
                result['dp_cp_group'] = pg_collection.dp_cp
            else:
                cp_size = getattr(config, 'context_parallel_size', 1)
                if cp_size == 1:
                    # If no context parallelism, dp_cp is same as dp
                    result['dp_cp_group'] = result['dp_group']
                else:
                    raise ValueError(
                        "dp_cp process group is required when context_parallel_size > 1 "
                        "but not provided in pg_collection"
                    )

            # 3. Handle expert data parallel group (DDP-specific: create if missing)
            if hasattr(pg_collection, 'expt_dp') and pg_collection.expt_dp is not None:
                result['expt_dp_group'] = pg_collection.expt_dp
            else:
                # Create a new group with just the current rank for DDP
                log_single_rank(
                    logger,
                    logging.WARNING,
                    "No expert data parallel group provided in pg_collection, "
                    "creating a new one with just the current rank",
                )
                result['expt_dp_group'] = torch.distributed.new_group(
                    ranks=[torch.distributed.get_rank()]
                )

            # 4. Handle intra groups based on optimizer instances
            if ddp_config.num_distributed_optimizer_instances == 1:
                result['intra_dp_cp_group'] = result['dp_cp_group']
                result['intra_expt_dp_group'] = result['expt_dp_group']
                result['inter_dist_opt_group'] = None
            else:
                # With multiple optimizer instances, groups must be provided
                if not (
                    hasattr(pg_collection, 'intra_dp_cp')
                    and hasattr(pg_collection, 'intra_expt_dp')
                    and hasattr(pg_collection, 'inter_dist_opt')
                ):
                    raise ValueError(
                        "intra_dp_cp, intra_expt_dp, and inter_dist_opt "
                        "process groups are required when using multiple optimizer "
                        "instances (>1) but not provided in pg_collection"
                    )
                result['intra_dp_cp_group'] = pg_collection.intra_dp_cp
                result['intra_expt_dp_group'] = pg_collection.intra_expt_dp
                result['inter_dist_opt_group'] = pg_collection.inter_dist_opt

            # 5. Model parallel groups (DDP-specific: tp, pp, ep instead of mp, expt_tp_pp)
            if not all(
                [
                    hasattr(pg_collection, 'tp'),
                    hasattr(pg_collection, 'pp'),
                    hasattr(pg_collection, 'ep'),
                ]
            ):
                raise ValueError(
                    "tp, pp and ep process groups are required but not provided in pg_collection"
                )
            result['tp_group'] = pg_collection.tp
            result['pp_group'] = pg_collection.pp
            result['ep_group'] = pg_collection.ep

            return result
