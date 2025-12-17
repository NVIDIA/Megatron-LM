# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Model and data parallel groups."""

import logging
import os
import warnings
from datetime import timedelta
from math import log2
from typing import Callable, List, Optional

import numpy as np
import torch

from .utils import GlobalMemoryBuffer, GlobalSymmetricMemoryBuffer, is_torch_min_version

logger = logging.getLogger(__name__)

try:
    import einops

    HAVE_EINOPS = True
except ImportError:
    HAVE_EINOPS = False

logger = logging.getLogger(__name__)

# Intra-layer model parallel group that the current rank belongs to.
_TENSOR_MODEL_PARALLEL_GROUP = None
# Inter-layer model parallel group that the current rank belongs to.
_PIPELINE_MODEL_PARALLEL_GROUP = None
# Model parallel group (both intra- and pipeline) that the current rank belongs to.
_MODEL_PARALLEL_GROUP = None
# Model parallel group (both intra-, pipeline, and expert) that the current rank belongs to.
# Embedding group.
_EMBEDDING_GROUP = None
# Position embedding group.
_POSITION_EMBEDDING_GROUP = None
# Data parallel group that the current rank belongs to.
_DATA_PARALLEL_GROUP = None
_DATA_PARALLEL_GROUP_GLOO = None
# tensor model parallel group and data parallel group combined
# used for fp8 and moe training
_TENSOR_AND_DATA_PARALLEL_GROUP = None

### Expert-related parallel states
# Naming convention:
# _EXPERT prefix in group name means it's used for expert layer in MoE models.
# _EXPERT_MODEL denotes expert parallelism which splits number of experts across the group.
# _EXPERT_TENSOR denotes tensor parallelism of expert which splits tensor across the group.
# _EXPERT_DATA denotes data parallelism of expert which replicates weight across the group.

# Expert model parallel group that current rank belongs to.
_EXPERT_MODEL_PARALLEL_GROUP = None
# Expert tensor parallel group that current rank belongs to.
_EXPERT_TENSOR_PARALLEL_GROUP = None
# Expert tensor and model combined parallel group
_EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP = None
# Expert tensor, model, pipeline combined parallel group
_EXPERT_TENSOR_MODEL_PIPELINE_PARALLEL_GROUP = None
# Expert data parallel group
_EXPERT_DATA_PARALLEL_GROUP = None
_EXPERT_DATA_PARALLEL_GROUP_GLOO = None
_INTRA_PARTIAL_EXPERT_DATA_PARALLEL_GROUP = None
_INTRA_PARTIAL_EXPERT_DATA_PARALLEL_GROUP_GLOO = None
_INTER_PARTIAL_EXPERT_DATA_PARALLEL_GROUP = None
# Parallel state values changed on the fly
_MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE = None
_MPU_EXPERT_MODEL_PARALLEL_RANK = None
_MPU_EXPERT_TENSOR_PARALLEL_WORLD_SIZE = None
_MPU_EXPERT_TENSOR_PARALLEL_RANK = None
### End of expert related parallel states

_VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = None
_VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None

# These values enable us to change the mpu sizes on the fly.
_MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = None
_MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None
_MPU_DATA_PARALLEL_WORLD_SIZE = None
_MPU_DATA_PARALLEL_RANK = None
_MPU_TENSOR_MODEL_PARALLEL_RANK = None
_MPU_PIPELINE_MODEL_PARALLEL_RANK = None

# A list of ranks that have a copy of the embedding.
_EMBEDDING_GLOBAL_RANKS = None

# A list of ranks that have a copy of the position embedding.
_POSITION_EMBEDDING_GLOBAL_RANKS = None

# A list of global ranks for each pipeline group to ease calculation of the source
# rank when broadcasting from the first or last pipeline stage.
_PIPELINE_GLOBAL_RANKS = None

# A list of global ranks for each data parallel group to ease calculation of the source
# rank when broadcasting weights from src to all other data parallel ranks
_DATA_PARALLEL_GLOBAL_RANKS = None

# A list of global ranks for each tensor model parallel group to ease calculation of
# the first local rank in the tensor model parallel group
_TENSOR_MODEL_PARALLEL_GLOBAL_RANKS = None

# A list of global ranks for each expert model parallel group to ease calculation of
# the first local rank in the expert model parallel group
_EXPERT_MODEL_PARALLEL_RANKS = None

# A list of global ranks for each model parallel group to ease calculation of
# the first local rank in the model parallel group
_MODEL_PARALLEL_GLOBAL_RANKS = None

# Context parallel group that the current rank belongs to
_CONTEXT_PARALLEL_GROUP = None
# A list of global ranks for each context parallel group to ease calculation of the
# destination rank when exchanging KV/dKV between context parallel_ranks
_CONTEXT_PARALLEL_GLOBAL_RANKS = None
# Hierarchical context parallel groups
_HIERARCHICAL_CONTEXT_PARALLEL_GROUPS = None
# Hybrid context parallel groups
_HYBRID_DP_CP_GROUPS = {}

# Data parallel group information with context parallel combined.
_DATA_PARALLEL_GROUP_WITH_CP = None
_DATA_PARALLEL_GROUP_WITH_CP_GLOO = None
_DATA_PARALLEL_GLOBAL_RANKS_WITH_CP = None

# Partial Data parallel group information with context parallel combined.
_INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP = None
_INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP_GLOO = None

# combined parallel group of TP and CP
_TENSOR_AND_CONTEXT_PARALLEL_GROUP = None

# combined parallel group of TP, DP, and CP used for fp8
_TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP = None

# Paralel group of all GPUs in a distributed optimizer instance
_INTRA_DISTRIBUTED_OPTIMIZER_INSTANCE_GROUP = None

# Memory buffers to avoid dynamic memory allocation
_GLOBAL_MEMORY_BUFFER = None

# Global symmetric memory buffer for inference
_GLOBAL_SYMMETRIC_MEMORY_BUFFER = None

# List of all process groups
# Used for updating the timeout for all process groups
# None represents the default process group
_global_process_group_list = None


def get_nccl_options(pg_name, nccl_comm_cfgs):
    """Set the NCCL process group options.

    Args:
        pg_name (str): process group name
        nccl_comm_cfgs (dict): nccl communicator configurations
    When an option (e.g., max_ctas) is not found in the config, use the NCCL default setting.
    """
    if pg_name in nccl_comm_cfgs:
        # When fields in nccl_options.config are not specified, NCCL applies default settings.
        # The default values for Hopper GPUs are as follows:
        # cga_cluster_size = 4, max_ctas = 32, min_ctas = 1
        # Default values may differ between GPU generations and NCCL versions.
        nccl_options = torch.distributed.ProcessGroupNCCL.Options(
            is_high_priority_stream=nccl_comm_cfgs[pg_name].get("is_high_priority_stream", False)
        )
        if "cga_cluster_size" in nccl_comm_cfgs[pg_name]:
            nccl_options.config.cga_cluster_size = nccl_comm_cfgs[pg_name]["cga_cluster_size"]
        if "max_ctas" in nccl_comm_cfgs[pg_name]:
            nccl_options.config.max_ctas = nccl_comm_cfgs[pg_name]["max_ctas"]
        if "min_ctas" in nccl_comm_cfgs[pg_name]:
            nccl_options.config.min_ctas = nccl_comm_cfgs[pg_name]["min_ctas"]
        if "net_name" in nccl_comm_cfgs[pg_name]:
            nccl_options.config.net_name = nccl_comm_cfgs[pg_name]["net_name"]
            # verify net_name value
            if nccl_options.config.net_name.lower() not in ["ib", "socket"]:
                raise RuntimeError(
                    f"net_name ({nccl_options.config.net_name}) is not supported."
                    f"Accepted values: 'IB' or 'socket'."
                )
        return nccl_options
    else:
        return None


def update_pg_timeout(
    timeout: timedelta, pg: Optional[torch._C._distributed_c10d.ProcessGroup] = None
):
    """Update the timeout for all process groups or a specific process group.
       Synchronize the process groups before updating the timeout.
    Args:
        timeout(datetime.timedelta): The timeout to set for the process group(s)
        pg(Optional[torch._C._distributed_c10d.ProcessGroup], default=None):
            The process group to update the timeout for.
            If None, all process groups are updated.
    """
    if hasattr(torch.distributed.distributed_c10d, "_set_pg_timeout"):
        torch.distributed.barrier(pg)
        torch.cuda.synchronize()
        try:
            if pg is None:
                global _global_process_group_list
                for group in _global_process_group_list:
                    torch.distributed.distributed_c10d._set_pg_timeout(timeout, group)
            else:
                torch.distributed.distributed_c10d._set_pg_timeout(timeout, pg)
        except Exception as e:
            logger.error(f"Error updating pg timeout: {e}")
            logger.error(f"Process group: {pg}")
            logger.error(f"Timeout: {timeout}")
            logger.error(f"Global process group list: {_global_process_group_list}")
            raise e


def create_group(
    ranks=None,
    timeout=None,
    backend=None,
    pg_options=None,
    use_local_synchronization=False,
    group_desc=None,
):
    """Creates a ProcessGroup."""
    kwargs = {
        "ranks": ranks,
        "timeout": timeout,
        "backend": backend,
        "pg_options": pg_options,
        "use_local_synchronization": use_local_synchronization,
        "group_desc": group_desc,
    }
    if not is_torch_min_version("2.4.0"):
        kwargs.pop("group_desc")
        if timeout is None:
            # Old version (e.g. v2.1.2) sets default_pg_timeout as default value to timeout
            # in function signature, then check tiemout value type.
            # New version sets None as default value to timeout in function signature. If value
            # is None, torch will give value according to the backend, then check type.
            # So need to unset timeout here if caller doesn't set value. Otherwise there is
            # type error.
            kwargs.pop("timeout")
    group = torch.distributed.new_group(**kwargs)
    global _global_process_group_list
    if _global_process_group_list is None:
        # None stands for the default process group
        _global_process_group_list = [None]
    if torch.distributed.get_rank() in ranks:
        _global_process_group_list.append(group)
    return group


def generate_masked_orthogonal_rank_groups(
    world_size: int, parallel_size: List[int], mask: List[bool]
) -> List[List[int]]:
    r"""Generate orthogonal parallel groups based on the parallel size and mask.

    Arguments:
        world_size (int): world size

        parallel_size (List[int]):
            The parallel size of each orthogonal parallel type. For example, if
            tensor_parallel_size = 2, pipeline_model_parallel_group = 3, data_parallel_size = 4,
            and the parallel mapping order is tp-pp-dp, then the parallel_size = [2, 3, 4].

        mask (List[bool]):
            The mask controls which parallel methods the generated groups represent. If mask[i] is
            True, it means the generated group contains the i-th parallelism method. For example,
            if parallel_size = [tp_size, pp_size, dp_size], and mask = [True, False , True], then
            the generated group is the `tp-dp` group, if the mask = [False, True, False], then the
            generated group is the `pp` group.

    Algorithm:
        For orthogonal parallelism, such as tp/dp/pp/cp, the global_rank and
        local_rank satisfy the following equation:
            global_rank = tp_rank + dp_rank * tp_size + pp_rank * tp_size * dp_size (1)
                tp_rank \in [0, tp_size)
                dp_rank \in [0, dp_size)
                pp_rank \in [0, pp_size)

        If we want to get the `dp_group` (tp_size * pp_size groups of dp_size ranks each.
        For example,  if the gpu size is 8 and order is 'tp-pp-dp', size is '2-2-2', and the
        dp_group here is [[0, 4], [1, 5], [2, 6], [3, 7]].)
        The tp_rank and pp_rank will be combined to form the `dp_group_index`.
            dp_group_index = tp_rank + pp_rank * tp_size (2)

        So, Given that tp_rank and pp_rank satisfy equation (2), and dp_rank in
        range(0, dp_size), the ranks in dp_group[dp_group_index] satisfies the
        equation (1).

        This function solve this math problem.

    For example, if the parallel_size = [tp_size, dp_size, pp_size] = [2, 3, 4],
    and the mask = [False, True, False]. Then,
        dp_group_index(0) = tp_rank(0) + pp_rank(0) * 2
        dp_group_index(1) = tp_rank(1) + pp_rank(0) * 2
        ...
        dp_group_index(7) = tp_rank(1) + pp_rank(3) * 2

        dp_group[0] = 0 + range(0, 3) * 2 + 0 = [0, 2, 4]
        dp_group[1] = 1 + range(0, 3) * 2 + 0 = [1, 3, 5]
        ...
        dp_group[7] = 1 + range(0, 3) * 2 + 3 * 2 * 3 = [19, 21, 23]
    """

    def prefix_product(a: List[int], init=1) -> List[int]:
        r = [init]
        for v in a:
            init = init * v
            r.append(init)
        return r

    def inner_product(a: List[int], b: List[int]) -> int:
        return sum([x * y for x, y in zip(a, b)])

    def decompose(index, shape, stride=None):
        """
        This function solve the math problem below:
            There is an equation:
                index = sum(idx[i] * stride[i])
            And given the value of index, stride.
            Return the idx.
        This function will be used to get the pp/dp/pp_rank
        from group_index and rank_in_group.
        """
        if stride is None:
            stride = prefix_product(shape)
        idx = [(index // d) % s for s, d in zip(shape, stride)]
        # stride is a prefix_product result. And the value of stride[-1]
        # is not used.
        assert (
            sum([x * y for x, y in zip(idx, stride[:-1])]) == index
        ), "idx {} with shape {} mismatch the return idx {}".format(index, shape, idx)
        return idx

    masked_shape = [s for s, m in zip(parallel_size, mask) if m]
    unmasked_shape = [s for s, m in zip(parallel_size, mask) if not m]

    global_stride = prefix_product(parallel_size)
    masked_stride = [d for d, m in zip(global_stride, mask) if m]
    unmasked_stride = [d for d, m in zip(global_stride, mask) if not m]

    group_size = prefix_product(masked_shape)[-1]
    num_of_group = world_size // group_size

    ranks = []
    for group_index in range(num_of_group):
        # get indices from unmaksed for group_index.
        decomposed_group_idx = decompose(group_index, unmasked_shape)
        rank = []
        for rank_in_group in range(group_size):
            # get indices from masked for rank_in_group.
            decomposed_rank_idx = decompose(rank_in_group, masked_shape)
            rank.append(
                inner_product(decomposed_rank_idx, masked_stride)
                + inner_product(decomposed_group_idx, unmasked_stride)
            )
        ranks.append(rank)
    return ranks


def create_hierarchical_groups(
    rank,
    ranks,
    hierarchical_group_sizes,
    create_gloo_process_groups=False,
    pg_options=None,
    timeout=None,
    group_desc=None,
):
    """Create hierarchical groups for a set of ranks.
    Taking a group size of 16 as example, so we have a total of 16 GPUs denoted by g0 ... g15.
    If the hierarchical group sizes are [2,2,4], we use 2 GPUs in the first and second level
    of sub-groups, and 4 GPUs in the last level of sub groups. The present function will
    create 8 level-1 sub-groups, 8 level-2 sub-groups and 4 level-3 sub-groups as:
        8 level-1 sub-groups:
            [g0, g1], [g2, g3], [g4, g5], [g6, g7], [g8, g9], [g10, g11], [g12, g13], [g14, g15]
        8 level-2 sub-groups:
            [g0, g2], [g1, g3], [g4, g6], [g5, g7], [g8, g10], [g9, g11], [g12, g14], [g13, g15]
        4 level-3 sub-groups:
            [g0, g4, g8, g12], [g1, g5, g9, g13], [g2, g6, g10, g14], [g3, g7, g11, g15]
    """

    if not HAVE_EINOPS:
        raise ImportError("einops is not installed. Please install it with `pip install einops`.")

    hierarchical_groups = []
    hierarchical_groups_gloo = []
    if not isinstance(pg_options, list):
        pg_options = [pg_options] * len(hierarchical_group_sizes)
    for level in range(len(hierarchical_group_sizes)):
        rearranged_ranks = einops.rearrange(
            np.array(ranks),
            "(l s u) -> (l u) s",
            u=int(np.prod(hierarchical_group_sizes[:level])),
            s=hierarchical_group_sizes[level],
            l=int(np.prod(hierarchical_group_sizes[level + 1 :])),
        ).tolist()
        for sub_ranks in rearranged_ranks:
            sub_group = create_group(
                sub_ranks,
                timeout=timeout,
                pg_options=pg_options[level],
                group_desc=f"HIERARCHICAL_{group_desc}_L{level}",
            )
            if create_gloo_process_groups:
                sub_group_gloo = create_group(
                    sub_ranks,
                    timeout=timeout,
                    backend="gloo",
                    pg_options=pg_options[level],
                    group_desc=f"HIERARCHICAL_{group_desc}_GLOO_L{level}",
                )
            else:
                sub_group_gloo = None
            if rank in sub_ranks:
                hierarchical_groups.append(sub_group)
                hierarchical_groups_gloo.append(sub_group_gloo)
    assert rank not in ranks or len(hierarchical_groups) == len(hierarchical_group_sizes)
    assert rank not in ranks or len(hierarchical_groups_gloo) == len(hierarchical_group_sizes)
    return hierarchical_groups, hierarchical_groups_gloo


def create_hybrid_dp_cp_groups(rank, ranks, pg_options):
    """
    Creates groups required for hybrid DPxCP.
    Creates a new group for every power of 2 up to the number of DPxCP ranks.
    Returns a dictionary indexed by group size.
    """
    hybrid_dp_cp_groups = {}
    # Generate group for every power of 2 up to the number of CP ranks
    # We limit the allowed group sizes in order to avoid excessive overhead.
    group_sizes = [2**i for i in range(int(log2(len(ranks))))][1:]
    for group_size in group_sizes:
        for i in range(0, len(ranks), group_size):
            group = create_group(
                ranks[i : i + group_size],
                pg_options=pg_options,
                group_desc=f"HYBRID_DP_CP_GROUP_{group_size}",
            )
            if rank in ranks[i : i + group_size]:
                assert (
                    group_size not in hybrid_dp_cp_groups
                ), f"Rank {rank} appears in multiple Hybrid DP CP groups of size {group_size}"
                hybrid_dp_cp_groups[group_size] = group
    return hybrid_dp_cp_groups


class RankGenerator(object):
    """A class for generating rank groups for different modes of parallelism."""

    def __init__(
        self, tp: int, ep: int, dp: int, pp: int, cp: int, order: str, rank_offset: int = 0
    ) -> None:
        assert (
            ep == 1 or cp == 1
        ), "Both EP and CP > 1 in not allow in one rank generator. \
            CP is only included in default RankGenerator, and EP only in expert RankGenerator."

        self.tp = tp
        self.ep = ep
        self.dp = dp
        self.pp = pp
        self.cp = cp
        self.rank_offset = rank_offset
        self.world_size = tp * dp * pp * cp * ep

        self.name_to_size = {
            "tp": self.tp,
            "pp": self.pp,
            "dp": self.dp,
            "ep": self.ep,
            "cp": self.cp,
        }
        self.order = order
        order = order.lower()

        for name in self.name_to_size.keys():
            if name not in order and self.name_to_size[name] != 1:
                raise RuntimeError(
                    f"The size of ({name}) is ({self.name_to_size[name]}), but you haven't"
                    f"specified the order ({self.order})."
                )
            elif name not in order:
                order = order + "-" + name

        self.order = order
        self.ordered_size = []

        for token in order.split("-"):
            self.ordered_size.append(self.name_to_size[token])

    def get_mask(self, order: str, token: str):
        """Create a mask for the specified tokens based on the given order.

        Args:
            order (str): The order of parallelism types (e.g., 'tp-dp-pp').
            token (str): The specific parallelism types to include in the mask,
                         separated by hyphens (e.g., 'tp-dp').
        """
        ordered_token = order.split("-")
        token_list = token.split("-")
        mask = [False] * len(ordered_token)
        for t in token_list:
            mask[ordered_token.index(t)] = True
        return mask

    def get_ranks(self, token):
        """Get rank group by input token.

        Args:
            token (str):
                Specify the ranks type that want to get. If we want
                to obtain multiple parallel types, we can use a hyphen
                '-' to separate them. For example, if we want to obtain
                the TP_DP group, the token should be 'tp-dp'.
        """
        mask = self.get_mask(self.order, token)
        ranks = generate_masked_orthogonal_rank_groups(self.world_size, self.ordered_size, mask)
        if self.rank_offset > 0:
            for rank_group in ranks:
                for i in range(len(rank_group)):
                    rank_group[i] += self.rank_offset
        return ranks


def default_embedding_ranks(pp_ranks):
    """Return the default ranks that constitute the stages on which the word embeddings live.
    For most models, these are the first and last pipeline stages."""
    if len(pp_ranks) == 1:
        return [pp_ranks[0]]
    else:
        return [pp_ranks[0], pp_ranks[-1]]


def default_position_embedding_ranks(pp_ranks):
    """Return the default ranks that constitute the stages on which the position embeddings live.
    For most models, this is only the first pipeline stage."""
    return [pp_ranks[0]]


def overwrite_nccl_comm_cfgs(nccl_comm_cfgs, pg_name, key_value_pair):
    """Overwrite the nccl_comm_cfgs for the given pg_name with the given key_value_pair."""
    if pg_name not in nccl_comm_cfgs:
        nccl_comm_cfgs[pg_name] = {}
    nccl_comm_cfgs[pg_name][key_value_pair[0]] = key_value_pair[1]


# pylint: disable=C0301
def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    virtual_pipeline_model_parallel_size: Optional[int] = None,
    pipeline_model_parallel_comm_backend: Optional[str] = None,
    use_sharp: bool = False,
    context_parallel_size: int = 1,
    hierarchical_context_parallel_sizes: Optional[List[int]] = None,
    expert_model_parallel_size: int = 1,
    num_distributed_optimizer_instances: int = 1,
    expert_tensor_parallel_size: Optional[int] = None,
    nccl_communicator_config_path: Optional[str] = None,
    distributed_timeout_minutes: int = 30,
    order: str = "tp-cp-ep-dp-pp",
    get_embedding_ranks: Optional[Callable[[List[int], Optional[int]], List[int]]] = None,
    get_position_embedding_ranks: Optional[Callable[[List[int], Optional[int]], List[int]]] = None,
    create_gloo_process_groups: bool = True,
    high_priority_stream_groups: Optional[List[str]] = None,
    sharp_enabled_group: Optional[str] = None,
    hybrid_context_parallel: bool = False,
) -> None:
    """Initialize model data parallel groups.

    Args:
        tensor_model_parallel_size (int, default = 1):
            The number of GPUs to split individual tensors across.

        pipeline_model_parallel_size (int, default = 1):
            The number of tensor parallel GPU groups to split the
            Transformer layers across. For example, if
            tensor_model_parallel_size is 4 and
            pipeline_model_parallel_size is 2, the model will be split
            into 2 groups of 4 GPUs.

        virtual_pipeline_model_parallel_size (int, optional):
            The number of stages that each pipeline group will have,
            interleaving as necessary. If None, no interleaving is
            performed. For example, if tensor_model_parallel_size is 1,
            pipeline_model_parallel_size is 4,
            virtual_pipeline_model_parallel_size is 2, and there are
            16 transformer layers in the model, the model will be
            split into 8 stages with two layers each and each GPU
            would get 2 stages as such (layer number starting with 1):

            GPU 0: [1, 2] [9, 10]
            GPU 1: [3, 4] [11, 12]
            GPU 2: [5, 6] [13, 14]
            GPU 3: [7, 8] [15, 16]

        pipeline_model_parallel_comm_backend (str, optional):
            The backend to use for pipeline parallel communication.
            If None, the default backend will be used.

        use_sharp (bool, default = False):
            Set the use of SHARP for the collective communications of
            data-parallel process groups. When `True`, run barrier
            within each data-parallel process group, which specifies
            the SHARP application target groups.

        context_parallel_size (int, default = 1):
            The number of tensor parallel GPU groups to split the
            network input sequence length across. Compute of attention
            module requires tokens of full sequence length, so GPUs
            in a context parallel group need to communicate with each
            other to exchange information of other sequence chunks.
            Each GPU and its counterparts in other tensor parallel
            groups compose a context parallel group.

            For example, assume we have 8 GPUs, if tensor model parallel
            size is 4 and context parallel size is 2, the network input
            will be split into two sequence chunks, which are processed
            by 2 different groups of 4 GPUs. One chunk is processed by
            GPU0-3, the other chunk is processed by GPU4-7. Four groups
            are build to do context parallel communications: [GPU0, GPU4],
            [GPU1, GPU5], [GPU2, GPU6], and [GPU3, GPU7].

            Context parallelism partitions sequence length, so it has no
            impact on weights, which means weights are duplicated among
            GPUs in a context parallel group. Hence, weight gradients
            all-reduce is required in backward. For simplicity, we piggyback
            GPUs of context parallelism on data parallel group for
            weight gradient all-reduce.

        expert_model_parallel_size (int, default = 1):
            The number of Mixture of Experts parallel GPUs in each expert
            parallel group.

        num_distributed_optimizer_instances (int, default = 1):
            The number of distributed optimizer replicas across the data-
            parallel domain.

        expert_tensor_parallel_size (int, default = tp_size):
            The number of GPUs to split individual tensors of expert.

        nccl_communicator_config_path (str, default = None):
            Path to the yaml file of NCCL communicator configurations.
            `min_ctas`, `max_ctas`, and `cga_cluster_size` can be set
            for each communicator.

        distributed_timeout_minutes (int, default = 30): Timeout, in
            minutes,for operations executed against distributed
            process groups. See PyTorch documentation at
            https://pytorch.org/docs/stable/distributed.html for
            caveats.

        order (str, default=tp-dp-pp):
            The rank initialization order of parallelism. Now we support
            tp-dp-pp and tp-pp-dp orders.

        get_embedding_ranks (Callable[[List[int], Optional[int]], List[int]], optional, default=None):
            A function that takes in a list of ranks for a pipeline group and returns
            those ranks that should have embeddings.

        get_position_embedding_ranks (Callable[[List[int], Optional[int]], List[int]], optional, default=None):
            A function that takes in a list of ranks for a pipeline group, and returns
            those ranks that should have position embeddings.

        create_gloo_process_groups (bool, default = True):
            Create Gloo process groups if set to True. If set to False, Gloo process groups are
            not created and calls to get Gloo process groups will result in assertion errors.

        high_priority_stream_groups (List[str], default = None):
            Specify which communicator groups should use high priority streams during creation.
            Assigning high priority to communication streams ensures that communication kernels
            are scheduled with higher priority, minimizing the exposed communication when it is
            overlapped with other computation kernels.
            Example: initialize_parallel_groups(..., high_priority_stream_groups=['dp_cp','ep_dp'])

        sharp_enabled_group (str, default = None):
            Specify which communicator group should use SHARP communication.
            This option is only valid when use_sharp is True.
            By default (None), it is enabled from dp group.
            Available options (choose one): [dp, dp_replica]

    Let's say we have a total of 16 GPUs denoted by g0 ... g15 and we
    use 2 GPUs to parallelize the model tensor, and 4 GPUs to parallelize
    the model pipeline. The present function will
    create 8 tensor model-parallel groups, 4 pipeline model-parallel groups
    and 8 data-parallel groups as:
        8 data_parallel groups:
            [g0, g2], [g1, g3], [g4, g6], [g5, g7], [g8, g10], [g9, g11], [g12, g14], [g13, g15]
        8 tensor model-parallel groups:
            [g0, g1], [g2, g3], [g4, g5], [g6, g7], [g8, g9], [g10, g11], [g12, g13], [g14, g15]
        4 pipeline model-parallel groups:
            [g0, g4, g8, g12], [g1, g5, g9, g13], [g2, g6, g10, g14], [g3, g7, g11, g15]
    Note that for efficiency, the caller should make sure adjacent ranks
    are on the same DGX box. For example if we are using 2 DGX-1 boxes
    with a total of 16 GPUs, rank 0 to 7 belong to the first box and
    ranks 8 to 15 belong to the second box.
    """
    # NCCL restricts IB SHARP usage to a single communicator group—the first one created
    # with NCCL_COLLNET_ENABLE=1. After this group is created, NCCL_COLLNET_ENABLE must be
    # set to 0 for subsequent groups.
    if "NCCL_COLLNET_ENABLE" in os.environ:
        del os.environ["NCCL_COLLNET_ENABLE"]

    if use_sharp:
        if sharp_enabled_group is None:
            # By default, SHARP is enabled from dp group.
            sharp_enabled_group = "dp"
        else:
            # Currently, only dp and dp_replica groups are supported for SHARP.
            assert sharp_enabled_group in ["dp", "dp_replica"], "Invalid sharp_enabled_group"
            if sharp_enabled_group == "dp_replica":
                assert (
                    num_distributed_optimizer_instances > 1
                ), "dp_replica group requires num_distributed_optimizer_instances > 1"
    else:
        assert (
            sharp_enabled_group is None
        ), "sharp_enabled_group is only valid when use_sharp is True"

    if get_embedding_ranks is None:
        get_embedding_ranks = default_embedding_ranks

    if get_position_embedding_ranks is None:
        get_position_embedding_ranks = default_position_embedding_ranks

    # Get world size and rank. Ensure some consistencies.
    assert torch.distributed.is_initialized()
    world_size: int = torch.distributed.get_world_size()

    model_size = tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size

    if world_size % model_size != 0:
        raise RuntimeError(f"world_size ({world_size}) is not divisible by {model_size}")

    data_parallel_size: int = world_size // model_size

    if virtual_pipeline_model_parallel_size is not None:
        if not pipeline_model_parallel_size > 1:
            raise RuntimeError(
                "pipeline-model-parallel size should be greater than 1 with interleaved schedule"
            )
        global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
        global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
        _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = 0
        _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = virtual_pipeline_model_parallel_size

    rank = torch.distributed.get_rank()

    nccl_comm_cfgs = {}
    if nccl_communicator_config_path is not None:
        try:
            import yaml
        except ImportError:
            raise RuntimeError(
                "Cannot import `yaml`. Setting custom nccl communicator configs "
                "requires the yaml package."
            )

        with open(nccl_communicator_config_path, "r") as stream:
            nccl_comm_cfgs = yaml.safe_load(stream)

    # Set is_high_priority_stream flag to the nccl_comm_cfgs if it is in high_priority_stream_groups
    high_priority_stream_groups = high_priority_stream_groups or []
    for pg_name in high_priority_stream_groups:
        overwrite_nccl_comm_cfgs(nccl_comm_cfgs, pg_name, ("is_high_priority_stream", True))

    decoder_rank_generator = RankGenerator(
        tp=tensor_model_parallel_size,
        ep=1,
        dp=data_parallel_size,
        pp=pipeline_model_parallel_size,
        cp=context_parallel_size,
        order=order,
        rank_offset=0,
    )

    # Build expert rank generator
    if expert_tensor_parallel_size is None:
        expert_tensor_parallel_size = tensor_model_parallel_size
    expert_tensor_model_pipeline_parallel_size = (
        expert_tensor_parallel_size * expert_model_parallel_size * pipeline_model_parallel_size
    )
    expert_data_parallel_size = world_size // expert_tensor_model_pipeline_parallel_size
    if world_size % expert_tensor_model_pipeline_parallel_size != 0:
        raise RuntimeError(
            f"world_size ({world_size}) is not divisible by expert_tensor_model_pipeline_parallel size ({expert_tensor_model_pipeline_parallel_size})"
        )

    # TODO: support expert specific ordering
    expert_decoder_rank_generator = RankGenerator(
        tp=expert_tensor_parallel_size,
        ep=expert_model_parallel_size,
        dp=expert_data_parallel_size,
        pp=pipeline_model_parallel_size,
        cp=1,
        order=order,
        rank_offset=0,
    )

    assert (
        order.endswith("pp")
        or pipeline_model_parallel_size == 1
        or expert_data_parallel_size == data_parallel_size
    ), "When not using pp-last rank ordering, the data parallel size of the attention and moe layers must be the same"

    assert decoder_rank_generator.get_ranks("pp") == expert_decoder_rank_generator.get_ranks(
        "pp"
    ), f"Pipeline parallel groups are expected to be the same for Non-Expert and Expert part, \
    but got {decoder_rank_generator.get_ranks('pp')} and {expert_decoder_rank_generator.get_ranks('pp')}"

    timeout = timedelta(minutes=distributed_timeout_minutes)

    # Build the data-parallel groups.
    global _DATA_PARALLEL_GROUP
    global _DATA_PARALLEL_GROUP_GLOO
    global _DATA_PARALLEL_GLOBAL_RANKS
    global _DATA_PARALLEL_GROUP_WITH_CP
    global _DATA_PARALLEL_GROUP_WITH_CP_GLOO
    global _DATA_PARALLEL_GLOBAL_RANKS_WITH_CP
    global _INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP
    global _INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP_GLOO
    assert _DATA_PARALLEL_GROUP is None, "data parallel group is already initialized"

    assert (
        data_parallel_size * context_parallel_size
    ) % num_distributed_optimizer_instances == 0, (
        "Data parallel size should be divisible by partial DistOpt shard factor"
    )
    intra_partial_data_parallel_size = (
        data_parallel_size * context_parallel_size
    ) // num_distributed_optimizer_instances

    # Set NCCL_COLLNET_ENABLE to 1 to enable SHARP for the dp group.
    if sharp_enabled_group == "dp":
        os.environ["NCCL_COLLNET_ENABLE"] = "1"

    # In case of using SHARP, the dp-cp group requires to use NCCL COLLNET feature.
    # Due to the hardware limitation, only the initially created communication group
    # is eligible for using the NCCL COLLNET feature.
    # Therefore, dp-cp group, which potentially requires SHARP-enablement,
    # need to be created before all the other groups
    for ranks_with_cp in decoder_rank_generator.get_ranks('dp-cp'):
        group_with_cp = create_group(
            ranks_with_cp,
            timeout=timeout,
            pg_options=get_nccl_options("dp_cp", nccl_comm_cfgs),
            group_desc="DATA_PARALLEL_GROUP_WITH_CP",
        )
        if create_gloo_process_groups:
            group_with_cp_gloo = create_group(
                ranks_with_cp,
                timeout=timeout,
                backend="gloo",
                group_desc="DATA_PARALLEL_GROUP_WITH_CP_GLOO",
            )
        else:
            group_with_cp_gloo = None
        if rank in ranks_with_cp:
            _DATA_PARALLEL_GROUP_WITH_CP = group_with_cp
            _DATA_PARALLEL_GROUP_WITH_CP_GLOO = group_with_cp_gloo
            _DATA_PARALLEL_GLOBAL_RANKS_WITH_CP = ranks_with_cp

        if num_distributed_optimizer_instances > 1:
            # Create groups for intra-partial DP domain
            for i in range(num_distributed_optimizer_instances):
                intra_partial_dp_ranks_with_cp = ranks_with_cp[
                    (i * intra_partial_data_parallel_size) : (
                        (i + 1) * intra_partial_data_parallel_size
                    )
                ]
                intra_partial_dp_group_with_cp = create_group(
                    intra_partial_dp_ranks_with_cp,
                    timeout=timeout,
                    pg_options=get_nccl_options("intra_dp_cp", nccl_comm_cfgs),
                    group_desc="INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP",
                )
                if create_gloo_process_groups:
                    intra_partial_dp_group_with_cp_gloo = create_group(
                        intra_partial_dp_ranks_with_cp,
                        timeout=timeout,
                        backend="gloo",
                        group_desc="INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP_GLOO",
                    )
                else:
                    intra_partial_dp_group_with_cp_gloo = None
                if rank in intra_partial_dp_ranks_with_cp:
                    _INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP = intra_partial_dp_group_with_cp
                    _INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP_GLOO = (
                        intra_partial_dp_group_with_cp_gloo
                    )
        else:
            _INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP = _DATA_PARALLEL_GROUP_WITH_CP
            _INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP_GLOO = _DATA_PARALLEL_GROUP_WITH_CP_GLOO

    # Apply SHARP to the dp group.
    if sharp_enabled_group == "dp":
        if rank == 0:
            logger.info(
                "The number of process groups to use SHARP with depends on the type "
                "of the network switch. Nvidia QM1 switch supports SAHRP up to 8 "
                "process groups and QM2 supports up to 256 process groups. We apply "
                "SHARP to the communications of the data-parallel domain. If the "
                "number of data-parallel process groups is larger than the max "
                "process groups that the network switch supports, the communication "
                "will fall back to non-SHARP operators. To enable SHARP, "
                "`#SBATCH_NETWORK=sharp` should be set in the sbatch script."
            )
        # PyTorch is performing lazy initialization of the communicator group.
        # Therefore, we need to perform a nccl call to ensure that the communicator group is created.
        torch.distributed.barrier(
            group=get_data_parallel_group(with_context_parallel=True),
            device_ids=[torch.cuda.current_device()],
        )
        torch.cuda.synchronize()
        # Set `NCCL_COLLNET_ENABLE=0` to restrict SHARP application to the dp group.
        if "NCCL_COLLNET_ENABLE" in os.environ:
            del os.environ["NCCL_COLLNET_ENABLE"]

    if hybrid_context_parallel:
        global _HYBRID_DP_CP_GROUPS
        for ranks_with_cp in decoder_rank_generator.get_ranks('dp-cp'):
            assert (
                len(ranks_with_cp) % 2 == 0
            ), "Hybrid context parallel requires an even number of ranks"
            _HYBRID_DP_CP_GROUPS.update(
                create_hybrid_dp_cp_groups(
                    rank, ranks_with_cp, get_nccl_options("dp_cp", nccl_comm_cfgs)
                )
            )
        # TODO: Are gloo groups needed for hybrid cp?

    for ranks in decoder_rank_generator.get_ranks('dp'):
        group = create_group(
            ranks,
            timeout=timeout,
            pg_options=get_nccl_options("dp", nccl_comm_cfgs),
            group_desc="DATA_PARALLEL_GROUP",
        )
        if create_gloo_process_groups:
            group_gloo = create_group(
                ranks, timeout=timeout, backend="gloo", group_desc="DATA_PARALLEL_GROUP_GLOO"
            )
        else:
            group_gloo = None
        if rank in ranks:
            _DATA_PARALLEL_GROUP = group
            _DATA_PARALLEL_GROUP_GLOO = group_gloo
            _DATA_PARALLEL_GLOBAL_RANKS = ranks

    # Build the context-parallel groups.
    global _CONTEXT_PARALLEL_GROUP
    global _CONTEXT_PARALLEL_GLOBAL_RANKS
    assert _CONTEXT_PARALLEL_GROUP is None, 'context parallel group is already initialized'
    for ranks in decoder_rank_generator.get_ranks('cp'):
        group = create_group(
            ranks,
            timeout=timeout,
            pg_options=get_nccl_options("cp", nccl_comm_cfgs),
            group_desc="CONTEXT_PARALLEL_GROUP",
        )
        if rank in ranks:
            _CONTEXT_PARALLEL_GROUP = group
            _CONTEXT_PARALLEL_GLOBAL_RANKS = ranks
        if hierarchical_context_parallel_sizes:
            assert np.prod(hierarchical_context_parallel_sizes) == context_parallel_size
            global _HIERARCHICAL_CONTEXT_PARALLEL_GROUPS
            hierarchical_groups, _ = create_hierarchical_groups(
                rank,
                ranks,
                hierarchical_context_parallel_sizes,
                create_gloo_process_groups=False,
                pg_options=get_nccl_options("hcp", nccl_comm_cfgs),
                timeout=timeout,
                group_desc="CONTEXT_PARALLEL_GROUP",
            )
            if rank in ranks:
                _HIERARCHICAL_CONTEXT_PARALLEL_GROUPS = hierarchical_groups

    # Build the model-parallel groups.
    global _MODEL_PARALLEL_GROUP
    global _MODEL_PARALLEL_GLOBAL_RANKS
    assert _MODEL_PARALLEL_GROUP is None, 'model parallel group is already initialized'
    for ranks in decoder_rank_generator.get_ranks('tp-pp'):
        group = create_group(
            ranks,
            timeout=timeout,
            pg_options=get_nccl_options("mp", nccl_comm_cfgs),
            group_desc="MODEL_PARALLEL_GROUP",
        )
        if rank in ranks:
            _MODEL_PARALLEL_GROUP = group
            _MODEL_PARALLEL_GLOBAL_RANKS = ranks

    # Build the tensor model-parallel groups.
    global _TENSOR_MODEL_PARALLEL_GROUP
    global _TENSOR_MODEL_PARALLEL_GLOBAL_RANKS
    assert (
        _TENSOR_MODEL_PARALLEL_GROUP is None
    ), 'tensor model parallel group is already initialized'
    for ranks in decoder_rank_generator.get_ranks('tp'):
        group = create_group(
            ranks,
            timeout=timeout,
            pg_options=get_nccl_options("tp", nccl_comm_cfgs),
            group_desc="TENSOR_MODEL_PARALLEL_GROUP",
        )
        if rank in ranks:
            _TENSOR_MODEL_PARALLEL_GROUP = group
            _TENSOR_MODEL_PARALLEL_GLOBAL_RANKS = ranks

    # Build the pipeline model-parallel groups and embedding groups
    # (first and last rank in each pipeline model-parallel group).
    global _PIPELINE_MODEL_PARALLEL_GROUP
    global _PIPELINE_GLOBAL_RANKS
    assert (
        _PIPELINE_MODEL_PARALLEL_GROUP is None
    ), "pipeline model parallel group is already initialized"
    global _EMBEDDING_GROUP
    global _EMBEDDING_GLOBAL_RANKS
    assert _EMBEDDING_GROUP is None, "embedding group is already initialized"
    global _POSITION_EMBEDDING_GROUP
    global _POSITION_EMBEDDING_GLOBAL_RANKS
    assert _POSITION_EMBEDDING_GROUP is None, "position embedding group is already initialized"
    if pipeline_model_parallel_comm_backend == "ucc":
        # The UCC backend provides two key benefits:
        # 1) Achieves better bandwidth utilization than NCCL when using InfiniBand links.
        # 2) Does not use GPU SM resources (Zero-SM), mitigating performance interference
        #    with overlapping compute kernels.

        # The UCC backend is recommended in the following cases:
        # 1) When the exposed pipeline-parallel (PP) communications are significant.
        #    - E.g., Pipeline parallelism with very less gradient accumulation steps.
        #    - It may provide better performance due to improved bandwidth utilization.
        # 2) When the critical-path pipeline stage has substantial PP-communication overlap.
        #    - E.g., Uneven pipeline parallelism.
        #    - It may provide better performance due to zero SM resource usage.
        if "CUDA_DEVICE_MAX_CONNECTIONS" in os.environ:
            # UCC backend requires CUDA_DEVICE_MAX_CONNECTIONS variable to be larger than 1,
            # to gurantee the overlapped UCC communications. If this environment variable is set to 1,
            # all the UCC communication will be serialized.
            assert (
                os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] != "1"
            ), "UCC-backend requires CUDA_DEVICE_MAX_CONNECTIONS > 1"

        # Setting up required environment variables for ucc backend
        #
        # "TORCH_UCC_BLOCKING_WAIT=none" allows non-blocking waits of the communiction handle
        # "UCC_EC_CUDA_STREAM_TASK_MODE" controls how CUDA execution engines (EC)
        # schedule tasks on CUDA streams.
        # "UCX_TLS" controls transport layer selection
        # "NSYS_UCP_COMM_PARAMS=1" enables capturing ucx tracing in nsys profiling
        # "UCX_RNDV_THRESH" controls threshold threshold for switching between
        # eager and rendezvous (RNDV) communication protocols.
        # "UCX_NET_DEVICES" select which network interfaces UCX should use.
        # "UCC_CL_BASIC_TLS" controls which Transport Layers are used by
        # the Basic Collective libraray

        os.environ["TORCH_UCC_BLOCKING_WAIT"] = (
            os.environ["TORCH_UCC_BLOCKING_WAIT"]
            if "TORCH_UCC_BLOCKING_WAIT" in os.environ
            else "none"
        )
        os.environ["UCC_EC_CUDA_STREAM_TASK_MODE"] = (
            os.environ["UCC_EC_CUDA_STREAM_TASK_MODE"]
            if "UCC_EC_CUDA_STREAM_TASK_MODE" in os.environ
            else "driver"
        )
        os.environ["UCX_TLS"] = (
            os.environ["UCX_TLS"] if "UCX_TLS" in os.environ else "ib,cuda_copy"
        )  # cuda_ipc (i.e., NVLink-enablement) will be later supported
        os.environ["NSYS_UCP_COMM_PARAMS"] = "1"
        os.environ["UCX_RNDV_THRESH"] = "0"
        os.environ["UCX_NET_DEVICES"] = "all"
        os.environ["UCC_CL_BASIC_TLS"] = "^sharp,nccl"

    for ranks in decoder_rank_generator.get_ranks('pp'):
        group = create_group(
            ranks,
            timeout=timeout,
            backend=pipeline_model_parallel_comm_backend,
            pg_options=(
                None
                if pipeline_model_parallel_comm_backend == "ucc"
                else get_nccl_options("pp", nccl_comm_cfgs)
            ),
            group_desc="PIPELINE_MODEL_PARALLEL_GROUP",
        )
        assert (
            pipeline_model_parallel_comm_backend == None
            or pipeline_model_parallel_comm_backend == "nccl"
            or pipeline_model_parallel_comm_backend == "ucc"
        ), f'"{pipeline_model_parallel_comm_backend}" backend for PP communication is currently not supported'

        if rank in ranks:
            if _PIPELINE_MODEL_PARALLEL_GROUP is None:
                _PIPELINE_MODEL_PARALLEL_GROUP = group
                _PIPELINE_GLOBAL_RANKS = ranks
            elif isinstance(_PIPELINE_GLOBAL_RANKS[0], list):
                _PIPELINE_MODEL_PARALLEL_GROUP.append(group)
                _PIPELINE_GLOBAL_RANKS.append(ranks)
            else:
                _PIPELINE_MODEL_PARALLEL_GROUP = [_PIPELINE_MODEL_PARALLEL_GROUP, group]
                _PIPELINE_GLOBAL_RANKS = [_PIPELINE_GLOBAL_RANKS, ranks]

        embedding_ranks = get_embedding_ranks(ranks)
        group = create_group(
            embedding_ranks,
            timeout=timeout,
            pg_options=get_nccl_options("embd", nccl_comm_cfgs),
            group_desc="EMBEDDING_GROUP",
        )
        if rank in embedding_ranks:
            _EMBEDDING_GROUP = group
            _EMBEDDING_GLOBAL_RANKS = embedding_ranks

        position_embedding_ranks = get_position_embedding_ranks(ranks)
        group = create_group(
            position_embedding_ranks,
            timeout=timeout,
            pg_options=get_nccl_options("pos_embd", nccl_comm_cfgs),
            group_desc="POSITION_EMBEDDING_GROUP",
        )
        if rank in position_embedding_ranks:
            _POSITION_EMBEDDING_GROUP = group
            _POSITION_EMBEDDING_GLOBAL_RANKS = position_embedding_ranks

    # Build the tensor + data parallel groups.
    global _TENSOR_AND_DATA_PARALLEL_GROUP
    global _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP
    assert (
        _TENSOR_AND_DATA_PARALLEL_GROUP is None
    ), 'Tensor + data parallel group is already initialized'
    for ranks in decoder_rank_generator.get_ranks('tp-dp-cp'):
        group = create_group(
            ranks,
            timeout=timeout,
            pg_options=get_nccl_options("tp_dp_cp", nccl_comm_cfgs),
            group_desc="TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP",
        )
        if rank in ranks:
            _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP = group
    for ranks in decoder_rank_generator.get_ranks('tp-dp'):
        group = create_group(
            ranks,
            timeout=timeout,
            pg_options=get_nccl_options("tp_dp", nccl_comm_cfgs),
            group_desc="TENSOR_AND_DATA_PARALLEL_GROUP",
        )
        if rank in ranks:
            _TENSOR_AND_DATA_PARALLEL_GROUP = group

    global _TENSOR_AND_CONTEXT_PARALLEL_GROUP
    assert (
        _TENSOR_AND_CONTEXT_PARALLEL_GROUP is None
    ), 'Tensor + context parallel group is already initialized'
    for ranks in decoder_rank_generator.get_ranks('tp-cp'):
        group = create_group(
            ranks,
            timeout=timeout,
            pg_options=get_nccl_options("tp_cp", nccl_comm_cfgs),
            group_desc="TENSOR_AND_CONTEXT_PARALLEL_GROUP",
        )
        if rank in ranks:
            _TENSOR_AND_CONTEXT_PARALLEL_GROUP = group

    ### Expert-related parallel groups initialization
    # Build the expert model parallel group
    global _EXPERT_MODEL_PARALLEL_GROUP, _EXPERT_MODEL_PARALLEL_RANKS
    assert _EXPERT_MODEL_PARALLEL_GROUP is None, 'Expert parallel group is already initialized'
    for ranks in expert_decoder_rank_generator.get_ranks('ep'):
        group = create_group(
            ranks,
            timeout=timeout,
            pg_options=get_nccl_options("ep", nccl_comm_cfgs),
            group_desc="EXPERT_MODEL_PARALLEL_GROUP",
        )
        if rank in ranks:
            _EXPERT_MODEL_PARALLEL_GROUP = group
            _EXPERT_MODEL_PARALLEL_RANKS = ranks

    # Build the expert tensor parallel group
    global _EXPERT_TENSOR_PARALLEL_GROUP
    assert (
        _EXPERT_TENSOR_PARALLEL_GROUP is None
    ), 'Expert tensor model parallel group is already initialized'
    for ranks in expert_decoder_rank_generator.get_ranks('tp'):
        group = create_group(
            ranks,
            timeout=timeout,
            pg_options=get_nccl_options("ep_tp", nccl_comm_cfgs),
            group_desc="EXPERT_TENSOR_PARALLEL_GROUP",
        )
        if rank in ranks:
            _EXPERT_TENSOR_PARALLEL_GROUP = group

    # Build the tensor + expert parallel groups
    global _EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP
    assert (
        _EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP is None
    ), 'Expert tensor + model parallel group is already initialized'
    for ranks in expert_decoder_rank_generator.get_ranks('tp-ep'):
        group = create_group(
            ranks,
            timeout=timeout,
            pg_options=get_nccl_options("tp_ep_mp", nccl_comm_cfgs),
            group_desc="EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP",
        )
        if rank in ranks:
            _EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP = group

    # Build the expert+tensor+pipeline parallel groups
    global _EXPERT_TENSOR_MODEL_PIPELINE_PARALLEL_GROUP
    assert (
        _EXPERT_TENSOR_MODEL_PIPELINE_PARALLEL_GROUP is None
    ), 'The expert_tensor_model_pipeline parallel group is already initialized'
    for ranks in expert_decoder_rank_generator.get_ranks('tp-ep-pp'):
        group = create_group(
            ranks,
            timeout=timeout,
            pg_options=get_nccl_options("tp_ep_pp", nccl_comm_cfgs),
            group_desc="EXPERT_TENSOR_MODEL_PIPELINE_PARALLEL_GROUP",
        )
        if rank in ranks:
            _EXPERT_TENSOR_MODEL_PIPELINE_PARALLEL_GROUP = group

    # Build the expert data parallel group
    global _EXPERT_DATA_PARALLEL_GROUP
    assert _EXPERT_DATA_PARALLEL_GROUP is None, "Expert data group is already initialized"
    global _EXPERT_DATA_PARALLEL_GROUP_GLOO
    assert _EXPERT_DATA_PARALLEL_GROUP_GLOO is None, "Expert data group-gloo is already initialized"
    global _INTRA_PARTIAL_EXPERT_DATA_PARALLEL_GROUP
    assert (
        _INTRA_PARTIAL_EXPERT_DATA_PARALLEL_GROUP is None
    ), "Intra partial expert data group is already initialized"
    global _INTRA_PARTIAL_EXPERT_DATA_PARALLEL_GROUP_GLOO
    assert (
        _INTRA_PARTIAL_EXPERT_DATA_PARALLEL_GROUP_GLOO is None
    ), "Intra partial expert data group-gloo is already initialized"
    global _INTER_PARTIAL_EXPERT_DATA_PARALLEL_GROUP
    assert (
        _INTER_PARTIAL_EXPERT_DATA_PARALLEL_GROUP is None
    ), "Inter partial expert data group is already initialized"

    assert (
        expert_data_parallel_size % num_distributed_optimizer_instances == 0
    ), "Expert data parallel size should be divisible by partial DistOpt shard factor"
    intra_partial_expert_data_parallel_size = (
        expert_data_parallel_size // num_distributed_optimizer_instances
    )

    for ranks in expert_decoder_rank_generator.get_ranks('dp'):
        group = create_group(
            ranks,
            timeout=timeout,
            pg_options=get_nccl_options("ep_dp", nccl_comm_cfgs),
            group_desc="EXPERT_DATA_PARALLEL_GROUP",
        )
        if create_gloo_process_groups:
            group_gloo = create_group(
                ranks, backend="gloo", group_desc="EXPERT_DATA_PARALLEL_GROUP_GLOO"
            )
        else:
            group_gloo = None
        if rank in ranks:
            _EXPERT_DATA_PARALLEL_GROUP = group
            _EXPERT_DATA_PARALLEL_GROUP_GLOO = group_gloo

        if num_distributed_optimizer_instances > 1:
            # Create groups for Partial DistOpt, one for intra-partial DP domain
            # Another for inter-partial DP domain

            # Set NCCL_COLLNET_ENABLE to 1 to enable SHARP for the dp_replica group.
            if sharp_enabled_group == "dp_replica":
                os.environ["NCCL_COLLNET_ENABLE"] = "1"
            hierarchical_groups, hierarchical_groups_gloo = create_hierarchical_groups(
                rank,
                ranks,
                [intra_partial_expert_data_parallel_size, num_distributed_optimizer_instances],
                create_gloo_process_groups=create_gloo_process_groups,
                pg_options=[
                    get_nccl_options("intra_ep_dp", nccl_comm_cfgs),
                    get_nccl_options("inter_ep_dp", nccl_comm_cfgs),
                ],
                timeout=timeout,
                group_desc="EXPERT_DATA_PARALLEL_GROUP",
            )
            if rank in ranks:
                _INTRA_PARTIAL_EXPERT_DATA_PARALLEL_GROUP = hierarchical_groups[0]
                _INTRA_PARTIAL_EXPERT_DATA_PARALLEL_GROUP_GLOO = hierarchical_groups_gloo[0]
                _INTER_PARTIAL_EXPERT_DATA_PARALLEL_GROUP = hierarchical_groups[1]

            if sharp_enabled_group == "dp_replica":
                # PyTorch is performing lazy initialization of the communicator group.
                # Therefore, we need to perform a nccl call to ensure that the communicator group is created.
                if _INTER_PARTIAL_EXPERT_DATA_PARALLEL_GROUP is not None:
                    torch.distributed.barrier(
                        group=_INTER_PARTIAL_EXPERT_DATA_PARALLEL_GROUP,
                        device_ids=[torch.cuda.current_device()],
                    )
                    torch.cuda.synchronize()
                # Set NCCL_COLLNET_ENABLE to 0 to restrict SHARP application to the dp_replica group.
                if "NCCL_COLLNET_ENABLE" in os.environ:
                    del os.environ["NCCL_COLLNET_ENABLE"]
        else:
            _INTRA_PARTIAL_EXPERT_DATA_PARALLEL_GROUP = _EXPERT_DATA_PARALLEL_GROUP
            _INTRA_PARTIAL_EXPERT_DATA_PARALLEL_GROUP_GLOO = _EXPERT_DATA_PARALLEL_GROUP_GLOO
    ### End of expert related parallel groups initialization

    # build the intra distributed optimizer instance group
    global _INTRA_DISTRIBUTED_OPTIMIZER_INSTANCE_GROUP
    assert (
        _INTRA_DISTRIBUTED_OPTIMIZER_INSTANCE_GROUP is None
    ), "Intra distributed optimizer instance group is already initialized"

    model_parallel_group_id = 0
    intra_dist_opt_ranks = []
    for ranks in expert_decoder_rank_generator.get_ranks('tp-ep-pp'):
        model_parallel_group_id += 1
        intra_dist_opt_ranks.extend(ranks)
        if model_parallel_group_id % intra_partial_expert_data_parallel_size == 0:
            intra_dist_opt_instance_group = create_group(
                intra_dist_opt_ranks,
                timeout=timeout,
                pg_options=get_nccl_options("intra_dist_opt_instance", nccl_comm_cfgs),
                group_desc="INTRA_DISTRIBUTED_OPTIMIZER_INSTANCE_GROUP",
            )
            if rank in intra_dist_opt_ranks:
                _INTRA_DISTRIBUTED_OPTIMIZER_INSTANCE_GROUP = intra_dist_opt_instance_group
            intra_dist_opt_ranks = []

    # Initialize global memory buffer
    # This isn't really "parallel state" but there isn't another good place to
    # put this. If we end up with a more generic initialization of megatron-core
    # we could stick it there
    _set_global_memory_buffer()


def is_initialized():
    """Useful for code segments that may be accessed with or without mpu initialization"""
    return _DATA_PARALLEL_GROUP is not None


def is_unitialized() -> bool:
    """Check if parallel state has been initialized

    Deprecated. Use is_initialized instead.

    """
    warnings.warn("is_unitialized is deprecated, use is_initialized instead", DeprecationWarning)
    return not is_initialized()


def model_parallel_is_initialized():
    """Check if model- and data-parallel groups are initialized."""
    if (
        _TENSOR_MODEL_PARALLEL_GROUP is None
        or _PIPELINE_MODEL_PARALLEL_GROUP is None
        or _DATA_PARALLEL_GROUP is None
    ):
        return False
    return True


def get_model_parallel_group(check_initialized=True):
    """Get the model-parallel group the caller rank belongs to."""
    if check_initialized:
        assert _MODEL_PARALLEL_GROUP is not None, "model parallel group is not initialized"
    return _MODEL_PARALLEL_GROUP


def get_tensor_model_parallel_group(check_initialized=True):
    """Get the tensor-model-parallel group the caller rank belongs to."""
    if check_initialized:
        assert (
            _TENSOR_MODEL_PARALLEL_GROUP is not None
        ), "tensor model parallel group is not initialized"
    return _TENSOR_MODEL_PARALLEL_GROUP


def get_pipeline_model_parallel_group(check_initialized=True):
    """Get the pipeline-model-parallel group the caller rank belongs to."""
    if check_initialized:
        assert (
            _PIPELINE_MODEL_PARALLEL_GROUP is not None
        ), "pipeline_model parallel group is not initialized"
    return _PIPELINE_MODEL_PARALLEL_GROUP


def get_data_parallel_group(with_context_parallel=False, partial_data_parallel=False):
    """Get the data-parallel group the caller rank belongs to."""
    if with_context_parallel:
        if partial_data_parallel:
            assert (
                _INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP is not None
            ), "Intra partial data parallel group is not initialized"
            return _INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP
        assert (
            _DATA_PARALLEL_GROUP_WITH_CP is not None
        ), "data parallel group with context parallel combined is not initialized"
        return _DATA_PARALLEL_GROUP_WITH_CP
    else:
        assert _DATA_PARALLEL_GROUP is not None, "data parallel group is not initialized"
        assert partial_data_parallel == False, "Partial DP for Optimizer needs to include CP"
        return _DATA_PARALLEL_GROUP


def get_data_parallel_group_gloo(with_context_parallel=False, partial_data_parallel=False):
    """Get the Gloo data-parallel group the caller rank belongs to."""
    if with_context_parallel:
        if partial_data_parallel:
            assert (
                _INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP_GLOO is not None
            ), "Intra partial data parallel group is not initialized"
            return _INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP_GLOO
        assert (
            _DATA_PARALLEL_GROUP_WITH_CP_GLOO is not None
        ), "data parallel group-gloo with context parallel combined is not initialized"
        return _DATA_PARALLEL_GROUP_WITH_CP_GLOO
    else:
        assert _DATA_PARALLEL_GROUP_GLOO is not None, "data parallel group-gloo is not initialized"
        assert partial_data_parallel == False, "Partial DP for Optimizer needs to include CP"
        return _DATA_PARALLEL_GROUP_GLOO


def get_context_parallel_group(check_initialized=True):
    """Get the context-parallel group the caller rank belongs to."""
    if check_initialized:
        assert _CONTEXT_PARALLEL_GROUP is not None, "context parallel group is not initialized"
    return _CONTEXT_PARALLEL_GROUP


def get_context_parallel_global_ranks(check_initialized=True):
    """Get all global ranks of the context-parallel group that the caller rank belongs to."""
    if check_initialized:
        assert (
            _CONTEXT_PARALLEL_GLOBAL_RANKS is not None
        ), "context parallel group is not initialized"
    return _CONTEXT_PARALLEL_GLOBAL_RANKS


def get_hierarchical_context_parallel_groups(check_initialized=True):
    """Get the inner ring of context parallel group the caller rank belongs to."""
    if check_initialized:
        assert _HIERARCHICAL_CONTEXT_PARALLEL_GROUPS is not None
    return _HIERARCHICAL_CONTEXT_PARALLEL_GROUPS


def get_hybrid_data_context_parallel_groups(check_initialized=True, group_size=None):
    """Get the hybrid context parallel groups the caller rank belongs to."""
    # If the group size is the same as the entire DPxCP group, return the original group
    if get_data_parallel_world_size(with_context_parallel=True) == group_size:
        if check_initialized:
            assert _DATA_PARALLEL_GROUP_WITH_CP is not None
        return _DATA_PARALLEL_GROUP_WITH_CP
    if check_initialized:
        assert _HYBRID_DP_CP_GROUPS is not None
    return _HYBRID_DP_CP_GROUPS[group_size]


def get_embedding_group(check_initialized=True):
    """Get the embedding group the caller rank belongs to."""
    if check_initialized:
        assert _EMBEDDING_GROUP is not None, "embedding group is not initialized"
    return _EMBEDDING_GROUP


def get_position_embedding_group(check_initialized=True):
    """Get the position embedding group the caller rank belongs to."""
    if check_initialized:
        assert _POSITION_EMBEDDING_GROUP is not None, "position embedding group is not initialized"
    return _POSITION_EMBEDDING_GROUP


def get_amax_reduction_group(with_context_parallel=False, tp_only_amax_red=False):
    """Get the FP8 amax reduction group the caller rank belongs to."""
    if with_context_parallel:
        if not tp_only_amax_red:
            assert (
                _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP is not None
            ), "FP8 amax reduction group is not initialized"
            return _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP
        else:
            assert (
                _TENSOR_AND_CONTEXT_PARALLEL_GROUP is not None
            ), "FP8 amax reduction group is not initialized"
            return _TENSOR_AND_CONTEXT_PARALLEL_GROUP
    else:
        if not tp_only_amax_red:
            assert (
                _TENSOR_AND_DATA_PARALLEL_GROUP is not None
            ), "FP8 amax reduction group is not initialized"
            return _TENSOR_AND_DATA_PARALLEL_GROUP
        else:
            assert (
                _TENSOR_MODEL_PARALLEL_GROUP is not None
            ), "FP8 amax reduction group is not initialized"
            return _TENSOR_MODEL_PARALLEL_GROUP


def get_tensor_and_data_parallel_group(check_initialized=True, with_context_parallel=False):
    """Get the tensor- and data-parallel group the caller rank belongs to."""
    if with_context_parallel:
        if check_initialized:
            assert (
                _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP is not None
            ), 'tensor and data parallel group is not initialized'
        return _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP
    else:
        if check_initialized:
            assert (
                _TENSOR_AND_DATA_PARALLEL_GROUP is not None
            ), 'tensor and data parallel group is not initialized'
        return _TENSOR_AND_DATA_PARALLEL_GROUP


def get_tensor_and_context_parallel_group(check_initialized=True):
    """Get the tensor- and context-parallel group the caller rank belongs to."""
    if check_initialized:
        assert (
            _TENSOR_AND_CONTEXT_PARALLEL_GROUP is not None
        ), "tensor and context parallel group is not initialized"
    return _TENSOR_AND_CONTEXT_PARALLEL_GROUP


def set_tensor_model_parallel_world_size(world_size):
    """Set the tensor-model-parallel size"""
    global _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = world_size


def set_pipeline_model_parallel_world_size(world_size):
    """Set the pipeline-model-parallel size"""
    global _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = world_size


def set_virtual_pipeline_model_parallel_world_size(world_size):
    """Set the pipeline-model-parallel size"""
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = world_size


def get_tensor_model_parallel_world_size():
    """Return world size for the tensor-model-parallel group."""
    global _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    if _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE is not None:
        return _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    return get_tensor_model_parallel_group().size()


def get_pipeline_model_parallel_world_size():
    """Return world size for the pipeline-model-parallel group."""
    global _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    if _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE is not None:
        return _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    return get_pipeline_model_parallel_group().size()


def set_tensor_model_parallel_rank(rank):
    """Set tensor-model-parallel rank."""
    global _MPU_TENSOR_MODEL_PARALLEL_RANK
    _MPU_TENSOR_MODEL_PARALLEL_RANK = rank


def set_pipeline_model_parallel_rank(rank):
    """Set pipeline-model-parallel rank."""
    global _MPU_PIPELINE_MODEL_PARALLEL_RANK
    _MPU_PIPELINE_MODEL_PARALLEL_RANK = rank


def get_tensor_model_parallel_rank():
    """Return caller's rank for the tensor-model-parallel group."""
    global _MPU_TENSOR_MODEL_PARALLEL_RANK
    if _MPU_TENSOR_MODEL_PARALLEL_RANK is not None:
        return _MPU_TENSOR_MODEL_PARALLEL_RANK
    return get_tensor_model_parallel_group().rank()


def get_pipeline_model_parallel_rank():
    """Return caller's rank for the pipeline-model-parallel group."""
    global _MPU_PIPELINE_MODEL_PARALLEL_RANK
    if _MPU_PIPELINE_MODEL_PARALLEL_RANK is not None:
        return _MPU_PIPELINE_MODEL_PARALLEL_RANK
    return torch.distributed.get_rank(group=get_pipeline_model_parallel_group())


def is_pipeline_first_stage(ignore_virtual=True, vp_stage=None):
    """Return True if in the first pipeline model-parallel stage, False otherwise."""
    if not ignore_virtual and get_virtual_pipeline_model_parallel_world_size() is not None:
        assert vp_stage is not None, "vp_stage must be passed if virtual pipeline is enabled"

        if vp_stage != 0:
            return False
    return get_pipeline_model_parallel_rank() == 0


def is_pipeline_last_stage(ignore_virtual=True, vp_stage=None):
    """Return True if in the last pipeline-model-parallel stage, False otherwise."""
    if not ignore_virtual and get_virtual_pipeline_model_parallel_world_size() is not None:
        assert vp_stage is not None, "vp_stage must be passed if virtual pipeline is enabled"

        if vp_stage != (get_virtual_pipeline_model_parallel_world_size() - 1):
            return False
    return get_pipeline_model_parallel_rank() == (get_pipeline_model_parallel_world_size() - 1)


def is_rank_in_embedding_group(ignore_virtual=True, vp_stage=None):
    """Return true if current rank is in embedding group, False otherwise."""
    rank = torch.distributed.get_rank()
    global _EMBEDDING_GLOBAL_RANKS
    if _EMBEDDING_GLOBAL_RANKS is None:
        return False
    if ignore_virtual:
        return rank in _EMBEDDING_GLOBAL_RANKS
    if rank in _EMBEDDING_GLOBAL_RANKS:
        if rank == _EMBEDDING_GLOBAL_RANKS[0]:
            return is_pipeline_first_stage(ignore_virtual=False, vp_stage=vp_stage)
        elif rank == _EMBEDDING_GLOBAL_RANKS[-1]:
            return is_pipeline_last_stage(ignore_virtual=False, vp_stage=vp_stage)
        else:
            return True
    return False


def is_rank_in_position_embedding_group():
    """Return true if current rank is in position embedding group, False otherwise."""
    rank = torch.distributed.get_rank()
    global _POSITION_EMBEDDING_GLOBAL_RANKS
    return _POSITION_EMBEDDING_GLOBAL_RANKS is not None and rank in _POSITION_EMBEDDING_GLOBAL_RANKS


def get_virtual_pipeline_model_parallel_rank():
    """Return the virtual pipeline-parallel rank."""
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
    return _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK


def set_virtual_pipeline_model_parallel_rank(rank):
    """Set the virtual pipeline-parallel rank."""
    warnings.warn(
        "set_virtual_pipeline_model_parallel_rank in global scope is deprecated. "
        "Pass vp_stage explicitly to is_pipeline_first_stage, is_pipeline_last_stage, etc.",
        DeprecationWarning,
    )
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
    _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = rank


def get_virtual_pipeline_model_parallel_world_size():
    """Return the virtual pipeline-parallel world size."""
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    return _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE


def get_tensor_model_parallel_src_rank():
    """Calculate the global rank corresponding to the first local rank
    in the tensor model parallel group."""
    assert (
        _TENSOR_MODEL_PARALLEL_GLOBAL_RANKS is not None
    ), "Tensor model parallel group is not initialized"
    return _TENSOR_MODEL_PARALLEL_GLOBAL_RANKS[0]


def get_model_parallel_src_rank():
    """Calculate the global rank corresponding to the first local rank
    in the model parallel group."""
    assert _MODEL_PARALLEL_GLOBAL_RANKS is not None, "Model parallel group is not initialized"
    return _MODEL_PARALLEL_GLOBAL_RANKS[0]


def get_data_parallel_src_rank(with_context_parallel=False):
    """Calculate the global rank corresponding to the first local rank
    in the data parallel group."""
    if with_context_parallel:
        assert (
            _DATA_PARALLEL_GLOBAL_RANKS_WITH_CP is not None
        ), "Data parallel group with context parallel combined is not initialized"
        return _DATA_PARALLEL_GLOBAL_RANKS_WITH_CP[0]
    else:
        assert _DATA_PARALLEL_GLOBAL_RANKS is not None, "Data parallel group is not initialized"
        return _DATA_PARALLEL_GLOBAL_RANKS[0]


def get_pipeline_model_parallel_first_rank():
    """Return the global rank of the first stage in the current rank's pipeline."""
    assert _PIPELINE_GLOBAL_RANKS is not None, "Pipeline parallel group is not initialized"
    return _PIPELINE_GLOBAL_RANKS[0]


def get_pipeline_model_parallel_last_rank():
    """Return the global rank of the last stage in the current rank's pipeline."""
    assert _PIPELINE_GLOBAL_RANKS is not None, "Pipeline parallel group is not initialized"
    last_rank_local = get_pipeline_model_parallel_world_size() - 1
    return _PIPELINE_GLOBAL_RANKS[last_rank_local]


def get_pipeline_model_parallel_next_rank():
    """Return the global rank that follows the caller in the pipeline."""
    assert _PIPELINE_GLOBAL_RANKS is not None, "Pipeline parallel group is not initialized"
    rank_in_pipeline = get_pipeline_model_parallel_rank()
    world_size = get_pipeline_model_parallel_world_size()
    return _PIPELINE_GLOBAL_RANKS[(rank_in_pipeline + 1) % world_size]


def get_pipeline_model_parallel_prev_rank():
    """Return the global rank that precedes the caller in the pipeline."""
    assert _PIPELINE_GLOBAL_RANKS is not None, "Pipeline parallel group is not initialized"
    rank_in_pipeline = get_pipeline_model_parallel_rank()
    world_size = get_pipeline_model_parallel_world_size()
    return _PIPELINE_GLOBAL_RANKS[(rank_in_pipeline - 1) % world_size]


def get_data_parallel_world_size(with_context_parallel=False, partial_data_parallel=False):
    """Return world size for the data parallel group."""
    global _MPU_DATA_PARALLEL_WORLD_SIZE
    if _MPU_DATA_PARALLEL_WORLD_SIZE is not None:
        return _MPU_DATA_PARALLEL_WORLD_SIZE
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return get_data_parallel_group(
            with_context_parallel=with_context_parallel, partial_data_parallel=partial_data_parallel
        ).size()
    else:
        return 0


def set_data_parallel_rank(rank):
    """Return world size for the data parallel group."""
    global _MPU_DATA_PARALLEL_RANK
    _MPU_DATA_PARALLEL_RANK = rank


def get_data_parallel_rank(with_context_parallel=False, partial_data_parallel=False):
    """Return caller's rank in the data-parallel group."""
    global _MPU_DATA_PARALLEL_RANK
    if _MPU_DATA_PARALLEL_RANK is not None:
        return _MPU_DATA_PARALLEL_RANK
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return get_data_parallel_group(
            with_context_parallel=with_context_parallel, partial_data_parallel=partial_data_parallel
        ).rank()
    else:
        return 0


def get_context_parallel_world_size():
    """Return world size for the context parallel group."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return get_context_parallel_group().size()
    else:
        return 0


def get_context_parallel_rank():
    """Return caller's rank in the context-parallel group."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return get_context_parallel_group().rank()
    else:
        return 0


def get_tensor_and_context_parallel_world_size():
    """Return world size for the tensor and context-parallel group."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return get_tensor_and_context_parallel_group().size()
    else:
        return 0


def get_tensor_and_context_parallel_rank():
    """Return caller's rank in the joint tensor-model-parallel and context-parallel group."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return get_tensor_and_context_parallel_group().rank()
    else:
        return 0


### Expert-related parallel states functions
def get_expert_model_parallel_group(check_initialized=True):
    """Get the expert-model-parallel group the caller rank belongs to."""
    if check_initialized:
        assert (
            _EXPERT_MODEL_PARALLEL_GROUP is not None
        ), "expert model parallel group is not initialized"
    return _EXPERT_MODEL_PARALLEL_GROUP


def get_expert_model_parallel_src_rank():
    """Calculate the global rank corresponding to the first local rank
    in the expert model parallel group."""
    assert (
        _EXPERT_MODEL_PARALLEL_RANKS is not None
    ), "Expert model parallel group is not initialized"
    return _EXPERT_MODEL_PARALLEL_RANKS[0]


def get_expert_model_parallel_world_size():
    """Return world size for the expert-model-parallel group."""
    if _MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE is not None:
        return _MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return get_expert_model_parallel_group().size()
    else:
        return 0


def set_expert_model_parallel_world_size(world_size):
    """Sets the expert-model-parallel world size."""
    global _MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE
    _MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE = world_size


def get_expert_model_parallel_rank():
    """Return caller's rank in the expert-model-parallel group."""
    if _MPU_EXPERT_MODEL_PARALLEL_RANK is not None:
        return _MPU_EXPERT_MODEL_PARALLEL_RANK
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return get_expert_model_parallel_group().rank()
    else:
        return 0


def set_expert_model_parallel_rank(rank):
    """Set expert-model-parallel rank."""
    global _MPU_EXPERT_MODEL_PARALLEL_RANK
    _MPU_EXPERT_MODEL_PARALLEL_RANK = rank


def get_expert_tensor_parallel_group(check_initialized=True):
    """Get the expert-tensor-parallel group the caller rank belongs to."""
    if check_initialized:
        assert (
            _EXPERT_TENSOR_PARALLEL_GROUP is not None
        ), "Expert tensor parallel group is not initialized"
    return _EXPERT_TENSOR_PARALLEL_GROUP


def get_expert_tensor_parallel_world_size():
    """Return world size for the expert tensor parallel group."""
    global _MPU_EXPERT_TENSOR_PARALLEL_WORLD_SIZE
    if _MPU_EXPERT_TENSOR_PARALLEL_WORLD_SIZE is not None:
        return _MPU_EXPERT_TENSOR_PARALLEL_WORLD_SIZE
    # Use tensor parallel group world size for backward compability otherwise
    if not _EXPERT_TENSOR_PARALLEL_GROUP:
        return _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    else:
        return get_expert_tensor_parallel_group().size()


def set_expert_tensor_parallel_world_size(world_size):
    "Set expert tensor model parallel size"
    global _MPU_EXPERT_TENSOR_PARALLEL_WORLD_SIZE
    _MPU_EXPERT_TENSOR_PARALLEL_WORLD_SIZE = world_size


def get_expert_tensor_parallel_rank():
    """Return my rank for the expert tensor parallel group."""
    global _MPU_EXPERT_TENSOR_PARALLEL_RANK
    if _MPU_EXPERT_TENSOR_PARALLEL_RANK is not None:
        return _MPU_EXPERT_TENSOR_PARALLEL_RANK
    # Use tensor parallel group rank for backward compability otherwise
    if not _EXPERT_TENSOR_PARALLEL_GROUP:
        return _MPU_TENSOR_MODEL_PARALLEL_RANK
    else:
        return get_expert_tensor_parallel_group().rank()


def set_expert_tensor_parallel_rank(rank):
    "Set expert tensor model parallel rank"
    global _MPU_EXPERT_TENSOR_PARALLEL_RANK
    _MPU_EXPERT_TENSOR_PARALLEL_RANK = rank


def get_expert_tensor_and_model_parallel_group(check_initialized=True):
    """Get the expert-tensor and expert-model group the caller rank belongs to."""
    if check_initialized:
        assert (
            _EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP is not None
        ), "Expert tensor and model parallel group is not initialized"
    return _EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP


def get_expert_tensor_and_model_parallel_world_size():
    """Return world size for the expert model parallel group times expert tensor parallel group."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        world_size = get_expert_tensor_and_model_parallel_group().size()
        return world_size
    else:
        return 0


def get_expert_tensor_and_model_parallel_rank():
    """Return caller's rank in the joint tensor- and expert-model-parallel group."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return get_expert_tensor_and_model_parallel_group().rank()
    else:
        return 0


def get_expert_tensor_model_pipeline_parallel_group(check_initialized=True):
    """Get expert tensor-model-pipeline parallel group."""
    if check_initialized:
        assert (
            _EXPERT_TENSOR_MODEL_PIPELINE_PARALLEL_GROUP is not None
        ), "Expert tensor-model-pipeline parallel group is not initialized"
    return _EXPERT_TENSOR_MODEL_PIPELINE_PARALLEL_GROUP


def get_expert_data_parallel_group(check_initialized=True, partial_expert_data_parallel=False):
    """Get expert data parallel group."""
    if partial_expert_data_parallel:
        if check_initialized:
            assert (
                _INTRA_PARTIAL_EXPERT_DATA_PARALLEL_GROUP is not None
            ), "Intra partial expert data parallel group is not initialized"
        return _INTRA_PARTIAL_EXPERT_DATA_PARALLEL_GROUP
    else:
        if check_initialized:
            assert (
                _EXPERT_DATA_PARALLEL_GROUP is not None
            ), "Expert data parallel group is not initialized"
        return _EXPERT_DATA_PARALLEL_GROUP


def get_data_modulo_expert_parallel_group(partial_expert_data_parallel=False):
    """[Deprecated] Get expert data parallel group."""
    warnings.warn(
        "get_data_modulo_expert_parallel_group is deprecated, please use "
        "get_expert_data_parallel_group instead.",
        DeprecationWarning,
    )
    return get_expert_data_parallel_group(partial_expert_data_parallel=partial_expert_data_parallel)


def get_expert_data_parallel_group_gloo(partial_expert_data_parallel=False):
    """Get expert data parallel group-gloo."""
    if partial_expert_data_parallel:
        assert (
            _INTRA_PARTIAL_EXPERT_DATA_PARALLEL_GROUP_GLOO is not None
        ), "Intra partial expert data parallel group-gloo is not initialized"
        return _INTRA_PARTIAL_EXPERT_DATA_PARALLEL_GROUP_GLOO
    else:
        assert (
            _EXPERT_DATA_PARALLEL_GROUP_GLOO is not None
        ), "Expert data parallel group-gloo is not initialized"
        return _EXPERT_DATA_PARALLEL_GROUP_GLOO


def get_expert_data_parallel_rank(partial_expert_data_parallel=False):
    """Return caller's rank in the expert data parallel group."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return get_expert_data_parallel_group(
            partial_expert_data_parallel=partial_expert_data_parallel
        ).rank()
    else:
        return 0


def get_expert_data_parallel_world_size(partial_expert_data_parallel=False):
    """Return world size for the expert data parallel group."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return get_expert_data_parallel_group(
            partial_expert_data_parallel=partial_expert_data_parallel
        ).size()
    else:
        return 0


def get_intra_distributed_optimizer_instance_group(check_initialized=True):
    """Get the group of all GPUs in a distributed optimizer instance."""
    if check_initialized:
        assert (
            _INTRA_DISTRIBUTED_OPTIMIZER_INSTANCE_GROUP is not None
        ), "Intra distributed optimizer instance group is not initialized"
    return _INTRA_DISTRIBUTED_OPTIMIZER_INSTANCE_GROUP


def get_inter_distributed_optimizer_instance_group(check_initialized=True):
    """Get the group spanning the different distributed optimizer instances.
    Attention and MLP/Expert share same inter-instance group, so only built
    inter_partial_expert_data_parallel_group, and return it at here.
    """
    if check_initialized:
        assert _INTER_PARTIAL_EXPERT_DATA_PARALLEL_GROUP is not None, (
            "Attention and MLP/Expert share same inter distributed optimize instance group, "
            "which has not been initialized"
        )
    return _INTER_PARTIAL_EXPERT_DATA_PARALLEL_GROUP


### End of expert-related functions region


def _set_global_memory_buffer():
    """Initialize global buffer."""
    global _GLOBAL_MEMORY_BUFFER
    assert _GLOBAL_MEMORY_BUFFER is None, "global memory buffer is already initialized"
    _GLOBAL_MEMORY_BUFFER = GlobalMemoryBuffer()


def _set_global_symmetric_memory_buffer():
    """Initialize global buffer."""
    global _GLOBAL_SYMMETRIC_MEMORY_BUFFER
    assert _GLOBAL_SYMMETRIC_MEMORY_BUFFER is None, "global memory buffer is already initialized"

    _GLOBAL_SYMMETRIC_MEMORY_BUFFER = GlobalSymmetricMemoryBuffer(
        size_in_mb=256,  # todo: set from an argument?
        process_group=get_tensor_model_parallel_group(),
    )


def get_global_memory_buffer():
    """Return the global GlobalMemoryBuffer object"""
    assert _GLOBAL_MEMORY_BUFFER is not None, "global memory buffer is not initialized"
    return _GLOBAL_MEMORY_BUFFER


def get_global_symmetric_memory_buffer():
    """Return the global GlobalSymmetricMemoryBuffer object"""
    assert (
        _GLOBAL_SYMMETRIC_MEMORY_BUFFER is not None
    ), "global symmetric memory buffer is not initialized"
    return _GLOBAL_SYMMETRIC_MEMORY_BUFFER


def destroy_global_memory_buffer():
    """Sets the global memory buffer to None"""
    global _GLOBAL_MEMORY_BUFFER
    _GLOBAL_MEMORY_BUFFER = None


def destroy_global_symmetric_memory_buffer():
    """Sets the global symmetric memory buffer to None"""
    global _GLOBAL_SYMMETRIC_MEMORY_BUFFER
    _GLOBAL_SYMMETRIC_MEMORY_BUFFER = None


def get_all_ranks():
    """Get caller's rank in tensor-model-parallel, data-parallel, context-parallel,
    pipeline-model-parallel and expert-model-parallel groups."""
    ranks = [
        get_tensor_model_parallel_rank(),
        get_data_parallel_rank(),
        get_context_parallel_rank(),
        get_pipeline_model_parallel_rank(),
        get_expert_model_parallel_rank(),
    ]
    return "_".join(map(lambda x: str(x or 0), ranks))


def destroy_model_parallel():
    """Set the groups to none."""
    global _MODEL_PARALLEL_GROUP
    _MODEL_PARALLEL_GROUP = None

    global _TENSOR_MODEL_PARALLEL_GROUP
    _TENSOR_MODEL_PARALLEL_GROUP = None

    global _PIPELINE_MODEL_PARALLEL_GROUP
    _PIPELINE_MODEL_PARALLEL_GROUP = None

    global _DATA_PARALLEL_GROUP
    _DATA_PARALLEL_GROUP = None

    global _DATA_PARALLEL_GROUP_WITH_CP
    _DATA_PARALLEL_GROUP_WITH_CP = None

    global _CONTEXT_PARALLEL_GROUP
    _CONTEXT_PARALLEL_GROUP = None

    global _CONTEXT_PARALLEL_GLOBAL_RANKS
    _CONTEXT_PARALLEL_GLOBAL_RANKS = None

    global _EMBEDDING_GROUP
    _EMBEDDING_GROUP = None

    global _POSITION_EMBEDDING_GROUP
    _POSITION_EMBEDDING_GROUP = None

    global _POSITION_EMBEDDING_GLOBAL_RANKS
    _POSITION_EMBEDDING_GLOBAL_RANKS = None

    global _TENSOR_AND_DATA_PARALLEL_GROUP
    _TENSOR_AND_DATA_PARALLEL_GROUP = None

    global _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP
    _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP = None

    global _TENSOR_AND_CONTEXT_PARALLEL_GROUP
    _TENSOR_AND_CONTEXT_PARALLEL_GROUP = None

    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
    _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = None

    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None

    global _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = None

    global _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None

    global _MPU_TENSOR_MODEL_PARALLEL_RANK
    _MPU_TENSOR_MODEL_PARALLEL_RANK = None

    global _MPU_PIPELINE_MODEL_PARALLEL_RANK
    _MPU_PIPELINE_MODEL_PARALLEL_RANK = None

    global _GLOBAL_MEMORY_BUFFER
    _GLOBAL_MEMORY_BUFFER = None

    global _GLOBAL_SYMMETRIC_MEMORY_BUFFER
    _GLOBAL_SYMMETRIC_MEMORY_BUFFER = None

    global _DATA_PARALLEL_GROUP_GLOO
    if (
        _DATA_PARALLEL_GROUP_GLOO is not None
        and torch.distributed.distributed_c10d._world.pg_map.get(_DATA_PARALLEL_GROUP_GLOO, None)
        is not None
    ):
        torch.distributed.destroy_process_group(_DATA_PARALLEL_GROUP_GLOO)
    _DATA_PARALLEL_GROUP_GLOO = None

    global _DATA_PARALLEL_GROUP_WITH_CP_GLOO
    if (
        _DATA_PARALLEL_GROUP_WITH_CP_GLOO is not None
        and torch.distributed.distributed_c10d._world.pg_map.get(
            _DATA_PARALLEL_GROUP_WITH_CP_GLOO, None
        )
        is not None
    ):
        torch.distributed.destroy_process_group(_DATA_PARALLEL_GROUP_WITH_CP_GLOO)
    _DATA_PARALLEL_GROUP_WITH_CP_GLOO = None

    # Destroy parallel state related to expert parallelism.
    global _EXPERT_MODEL_PARALLEL_GROUP
    _EXPERT_MODEL_PARALLEL_GROUP = None

    global _MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE
    _MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE = None

    global _MPU_EXPERT_MODEL_PARALLEL_RANK
    _MPU_EXPERT_MODEL_PARALLEL_RANK = None

    global _EXPERT_TENSOR_PARALLEL_GROUP
    _EXPERT_TENSOR_PARALLEL_GROUP = None

    global _MPU_EXPERT_TENSOR_PARALLEL_WORLD_SIZE
    _MPU_EXPERT_TENSOR_PARALLEL_WORLD_SIZE = None

    global _MPU_EXPERT_TENSOR_PARALLEL_RANK
    _MPU_EXPERT_TENSOR_PARALLEL_RANK = None

    global _EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP
    _EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP = None

    global _EXPERT_TENSOR_MODEL_PIPELINE_PARALLEL_GROUP
    _EXPERT_TENSOR_MODEL_PIPELINE_PARALLEL_GROUP = None

    global _EXPERT_DATA_PARALLEL_GROUP
    _EXPERT_DATA_PARALLEL_GROUP = None

    global _EXPERT_DATA_PARALLEL_GROUP_GLOO
    if (
        _EXPERT_DATA_PARALLEL_GROUP_GLOO is not None
        and torch.distributed.distributed_c10d._world.pg_map.get(
            _EXPERT_DATA_PARALLEL_GROUP_GLOO, None
        )
        is not None
    ):
        torch.distributed.destroy_process_group(_EXPERT_DATA_PARALLEL_GROUP_GLOO)
    _EXPERT_DATA_PARALLEL_GROUP_GLOO = None

    global _INTRA_PARTIAL_EXPERT_DATA_PARALLEL_GROUP
    _INTRA_PARTIAL_EXPERT_DATA_PARALLEL_GROUP = None

    global _INTRA_PARTIAL_EXPERT_DATA_PARALLEL_GROUP_GLOO
    if (
        _INTRA_PARTIAL_EXPERT_DATA_PARALLEL_GROUP_GLOO is not None
        and torch.distributed.distributed_c10d._world.pg_map.get(
            _INTRA_PARTIAL_EXPERT_DATA_PARALLEL_GROUP_GLOO, None
        )
        is not None
    ):
        torch.distributed.destroy_process_group(_INTRA_PARTIAL_EXPERT_DATA_PARALLEL_GROUP_GLOO)
    _INTRA_PARTIAL_EXPERT_DATA_PARALLEL_GROUP_GLOO = None

    global _INTER_PARTIAL_EXPERT_DATA_PARALLEL_GROUP
    _INTER_PARTIAL_EXPERT_DATA_PARALLEL_GROUP = None
    # End of expert parallelism destroy.

    global _INTRA_DISTRIBUTED_OPTIMIZER_INSTANCE_GROUP
    _INTRA_DISTRIBUTED_OPTIMIZER_INSTANCE_GROUP = None

    global _global_process_group_list
    _global_process_group_list = None
