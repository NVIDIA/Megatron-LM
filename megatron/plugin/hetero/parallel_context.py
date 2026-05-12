import os
import math
import warnings
import itertools
import operator
import warnings
import dataclasses
from typing import List, Optional
from datetime import timedelta
from functools import cmp_to_key
from collections import defaultdict
from typing import Callable, List, Optional

import torch

try:
    import flagcx
except:
    warnings.warn(
            "flagcx is not installed, you can't use flagcx backend for communication.",
            ImportWarning,
        )

_GLOBAL_PARALLEL_CONTEXT = None

from megatron.plugin.platform import get_platform
cur_platform = get_platform()


def _ensure_var_is_initialized(var, name):
    """Make sure the input variable is not None."""
    # Refer to the same function from megatron/megatron/training/global_vars.py
    assert var is not None, "{} is not initialized.".format(name)


def _ensure_var_is_not_initialized(var, name):
    """Make sure the input variable is not None."""
    # Refer to the same function from megatron/megatron/training/global_vars.py
    assert var is None, "{} is already initialized.".format(name)


def get_parallel_context():
    """Return heterogenous parallel context."""
    return _GLOBAL_PARALLEL_CONTEXT


def set_parallel_context(args):
    """Initialize heterogenous parallel context."""
    global _GLOBAL_PARALLEL_CONTEXT
    _ensure_var_is_not_initialized(_GLOBAL_PARALLEL_CONTEXT, "parallel context")
    _GLOBAL_PARALLEL_CONTEXT = ParallelContext(args)


def get_nccl_options(pg_name, nccl_comm_cfgs):
    from megatron.core.parallel_state import get_nccl_options
    return get_nccl_options(pg_name, nccl_comm_cfgs)


def get_group_name(token, is_expert=False):
        # Add a prefix exp to form the expert process group names
        # Make the group name is unique for the expert and non-expert process groups
        if not is_expert:
            return token
        else:
            return f"exp_{token}"

def get_nccl_option_name(token, is_expert=False):
    if not is_expert:
        names = {
            "dp": "dp",
            "dp-cp": "dp_cp",
            "intra-dp-cp": "dp_cp",
            "inter-dp-cp": "dp_cp",
            "dp-cp": "dp_cp",
            "cp": "cp",
            "hierachical-cp": "hcp",
            "tp-pp": "mp",
            "tp": "tp",
            "pp": "pp",
            "tp-dp-cp": "tp_dp_cp",
            "tp-dp": "tp_dp",
            "tp-cp": "tp_cp",
        }
        name = names.get(token, None)
        if name is None:
            raise ValueError(f"Invalid token: {token}")
    else:
        names = {
            "ep": "ep",
            "tp": "ep_tp",
            "tp-ep": "tp_exp",
            "tp-ep-pp": "tp_ep_mp",
            "dp": "ep_dp",
        }
        name = names.get(token, None)
        if name is None:
            raise ValueError(f"Invalid token: {token}")
    return name

def create_group(
    ranks=None,
    timeout=None,
    backend=None,
    pg_options=None,
    use_local_synchronization=False,
    group_desc=None,
):
    from megatron.core.parallel_state import create_group
    return create_group(
        ranks=ranks,
        timeout=timeout,
        backend=backend,
        pg_options=pg_options,
        use_local_synchronization=use_local_synchronization,
        group_desc=group_desc,
    )

def find_overlapped_mapping(dim1, dim2, global_size=None):
    """
    Finds the overlapped mapping between two dimensions within an optional global size. Please refer to https://eli.thegreenplace.net/2008/08/15/intersection-of-1d-segments for details.
    """
    # Calculate the least common multiple (LCM) of dim1 and dim2, or use global_size if provided
    dim_lcm = global_size if global_size else math.lcm(dim1, dim2)
    # Generate segments for dim1 and dim2
    dim1_segments_len = dim_lcm // dim1
    dim1_segments = [(i * dim1_segments_len, (i + 1) * dim1_segments_len) for i in range(dim1)]
    dim2_segments_len = dim_lcm // dim2
    dim2_segments = [(i * dim2_segments_len, (i + 1) * dim2_segments_len) for i in range(dim2)]
    # Initialize the mapping of overlapped segments
    overlapped_mapping = {i: [] for i in range(dim1)}
    # Calculate overlaps between dim1 and dim2 segments
    for i, (start1, end1) in enumerate(dim1_segments):
        for j, (start2, end2) in enumerate(dim2_segments):
            if start1 < end2 and end1 > start2:  # Check if segments overlap
                # Calculate the overlap offsets relative to the start of the dim1 segment
                local_overlap_start1 = max(start1, start2) - start1
                local_overlap_end1 = min(end1, end2) - start1
                overlapped_mapping[i].append(
                    (j, local_overlap_start1, local_overlap_end1)
                )
    return overlapped_mapping


class RankMapper:
    def __init__(self, args):
        assert (
            torch.distributed.is_initialized()
        ), "torch.distributed is not initialized"
        self._world_size = torch.distributed.get_world_size()
        # The order of device types is very import for creating the logical rank.
        # Users should make sure the order satisfies their needs.
        self._hetero_device_types = args.hetero_device_types
        self._hetero_current_device_type = args.hetero_current_device_type
        self._rank_infos = {}
        self._physical_rank_to_logical_rank = {}
        self._logical_rank_to_physical_rank = {}
        self.build_rank_mapping()

    def build_rank_mapping(self):
        # Collect all rank infos.
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        all_rank_infos = [None] * world_size
        cur_rank_info = {'rank': rank,
                         'device_type': self._hetero_current_device_type}
        torch.distributed.all_gather_object(
            all_rank_infos, cur_rank_info)
        physical_ranks = []
        for info in all_rank_infos:
            self._rank_infos[info['rank']] = info
            physical_ranks.append(info['rank'])

        # Sort the physical ranks by device type and rank.
        def _compare(rank1, rank2):
            device_type1 = self._rank_infos[rank1]['device_type']
            device_type2 = self._rank_infos[rank2]['device_type']
            if self._hetero_device_types \
                and self._hetero_device_types.index(device_type1) < self._hetero_device_types.index(device_type2):
                return -1
            elif self._hetero_device_types \
                and self._hetero_device_types.index(device_type1) > self._hetero_device_types.index(device_type2):
                return 1
            else:
                return rank1 - rank2
        sorted_physical_ranks = sorted(
            physical_ranks, key=cmp_to_key(_compare))

        # Build the mapping between physical rank and logical rank
        for logical_rank, physical_rank in enumerate(sorted_physical_ranks):
            self._physical_rank_to_logical_rank[physical_rank] = logical_rank
            self._logical_rank_to_physical_rank[logical_rank] = physical_rank

    def to_physical_ranks(self, logical_ranks: list) -> list:
        """Converts logical ranks to physical ranks."""
        physical_ranks = []
        for logical_rank in logical_ranks:
            physical_ranks.append(
                self._logical_rank_to_physical_rank[logical_rank])
        return physical_ranks

    def to_logical_ranks(self, physical_ranks: list) -> list:
        """Converts physical ranks to logical ranks."""
        logical_ranks = []
        for physical_rank in physical_ranks:
            logical_ranks.append(
                self._physical_rank_to_logical_rank[physical_rank])
        return logical_ranks


class ProcessMesh:
    """ Define n-dimensional Cartesian process topology. """

    def __init__(
        self,
        data_parallel_size: int = 1,
        tensor_model_parallel_size: int = 1,
        pipeline_model_parallel_size: int = 1,
        virtual_pipeline_model_parallel_size: Optional[int] = None,
        pipeline_model_parallel_split_rank: Optional[int] = None,
        use_sharp: bool = False,
        context_parallel_size: int = 1,
        hierarchical_context_parallel_sizes: Optional[List[int]] = None,
        expert_model_parallel_size: int = 1,
        num_distributed_optimizer_instances: int = 1,
        expert_tensor_parallel_size: Optional[int] = None,
        nccl_communicator_config_path: Optional[str] = None,
        distributed_timeout_minutes: int = 30,
        order: str = "tp-cp-ep-dp-pp",
        offset: int = 0,
        rank_mapper: RankMapper = None,
        args: dict = None,
    ):
        assert torch.distributed.is_initialized()
        self._data_parallel_size = data_parallel_size
        self._tensor_model_parallel_size = tensor_model_parallel_size
        self._pipeline_model_parallel_size = pipeline_model_parallel_size
        self._virtual_pipeline_model_parallel_size = virtual_pipeline_model_parallel_size
        self._pipeline_model_parallel_split_rank = pipeline_model_parallel_split_rank
        self._use_sharp = use_sharp
        self._context_parallel_size = context_parallel_size
        self._hierarchical_context_parallel_sizes = hierarchical_context_parallel_sizes
        self._expert_model_parallel_size = expert_model_parallel_size
        self._num_distributed_optimizer_instances = num_distributed_optimizer_instances
        self._intra_partial_data_parallel_size = data_parallel_size * context_parallel_size // num_distributed_optimizer_instances
        self._expert_tensor_parallel_size = expert_tensor_parallel_size
        self._order = order
        self._offset = offset
        self._args = args
        self.create_gloo_process_groups = getattr(args, 'use_gloo_process_groups', True)

        self._timeout = timedelta(minutes=distributed_timeout_minutes)
        self._rank = torch.distributed.get_rank()
        self._world_size = (
            self._tensor_model_parallel_size
            * self._pipeline_model_parallel_size
            * self._context_parallel_size
            * self._data_parallel_size
        )
        self._distributed_backend = args.distributed_backend

        if self._virtual_pipeline_model_parallel_size is not None:
            if not self._pipeline_model_parallel_size > 2:
                raise RuntimeError(
                    "pipeline-model-parallel size should be greater than 2 with interleaved schedule"
                )

            self._virtual_pipeline_model_parallel_rank = 0
            self._virtual_pipeline_model_parallel_world_size = self._virtual_pipeline_model_parallel_size

        self._nccl_comm_cfgs = {}
        if nccl_communicator_config_path is not None:
            try:
                import yaml
            except ImportError:
                raise RuntimeError(
                    "Cannot import `yaml`. Setting custom nccl communicator configs "
                    "requires the yaml package."
                )

            with open(nccl_communicator_config_path, "r") as stream:
                self._nccl_comm_cfgs = yaml.safe_load(stream)

        from megatron.core.parallel_state import RankGenerator
        self._rank_generator = RankGenerator(
            tp=self._tensor_model_parallel_size,
            ep=1,
            dp=self._data_parallel_size,
            pp=self._pipeline_model_parallel_size,
            cp=self._context_parallel_size,
            order=self._order,
        )

        # Build expert rank generator
        self._expert_tensor_model_pipeline_parallel_size = (
            self._expert_tensor_parallel_size
            * self._expert_model_parallel_size
            * self._pipeline_model_parallel_size
        )
        self._expert_data_parallel_size = self._world_size // self._expert_tensor_model_pipeline_parallel_size
        if self._world_size % self._expert_tensor_model_pipeline_parallel_size != 0:
            raise RuntimeError(
                f"decoder world_size ({self._world_size}) is not divisible by expert_tensor_model_pipeline_parallel size ({self._expert_tensor_model_pipeline_parallel_size})"
            )

        self._expert_rank_generator = RankGenerator(
            tp=self._expert_tensor_parallel_size,
            ep=self._expert_model_parallel_size,
            dp=self._expert_data_parallel_size,
            pp=self._pipeline_model_parallel_size,
            cp=1,
            order=self._order,
        )

        self._rank_mapper = rank_mapper
        self._group_ranks = {} # group_ranks belongs to the current rank
        self._all_group_ranks = defaultdict(list) # group_ranks belongs to the current process mesh
        self._process_groups = {} # process groups belongs to the current rank
        self._process_groups_gloo = {} # process groups belongs to the current rank with gloo backend

        self.build_all_process_groups()

    def build_process_group(
        self, token, is_expert=False, gloo=False, create_gloo_process_groups=True
    ):
        if not is_expert:
            logical_ranks_list = self._rank_generator.get_ranks(token)
        else:
            logical_ranks_list = self._expert_rank_generator.get_ranks(token)
        # Add the offset for each ranks of the current process mesh
        for logical_ranks in logical_ranks_list:
            for i in range(len(logical_ranks)):
                logical_ranks[i] += self._offset

        for logical_ranks in logical_ranks_list:
            group_name = get_group_name(token, is_expert=is_expert)
            nccl_option_name = get_nccl_option_name(token, is_expert=is_expert)
            pg_options = get_nccl_options(nccl_option_name, self._nccl_comm_cfgs)
            ranks = self._rank_mapper.to_physical_ranks(logical_ranks)
            group = create_group(ranks, timeout=self._timeout, backend=self._distributed_backend, pg_options=pg_options, group_desc=group_name)
            if gloo:
                if create_gloo_process_groups:
                    group_gloo = create_group(ranks, timeout=self._timeout, backend="gloo", group_desc=group_name+"_gloo")
                else:
                    group_gloo = None
            self._all_group_ranks[group_name].append(ranks)
            if self._rank in ranks:
                self._group_ranks[group_name] = ranks
                self._process_groups[group_name] = group
                if gloo:
                    self._process_groups_gloo[group_name] = group_gloo

            if token == "dp-cp" and not is_expert:
                self._build_dist_opt_process_groups(token, ranks, pg_options, group, group_gloo, create_gloo_process_groups=create_gloo_process_groups)

            if token == "cp" and not is_expert:
                self._build_hierarchical_cp_groups(ranks, pg_options)

    def _build_dist_opt_process_groups(self, token, ranks, pg_options, group, group_gloo, create_gloo_process_groups=True):
        if self._num_distributed_optimizer_instances > 1:
            # Create groups for Partial DistOpt, one for intra-partial DP domain
            # Another for inter-partial DP domain
            for i in range(self._num_distributed_optimizer_instances):
                intra_partial_data_parallel_ranks_with_cp = ranks[
                    (i * self._intra_partial_data_parallel_size) : (
                        (i + 1) * self._intra_partial_data_parallel_size
                    )
                ]

                if token == "dp-cp":
                    group_name = "intra-dp-cp"
                else:
                    raise ValueError(f"Invalid token: {token}")
                self._all_group_ranks[group_name].append(intra_partial_data_parallel_ranks_with_cp)

                intra_partial_data_parallel_group_with_cp = create_group(
                    intra_partial_data_parallel_ranks_with_cp,
                    timeout=self._timeout,
                    backend=self._distributed_backend,
                    pg_options=pg_options,
                    group_desc=group_name,
                )
                if create_gloo_process_groups:
                    intra_partial_data_parallel_group_with_cp_gloo = create_group(
                        intra_partial_data_parallel_ranks_with_cp,
                        timeout=self._timeout,
                        backend="gloo",
                        group_desc=group_name+"_gloo")
                else:
                    intra_partial_data_parallel_group_with_cp_gloo = None

                if self._rank in intra_partial_data_parallel_ranks_with_cp:
                    self._group_ranks[group_name] = intra_partial_data_parallel_ranks_with_cp
                    self._process_groups[group_name] = intra_partial_data_parallel_group_with_cp
                    self._process_groups_gloo[group_name] = intra_partial_data_parallel_group_with_cp_gloo

            for i in range(self._intra_partial_data_parallel_size):
                inter_partial_data_parallel_ranks_with_cp = ranks[
                    i::self._intra_partial_data_parallel_size
                ]

                if token == "dp-cp":
                    group_name = "intra-dp-cp"
                else:
                    raise ValueError(f"Invalid token: {token}")

                self._all_group_ranks[group_name].append(inter_partial_data_parallel_ranks_with_cp)

                inter_partial_data_parallel_group_with_cp = create_group(
                    inter_partial_data_parallel_ranks_with_cp,
                    timeout=self._timeout,
                    backend=self._distributed_backend,
                    pg_options=pg_options,
                    group_desc=group_name,
                )

                if self._rank in inter_partial_data_parallel_ranks_with_cp:
                    self._process_groups[group_name] = inter_partial_data_parallel_group_with_cp
        else:
            if token == "dp-cp":
                group_name = "intra-dp-cp"
            else:
                raise ValueError(f"Invalid token: {token}")
            self._all_group_ranks[group_name].append(ranks)
            if self._rank in ranks:
                self._process_groups[group_name] = group
                self._process_groups_gloo[group_name] = group_gloo

    def _build_hierarchical_cp_groups(self, ranks, pg_options):
        if self._hierarchical_context_parallel_sizes:
            group_name = "hierarchical-cp"
            from megatron.core.parallel_state import create_hierarchical_parallel_groups
            hierarchical_cp_groups = self._process_groups.get(group_name, [])
            hierarchical_cp_groups += create_hierarchical_parallel_groups(
                self._rank,
                ranks,
                self._context_parallel_size,
                self._hierarchical_context_parallel_sizes,
                pg_options,
            )
            self._process_groups[group_name] = hierarchical_cp_groups

    def build_all_process_groups(self):
        self.build_process_group("dp", is_expert=False, gloo=True, create_gloo_process_groups=self.create_gloo_process_groups)
        self.build_process_group("dp-cp", is_expert=False, gloo=True, create_gloo_process_groups=self.create_gloo_process_groups)

        # Apply SHARP to DP process groups
        if self._use_sharp:
            if self._rank == 0:
                print(
                    "The number of process groups to use SHARP with depends on the type "
                    "of the network switch. Nvidia QM1 switch supports SAHRP up to 8 "
                    "process groups and QM2 supports up to 256 process groups. We apply "
                    "SHARP to the communications of the data-parallel domain. If the "
                    "number of data-parallel process groups is larger than the max "
                    "process groups that the network switch supports, the communication "
                    "will fall back to non-SHARP operators. To enable SHARP, "
                    "`#SBATCH_NETWORK=sharp` should be set in the sbatch script."
                )
            torch.distributed.barrier(
                group=self.get_process_group("dp-cp"),
                device_ids=[cur_platform.current_device()],
            )
            # Set `NCCL_COLLNET_ENABLE=0` to restrict SHARP application to DP process groups
            os.environ["NCCL_COLLNET_ENABLE"] = "0"

        self.build_process_group("cp", is_expert=False, gloo=False)
        self.build_process_group("tp-pp", is_expert=False, gloo=False)
        self.build_process_group('tp', is_expert=False, gloo=False)
        self.build_process_group("pp", is_expert=False, gloo=False)
        self.build_process_group("tp-dp-cp", is_expert=False, gloo=False)
        self.build_process_group("tp-dp", is_expert=False, gloo=False)
        self.build_process_group("tp-cp", is_expert=False, gloo=False)
        # build expert process groups
        self.build_process_group("ep", is_expert=True, gloo=False)
        self.build_process_group("tp", is_expert=True, gloo=False)
        self.build_process_group("tp-ep", is_expert=True, gloo=False)
        self.build_process_group("tp-ep-pp", is_expert=True, gloo=False)
        self.build_process_group("dp", is_expert=True, gloo=True, create_gloo_process_groups=self.create_gloo_process_groups)

    def get_parallel_size(self, token, is_expert=False):
        if not is_expert:
            parallel_sizes = self._rank_generator.ordered_size
            order = self._rank_generator.order
        else:
            parallel_sizes = self._expert_rank_generator.ordered_size
            order = self._expert_rank_generator.order
        order = order.split("-")
        if token in order:
            return parallel_sizes[order.index(token)]
        else:
            raise ValueError(f"Invalid token: {token}")

    def get_process_group(
        self,
        token,
        is_expert=False,
        gloo=False,
        check_initialized=False,
    ):
        group_name = get_group_name(token, is_expert=is_expert)
        if gloo:
            group = self._process_groups_gloo.get(group_name, None)
        else:
            group = self._process_groups.get(group_name, None)
        if check_initialized:
            assert (
                group is not None
            ), f"Process group {group_name} is not initialized."
        return group

    def get_process_group_size(self, token, is_expert=False, gloo=False):
        group_name = get_group_name(token, is_expert=is_expert)
        if gloo:
            return torch.distributed.get_world_size(self._process_groups_gloo[group_name])
        else:
            return torch.distributed.get_world_size(self._process_groups[group_name])

    def get_process_group_ranks(
        self, token, is_expert=False, check_initialized=False
    ):
        group_name = get_group_name(token, is_expert=is_expert)
        ranks = self._group_ranks.get(group_name, None)
        if check_initialized:
            assert (
                ranks is not None
            ), f"Process group {group_name} is not initialized."
        return ranks

    def get_all_process_group_ranks(
        self, token, is_expert=False, check_initialized=False
    ):
        group_name = get_group_name(token, is_expert=is_expert)
        ranks = self._all_group_ranks.get(group_name, None)
        if check_initialized:
            assert (
                ranks is not None
            ), f"Process group {group_name} is not initialized."
        return ranks

    def get_transformer_config(self):
        return self._transformer_config

    def get_ddp_config(self):
        return self._ddp_config

    def get_optimizer_config(self):
        return (self._optimizer_config, self._optimizer_config_overrides)
    
    def logical_coords_to_physical_ranks(self, coords, is_expert=False):
        def _prefix_product(a: List[int], init=1) -> List[int]:
            r = [init]
            for v in a:
                init = init * v
                r.append(init)
            return r
        for coord in coords:
            assert len(coord) == 4
        if not is_expert:
            sizes = self._rank_generator.ordered_size
            # Skip the axes related to expert parallelism
            # given the order tp-cp-ep-dp-pp --> tp-cp-dp-pp
            new_sizes = [val for idx, val in enumerate(sizes) if idx != 3]
        else:
            sizes = self._expert_rank_generator.ordered_size
            # Skip the axes related to cp parallelism
            # given the order tp-cp-ep-dp-pp --> tp-ep-dp-pp
            new_sizes = [val for idx, val in enumerate(sizes) if idx != 2]
        assert len(new_sizes) == len(coords[0]), f"new_sizes: {new_sizes}, coords[0]: {coords[0]}"
        strides = _prefix_product(new_sizes)
        logical_ranks = []
        for coord in coords:
            logical_rank = sum([c * s for c, s in zip(coord, strides)]) + self._offset
            logical_ranks.append(logical_rank)
        ranks = self._rank_mapper.to_physical_ranks(logical_ranks)
        return ranks


class ParallelContext:
    def __init__(self, args):
        assert args.context_parallel_size == 1, "Context parallelism is not supported."
        assert args.num_distributed_optimizer_instances == 1, "Distributed optimizer is not supported."
        assert torch.distributed.is_initialized()
        self._is_initialized = False
        self._args = args
        self._current_process_mesh_index = 0
        self._process_meshes = []
        self._rank_to_process_mesh = {}
        self._inter_mesh_group_ranks = defaultdict(list)
        self._inter_mesh_process_groups_pp = {} # (src_rank, dst_rank) -> bool
        self._inter_mesh_process_groups_dp = {} # (src_rank, dst_rank) -> bool
        self._inter_mesh_process_groups_edp = {} # (src_rank, dst_rank) -> bool
        # (src_rank, local_tensor_shape, next) -> (dst_rank, (dp_start, dp_end), (sp_start, sp_end), local_hidden_size)
        self._inter_mesh_tensor_slices = {}
        self._inter_mesh_tensor_slices_for_embd_group = {}

        self._global_group_ranks = defaultdict(list) # current rank: {group_name: [[ranks0], [ranks1], ...], ...}
        self._global_all_group_ranks = defaultdict(list) # all_rank: {group_name -> [[rank0], [rank1], [rank2], ...], ...}
        self._global_process_groups = defaultdict(list) # current rank: {group_name: [group0, group1, ...], ...}
        self._global_process_group_to_ranks = {}
        self._global_parallel_world_sizes = {}
        self._global_parallel_ranks = {}
        self._timeout = timedelta(minutes=self._args.distributed_timeout_minutes)

        self._rank = torch.distributed.get_rank()
        self._rank_mapper = RankMapper(args)
        self.build_all_process_meshes()
        self.build_all_inter_mesh_process_groups()
        self.build_global_process_groups()
        from megatron.core.utils import GlobalMemoryBuffer
        self._global_memory_buffer = GlobalMemoryBuffer()

        # Initialize the associated configs
        self._tranformer_config = None
        self._ddp_config = None
        self._optimizer_config = None
        self._optimizer_config_overrides = None
        self._dataset_config = None

        self.build_config()

        self._is_initialized = True

    def is_initialized(self):
        return self._is_initialized

    def build_all_process_meshes(self):
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        logical_rank = self._rank_mapper.to_logical_ranks([rank])[0]
        accumulated_world_size = 0
        process_mesh_idx = 0
        for tp, cp, ep, dp, pp in self._args.hetero_process_meshes:
            if self._args.expert_tensor_parallel_size_per_process_mesh is not None:
                expert_tensor_parallel_size = self._args.expert_tensor_parallel_size_per_process_mesh[process_mesh_idx]
            elif self._args.expert_tensor_parallel_size is None:
                expert_tensor_parallel_size = tp
            else:
                expert_tensor_parallel_size = self._args.expert_tensor_parallel_size
            process_mesh = ProcessMesh(
                data_parallel_size=dp,
                tensor_model_parallel_size=tp,
                pipeline_model_parallel_size=pp,
                context_parallel_size=cp,
                expert_model_parallel_size=ep,
                nccl_communicator_config_path=self._args.nccl_communicator_config_path,
                distributed_timeout_minutes=self._args.distributed_timeout_minutes,
                order='tp-cp-ep-dp-pp' if not self._args.use_tp_pp_dp_mapping else 'tp-pp-dp',
                offset=accumulated_world_size,
                rank_mapper=self._rank_mapper,
                args=self._args,
                expert_tensor_parallel_size=expert_tensor_parallel_size,
            )
            if (
                logical_rank >= accumulated_world_size
                and logical_rank < accumulated_world_size + process_mesh._world_size
            ):
                self._current_process_mesh_index = len(self._process_meshes)
            accumulated_world_size += process_mesh._world_size
            self._process_meshes.append(process_mesh)
            process_mesh_idx += 1

        if world_size != accumulated_world_size:
            raise RuntimeError(
                f"World size mismatch. Expected {world_size}, but got {accumulated_world_size}"
            )

        all_rank_to_process_mesh = [None for _ in range(world_size)]
        cur_rank_to_process_mesh = {
            "rank": rank,
            "process_mesh_idx": self._current_process_mesh_index,
        }
        torch.distributed.all_gather_object(
            all_rank_to_process_mesh, cur_rank_to_process_mesh
        )
        for item in all_rank_to_process_mesh:
            self._rank_to_process_mesh[item["rank"]] = self._process_meshes[
                item["process_mesh_idx"]
            ]

    def build_inter_mesh_process_groups(self, process_mesh1, process_mesh2):
        tp1 = process_mesh1.get_parallel_size("tp", is_expert=False)
        cp1 = process_mesh1.get_parallel_size("cp", is_expert=False)
        dp1 = process_mesh1.get_parallel_size("dp", is_expert=False)
        tp2 = process_mesh2.get_parallel_size("tp", is_expert=False)
        cp2 = process_mesh2.get_parallel_size("cp", is_expert=False)
        dp2 = process_mesh2.get_parallel_size("dp", is_expert=False)

        if not(tp1 == 1 and tp2 == 1):
            sp1 = tp1 * cp1
            sp2 = tp2 * cp2
        else:
            sp1 = cp1
            sp2 = cp2
        sp_overlapped_mapping = find_overlapped_mapping(sp1, sp2)
        dp_overlapped_mapping = find_overlapped_mapping(dp1, dp2)
        src_pp_dims = [process_mesh1.get_parallel_size("pp") - 1]
        dst_pp_dims = [0]

        # find pp group connection
        for s in range(sp1):
            # i is tp, j is cp, k is dp,
            src_i, src_j = s % tp1, s // tp1
            for k in range(dp1):
                src_coord = [src_i, src_j, k, src_pp_dims[0]]
                dst_sp_dims = [dim for dim, _, _ in sp_overlapped_mapping[s]]
                dst_dp_dims = [dim for dim, _, _ in dp_overlapped_mapping[k]]
                dst_coords = list(
                    itertools.product(dst_sp_dims, dst_dp_dims, dst_pp_dims)
                )
                src_rank = process_mesh1.logical_coords_to_physical_ranks(
                    [src_coord]
                )[0]
                for dst_coord in dst_coords:
                    sp_dim, dp_dim, pp_dim = dst_coord
                    dst_coord = [sp_dim % tp2, sp_dim // tp2, dp_dim, pp_dim]
                    dst_rank = process_mesh2.logical_coords_to_physical_ranks(
                        [dst_coord]
                    )[0]
                    # NOTE: There is no need to create a group for the commnetting boundary.
                    #       We will create the `pp` group in the `build_global_process_groups` function.
                    # ranks = [src_rank, dst_rank]
                    # timeout = max(process_mesh1._timeout, process_mesh2._timeout)
                    # group = torch.distributed.new_group(ranks, timeout=timeout)
                    self._inter_mesh_process_groups_pp[(src_rank, dst_rank)] = True

        # find mp(tp+pp) group connection
        for k in range(dp1):
            src_coord = [tp1 - 1, cp1 - 1, k, src_pp_dims[0]]
            dst_dp_dims = [dim for dim, _, _ in dp_overlapped_mapping[k]]
            dst_coords = list(
                itertools.product([0], [0], dst_dp_dims, dst_pp_dims)
            )
            src_rank = process_mesh1.logical_coords_to_physical_ranks(
                [src_coord]
            )[0]
            for dst_coord in dst_coords:
                dst_rank = process_mesh2.logical_coords_to_physical_ranks(
                    [dst_coord]
                )[0]
                self._inter_mesh_process_groups_dp[(src_rank, dst_rank)] = True

        # ep-related process groups
        etp1 = process_mesh1.get_parallel_size("tp", is_expert=True)
        ep1 = process_mesh1.get_parallel_size("ep", is_expert=True)
        edp1 = process_mesh1.get_parallel_size("dp", is_expert=True)

        edp2 = process_mesh2.get_parallel_size("dp", is_expert=True)
        edp_overlapped_mapping = find_overlapped_mapping(edp1, edp2)
        src_pp_dims = [process_mesh1.get_parallel_size("pp") - 1]
        dst_pp_dims = [0]
        # find tp+ep+pp group connection
        for k in range(edp1):
            src_coord = [etp1 - 1, ep1 - 1, k, src_pp_dims[0]]
            dst_edp_dims = [dim for dim, _, _ in edp_overlapped_mapping[k]]
            dst_coords = list(
                itertools.product([0], [0], dst_edp_dims, dst_pp_dims)
            )
            src_rank = process_mesh1.logical_coords_to_physical_ranks(
                [src_coord], is_expert=True
            )[0]
            for dst_coord in dst_coords:
                dst_rank = process_mesh2.logical_coords_to_physical_ranks(
                    [dst_coord], is_expert=True
                )[0]
                self._inter_mesh_process_groups_edp[(src_rank, dst_rank)] = True

    def build_all_inter_mesh_process_groups(self):
        if len(self._process_meshes) == 1:
            return

        for i in range(len(self._process_meshes) - 1):
            self.build_inter_mesh_process_groups(
                self._process_meshes[i], self._process_meshes[i + 1]
            )
    def build_intra_dist_opt_process_groups(self):
        """TODO: Support intra-dist-opt process groups for partial data parallelism """
        pass

    def build_global_process_groups(self):
        """ Build global process groups across all process meshes. The global process groups are used for the communication
            between different pipeline stages. Heteregonous process groups except for the default process groups are all here"""
        # build global pipeline process groups
        def _backtrack(mesh_index, prev_rank, path, token = "pp", is_expert=False):
            assert token in ["tp-pp", "pp", "tp-ep-pp"], f"Invalid token: {token} for inter-mesh process groups"
            group_name = get_group_name(token, is_expert=is_expert)
            if mesh_index == len(self._process_meshes):
                aggregated_ranks = [rank for ranks in path for rank in ranks]
                self._global_all_group_ranks[group_name].append(aggregated_ranks)
                # NOTE: "use_local_synchronization=True" works well in torhch <= 2.5, but it causes hang in torch >= 2.6
                group = create_group(aggregated_ranks, timeout=self._timeout, use_local_synchronization=False, group_desc=group_name)
                if self._rank in aggregated_ranks:
                    self._global_process_groups[group_name].append(group)
                    self._global_group_ranks[group_name].append(aggregated_ranks)
                    self._global_process_group_to_ranks[group] = aggregated_ranks
                return
            current_mesh = self._process_meshes[mesh_index]
            ranks_list = current_mesh.get_all_process_group_ranks(token, is_expert=is_expert, check_initialized=True)
            valid_ranks_list = []
            for ranks in ranks_list:
                mesh_is_connected = False
                for prev_path_ranks in path:
                    for prev_path_rank in prev_path_ranks:
                        if token == "pp" and (prev_path_rank, ranks[0]) in self._inter_mesh_process_groups_pp:
                            mesh_is_connected = True
                        elif token == "tp-pp" and (prev_path_rank, ranks[0]) in self._inter_mesh_process_groups_dp:
                            mesh_is_connected = True
                        elif token == "tp-ep-pp" and (prev_path_rank, ranks[0]) in self._inter_mesh_process_groups_edp:
                            mesh_is_connected = True
                if prev_rank == -1 or mesh_is_connected:
                    valid_ranks_list.append(ranks)
            for ranks in valid_ranks_list:
                path.append(ranks)
                _backtrack(mesh_index + 1, ranks[-1], path, token=token, is_expert=is_expert)
                path.pop()
        # build the global process groups which across the different Processmesh
        _backtrack(0, -1, path=[], token="tp-pp", is_expert=False)
        _backtrack(0, -1, path=[], token="pp", is_expert=False)
        _backtrack(0, -1, path=[], token="tp-ep-pp", is_expert=True)

        # 'last_rank' is the last rank of the last pipeline stage
        pp_ranks = self.get_global_all_group_ranks("pp")
        self._global_parallel_ranks["last_rank"] = pp_ranks[-1][-1] if isinstance(pp_ranks[0], list) else pp_ranks[-1]

        # build global embedding process groups
        for ranks in self._global_all_group_ranks["pp"]: # NOTE: Make sure all the ranks to execute the "create_group" API.
            if len(ranks) > 1:
                embedding_ranks = [ranks[0], ranks[-1]]
                position_embedding_ranks = [ranks[0]]
                # `pp_split` is similar to the `pipeline_model_parallel_split_rank` in parallel_state
                if "pp_split" in self._global_parallel_ranks.keys() and self._global_parallel_ranks["pp_split"] is not None:
                    split_rank = self._global_parallel_ranks["pp_split"]
                    if ranks[split_rank] not in embedding_ranks:
                        embedding_ranks = [
                            ranks[0],
                            ranks[split_rank],
                            ranks[-1],
                        ]
                    if ranks[split_rank] not in position_embedding_ranks:
                        position_embedding_ranks = [
                            ranks[0],
                            ranks[split_rank],
                        ]
            else:
                embedding_ranks = ranks
                position_embedding_ranks = ranks
            group = create_group(embedding_ranks, timeout=self._timeout, use_local_synchronization=False, group_desc="embd")
            if self._rank in embedding_ranks and ("embd" not in self._global_group_ranks or embedding_ranks not in self._global_group_ranks["embd"]):
                self._global_process_groups["embd"].append(group)
                self._global_process_group_to_ranks[group] = embedding_ranks
                self._global_group_ranks["embd"].append(embedding_ranks)

            group = create_group(position_embedding_ranks, timeout=self._timeout, use_local_synchronization=False, group_desc="embd_pos")
            if self._rank in position_embedding_ranks:
                self._global_process_groups["embd_pos"].append(group)
                self._global_process_group_to_ranks[group] = position_embedding_ranks

            if self._rank in ranks:
                self._global_group_ranks["embd_pos"].append(position_embedding_ranks)

    def get_inter_mesh_process_group(self, src_rank, dst_rank):
        if (src_rank, dst_rank) in self._inter_mesh_process_groups_pp:
            return self._inter_mesh_process_groups_pp[(src_rank, dst_rank)]
        elif (dst_rank, src_rank) in self._inter_mesh_process_groups_pp:
            return self._inter_mesh_process_groups_pp[(dst_rank, src_rank)]
        else:
            raise RuntimeError(
                f"ProcessGroup [{src_rank}, {dst_rank}] does not exist."
            )

    def get_inter_mesh_tensor_slices(self, rank, local_tensor_shape, next=True):
        if (rank, local_tensor_shape, next) in self._inter_mesh_tensor_slices:
            return self._inter_mesh_tensor_slices[(rank, local_tensor_shape, next)]
        process_mesh1 = self._process_meshes[self._current_process_mesh_index]
        if next:
            process_mesh2 = self.get_next_process_mesh()
            # first stage of the next process mesh
            src_pp_dims = [process_mesh1.get_parallel_size("pp", is_expert=False) - 1]
            dst_pp_dims = [0]
        else:
            process_mesh2 = self.get_prev_process_mesh()
            # last stage of the previous process mesh
            src_pp_dims = [0]
            dst_pp_dims = [process_mesh2.get_parallel_size("pp") - 1]
        tp1 = process_mesh1.get_parallel_size("tp", is_expert=False)
        cp1 = process_mesh1.get_parallel_size("cp", is_expert=False)
        dp1 = process_mesh1.get_parallel_size("dp", is_expert=False)
        tp2 = process_mesh2.get_parallel_size("tp", is_expert=False)
        cp2 = process_mesh2.get_parallel_size("cp", is_expert=False)
        dp2 = process_mesh2.get_parallel_size("dp", is_expert=False)

        # Assume that the tensor shape is (seq_len, batch_size, hidden_size)
        local_seq_len, local_batch_size, local_hidden_size = local_tensor_shape
        if not(tp1 == 1 and tp2 == 1):
            global_seq_len = local_seq_len * tp1 * cp1
            sp1 = tp1 * cp1
            sp2 = tp2 * cp2
        else:
            global_seq_len = local_seq_len * cp1
            sp1 = cp1
            sp2 = cp2
        global_batch_size = local_batch_size * dp1
        sp_overlapped_mapping = find_overlapped_mapping(sp1, sp2, global_seq_len)
        dp_overlapped_mapping = find_overlapped_mapping(dp1, dp2, global_batch_size)
        for s in range(sp1):
            src_i, src_j = s % tp1, s // tp1
            for k in range(dp1):
                src_coord = [src_i, src_j, k, src_pp_dims[0]]
                dst_sp_dims = [c for c, _, _ in sp_overlapped_mapping[s]]
                dst_dp_dims = [c for c, _, _ in dp_overlapped_mapping[k]]
                dst_coords = list(
                    itertools.product(dst_sp_dims, dst_dp_dims, dst_pp_dims)
                )
                src_sp_starts = [s for _, s, _ in sp_overlapped_mapping[s]]
                src_dp_starts = [s for _, s, _ in dp_overlapped_mapping[k]]
                src_starts = list(itertools.product(src_sp_starts, src_dp_starts))
                src_sp_ends = [e for _, _, e in sp_overlapped_mapping[s]]
                src_dp_ends = [e for _, _, e in dp_overlapped_mapping[k]]
                src_ends = list(itertools.product(src_sp_ends, src_dp_ends))
                src_rank = process_mesh1.logical_coords_to_physical_ranks([src_coord])[0]
                for i, dst_coord in enumerate(dst_coords):
                    sp_dim, dp_dim, pp_dim = dst_coord
                    dst_coord = [sp_dim % tp2, sp_dim // tp2, dp_dim, pp_dim]
                    dst_rank = process_mesh2.logical_coords_to_physical_ranks([dst_coord])[0]
                    sp_start, dp_start = src_starts[i]
                    sp_end, dp_end = src_ends[i]
                    if (src_rank, local_tensor_shape, next) not in self._inter_mesh_tensor_slices:
                        self._inter_mesh_tensor_slices[(src_rank, local_tensor_shape, next)] = []
                    self._inter_mesh_tensor_slices[
                        (src_rank, local_tensor_shape, next)
                    ].append(
                        (
                            dst_rank,
                            (dp_start, dp_end),
                            (sp_start, sp_end),
                            local_hidden_size,
                        )
                    )
        return self._inter_mesh_tensor_slices[(rank, local_tensor_shape, next)]

    def get_dp_coef_when_recv_backward(self) -> float:
        if self._args.calculate_per_token_loss:
            return 1.0
        recv_rank_dp_size = self.get_current_process_mesh().get_parallel_size("dp", is_expert=False)
        send_rank_dp_size = self.get_next_process_mesh().get_parallel_size("dp", is_expert=False)
        if recv_rank_dp_size == send_rank_dp_size:
            return 1.0
        return float(recv_rank_dp_size) / float(send_rank_dp_size)

    def get_current_process_mesh(self):
        assert self._current_process_mesh_index < len(self._process_meshes)
        return self._process_meshes[self._current_process_mesh_index]

    def get_prev_process_mesh(self):
        assert self._current_process_mesh_index - 1 >= 0
        return self._process_meshes[self._current_process_mesh_index - 1]

    def get_next_process_mesh(self):
        assert self._current_process_mesh_index + 1 < len(self._process_meshes)
        return self._process_meshes[self._current_process_mesh_index + 1]

    def get_global_process_group(self, token, is_expert=False, check_initialized=False):
        group_name = get_group_name(token, is_expert=is_expert)
        group = self._global_process_groups.get(group_name, None)
        if check_initialized:
            assert (
                group is not None
            ), f"Process group {group_name} is not initialized."
        return group

    def get_global_group_ranks(self, token, is_expert=False, check_initialized=False):
        group_name = get_group_name(token, is_expert=is_expert)
        ranks = self._global_group_ranks.get(group_name, None)
        if check_initialized:
            assert (
                ranks is not None
            ), f"Process group {group_name} is not initialized."
        return ranks

    def get_global_all_group_ranks(self, token, is_expert=False, check_initialized=False):
        group_name = get_group_name(token, is_expert=is_expert)
        ranks = self._global_all_group_ranks.get(group_name, None)
        if check_initialized:
            assert (
                ranks is not None
            ), f"Process group {group_name} is not initialized."
        return ranks

    def get_model_parallel_group(self, check_initialized=True):
        """Get the model parallel group the caller rank belongs to."""
        group = self.get_global_process_group("tp-pp", is_expert=False, check_initialized=True)
        if check_initialized:
            assert group is not None, 'model parallel group is not initialized'
        return group

    def get_model_parallel_src_rank(self):
        """Calculate the global rank corresponding to the first local rank
        in the model parallel group."""
        ranks = self.get_global_group_ranks("tp-pp", is_expert=False, check_initialized=True)
        return ranks[0]

    def get_tensor_model_parallel_group(self, check_initialized=True):
        """Get the tensor model parallel group the caller rank belongs to."""
        current_process_mesh = self._process_meshes[self._current_process_mesh_index]
        return current_process_mesh.get_process_group(
            "tp", is_expert=False, gloo=False, check_initialized=check_initialized
        )

    def get_pipeline_model_parallel_group(self, check_initialized=True, local_pp_group=False):
        """Get the pipeline model parallel group the caller rank belongs to."""
        if local_pp_group:
            current_process_mesh = self._process_meshes[self._current_process_mesh_index]
            return current_process_mesh.get_process_group("pp", is_expert=False, gloo=False, check_initialized=check_initialized)
        group = self._global_process_groups.get("pp", None)
        if check_initialized:
            assert group is not None, "pipeline_model parallel group is not initialized"
        return self._global_process_groups["pp"]

    def get_data_parallel_group(self, with_context_parallel=False, partial_data_parallel=False):
        """Get the data parallel group the caller rank belongs to."""
        current_process_mesh = self._process_meshes[self._current_process_mesh_index]
        if with_context_parallel:
            if partial_data_parallel:
                return current_process_mesh.get_process_group(
                    "intra-dp-cp", is_expert=False, gloo=False, check_initialized=True
                )
            return current_process_mesh.get_process_group(
                "dp-cp", is_expert=False, gloo=False, check_initialized=True
            )
        else:
            return current_process_mesh.get_process_group(
                "dp", is_expert=False, gloo=False, check_initialized=True
            )

    def get_data_parallel_group_gloo(self, with_context_parallel=False, partial_data_parallel=False):
        """Get the data parallel group-gloo the caller rank belongs to."""
        current_process_mesh = self._process_meshes[self._current_process_mesh_index]
        if with_context_parallel:
            if partial_data_parallel:
                return current_process_mesh.get_process_group(
                    "intra-dp-cp", is_expert=False, gloo=True, check_initialized=True
                )
            return current_process_mesh.get_process_group(
                "dp-cp", is_expert=False, gloo=True, check_initialized=True
            )
        else:
            return current_process_mesh.get_process_group(
                "dp", is_expert=False, gloo=True, check_initialized=True
            )

    def get_inter_partial_data_parallel_group(self):
        """Get the group spanning the different partial data-parallel groups."""
        current_process_mesh = self._process_meshes[self._current_process_mesh_index]
        return current_process_mesh.get_process_group(
            "inter-dp-cp", is_expert=False, gloo=False, check_initialized=True
        )

    def get_context_parallel_group(self, check_initialized=True):
        """Get the context parallel group the caller rank belongs to."""
        current_process_mesh = self._process_meshes[self._current_process_mesh_index]
        return current_process_mesh.get_process_group(
            "cp", is_expert=False, gloo=False, check_initialized=check_initialized
        )

    def get_context_parallel_global_ranks(self, check_initialized=True):
        """Get all global ranks of the context parallel group that the caller rank belongs to."""
        current_process_mesh = self._process_meshes[self._current_process_mesh_index]
        return current_process_mesh.get_process_group_ranks(
            "cp", is_expert=False, check_initialized=check_initialized
        )

    def get_hierarchical_context_parallel_groups(self, check_initialized=True):
        """Get the inner ring of context parallel group the caller rank belongs to."""
        current_process_mesh = self._process_meshes[self._current_process_mesh_index]
        return current_process_mesh.get_process_group(
            "hierarchical-cp", is_expert=False, gloo=False, check_initialized=check_initialized
        )

    def get_embedding_group(self, check_initialized=True):
        """Get the embedding group the caller rank belongs to."""
        groups = self._global_process_groups.get("embd", None)
        if check_initialized:
            assert groups is not None, 'embedding group is not initialized'
        return groups

    def get_position_embedding_group(self, check_initialized=True):
        """Get the position embedding group the caller rank belongs to."""
        groups = self._global_process_groups.get("embd_pos", None)
        if check_initialized:
            assert groups is not None, 'Position embedding group is not initialized'
        if groups is None:
            return None
        for group in groups:
            if self._rank in self._global_process_group_to_ranks[group]:
                pos_embd_group = group
                break
        return pos_embd_group

    def get_amax_reduction_group(self, with_context_parallel=False, tp_only_amax_red=False):
        """Get the FP8 amax reduction group the caller rank belongs to."""
        current_process_mesh = self._process_meshes[self._current_process_mesh_index]
        if with_context_parallel:
            if not tp_only_amax_red:
                return current_process_mesh.get_process_group(
                    "tp-dp-cp", is_expert=False, gloo=False, check_initialized=True
                )
            else:
                return current_process_mesh.get_process_group(
                    "tp-cp", is_expert=False, gloo=False, check_initialized=True
                )
        else:
            if not tp_only_amax_red:
                return current_process_mesh.get_process_group(
                    "tp-dp", is_expert=False, gloo=False, check_initialized=True
                )
            else:
                return current_process_mesh.get_process_group(
                    "tp", is_expert=False, gloo=False, check_initialized=True
                )

    def get_tensor_and_data_parallel_group(self, with_context_parallel=False):
        """Get the tensor and data parallel group the caller rank belongs to."""
        current_process_mesh = self._process_meshes[self._current_process_mesh_index]
        if with_context_parallel:
            return current_process_mesh.get_process_group(
                "tp-dp-cp", is_expert=False, gloo=False, check_initialized=True
            )
        else:
            return current_process_mesh.get_process_group(
                "tp-dp", is_expert=False, gloo=False, check_initialized=True
            )

    def get_tensor_and_context_parallel_group(self, check_initialized=True):
        """Get the tensor- and context-parallel group the caller rank belongs to."""
        current_process_mesh = self._process_meshes[self._current_process_mesh_index]
        return current_process_mesh.get_process_group(
            "tp-cp", is_expert=False, gloo=False, check_initialized=check_initialized
        )

    def set_tensor_model_parallel_world_size(self, world_size):
        """Set the tensor model parallel size"""
        self._global_parallel_world_sizes["tp"] = world_size

    def set_pipeline_model_parallel_world_size(self, world_size):
        """Set the pipeline model parallel size"""
        self._global_parallel_world_sizes["pp"] = world_size

    def set_virtual_pipeline_model_parallel_world_size(self, world_size):
        """Set the pipeline model parallel size"""
        self._global_parallel_world_sizes["vpp"] = world_size

    def get_tensor_model_parallel_world_size(self):
        """Return world size for the tensor model parallel group."""
        size = self._global_parallel_world_sizes.get("tp", None)
        if size is not None:
            return size
        return torch.distributed.get_world_size(group=self.get_tensor_model_parallel_group())

    def get_pipeline_model_parallel_world_size(self, group=None):
        """Return world size for the pipeline model parallel group."""
        size = self._global_parallel_world_sizes.get("pp", None)
        if size is not None:
            return size
        if group is None:
            group = self.get_pipeline_model_parallel_group()[0]
        return torch.distributed.get_world_size(group)

    def set_tensor_model_parallel_rank(self, rank):
        """Set tensor model parallel rank."""
        self._global_parallel_ranks["tp"] = rank

    def set_pipeline_model_parallel_rank(self, rank):
        """Set pipeline model parallel rank."""
        self._global_parallel_ranks["pp"] = rank

    def set_pipeline_model_parallel_split_rank(self, rank):
        """Set pipeline model parallel split rank."""
        self._global_parallel_ranks["pp-split"] = rank

    def get_tensor_model_parallel_rank(self):
        """Return my rank for the tensor model parallel group."""
        rank = self._global_parallel_ranks.get("tp", None)
        if rank is not None:
            return rank
        return torch.distributed.get_rank(group=self.get_tensor_model_parallel_group())

    def get_pipeline_model_parallel_rank(self, group=None):
        """Return my rank for the pipeline model parallel group."""
        rank = self._global_parallel_ranks.get("pp", None)
        if rank is not None:
            return rank
        if group is None:
            group = self.get_pipeline_model_parallel_group()[0]
        if group not in self._global_process_group_to_ranks: # local pipeline group
            return torch.distributed.get_rank(group=group)
        else:
            ranks = self._global_process_group_to_ranks[group]
            return ranks.index(self._rank)

    def get_pipeline_model_parallel_split_rank(self):
        """Return pipeline model parallel split rank."""
        return self._global_parallel_ranks.get("pp-split", None)

    def is_pipeline_first_stage(self, ignore_virtual=False, group=None):
        """Return True if in the first pipeline model-parallel stage, False otherwise."""
        if not ignore_virtual:
            if (
                self.get_virtual_pipeline_model_parallel_world_size() is not None
                and self.get_virtual_pipeline_model_parallel_rank() != 0
            ):
                return False
        return self.get_pipeline_model_parallel_rank(group) == 0

    def is_pipeline_last_stage(self, ignore_virtual=False, group=None):
        """Return True if in the last pipeline model-parallel stage, False otherwise."""
        if not ignore_virtual:
            virtual_pipeline_model_parallel_world_size = (
                self.get_virtual_pipeline_model_parallel_world_size()
            )
            if (
                virtual_pipeline_model_parallel_world_size is not None
                and self.get_virtual_pipeline_model_parallel_rank()
                != (virtual_pipeline_model_parallel_world_size - 1)
            ):
                return False
        return self.get_pipeline_model_parallel_rank(group) == (self.get_pipeline_model_parallel_world_size(group) - 1)

    def is_rank_in_embedding_group(self, ignore_virtual=False, group=None):
        """Return true if current rank is in embedding group, False otherwise."""
        rank = torch.distributed.get_rank()
        if group is None:
            group = self._global_process_groups.get("embd", None)
            if group is None:
                return False
            else:
                group = group[0]
        ranks = self._global_process_group_to_ranks[group]
        ignore_virtual = True
        if ignore_virtual:
            return rank in ranks
        if rank in ranks:
            if rank == ranks[0]:
                return self.is_pipeline_first_stage(ignore_virtual=False, group=group)
            elif rank == ranks[-1]:
                return self.is_pipeline_last_stage(ignore_virtual=False, group=group)
            else:
                return True
        return False

    def is_rank_in_position_embedding_group(self, group=None):
        """Return true if current rank is in position embedding group, False otherwise."""
        rank = torch.distributed.get_rank()
        if group is None:
            group = self._global_process_groups.get("embd_pos", None)
            if group is None:
                return False
            else:
                group = group[0]
        ranks = self._global_process_group_to_ranks[group]
        return rank in ranks

    def is_pipeline_stage_before_split(self, rank=None, group=None):
        """Return True if pipeline stage executes encoder block for a model
        with both encoder and decoder."""
        if self.get_pipeline_model_parallel_world_size(group) == 1:
            return True
        if rank is None:
            rank = self.get_pipeline_model_parallel_rank(group)
        split_rank = self.get_pipeline_model_parallel_split_rank()
        if split_rank is None:
            return True
        if rank < split_rank:
            return True
        return False

    def is_pipeline_stage_after_split(self, rank=None, group=None):
        """Return True if pipeline stage executes decoder block for a model
        with both encoder and decoder."""
        if self.get_pipeline_model_parallel_world_size(group) == 1:
            return True
        if rank is None:
            rank = self.get_pipeline_model_parallel_rank(group)
        split_rank = self.get_pipeline_model_parallel_split_rank()
        if split_rank is None:
            return True
        if rank >= split_rank:
            return True
        return False

    def is_inside_encoder(self, rank=None) -> bool:
        """Return True if pipeline stage executes encoder block.
        This function implicitly assumes we have a model with both
        encoder and decoder."""
        if self.get_pipeline_model_parallel_world_size() == 1:
            return True
        if rank is None:
            rank = self.get_pipeline_model_parallel_rank()
        pipeline_model_parallel_decoder_start = self.get_pipeline_model_parallel_decoder_start()
        # _PIPELINE_MODEL_PARALLEL_DECODER_START == None means that the
        # encoder shares the first pipeline rank with the decoder
        if pipeline_model_parallel_decoder_start is None and rank == 0:
            return True
        # _PIPELINE_MODEL_PARALLEL_DECODER_START != None means that the
        # encoder is on it's own pipeline ranks before the decoder
        if (
            pipeline_model_parallel_decoder_start is not None
            and rank < pipeline_model_parallel_decoder_start
        ):
            return True
        return False

    def is_inside_decoder(self, rank=None) -> bool:
        """Return True if pipeline stage executes decoder block for a model
        with both encoder and decoder."""
        if self.get_pipeline_model_parallel_world_size() == 1:
            return True
        if rank is None:
            rank = self.get_pipeline_model_parallel_rank()
        pipeline_model_parallel_decoder_start = self.get_pipeline_model_parallel_decoder_start()
        if pipeline_model_parallel_decoder_start is None:
            return True
        if rank >= pipeline_model_parallel_decoder_start:
            return True
        return False

    def get_pipeline_model_parallel_decoder_start(self) -> int:
        """Return decoder start rank (if encoder pipeline parallelism is set)."""
        return self._global_parallel_ranks.get("pp-decoder-start", None)

    def is_pipeline_stage_at_split(self, group=None):
        """Return true if pipeline stage executes decoder block and next
        stage executes encoder block for a model with both encoder and
        decoder."""
        rank = self.get_pipeline_model_parallel_rank(group)
        return self.is_pipeline_stage_before_split(rank, group) and self.is_pipeline_stage_after_split(rank + 1, group)

    def get_virtual_pipeline_model_parallel_rank(self):
        """Return the virtual pipeline-parallel rank."""
        return self._global_parallel_ranks.get("vpp", None)

    def set_virtual_pipeline_model_parallel_rank(self, rank):
        """Set the virtual pipeline-parallel rank."""
        self._global_parallel_ranks["vpp"] = rank

    def get_virtual_pipeline_model_parallel_world_size(self):
        """Return the virtual pipeline-parallel world size."""
        return self._global_parallel_world_sizes.get("vpp", None)

    def get_tensor_model_parallel_src_rank(self):
        """Calculate the global rank corresponding to the first local rank
        in the tensor model parallel group."""
        current_process_mesh = self._process_meshes[self._current_process_mesh_index]
        ranks = current_process_mesh.get_process_group_ranks(
            "tp", is_expert=False, check_initialized=True
        )
        return ranks[0]

    def get_data_parallel_src_rank(self, with_context_parallel=False):
        """Calculate the global rank corresponding to the first local rank
        in the data parallel group."""
        current_process_mesh = self._process_meshes[self._current_process_mesh_index]
        if with_context_parallel:
            ranks = current_process_mesh.get_process_group_ranks(
                "dp-cp", is_expert=False, check_initialized=True
            )
        else:
            ranks = current_process_mesh.get_process_group_ranks(
                "dp", is_expert=False, check_initialized=True
            )
        return ranks[0]

    def get_pipeline_model_parallel_first_rank(self, group=None):
        """Return the global rank of the first process in the pipeline for the
        current tensor parallel group"""
        if group is None:
            group = self.get_pipeline_model_parallel_group()[0]
        ranks = self._global_process_group_to_ranks.get(group, None)
        assert ranks is not None, "Pipeline parallel group is not initialized"
        return ranks[0]

    def get_pipeline_model_parallel_last_rank(self, group=None):
        """Return the global rank of the last process in the pipeline for the
        current tensor parallel group"""
        if group is None:
            group = self.get_pipeline_model_parallel_group()[0]
        ranks = self._global_process_group_to_ranks.get(group, None)
        assert ranks is not None, "Pipeline parallel group is not initialized"
        last_rank_local = self.get_pipeline_model_parallel_world_size(group) - 1
        return ranks[last_rank_local]

    def get_pipeline_model_parallel_next_rank(self, group=None):
        """Return the global rank that follows the caller in the pipeline"""
        if group is None:
            group = self.get_pipeline_model_parallel_group()[0]
        ranks = self._global_process_group_to_ranks.get(group, None)
        if ranks is None: # local pipeline group
            current_process_mesh = self._process_meshes[self._current_process_mesh_index]
            ranks = current_process_mesh.get_process_group_ranks(token="pp", is_expert=False, check_initialized=True)
            assert ranks is not None, "Pipeline parallel group is not initialized"
            rank_in_pipeline = torch.distributed.get_rank(group=group)
            world_size = self.get_pipeline_model_parallel_world_size(group)
            return ranks[(rank_in_pipeline + 1) % world_size]
        rank_in_pipeline = ranks.index(self._rank)
        world_size = self.get_pipeline_model_parallel_world_size(group)
        return ranks[(rank_in_pipeline + 1) % world_size]

    def get_pipeline_model_parallel_prev_rank(self, group=None):
        """Return the global rank that preceeds the caller in the pipeline"""
        if group is None:
            group = self.get_pipeline_model_parallel_group()[0]
        ranks = self._global_process_group_to_ranks.get(group, None)
        if ranks is None: # local pipeline group
            current_process_mesh = self._process_meshes[self._current_process_mesh_index]
            ranks = current_process_mesh.get_process_group_ranks(token="pp", is_expert=False, check_initialized=True)
            assert ranks is not None, "Pipeline parallel group is not initialized"
            rank_in_pipeline = torch.distributed.get_rank(group=group)
            world_size = self.get_pipeline_model_parallel_world_size(group)
            return ranks[(rank_in_pipeline - 1) % world_size]
        rank_in_pipeline = ranks.index(self._rank)
        world_size = self.get_pipeline_model_parallel_world_size(group)
        return ranks[(rank_in_pipeline - 1) % world_size]

    def get_last_rank_when_using_pipeline(self):
        """Return the global rank of the last process in the pipeline"""
        assert (
            self._global_parallel_ranks.get("last_rank", None) is not None
        ), "Last rank when using pipeline is not initialized"
        return self._global_parallel_ranks["last_rank"]

    def get_data_parallel_world_size(
        self,
        with_context_parallel=False,
        partial_data_parallel=False,
    ):
        """Return world size for the data parallel group."""
        size = self._global_parallel_world_sizes.get("dp", None)
        if size is not None:
            return size
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_world_size(
                group=self.get_data_parallel_group(
                    with_context_parallel=with_context_parallel,
                    partial_data_parallel=partial_data_parallel,
                )
            )
        else:
            return 0

    def set_data_parallel_rank(self, rank):
        """Return world size for the data parallel group."""
        self._global_parallel_ranks["dp"] = rank

    def get_data_parallel_rank(
        self,
        with_context_parallel=False,
        partial_data_parallel=False,
    ):
        """Return my rank for the data parallel group."""
        rank = self._global_parallel_ranks.get("dp", None)
        if rank is not None:
            return rank
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank(
                group=self.get_data_parallel_group(
                    with_context_parallel=with_context_parallel,
                    partial_data_parallel=partial_data_parallel,
                )
            )
        else:
            return 0

    def get_context_parallel_world_size(self):
        """Return world size for the context parallel group."""
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_world_size(group=self.get_context_parallel_group())
        else:
            return 0

    def get_context_parallel_rank(self):
        """Return my rank for the context parallel group."""
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank(group=self.get_context_parallel_group())
        else:
            return 0

    def get_tensor_and_context_parallel_world_size(self):
        """Return world size for the tensor and context-parallel group."""
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_world_size(group=self.get_tensor_and_context_parallel_group())
        else:
            return 0

    def get_intra_distributed_optimizer_instance_group(self):
        return torch.distributed.group.WORLD

    def get_tensor_and_context_parallel_rank(self):
        """Return caller's rank in the joint tensor-model-parallel and context-parallel group."""
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank(group=self.get_tensor_and_context_parallel_group())
        else:
            return 0

    ### Expert-related parallel states functions
    def get_expert_model_parallel_group(self, check_initialized=True):
        """Get the expert-model-parallel group the caller rank belongs to."""
        current_process_mesh = self._process_meshes[self._current_process_mesh_index]
        return current_process_mesh.get_process_group(
            "ep", is_expert=True, gloo=False, check_initialized=check_initialized
        )

    def get_expert_model_parallel_world_size(self):
        """Return world size for the expert-model-parallel group."""
        group_name = get_group_name("ep", is_expert=True)
        size = self._global_parallel_world_sizes.get(group_name, None)
        if size is not None:
            return size
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_world_size(group=self.get_expert_model_parallel_group())
        else:
            return 0

    def set_expert_model_parallel_world_size(self, world_size):
        """Sets the expert-model-parallel world size."""
        group_name = get_group_name("ep", is_expert=True)
        self._global_parallel_world_sizes[group_name] = world_size

    def get_expert_model_parallel_rank(self):
        """Return caller's rank in the expert-model-parallel group."""
        group_name = get_group_name("ep", is_expert=True)
        rank = self._global_parallel_ranks.get(group_name, None)
        if rank is not None:
            return rank
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank(group=self.get_expert_model_parallel_group())
        else:
            return 0

    def set_expert_model_parallel_rank(self, rank):
        """Set expert-model-parallel rank."""
        group_name = get_group_name("ep", is_expert=True)
        self._global_parallel_ranks[group_name] = rank

    def get_expert_tensor_parallel_group(self, check_initialized=True):
        """Get the expert-tensor-parallel group the caller rank belongs to."""
        current_process_mesh = self._process_meshes[self._current_process_mesh_index]
        return current_process_mesh.get_process_group(
            "tp", is_expert=True, gloo=False, check_initialized=check_initialized
        )

    def get_expert_tensor_parallel_world_size(self):
        """Return world size for the expert tensor parallel group."""
        group_name = get_group_name("tp", is_expert=True)
        size = self._global_parallel_world_sizes.get(group_name, None)
        if size is not None:
            return size
        # Use tensor parallel group world size for backward compability otherwise
        if not self.get_expert_tensor_parallel_group():
            return self.get_tensor_model_parallel_world_size()
        else:
            return torch.distributed.get_world_size(group=self.get_expert_tensor_parallel_group())

    def set_expert_tensor_parallel_world_size(self, world_size):
        "Set expert tensor model parallel size"
        group_name = get_group_name("tp", is_expert=True)
        self._global_parallel_world_sizes[group_name] = world_size

    def get_expert_tensor_parallel_rank(self):
        """Return my rank for the expert tensor parallel group."""
        group_name = get_group_name("tp", is_expert=True)
        rank = self._global_parallel_ranks.get(group_name, None)
        if rank is not None:
            return rank
        # Use tensor parallel group rank for backward compability otherwise
        if not self.get_expert_tensor_parallel_group():
            return self.get_tensor_model_parallel_rank()
        else:
            return torch.distributed.get_rank(group=self.get_expert_tensor_parallel_group())

    def set_expert_tensor_parallel_rank(self, rank):
        "Set expert tensor model parallel rank"
        group_name = get_group_name("tp", is_expert=True)
        self._global_parallel_ranks[group_name] = rank

    def get_expert_tensor_and_model_parallel_group(self, check_initialized=True):
        """Get the expert-tensor and expert-model group the caller rank belongs to."""
        current_process_mesh = self._process_meshes[self._current_process_mesh_index]
        return current_process_mesh.get_process_group(
            "tp-ep", is_expert=True, gloo=False, check_initialized=check_initialized
        )

    def get_expert_tensor_and_model_parallel_world_size(self):
        """Return world size for the expert model parallel group times expert tensor parallel group."""
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size(
                group=self.get_expert_tensor_and_model_parallel_group()
            )
            return world_size
        else:
            return 0

    def get_expert_tensor_and_model_parallel_rank(self):
        """Return caller's rank in the joint tensor- and expert-model-parallel group."""
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank(group=self.get_expert_tensor_and_model_parallel_group())
        else:
            return 0

    def get_expert_tensor_model_pipeline_parallel_group(self, check_initialized=True):
        """Get expert tensor-model-pipeline parallel group."""
        group = self.get_global_process_group("tp-ep-pp", is_expert=True, check_initialized=True)
        if check_initialized:
            assert group is not None, 'expert tensor-model-pipeline parallel group is not initialized'
        return group

    def get_expert_data_parallel_group(self):
        """Get expert data parallel group."""
        current_process_mesh = self._process_meshes[self._current_process_mesh_index]
        return current_process_mesh.get_process_group(
            "dp", is_expert=True, gloo=False, check_initialized=True)

    def get_data_modulo_expert_parallel_group(self):
        """[Deprecated] Get expert data parallel group."""
        warnings.warn(
            "get_data_modulo_expert_parallel_group is deprecated, please use "
            "get_expert_data_parallel_group instead.",
            DeprecationWarning,
        )
        self.get_expert_data_parallel_group()

    def get_expert_data_parallel_group_gloo(self):
        """Get expert data parallel group-gloo."""
        current_process_mesh = self._process_meshes[self._current_process_mesh_index]
        return current_process_mesh.get_process_group(
            "dp", is_expert=True, gloo=True, check_initialized=True)

    def get_expert_data_parallel_rank(self):
        """Return caller's rank in the expert data parallel group."""
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank(group=self.get_expert_data_parallel_group())
        else:
            return 0

    def get_expert_data_parallel_world_size(self):
        """Return caller's rank in the expert data parallel group."""
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank(group=self.get_expert_data_parallel_group()).size()
        else:
            return 0

    ### End of expert-related functions region

    def set_global_memory_buffer(self):
        """Initialize global buffer"""
        assert self._global_memory_buffer is None, 'global memory buffer is already initialized'
        self._global_memory_buffer = GlobalMemoryBuffer()

    def get_global_memory_buffer(self):
        """Return the global GlobalMemoryBuffer object"""
        assert self._global_memory_buffer is not None, 'global memory buffer is not initialized'
        return self._global_memory_buffer

    def destroy_global_memory_buffer(self):
        """Sets the global memory buffer to None"""
        self._global_memory_buffer = None

    def build_config(self):
        def _build_ddp_config(args):
            from megatron.core.distributed import DistributedDataParallelConfig
            kwargs = {}
            for f in dataclasses.fields(DistributedDataParallelConfig):
                if hasattr(args, f.name):
                    kwargs[f.name] = getattr(args, f.name)
            kwargs['grad_reduce_in_fp32'] = args.accumulate_allreduce_grads_in_fp32
            kwargs['check_for_nan_in_grad'] = args.check_for_nan_in_loss_and_grad
            kwargs['bucket_size'] = args.ddp_bucket_size
            kwargs['average_in_collective'] = args.ddp_average_in_collective
            ddp_config = DistributedDataParallelConfig(**kwargs)
            return ddp_config

        def _build_optimzer_config(args):
            # Use specific optimizer config class based on optimizer type, matching Megatron-LM-FL behavior
            from megatron.training.training import get_megatron_optimizer_config
            config, config_overrides = get_megatron_optimizer_config(args)
            return config, config_overrides

        def _build_dataset_config(args):
            from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
            from megatron.training import get_tokenizer
            from megatron.core.datasets.utils import get_blend_from_list
            from megatron.training.datasets.sft_dataset_fs import SFTDatasetConfig

            if args.apply_sft_dataset_separated_loss_mask_if_existed:
                tokenizer = get_tokenizer()

                return SFTDatasetConfig(
                    random_seed=args.seed,
                    sequence_length=args.seq_length,
                    blend=get_blend_from_list(args.data_path),
                    blend_per_split=[
                        get_blend_from_list(args.train_data_path),
                        get_blend_from_list(args.valid_data_path),
                        get_blend_from_list(args.test_data_path)
                    ],
                    split=args.split,
                    num_dataset_builder_threads=args.num_dataset_builder_threads,
                    path_to_cache=args.data_cache_path,
                    mmap_bin_files=args.mmap_bin_files,
                    tokenizer=tokenizer,
                    reset_position_ids=args.reset_position_ids,
                    reset_attention_mask=args.reset_attention_mask,
                    eod_mask_loss=args.eod_mask_loss,
                    create_attention_mask=args.create_attention_mask_in_dataloader,
                    apply_sft_dataset_separated_loss_mask_if_existed=args.apply_sft_dataset_separated_loss_mask_if_existed,
                )
            else:
                tokenizer = get_tokenizer()

                return GPTDatasetConfig(
                    random_seed=args.seed,
                    sequence_length=args.seq_length,
                    blend=get_blend_from_list(args.data_path),
                    blend_per_split=[
                        get_blend_from_list(args.train_data_path),
                        get_blend_from_list(args.valid_data_path),
                        get_blend_from_list(args.test_data_path)
                    ],
                    split=args.split,
                    num_dataset_builder_threads=args.num_dataset_builder_threads,
                    path_to_cache=args.data_cache_path,
                    mmap_bin_files=args.mmap_bin_files,
                    tokenizer=tokenizer,
                    reset_position_ids=args.reset_position_ids,
                    reset_attention_mask=args.reset_attention_mask,
                    eod_mask_loss=args.eod_mask_loss,
                    create_attention_mask=args.create_attention_mask_in_dataloader,
                    object_storage_cache_path=args.object_storage_cache_path,
                    mid_level_dataset_surplus=args.mid_level_dataset_surplus,
                )

        from megatron.training.arguments import core_transformer_config_from_args
        self._transformer_config = core_transformer_config_from_args(self._args)
        self._ddp_config = _build_ddp_config(self._args)
        self._optimizer_config, self._optimizer_config_overrides = _build_optimzer_config(self._args)
        self._dataset_config = _build_dataset_config(self._args)

    def get_transformer_config(self):
        return self._transformer_config

    def get_ddp_config(self):
        return self._ddp_config

    def get_optimizer_config(self):
        return (self._optimizer_config, self._optimizer_config_overrides)

    def get_dataset_config(self):
        return self._dataset_config
