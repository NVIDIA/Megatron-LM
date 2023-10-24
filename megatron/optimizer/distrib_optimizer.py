# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

"""Megatron distributed optimizer."""


from apex.optimizers import FusedAdam as Adam
import math
import torch

from megatron import get_args
from megatron import get_timers
from megatron import print_rank_0
from megatron.core import mpu, tensor_parallel
from megatron.model.module import param_is_not_shared

from .optimizer import MixedPrecisionOptimizer, _zero_grad_group_helper
from .utils import shard_buffer



class Range:
    """
    A range represents a start and end points for indexing a shard
    from a full tensor.
    """
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.size = end - start
    def normalize(self, start = 0):
        return Range(start, start + self.size)
    def __str__(self):
        return "%d,%d [%d]" % (self.start, self.end, self.size)
    def __len__(self):
        return self.end - self.start


class DistributedOptimizer(MixedPrecisionOptimizer):
    """Distributed optimizer, for all data types (fp16, bf16, and fp32).

    Arguments:
        optimizer: base optimizer such as Adam or SGD
        clip_grad: clip gradeints with this global L2 norm. Note
            that clipping is ignored if clip_grad == 0
        log_num_zeros_in_grad: return number of zeros in the gradients.
        check_for_nan_in_grad: check if gradients have a NaN.
        params_have_main_grad: flag indicating if parameters have
            a `main_grad` field. If this is set, we are assuming
            that the model parameters are store in the `main_grad`
            field instead of the typical `grad` field. This happens
            for the DDP cases where there is a continuous buffer
            holding the gradients. For example for bfloat16, we want
            to do gradient accumulation and all-reduces in float32
            and as a result we store those gradients in the main_grad.
            Note that main grad is not necessarily in float32.
        fp16: if true, the model is running in fp16.
        bf16: if true, the model is running in bfloat16.
        grad_scaler: used for scaling gradients. Note that this can be
            None. This case happens when `bf16 = True` and we don't
            use any loss scale. Note that for `bf16 = True`, we can have
            a constnat gradient scaler. Also for `bf16 = False`, we
            always require a grad scaler.
        models: list of models (i.e., the virtual pipelining models). This
            is used by the distributed optimizer for mapping parameters.
    """

    @classmethod
    def build_model_gbuf_param_range_map(cls, model, dtype, gbuf_world_range, bucket_offset):
        """
        Build mapping from param reference to grad buffer shard ranges.

        This method builds a mapping from parameter references to grad
        buffer shard ranges, specific to each data-parallel (DP) rank's
        set of 'owned' parameters. Each grad buffer (padded to be an even
        multiple of DP-world-size) is conceptually divided into DP-world-size
        contiguous regions, where each DP rank 'owns' a contiguous regions.
        Ownership in this sense means DP rank is responsible for reducing
        the relevant subset of grads, and updating the relevant subset of
        params.

        This conceptual partitioning of the grad buffer does NOT respect
        parameter boundaries, and as such it is assumed that each created
        range references a shard (or subset) of the full parameter. It is
        easiest to think of each DP rank as operating (i.e., reducing,
        gathering) purely on views into the grad buffer, for all model-to-
        main & main-to-model operations.

        This method creates four ranges:
        - The param's range within the entire grad buffer (i.e., world index).
        - The param's range within the relevant grad bucket's buffer.
        - The param's range within the DP rank's local view of the grad buffer.
        - The param's range within itself (i.e., its shard).
        """

        # Param range map.
        param_world_index_map = model.grad_buffer_param_index_map[dtype]
        param_range_map = {}
        for param, param_world_indexes in param_world_index_map.items():

            # Param range.
            param_world_start, param_world_end, _ = param_world_indexes
            param_local_start = max(
                0,
                param_world_start - gbuf_world_range.start)
            param_local_end = min(
                gbuf_world_range.size,
                param_world_end - gbuf_world_range.start)

            # Add param, if within local gbuf range.
            if param_local_end > param_local_start:
                param_local_range = Range(param_local_start, param_local_end)
                param_world_range = param_local_range.normalize(
                    param_local_start + gbuf_world_range.start)
                param_world_range_in_bucket = Range(param_world_range.start-bucket_offset,
                                                    param_world_range.end-bucket_offset)
                sub_param_start = max(0, gbuf_world_range.start-param_world_start)
                sub_param_range = param_local_range.normalize(sub_param_start)
                param_range_map[param] = {
                    "gbuf_world" : param_world_range,
                    "gbuf_world_in_bucket": param_world_range_in_bucket,
                    "gbuf_local" : param_local_range,
                    "param" : sub_param_range,
                }

        return param_range_map


    @classmethod
    def build_model_gbuf_range(cls, model, dtype, bucket_index):
        """
        Build mapping between params and their grad buffers.

        This method does the initial setup for the method above. This setup
        includes determining the shard ranges into the DDP's grad buffer for
        each data-parallel (DP) rank. Each DP rank keeps range info for
        all other DP ranks, for the purpose of creating args for
        reduce-scatter and all-gather.
        """

        data_parallel_rank = mpu.get_data_parallel_rank(with_context_parallel=True)
        data_parallel_world_size = mpu.get_data_parallel_world_size(with_context_parallel=True)

        bucket = model.grad_buffers[dtype].buckets[bucket_index]
        bucket_buffer = bucket.data
        gbuf_size = bucket_buffer.numel()
        assert gbuf_size % data_parallel_world_size == 0, \
            f"Each bucket's buffer size should be divisible by {data_parallel_world_size}"
        max_gbuf_range_size = gbuf_size // data_parallel_world_size

        # All world ranges (i.e., across all data parallel ranks).
        gbuf_world_all_ranges = []
        for r in range(data_parallel_world_size):
            # Compute start of chunk in this bucket.
            gbuf_world_start = r * max_gbuf_range_size
            gbuf_world_end = min(gbuf_size, gbuf_world_start+max_gbuf_range_size)
            # Add bucket's offset in grad buffer.
            gbuf_world_range = Range(gbuf_world_start + bucket.offset,
                                     gbuf_world_end + bucket.offset)
            gbuf_world_all_ranges.append(gbuf_world_range)

        # Local DP's ranges.
        gbuf_world_range = gbuf_world_all_ranges[data_parallel_rank]

        # Get each param's ranges.
        param_range_map = cls.build_model_gbuf_param_range_map(model,
                                                               dtype,
                                                               gbuf_world_range,
                                                               bucket.offset)

        # Group into dict.
        data = {
            "param_map" : param_range_map,
        }

        return data


    @classmethod
    def build_model_gbuf_range_map(cls, model):
        """
        Create param-to-grad-buffer mappings, for grad buffer data types
        within a specific virtual model.
        """
        # Iterate through all buckets to construct param ranges that this rank "owns"
        # (the dp_rank'th shard of each bucket, where each shard is 1/dp_world_size
        # of the bucket).
        return {
            dtype : [cls.build_model_gbuf_range(model, dtype, bucket_index)
                     for bucket_index in range(len(model.grad_buffers[dtype].buckets))]
            for dtype in model.grad_buffers
        }


    @classmethod
    def build_model_param_gbuf_map(cls, model_gbuf_ranges):
        """
        Create a reverse of the model_gbuf_ranges, for referencing in
        opposite direction.
        """
        param_gbuf_map = {}
        for model_index, model_gbuf_range_map in enumerate(model_gbuf_ranges):
            for dtype, gbuf_range_map_for_all_buckets in model_gbuf_range_map.items():
                for bucket_index, gbuf_range_map in enumerate(gbuf_range_map_for_all_buckets):
                    for param, _ in gbuf_range_map["param_map"].items():
                        assert param not in param_gbuf_map, \
                            "Param should not be in param_gbuf_map; each param only belongs to a single bucket"
                        param_gbuf_map[param] = (model_index, dtype, bucket_index)
        return param_gbuf_map


    @classmethod
    def build_optimizer_group_ranges(cls, param_groups, model_gbuf_ranges):
        """
        Create optimizer groups.

        Given the set of parameter shard ranges that are owned by the current
        data-parallel (DP) rank, gather the set of parameters that will be
        used (in the method below) to create the current DP's optimizer
        groups.
        """

        num_groups = len(param_groups)

        # Param group map.
        # World param group map.
        # - Store a mapping of <model_parameter:group_index> for all parameters
        #   across all DP ranks. This is necessary because it is our first
        #   cross reference between the DDP mappings and the optimizer group
        #   parameters. This mapping only for use in the next step of building
        #   the local mapping over this DP rank's parameters.
        world_param_group_map = {}
        for group_index, group in enumerate(param_groups):
            for param in group["params"]:
                assert param.requires_grad
                world_param_group_map[param] = group_index

        # Optimizer group ranges & param-group mapping.
        # - Build a mapping from groups to their contained parameters, and also
        #   from parameters to their containing group index and order within
        #   the group. The group index and order are particularly important for
        #   saving and loading checkpoints.
        local_param_group_map = {}
        group_ranges = [ {"params": []} for _ in param_groups ]
        for model_gbuf_range_map in model_gbuf_ranges:
            for dtype, gbuf_range_map_for_all_buckets in model_gbuf_range_map.items():
                for gbuf_range_map in gbuf_range_map_for_all_buckets:
                    for param in gbuf_range_map["param_map"]:
                        group_index = world_param_group_map[param]
                        group_range = group_ranges[group_index]
                        group_range["params"].append(param)
                        local_param_group_map[param] = \
                            (group_index, len(group_range["params"]) - 1)

        # Squeeze zero-size group ranges.
        for group_index, group_range in enumerate(group_ranges):
            group_range["orig_group"] = param_groups[group_index]
            group_range["orig_group_idx"] = param_groups[group_index]

        return local_param_group_map, group_ranges


    @classmethod
    def build_model_and_main_param_groups(cls,
                                          model_gbuf_ranges,
                                          param_gbuf_map,
                                          opt_group_ranges):
        """
        Create main parameter groups needed for the optimizer step.

        These groups encompass both: 1) groups used by this class, for
        reducing/gather, and 2) groups used by the inner optimizer for the
        parameter update. Given that the conceptual grad buffer partitioning
        (created in earlier method) doesn't respect parameter boundaries,
        the optimizer operates on shards of the model parameters, rather than
        the full parameters.
        """

        # Parameter groups:
        #   model_float16_groups: original float16 parameters
        #   model_fp32_groups: original fp32 parameters
        #   shard_float16_groups: shards of original float16 parameters
        #   shard_fp32_groups: shards of original fp32 parameters
        #   shard_fp32_from_float16_groups: fp32 copy of float16 parameters
        model_float16_groups = []
        model_fp32_groups = []
        shard_float16_groups = []
        shard_fp32_groups = []
        shard_fp32_from_float16_groups = []

        # Allocate (or slice) each group's param shard.
        for group_index, group_range in enumerate(opt_group_ranges):

            # Params of this group.
            model_float16_params_this_group = []
            model_fp32_params_this_group = []
            shard_float16_params_this_group = []
            shard_fp32_params_this_group = []
            shard_fp32_from_float16_params_this_group = []
            model_float16_groups.append(model_float16_params_this_group)
            model_fp32_groups.append(model_fp32_params_this_group)
            shard_float16_groups.append(shard_float16_params_this_group)
            shard_fp32_groups.append(shard_fp32_params_this_group)
            shard_fp32_from_float16_groups.append(
                shard_fp32_from_float16_params_this_group)

            for model_param in group_range["params"]:

                assert model_param.requires_grad

                model_index, dtype, bucket_index = param_gbuf_map[model_param]
                gbuf_range = model_gbuf_ranges[model_index][dtype][bucket_index]
                param_range = gbuf_range["param_map"][model_param]["param"]

                # fp16, bf16 params.
                if model_param.type() in ['torch.cuda.HalfTensor',
                                          'torch.cuda.BFloat16Tensor']:

                    # Clone model -> main.
                    shard_model_param = model_param.detach().view(-1) \
                        [param_range.start:param_range.end]
                    shard_main_param = shard_model_param.clone().float()
                    tensor_parallel.copy_tensor_model_parallel_attributes(
                        shard_model_param, model_param)
                    tensor_parallel.copy_tensor_model_parallel_attributes(
                        shard_main_param, model_param)
                    if hasattr(model_param, 'shared'):
                        shard_model_param.shared = model_param.shared
                        shard_main_param.shared = model_param.shared

                    # Add to group.
                    model_float16_params_this_group.append(model_param)
                    shard_float16_params_this_group.append(shard_model_param)
                    shard_fp32_from_float16_params_this_group.append(shard_main_param)

                # fp32 params.
                elif model_param.type() == 'torch.cuda.FloatTensor':
                    shard_model_param = model_param.view(-1) \
                        [param_range.start:param_range.end]
                    model_fp32_params_this_group.append(model_param)
                    shard_fp32_params_this_group.append(shard_model_param)
                    tensor_parallel.copy_tensor_model_parallel_attributes(
                        shard_model_param, model_param)
                    if hasattr(model_param, 'shared'):
                        shard_model_param.shared = model_param.shared

                else:
                    raise TypeError('Wrapped parameters must be one of '
                                    'torch.cuda.FloatTensor,  '
                                    'torch.cuda.HalfTensor, or '
                                    'torch.cuda.BFloat16Tensor. '
                                    'Received {}'.format(model_param.type()))

            # Update optimizer's params.
            group_range["orig_group"]["params"] = [
                *shard_fp32_params_this_group,
                *shard_fp32_from_float16_params_this_group,
            ]

        return (
            model_float16_groups,
            model_fp32_groups,
            shard_float16_groups,
            shard_fp32_groups,
            shard_fp32_from_float16_groups,
        )


    def __init__(self, optimizer, clip_grad, log_num_zeros_in_grad,
                 check_for_nan_in_grad, params_have_main_grad, fp16,
                 bf16, params_dtype, grad_scaler, models):
        """
        See top of class definition for argument descriptions.

        The steps in this method create the core mapping between DDP grad
        buffers, parameters, and parameter shard ranges, that is needed for
        converting between model param indexes and main parameter shard
        indexes. This method also updates the optimizer parameter groups
        with the newly created shards.
        """

        super().__init__(
            optimizer, clip_grad, log_num_zeros_in_grad,
            check_for_nan_in_grad, params_have_main_grad,
            fp16, bf16, params_dtype, grad_scaler, models)

        assert isinstance(optimizer, Adam), \
            "Only Adam currently supported, due to checkpointing requirements."

        # Model grad buffer ranges.
        self.model_gbuf_ranges = []
        self.bucket_sizes = []
        for model_index, model in enumerate(self.models):
            self.bucket_sizes.append(model.bucket_size)
            self.model_gbuf_ranges.append(self.build_model_gbuf_range_map(model))
        self.model_param_gbuf_map = \
            self.build_model_param_gbuf_map(self.model_gbuf_ranges)

        # Optimizer ranges.
        self.model_param_group_index_map, self.opt_group_ranges = \
            self.build_optimizer_group_ranges(self.optimizer.param_groups,
                                              self.model_gbuf_ranges)

        # Allocate main param shards.
        (
            self.model_float16_groups,
            self.model_fp32_groups,
            self.shard_float16_groups,
            self.shard_fp32_groups,
            self.shard_fp32_from_float16_groups,
        ) = self.build_model_and_main_param_groups(self.model_gbuf_ranges,
                                                   self.model_param_gbuf_map,
                                                   self.opt_group_ranges)

        # Initialize param buffers.
        # - These are views on the DDP model's grad buffers, that share
        #   storage & have their own dtype. This is safe because the param
        #   dtype size is always <= grad dtype size.
        self.param_buffers = []
        for model_index, model in enumerate(self.models):
            current_param_buffers = {}
            for dtype, grad_buffer in model.grad_buffers.items():
                current_param_buffers[dtype] = []
                for bucket in grad_buffer.buckets:

                    # Handle older/newer method for getting untyped storage.
                    try:
                        storage = bucket.data.storage()._untyped()
                    except:
                        storage = bucket.data.storage().untyped()

                    # Typed param buffer.
                    param_buffer = torch.tensor(
                        storage,
                        dtype = params_dtype,
                        device = bucket.data.device)
                    # .storage() ignores views / slices, so param_buffer now points to the start
                    # of the grad_buffer instead of to the start of each bucket. As a result,
                    # add bucket.offset to make sure param_buffers don't point to the same region
                    # of memory.
                    param_buffer = param_buffer[bucket.offset:bucket.offset+bucket.data.numel()]
                    current_param_buffers[dtype].append(param_buffer)
            self.param_buffers.append(current_param_buffers)

        # Update optimizer groups.
        # - Also, leverage state_dict() and load_state_dict() to
        #   recast preexisting per-param state tensors.
        self.optimizer.param_groups = \
            [ g["orig_group"] for g in self.opt_group_ranges ]
        self.optimizer.load_state_dict(self.optimizer.state_dict())


    def get_model_param_range_map(self, param):
        """
        Given a model param, get the index sub-range of the param that this
        data-parallel rank owns.
        """
        model_index, dtype, bucket_index = self.model_param_gbuf_map[param]
        gbuf_range_map = self.model_gbuf_ranges[model_index][dtype][bucket_index]
        param_range_map = gbuf_range_map["param_map"][param]
        return param_range_map


    def get_model_parallel_group(self):
        """
        With the distributed optimizer, the model parallel group is the
        entire world.
        """
        return None


    def state_dict(self):
        """
        The state dict contains all non-DP-rank-dependent (i.e., non-parameter-
        related) optimizer variables. The returned state dict can be stored in
        the standard model/RNG checkpoint file. The parameter and dependent
        optimizer state (e.g., exp_avg, exp_avg_sq) are stored in a separate
        checkpoint file by calling 'save_parameter_state()'.
        """

        state_dict = {}

        # Optimizer state (do not store parameter state here).
        state_dict['optimizer'] = {
            k : v
            for k, v in self.optimizer.state_dict().items()
            if k != "state"
        }
        for param_group in state_dict["optimizer"]["param_groups"]:
            del param_group["params"]

        # Grad scaler state.
        if self.grad_scaler:
            state_dict['grad_scaler'] = self.grad_scaler.state_dict()

        return state_dict


    def load_state_dict(self, state_dict):
        """Load the state dict.

        As detailed in state_dict(), the state dict contains all non-
        parameter-related variables. This method is notably longer than
        state_dict(), because the Torch optimizers state has yet to be
        allocated at this point, and so we must do a cross referencing between
        the optimizers state (and the ordering it expects for parameter state)
        and this DP rank's shards. The optimizer at this point does not contain
        any tensor dimension information, so we must get these dimensions from
        the DP shards mapped during DistributedOptimizer.__init__().

        The tensor parameter state is loaded via load_parameter_state(), and
        so this method also must populate the loaded state dict with dummy
        tensor data (i.e., via torch.empty() below). This will be overwritten
        during load_parameter_state().

        ** Note: Torch optimizer's state structure. **
        The Torch optimizer stores its state in two levels. The top level is a
        list of groups, where each group contains a list of integer indexes
        (corresponding to parameters) that index into a master parameter list
        that is shared by all groups. As such, three values are necessary for
        maintaining this ordering:

        - group_index : The group to which a parameter belongs.
        - group_order : The index of a parameter within its group.
        - state_order : The index of a parameter within the shared parameter
            list.
        """

        # Get the Torch optimizer's state dict.
        # - This 'inner' optimizer at this point is unallocated, and only
        #   contains an integer odering of parameters within each group, and
        #   the ordering of parameters within its flattened parameter state
        #   list.
        inner_state_dict = self.optimizer.state_dict()
        state_dict_param_groups = [{
            **group,
            "params" : list(inner_state_dict["param_groups"][idx]["params"]),
        } for idx, group in enumerate(state_dict["optimizer"]["param_groups"])]

        # Allocate 'dummy' data for optimizer state (i.e., torch.empty() below)
        # - Real data is overwritten during load_parameter_state().
        state_dict_state = []
        for gbuf_range_maps in self.model_gbuf_ranges:
            for gbuf_range_map_for_all_buckets in gbuf_range_maps.values():
                for gbuf_range_map in gbuf_range_map_for_all_buckets:
                    for model_param, param_range_map in \
                        gbuf_range_map["param_map"].items():

                        # Get parameter ordering information (see method docstring
                        # for details).
                        group_index, group_order = \
                            self.model_param_group_index_map[model_param]
                        state_order = inner_state_dict["param_groups"] \
                            [group_index]["params"][group_order]

                        # Allocate dummy tensors.
                        numel = len(param_range_map["gbuf_world"])
                        init_shard = lambda : torch.empty(
                            (numel,),
                            dtype=torch.float32,
                            device=torch.cuda.current_device())

                        state_dict_state.append((state_order, {
                            "exp_avg" : init_shard(),
                            "exp_avg_sq" : init_shard(),
                        }))

        # Sort by state order (see method docstring for details).
        state_dict_state.sort(key = lambda s : s[0])
        state_dict_state = {s[0]:s[1] for s in state_dict_state}

        # Optimizer.
        self.optimizer.load_state_dict({
            "state" : state_dict_state,
            "param_groups" : state_dict_param_groups,
        })

        # Grad scaler.
        if 'grad_scaler' not in state_dict:
            if self.fp16:
                print_rank_0('***WARNING*** found an old checkpoint, will not '
                             'load grad scaler ...')
        else:
            if self.grad_scaler:
                self.grad_scaler.load_state_dict(state_dict['grad_scaler'])
            else:
                print_rank_0('***WARNING*** fould the grad scaler in the '
                             'checkpoint but it is None in the class. '
                             'Skipping loading grad scaler ...')


    def save_parameter_state(self, filename):
        """Save parameter state (i.e., parameter & optimizer tensors).

        This method performs three steps:
        - For each DP rank, copy param & optimizer shards to contiguous CPU
          buffers. (e.g., one buffer each for main_param, exp_avg, and
          exp_avg_sq).
        - Gather contiguous buffers on DP rank 0 and concatenate to world
          buffers.
        - Save world buffers to disk (i.e., distrib_opt.pt).
        """

        # Data parallelism variables.
        data_parallel_world_size = mpu.get_data_parallel_world_size(with_context_parallel=True)
        data_parallel_rank = mpu.get_data_parallel_rank(with_context_parallel=True)
        data_parallel_group_gloo = mpu.get_data_parallel_group_gloo(with_context_parallel=True)
        data_parallel_global_ranks = list(mpu._DATA_PARALLEL_GLOBAL_RANKS_WITH_CP)

        # Collect param states.
        state = {"bucket_sizes": self.bucket_sizes}
        for model_idx, gbuf_range_maps in enumerate(self.model_gbuf_ranges):

            # Iterate grad buffers (by data type).
            dtype_state = {}
            assert len(gbuf_range_maps) == 1, "single dtype supported, for now."
            for dtype, gbuf_range_map_for_all_buckets in gbuf_range_maps.items():
                world_tensors = {}
                for bucket_idx, gbuf_range_map in enumerate(gbuf_range_map_for_all_buckets):

                    # Compute local DP contiguous shard's size.
                    model = self.models[model_idx]
                    gbuf_world_numel = model.grad_buffers[dtype].buckets[bucket_idx].data.numel()
                    assert gbuf_world_numel % data_parallel_world_size == 0
                    gbuf_local_numel = gbuf_world_numel // data_parallel_world_size
                    local_shards = {key: torch.empty((gbuf_local_numel,),
                                                     dtype=torch.float32,
                                                     device="cpu")
                                    for key in ("param", "exp_avg", "exp_avg_sq")}

                    # Build contiguous DP rank shards (for param + optim states).
                    for model_param, param_range_map in \
                        gbuf_range_map["param_map"].items():

                        # Main param & optimizer states.
                        group_index, group_order = \
                            self.model_param_group_index_map[model_param]
                        main_param = self.optimizer.param_groups \
                            [group_index]["params"][group_order]
                        optim_state = self.optimizer.state[main_param]

                        tensors = {
                            "param" : main_param,
                            **optim_state,
                        }

                        # Copy states into contiguous shard.
                        gbuf_local_start = param_range_map["gbuf_local"].start
                        gbuf_local_end = param_range_map["gbuf_local"].end
                        for key in local_shards:
                            local_shards[key][gbuf_local_start:gbuf_local_end] \
                                .data.copy_(tensors[key].detach().cpu())

                    # Gather contiguous shards on DP rank 0.
                    for key, send_tensor in local_shards.items():

                        # Gather tensor list.
                        if data_parallel_rank == 0:
                            recv_tensors = [torch.empty((gbuf_local_numel,),
                                                        dtype=torch.float32,
                                                        device="cpu")
                                            for _ in range(data_parallel_world_size)]
                        else:
                            recv_tensors = None

                        # Gather.
                        torch.distributed.gather(
                            send_tensor,
                            recv_tensors,
                            data_parallel_global_ranks[0],
                            data_parallel_group_gloo,
                        )

                        # Concatenate.
                        if data_parallel_rank == 0:
                            if key not in world_tensors:
                                world_tensors[key] = []
                            world_tensors[key].append(torch.cat(recv_tensors))

                # Collect world state.
                dtype_state[dtype] = world_tensors
            state[model_idx] = dtype_state

        # Save param state.
        if data_parallel_rank == 0:
            torch.save(state, filename)


    def load_parameter_state(self, filename):
        """Load parameter state (i.e., parameter & optimizer tensors).

        This method performs the reverse of save_parameter_state():
        - Load world buffers from disk (i.e., distrib_opt.pt).
        - Scatter contiguous buffers from DP rank 0 to each DP rank (each DP
          rank receives its relevant subset of the world buffers).
        - For each DP rank, copy param & optimizer shards from contiguous CPU
          buffers. (e.g., one buffer each for main_param, exp_avg, and
          exp_avg_sq).
        """

        # Data parallelism variables.
        data_parallel_world_size = mpu.get_data_parallel_world_size(with_context_parallel=True)
        data_parallel_rank = mpu.get_data_parallel_rank(with_context_parallel=True)
        data_parallel_group_gloo = mpu.get_data_parallel_group_gloo(with_context_parallel=True)
        data_parallel_global_ranks = list(mpu._DATA_PARALLEL_GLOBAL_RANKS_WITH_CP)

        # Load on DP rank 0.
        if data_parallel_rank == 0:
            loaded_state = torch.load(filename)
            if "bucket_sizes" in loaded_state:
                bucket_sizes_in_checkpoint = loaded_state["bucket_sizes"]
                assert self.bucket_sizes == bucket_sizes_in_checkpoint, \
                    f"Bucket sizes need to be the same in current run ({self.bucket_sizes}) and checkpoint ({bucket_sizes_in_checkpoint})"

        # Scatter tensors to all DP ranks.
        for model_idx, gbuf_range_maps in enumerate(self.model_gbuf_ranges):
            for dtype, gbuf_range_map_for_all_buckets in gbuf_range_maps.items():
                for bucket_idx, gbuf_range_map in enumerate(gbuf_range_map_for_all_buckets):

                    # Compute local DP contiguous shard's size.
                    model = self.models[model_idx]
                    gbuf_world_numel = model.grad_buffers[dtype].buckets[bucket_idx].data.numel()
                    assert gbuf_world_numel % data_parallel_world_size == 0
                    gbuf_local_numel = gbuf_world_numel // data_parallel_world_size

                    # Contiguous local shards (received from DP rank 0).
                    local_shards = {key: torch.empty((gbuf_local_numel,),
                                                     dtype=torch.float32,
                                                     device="cpu")
                                    for key in ("param", "exp_avg", "exp_avg_sq")}

                    # Scatter local shards from DP rank 0.
                    for key, recv_tensor in local_shards.items():

                        # Scatter tensor list.
                        if data_parallel_rank == 0:
                            world_tensor = loaded_state[model_idx][dtype][key][bucket_idx]
                            gbuf_start_idxs = \
                                list(range(0, gbuf_world_numel, gbuf_local_numel))
                            send_tensors = [world_tensor[i:(i+gbuf_local_numel)]
                                            for i in gbuf_start_idxs]
                        else:
                            send_tensors = None

                        # Scatter.
                        torch.distributed.scatter(
                            recv_tensor,
                            send_tensors,
                            data_parallel_global_ranks[0],
                            data_parallel_group_gloo,
                        )

                    # Copy local contiguous shards to param/optim shards.
                    for model_param, param_range_map in \
                        gbuf_range_map["param_map"].items():

                        # Main param & optimizer states.
                        group_index, group_order = \
                            self.model_param_group_index_map[model_param]
                        main_param = self.optimizer.param_groups \
                            [group_index]["params"][group_order]
                        optim_state = self.optimizer.state[main_param]

                        tensors = {
                            "param" : main_param,
                            **optim_state,
                        }

                        # Copy states into contiguous shard.
                        gbuf_local_start = param_range_map["gbuf_local"].start
                        gbuf_local_end = param_range_map["gbuf_local"].end
                        for key in local_shards:
                            tensors[key].data.copy_(
                                local_shards[key][gbuf_local_start:gbuf_local_end])


    def zero_grad(self, set_to_none=True):
        """
        Zero grads.

        We only need to zero the model related parameters, i.e.,
        model_float16_groups & model_fp32_groups. We additionally zero
        the remaining groups as a memory optimization to reduce
        fragmentation; in the case of set_to_none==True, the space
        used by this field can be safely deallocated at this point.
        """
        for groups in (
                self.model_float16_groups,
                self.model_fp32_groups,
                self.shard_float16_groups, # grad empty/unused here?
                self.shard_fp32_groups, # throws grad-access warning
                self.shard_fp32_from_float16_groups):
            for group in groups:
                _zero_grad_group_helper(group, set_to_none)


    @staticmethod
    def get_model_buffer_dp_views(model_buffers):
        """
        Get shard views of each of the DDP's param/grad buffers.

        In this nested list, the top level is grouped by the virtual model
        index and the buffer's data type. The sub-level is a list of
        shards of that buffer, where each shard in the list represents
        a contiguous view of the buffer, that is owned by a data-parallel
        rank. The shard boundary does not respect parameter boundaries, and
        so the elements of some parameters are split across data parallel
        ranks.

        Additionally, return references to the entire buffers, for use
        in _reduce_scatter_base and _all_gather_base.
        """

        # Buffer views.
        view_items = []
        for model_index, buffers in enumerate(model_buffers):
            for dtype, buf_for_all_buckets in buffers.items():
                for bucket_index, buf in enumerate(buf_for_all_buckets):
                    buf_views = shard_buffer(buf)
                    view_items.append((model_index, dtype, bucket_index, buf, buf_views))

        return view_items


    def get_model_param_buffer_dp_views(self):
        return self.get_model_buffer_dp_views(self.param_buffers)


    def gather_model_params(self, args, timers):
        """
        All-gather updated model params.

        The DDP's param buffer is used for the all-gather, and thus no
        tensors are dynamically allocated. After the all-gather, the params
        can be copied from the param buffer to the param.
        """

        timers('params-all-gather', log_level=1).start(
            barrier=args.barrier_with_L1_time)

        data_parallel_rank = mpu.get_data_parallel_rank(with_context_parallel=True)
        data_parallel_group = mpu.get_data_parallel_group(with_context_parallel=True)

        # All-gather updated main params.
        # - All param buffer views are guaranteed to have the same num elements
        #   across all data parallel ranks, due to grad buffer padding that is
        #   done in distributed.py, and extended to the param buffers. Thus,
        #   all sub-views will have consistent start/end indexes across data
        #   parallel ranks.
        pbuf_view_items = self.get_model_param_buffer_dp_views()
        for (_, _, _, pbuf, pbuf_views) in pbuf_view_items:
            torch.distributed._all_gather_base(
                pbuf,
                pbuf_views[data_parallel_rank],
                group = data_parallel_group,
            )

        # Copy from param buffer to each param.
        for model_id, model in enumerate(self.models):
            for dtype, param_map in model.grad_buffer_param_index_map.items():
                for param, (buf_start, buf_end, bucket_index) in param_map.items():
                    bucket_offset = model.grad_buffers[dtype].buckets[bucket_index].offset
                    param_buf = self.param_buffers[model_id][dtype][bucket_index]
                    # buf_start and buf_end store position of this parameter in the full grad_buffer,
                    # so need to adjust these indices (by subtracting out bucket_offset) since we
                    # have independent param_bufs for each bucket.
                    param_buf_shard = param_buf[buf_start-bucket_offset:buf_end-bucket_offset]
                    assert param.data.nelement() == param_buf_shard.nelement()
                    param.view(-1).detach().copy_(param_buf_shard)

        timers('params-all-gather').stop()


    def _collect_main_grad_data_for_unscaling(self):
        """
        Note: this should be equivalent to the float-16 optimizer's method,
        but writtent differently, so the two should be combined.
        """
        return [
            param.grad.data
            for group in self.optimizer.param_groups
            for param in group["params"]
        ]


    def _get_model_and_main_params_data_float16(self):
        """
        Get aligned list of model and main params.
        """
        model_data = []
        main_data = []
        for model_group, main_group in zip(self.shard_float16_groups,
                                           self.shard_fp32_from_float16_groups):
            for model_param, main_param in zip(model_group, main_group):
                model_data.append(model_param.data)
                main_data.append(main_param.data)
        return model_data, main_data


    def _copy_model_grads_to_main_grads(self):
        """
        Copy model grads to main grads.

        Since this step follows a reduce-scatter through the DDP's grad
        buffer, this method is responsible for copying the updated grads
        from the grad buffer to the main shard's grad field.
        """

        # Utility method for copying group grads.
        def copy_group_grads(model_groups, shard_main_groups):
            for model_group, shard_main_group in zip(model_groups,
                                                     shard_main_groups):
                for model_param, shard_main_param in zip(model_group,
                                                         shard_main_group):

                    param_range_map = self.get_model_param_range_map(model_param)
                    param_range = param_range_map["param"]
                    assert param_range.size == shard_main_param.nelement()

                    model_grad = model_param.main_grad
                    shard_model_grad = model_grad.view(-1) \
                        [param_range.start:param_range.end]
                    shard_main_param.grad = shard_model_grad.float()

        # Copy model groups to shard groups.
        copy_group_grads(self.model_float16_groups,
                         self.shard_fp32_from_float16_groups)
        copy_group_grads(self.model_fp32_groups,
                         self.shard_fp32_groups)


    def _copy_main_params_to_model_params(self):
        """
        Copy main params to model params.

        Since this step is followed by an all-gather through the DDP's grad
        buffer, this method is responsible for copying the updated params
        from the main shards into the correct position in the grad buffer.
        """

        # Utility method for copying group params.
        def copy_group_params(shard_main_groups, model_groups):
            for shard_main_group, model_group in zip(shard_main_groups,
                                                     model_groups):
                for shard_main_param, model_param in zip(shard_main_group,
                                                         model_group):

                    param_range_map = self.get_model_param_range_map(model_param)
                    world_range = param_range_map["gbuf_world_in_bucket"]

                    assert world_range.size == shard_main_param.nelement()

                    model_id, dtype, bucket_id = self.model_param_gbuf_map[model_param]
                    model_param_buffer = self.param_buffers[model_id][dtype][bucket_id]

                    shard_model_param = model_param_buffer.view(-1) \
                        [world_range.start:world_range.end]

                    shard_model_param.data.copy_(shard_main_param)

        # Copy shard groups to model groups.
        copy_group_params(self.shard_fp32_from_float16_groups,
                          self.model_float16_groups)
        copy_group_params(self.shard_fp32_groups,
                          self.model_fp32_groups)


    def _copy_model_params_to_main_params(self):
        """
        Copy model params to main params.

        During finetuning, this method is used to reload the main params from
        the model params. This copy does not make use of the grad buffer as
        an intermediary.
        """

        # Utility method for copying group params.
        def copy_group_params(model_groups, shard_main_groups):
            for model_group, shard_main_group in zip(model_groups,
                                                     shard_main_groups):
                for model_param, shard_main_param in zip(model_group,
                                                         shard_main_group):

                    param_range_map = self.get_model_param_range_map(model_param)
                    param_range = param_range_map["param"]
                    assert param_range.size == shard_main_param.nelement()

                    shard_model_param = model_param.view(-1) \
                        [param_range.start:param_range.end]
                    shard_main_param.data.copy_(shard_model_param)

        # Copy model groups to shard groups.
        copy_group_params(self.model_float16_groups,
                          self.shard_fp32_from_float16_groups)
        copy_group_params(self.model_fp32_groups,
                          self.shard_fp32_groups)
