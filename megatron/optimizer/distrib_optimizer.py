# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

"""Megatron distributed optimizer."""


import math
import torch

from megatron import get_args
from megatron import get_timers
from megatron import mpu
from megatron.model.module import param_is_not_shared
from megatron.mpu.layers import param_is_not_tensor_parallel_duplicate

from .optimizer import MixedPrecisionOptimizer, _zero_grad_group_helper
# from .optimizer import Float16OptimizerWithFloat16Params

# >>>
from lutil import pax, tp, print_seq
# <<<

# >>>
# class Shard:
class Range:
# <<<

    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.size = end - start
    def normalize(self, start = 0):
        return Range(start, start + self.size)
    def __str__(self):
        return "%d,%d [%d]" % (self.start, self.end, self.size)


# class DistributedOptimizer(Float16OptimizerWithFloat16Params):
class DistributedOptimizer(MixedPrecisionOptimizer):

    @classmethod
    def build_model_gbuf_param_range_map(cls, model, dtype, gbuf_world_range):

        # Param range map.
        param_world_index_map = model._grad_buffer_param_index_map[dtype]
        param_range_map = {}
        for param, param_world_indexes in param_world_index_map.items():

            # Param range.
            param_world_start, param_world_end = param_world_indexes
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
                sub_param_start = max(0, gbuf_world_range.start-param_world_start)
                sub_param_range = param_local_range.normalize(sub_param_start)
                param_range_map[param] = {
                    "gbuf_world" : param_world_range,
                    "gbuf_local" : param_local_range,
                    "param" : sub_param_range,
                }

        return param_range_map

    @classmethod
    def build_model_gbuf_range(cls, model, dtype):

        data_parallel_rank = mpu.get_data_parallel_rank()
        data_parallel_world_size = mpu.get_data_parallel_world_size()

        # Grad buffer range.
        grad_buffer = model._grad_buffers[dtype]
        gbuf_size = grad_buffer.numel
        max_gbuf_range_size = int(math.ceil(gbuf_size / data_parallel_world_size))

        # All world ranges. (i.e., across all data parallel ranks)
        gbuf_world_all_ranges = []
        for r in range(data_parallel_world_size):
            gbuf_world_start = r * max_gbuf_range_size
            gbuf_world_end = min(gbuf_size, gbuf_world_start+max_gbuf_range_size)
            gbuf_world_range = Range(gbuf_world_start, gbuf_world_end)
            gbuf_world_all_ranges.append(gbuf_world_range)

        # Local DP's ranges.
        gbuf_world_range = gbuf_world_all_ranges[data_parallel_rank]
        gbuf_local_range = gbuf_world_range.normalize()

        # Get each param's ranges.
        param_range_map = cls.build_model_gbuf_param_range_map(model,
                                                               dtype,
                                                               gbuf_world_range)

        # Altogether.
        data = {
            "local" : gbuf_local_range,
            "world" : gbuf_world_range,
            "world_all" : gbuf_world_all_ranges,
            "param_map" : param_range_map,
            "max_range_size" : max_gbuf_range_size,
        }

        return data

    @classmethod
    def build_model_gbuf_range_map(cls, model):
        return {
            dtype : cls.build_model_gbuf_range(model, dtype)
            for dtype in model._grad_buffers
        }

    @classmethod
    def build_model_param_gbuf_map(cls, model_gbuf_ranges):
        '''Create a reverse of the model_gbuf_ranges, for referencing in
        opposite direction.'''
        param_gbuf_map = {}
        for model_index, model_gbuf_range_map in enumerate(model_gbuf_ranges):
            for dtype, gbuf_range_map in model_gbuf_range_map.items():
                for param, param_range_map in gbuf_range_map["param_map"].items():
                    param_gbuf_map[param] = (model_index, dtype)
        return param_gbuf_map

    # >>>
    # @classmethod
    # def build_optimizer_group_ranges(cls, param_groups, model_gbuf_ranges):

    #     num_groups = len(param_groups)

    #     # Param group map.
    #     param_group_map = {}
    #     for group_index, group in enumerate(param_groups):
    #         for param in group["params"]:
    #             assert param.requires_grad
    #             param_group_map[param] = group_index

    #     # Optimizer group ranges.
    #     group_ranges = [ {"size": 0, "param_map": {}} for _ in param_groups ]
    #     for model_gbuf_range_map in model_gbuf_ranges:
    #         for dtype, gbuf_range_map in model_gbuf_range_map.items():
    #             for param in gbuf_range_map["param_map"]:
                    
    #                 group_index = param_group_map[param]
    #                 group_range = group_ranges[group_index]
    #                 param_size = gbuf_range_map["param_map"][param]["param"].size

    #                 param_group_start = group_range["size"]
    #                 param_group_end = param_group_start + param_size
    #                 param_group_range = Range(param_group_start, param_group_end)

    #                 group_range["size"] += param_size
    #                 group_range["param_map"][param] = param_group_range

    #     # Squeeze zero-size group ranges.
    #     for group_index, group_range in enumerate(group_ranges):
    #         group_range["orig_group"] = param_groups[group_index]
    #     group_ranges = [ g for g in group_ranges if g["size"] > 0 ]

    #     return group_ranges
    @classmethod
    def build_optimizer_group_ranges(cls, param_groups, model_gbuf_ranges):

        num_groups = len(param_groups)

        # Param group map.
        param_group_map = {}
        for group_index, group in enumerate(param_groups):
            for param in group["params"]:
                assert param.requires_grad
                param_group_map[param] = group_index

        # Optimizer group ranges.
        # >>>
        # group_ranges = [ {"size": 0, "param_map": {}} for _ in param_groups ]
        group_ranges = [ {"params": []} for _ in param_groups ]
        # group_ranges = [ [] for _ in param_groups ]
        # <<<
        for model_gbuf_range_map in model_gbuf_ranges:
            for dtype, gbuf_range_map in model_gbuf_range_map.items():
                for param in gbuf_range_map["param_map"]:
                    group_index = param_group_map[param]
                    group_range = group_ranges[group_index]
                    group_range["params"].append(param)

        # Squeeze zero-size group ranges.
        for group_index, group_range in enumerate(group_ranges):
            group_range["orig_group"] = param_groups[group_index]
        group_ranges = [ g for g in group_ranges if len(g["params"]) > 0 ]

        # >>>
        # print_seq("group ranges / len = %s." %
        #           ", ".join(str(len(s["params"])) for s in group_ranges))
        # <<<

        return group_ranges
    # <<<

    # >>>
    # @classmethod
    # def allocate_main_param_shards(cls, opt_group_ranges):

    #     # Allocator method.
    #     allocate_shard = lambda shard_size, dtype : torch.empty(
    #         (shard_size,),
    #         dtype = dtype,
    #         device = torch.cuda.current_device(),
    #         requires_grad = True)

    #     # Allocate each group's param/grad shard.
    #     for group_index, group_range in enumerate(opt_group_ranges):

    #         group_size = group_range["size"]
    #         assert group_size != 0, "temporary check ... remove me."

    #         # Allocate shard.
    #         main_param = allocate_shard(group_size, torch.float)
    #         main_param.grad = allocate_shard(group_size, torch.float)
    #         mpu.set_tensor_model_parallel_attributes(main_param, True, 0, 1)

    #         # Update group's param.
    #         group_range["orig_group"]["params"] = [ main_param ]
    @classmethod
    # def allocate_main_params(cls, opt_group_ranges):
    # def allocate_or_view_main_param_shards(cls,
    def build_model_and_main_param_groups(cls,
                                        model_gbuf_ranges,
                                        param_gbuf_map,
                                        opt_group_ranges):

        # Three groups of parameters:
        #   float16_groups: original float16 parameters
        #   fp32_from_float16_groups: fp32 copy of float16 parameters
        #   fp32_groups: original fp32 parameters
        full_float16_groups = []
        full_fp32_groups = []
        shard_float16_groups = []
        shard_fp32_groups = []
        shard_fp32_from_float16_groups = []

        # Allocate each group's param shard.
        for group_index, group_range in enumerate(opt_group_ranges):

            # Params of this group.
            full_float16_params_this_group = []
            full_fp32_params_this_group = []
            shard_float16_params_this_group = []
            shard_fp32_params_this_group = []
            shard_fp32_from_float16_params_this_group = []
            full_float16_groups.append(full_float16_params_this_group)
            full_fp32_groups.append(full_fp32_params_this_group)
            shard_float16_groups.append(shard_float16_params_this_group)
            shard_fp32_groups.append(shard_fp32_params_this_group)
            shard_fp32_from_float16_groups.append(
                shard_fp32_from_float16_params_this_group)

            for model_param in group_range["params"]:

                model_index, dtype = param_gbuf_map[model_param]
                gbuf_range = model_gbuf_ranges[model_index][dtype]
                param_range = gbuf_range["param_map"][model_param]["param"]

                # fp16, bf16 params.
                if model_param.type() in ['torch.cuda.HalfTensor',
                                          'torch.cuda.BFloat16Tensor']:

                    # Clone model -> main.
                    shard_model_param = \
                        model_param.detach()[param_range.start:param_range.end]
                    shard_main_param = shard_model_param.clone().float()
                    mpu.copy_tensor_model_parallel_attributes(
                        shard_model_param, model_param)
                    mpu.copy_tensor_model_parallel_attributes(
                        shard_main_param, model_param)
                    if hasattr(model_param, 'shared'):
                        shard_model_param.shared = model_param.shared
                        shard_main_param.shared = model_param.shared

                    # Add to group.
                    full_float16_params_this_group.append(model_param)
                    shard_float16_params_this_group.append(shard_model_param)
                    shard_fp32_from_float16_params_this_group.append(shard_main_param)

                # fp32 params.
                elif param.type() == 'torch.cuda.FloatTensor':
                    shard_model_param = \
                        model_param[param_range.start:param_range.end]
                    full_fp32_params_this_group.append(model_param)
                    shard_fp32_params_this_group.append(shard_model_param)

                else:
                    raise TypeError('Wrapped parameters must be one of '
                                    'torch.cuda.FloatTensor,  '
                                    'torch.cuda.HalfTensor, or '
                                    'torch.cuda.BFloat16Tensor. '
                                    'Received {}'.format(param.type()))

                # # Add to group.
                # group_main_params.append(main_param)

            group_range["orig_group"]["params"] = [
                *shard_fp32_params_this_group,
                *shard_fp32_from_float16_params_this_group,
            ]

        return (
            full_float16_groups,
            full_fp32_groups,
            shard_float16_groups,
            shard_fp32_groups,
            shard_fp32_from_float16_groups,
        )
    # <<<

    # >>>
    # @classmethod
    # def build_main_grad_views_for_grad_norm(cls, opt_group_ranges, optimizer):

    #     grad_views = []
    #     for group_index, opt_group_range in enumerate(opt_group_ranges):
    #         opt_grad = optimizer.param_groups[group_index]["params"][0].grad
    #         for param, range in opt_group_range["param_map"].items():
    #             if param_is_not_shared(param) and \
    #                param_is_not_tensor_parallel_duplicate(param):
                    
    #                 grad_view = opt_grad[range.start:range.end]
    #                 grad_views.append(grad_view)

    #     return grad_views
    # <<<

    def __init__(self, optimizer, clip_grad, log_num_zeros_in_grad,
                 params_have_main_grad, use_contiguous_buffers_in_local_ddp,
                 fp16, bf16, grad_scaler, models):

        super().__init__(
            optimizer, clip_grad, log_num_zeros_in_grad,
            params_have_main_grad, use_contiguous_buffers_in_local_ddp,
            fp16, bf16, grad_scaler, models)

        # Verify that contiguous buffers are being used
        # - Note: this should already be checked in arguments.py
        # >>>
        # args = get_args()
        # assert args.use_contiguous_buffers_in_local_ddp
        assert use_contiguous_buffers_in_local_ddp
        # <<<

        # Model grad buffer ranges.
        self.model_gbuf_ranges = []
        for model_index, model in enumerate(self.models):
            self.model_gbuf_ranges.append(self.build_model_gbuf_range_map(model))
        self.model_param_gbuf_map = \
            self.build_model_param_gbuf_map(self.model_gbuf_ranges)

        # Optimizer ranges.
        self.opt_group_ranges = self.build_optimizer_group_ranges(
            self.optimizer.param_groups,
            self.model_gbuf_ranges)

        # Allocate main param shards.
        (
            self.full_float16_groups,
            self.full_fp32_groups,
            self.shard_float16_groups,
            self.shard_fp32_groups,
            self.shard_fp32_from_float16_groups,
        ) = self.build_model_and_main_param_groups(self.model_gbuf_ranges,
                                                   self.model_param_gbuf_map,
                                                   self.opt_group_ranges)

        # print_seq("16 [%d], 16x32 [%d], 32 [%d]." % (
        #     sum(len(g) for g in self.float16_groups),
        #     sum(len(g) for g in self.fp32_from_float16_groups),
        #     sum(len(g) for g in self.fp32_groups),
        # ))

        # Update optimizer groups.
        # - Also, leverage state_dict() and load_state_dict() to
        #   recast preexisting per-param state tensors.
        self.optimizer.param_groups = \
            [ g["orig_group"] for g in self.opt_group_ranges ]
        self.optimizer.load_state_dict(self.optimizer.state_dict())

        # >>>
        # # Initialize main params.
        # self._copy_model_params_to_main_params()
        # <<<

        # >>>
        # # Params for grad norm.
        # self.main_grad_views_for_grad_norm = self.build_main_grad_views_for_grad_norm(
        #     self.opt_group_ranges,
        #     self.optimizer)
        # <<<


    def get_model_param_range_map(self, param):
        model_index, dtype = self.model_param_gbuf_map[param]
        gbuf_range_map = self.model_gbuf_ranges[model_index][dtype]
        param_range_map = gbuf_range_map["param_map"][param]
        
        # >>>
        # pax(0, {
        #     "param" : param,
        #     "model_index" : model_index,
        #     "dtype" : str(dtype),
        #     "gbuf_range_map" : gbuf_range_map,
        #     "param_range_map" : param_range_map,
        # })
        # <<<

        return param_range_map


    def get_model_parallel_group(self):
        return None


    # def get_main_params(self):
    #     return [ g["params"][0] for g in self.optimizer.param_groups ]
    # def get_main_grads(self):
    #     return [ p.grad for p in self.get_main_params() ]
    # def get_main_param(self, group_index):
    #     return self.get_main_params()[group_index]
    # def get_main_grad(self, group_index):
    #     return self.get_main_param(group_index).grad


    # >>>
    # def get_main_grads_for_grad_norm(self):
    #     return self.main_grad_views_for_grad_norm
    def get_main_grads_for_grad_norm(self):
        raise Exception("does 'super' work?")
    # <<<


    # def state_dict(self):
    #     state_dict = {}
    #     state_dict['optimizer'] = self.optimizer.state_dict()
    #     if self.grad_scaler:
    #         state_dict['grad_scaler'] = self.grad_scaler.state_dict()
    #     state_dict['groups'] = [g['params'] for g in self.optimizer.param_groups]
    #     return state_dict
    def state_dict(self):
        raise Exception("fix me.")


    # def load_state_dict(self, state_dict):
    #     # Optimizer.
    #     optimizer_key = 'optimizer'
    #     if optimizer_key not in state_dict:
    #         optimizer_key = 'optimizer_state_dict'
    #         print_rank_0('***WARNING*** loading optimizer from '
    #                      'an old checkpoint ...')
    #     self.optimizer.load_state_dict(state_dict[optimizer_key])

    #     # Grad scaler.
    #     if 'grad_scaler' not in state_dict:
    #         print_rank_0('***WARNING*** found an old checkpoint, will not '
    #                      'load grad scaler ...')
    #     else:
    #         if self.grad_scaler:
    #             self.grad_scaler.load_state_dict(state_dict['grad_scaler'])
    #         else:
    #             print_rank_0('***WARNING*** fould the grad scaler in the '
    #                          'checkpoint but it is None in the class. '
    #                          'Skipping loading grad scaler ...')

    #     # Copy data for the main params.
    #     current_groups = [ g["params"] for g in self.optimizer.param_groups ]
    #     assert "groups" in state_dict, "key 'groups' not in state_dict."
    #     for current_group, saved_group in zip(current_groups, state_dict["groups"]):
    #         for current_param, saved_param in zip(current_group, saved_group):
    #             current_param.data.copy_(saved_param.data)
    def load_state_dict(self, state_dict):
        raise Exception("hi.")

    # def zero_grad(self, set_to_none=True):

    #     # Collect model params.
    #     model_params = []
    #     for model in self.models:
    #         for dtype, param_map in model._grad_buffer_param_index_map.items():
    #             model_params.extend(param_map.keys())

    #     # Distributed optimizer requires contiguous buffer; don't set to None.
    #     _zero_grad_group_helper(model_params, set_to_none = False)
    # def zero_grad(self, set_to_none=True):
    #     raise Exception("does 'super' work?")
    # >>>
    def zero_grad(self, set_to_none=True):
        """We only need to zero the model related parameters, i.e.,
        float16_groups & fp32_groups. We additionally zero
        fp32_from_float16_groups as a memory optimization to reduce
        fragmentation; in the case of set_to_none==True, the space
        used by this field can be safely deallocated at this point."""
        for groups in (
                self.full_float16_groups,
                self.full_fp32_groups,
                self.shard_fp32_from_float16_groups):
            for group in groups:
                _zero_grad_group_helper(group, set_to_none)
    # <<<


    def get_model_grad_buffer_dp_views(self):

        data_parallel_world_size = mpu.get_data_parallel_world_size()

        # Grad buffer views.
        gbuf_view_items = []
        for model_index, model in enumerate(self.models):
            for dtype, gbuf in model._grad_buffers.items():

                assert gbuf.numel_padded % data_parallel_world_size == 0
                shard_size = int(gbuf.numel_padded / data_parallel_world_size)
                gbuf_views = [gbuf.data[(r*shard_size):((r+1)*shard_size)]
                              for r in range(data_parallel_world_size)]
                gbuf_view_items.append((model_index, dtype, gbuf.data, gbuf_views))

        return gbuf_view_items

    def reduce_model_grads(self, args, timers):
        '''Note: this is a different order of reduction, versus the non-
           distributed optimizer, which reduces: 1) all grads, 2) embedding
           grads.
        '''

        # >>>
        # print_seq([
        #     tp(b.data)
        #     for m in self.models
        #     for b in m._grad_buffers.values()
        # ])
        # <<<

        # All-reduce embedding grads.
        timers('backward-embedding-all-reduce').start()
        self.allreduce_embedding_grads(args)
        timers('backward-embedding-all-reduce').stop()

        # Reduce-scatter setup.
        timers('backward-params-all-reduce').start()
        data_parallel_rank = mpu.get_data_parallel_rank()
        data_parallel_world_size = mpu.get_data_parallel_world_size()
        data_parallel_group = mpu.get_data_parallel_group()

        # Scale grad buffers by '1 / data_parallel_world_size'.
        for model in self.models:
            for dtype, gbuf in model._grad_buffers.items():
                gbuf.data /= data_parallel_world_size

        # Reduce-scatter all grads.
        gbuf_view_items = self.get_model_grad_buffer_dp_views()
        for index, (model_index, dtype, gbuf, gbuf_views) in enumerate(gbuf_view_items):
            torch.distributed._reduce_scatter_base(
                gbuf_views[data_parallel_rank],
                gbuf,
                group = data_parallel_group,
            )

        timers('backward-params-all-reduce').stop()

    def gather_model_params(self, args, timers):

        raise Exception("hi.")

        timers('backward-params-all-gather').start()

        data_parallel_rank = mpu.get_data_parallel_rank()
        data_parallel_group = mpu.get_data_parallel_group()

        # All-gather updated main params.
        # - All grad buffer views are guaranteed to have the same num elements
        #   across all data parallel ranks, with grad buffer padding that is done
        #   in distributed.py. Thus, all sub-views will have consistent start/end
        #   indexes across data parallel ranks.
        gbuf_view_items = self.get_model_grad_buffer_dp_views()
        for index, (model_index, dtype, gbuf, gbuf_views) in enumerate(gbuf_view_items):
            torch.distributed._all_gather_base(
                gbuf,
                gbuf_views[data_parallel_rank],
                group = data_parallel_group,
            )

        # Each model param now contains its updated values in its
        # '.main_grad' field.
        for model in self.models:
            for dtype, param_map in model._grad_buffer_param_index_map.items():
                for param in param_map:
                    param.detach().copy_(param.main_grad)

        timers('backward-params-all-gather').stop()

    def _collect_main_grad_data_for_unscaling(self):
        raise Exception("hi.")
        return [ g.data for g in self.get_main_grads() ]

    # >>>
    # def _copy_model_params_to_main_params(self):

    #     for group_index, group_range in enumerate(self.opt_group_ranges):
    #         main_param = self.get_main_param(group_index)
    #         for model_param, main_range in group_range["param_map"].items():

    #             # Model range.
    #             # model_index, dtype = self.param_gbuf_map[model_param]
    #             # model_range = self.model_gbuf_ranges \
    #             #     [model_index][dtype]["param_map"][model_param]["param"]
    #             model_range = self.get_model_param_range_map(model_param)["param"]

    #             assert main_range.size == model_range.size

    #             # Copy shard data.
    #             main_view = main_param[main_range.start:main_range.end]
    #             model_view = model_param.view(-1)[model_range.start:model_range.end]

    #             main_view.detach().copy_(model_view)
    def _copy_model_params_to_main_params(self):
        raise Exception("check if super's copy works.")
    # <<<

    # >>>
    # def _copy_model_grads_to_main_grads(self):

    #     for group_index, group_range in enumerate(self.opt_group_ranges):
    #         for model_param, main_range in group_range["param_map"].items():

    #             # Model range.
    #             # model_index, dtype = self.param_gbuf_map[model_param]
    #             # model_range = self.model_gbuf_ranges \
    #             #     [model_index][dtype]["param_map"][model_param]["gbuf_world"]
    #             model_range = self.get_model_param_range_map(model_param)["gbuf_world"]

    #             assert main_range.size == model_range.size

    #             # Copy from DDP's contiguous buffer to main shard's grad.
    #             model_grad = self.models[model_index]._grad_buffers[dtype].data
    #             main_grad = self.get_main_grad(group_index)

    #             # Copy sub-range within tensor.
    #             model_view = model_grad[model_range.start:model_range.end]
    #             main_view = main_grad[main_range.start:main_range.end]

    #             main_view.detach().copy_(model_view)
    # def _copy_model_grads_to_main_grads(self):
    #     super()._copy_model_grads_to_main_grads()
    #     raise Exception("check main param '.grad'.")

    #     for group in self.optimizer.param_groups:
    #         for param in group["params"]:
    #             param.grad = 
    def _copy_model_grads_to_main_grads(self):

        # >>>
        # print_seq([
        #     "grad = %s." % tp(p.grad)
        #     for g in self.optimizer.param_groups
        #     for p in g["params"]
        # ])
        # <<<

        # This only needs to be done for the float16 group.
        for full_model_group, shard_main_group in zip(
                self.full_float16_groups,
                self.shard_fp32_from_float16_groups):
            for full_model_param, shard_main_param in zip(full_model_group,
                                                          shard_main_group):

                param_range_map = self.get_model_param_range_map(full_model_param)
                param_range = param_range_map["param"]
                full_model_grad = full_model_param.main_grad
                shard_model_grad = \
                    full_model_grad[param_range.start:param_range.end]
                shard_main_param.grad = shard_model_grad.float()

                # >>>
                if full_model_param.nelement() != shard_main_param.nelement():
                    pax(0, {
                        "param_range_map" : param_range_map,
                        "param_range" : param_range,
                        "full_model_param" : tp(full_model_param),
                        "full_model_grad" : tp(full_model_grad),
                        "shard_model_grad" : tp(shard_model_grad),
                        "shard_main_grad" : tp(shard_main_param.grad),
                        "shard_main_param" : tp(shard_main_param),
                    })
                # <<<

        # For fp32 grads, we need to reset the grads to main grad.
        for group in self.fp32_groups:
            for param in group:
                param.grad = param.main_grad

        # >>>
        print_seq([
            "grad = %s." % tp(p.grad)
            for g in self.optimizer.param_groups
            for p in g["params"]
        ])
        # <<<

    # <<<

    # >>>
    # def _copy_main_params_to_model_params(self):

    #     for group_index, group_range in enumerate(self.opt_group_ranges):
    #         for model_param, main_range in group_range["param_map"].items():

    #             # model_index, dtype = self.param_gbuf_map[model_param]
    #             # model_range = self.model_gbuf_ranges \
    #             #     [model_index][dtype]["param_map"][model_param]["gbuf_world"]
    #             model_range = self.get_model_param_range_map(model_param)["gbuf_world"]

    #             assert main_range.size == model_range.size

    #             # Use DDP's contiguous buffer to temporarily hold params.
    #             model_param = self.models[model_index]._grad_buffers[dtype].data
    #             main_param = self.get_main_param(group_index)

    #             # Copy sub-range within tensor.
    #             model_view = model_param[model_range.start:model_range.end]
    #             main_view = main_param[main_range.start:main_range.end]

    #             model_view.detach().copy_(main_view)
    # def _copy_main_params_to_model_params(self):
    #     super()._copy_main_params_to_model_params()
    #     raise Exception("check main param '.grad'.")
    def _copy_main_params_to_model_params(self):
        raise Exception("hi.")

        # This only needs to be done for the float16 group.
        for model_group, main_group in zip(self.float16_groups,
                                           self.fp32_from_float16_groups):
            for model_param, main_param in zip(model_group, main_group):
                model_param.main_grad.detach().copy_(main_param)

        # For fp32 grads, we need to reset the grads to main grad.
        for group in self.fp32_groups:
            for param in group:
                param.main_grad.detach().copy_(param)
    # <<<
