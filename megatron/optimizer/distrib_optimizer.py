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

# >>>
from lutil import pax, tp, print_seq
# <<<


class Range:

    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.size = end - start
    def normalize(self, start = 0):
        return Range(start, start + self.size)
    def __str__(self):
        return "%d,%d [%d]" % (self.start, self.end, self.size)


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
        group_ranges = [ {"params": []} for _ in param_groups ]
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

        return group_ranges


    @classmethod
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

        # Allocate (or slice) each group's param shard.
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

                assert model_param.requires_grad

                model_index, dtype = param_gbuf_map[model_param]
                gbuf_range = model_gbuf_ranges[model_index][dtype]
                param_range = gbuf_range["param_map"][model_param]["param"]

                # fp16, bf16 params.
                if model_param.type() in ['torch.cuda.HalfTensor',
                                          'torch.cuda.BFloat16Tensor']:

                    # Clone model -> main.
                    shard_model_param = model_param.detach().view(-1) \
                        [param_range.start:param_range.end]
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
                elif model_param.type() == 'torch.cuda.FloatTensor':
                    shard_model_param = model_param.view(-1) \
                        [param_range.start:param_range.end]
                    full_fp32_params_this_group.append(model_param)
                    shard_fp32_params_this_group.append(shard_model_param)
                    mpu.copy_tensor_model_parallel_attributes(
                        shard_model_param, model_param)

                else:
                    raise TypeError('Wrapped parameters must be one of '
                                    'torch.cuda.FloatTensor,  '
                                    'torch.cuda.HalfTensor, or '
                                    'torch.cuda.BFloat16Tensor. '
                                    'Received {}'.format(param.type()))

            # Update optimizer's params.
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


    def __init__(self, optimizer, clip_grad, log_num_zeros_in_grad,
                 params_have_main_grad, use_contiguous_buffers_in_local_ddp,
                 fp16, bf16, grad_scaler, models):

        super().__init__(
            optimizer, clip_grad, log_num_zeros_in_grad,
            params_have_main_grad, use_contiguous_buffers_in_local_ddp,
            fp16, bf16, grad_scaler, models)

        # Verify that contiguous buffers are being used
        # - Note: this should already be checked in arguments.py
        assert use_contiguous_buffers_in_local_ddp

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

        # Update optimizer groups.
        # - Also, leverage state_dict() and load_state_dict() to
        #   recast preexisting per-param state tensors.
        self.optimizer.param_groups = \
            [ g["orig_group"] for g in self.opt_group_ranges ]
        self.optimizer.load_state_dict(self.optimizer.state_dict())


    def get_model_param_range_map(self, param):
        model_index, dtype = self.model_param_gbuf_map[param]
        gbuf_range_map = self.model_gbuf_ranges[model_index][dtype]
        param_range_map = gbuf_range_map["param_map"][param]
        return param_range_map


    def get_model_parallel_group(self):
        return None


    def state_dict(self):
        state_dict = {}
        state_dict['optimizer'] = self.optimizer.state_dict()
        if self.grad_scaler:
            state_dict['grad_scaler'] = self.grad_scaler.state_dict()
        state_dict['shard_fp32_from_float16_groups'] = \
            self.shard_fp32_from_float16_groups
        return state_dict


    def load_state_dict(self, state_dict):

        # Optimizer.
        optimizer_key = 'optimizer'
        if optimizer_key not in state_dict:
            optimizer_key = 'optimizer_state_dict'
            print_rank_0('***WARNING*** loading optimizer from '
                         'an old checkpoint ...')
        self.optimizer.load_state_dict(state_dict[optimizer_key])

        # Grad scaler.
        if 'grad_scaler' not in state_dict:
            print_rank_0('***WARNING*** found an old checkpoint, will not '
                         'load grad scaler ...')
        else:
            if self.grad_scaler:
                self.grad_scaler.load_state_dict(state_dict['grad_scaler'])
            else:
                print_rank_0('***WARNING*** fould the grad scaler in the '
                             'checkpoint but it is None in the class. '
                             'Skipping loading grad scaler ...')

        # Copy data for the main params.
        for current_group, saved_group in zip(
                self.shard_fp32_from_float16_groups,
                state_dict["shard_fp32_from_float16_groups"]):
            for current_param, saved_param in zip(current_group, saved_group):
                current_param.data.copy_(saved_param.data)


    def zero_grad(self, set_to_none=True):
        """We only need to zero the model related parameters, i.e.,
        float16_groups & fp32_groups. We additionally zero
        fp32_from_float16_groups as a memory optimization to reduce
        fragmentation; in the case of set_to_none==True, the space
        used by this field can be safely deallocated at this point."""
        # >>>
        # params = [ p for g in self.shard_fp32_groups for p in g ]
        # pax(0, {
        #     "shard_fp32_groups" : self.shard_fp32_groups,
        #     "params" : params,
        #     "grads" : [ p.grad for p in params ],
        # })
        # <<<
        for groups in (
                self.full_float16_groups,
                self.full_fp32_groups,
                self.shard_float16_groups, # grad empty/unused here?
                self.shard_fp32_groups, # throws grad-access warning
                self.shard_fp32_from_float16_groups):
            for group in groups:
                _zero_grad_group_helper(group, set_to_none)


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
        return [
            param.grad.data
            for group in self.optimizer.param_groups
            for param in group["params"]
        ]


    def _get_model_and_main_params_data_float16(self):
        model_data = []
        main_data = []
        for model_group, main_group in zip(self.shard_float16_groups,
                                           self.shard_fp32_from_float16_groups):
            for model_param, main_param in zip(model_group, main_group):
                model_data.append(model_param.data)
                main_data.append(main_param.data)
        return model_data, main_data


    def _copy_model_grads_to_main_grads(self):

        def copy_group_grads(full_model_groups, shard_main_groups):
            for full_model_group, shard_main_group in zip(full_model_groups,
                                                          shard_main_groups):
                for full_model_param, shard_main_param in zip(full_model_group,
                                                              shard_main_group):

                    param_range_map = self.get_model_param_range_map(full_model_param)
                    param_range = param_range_map["param"]
                    assert param_range.size == shard_main_param.nelement()

                    full_model_grad = full_model_param.main_grad
                    shard_model_grad = full_model_grad.view(-1) \
                        [param_range.start:param_range.end]
                    shard_main_param.grad = shard_model_grad.float()

        copy_group_grads(self.full_float16_groups,
                         self.shard_fp32_from_float16_groups)
        copy_group_grads(self.full_fp32_groups,
                         self.shard_fp32_groups)


    def _copy_main_params_to_model_params(self):

        def copy_group_params(shard_main_groups, full_model_groups):
            for shard_main_group, full_model_group in zip(shard_main_groups,
                                                          full_model_groups):
                for shard_main_param, full_model_param in zip(shard_main_group,
                                                              full_model_group):

                    param_range_map = self.get_model_param_range_map(full_model_param)
                    param_range = param_range_map["param"]
                    assert param_range.size == shard_main_param.nelement()

                    full_model_grad = full_model_param.main_grad
                    shard_model_grad = full_model_grad.view(-1) \
                        [param_range.start:param_range.end]

                    shard_model_grad.data.copy_(shard_main_param)

        copy_group_params(self.shard_fp32_from_float16_groups,
                          self.full_float16_groups)
        copy_group_params(self.shard_fp32_groups,
                          self.full_fp32_groups)
