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

from .optimizer import MixedPrecisionOptimizer, _zero_grad_group_helper


class Shard:
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.size = end - start
    def normalize(self, start = 0):
        return Shard(start, start + self.size)
    def __str__(self):
        return "%d,%d [%d]" % (self.start, self.end, self.size)


class DistributedOptimizer(MixedPrecisionOptimizer):

    @classmethod
    def get_model_gbuf_param_shard_map(cls, model, dtype, gbuf_world_shard):

        # Param shard map.
        param_world_index_map = model._grad_buffer_param_index_map[dtype]
        param_shard_map = {}
        for param, param_world_indexes in param_world_index_map.items():

            # Shard range.
            param_world_start, param_world_end = param_world_indexes
            param_local_start = max(
                0,
                param_world_start - gbuf_world_shard.start)
            param_local_end = min(
                gbuf_world_shard.size,
                param_world_end - gbuf_world_shard.start)

            # Add shard, if within range.
            if param_local_end > param_local_start:
                param_local_shard = Shard(param_local_start, param_local_end)
                param_world_shard = param_local_shard.normalize(
                    param_local_start + gbuf_world_shard.start)
                sub_param_start = max(0, gbuf_world_shard.start-param_world_start)
                sub_param_shard = param_local_shard.normalize(sub_param_start)
                param_shard_map[param] = {
                    "gbuf_world" : param_world_shard,
                    "gbuf_local" : param_local_shard,
                    "param" : sub_param_shard,
                }

        return param_shard_map

    @classmethod
    def get_model_gbuf_shard(cls, model, dtype):

        data_parallel_rank = mpu.get_data_parallel_rank()
        data_parallel_world_size = mpu.get_data_parallel_world_size()

        # Grad buffer shard.
        grad_buffer = model._grad_buffers[dtype]
        gbuf_size = grad_buffer.numel
        max_gbuf_shard_size = int(math.ceil(gbuf_size / data_parallel_world_size))

        # All world shards. (i.e., across all data parallel ranks)
        gbuf_world_all_shards = []
        for r in range(data_parallel_world_size):
            gbuf_world_start = r * max_gbuf_shard_size
            gbuf_world_end = min(gbuf_size, gbuf_world_start+max_gbuf_shard_size)
            gbuf_world_shard = Shard(gbuf_world_start, gbuf_world_end)
            gbuf_world_all_shards.append(gbuf_world_shard)

        # Local DP's shards.
        gbuf_world_shard = gbuf_world_all_shards[data_parallel_rank]
        gbuf_local_shard = gbuf_world_shard.normalize()

        # Get each param's shards.
        param_shard_map = cls.get_model_gbuf_param_shard_map(model,
                                                             dtype,
                                                             gbuf_world_shard)

        # Altogether.
        data = {
            "local" : gbuf_local_shard,
            "world" : gbuf_world_shard,
            "world_all" : gbuf_world_all_shards,
            "param_map" : param_shard_map,
            "max_shard_size" : max_gbuf_shard_size,
        }

        return data

    @classmethod
    def get_model_gbuf_shard_map(cls, model):
        return {
            dtype : cls.get_model_gbuf_shard(model, dtype)
            for dtype in model._grad_buffers
        }

    @classmethod
    def get_param_gbuf_map(cls, model_gbuf_shards):
        '''Create a reverse of the model_gbuf_shards, for referencing in
        opposite direction.'''
        param_gbuf_map = {}
        for model_index, model_gbuf_shard_map in enumerate(model_gbuf_shards):
            for dtype, gbuf_shard_map in model_gbuf_shard_map.items():
                for param, param_shard_map in gbuf_shard_map["param_map"].items():
                    param_gbuf_map[param] = (model_index, dtype)
        return param_gbuf_map

    @classmethod
    def get_optimizer_group_shards(cls, param_groups, model_gbuf_shards):

        num_groups = len(param_groups)

        # Param group map.
        param_group_map = {}
        for group_index, group in enumerate(param_groups):
            for param in group["params"]:
                assert param.requires_grad
                param_group_map[param] = group_index

        # Optimizer group shards.
        group_shards = [ {"size": 0, "param_map": {}} for _ in param_groups ]
        for model_gbuf_shard_map in model_gbuf_shards:
            for dtype, gbuf_shard_map in model_gbuf_shard_map.items():
                for param in gbuf_shard_map["param_map"]:
                    
                    group_index = param_group_map[param]
                    group_shard = group_shards[group_index]
                    param_size = gbuf_shard_map["param_map"][param]["param"].size

                    param_group_start = group_shard["size"]
                    param_group_end = param_group_start + param_size
                    param_group_shard = Shard(param_group_start, param_group_end)

                    group_shard["size"] += param_size
                    group_shard["param_map"][param] = param_group_shard

        # Squeeze zero-size group shards.
        for group_index, group_shard in enumerate(group_shards):
            group_shard["orig_group"] = param_groups[group_index]
        group_shards = [ g for g in group_shards if g["size"] > 0 ]

        return group_shards

    @classmethod
    def allocate_main_param_shards(cls, opt_group_shards):

        # Allocator method.
        allocate_shard = lambda shard_size, dtype : torch.empty(
            (shard_size,),
            dtype = dtype,
            device = torch.cuda.current_device(),
            requires_grad = True)

        # Allocate each group's param/grad shard.
        for group_index, group_shard in enumerate(opt_group_shards):

            group_size = group_shard["size"]
            assert group_size != 0, "temporary check ... remove me."

            # Allocate shard.
            main_param = allocate_shard(group_size, torch.float)
            main_param.grad = allocate_shard(group_size, torch.float)
            mpu.set_tensor_model_parallel_attributes(main_param, True, 0, 1)

            # Update group's param.
            group_shard["orig_group"]["params"] = [ main_param ]


    def __init__(self, optimizer, clip_grad, log_num_zeros_in_grad,
                 params_have_main_grad, use_contiguous_buffers_in_local_ddp,
                 bf16, grad_scaler, models):

        super().__init__(
            optimizer, clip_grad, log_num_zeros_in_grad,
            params_have_main_grad, use_contiguous_buffers_in_local_ddp,
            bf16, grad_scaler, models)

        # Verify that contiguous buffers are being used
        # - Note: this should already be checked in arguments.py
        args = get_args()
        assert args.use_contiguous_buffers_in_local_ddp

        # Model grad buffer shards.
        self.model_gbuf_shards = []
        for model_index, model in enumerate(self.models):
            self.model_gbuf_shards.append(self.get_model_gbuf_shard_map(model))
        self.param_gbuf_map = self.get_param_gbuf_map(self.model_gbuf_shards)

        # Optimizer shards.
        self.opt_group_shards = self.get_optimizer_group_shards(
            self.optimizer.param_groups,
            self.model_gbuf_shards)

        # Allocate main param shards.
        self.allocate_main_param_shards(self.opt_group_shards)

        # Update optimizer groups.
        # - Also, leverage state_dict() and load_state_dict() to
        #   recast preexisting per-param state tensors.
        self.optimizer.param_groups = \
            [ g["orig_group"] for g in self.opt_group_shards ]
        self.optimizer.load_state_dict(self.optimizer.state_dict())

        # Initialize main params.
        self._copy_model_params_to_main_params()

    def get_model_parallel_group(self):
        return None

    # >>>
    @staticmethod
    def has_nan_debug(tensors):
        if isinstance(tensors, torch.Tensor):
            tensors = [ tensors ]
        assert isinstance(tensors, list)
        has_nans = [ (not torch.all(torch.isfinite(t)).item()) for t in tensors ]
        has_nan = any(has_nans)
        return has_nan
    # def get_local_model_param_views(self):
    #     '''** FOR DEBUGGING. **'''
    #     model_param_views = []
    #     for group_index, opt_group_shard in enumerate(self.opt_group_shards):
    #         for param, opt_shard in opt_group_shard["param_map"].items():
    #             model_index, dtype = self.param_gbuf_map[param]
    #             gbuf_shard_map = \
    #                 self.model_gbuf_shards[model_index][dtype]["param_map"][param]
    #             model_param_shard = gbuf_shard_map["param"]
    #             model_param_views.append(
    #                 param.view(-1)[model_param_shard.start:model_param_shard.end])
    #     return model_param_views
    # def get_local_model_grad_views(self):
    #     '''** FOR DEBUGGING. **'''
    #     model_grad_views = []
    #     for group_index, opt_group_shard in enumerate(self.opt_group_shards):
    #         for param, opt_shard in opt_group_shard["param_map"].items():
    #             model_index, dtype = self.param_gbuf_map[param]
    #             gbuf = self.models[model_index]._grad_buffers[dtype].data
    #             gbuf_shard_map = \
    #                 self.model_gbuf_shards[model_index][dtype]["param_map"][param]
    #             gbuf_world_shard = gbuf_shard_map["gbuf_world"]
    #             model_grad_views.append(
    #                 gbuf[gbuf_world_shard.start:gbuf_world_shard.end])
    #     return model_grad_views
    # def get_world_model_params(self):
    #     '''** FOR DEBUGGING. **'''
    #     return [ p for m in self.models for p in m.parameters() ]
    # def get_world_model_grads(self):
    #     '''** FOR DEBUGGING. **'''
    #     return [ p.main_grad for p in self.get_world_model_params() ]
    # <<<

    def get_main_params(self):
        return [ g["params"][0] for g in self.optimizer.param_groups ]
    def get_main_grads(self):
        return [ p.grad for p in self.get_main_params() ]
    def get_main_param(self, group_index):
        return self.get_main_params()[group_index]
    def get_main_grad(self, group_index):
        return self.get_main_param(group_index).grad

    def state_dict(self):
        state_dict = {}
        state_dict['optimizer'] = self.optimizer.state_dict()
        if self.grad_scaler:
            state_dict['grad_scaler'] = self.grad_scaler.state_dict()
        state_dict['groups'] = [g['params'] for g in self.optimizer.param_groups]
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
        current_groups = [ g["params"] for g in self.optimizer.param_groups ]
        assert "groups" in state_dict, "key 'groups' not in state_dict."
        for current_group, saved_group in zip(current_groups, state_dict["groups"]):
            for current_param, saved_param in zip(current_group, saved_group):
                current_param.data.copy_(saved_param.data)

    def zero_grad(self, set_to_none=True):

        # Collect model params.
        model_params = []
        for model in self.models:
            for dtype, param_map in model._grad_buffer_param_index_map.items():
                model_params.extend(param_map.keys())

        # Distributed optimizer requires contiguous buffer; don't set to None.
        _zero_grad_group_helper(model_params, set_to_none = False)

    # >>>
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
                gbuf_view_items.append((model_index, dtype, gbuf_views))

        return gbuf_view_items
    def get_model_grad_buffer_dp_views_SUB(self, sub_view_numel):
        gbuf_view_items = self.get_model_grad_buffer_dp_views()
        sub_view_items = []
        for model_index, dtype, gbuf_views in gbuf_view_items:

            # ** Sanity check. ** (should be unnecessary; see comment above)
            view_numel = gbuf_views[0].nelement()
            for view in gbuf_views:
                assert view.nelement() == view_numel

            for start_index in range(0, view_numel, sub_view_numel):
                end_index = min(view_numel, start_index + sub_view_numel)
                sub_views = [ t[start_index:end_index] for t in gbuf_views ]
                sub_view_items.append((model_index, dtype, sub_views))

        # >>>
        from lutil import pax
        pax(0, {
            "gbuf_view_items" : [(a,b,c.shape) for a,b,c in gbuf_view_items],
            "sub_view_items" : [(a,b,c.shape) for a,b,c in sub_view_items],
        })
        # <<<

        return sub_view_items
    # def get_model_grad_buffers_SINGLE(self):

    #     data_parallel_world_size = mpu.get_data_parallel_world_size()

    #     # Grad buffers.
    #     gbuf_items = []
    #     for model_index, model in enumerate(self.models):
    #         for dtype, gbuf in model._grad_buffers.items():

    #             assert gbuf.numel_padded % data_parallel_world_size == 0
    #             shard_size = int(gbuf.numel_padded / data_parallel_world_size)
    #             gbuf_items.append((model_index, dtype, gbuf.data))

    #     return gbuf_items
    # <<<

    # >>>
    def reduce_model_grads_0(self, args, timers):
        '''Note: this is a different order of reduction, versus the non-
           distributed optimizer, which reduces: 1) all grads, 2) embedding
           grads.
        '''

        # All-reduce embedding grads.
        timers('backward-embedding-all-reduce').start()
        self.allreduce_embedding_grads(args)
        timers('backward-embedding-all-reduce').stop()

        # Reduce-scatter all grads.
        timers('backward-params-all-reduce').start()
        data_parallel_rank = mpu.get_data_parallel_rank()
        data_parallel_world_size = mpu.get_data_parallel_world_size()
        data_parallel_group = mpu.get_data_parallel_group()

        gbuf_view_items = self.get_model_grad_buffer_dp_views()
        for model_index, dtype, gbuf_views in gbuf_view_items:
            gbuf = self.models[model_index]._grad_buffers[dtype].data
            gbuf /= data_parallel_world_size
            torch.distributed.reduce_scatter(
                gbuf_views[data_parallel_rank],
                gbuf_views,
                group = data_parallel_group,
            )
        timers('backward-params-all-reduce').stop()
    def reduce_model_grads_1(self, args, timers):
        '''Note: this is a different order of reduction, versus the non-
           distributed optimizer, which reduces: 1) all grads, 2) embedding
           grads.
        '''

        # All-reduce embedding grads.
        timers('backward-embedding-all-reduce').start()
        self.allreduce_embedding_grads(args)
        timers('backward-embedding-all-reduce').stop()

        # Reduce-scatter all grads.
        timers('backward-params-all-reduce').start()
        data_parallel_rank = mpu.get_data_parallel_rank()
        data_parallel_world_size = mpu.get_data_parallel_world_size()
        data_parallel_group = mpu.get_data_parallel_group()

        sub_numel = 1 * 1048576
        sub_view_items = self.get_model_grad_buffer_dp_views_SUB(sub_numel)
        for model_index, dtype, gbuf_views in gbuf_view_items:
            gbuf = self.models[model_index]._grad_buffers[dtype].data
            gbuf /= data_parallel_world_size
            torch.distributed.reduce_scatter(
                gbuf_views[data_parallel_rank],
                gbuf_views,
                group = data_parallel_group,
            )
        timers('backward-params-all-reduce').stop()
    def reduce_model_grads(self, *args):
        # >>>
        # return
        # <<<
        # self.reduce_model_grads_0(*args)
        self.reduce_model_grads_1(*args)
    # <<<

    # >>>
    def gather_model_params_0(self, args, timers):

        timers('backward-params-all-gather').start()

        data_parallel_rank = mpu.get_data_parallel_rank()
        data_parallel_group = mpu.get_data_parallel_group()

        # All-gather updated main params.
        gbuf_view_items = self.get_model_grad_buffer_dp_views()
        for model_index, dtype, gbuf_views in gbuf_view_items:
            torch.distributed.all_gather(
                gbuf_views,
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
    def gather_model_params_1(self, args, timers):

        timers('backward-params-all-gather').start()

        data_parallel_rank = mpu.get_data_parallel_rank()
        data_parallel_group = mpu.get_data_parallel_group()

        # All-gather updated main params.
        # - All grad buffer views are guaranteed to have the same num elements
        #   across all data parallel ranks, with grad buffer padding that is done
        #   in distributed.py. Thus, all sub-views will have consistent start/end
        #   indexes across data parallel ranks.
        gbuf_view_items = self.get_model_grad_buffer_dp_views()

        # sub_view_numel = 1 * 1024
        # sub_view_numel = 1 * 131072
        sub_view_numel = 1 * 1048576
        for model_index, dtype, gbuf_views in gbuf_view_items:

            # ** Sanity check. ** (should be unnecessary; see comment above)
            view_numel = gbuf_views[0].nelement()
            for view in gbuf_views:
                assert view.nelement() == view_numel

            for start_index in range(0, view_numel, sub_view_numel):

                end_index = min(view_numel, start_index + sub_view_numel)
                sub_views = [ t[start_index:end_index] for t in gbuf_views ]

                torch.distributed.all_gather(
                    sub_views,
                    sub_views[data_parallel_rank],
                    group = data_parallel_group,
                )

        # Each model param now contains its updated values in its
        # '.main_grad' field.
        for model in self.models:
            for dtype, param_map in model._grad_buffer_param_index_map.items():
                for param in param_map:
                    param.detach().copy_(param.main_grad)

        timers('backward-params-all-gather').stop()
    # def gather_model_params_2(self, args, timers):

    #     raise Exception("_all_gather_base not applicable when each DP rank owns contiguous range of grad buffer.")

    #     timers('backward-params-all-gather').start()

    #     data_parallel_rank = mpu.get_data_parallel_rank()
    #     data_parallel_world_size = mpu.get_data_parallel_world_size()
    #     data_parallel_group = mpu.get_data_parallel_group()

    #     # All-gather updated main params.
    #     # - All grad buffer views are guaranteed to have the same num elements
    #     #   across all data parallel ranks, with grad buffer padding that is done
    #     #   in distributed.py. Thus, all sub-views will have consistent start/end
    #     #   indexes across data parallel ranks.
    #     gbuf_items = self.get_model_grad_buffers_SINGLE()

    #     # local_sub_numel = 1 * 1024
    #     # local_sub_numel = 1 * 131072
    #     ideal_local_numel = 128 * 1048576
    #     ideal_world_numel = data_parallel_world_size * ideal_local_numel
    #     for model_index, dtype, gbuf in gbuf_items:

    #         gbuf_numel = gbuf.nelement()

    #         # >>>
    #         # from lutil import pax
    #         # pax(0, {
    #         #     "gbuf_items" : [ (a, b, c.shape) for a, b, c in gbuf_items ],
    #         #     "gbuf" : str(gbuf.shape),
    #         #     "gbuf_numel" : gbuf_numel,
    #         #     "local_sub_numel" : local_sub_numel,
    #         #     "world_sub_numel" : world_sub_numel,
    #         # })
    #         # <<<

    #         for world_start_index in range(0, gbuf_numel, ideal_world_numel):
    #             world_end_index = \
    #                 min(gbuf_numel, world_start_index + ideal_world_numel)
    #             world_numel = world_end_index - world_start_index
    #             assert world_numel % data_parallel_world_size == 0
    #             local_numel = int(world_numel / data_parallel_world_size)
    #             local_start_index = \
    #                 world_start_index + data_parallel_rank * local_numel
    #             local_end_index = \
    #                 min(gbuf_numel, local_start_index + local_numel)

    #             try:
    #                 world_view = gbuf[world_start_index:world_end_index]
    #                 local_view = gbuf[local_start_index:local_end_index]
    #             except:
    #                 # >>>
    #                 from lutil import pax
    #                 pax(0, {
    #                     "world_start_index" : world_start_index,
    #                     "world_end_index" : world_end_index,
    #                     "local_start_index" : local_start_index,
    #                     "local_end_index" : local_end_index,
    #                 })
    #                 # <<<
                
    #             try:
    #                 torch.distributed._all_gather_base(
    #                     world_view,
    #                     local_view,
    #                     group = data_parallel_group,
    #                 )
    #             except:
    #                 # >>>
    #                 from lutil import pax
    #                 pax(0, {
    #                     "data_parallel_rank" : data_parallel_rank,
    #                     # "local_sub_numel" : local_sub_numel,
    #                     # "world_sub_numel" : world_sub_numel,
    #                     "world_start_index" : world_start_index,
    #                     "world_end_index" : world_end_index,
    #                     "local_start_index" : local_start_index,
    #                     "local_end_index" : local_end_index,
    #                     "gbuf" : str(gbuf.shape),
    #                     "world_view" : str(world_view.shape),
    #                     "local_view" : str(local_view.shape),
    #                     "local_sub_numel / ideal" : local_sub_numel,
    #                     "local_sub_numel / act" :
    #                     local_end_index - local_start_index,
    #                 })
    #                 # <<<

    #             # >>>
    #             # from lutil import pax, tp
    #             # pax(0, {
    #             #     # "gbuf" : tp(gbuf),
    #             #     "world range" : "%d, %d"%(world_start_index, world_end_index),
    #             #     "local range" : "%d, %d"%(local_start_index, local_end_index),
    #             #     "world_view" : tp(world_view),
    #             #     "local_view" : tp(local_view),
    #             #     "gbuf view" : tp(gbuf[world_start_index:world_end_index]),
    #             # })
    #             # <<<

    #     # >>>
    #     for model_index, dtype, gbuf in gbuf_items:
    #         if self.has_nan_debug(gbuf):
    #             raise Exception("hi.")
    #     # from lutil import pax, tp
    #     # pax(0, {
    #     #     "gbuf_items" : [ (a, b, tp(c)) for a, b, c in gbuf_items ],
    #     # })
    #     # <<<

    #     # Each model param now contains its updated values in its
    #     # '.main_grad' field.
    #     for model in self.models:
    #         for dtype, param_map in model._grad_buffer_param_index_map.items():
    #             for param in param_map:
    #                 param.detach().copy_(param.main_grad)
    #                 # >>>
    #                 if self.has_nan_debug(param):
    #                     raise Exception("wha?")
    #                 # <<<

    #     timers('backward-params-all-gather').stop()
    def gather_model_params(self, *args):
        # >>>
        return
        # <<<
        # self.gather_model_params_0(*args)
        self.gather_model_params_1(*args)
        # self.gather_model_params_2(*args)

        # ~~~
        # self.debug_model(0, "after / gather_model_params", 0)
    # <<<

    def _collect_main_grad_data_for_unscaling(self):
        return [ g.data for g in self.get_main_grads() ]

    def _copy_model_params_to_main_params(self):

        for group_index, group_shard in enumerate(self.opt_group_shards):
            main_param = self.get_main_param(group_index)
            for model_param, main_shard in group_shard["param_map"].items():

                # Model shard.
                model_index, dtype = self.param_gbuf_map[model_param]
                model_shard = self.model_gbuf_shards \
                    [model_index][dtype]["param_map"][model_param]["param"]

                assert main_shard.size == model_shard.size

                # Copy shard data.
                main_view = main_param[main_shard.start:main_shard.end]
                model_view = model_param.view(-1)[model_shard.start:model_shard.end]

                main_view.detach().copy_(model_view)


    def _copy_model_grads_to_main_grads(self):

        for group_index, group_shard in enumerate(self.opt_group_shards):
            for model_param, main_shard in group_shard["param_map"].items():

                # Model shard.
                model_index, dtype = self.param_gbuf_map[model_param]
                model_shard = self.model_gbuf_shards \
                    [model_index][dtype]["param_map"][model_param]["gbuf_world"]

                assert main_shard.size == model_shard.size

                # Copy from DDP's contiguous buffer to main shard's grad.
                model_grad = self.models[model_index]._grad_buffers[dtype].data
                main_grad = self.get_main_grad(group_index)

                # Copy sub-range within tensor.
                model_view = model_grad[model_shard.start:model_shard.end]
                main_view = main_grad[main_shard.start:main_shard.end]

                main_view.detach().copy_(model_view)


    def _copy_main_params_to_model_params(self):

        for group_index, group_shard in enumerate(self.opt_group_shards):
            for model_param, main_shard in group_shard["param_map"].items():

                model_index, dtype = self.param_gbuf_map[model_param]
                model_shard = self.model_gbuf_shards \
                    [model_index][dtype]["param_map"][model_param]["gbuf_world"]

                assert main_shard.size == model_shard.size

                # Use DDP's contiguous buffer to temporarily hold params.
                model_param = self.models[model_index]._grad_buffers[dtype].data
                main_param = self.get_main_param(group_index)

                # Copy sub-range within tensor.
                model_view = model_param[model_shard.start:model_shard.end]
                main_view = main_param[main_shard.start:main_shard.end]

                model_view.detach().copy_(main_view)

