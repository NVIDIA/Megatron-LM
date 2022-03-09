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

# >>>
from lutil import pax, tp
DEBUG_ITERATION = 2 # 10
# <<<


class Shard:
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.size = end - start
    def normalize(self, start = 0):
        return Shard(start, start + self.size)
    def __str__(self):
        return "%d,%d [%d]" % (self.start, self.end, self.size)


# class Float16DistributedOptimizer(Float16OptimizerWithFloat16Params):
# class Float16DistributedOptimizer(MegatronOptimizer):
# class Float16DistributedOptimizer(BaseFloat16Optimizer):
# class DistributedOptimizer(MegatronOptimizer):
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
                # param_world_shard = param_local_shard.normalize(param_world_start)
                param_world_shard = param_local_shard.normalize(
                    param_local_start + gbuf_world_shard.start)
                sub_param_start = max(0, gbuf_world_shard.start-param_world_start)
                sub_param_shard = param_local_shard.normalize(sub_param_start)
                param_shard_map[param] = {
                    "gbuf_world" : param_world_shard,
                    "gbuf_local" : param_local_shard,
                    "param" : sub_param_shard,
                }

        # pax(0, {"param_shard_map": [ str((str(p.shape), s)) for p,s in param_shard_map.items() ]})

        return param_shard_map

    @classmethod
    def get_model_gbuf_shard(cls, model, dtype):

        data_parallel_rank = mpu.get_data_parallel_rank()
        data_parallel_world_size = mpu.get_data_parallel_world_size()

        # Grad buffer shard.
        grad_buffer = model._grad_buffers[dtype]
        gbuf_size = grad_buffer.numel
        max_gbuf_shard_size = int(math.ceil(gbuf_size / data_parallel_world_size))

        gbuf_world_all_shards = []
        for r in range(data_parallel_world_size):
            gbuf_world_start = r * max_gbuf_shard_size
            gbuf_world_end = min(gbuf_size, gbuf_world_start+max_gbuf_shard_size)
            gbuf_world_shard = Shard(gbuf_world_start, gbuf_world_end)
            gbuf_world_all_shards.append(gbuf_world_shard)
            # >>>
            # if max_gbuf_shard_size != gbuf_world_shard.size:
            #     raise Exception("%d: smaller, rank %d. [ %d -> %d vs. %d]" % (
            #         data_parallel_rank,
            #         r,
            #         gbuf_size,
            #         max_gbuf_shard_size,
            #         gbuf_world_shard.size,
            #     ))
            # <<<
        gbuf_world_shard = gbuf_world_all_shards[data_parallel_rank]
        gbuf_local_shard = gbuf_world_shard.normalize()

        # Param shards.
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

        # pax(0, {"data": data})

        return data

    @classmethod
    def get_model_gbuf_shard_map(cls, model):
        return {
            dtype : cls.get_model_gbuf_shard(model, dtype)
            for dtype in model._grad_buffers
        }

    @classmethod
    def get_param_gbuf_map(cls, model_gbuf_shards):

        param_gbuf_map = {}
        for model_index, model_gbuf_shard_map in enumerate(model_gbuf_shards):
            for dtype, gbuf_shard_map in model_gbuf_shard_map.items():
                for param, param_shard_map in gbuf_shard_map["param_map"].items():
                    # assert param not in param_size_map
                    # param_size_map[param] = param_shard_map["local"].size
                    param_gbuf_map[param] = (model_index, dtype)
                    # pax(0, {
                    #     "dtype" : dtype,
                    #     "gbuf_shard_map" : gbuf_shard_map,
                    #     "param" : tp(param),
                    #     "param_shard_map" : param_shard_map,
                    # })

        # pax(0, {
        #     "model_gbuf_shards" : model_gbuf_shards,
        #     # "param_size_map" :
        #     # [ (str(p.shape), s) for p, s in param_size_map.items() ],
        #     "param_gbuf_map" : param_gbuf_map,
        # })

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

                    # group_shard["max_size"] = gbuf_shard_map["max_shard_size"]
                    group_shard["size"] += param_size
                    group_shard["param_map"][param] = param_group_shard

                    # pax(0, {"gbuf_shard_map": gbuf_shard_map})
                    # >>>
                    # if torch.distributed.get_rank() == 1:
                    #     print(">>> [%d] ... group %d, size %d, param %s. <<<" % (
                    #         torch.distributed.get_rank(),
                    #         group_index,
                    #         param_size,
                    #         str(tuple(param.shape)),
                    #     ))
                    # <<<

        # Squeeze zero-size group shards.
        for group_index, group_shard in enumerate(group_shards):
            group_shard["orig_group"] = param_groups[group_index]
        group_shards = [ g for g in group_shards if g["size"] > 0 ]

        # [ ... x ... ] Synchronize group sizes across ranks.
        
        # pax(0, {
        #     "param_group_map": [
        #         (g, str(p.shape))
        #         for p, g in param_group_map.items()
        #     ],
        #     "group_shards" : group_shards,
        # })

        return group_shards

    @classmethod
    def allocate_main_param_shards(cls, opt_group_shards):

        # Allocate main param/grad shard.
        # ** torch.nn.Parameter ??
        # ** MemoryBuffer ??
        allocate_shard = lambda shard_size, dtype : torch.empty(
            (shard_size,),
            dtype = dtype,
            device = torch.cuda.current_device(),
            requires_grad = True)
        
        # main_param_shards = []
        for group_index, group_shard in enumerate(opt_group_shards):

            # pax(0, {
            #     "group_shard" : group_shard,
            # })

            group_size = group_shard["size"]
            assert group_size != 0, "temporary check ... remove me."

            # ** todo: for dtype in model_main_dtypes ........ **

            # Allocate shard.
            # if group_size == 0:
            #     main_param = None
            # else:
            main_param = allocate_shard(group_size, torch.float)
            main_param.grad = allocate_shard(group_size, torch.float)
            mpu.set_tensor_model_parallel_attributes(main_param, True, 0, 1)

            # main_param_shards.append(main_param)
            group_shard["orig_group"]["params"] = [ main_param ]

            # # Update optimizer group.
            # self.optimizer.param_groups[group_index]["params"] = [ main_param ]

        # pax(1, {
        #     "opt_group_shards" : opt_group_shards,
        #     "main_param_shards" : main_param_shards,
        # })

        # return main_param_shards

    def __init__(self, optimizer, clip_grad, log_num_zeros_in_grad,
                 params_have_main_grad, use_contiguous_buffers_in_local_ddp,
                 bf16, grad_scaler, models):

        super().__init__(
            optimizer, clip_grad, log_num_zeros_in_grad,
            params_have_main_grad, use_contiguous_buffers_in_local_ddp,
            bf16, grad_scaler, models)

        # >>>
        args = get_args()
        assert args.use_contiguous_buffers_in_local_ddp # already checked in args
        # <<<

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

        # >>>
        # pax(0, {
        #     "model_gbuf_shards" : self.model_gbuf_shards,
        #     "opt_group_shards" : self.opt_group_shards,
        #     "main_param_shards" : self.main_param_shards,
        # })
        # <<<

        # Update optimizer groups.
        # - Also, leverage state_dict() and load_state_dict() to
        #   recast preexisting per-param state tensors.
        self.optimizer.param_groups = \
            [ g["orig_group"] for g in self.opt_group_shards ]
        self.optimizer.load_state_dict(self.optimizer.state_dict())

        # pax(0, {
        #     # "opt_group_shards" : self.opt_group_shards,
        #     # "param_groups" : self.optimizer.param_groups,
        #     "optimizer" : self.optimizer,
        #     "optimizer / state" : self.optimizer.state,
        # })
        # pax(1, {
        #     "optimizer" : self.optimizer,
        #     **{"optimizer / param_groups / %d" % i : g
        #        for i, g in enumerate(self.optimizer.param_groups)},
        #     "optimizer / state" : self.optimizer.state,
        #     "optimizer / state_dict" : self.optimizer.state_dict(),
        # })

        # Initialize main params.
        self._copy_model_params_to_main_params()

    def get_model_parallel_group(self):
        # >>>
        # i.e., no param replication across this group
        # <<<
        return None

    # @staticmethod
    # def has_nan_debug(tensors):
    #     if isinstance(tensors, torch.Tensor):
    #         tensors = [ tensors ]
    #     assert isinstance(tensors, list)
    #     has_nans = [ (not torch.all(torch.isfinite(t)).item()) for t in tensors ]
    #     has_nan = any(has_nans)
    #     return has_nan
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

    def get_main_params(self):
        return [ g["params"][0] for g in self.optimizer.param_groups ]
    def get_main_grads(self):
        return [ p.grad for p in self.get_main_params() ]
    def get_main_param(self, group_index):
        # return self.optimizer.param_groups[group_index]["params"][0]
        return self.get_main_params()[group_index]
    def get_main_grad(self, group_index):
        return self.get_main_param(group_index).grad

    def load_state_dict(self):
        raise Exception("hi.")
    def reload_model_params(self):
        raise Exception("hi.")
    def state_dict(self):
        raise Exception("hi.")

    def zero_grad(self, set_to_none=True):

        model_params = []
        for model in self.models:
            for dtype, param_map in model._grad_buffer_param_index_map.items():
                model_params.extend(param_map.keys())
        # main_params = []
        # for main_group in self.optimizer.param_groups:
        #     main_params.extend(main_group["params"])

        # ** using contiguous buffer; don't set_to_none **
        _zero_grad_group_helper(model_params, set_to_none = False) # set_to_none)
        # _zero_grad_group_helper(params, set_to_none = False)

        # pax(0, {"model_params": model_params})

    # def get_model_grad_buffer_dp_views(self):

    #     # >>>
    #     # ** only contiguous grad buffer supported, for now [ TEMPORARY ] **
    #     args = get_args()
    #     assert args.use_contiguous_buffers_in_local_ddp
    #     # <<<

    #     # Grad buffer views.
    #     gbuf_view_items = []
    #     for model_index, model in enumerate(self.models):
    #         for dtype, gbuf_shard in self.model_gbuf_shards[model_index].items():
    #             world_shards = gbuf_shard["world_all"]
    #             gbuf = model._grad_buffers[dtype].data
    #             gbuf_views = [ gbuf[s.start:s.end] for s in world_shards ]
    #             gbuf_view_items.append((model_index, dtype, gbuf_views))

    #             # pax(0, {
    #             #     "world_shards" : world_shards,
    #             #     "gbuf_views" : gbuf_views,
    #             # })

    #     pax(0, {
    #         "gbuf_view_items" : gbuf_view_items,
    #         **{
    #             "views / %d" % i : item[2]
    #             for i, item in enumerate(gbuf_view_items)
    #         },
    #     })

    #     return gbuf_view_items
    def get_model_grad_buffer_dp_views(self):

        # >>>
        # ** only contiguous grad buffer supported, for now [ TEMPORARY ] **
        args = get_args()
        assert args.use_contiguous_buffers_in_local_ddp
        # <<<

        # data_parallel_rank = mpu.get_data_parallel_rank()
        data_parallel_world_size = mpu.get_data_parallel_world_size()

        # Grad buffer views.
        gbuf_view_items = []
        for model_index, model in enumerate(self.models):
            for dtype, gbuf in model._grad_buffers.items():

                # gbuf_size = gbuf.numel_padded
                assert gbuf.numel_padded % data_parallel_world_size == 0
                shard_size = int(gbuf.numel_padded / data_parallel_world_size)
                # pax(0, {
                #     "numel" : gbuf.numel,
                #     "numel_padded" : gbuf.numel_padded,
                #     "shard_size / f" : gbuf.numel_padded/data_parallel_world_size,
                #     "shard_size / i" : shard_size,
                # })
                gbuf_views = [gbuf.data[(r*shard_size):((r+1)*shard_size)]
                              for r in range(data_parallel_world_size)]
                gbuf_view_items.append((model_index, dtype, gbuf_views))

        # pax(0, {
        #     "gbuf_view_items" : gbuf_view_items,
        #     **{
        #         "views / %d" % i : item[2]
        #         for i, item in enumerate(gbuf_view_items)
        #     },
        # })

        return gbuf_view_items

    def reduce_grads(self, model):

        # >>>
        from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP

        from megatron import get_args
        from megatron import get_timers
        from megatron.model import DistributedDataParallel as LocalDDP
        from megatron.model import Float16Module
        from megatron.utils import unwrap_model

        args = get_args()
        timers = get_timers()
        # <<<

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Sync word embedding params.

        # ... todo ...

        # All-reduce word_embeddings' grad across first and last stages to ensure
        # that word_embeddings parameters stay in sync.
        # This should only run for models that support pipelined model parallelism
        # (BERT and GPT-2).
        timers('backward-embedding-all-reduce').start()
        if mpu.is_rank_in_embedding_group(ignore_virtual=True) and \
                mpu.get_pipeline_model_parallel_world_size() > 1:
            # >>>
            # raise Exception("[fix] ready for weight sync?")
            # <<<
            if mpu.is_pipeline_first_stage(ignore_virtual=True):
                unwrapped_model = model[0]
            elif mpu.is_pipeline_last_stage(ignore_virtual=True):
                unwrapped_model = model[-1]
            else:  # We do not support the interleaved schedule for T5 yet.
                unwrapped_model = model[0]
            unwrapped_model = unwrap_model(
                unwrapped_model, (torchDDP, LocalDDP, Float16Module))

            if unwrapped_model.share_word_embeddings:
                word_embeddings_weight = unwrapped_model.word_embeddings_weight()
                # >>>
                if args.DDP_impl == 'local':
                    grad = word_embeddings_weight.main_grad
                else:
                    raise Exception("only 'main_grad' supported for distrib-opt.")
                    grad = word_embeddings_weight.grad
                torch.distributed.all_reduce(grad, group=mpu.get_embedding_group())
                # +++
                # grad_shard = optimizer.get_grad_shard(word_embeddings)
                # torch.distributed.all_reduce(grad_shard,
                #                              group=mpu.get_embedding_group())
                # <<<

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Sync T5 position embedding params.

        # ... todo ...

        # All-reduce position_embeddings grad across first (encoder) and split (decoder) 
        # stages to ensure that position embeddings parameters stay in sync.
        # This should only run for T5 models with pipeline parallelism
        if mpu.is_rank_in_position_embedding_group() and \
                mpu.get_pipeline_model_parallel_world_size() > 1 and \
                args.pipeline_model_parallel_split_rank is not None:
            # >>>
            raise Exception("[fix] ready for t5 sync?")
            # <<<
            unwrapped_model = model[0]
            unwrapped_model = unwrap_model(
                unwrapped_model, (torchDDP, LocalDDP, Float16Module))
            assert args.DDP_impl == 'local', \
                'T5 model is only supported with local DDP mode'
            # >>>
            grad = unwrapped_model.language_model.embedding.position_embeddings.weight.main_grad
            torch.distributed.all_reduce(grad, group=mpu.get_position_embedding_group())
            # +++
            # grad_shard = optimizer.get_grad_shard(
            #     unwrapped_model.language_model.embedding.position_embeddings.weight)
            # torch.distributed.all_reduce(grad_shard,
            #                              group=mpu.get_position_embedding_group())
            # <<<
        timers('backward-embedding-all-reduce').stop()

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Reduce-scatter.
        # timers('backward-params-reduce-scatter').start()
        timers('backward-params-all-reduce').start()
        data_parallel_rank = mpu.get_data_parallel_rank()
        data_parallel_world_size = mpu.get_data_parallel_world_size()
        data_parallel_group = mpu.get_data_parallel_group()

        gbuf_view_items = self.get_model_grad_buffer_dp_views()

        # pax(0, {"gbuf_views": [g for item in gbuf_view_items for g in item[2]]})
        # pax(0, {"gbufs": [
        #     g.data
        #     for m in self.models
        #     for g in m._grad_buffers.values()
        # ]})

        # >>>
        # buffer_.data /= mpu.get_data_parallel_world_size()
        # torch.distributed.all_reduce(
        #     buffer_.data, group=mpu.get_data_parallel_group())
        # <<<

        # >>>
        # self.debug_main_param(0, "before reduce scatter")
        # self.debug_main_grad(0, "before reduce scatter")
        # <<<

        for model_index, dtype, gbuf_views in gbuf_view_items:
            # coalesced /= mpu.get_data_parallel_world_size()
            gbuf = self.models[model_index]._grad_buffers[dtype].data

            # >>>
            # ~~ distributed.py ~~
            # gbuf /= data_parallel_world_size
            # torch.distributed.all_reduce(gbuf, group=data_parallel_group)
            # pax(0, {
            #     "gbuf" : tp(gbuf),
            # })
            # <<<

            # torch.mul(gbuf.data, 1. / data_parallel_world_size, out = gbuf.data)
            # gbuf_views = [ t / data_parallel_world_size for t in gbuf_views ]
            gbuf /= data_parallel_world_size

            # if 1:
            # try:
            # pax(0, {"gbuf_views": gbuf_views})
            torch.distributed.reduce_scatter(
                gbuf_views[data_parallel_rank],
                gbuf_views,
                group = data_parallel_group,
            )
            # except:
            #     pax(0, {
            #         "data_parallel_rank" : data_parallel_rank,
            #         "gbuf_views" : gbuf_views,
            #     })
            # else:
            #     torch.distributed.all_reduce(
            #         gbuf,
            #         group = data_parallel_group,
            #     )
        # timers('backward-params-reduce-scatter').stop()
        timers('backward-params-all-reduce').stop()
            
        # pax(0, {"gbuf_views": [g for item in gbuf_view_items for g in item[2]]})

    def gather_params(self, ITERATION):

        # >>>
        timers = get_timers()
        # <<<

        timers('backward-params-all-gather').start()

        data_parallel_rank = mpu.get_data_parallel_rank()
        data_parallel_group = mpu.get_data_parallel_group()

        gbuf_view_items = self.get_model_grad_buffer_dp_views()

        # All-gather updated main params.
        for model_index, dtype, gbuf_views in gbuf_view_items:
            torch.distributed.all_gather(
                gbuf_views,
                gbuf_views[data_parallel_rank],
                group = data_parallel_group,
            )

        # Each model param now contains its updated values in its
        # '.main_grad' field.
        # for param in self.param_gbuf_map: # ... incomplete param list.
        for model in self.models:
            for dtype, param_map in model._grad_buffer_param_index_map.items():
                for param in param_map:
                    param.detach().copy_(param.main_grad)

        timers('backward-params-all-gather').stop()

        # pax(0, {"gbuf_view_items": gbuf_view_items})

        # >>>
        # self.debug_main(ITERATION, "after/inside gather_params.", 0)
        # self.debug_model(ITERATION, "after/inside gather_params.", 0)

        # if ITERATION == 2:
        #     pax(1, {
        #         "ITERATION" : ITERATION,
        #         # "gbufs" : [
        #         #     tp(b.data)
        #         #     for m in self.models
        #         #     for b in m._grad_buffers.values()
        #         # ],
        #         "param_gbuf_map" : [ str(tuple(p.shape)) for p in self.param_gbuf_map ],
        #     })
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


    def _copy_model_grads_to_main_grads(self, ITERATION):

        for group_index, group_shard in enumerate(self.opt_group_shards):
            for model_param, main_shard in group_shard["param_map"].items():

                # Model shard.
                model_index, dtype = self.param_gbuf_map[model_param]
                model_shard = self.model_gbuf_shards \
                    [model_index][dtype]["param_map"][model_param]["gbuf_world"]

                assert main_shard.size == model_shard.size

                # pax(0, {
                #     "model_param" : tp(model_param),
                #     "main_shard" : str(main_shard),
                #     "param shard" : self.model_gbuf_shards \
                #     [model_index][dtype]["param_map"][model_param],
                # })

                # Copy from DDP's contiguous buffer to main shard's grad.
                model_grad = self.models[model_index]._grad_buffers[dtype].data
                main_grad = self.get_main_grad(group_index)

                # Copy sub-range within tensor.
                model_view = model_grad[model_shard.start:model_shard.end]
                main_view = main_grad[main_shard.start:main_shard.end]

                main_view.detach().copy_(model_view)

                # pax(0, {
                #     "group_index" : group_index,
                #     "group_shard" : group_shard,
                #     # "param" : tp(param),
                #     "model_index" : model_index,
                #     "dtype" : str(dtype),
                #     "model_grad" : tp(model_grad),
                #     "main_grad" : tp(main_grad),
                #     "model_view" : tp(model_view),
                #     "main_view" : tp(main_view),
                #     "model_shard" : str(model_shard),
                #     "main_shard" : str(main_shard),
                # })

        # >>>
        # if 1 or ITERATION == DEBUG_ITERATION:
        #     pax(0, {
        #         "** branch **" : "** fix. **",
        #         "ITERATION" : ITERATION,
        #         # "model grads" : self.get_world_model_grads(),
        #         "main_grads" : self.get_main_grads(),
        #         "group shards" : [
        #             "group %d; %s" % (grp_idx, main_shard)
        #             for grp_idx, grp_shard in enumerate(self.opt_group_shards)
        #             for model_param, main_shard in grp_shard["param_map"].items()
        #         ],
        #     })
        # <<<


    def _copy_main_params_to_model_params(self, ITERATION):

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

                # Debug.
                # pax(1, {
                #     "group_index" : group_index,
                #     "group_shard" : group_shard,
                #     "model_param" : tp(model_param),
                #     "model_index" : model_index,
                #     "dtype" : str(dtype),
                #     "model_param" : tp(model_param),
                #     "main_param" : tp(main_param),
                #     "model_view" : tp(model_view),
                #     "main_view" : tp(main_view),
                #     "model_shard" : str(model_shard),
                #     "main_shard" : str(main_shard),
                # })

        # >>>
        # if ITERATION == DEBUG_ITERATION:
        #     pax(0, {
        #         "** branch **" : "** fix. **",
        #         "ITERATION" : ITERATION,
        #         "model params" : self.get_world_model_params(),
        #     })
        # <<<

# <<<


