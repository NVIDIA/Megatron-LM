# Copyright (c) 2024 Alibaba PAI and Nvidia Megatron-LM Team.
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
import numpy as np
import torch

from typing import *
from megatron.training.memstats_collector import MemStats

from .. import tensor_parallel
from ..distributed import ParamAndGradBuffer
from ..transformer.module import param_is_not_shared
from .chunk import ChunkManager
from .chunk.manager import get_rank
from .clip_grads import get_grad_norm_fp32
from .distrib_optimizer import DistributedOptimizer
from .grad_scaler import MegatronGradScaler
from .hybrid_adam import HybridAdam
from .optimizer_config import OptimizerConfig
from ..utils import is_float8tensor
try:
    # This will be used when "--fp8-param-gather" is enabled.
    # When BF16/FP16 parameters don't exist, we need to cast the FP32 main parameters to
    # FP8 directly in the optimizer.
    from transformer_engine.pytorch.cpp_extensions import cast_to_fp8
except:
    pass
__all__ = ['OffloadDistributedOptimizer']


class OffloadDistributedOptimizer(DistributedOptimizer):

    def _build_model_and_main_param_groups(self, *args, **kwargs):
        """
        This function overrides DO._build_model_and_main_param_groups
        """
        return None, None, None, None, None

    def _build_model_and_main_param_groups_actual(
        self,
        gbuf_ranges: List[Dict],
        param_gbuf_map: Dict[torch.nn.Parameter, Tuple],
        opt_group_ranges: List,
    ):
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
        shard_fp32_from_float32_groups = []

        # Allocate (or slice) each group's param shard.
        for group_range in opt_group_ranges:

            # Params of this group.
            model_float16_params_this_group = []
            model_fp32_params_this_group = []
            shard_float16_params_this_group = []
            shard_fp32_params_this_group = []
            shard_fp32_from_float16_params_this_group = []
            shard_fp32_from_float32_params_this_group = []
            model_float16_groups.append(model_float16_params_this_group)
            model_fp32_groups.append(model_fp32_params_this_group)

            # Views of each sharded parameters
            shard_float16_groups.append(shard_float16_params_this_group)
            shard_fp32_groups.append(shard_fp32_params_this_group)

            # Hybrid FP32 copies of sharded parameters
            shard_fp32_from_float16_groups.append(shard_fp32_from_float16_params_this_group)
            shard_fp32_from_float32_groups.append(shard_fp32_from_float32_params_this_group)

            for model_param in group_range["params"]:
                assert model_param.requires_grad

                gbuf_index, dtype, bucket_index = param_gbuf_map[model_param]
                gbuf_range = gbuf_ranges[gbuf_index][dtype][bucket_index]
                param_range = gbuf_range["param_map"][model_param]["param"]

                # fp16, bf16 params.
                if model_param.type() in ['torch.cuda.HalfTensor', 'torch.cuda.BFloat16Tensor']:
                    # Clone model -> main.
                    shard_model_param = model_param.detach().view(-1)[
                        param_range.start : param_range.end
                    ]

                    # If we use FP8 params to initialize FP32 main params (compared to using the
                    # bf16/fp16 params to initialize the main params), there will be a loss of
                    # precision at the beginning of training (this problem will not occur if the
                    # training is long enough or if the main params are loaded from a checkpoint).
                    if is_float8tensor(model_param) and hasattr(
                        model_param, 'get_high_precision_init_val'
                    ):
                        shard_main_param = (
                            model_param.get_high_precision_init_val()
                            .view(-1)[param_range.start : param_range.end]
                            .clone()
                            .to(shard_model_param.device)
                            .float()
                        )
                        model_param.clear_high_precision_init_val()
                    else:
                        shard_main_param = shard_model_param.clone().float()
                    self.chunk_manager.register_tensor(
                        shard_main_param, 'shard_fp32_from_float16_params'
                    )

                    tensor_parallel.copy_tensor_model_parallel_attributes(
                        shard_model_param, model_param
                    )
                    tensor_parallel.copy_tensor_model_parallel_attributes(
                        shard_main_param, model_param
                    )
                    if hasattr(model_param, 'shared'):
                        shard_model_param.shared = model_param.shared
                        shard_main_param.shared = model_param.shared

                    # Add to group.
                    model_float16_params_this_group.append(model_param)
                    shard_float16_params_this_group.append(shard_model_param)

                    # NOTE: view of shard params, possible on CPU or CUDA
                    shard_fp32_from_float16_params_this_group.append(shard_main_param)

                # fp32 params.
                elif model_param.type() == 'torch.cuda.FloatTensor':
                    shard_model_param = model_param.view(-1)[param_range.start : param_range.end]

                    shard_main_param = shard_model_param.clone()
                    self.chunk_manager.register_tensor(
                        shard_main_param.clone().float(), 'shard_fp32_from_float16_params'
                    )

                    tensor_parallel.copy_tensor_model_parallel_attributes(
                        shard_model_param, model_param
                    )
                    tensor_parallel.copy_tensor_model_parallel_attributes(
                        shard_main_param, model_param
                    )
                    if hasattr(model_param, 'shared'):
                        shard_model_param.shared = model_param.shared
                        shard_main_param.shared = model_param.shared

                    model_fp32_params_this_group.append(model_param)
                    shard_fp32_params_this_group.append(shard_model_param)
                    shard_fp32_from_float32_params_this_group.append(shard_main_param)

                else:
                    raise TypeError(
                        'Wrapped parameters must be one of '
                        'torch.cuda.FloatTensor,  '
                        'torch.cuda.HalfTensor, or '
                        'torch.cuda.BFloat16Tensor. '
                        'Received {}'.format(model_param.type())
                    )

            # Update optimizer's params. [Hybrid]
            group_range["orig_group"]["params"] = [
                *shard_fp32_from_float32_params_this_group,
                *shard_fp32_from_float16_params_this_group,
            ]

        return (
            model_float16_groups,
            model_fp32_groups,
            shard_float16_groups,
            shard_fp32_groups,
            shard_fp32_from_float16_groups,
            shard_fp32_from_float32_groups,
        )

    def collect_shard_param_numel(
        self,
        gbuf_ranges: List[Dict],
        param_gbuf_map: Dict[torch.nn.Parameter, Tuple],
        opt_group_ranges: List,
    ):

        numels = np.zeros([sum(len(group_range["params"]) for group_range in opt_group_ranges)])
        ptr = 0
        for group_range in opt_group_ranges:
            for model_param in group_range["params"]:
                assert model_param.requires_grad

                gbuf_index, dtype, bucket_index = param_gbuf_map[model_param]
                gbuf_range = gbuf_ranges[gbuf_index][dtype][bucket_index]
                param_range = gbuf_range["param_map"][model_param]["param"]
                numels[ptr] = param_range.end - param_range.start
                ptr += 1

        return numels

    def __init__(
        self,
        optimizer: HybridAdam,
        config: OptimizerConfig,
        grad_scaler: MegatronGradScaler,
        init_state_fn: Optional[Callable],
        per_model_buffers: Dict[int, List[ParamAndGradBuffer]],
        data_parallel_group: torch.distributed.ProcessGroup,
        data_parallel_group_gloo: torch.distributed.ProcessGroup,
        data_parallel_group_idx: int,
    ):

        assert (
            config.optimizer_offload_auto_threshold % (1024**2) == 0
            and config.optimizer_offload_auto_threshold > 0
        ), "auto offload threshold should be divided by 2**20"
        assert 0 <= config.optimizer_offload_fraction <= 1, "Offload fraction should be in [0, 1] !"
        assert config.optimizer_offload_policy in [
            'static',
            'auto',
        ], "Only support static or auto placement policy!"
        self.optimizer_offload_fraction = config.optimizer_offload_fraction
        self.optimizer_offload_auto_threshold: int = config.optimizer_offload_auto_threshold
        self.policy = config.optimizer_offload_policy

        super().__init__(
            optimizer,
            config,
            grad_scaler,
            init_state_fn,
            per_model_buffers,
            data_parallel_group,
            data_parallel_group_gloo,
            data_parallel_group_idx,
        )

        # In bf16 model training
        self.grad_dtype_in_buffer = None
        for _, buffers in per_model_buffers.items():
            for buffer in buffers:
                if self.grad_dtype_in_buffer is not None:
                    assert (
                        buffer.grad_dtype == self.grad_dtype_in_buffer
                    ), "Currently only support consistent grad dtype!"
                self.grad_dtype_in_buffer = buffer.grad_dtype

        self.chunk_manager = ChunkManager(
            chunk_size=(
                config.optimizer_offload_chunk_size
                if config.optimizer_offload_chunk_size > 0
                else ChunkManager.find_best_chunk_size(
                    self.collect_shard_param_numel(
                        self.gbuf_ranges, self.model_param_gbuf_map, self.opt_group_ranges
                    ),
                    512,  # NOTE: search chunk size in [32MB, 544MB]
                )
            ),
            init_device='cpu',
            is_fp32_grad=self.grad_dtype_in_buffer == torch.float32,
        )

        # NOTE: Allocate main param shards, all buffer will be on cpu.
        (
            self.model_float16_groups,
            self.model_fp32_groups,
            self.shard_float16_groups,
            self.shard_fp32_groups,
            self.shard_fp32_from_float16_groups,
            self.shard_fp32_from_float32_groups,
        ) = self._build_model_and_main_param_groups_actual(
            self.gbuf_ranges, self.model_param_gbuf_map, self.opt_group_ranges
        )

        self.chunk_manager.close_all_groups()

        # Update optimizer groups.
        # - Also, leverage state_dict() and load_state_dict() to
        #   recast preexisting per-param state tensors.
        self.optimizer.param_groups = [g["orig_group"] for g in self.opt_group_ranges]
        self.optimizer.load_state_dict(self.optimizer.state_dict())

        # NOTE: alloc grad buffer for each parameter
        self.chunk_manager.create_grads()

        # NOTE: also alloc Adam states for each parameter
        exp_avg = self.chunk_manager.alloc_paired_tensors(torch.float32)
        exp_avg_sq = self.chunk_manager.alloc_paired_tensors(torch.float32)

        for t, chunk_list in self.chunk_manager.paired_chunk_map.items():
            assert len(chunk_list) == 2

        for group in self.optimizer.param_groups:
            for _, p in enumerate(group["params"]):
                state = self.state[p]
                assert len(state) == 0
                state["step"] = 0
                # gradient momentums
                state["exp_avg"] = exp_avg[p]
                # gradient variances
                state["exp_avg_sq"] = exp_avg_sq[p]
                self.optimizer._post_state_init(p)

        if self.policy == 'static':
            # NOTE: select partial chunks to GPU
            total_memory = self.chunk_manager.total_mem['cpu']
            budget = round((1 - self.optimizer_offload_fraction) * total_memory)
            if budget > 0:
                for _, chunks in self.chunk_manager.chunk_groups.items():
                    for chunk in chunks:
                        self.chunk_manager.move_chunk(chunk, torch.cuda.current_device(), True)
                        if self.chunk_manager.total_mem['cuda'] >= budget:
                            break
                    if self.chunk_manager.total_mem['cuda'] >= budget:
                        break
        # Total: (2 + 4 + 4) = 10M or (2 + 4 + 4 + 4) = 14M [if an extra fp32 grad chunk is required]
        print('After initialization, parameter chunks use mem: ', self.chunk_manager.total_mem)

    def zero_grad(self, set_to_none=True):
        """
        Zeroes grads for the model related parameters, i.e., model_float16_groups
        and model_fp32_groups. We additionally zero the remaining groups as a
        memory optimization to reduce fragmentation; in the case of
        set_to_none==True, the space used by this field can be safely deallocated.

        Args:
            set_to_none (bool): if true, set grads to None.
        """
        from .optimizer import (
            _zero_grad_group_helper,
        )
        for groups in (
            self.model_float16_groups,
            self.model_fp32_groups,
            self.shard_float16_groups,  # grad empty/unused here?
            self.shard_fp32_groups,  # throws grad-access warning
            self.shard_fp32_from_float16_groups,
            self.shard_fp32_from_float32_groups,
        ):
            for group in groups:
                _zero_grad_group_helper(group, set_to_none=set_to_none)

        # If overlapping param all-gather with forward compute, launch all-gather
        # for first accessed bucket here before forward compute is initiated.
        # The all-gather for the next bucket will be launched in the forward
        # pre-hook when this all-gather finishes (to ensure that the communication
        # kernels don't head-of-line block the compute kernels since we run with
        # CUDA_DEVICE_MAX_CONNECTIONS=1 to support sequence parallelism).
        if self.overlap_param_gather:
            self._dispatch_gather_model_params(all_gather_handle_index=0)

    def _get_model_and_main_params_data_float32(self):
        """
        Get aligned list of model and main params.
        """
        model_data = []
        main_data = []
        for model_group, main_group in zip(
            self.shard_float16_groups, self.shard_fp32_from_float32_groups
        ):
            for model_param, main_param in zip(model_group, main_group):
                model_data.append(model_param.data)
                main_data.append(main_param.data)
        return model_data, main_data

    def _collect_grads(self):
        shard_main_param_id_to_shard_main_grad_mapping = {}
        shard_main_grads = []

        # Utility method for copying group grads.
        def collect_group_grads(model_groups, shard_main_groups):
            for model_group, shard_main_group in zip(model_groups, shard_main_groups):
                for model_param, shard_main_param in zip(model_group, shard_main_group):

                    param_range_map = self._get_model_param_range_map(model_param)
                    param_range = param_range_map["param"]
                    assert param_range.size == shard_main_param.nelement()

                    model_grad = model_param.main_grad
                    shard_model_grad = model_grad.view(-1)[param_range.start : param_range.end]

                    shard_main_grads.append(shard_model_grad.float())
                    shard_main_param_id_to_shard_main_grad_mapping[id(shard_main_param)] = (
                        shard_main_grads[-1]
                    )

        # Copy model groups to shard groups.
        collect_group_grads(self.model_float16_groups, self.shard_fp32_from_float16_groups)
        collect_group_grads(self.model_fp32_groups, self.shard_fp32_from_float32_groups)
        return shard_main_grads, shard_main_param_id_to_shard_main_grad_mapping

    def _dispatch_grads(self, params, main_param_id_to_main_grad_mapping):
        if params is None:
            params = self.get_parameters()
        for param in params:
            if id(param) in main_param_id_to_main_grad_mapping:
                if param.grad is None:
                    param.grad = main_param_id_to_main_grad_mapping[id(param)].to(
                        param.device, non_blocking=True
                    )
                else:
                    param.grad.data.copy_(main_param_id_to_main_grad_mapping[id(param)])

    def _copy_main_params_to_model_params(self):
        """
        Copy main params to model params.

        Since this step is followed by an all-gather through the DDP's grad
        buffer, this method is responsible for copying the updated params
        from the main shards into the correct position in the grad buffer.
        """

        # Utility method for copying group params.
        def copy_group_params(shard_main_groups, model_groups):
            for shard_main_group, model_group in zip(shard_main_groups, model_groups):
                for shard_main_param, model_param in zip(shard_main_group, model_group):

                    param_range_map = self._get_model_param_range_map(model_param)
                    world_range = param_range_map["gbuf_world_in_bucket"]

                    assert world_range.size == shard_main_param.nelement()

                    gbuf_index, _, bucket_id = self.model_param_gbuf_map[model_param]
                    model_param_buffer = self.buffers[gbuf_index].buckets[bucket_id].param_data

                    shard_model_param = model_param_buffer.view(-1)[
                        world_range.start : world_range.end
                    ]

                    if is_float8tensor(model_param):
                        # 1. When "--fp8-param-gather" is disabled, the main param is first cast to
                        #    BF16/FP16, and then cast to FP8, so the amax_history is calculated
                        #    using BF16/FP16 param.
                        # 2. When "--fp8-param-gather" is enabled, we can cast the FP32 main param
                        #    to FP8 directly, which results in slightly different results with
                        #    higher speed. In theory, this does not affect convergence.
                        # TODO: The following code maintains the logic of the point-1 above. It can
                        # be deleted if it is not necessary.
                        shard_main_param = shard_main_param.to(model_param.dtype)

                        cast_to_fp8(
                            shard_main_param.view(1, -1),
                            model_param._fp8_meta['scaling_fwd'],
                            model_param._fp8_meta_index,
                            model_param._fp8_dtype,
                            out=shard_model_param.view(1, -1),
                        )
                    else:
                        shard_model_param.data.copy_(shard_main_param)

        # Copy shard groups to model groups.
        copy_group_params(self.shard_fp32_from_float16_groups, self.model_float16_groups)
        copy_group_params(self.shard_fp32_from_float32_groups, self.model_fp32_groups)

    def _copy_model_params_to_main_params(self):
        """
        Copy model params to main params.

        During finetuning, this method is used to reload the main params from
        the model params. This copy does not make use of the grad buffer as
        an intermediary.
        """

        # Utility method for copying group params.
        def copy_group_params(model_groups, shard_main_groups):
            for model_group, shard_main_group in zip(model_groups, shard_main_groups):
                for model_param, shard_main_param in zip(model_group, shard_main_group):

                    param_range_map = self._get_model_param_range_map(model_param)
                    param_range = param_range_map["param"]
                    assert param_range.size == shard_main_param.nelement()

                    shard_model_param = model_param.view(-1)[param_range.start : param_range.end]
                    shard_main_param.data.copy_(shard_model_param)

        # Copy model groups to shard groups.
        copy_group_params(self.model_float16_groups, self.shard_fp32_from_float16_groups)
        copy_group_params(self.model_fp32_groups, self.shard_fp32_from_float32_groups)

    @torch.no_grad()
    def step_with_ready_grads(self) -> bool:
        """Step the optimizer with ready gradients, return successful.
        Under the hood, either launch synchronous param all-gathers or get ready to launch
        asynchorous all-gathers that get overlapped with the next forward pass.
        """
        self.update_successful = super().step_with_ready_grads()

        timers = self.config.timers
        if timers is not None:
            timers('params-all-gather', log_level=1).start(barrier=self.config.barrier_with_L1_time)
        # If not overlapping all-gather for parameters, launch synchronous all-gather
        # communication calls here. If overlapping all-gather for parameters, the following
        # call to _gather_all_model_params is a no-op: the first all-gather is launched
        # asynchronously in the next optimizer.zero_grad() call and subsequent all-gathers
        # are launched in the forward pre-hook.
        self._reset_metadata_and_sync_gather_all_model_params(force_sync=False)
        if timers is not None:
            timers('params-all-gather').stop()

        return self.update_successful

    def update_layout(self, mem_stats: MemStats = None, threshold: int = None):
        if mem_stats is None:
            return
        if threshold is None:
            threshold = self.optimizer_offload_auto_threshold
        # NOTE: assume in optimizer.step(), we need less non-model data
        # than forward-backward step, therefore make
        # [chunk mem in CUDA] + threshold <= available space
        model_data = mem_stats._prev_md_cuda
        chunk_mem = self.chunk_manager.total_mem['cuda']
        non_model_data = mem_stats.max_non_model_data('cuda')

        current_usage = torch.cuda.memory_reserved() - model_data - non_model_data
        available_space = torch.cuda.mem_get_info()[0] + current_usage - threshold

        # NOTE: small chunks are preferred to being moved.
        # We find this strategy is more stable than random select,
        if available_space < 0:
            for _, chunk_group in self.chunk_manager.chunk_groups.items():
                for chunk in chunk_group:
                    if chunk.device_type == 'cpu':
                        continue
                    released_mem = self.chunk_manager.calc_size_in_device(chunk, 'cuda')
                    self.chunk_manager.move_chunk(chunk, 'cpu', async_move=False)
                    available_space += released_mem
                    if available_space >= 0:
                        break
                if available_space >= 0:
                    break

        # otherwise try to move chunk to CUDA without violating memory constraints
        chunk_and_its_size = []
        for _, chunk_group in self.chunk_manager.chunk_groups.items():
            for chunk in chunk_group:
                if chunk.device_type == 'cuda':
                    continue
                required_mem = self.chunk_manager.calc_size_in_device(
                    chunk, 'cuda')
                chunk_and_its_size.append(
                    (chunk, required_mem)
                )

        chunk_and_its_size.sort(key=lambda x: x[1])
        for chunk, required_mem in chunk_and_its_size:
            if required_mem < available_space:
                self.chunk_manager.move_chunk(
                    chunk, torch.cuda.current_device()
                )
                available_space -= required_mem
                
    def prepare_grads(self) -> bool:
        timers = self.config.timers
        if timers is not None:
            timers('optimizer-update-layout', log_level=1).start(
                barrier=self.config.barrier_with_L1_time
            )
        if self.policy == 'auto':
            self.update_layout(self._mem_stats)
            self._mem_stats = None

        if timers is not None:
            timers('optimizer-update-layout').stop()      

        (
            self._main_grads, 
            self._main_param_id_to_main_grad_mapping
        ) = self._collect_grads()
        
        # 2. unscale / check inf
        # Reset found inf.
        if self.grad_scaler:
            if timers is not None:
                timers('optimizer-unscale-and-check-inf', log_level=1).start(
                    barrier=self.config.barrier_with_L1_time
                )

            self.found_inf.fill_(0.0)

            # Unscale and set found inf/nan
            torch._amp_foreach_non_finite_check_and_unscale_(
                self._main_grads, self.found_inf, self.grad_scaler.inv_scale
            )

            # Update across all model parallel instances.
            torch.distributed.all_reduce(
                self.found_inf,
                op=torch.distributed.ReduceOp.MAX,
                group=self.get_model_parallel_group(),
            )

            # Check for nan.
            found_inf_flag = self.found_inf.item() > 0
            if timers is not None:
                timers('optimizer-unscale-and-check-inf').stop()

            if found_inf_flag:
                self._main_grads = None
                self._main_param_id_to_main_grad_mapping = None
            return found_inf_flag
        return False
    
    def get_main_grads_for_grad_norm(self):
        main_param_id_to_main_grad_mapping = \
            self._main_param_id_to_main_grad_mapping
        params = self.get_parameters()
        grads_for_norm = []
        for param in params:
            # O(n) to O(n^2)
            if id(param) not in main_param_id_to_main_grad_mapping:
                continue
            grad = main_param_id_to_main_grad_mapping[id(param)]
            is_not_shared = param_is_not_shared(param)
            is_not_tp_duplicate = tensor_parallel.param_is_not_tensor_parallel_duplicate(param)
            if is_not_shared and is_not_tp_duplicate:
                grads_for_norm.append(grad)
        return grads_for_norm

    def clip_grad_norm(self, clip_grad: float) -> float:
        grads_for_norm = self.get_main_grads_for_grad_norm()
        total_norm = get_grad_norm_fp32(
            grads_for_norm, model_parallel_group=self.get_model_parallel_group()
        )
        from .optimizer import (
            multi_tensor_applier,
            multi_tensor_scale_impl,
        )
        # Grads.
        grads = []
        for g in self._main_grads:
            assert g.type() == 'torch.cuda.FloatTensor'
            grads.append(g.detach())

        # Scale.
        clip_coeff = clip_grad / (total_norm + 1.0e-6)
        if clip_coeff < 1.0:
            dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device='cuda')
            multi_tensor_applier(
                multi_tensor_scale_impl, dummy_overflow_buf, [grads, grads], clip_coeff
            )

        return total_norm

    def count_zeros(self) -> float:
        main_param_id_to_main_grad_mapping = \
            self._main_param_id_to_main_grad_mapping
        params = self.get_parameters()
        total_num_zeros = torch.tensor([0.0], dtype=torch.float, device='cuda')
        for param in params:
            # O(n) to O(n^2)
            if id(param) not in main_param_id_to_main_grad_mapping:
                continue
            grad = main_param_id_to_main_grad_mapping[id(param)]
            is_not_shared = param_is_not_shared(param)
            is_not_tp_duplicate = tensor_parallel.param_is_not_tensor_parallel_duplicate(param)
            if is_not_shared and is_not_tp_duplicate:
                grad = grad.detach()
                num_zeros = grad.numel() - torch.count_nonzero(grad)
                total_num_zeros = num_zeros + total_num_zeros

        # Sum across all model-parallel GPUs.
        torch.distributed.all_reduce(
            total_num_zeros, 
            op=torch.distributed.ReduceOp.SUM, 
            group=self.get_model_parallel_group()
        )
        total_num_zeros = total_num_zeros.item()
        return total_num_zeros

    def step_with_ready_grads(self):
        timers = self.config.timers
        if timers is not None:
            timers('optimizer-copy-grad-to-cpu-and-gpu', log_level=1).start(
                barrier=self.config.barrier_with_L1_time
            )
        # 4. move these grads to CPU
        self.chunk_manager.attach_grad()
        params = self.get_parameters()
        self._dispatch_grads(
            params, self._main_param_id_to_main_grad_mapping
        )

        self._main_param_id_to_main_grad_mapping = None
        self._main_grads = None

        if timers is not None:
            timers('optimizer-copy-grad-to-cpu-and-gpu').stop()

        return super().step_with_ready_grads()

    def step(self, mem_stats=None):
        self._mem_stats = mem_stats
        return super().step()