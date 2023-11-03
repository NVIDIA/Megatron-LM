import functools
import logging
import queue

import torch.cuda.nvtx
from torch.profiler import profile, record_function

from megatron import get_args, get_timers
from megatron.core import parallel_state
from megatron.core.distributed.finalize_model_grads import _allreduce_embedding_grads
from megatron.core.utils import get_model_config


class WeightGradStore:

    cache = []
    weight_grad_queue = [queue.Queue(), queue.Queue()]
    optimizer = None

    @classmethod
    def set_optimizer(cls, optimizer):
        cls.optimizer = optimizer

    @classmethod
    def put(cls, total_input, grad_output, weight, func):
        # func(total_input, grad_output, weight.main_grad)
        cls.cache.append((total_input, grad_output, weight, func))

    @classmethod
    def flush(cls, chunk=0):
        cls.weight_grad_queue[chunk].put(cls.cache)
        cls.cache = []

    @classmethod
    def pop(cls, chunk=0):
        if cls.weight_grad_queue[chunk].qsize() > 0:
            stored_grads = cls.weight_grad_queue[chunk].get()
            for total_input, grad_output, weight, func in stored_grads:
                func(total_input, grad_output, weight.main_grad)
        else:
            raise Exception("Pop empty queue.")

    @classmethod
    def clear(cls, model, chunk=0):
        weight_grad_tasks = []
        while cls.weight_grad_queue[chunk].qsize() > 0:
            stored_grads = cls.weight_grad_queue[chunk].get()
            if len(weight_grad_tasks) == 0:
                for _ in stored_grads:
                    weight_grad_tasks.append([])
            else:
                assert len(weight_grad_tasks) == len(stored_grads)
            for i, task in enumerate(stored_grads):
                weight_grad_tasks[i].append(task)
        assert cls.optimizer is not None
        args = get_args()
        timers = get_timers()
        weight_params = []
        handles = []
        # if model.has_reset:
        #     handles += model.allreduce_gradients()

        output_layer_weight = None

        if parallel_state.is_pipeline_last_stage():
            assert len(weight_grad_tasks) > 0
            output_layer_grads = weight_grad_tasks[0]
            for j in range(len(output_layer_grads)):
                total_input, grad_output, weight, func = output_layer_grads[j]
                if output_layer_weight is None:
                    output_layer_weight = weight
                assert output_layer_weight is weight
                func(total_input, grad_output, weight.main_grad)
                output_layer_grads[j] = None  # release memory
            weight_grad_tasks = weight_grad_tasks[1:]

        config = get_model_config(model)
        handles += _allreduce_embedding_grads([model], config, async_op=True)
        # if parallel_state.is_pipeline_last_stage() or parallel_state.is_pipeline_first_stage():
        #     from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
        #
        #     from megatron.model import DistributedDataParallel as LocalDDP
        #     from megatron.model import Float16Module
        #     from megatron.utils import unwrap_model
        #
        #     unwrapped_model = unwrap_model(model, (torchDDP, LocalDDP, Float16Module))
        #     if unwrapped_model.share_embeddings_and_output_weights:
        #         weight = unwrapped_model.shared_embedding_or_output_weight()
        #         weight_params.append(weight)
        #     elif parallel_state.is_pipeline_last_stage():
        #         weight_params.append(output_layer_weight)
        #         if model.has_reset:
        #             handles += model.allreduce_weight_gradients([output_layer_weight])
        # if model.has_reset:
        #     handles += cls.optimizer.reduce_model_grads(args, timers)

        for i in range(len(weight_grad_tasks)):
            tasks = weight_grad_tasks[i]
            param = None
            for j in range(len(tasks)):
                total_input, grad_output, weight, func = tasks[j]
                if param is None:
                    param = weight
                assert param is weight
                assert not (weight is output_layer_weight)
                func(total_input, grad_output, weight.main_grad)
                tasks[j] = None  # release memory
            weight_params.append(param)
            # if model.has_reset:
            #     handles += model.allreduce_weight_gradients([param])
            weight_grad_tasks[i] = None  # release memory

        # timers('wait_all_reduce', log_level=1).start(barrier=False)
        # if not model.has_reset:
        #     handles += model.allreduce_gradients()
        #     for handle in handles:
        #         if handle is not None:
        #             handle.wait()
        #     handles = cls.optimizer.reduce_model_grads(args, timers)
        for handle in handles:
            if handle is not None:
                handle.wait()
        # timers('wait_all_reduce').stop()
        # model.reset_buffer(weight_params)
