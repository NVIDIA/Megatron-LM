# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Full iteration CUDA graph for training."""

import logging

import torch

from megatron.core.tensor_parallel.random import get_all_rng_states

logger = logging.getLogger(__name__)


class OptimizerCudaGraphWrapper:
    """Wrapper class to enable FullIterationCUDAgraph."""

    curr_iteration = 0
    cuda_graph = None
    result = None # result of the optimizer.step() function

    def __init__(
        self,
        optimizer_step_func,
        cuda_graph_warmup_steps=1,
    ):
        self.optimizer_step_func = optimizer_step_func
        self.cuda_graph_warmup_steps = cuda_graph_warmup_steps

    def __call__(self, *args, **kwargs):
        assert len(args) == 0, 'optimizer.step() does not accept positional args'
        assert len(kwargs) == 0, 'optimizer.step() does not accept keyword args'

        curr_iteration = self.curr_iter()
        if curr_iteration == self.cuda_graph_warmup_steps:
            logger.info(f'Capture CUDA graph for type(self.optimizer).__name__ optimizer!!!')
            torch.distributed.barrier()
            assert OptimizerCudaGraphWrapper.cuda_graph is None
            OptimizerCudaGraphWrapper.cuda_graph = torch.cuda.CUDAGraph()
            for _, state in get_all_rng_states().items():
                OptimizerCudaGraphWrapper.cuda_graph.register_generator_state(state)
            torch.cuda.synchronize()
            capture_stream = torch.cuda.Stream()
            with torch.cuda.graph(
                OptimizerCudaGraphWrapper.cuda_graph,
                stream=capture_stream,
            ):
                OptimizerCudaGraphWrapper.result = self.optimizer_step_func()
            torch.cuda.synchronize()
            torch.distributed.barrier()
            logger.info(f'Optimizer CUDA graph capture done!!!')
        if OptimizerCudaGraphWrapper.cuda_graph is None:
            OptimizerCudaGraphWrapper.result = self.optimizer_step_func()
        else:
            OptimizerCudaGraphWrapper.cuda_graph.replay()
        OptimizerCudaGraphWrapper.curr_iteration += 1
        return OptimizerCudaGraphWrapper.result

    def curr_iter(self):
        """Return current training iteration."""
        return OptimizerCudaGraphWrapper.curr_iteration

    def next_iter(self):
        """Increment current training iteration."""
        OptimizerCudaGraphWrapper.curr_iteration += 1

    def __del__(self):
        print(f"Destructor called for {type(self.optimizer_step_func).__name__} optimizer!!!")
        if OptimizerCudaGraphWrapper.cuda_graph is not None:
            del OptimizerCudaGraphWrapper.cuda_graph
            OptimizerCudaGraphWrapper.cuda_graph = None
        if OptimizerCudaGraphWrapper.result is not None:
            OptimizerCudaGraphWrapper.result = None
