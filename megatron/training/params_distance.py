# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

"""L2 distance of the model parameters from the parameters a run started from."""

import os

import torch

from megatron.core import mpu
from megatron.core.tensor_parallel import param_is_not_tensor_parallel_duplicate
from megatron.training.utils import print_rank_0


class ParamsDistanceFromStart:
    """Tracks ||theta_t - theta_0|| for --log-params-distance-from-start.

    theta_0 is the model's (bf16) parameters when a run starts at iteration 0, e.g. right after
    a finetune loads its pretrained checkpoint. It is kept on the GPU and saved under
    ``save_dir`` so a resumed run reloads it instead of restarting from its resume point. The
    distance is computed from the model parameters rather than fp32 main parameters, so it is
    accurate to about the bf16 rounding of each weight.

    Args:
        model (list): The model chunks
        save_dir (str): The run's --save directory
        iteration (int): The iteration training starts at
    """

    def __init__(self, model, save_dir, iteration):
        self.params = [
            param
            for chunk in model
            for param in chunk.parameters()
            if param_is_not_tensor_parallel_duplicate(param)
        ]
        tp_rank = mpu.get_tensor_model_parallel_rank()
        pp_rank = mpu.get_pipeline_model_parallel_rank()
        path = os.path.join(save_dir, 'params_start', f'mp_rank_{tp_rank:02d}_{pp_rank:03d}.pt')

        self.start = None
        if os.path.exists(path):
            self.start = torch.load(path, map_location='cuda', weights_only=True)
            assert len(self.start) == len(self.params), f'{path} does not match the model'
            print_rank_0(f'> params distance: measuring from {os.path.dirname(path)}')
        elif iteration == 0:
            self.start = [param.detach().clone() for param in self.params]
            if mpu.get_data_parallel_rank() == 0:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                torch.save(self.start, path + '.tmp')
                os.replace(path + '.tmp', path)
            print_rank_0('> params distance: saved the starting parameters')
        else:
            print_rank_0(
                f'> params distance: disabled, resuming at iteration {iteration} without '
                f'saved starting parameters in {os.path.dirname(path)}'
            )

        # ||theta_0||, so the distance can also be reported relative to the starting weights,
        # which makes it comparable between runs whose weights differ in scale.
        self.start_norm = None
        if self.start is not None:
            self.start_norm = self._global_norm(start.float() for start in self.start)

    @staticmethod
    def _global_norm(tensors):
        squared = torch.zeros(1, dtype=torch.float32, device='cuda')
        for tensor in tensors:
            squared += tensor.square().sum()
        # Tensor- and pipeline-parallel ranks hold different parameters; data-parallel ranks
        # hold the same ones, so only the model-parallel group is summed.
        torch.distributed.all_reduce(squared, group=mpu.get_model_parallel_group())
        return squared.sqrt().item()

    def compute(self):
        """Return (||theta_t - theta_0||, that distance / ||theta_0||), or None when there are
        no starting parameters."""
        if self.start is None:
            return None
        distance = self._global_norm(
            param.detach().float() - start.float()
            for param, start in zip(self.params, self.start)
        )
        return distance, distance / self.start_norm
