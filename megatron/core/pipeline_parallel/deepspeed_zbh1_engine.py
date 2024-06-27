from megatron.core.tensor_parallel.weight_grad_store import WeightGradStore

from deepspeed.runtime.pipe.engine import PipelineEngine
from deepspeed.utils.timer import BACKWARD_MICRO_TIMER, \
    BACKWARD_GLOBAL_TIMER, BACKWARD_INNER_MICRO_TIMER, BACKWARD_INNER_GLOBAL_TIMER
from deepspeed.runtime.utils import PartitionedTensor
from deepspeed.accelerator import get_accelerator

import torch
from torch.cuda.amp import custom_bwd
from packaging import version


from megatron.core.parallel_state import (
    get_tensor_model_parallel_world_size,
    get_tensor_model_parallel_group,
    get_global_memory_buffer,
)

def _exec_backward_only_pass(self, buffer_id):
    assert self.optimizer is not None, "must provide optimizer during " \
                                        "init in order to use backward"

    self.mem_status('BEFORE BWD ONLY', reset_max=True)
    from megatron.core.tensor_parallel.layers import LinearWithGradAccumulationAndAsyncCommunication
    WeightGradStore.set_combine_bw(False)
    # The last stage just runs backward on the loss using DeepSpeed's typical
    # mechanisms.
    if self.is_last_stage():
        super(PipelineEngine, self).backward(self.loss)
        WeightGradStore.flush()
        self.mem_status('AFTER BWD ONLY')

        WeightGradStore.set_combine_bw(True)
        return

    outputs = self.pipe_buffers['outputs'][buffer_id]

    if self.wall_clock_breakdown():
        self.timers(BACKWARD_MICRO_TIMER).start()
        self.timers(BACKWARD_GLOBAL_TIMER).start()
        self.timers(BACKWARD_INNER_MICRO_TIMER).start()
        self.timers(BACKWARD_INNER_GLOBAL_TIMER).start()

    # Reconstruct if we previously partitioned the output. We must be
    # careful to also restore the computational graph of the tensors we partitioned.
    if self.is_pipe_partitioned:
        if self.is_grad_partitioned:
            if self.pipe_partition_output_meta_cache is None:
                self.pipe_partition_output_meta_cache = outputs[0].to('cpu')
            part_output = PartitionedTensor.from_meta(meta=self.pipe_partition_output_meta_cache,
                                                        local_part=outputs[1],
                                                        group=self.grid.get_slice_parallel_group())
            self.pipe_buffers['output_tensors'][buffer_id].data = part_output.full()
            outputs = (self.pipe_buffers['output_tensors'][buffer_id], *outputs[2:])
        else:
            # Already restored from partition
            self.pipe_buffers['output_tensors'][buffer_id].data = outputs[0]
            outputs = (self.pipe_buffers['output_tensors'][buffer_id], *outputs[1:])

    grad_tensors = self.grad_layer
    if self.is_grad_partitioned:
        if self.grad_partition_grad_layer_meta_cache is None:
            self.grad_partition_grad_layer_meta_cache = self.grad_layer[0].to('cpu')
        part_grad = PartitionedTensor.from_meta(meta=self.grad_partition_grad_layer_meta_cache,
                                                local_part=self.grad_layer[1],
                                                group=self.grid.get_slice_parallel_group())
        grad_tensors = (part_grad.full(), *grad_tensors[2:])
        part_grad = None

    if self.using_bf16_optimizer and not self.is_last_stage():
        # manually call because we don't call optimizer.backward()
        self.optimizer.clear_lp_grads()

    # This handles either a single tensor or tuple of tensors.
    
    if isinstance(outputs, tuple):
        out_tensors = [t for t in outputs if t.is_floating_point()]
        assert len(out_tensors) == len(grad_tensors)
        torch.autograd.backward(tensors=out_tensors, grad_tensors=grad_tensors)
    else:
        torch.autograd.backward(tensors=(outputs, ), grad_tensors=(grad_tensors, ))
    

    WeightGradStore.flush()

    if self.using_bf16_optimizer and not self.is_last_stage():
        # manually call because we don't call optimizer.backward()
        self.optimizer.update_hp_grads(clear_lp_grads=False)

    # Free up the memory from the output of forward()
    self.pipe_buffers['output_tensors'][buffer_id] = None
    self.pipe_buffers['outputs'][buffer_id] = None
    grad_tensors = None
    
    WeightGradStore.set_combine_bw(True)

    if self.wall_clock_breakdown():
        self.timers(BACKWARD_INNER_MICRO_TIMER).stop()
        self.timers(BACKWARD_INNER_GLOBAL_TIMER).stop()
        self.timers(BACKWARD_MICRO_TIMER).stop()
        self.timers(BACKWARD_GLOBAL_TIMER).stop()

def _exec_weight_pass(self):
    if self.using_bf16_optimizer:
        # manually call because we don't call optimizer.backward()
        self.optimizer.clear_lp_grads()
    WeightGradStore.pop()
    if self.using_bf16_optimizer:
        self.optimizer.update_hp_grads(clear_lp_grads=False)