from deepspeed.runtime.pipe.schedule import PipeSchedule, PipeInstruction, BufferOpInstruction, \
    LoadMicroBatch, RecvActivation, SendActivation, RecvGrad, SendGrad, \
    ForwardPass, BackwardPass, ReduceGrads, ReduceTiedGrads, OptimizerStep
from megatron import get_args

class ZeroBubbleH1Pipeline(PipeSchedule):
    """A schedule for training a batch using hybrid parallelism.

    Pipeline parallelism is extracted through gradient accumulation and thus
    convergence follows that of a data parallel approach with the same batch
    size.
    """

    def steps(self):
        num_warmup_microbatches = self.stages - self.stage_id

        forward = 0
        backward = 0
        weight = 0

        # F section
        for _ in range(num_warmup_microbatches - 1):
            if forward == self.micro_batches:
                continue
            forward_id = self.get_buffer_id(forward)
            forward += 1

            cmds = []
            if not self.is_first_stage:
                cmds.append(RecvActivation(forward_id))
            if self.is_first_stage or self.is_last_stage:
                cmds.append(LoadMicroBatch(forward_id))
            cmds.append(ForwardPass(forward_id))
            if not self.is_last_stage:
                cmds.append(SendActivation(forward_id))
            yield cmds

        # FB section
        for _ in range(self.stage_id):
            if forward == self.micro_batches:
                continue
            forward_id = self.get_buffer_id(forward)
            backward_id = self.get_buffer_id(backward)
            forward += 1
            backward += 1

            cmds = []
            if not self.is_first_stage:
                cmds.append(RecvActivation(forward_id))
            if self.is_first_stage or self.is_last_stage:
                cmds.append(LoadMicroBatch(forward_id))
            cmds.append(ForwardPass(forward_id))
            if not self.is_last_stage:
                cmds.append(RecvGrad(backward_id))
                cmds.append(SendActivation(forward_id))
            cmds.append(BackwardOnlyPass(backward_id))
            if not self.is_first_stage:
                cmds.append(SendGrad(backward_id))
            yield cmds
        
        # FBW section
        while forward < self.micro_batches:
            forward_id = self.get_buffer_id(forward)
            backward_id = self.get_buffer_id(backward)
            forward += 1
            backward += 1
            weight += 1

            cmds = []
            if not self.is_first_stage:
                cmds.append(RecvActivation(forward_id))
            if self.is_first_stage or self.is_last_stage:
                cmds.append(LoadMicroBatch(forward_id))
            cmds.append(ForwardPass(forward_id))
            if not self.is_last_stage:
                cmds.append(RecvGrad(backward_id))
                cmds.append(SendActivation(forward_id))
            if self.is_first_stage:
                cmds.append(BackwardPass(backward_id))
            elif forward == self.micro_batches:
                cmds.append(BackwardOnlyPass(backward_id))
                cmds.append(SendGrad(backward_id))
                cmds.append(WeightPass())
            else:
                if get_args().enable_zbh1_exact_semantics:
                    cmds.append(BackwardOnlyPass(backward_id))
                    cmds.append(SendGrad(backward_id)) 
                    cmds.append(WeightPass())
                else:
                    cmds.append(BackwardPass(backward_id))
                    cmds.append(SendGrad(backward_id))
            yield cmds

        #BW section
        while backward < self.micro_batches:
            backward_id = self.get_buffer_id(backward)
            backward += 1
            weight += 1

            cmds = []
            if not self.is_last_stage:
                cmds.append(RecvGrad(backward_id))
            if self.is_first_stage:
                cmds.append(BackwardPass(backward_id))
            else:
                cmds.append(BackwardOnlyPass(backward_id))
                cmds.append(SendGrad(backward_id))
                cmds.append(WeightPass())
            yield cmds
        
        #W section
        while weight < self.micro_batches:
            weight += 1
            yield [WeightPass()]
        
        yield [ReduceTiedGrads(), ReduceGrads(), OptimizerStep()]

    def get_buffer_id(self, microbatch_id):
        num_warmup_microbatches = self.stages - self.stage_id
        return microbatch_id % num_warmup_microbatches


##Additional Instruction classes
class BackwardOnlyPass(BufferOpInstruction):
    """Compute a backward pass and accumulate gradients.

    Roughly:

    .. code-block:: python

        outputs = buffers['outputs'][buffer_id]
        gradients = buffers['gradients'][buffer_id]
        torch.autograd.backward(tensors=outputs,
                                grad_tensors=gradients, inputs = input_tensor)
    """
    pass

class WeightPass(PipeInstruction):
    """Compute a weight pass and accumulate gradients.

    Roughly:

    .. code-block:: python

        torch.autograd.backward(tensors=outputs,
                                grad_tensors=gradients, inputs = model.parameters())
    """
    pass
