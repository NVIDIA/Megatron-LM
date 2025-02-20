from typing import Callable, Union, Any, Tuple
import torch
from torch import Tensor


class StreamRelease(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, event, stream, *inputs) -> Union[Tensor, Tuple[Tensor]]:
        ctx.event = event
        ctx.stream = stream
        ctx.event.record(stream)
        return inputs if len(inputs) > 1 else inputs[0]

    @staticmethod
    def backward(ctx: Any, *grad_outputs) -> Union[Tensor, Tuple[Tensor]]:
        event = ctx.event
        stream = ctx.stream
        event.wait(stream)
        return (None, None) + grad_outputs


class StreamAcquire(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, event, stream, *inputs: Tensor) -> Union[Tensor, Tuple[Tensor]]:
        ctx.event = event
        ctx.stream = stream
        ctx.event.wait(stream)
        # multiple in, multiple out
        return inputs if len(inputs) > 1 else inputs[0]

    @staticmethod
    def backward(ctx: Any, *grad_outputs) -> Union[Tensor, Tuple[Tensor]]:
        event = ctx.event
        stream = ctx.stream
        event.record(stream)
        return (None, None) + grad_outputs


class ScheduleNode:

    def __init__(self, forward_func, stream, event, name="schedule_node"):
        self.name = name
        self.forward_func = forward_func
        self.stream = stream
        self.event = event
        self.inputs = None
        self.outputs = None

    def forward(self, *inputs):
        torch.cuda.nvtx.range_push(f"{self.name} forward")
        self.inputs = [e.detach() if e is not None else None for e in inputs]
        for i, input in enumerate(self.inputs):
            if input is not None:
                input.requires_grad = inputs[i].requires_grad

        data = StreamAcquire.apply(self.event, self.stream, *self.inputs)
        # pack args to tuple
        if not isinstance(data, tuple):
            data = (data,)

        with torch.cuda.stream(self.stream):
            data = self.forward_func(*data)

        # pack args to tuple
        if not isinstance(data, tuple):
            data = (data,)

        data = StreamRelease.apply(self.event, self.stream, *data)
        self.output = data
        torch.cuda.nvtx.range_pop()
        return self.output

    def get_output(self):
        return self.output

    def backward(self, *output_grad):

        torch.cuda.nvtx.range_push(f"{self.name} backward")
        # not multiple input
        if len(output_grad) == 1:
            output_grad = output_grad[0]
        with torch.cuda.stream(self.stream):
            torch.autograd.backward(self.output, grad_tensors=output_grad, retain_graph=True)
        torch.cuda.nvtx.range_pop()
        return self.get_grad()

    def get_grad(self):
        grad = [e.grad if e is not None else None for e in self.inputs]
        # multiple in, multiple out
        if len(grad) == 1:
            grad = grad[0]
        return grad


def schedule_1f1b(f_model_chunk, f_input, f_context, b_output, b_grad, b_model_chunk, b_context):
    pass
