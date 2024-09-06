# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import logging
import time
from enum import Enum

import torch

try:
    from transformer_engine.pytorch import make_graphed_callables
    from transformer_engine.pytorch.fp8 import FP8GlobalStateManager

    HAVE_TE_GRAPHS = True
except:
    HAVE_TE_GRAPHS = False


class GraphStatus(Enum):
    """An Enum to track if a cudagraph is ready to perform a forward or backward pass."""

    FWD_READY = 0
    BWD_READY = 1


class GraphStatusFunc(torch.autograd.Function):
    """Inserts a node into the autograd graph that tracks whether an object has an outstanding
    backward pass by toggling the value of GraphStatus. This is mainly used to detect when to create
    multiple graphs per transformer layer for pipeline parallelism.
    We don't use backward module hooks as they change forward output tensors to views, see:
    https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_full_backward_hook
    """

    @staticmethod
    def forward(ctx, runner, obj):
        """Occurs immediately before the graph's forward pass.
        Marks the graph's backward pass as ready."""
        ctx.runner = runner
        runner.status = GraphStatus.BWD_READY
        return obj

    @staticmethod
    def backward(ctx, grad):
        """Occurs immediately after the graph's backward pass.
        Marks the graph's forward pass as ready."""
        assert ctx.runner.status == GraphStatus.BWD_READY
        ctx.runner.status = GraphStatus.FWD_READY
        return None, grad


class TensorDescription:
    """Records the attributes of a tensor. Used to check if a
    tensor argument matches the tensor with which the module
    was graph captured with."""

    def __init__(self, tensor):
        self.shape = tuple(tensor.shape)
        self.dtype = tensor.dtype
        self.device = tensor.device

    def matches_tensor(self, tensor):
        """Check if 'tensor' matches the attributes of this TensorDescription."""

        assert torch.is_tensor(tensor)
        return (
            tensor.shape == self.shape
            and tensor.dtype == self.dtype
            and tensor.device == self.device
        )


class CudaGraphCallable(torch.nn.Module):
    """Wraps a module to be cudagraphable, records the output of the cudagraph.
    Reinserts non-tensor args, kwargs that were previously filtered out by 'get_tensor_args'.
    """

    def __init__(self, module, groundtruth_args, groundtruth_kwargs):
        super().__init__()
        self.add_module('base_module', module)

        # The Pytorch cudagraph API requires only tensor inputs, so we strip
        # non-tensor arguments and reinsert them in forward() using these groundtruth attributes.
        # We will also check future calls to the cudagraph against these to ensure the cudagraph
        # is called with the same inputs as it was captured with.
        self.groundtruth_outputs = []
        self.groundtruth_args = tuple(
            TensorDescription(a) if torch.is_tensor(a) else a for a in groundtruth_args
        )
        self.groundtruth_kwargs = {
            k: TensorDescription(v) if torch.is_tensor(v) else v
            for k, v in groundtruth_kwargs.items()
        }

    def forward(self, *arg_tensors, **kwarg_tensors):
        """Call the forward pass of the cudagraph. Also checks the outputs
        of the cudagraph matches what the graph was traced with."""

        args = list(self.groundtruth_args)
        arg_tensors = list(arg_tensors)
        for idx, groundtruth_arg in enumerate(self.groundtruth_args):
            if isinstance(groundtruth_arg, TensorDescription):
                args[idx] = arg_tensors.pop(0)

        kwargs = dict(self.groundtruth_kwargs)
        for k, v in self.groundtruth_kwargs.items():
            if isinstance(v, TensorDescription):
                kwargs[k] = kwarg_tensors[k]

        # Use forward() instead of __call__ to avoid triggering hooks
        out = self.base_module.forward(*args, **kwargs)
        if torch.is_tensor(out):
            out = tuple(out)

        self.groundtruth_outputs = [TensorDescription(o) if torch.is_tensor(o) else o for o in out]

        out = tuple(o for o in out if torch.is_tensor(o))
        assert (
            len(out) > 0
        ), """A graphed module returned no tensors in training mode, however the graphed module 
            must output at least one tensor, so that a corresponding backward node
            may be registered in the autograd graph."""

        if len(out) == 1:
            return out[0]
        return out


class CudaGraphRunner(torch.nn.Module):
    """Wraps a single cudagraph and its expected arguments. Checks that
    the provided args are the same as what the graph was traced with.
    """

    def __init__(self, graphed_module, wrapped_module):
        super().__init__()

        self.graphed_module = graphed_module
        self.groundtruth_args = wrapped_module.groundtruth_args
        self.groundtruth_kwargs = wrapped_module.groundtruth_kwargs
        self.groundtruth_outputs = wrapped_module.groundtruth_outputs
        self.status = GraphStatus.FWD_READY

    def static_args_match(self, args, kwargs):
        """Check the the passed args, kwargs match with the arg, kwargs
        the graph was created with."""

        def check(val, ref):
            if isinstance(ref, TensorDescription):
                return ref.matches_tensor(val)
            return ref == val

        if len(args) != len(self.groundtruth_args):
            return False
        for idx, groundtruth_arg in enumerate(self.groundtruth_args):
            if not check(args[idx], groundtruth_arg):
                return False

        if kwargs.keys() != self.groundtruth_kwargs.keys():
            return False
        for k, v in self.groundtruth_kwargs.items():
            if not check(kwargs[k], v):
                return False
        return True

    def forward(self, args, kwargs, is_first_microbatch=None):
        """Call the forward pass of the cuda graph."""
        if self.training and torch.is_grad_enabled():
            args = list(args)
            for pos in range(len(args)):
                if torch.is_tensor(args[pos]):
                    args[pos] = GraphStatusFunc.apply(self, args[pos])
            for k, v in kwargs.items():
                if torch.is_tensor(v):
                    kwargs[k] = GraphStatusFunc.apply(self, v)

        ret_tensors = self.graphed_module(is_first_microbatch=is_first_microbatch, *args, **kwargs)
        ret_tensors = [ret_tensors] if torch.is_tensor(ret_tensors) else list(ret_tensors)
        out = tuple(
            ret_tensors.pop(0) if isinstance(o, TensorDescription) else o
            for o in self.groundtruth_outputs
        )

        # Check that the static graph matches what was recorded during graph capture
        assert len(out) == len(self.groundtruth_outputs)
        for idx, o in enumerate(self.groundtruth_outputs):
            if isinstance(o, TensorDescription):
                assert o.matches_tensor(out[idx])
            else:
                assert o == out[idx]

        if len(out) == 1:
            return out[0]
        return out


class CudaGraphManager(torch.nn.Module):
    """Creates and runs cudagraphs for a megatron module."""

    def __init__(self):
        super().__init__()
        self.cudagraph_runners = []
        self.is_first_microbatch = True
        assert HAVE_TE_GRAPHS, "CudaGraphManager currently requires TransformerEngine"

        # Cudagraph stream capture requires no operations on the default stream prior to the
        # capture, so change to a side stream. At graph capture change it back.
        self.stream = torch.cuda.current_stream()
        torch.cuda.set_stream(torch.cuda.Stream())

    def __call__(self, megatron_module, args, kwargs):
        """Calls the forward pass of the cudagraphed module.

        Args:
            megatron_module (torch.nn.module): The megatron module to be graphed and run

            args (tuple):  The positional args to be passed to the module.

            kwargs (dict):  The keyword args to be passed to the module.

        """

        # param.data_ptr() below is used to trigger any hooks that have attached to the parameter.
        # Specifically, this is trying to trigger the param sync hook for the APEX optimizer, which
        # triggers param syncs by hooking into any param references.
        # However cudagraphs disables this, so we workaround by manually referencing params here.
        # For more information see:
        # https://github.com/NVIDIA/apex/blob/7001836/apex/contrib/optimizers/distributed_fused_adam.py#L885C9
        for param in megatron_module.parameters():
            param.data_ptr()

        runner = None
        for _runner in self.cudagraph_runners:
            if _runner.static_args_match(args, kwargs) and _runner.status == GraphStatus.FWD_READY:
                runner = _runner
                break

        if runner is None:
            runner = self.create_cudagraph_module(megatron_module, args, kwargs)
            self.cudagraph_runners.append(runner)
            logging.getLogger(__name__).info(
                f"Creating cudagraph; now have {len(self.cudagraph_runners)}"
            )

        tensor_args, tensor_kwargs = self.get_tensor_args(args, kwargs)
        out = runner(tensor_args, tensor_kwargs, is_first_microbatch=self.is_first_microbatch)
        self.is_first_microbatch = False
        return out

    def get_tensor_args(self, args, kwargs):
        """Filter out non-tensor arguments from args and kwargs.
        Needed since 'make_graphed_callables' expects Torch.tensor arg, kwargs."""
        tensor_kwargs = {}
        for k, v in kwargs.items():
            if torch.is_tensor(v):
                tensor_kwargs[k] = v
        tensor_args = tuple(arg for arg in args if torch.is_tensor(arg))
        return tensor_args, tensor_kwargs

    def create_cudagraph_module(self, megatron_module, args, kwargs):
        """Record the graph capture stream. Runs warmup iterations of
        megatron_module, and creates a autograd function, where the
        forward, backward functions are the cudagraphs of module's forward,
        backward passes. Finally wraps this cudagraph function with a CudaGraphRunner.
        """

        torch.cuda.synchronize()
        torch.cuda.set_stream(self.stream)
        start = time.time()

        wrapped_module = CudaGraphCallable(megatron_module, args, kwargs)
        sample_args, sample_kwargs = self.get_tensor_args(args, kwargs)

        # Cudagraphs require no autograd history recorded on sample inputs
        sample_args_detached = tuple(n.detach() for n in sample_args)
        sample_kwargs_detached = {k: v.detach() for k, v in sample_kwargs.items()}
        sample_args_copy = tuple(torch.clone(n) for n in sample_args_detached)
        sample_kwargs_copy = {k: torch.clone(v) for k, v in sample_kwargs_detached.items()}

        # Zero out input args inplace so cudagraph warmup doesnt affect grads
        for orig, detach in zip(sample_args, sample_args_detached):
            detach.zero_()
            detach.requires_grad = orig.requires_grad
        for k, detach in sample_kwargs_detached.items():
            detach.zero_()
            detach.requires_grad = sample_kwargs[k].requires_grad

        fp8_enabled = megatron_module.config.fp8 is not None
        fp8_recipe = FP8GlobalStateManager.get_fp8_recipe() if fp8_enabled else None
        graphed_module = make_graphed_callables(
            modules=wrapped_module,
            sample_args=sample_args_detached,
            sample_kwargs=sample_kwargs_detached,
            _order=[1, -1],
            allow_unused_input=True,
            fp8_enabled=fp8_enabled,
            fp8_recipe=fp8_recipe,
            fp8_weight_caching=True,
        )

        # Restore zeroed out sample args
        # Detach again since pytorch prohibits inplace ops on leaf nodes
        for orig, copy in zip(sample_args, sample_args_copy):
            orig.detach().copy_(copy)
        for k, orig in sample_kwargs.items():
            orig.detach().copy_(sample_kwargs_copy[k])

        logging.getLogger(__name__).info(f'Time spent in cudagraph capture: {time.time() - start}s')
        return CudaGraphRunner(graphed_module, wrapped_module)
