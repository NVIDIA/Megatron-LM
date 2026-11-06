# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CUDA graph storage and autograd boundary for native mHC mapping computation.

Residual branches, parameter-gradient accumulation, and distributed hooks remain
outside these graphs. Private parameter leaves share the live parameter storage
without inheriting its accumulation hooks.
"""

from contextlib import contextmanager
from typing import TYPE_CHECKING, Iterator, Optional, Tuple
from weakref import ref

import torch
from torch import Tensor

if TYPE_CHECKING:
    from megatron.core.distributed import DistributedDataParallel
    from megatron.core.transformer.hyper_connection import HyperConnectionModule


class _MappingBoundary(torch.autograd.Function):
    """Return independent outputs and apply each parameter gradient exactly once."""

    @staticmethod
    def forward(ctx, graph, x: Tensor, *parameters: Tensor):
        """Replay mapping forward while retaining ownership of its saved activations."""
        ctx.graph = graph
        ctx.save_for_backward(x, *parameters)
        ctx.set_materialize_grads(False)
        graph.owner = ref(ctx)
        graph._wait()
        graph.input.copy_(x)
        graph.forward_graph.replay()
        outputs = tuple(value.detach().clone() for value in graph.outputs)
        graph._record()
        graph.replays["forward"] += 1
        return outputs

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, *grad_outputs: Optional[Tensor]):
        """Replay the actual incoming VJP without capturing parameter hooks."""
        graph = ctx.graph
        x, *parameters = ctx.saved_tensors  # Enforce PyTorch's input/weight version checks.
        if graph.owner is None or graph.owner() is not ctx:
            raise RuntimeError("mHC mapping graph activations were overwritten before backward.")
        if not graph._matches(x, check_autocast=False):
            raise RuntimeError("mHC mapping graph configuration changed before backward.")
        graph._wait()
        if any(grad is None for grad in grad_outputs):
            # An unused output must not backpropagate zeros through its branch:
            # doing so changes signed zeros and can turn 0 * inf into NaN.
            gradients = graph._partial_backward(x, grad_outputs)
            graph.replays["backward_fallback"] += 1
        else:
            for target, grad in zip(graph.grad_outputs, grad_outputs):
                target.copy_(grad)
            graph.backward_graph.replay()
            gradients = tuple(value.detach().clone() for value in graph.gradients)
            graph.replays["backward"] += 1
        graph._record()
        graph.owner = None
        parameter_grads = iter(gradients[1:])
        return (
            None,
            gradients[0],
            *(next(parameter_grads) if p.requires_grad else None for p in parameters),
        )


class _MHCMappingsGraph:
    """Own one native-shape training context and a separate evaluation graph."""

    def __init__(
        self,
        module: "HyperConnectionModule",
        sample: Tensor,
        warmup_steps: int,
        ddp_model: Optional["DistributedDataParallel"] = None,
    ) -> None:
        if module.config.use_fused_mhc:
            raise ValueError("mHC mapping CUDA graphs require use_fused_mhc=False.")
        if module.config.variable_seq_lengths:
            raise ValueError("mHC mapping CUDA graphs require fixed sequence lengths.")
        if module.config.cuda_graph_impl != "none":
            raise ValueError(
                "mHC mapping CUDA graphs cannot be combined with another graph implementation."
            )
        if not sample.is_cuda or not sample.is_contiguous() or sample.numel() == 0:
            raise ValueError("mHC mapping CUDA graphs require a nonempty contiguous CUDA sample.")
        if sample.ndim != 3 or sample.shape[-1] != module.n * module.hidden_size:
            raise ValueError(
                "The mHC mapping sample must have shape [sequence, batch, streams * hidden]."
            )
        if sample.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("mHC mapping CUDA graphs support FP16, BF16, and FP32 inputs.")
        if warmup_steps < 1:
            raise ValueError("mHC mapping CUDA graphs require at least one warmup step.")
        if torch.is_inference_mode_enabled():
            raise ValueError("Prepare mHC mapping CUDA graphs outside inference_mode.")
        if torch.cuda.is_current_stream_capturing():
            raise ValueError("Prepare mHC mapping CUDA graphs outside graph capture.")
        self.saved_tensor_hooks = getattr(
            torch._C._autograd, "_top_saved_tensors_default_hooks", None
        )
        if self.saved_tensor_hooks is None:
            raise ValueError("mHC mapping CUDA graphs require saved-tensor hook introspection.")
        if self.saved_tensor_hooks(False) is not None:
            raise ValueError("Prepare mHC mapping CUDA graphs outside saved-tensor hooks.")
        self.active_torch_ddp = getattr(
            torch.nn.parallel.DistributedDataParallel, "_get_active_ddp_module", None
        )
        if self.active_torch_ddp is None:
            raise ValueError("mHC mapping CUDA graphs require DDP context introspection.")
        self.torch_ddp_seen = False
        self.module = module
        self.original = module._compute_mappings
        self.parameters = tuple(module.parameters())
        if any(parameter.device != sample.device for parameter in self.parameters):
            raise ValueError("The mHC sample and parameters must be on the same CUDA device.")
        if ddp_model is None and any(
            getattr(p, "main_grad", None) is not None for p in self.parameters
        ):
            raise ValueError("Pass ddp_model when preparing mappings wrapped in Megatron DDP.")
        if ddp_model is not None:
            if not hasattr(ddp_model, "param_to_bucket_group") or any(
                p.requires_grad and p not in ddp_model.param_to_bucket_group
                for p in self.parameters
            ):
                raise ValueError("ddp_model must be the owning Megatron DDP wrapper.")
            config = ddp_model.ddp_config
            if config.overlap_param_gather or config.grad_buffer_offload:
                raise ValueError(
                    "mHC mapping graphs do not support overlapped parameter gathering "
                    "or gradient-buffer offload."
                )
            groups = ddp_model.bucket_groups + ddp_model.expert_parallel_bucket_groups
            if config.overlap_grad_reduce and len(groups) > 1:
                raise ValueError(
                    "Overlapped mHC mapping gradients require a single bucket group; "
                    "disable overlapped gradient reduction for multiple groups."
                )
        self.ddp_model = ref(ddp_model) if ddp_model is not None else None
        self.communication_signature = self._communication_signature()
        self.private_parameters = tuple(
            torch.nn.Parameter(parameter.detach(), requires_grad=parameter.requires_grad)
            for parameter in self.parameters
        )
        self.active_parameters = tuple(p for p in self.private_parameters if p.requires_grad)
        self.owner = None
        self.replays = {
            key: 0 for key in ("forward", "backward", "evaluation", "fallback", "backward_fallback")
        }
        self.input_signature = (sample.shape, sample.stride(), sample.dtype, sample.device)
        self.parameter_signature = self._parameter_signature()
        self.math_signature = self._math_signature()
        with torch.cuda.device(sample.device):
            self.input = sample.detach().clone().requires_grad_(True)
            self.event = torch.cuda.Event()
            self.event.record()
            self._capture(warmup_steps)

    def _parameter_signature(self) -> tuple:
        return tuple(
            (id(p), p.data_ptr(), p.shape, p.stride(), p.dtype, p.device, p.requires_grad)
            for p in self.module.parameters()
        )

    def _communication_signature(self) -> tuple:
        buffers = tuple(
            (
                (id(getattr(p, "main_grad", None)), getattr(p, "main_grad", None).data_ptr())
                if getattr(p, "main_grad", None) is not None
                else None
            )
            for p in self.module.parameters()
        )
        if self.ddp_model is None:
            return (None, buffers)
        ddp = self.ddp_model()
        if ddp is None:
            return ("released", buffers)
        config = ddp.ddp_config
        return (
            id(ddp),
            config.overlap_grad_reduce,
            config.overlap_param_gather,
            config.grad_buffer_offload,
            tuple(id(group) for group in ddp.bucket_groups + ddp.expert_parallel_bucket_groups),
            tuple(id(ddp.param_to_bucket_group.get(p)) for p in self.module.parameters()),
            buffers,
        )

    def _math_signature(self) -> tuple:
        module = self.module
        autocast = torch.is_autocast_enabled("cuda")
        return (
            module.n,
            module.hidden_size,
            module.sinkhorn_iterations,
            module.sinkhorn_eps,
            module.compute_h_eps,
            module.norm_eps,
            module.config.mhc_keep_mappings_in_fp32,
            module.config.mhc_norm_eps_inside_sqrt,
            module.config.use_fused_mhc,
            autocast,
            torch.get_autocast_dtype("cuda") if autocast else None,
            torch.get_float32_matmul_precision(),
            torch.are_deterministic_algorithms_enabled(),
            module.config.cuda_graph_impl,
        )

    def _matches(self, x: Tensor, check_autocast: bool = True) -> bool:
        current_math = self._math_signature()
        captured_math = self.math_signature
        if not check_autocast:
            current_math = current_math[:9] + current_math[11:]
            captured_math = captured_math[:9] + captured_math[11:]
        return (
            (x.shape, x.stride(), x.dtype, x.device) == self.input_signature
            and self._parameter_signature() == self.parameter_signature
            and self._communication_signature() == self.communication_signature
            and current_math == captured_math
        )

    def _wait(self) -> None:
        torch.cuda.current_stream(self.input.device).wait_event(self.event)

    def _record(self) -> None:
        self.event.record(torch.cuda.current_stream(self.input.device))

    @contextmanager
    def _aliases(self) -> Iterator[None]:
        replacements = {id(a): b for a, b in zip(self.parameters, self.private_parameters)}
        saved = []
        try:
            for module in self.module.modules():
                for name, parameter in module._parameters.items():
                    replacement = replacements.get(id(parameter))
                    if replacement is not None:
                        saved.append((module, name, parameter))
                        module._parameters[name] = replacement
            yield
        finally:
            for module, name, parameter in saved:
                module._parameters[name] = parameter

    def _capture(self, warmup_steps: int) -> None:
        stream = torch.cuda.Stream(device=self.input.device)
        stream.wait_stream(torch.cuda.current_stream(self.input.device))
        inputs = (self.input, *self.active_parameters)
        with self._aliases(), torch.cuda.stream(stream), torch.enable_grad():
            for _ in range(warmup_steps):
                outputs = self.original(self.input)
                # Backward normally runs outside forward autocast. Warm its
                # compiled specializations in the same context used by capture.
                with torch.autocast("cuda", enabled=False):
                    torch.autograd.grad(
                        outputs, inputs, tuple(torch.zeros_like(y) for y in outputs)
                    )
        stream.synchronize()
        self.forward_graph = torch.cuda.CUDAGraph()
        self.backward_graph = torch.cuda.CUDAGraph()
        with self._aliases(), torch.enable_grad():
            with torch.cuda.graph(self.forward_graph, stream=stream):
                self.outputs = self.original(self.input)
            self.grad_outputs = tuple(torch.zeros_like(y) for y in self.outputs)
            with (
                torch.autocast("cuda", enabled=False),
                torch.cuda.graph(
                    self.backward_graph, stream=stream, pool=self.forward_graph.pool()
                ),
            ):
                self.gradients = torch.autograd.grad(self.outputs, inputs, self.grad_outputs)
        torch.cuda.current_stream(self.input.device).wait_stream(stream)
        with self._aliases(), torch.no_grad(), torch.cuda.stream(stream):
            for _ in range(warmup_steps):
                self.original(self.input)
        stream.synchronize()
        # A separate pool keeps evaluation from overwriting saved training intermediates.
        self.evaluation_graph = torch.cuda.CUDAGraph()
        with (
            self._aliases(),
            torch.no_grad(),
            torch.cuda.graph(self.evaluation_graph, stream=stream),
        ):
            self.evaluation_outputs = self.original(self.input)
        torch.cuda.current_stream(self.input.device).wait_stream(stream)
        self._record()

    def _partial_backward(
        self, x: Tensor, grad_outputs: Tuple[Optional[Tensor], ...]
    ) -> Tuple[Optional[Tensor], ...]:
        with (
            self._aliases(),
            torch.enable_grad(),
            torch.autocast("cuda", enabled=self.math_signature[9], dtype=self.math_signature[10]),
        ):
            private_x = x.detach().requires_grad_(True)
            outputs = self.original(private_x)
            used = [
                (output, grad) for output, grad in zip(outputs, grad_outputs) if grad is not None
            ]
            if not used:
                return (None,) * (1 + len(self.active_parameters))
            values, grads = zip(*used)
        with torch.enable_grad():
            gradients = torch.autograd.grad(
                values, (private_x, *self.active_parameters), grads, allow_unused=True
            )
        return gradients

    def __call__(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        # Returning the mapping VJP together changes PyTorch DDP's parameter-ready
        # order and thus its rebuilt reduction buckets. Keep native execution to
        # preserve exact reduction arithmetic. Latch this choice for reentrant
        # recomputation, which runs outside DDP's forward context.
        self.torch_ddp_seen = self.torch_ddp_seen or self.active_torch_ddp() is not None
        if not self.torch_ddp_seen and not self._matches(x):
            # A new shape can generalize compiled reductions and change their
            # floating-point arithmetic. Do not mix those with captured kernels.
            raise RuntimeError(
                "mHC mapping graph metadata changed; disable or prepare the graphs again."
            )
        if (
            self.torch_ddp_seen
            or (torch.is_grad_enabled() and not x.requires_grad)
            # Non-reentrant checkpoints and activation offloading must observe
            # the same native saved tensors in forward and recomputation.
            or self.saved_tensor_hooks(False) is not None
            or (self.owner is not None and self.owner() is not None)
            or torch.cuda.is_current_stream_capturing()
        ):
            self.replays["fallback"] += 1
            return self.original(x)
        if torch.is_grad_enabled() and (
            x.requires_grad or any(p.requires_grad for p in self.parameters)
        ):
            return _MappingBoundary.apply(self, x, *self.parameters)
        with torch.no_grad():
            self._wait()
            self.input.copy_(x)
            self.evaluation_graph.replay()
            outputs = tuple(value.clone() for value in self.evaluation_outputs)
            self._record()
        self.replays["evaluation"] += 1
        return outputs
