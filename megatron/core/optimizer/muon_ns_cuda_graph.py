# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# License for these changes: BSD-3-Clause.

"""Opt-in replay of the existing local Muon Newton–Schulz computation."""

import threading
from typing import Callable, Optional

import torch


class MuonNewtonSchulzCudaGraphs:
    """Replay the existing NS kernels, retaining precision and iteration count.

    The cache contains CUDA workspaces only, not parameters or optimizer state.
    The unmodified callable remains the fallback for distributed NS and TP>1.
    """

    def __init__(self, reference: Callable[..., torch.Tensor], max_graphs: int = 32) -> None:
        if max_graphs < 1:
            raise ValueError("max_graphs must be positive")
        self.reference = reference
        self.max_graphs = max_graphs
        self.cache: dict[tuple, tuple[torch.Tensor, torch.Tensor, torch.cuda.CUDAGraph]] = {}
        self.lock = threading.Lock()

    def __call__(
        self,
        grad: torch.Tensor,
        steps: int,
        coefficient_type: str,
        tp_group: Optional[torch.distributed.ProcessGroup],
        partition_dim: Optional[int] = None,
        tp_mode: str = 'duplicated',
        use_syrk: bool = False,
    ) -> torch.Tensor:
        arguments = dict(
            steps=steps,
            coefficient_type=coefficient_type,
            tp_group=tp_group,
            partition_dim=partition_dim,
            tp_mode=tp_mode,
        )
        if use_syrk:
            arguments['use_syrk'] = True
        if (
            not grad.is_cuda
            or torch.compiler.is_compiling()
            or not grad.is_contiguous()
            or grad.ndim not in (2, 3)
            or torch.get_float32_matmul_precision() != 'medium'
            or grad.dtype != torch.float32
            or tp_mode != 'duplicated'
            or (partition_dim is not None and (tp_group is None or tp_group.size() != 1))
            or torch.cuda.is_current_stream_capturing()
        ):
            return self.reference(grad, **arguments)
        # Match singleton gather/cat's contiguous layout. Do not change the
        # distributed mode's partition-dependent transpose behavior.
        original_arguments = arguments.copy()
        arguments['partition_dim'] = None
        key = (
            grad.device,
            tuple(grad.shape),
            steps,
            coefficient_type,
            torch.get_float32_matmul_precision(),
            torch.cuda.current_stream(grad.device).cuda_stream,
            use_syrk,
            torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        )
        # Clone the replay output before returning: callers must own a stable
        # result even after another invocation reuses this graph's workspace.
        with self.lock:
            new_shape = key not in self.cache
            if new_shape:
                if len(self.cache) >= self.max_graphs:
                    return self.reference(grad.contiguous(), **arguments)
                rng_before = torch.cuda.get_rng_state(grad.device).clone()
                current = torch.cuda.current_stream(grad.device)
                side = torch.cuda.Stream(device=grad.device)
                static = torch.empty_like(grad, memory_format=torch.contiguous_format)
                static.copy_(grad)
                side.wait_stream(current)
                with torch.cuda.stream(side):
                    for _ in range(3):
                        self.reference(static, **arguments)
                side.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=side, capture_error_mode='thread_local'):
                    output = self.reference(static, **arguments)
                current.wait_stream(side)
                self.cache[key] = (static, output, graph)
            static, output, graph = self.cache[key]
            static.copy_(grad)
            graph.replay()
            actual = output.clone()
            if new_shape:
                expected = self.reference(grad, **original_arguments)
                if not torch.equal(actual, expected) or not torch.equal(
                    rng_before, torch.cuda.get_rng_state(grad.device)
                ):
                    del self.cache[key]
                    raise RuntimeError(
                        'Muon NS CUDA graph changed the first-use result or CUDA RNG state '
                        f'for shape {tuple(grad.shape)}. Disable --muon-ns-cuda-graphs.'
                    )
            return actual
