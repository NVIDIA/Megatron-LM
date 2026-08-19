# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""SiTU-GLU activation builders with an MCore-local CuTe DSL fallback."""

from __future__ import annotations

import math
from collections.abc import Sequence
from functools import lru_cache
from typing import Any, Optional

import torch
from transformer_engine.pytorch.cpu_offload import is_cpu_offload_enabled, mark_activation_offload
from transformer_engine.pytorch.ops._common import maybe_dequantize
from transformer_engine.pytorch.ops.op import BasicOperation, OperationContext
from transformer_engine.pytorch.utils import clear_tensor_data


def situ_glu_reference(
    input_: torch.Tensor, beta1: float = 4.0, beta2: float = 25.0
) -> torch.Tensor:
    """Compute SiTU-GLU with the gate in the first half of the last dimension."""
    gate, up = input_.chunk(2, dim=-1)
    return (beta1 * torch.tanh(gate / beta1) * torch.sigmoid(gate)) * (
        beta2 * torch.tanh(up / beta2)
    )


def _validate_betas(beta1: float, beta2: float) -> tuple[float, float]:
    beta1 = float(beta1)
    beta2 = float(beta2)
    if not math.isfinite(beta1) or beta1 <= 0.0:
        raise ValueError(f"SiTU-GLU beta1 must be finite and positive, got {beta1}.")
    if not math.isfinite(beta2) or beta2 <= 0.0:
        raise ValueError(f"SiTU-GLU beta2 must be finite and positive, got {beta2}.")
    return beta1, beta2


try:
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack, make_fake_stream

    _CUTE_AVAILABLE = True
except ImportError:
    cuda = None
    cutlass = None
    cute = None
    from_dlpack = None
    make_fake_stream = None
    _CUTE_AVAILABLE = False


if _CUTE_AVAILABLE:

    @cute.kernel
    def _situ_glu_forward_kernel(
        input_: cute.Tensor,
        output: cute.Tensor,
        rows: cutlass.Int32,
        width: cutlass.Int32,
        beta1: cutlass.Float32,
        beta2: cutlass.Float32,
        inv_beta1: cutlass.Float32,
        inv_beta2: cutlass.Float32,
        beta_product: cutlass.Float32,
        interleave_size: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        index = bidx * 256 + tidx
        elements = rows * width
        if index < elements:
            row = index // width
            col = index - row * width
            gate_col = col
            up_col = col + width
            if interleave_size > 0:
                block = col // interleave_size
                offset = col - block * interleave_size
                gate_col = block * 2 * interleave_size + offset
                up_col = gate_col + interleave_size
            gate = cutlass.Float32(input_[row, gate_col])
            up = cutlass.Float32(input_[row, up_col])
            gate_tanh = cute.math.tanh(gate * inv_beta1, fastmath=True)
            up_tanh = cute.math.tanh(up * inv_beta2, fastmath=True)
            sigmoid = cutlass.Float32(0.0)
            if beta1 == cutlass.Float32(4.0):
                reciprocal = cute.arch.rcp_approx(cutlass.Float32(1.0) + gate_tanh * gate_tanh)
                sigmoid = cutlass.Float32(0.5) + gate_tanh * reciprocal
            else:
                sigmoid = cute.arch.rcp_approx(
                    cutlass.Float32(1.0) + cute.math.exp(-gate, fastmath=True)
                )
            output[row, col] = (beta_product * gate_tanh * sigmoid * up_tanh).to(
                output.element_type
            )

    @cute.jit
    def _situ_glu_forward_launch(
        input_: cute.Tensor,
        output: cute.Tensor,
        rows: cutlass.Int32,
        width: cutlass.Int32,
        beta1: cutlass.Float32,
        beta2: cutlass.Float32,
        inv_beta1: cutlass.Float32,
        inv_beta2: cutlass.Float32,
        beta_product: cutlass.Float32,
        interleave_size: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        _situ_glu_forward_kernel(
            input_,
            output,
            rows,
            width,
            beta1,
            beta2,
            inv_beta1,
            inv_beta2,
            beta_product,
            interleave_size,
        ).launch(grid=(cute.ceil_div(rows * width, 256), 1, 1), block=(256, 1, 1), stream=stream)

    @cute.kernel
    def _situ_glu_backward_kernel(
        grad_output: cute.Tensor,
        input_: cute.Tensor,
        grad_input: cute.Tensor,
        rows: cutlass.Int32,
        width: cutlass.Int32,
        beta1: cutlass.Float32,
        beta2: cutlass.Float32,
        inv_beta1: cutlass.Float32,
        inv_beta2: cutlass.Float32,
        interleave_size: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        index = bidx * 256 + tidx
        elements = rows * width
        if index < elements:
            row = index // width
            col = index - row * width
            grad = cutlass.Float32(grad_output[row, col])
            gate_col = col
            up_col = col + width
            if interleave_size > 0:
                block = col // interleave_size
                offset = col - block * interleave_size
                gate_col = block * 2 * interleave_size + offset
                up_col = gate_col + interleave_size
            gate = cutlass.Float32(input_[row, gate_col])
            up = cutlass.Float32(input_[row, up_col])
            gate_tanh = cute.math.tanh(gate * inv_beta1, fastmath=True)
            up_tanh = cute.math.tanh(up * inv_beta2, fastmath=True)
            sigmoid = cutlass.Float32(0.0)
            gate_grad = cutlass.Float32(0.0)
            if beta1 == cutlass.Float32(4.0):
                reciprocal = cute.arch.rcp_approx(cutlass.Float32(1.0) + gate_tanh * gate_tanh)
                sigmoid = cutlass.Float32(0.5) + gate_tanh * reciprocal
                gate_grad = (cutlass.Float32(1.0) - gate_tanh * gate_tanh) * (
                    cutlass.Float32(0.5)
                    + cutlass.Float32(2.0) * gate_tanh * reciprocal * reciprocal
                )
            else:
                sigmoid = cute.arch.rcp_approx(
                    cutlass.Float32(1.0) + cute.math.exp(-gate, fastmath=True)
                )
                gate_grad = (cutlass.Float32(1.0) - gate_tanh * gate_tanh) * sigmoid
                gate_grad += beta1 * gate_tanh * sigmoid * (cutlass.Float32(1.0) - sigmoid)
            gate_value = beta1 * gate_tanh * sigmoid
            up_value = beta2 * up_tanh
            up_grad = cutlass.Float32(1.0) - up_tanh * up_tanh
            grad_input[row, gate_col] = (grad * up_value * gate_grad).to(grad_input.element_type)
            grad_input[row, up_col] = (grad * gate_value * up_grad).to(grad_input.element_type)

    @cute.jit
    def _situ_glu_backward_launch(
        grad_output: cute.Tensor,
        input_: cute.Tensor,
        grad_input: cute.Tensor,
        rows: cutlass.Int32,
        width: cutlass.Int32,
        beta1: cutlass.Float32,
        beta2: cutlass.Float32,
        inv_beta1: cutlass.Float32,
        inv_beta2: cutlass.Float32,
        interleave_size: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        _situ_glu_backward_kernel(
            grad_output,
            input_,
            grad_input,
            rows,
            width,
            beta1,
            beta2,
            inv_beta1,
            inv_beta2,
            interleave_size,
        ).launch(grid=(cute.ceil_div(rows * width, 256), 1, 1), block=(256, 1, 1), stream=stream)


def _gpu_arch() -> str:
    capability = torch.cuda.get_device_capability()
    arch = {(9, 0): "sm_90a", (10, 0): "sm_100a", (10, 3): "sm_103a"}.get(capability)
    if arch is None:
        raise RuntimeError(
            f"SiTU-GLU CuTe DSL kernels do not support compute capability {capability}."
        )
    return arch


def _cute_tensor(tensor: torch.Tensor):
    tensor = from_dlpack(tensor.detach(), assumed_align=16, enable_tvm_ffi=True)
    return tensor.mark_layout_dynamic(leading_dim=1)


@lru_cache(maxsize=None)
def _compile_forward(dtype: torch.dtype, device_index: int, rows: int, double_width: int):
    with torch.cuda.device(device_index):
        fake_input = torch.empty((rows, double_width), device="cuda", dtype=dtype)
        fake_output = torch.empty((rows, double_width // 2), device="cuda", dtype=dtype)
        return cute.compile(
            _situ_glu_forward_launch,
            _cute_tensor(fake_input),
            _cute_tensor(fake_output),
            cutlass.Int32(1),
            cutlass.Int32(1),
            cutlass.Float32(4.0),
            cutlass.Float32(25.0),
            cutlass.Float32(0.25),
            cutlass.Float32(0.04),
            cutlass.Float32(100.0),
            cutlass.Int32(0),
            make_fake_stream(use_tvm_ffi_env_stream=False),
            options=f"--enable-tvm-ffi --gpu-arch {_gpu_arch()}",
        )


@lru_cache(maxsize=None)
def _compile_backward(dtype: torch.dtype, device_index: int, rows: int, double_width: int):
    with torch.cuda.device(device_index):
        fake_grad = torch.empty((rows, double_width // 2), device="cuda", dtype=dtype)
        fake_input = torch.empty((rows, double_width), device="cuda", dtype=dtype)
        return cute.compile(
            _situ_glu_backward_launch,
            _cute_tensor(fake_grad),
            _cute_tensor(fake_input),
            _cute_tensor(fake_input),
            cutlass.Int32(1),
            cutlass.Int32(1),
            cutlass.Float32(4.0),
            cutlass.Float32(25.0),
            cutlass.Float32(0.25),
            cutlass.Float32(0.04),
            cutlass.Int32(0),
            make_fake_stream(use_tvm_ffi_env_stream=False),
            options=f"--enable-tvm-ffi --gpu-arch {_gpu_arch()}",
        )


def _validate_input(input_: torch.Tensor, interleave_size: int = 0) -> torch.Tensor:
    if not _CUTE_AVAILABLE:
        raise RuntimeError("SiTU-GLU requires the nvidia-cutlass-dsl package.")
    if not input_.is_cuda:
        raise ValueError("SiTU-GLU CuTe DSL kernels require a CUDA tensor.")
    if input_.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("SiTU-GLU CuTe DSL kernels support BF16 and FP16 tensors.")
    if input_.shape[-1] % 2:
        raise ValueError("SiTU-GLU input width must be even.")
    if interleave_size < 0:
        raise ValueError("SiTU-GLU interleave size must be non-negative.")
    if interleave_size and input_.shape[-1] % (2 * interleave_size):
        raise ValueError("SiTU-GLU input width must be divisible by twice the interleave size.")
    return input_.contiguous().view(-1, input_.shape[-1])


def _situ_glu_forward(
    input_: torch.Tensor, beta1: float, beta2: float, interleave_size: int = 0
) -> torch.Tensor:
    input_2d = _validate_input(input_, interleave_size)
    rows, double_width = input_2d.shape
    output = torch.empty((rows, double_width // 2), device=input_.device, dtype=input_.dtype)
    launcher = _compile_forward(input_.dtype, input_.device.index, rows, double_width)
    launcher(
        input_2d,
        output,
        cutlass.Int32(rows),
        cutlass.Int32(double_width // 2),
        cutlass.Float32(beta1),
        cutlass.Float32(beta2),
        cutlass.Float32(1.0 / beta1),
        cutlass.Float32(1.0 / beta2),
        cutlass.Float32(beta1 * beta2),
        cutlass.Int32(interleave_size),
        cuda.CUstream(torch.cuda.current_stream(input_.device).cuda_stream),
    )
    return output.view(*input_.shape[:-1], double_width // 2)


def _situ_glu_backward(
    grad_output: torch.Tensor,
    input_: torch.Tensor,
    beta1: float,
    beta2: float,
    interleave_size: int = 0,
) -> torch.Tensor:
    input_2d = _validate_input(input_, interleave_size)
    grad_output_2d = grad_output.contiguous().view(-1, grad_output.shape[-1])
    grad_input = torch.empty_like(input_2d)
    rows, double_width = input_2d.shape
    launcher = _compile_backward(input_.dtype, input_.device.index, rows, double_width)
    launcher(
        grad_output_2d,
        input_2d,
        grad_input,
        cutlass.Int32(rows),
        cutlass.Int32(double_width // 2),
        cutlass.Float32(beta1),
        cutlass.Float32(beta2),
        cutlass.Float32(1.0 / beta1),
        cutlass.Float32(1.0 / beta2),
        cutlass.Int32(interleave_size),
        cuda.CUstream(torch.cuda.current_stream(input_.device).cuda_stream),
    )
    return grad_input.view_as(input_)


class _SiTUGLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, input_: torch.Tensor, beta1: float, beta2: float, interleave_size: int
    ) -> torch.Tensor:
        """Run SiTU-GLU forward and save its input for the custom backward."""
        ctx.save_for_backward(input_)
        ctx.beta1 = beta1
        ctx.beta2 = beta2
        ctx.interleave_size = interleave_size
        return _situ_glu_forward(input_, beta1, beta2, interleave_size)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Compute the input gradient for the custom SiTU-GLU operation."""
        (input_,) = ctx.saved_tensors
        return (
            _situ_glu_backward(grad_output, input_, ctx.beta1, ctx.beta2, ctx.interleave_size),
            None,
            None,
            None,
        )


class CuTeDSLSiTUGLU(torch.nn.Module):
    """Standalone SiTU-GLU used by shared experts and unfused routed experts."""

    def __init__(self, beta1: float = 4.0, beta2: float = 25.0, interleave_size: int = 0) -> None:
        super().__init__()
        self.beta1, self.beta2 = _validate_betas(beta1, beta2)
        self.interleave_size = int(interleave_size)
        if self.interleave_size < 0:
            raise ValueError("SiTU-GLU interleave size must be non-negative.")

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """Apply standalone SiTU-GLU to the input tensor."""
        return _SiTUGLUFunction.apply(input_, self.beta1, self.beta2, self.interleave_size)


def _get_te_situ_glu_ops() -> Optional[tuple[type[torch.nn.Module], type[torch.nn.Module]]]:
    """Import the complete public Transformer Engine SiTU-GLU interface."""
    try:
        from transformer_engine.pytorch.ops import ScaledSiTUGLU, SiTUGLU
    except ImportError:
        return None
    return SiTUGLU, ScaledSiTUGLU


class CuTeDSLScaledSiTUGLU(BasicOperation):
    """MCore-local scaled SiTU-GLU fallback for TE's operation-fuser interface."""

    num_extra_inputs: int = 1

    def __init__(
        self,
        glu_interleave_size: Optional[int] = None,
        *,
        activation_recompute_in_mlp: bool = False,
        beta1: float = 4.0,
        beta2: float = 25.0,
    ) -> None:
        super().__init__()
        if activation_recompute_in_mlp:
            raise ValueError(
                f"{self.__class__.__name__} does not support activation recomputation "
                "in the fused grouped MLP"
            )
        self.beta1, self.beta2 = _validate_betas(beta1, beta2)
        self.glu_interleave_size = glu_interleave_size
        if self.glu_interleave_size is not None:
            self.glu_interleave_size = int(self.glu_interleave_size)
            if self.glu_interleave_size <= 0:
                raise ValueError("SiTU-GLU interleave size must be positive when set.")

    def op_forward(self, *args, **kwargs) -> None:
        """Reject direct BasicOperation execution with a hidden scale input."""
        raise RuntimeError(
            f"{self.__class__.__name__} expects its row scales as an operation-fuser input."
        )

    def op_backward(self, *args, **kwargs) -> None:
        """Reject direct BasicOperation backward with a hidden scale input."""
        raise RuntimeError(
            f"{self.__class__.__name__} expects its row scales as an operation-fuser input."
        )

    def fuser_forward(
        self,
        basic_op_ctxs: list[OperationContext],
        input_: torch.Tensor,
        *,
        basic_op_extra_inputs: Sequence[Sequence[Optional[torch.Tensor]]],
        prev_op_grad_output_quantizer: Optional[Any],
        next_op_input_quantizer: Optional[Any],
        basic_op_kwargs: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, Sequence[Sequence[Optional[torch.Tensor]]]]:
        """Apply local SiTU-GLU and the routed-token row scales."""
        del prev_op_grad_output_quantizer, next_op_input_quantizer, basic_op_kwargs
        scales = basic_op_extra_inputs[0][0]
        if torch.is_autocast_enabled():
            dtype = torch.get_autocast_dtype("cuda")
        elif isinstance(input_, torch.Tensor):
            dtype = input_.dtype
        else:
            dtype = scales.dtype
        input_ = maybe_dequantize(input_, dtype)
        scales = maybe_dequantize(scales, dtype)
        output = _situ_glu_forward(
            input_, self.beta1, self.beta2, int(self.glu_interleave_size or 0)
        )
        output = output * scales.unsqueeze(-1)

        ctx = basic_op_ctxs[0]
        if ctx.requires_grad:
            if is_cpu_offload_enabled():
                mark_activation_offload(input_)
            ctx.input_requires_grad = True
            ctx.extra_input_requires_grad = scales.requires_grad
            ctx.dtype = dtype
            ctx.save_for_backward(input_, scales)
        return output, [()]

    def fuser_backward(
        self,
        basic_op_ctxs: list[OperationContext],
        grad_output: torch.Tensor,
        *,
        basic_op_grad_extra_outputs: Sequence[Sequence[Optional[torch.Tensor]]],
    ) -> tuple[
        torch.Tensor,
        Sequence[Sequence[Optional[torch.Tensor]]],
        Sequence[Sequence[Optional[torch.Tensor]]],
    ]:
        """Differentiate the local activation and optional row scales."""
        del basic_op_grad_extra_outputs
        ctx = basic_op_ctxs[0]
        input_, scales = ctx.saved_tensors
        input_ = maybe_dequantize(input_, ctx.dtype)
        scales = maybe_dequantize(scales, ctx.dtype)
        grad_output = maybe_dequantize(grad_output, ctx.dtype)
        interleave_size = int(self.glu_interleave_size or 0)
        grad_input = _situ_glu_backward(
            grad_output * scales.unsqueeze(-1), input_, self.beta1, self.beta2, interleave_size
        )
        grad_scales = None
        if ctx.extra_input_requires_grad:
            output = _situ_glu_forward(input_, self.beta1, self.beta2, interleave_size)
            grad_scales = torch.linalg.vecdot(output, grad_output)
        clear_tensor_data(ctx.saved_tensors[0])
        return grad_input, [()], [(grad_scales,)]


def make_situ_glu(
    beta1: float = 4.0,
    beta2: float = 25.0,
    *,
    cache_quantized_input: bool = False,
    glu_interleave_size: Optional[int] = None,
) -> torch.nn.Module:
    """Construct native TE SiTU-GLU or the standalone MCore CuTe fallback."""
    beta1, beta2 = _validate_betas(beta1, beta2)
    te_ops = _get_te_situ_glu_ops()
    if te_ops is not None:
        situ_glu, _ = te_ops
        return situ_glu(
            beta1=beta1,
            beta2=beta2,
            cache_quantized_input=cache_quantized_input,
            glu_interleave_size=glu_interleave_size,
        )
    if cache_quantized_input:
        raise RuntimeError(
            "activation_func_fp8_input_store for SiTU-GLU requires "
            "https://github.com/NVIDIA/TransformerEngine/pull/3402."
        )
    return CuTeDSLSiTUGLU(beta1=beta1, beta2=beta2, interleave_size=int(glu_interleave_size or 0))


def make_scaled_situ_glu(
    beta1: float = 4.0,
    beta2: float = 25.0,
    *,
    glu_interleave_size: Optional[int] = None,
    activation_recompute_in_mlp: bool = False,
) -> torch.nn.Module:
    """Construct native TE scaled SiTU-GLU or the MCore CuTe fuser fallback."""
    beta1, beta2 = _validate_betas(beta1, beta2)
    te_ops = _get_te_situ_glu_ops()
    if te_ops is not None:
        _, scaled_situ_glu = te_ops
        return scaled_situ_glu(
            glu_interleave_size=glu_interleave_size,
            activation_recompute_in_mlp=activation_recompute_in_mlp,
            beta1=beta1,
            beta2=beta2,
        )
    return CuTeDSLScaledSiTUGLU(
        glu_interleave_size=glu_interleave_size,
        activation_recompute_in_mlp=activation_recompute_in_mlp,
        beta1=beta1,
        beta2=beta2,
    )


__all__ = [
    "CuTeDSLSiTUGLU",
    "CuTeDSLScaledSiTUGLU",
    "make_scaled_situ_glu",
    "make_situ_glu",
    "situ_glu_reference",
]
