# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CuTe DSL SiTU-GLU activation and cuDNN grouped-MLP integration."""

from __future__ import annotations

import inspect
import math
from functools import lru_cache
from typing import Optional

import torch


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


try:
    from transformer_engine.pytorch.ops import ScaledSwiGLU as _ScaledSwiGLU
except ImportError:
    _ScaledSwiGLU = torch.nn.Module


class ScaledSiTUGLU(_ScaledSwiGLU):
    """TE scaled activation shell recognized by its fused grouped-MLP matcher."""

    def __init__(self, beta1: float = 4.0, beta2: float = 25.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.beta1, self.beta2 = _validate_betas(beta1, beta2)

    def _glu_forward(self, input_: torch.Tensor) -> torch.Tensor:
        # TE's legacy _ScaledGLU API removes interleaving before this callback.
        return _situ_glu_forward(input_, self.beta1, self.beta2)

    def _glu_backward(self, grad_output: torch.Tensor, input_: torch.Tensor) -> torch.Tensor:
        # TE's legacy _ScaledGLU API restores interleaving after this callback.
        return _situ_glu_backward(grad_output, input_, self.beta1, self.beta2)

    def _scaled_glu_forward(self, input_: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        """Implement the TE 2.17+ scaled-GLU fallback API."""
        output = _situ_glu_forward(
            input_, self.beta1, self.beta2, int(self.glu_interleave_size or 0)
        )
        return output * scales.unsqueeze(-1)

    def _scaled_glu_backward(
        self,
        grad_output: torch.Tensor,
        input_: torch.Tensor,
        scales: torch.Tensor,
        *,
        compute_scale_grad: bool,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Implement the TE 2.17+ scaled-dGLU fallback API."""
        output = None
        if compute_scale_grad:
            output = _situ_glu_forward(
                input_, self.beta1, self.beta2, int(self.glu_interleave_size or 0)
            )
        grad_input = _situ_glu_backward(
            grad_output * scales.unsqueeze(-1),
            input_,
            self.beta1,
            self.beta2,
            int(self.glu_interleave_size or 0),
        )
        grad_scales = None
        if output is not None:
            grad_scales = torch.linalg.vecdot(output, grad_output)
        return grad_input, grad_scales


def make_situ_glu(beta1: float = 4.0, beta2: float = 25.0) -> torch.nn.Module:
    """Construct native TE SiTU-GLU when available, otherwise the local CuTe fallback."""
    beta1, beta2 = _validate_betas(beta1, beta2)
    try:
        from transformer_engine.pytorch import ops as te_ops
    except ImportError:
        te_ops = None
    native_cls = getattr(te_ops, "SiTUGLU", None)
    if native_cls is not None:
        return native_cls(beta1=beta1, beta2=beta2)
    return CuTeDSLSiTUGLU(beta1=beta1, beta2=beta2)


def make_scaled_situ_glu(
    beta1: float = 4.0, beta2: float = 25.0, *, install_grouped_fallback: bool = False, **kwargs
) -> torch.nn.Module:
    """Construct native TE scaled SiTU-GLU or install and use the local fused fallback."""
    beta1, beta2 = _validate_betas(beta1, beta2)
    try:
        from transformer_engine.pytorch import ops as te_ops
    except ImportError:
        te_ops = None
    native_cls = getattr(te_ops, "ScaledSiTUGLU", None)
    if native_cls is not None:
        return native_cls(beta1=beta1, beta2=beta2, **kwargs)
    if install_grouped_fallback:
        install_grouped_situ_glu_kernels(beta1, beta2)
    return ScaledSiTUGLU(beta1=beta1, beta2=beta2, **kwargs)


_GROUPED_KERNELS_INSTALLED = False
_GROUPED_KERNEL_PARAMETERS: Optional[tuple[float, float]] = None


def _install_operand_major_mode_compat() -> bool:
    """Restore the nvgpu re-export expected by cuDNN Frontend 1.26.0."""
    from cutlass.cute import nvgpu

    if hasattr(nvgpu, "OperandMajorMode"):
        return False
    from cutlass.cute.nvgpu.tcgen05 import OperandMajorMode

    nvgpu.OperandMajorMode = OperandMajorMode
    return True


def _install_nvvm_atomicrmw_compat() -> None:
    """Bridge the explicit-result NVVM binding used by some CuTe DSL wheels."""
    nvvm = cutlass._mlir.dialects.nvvm
    atomicrmw = nvvm.atomicrmw
    parameters = inspect.signature(atomicrmw).parameters
    result = parameters.get("res")
    if result is None or result.default is not inspect.Parameter.empty:
        return
    if getattr(atomicrmw, "_mcore_infers_result_type", False):
        return

    def atomicrmw_with_inferred_result(*args, **kwargs):
        if not args and "res" not in kwargs:
            return atomicrmw(kwargs["a"].type, **kwargs)
        return atomicrmw(*args, **kwargs)

    atomicrmw_with_inferred_result._mcore_infers_result_type = True
    nvvm.atomicrmw = atomicrmw_with_inferred_result


def _install_blockscaled_tiled_mma_compat() -> None:
    """Adapt the CuTe DSL 4.4 helper to the 4.5 call shape used by cuDNN 1.26."""
    from cutlass.utils import blackwell_helpers

    make_tiled_mma = blackwell_helpers.make_blockscaled_trivial_tiled_mma
    parameters = inspect.signature(make_tiled_mma).parameters
    if "b_dtype" in parameters:
        return
    if getattr(make_tiled_mma, "_mcore_accepts_separate_ab_dtypes", False):
        return

    def make_tiled_mma_with_separate_ab_dtypes(*args, loc=None, ip=None, **kwargs):
        if "b_dtype" in kwargs:
            a_dtype = kwargs.pop("a_dtype")
            b_dtype = kwargs.pop("b_dtype")
            if a_dtype is not b_dtype:
                raise TypeError("CuTe DSL 4.4 block-scaled MMA requires matching A/B dtypes.")
            kwargs["ab_dtype"] = a_dtype
        elif len(args) >= 2 and inspect.isclass(args[1]):
            a_dtype, b_dtype = args[:2]
            if a_dtype is not b_dtype:
                raise TypeError("CuTe DSL 4.4 block-scaled MMA requires matching A/B dtypes.")
            args = (a_dtype, *args[2:])
        return make_tiled_mma(*args, loc=loc, ip=ip, **kwargs)

    make_tiled_mma_with_separate_ab_dtypes._mcore_accepts_separate_ab_dtypes = True
    blackwell_helpers.make_blockscaled_trivial_tiled_mma = make_tiled_mma_with_separate_ab_dtypes


def _install_pointer_property_compat() -> None:
    """Expose the pointer accessor expected by cuDNN 1.26 on CuTe DSL 4.4."""
    from cutlass.cute import core

    if hasattr(core._Pointer, "ptr"):
        return
    core._Pointer.ptr = property(lambda self: self)


def install_grouped_situ_glu_kernels(beta1: float, beta2: float) -> None:
    """Replace cuDNN Frontend's SwiGLU kernel classes before their first compilation."""
    global _GROUPED_KERNEL_PARAMETERS, _GROUPED_KERNELS_INSTALLED
    beta1, beta2 = _validate_betas(beta1, beta2)
    if _GROUPED_KERNELS_INSTALLED:
        if _GROUPED_KERNEL_PARAMETERS != (beta1, beta2):
            raise RuntimeError(
                "Grouped SiTU-GLU kernels are already installed with "
                f"beta1={_GROUPED_KERNEL_PARAMETERS[0]}, beta2={_GROUPED_KERNEL_PARAMETERS[1]}."
            )
        return
    if not _CUTE_AVAILABLE:
        raise RuntimeError("Grouped SiTU-GLU requires nvidia-cutlass-dsl.")
    installed_operand_major_mode_compat = _install_operand_major_mode_compat()
    _install_blockscaled_tiled_mma_compat()
    _install_pointer_property_compat()
    _install_nvvm_atomicrmw_compat()

    from cudnn.grouped_gemm.grouped_gemm_dglu import api as dglu_api
    from cudnn.grouped_gemm.grouped_gemm_dglu.moe_blockscaled_grouped_gemm_dglu_dbias import (
        BlockScaledMoEGroupedGemmDgluDbiasKernel,
    )
    from cudnn.grouped_gemm.grouped_gemm_glu import api as glu_api
    from cudnn.grouped_gemm.grouped_gemm_glu.moe_blockscaled_grouped_gemm_glu_bias import (
        BlockScaledMoEGroupedGemmGluBiasKernel,
    )

    try:
        from cudnn.grouped_gemm.grouped_gemm_glu_hadamard import api as glu_hadamard_api

        # pylint: disable-next=line-too-long
        from cudnn.grouped_gemm.grouped_gemm_glu_hadamard.moe_blockscaled_grouped_gemm_glu_hadamard import (
            BlockScaledMoEGroupedGemmGluHadamardKernel,
        )
    except ImportError:
        try:
            from cudnn.gemm.cutedsl.grouped.glu_hadamard import api as glu_hadamard_api

            # pylint: disable-next=line-too-long
            from cudnn.gemm.cutedsl.grouped.glu_hadamard.moe_blockscaled_grouped_gemm_glu_hadamard import (
                BlockScaledMoEGroupedGemmGluHadamardKernel,
            )
        except ImportError:
            glu_hadamard_api = None
            BlockScaledMoEGroupedGemmGluHadamardKernel = None

    required_forward = {"tCompute", "acc_vec_up", "acc_vec_gate", "mProb"}
    required_backward = {
        "acc_vec",
        "ab1_vec_load",
        "ab2_vec_load",
        "mProb",
        "beta_val",
        "square_alpha",
        "dprob_swiglu",
    }
    if set(inspect.signature(BlockScaledMoEGroupedGemmGluBiasKernel.swiglu_act).parameters) != (
        required_forward | {"self"}
    ):
        raise RuntimeError("Unsupported cuDNN Frontend grouped GLU kernel signature.")
    if set(inspect.signature(BlockScaledMoEGroupedGemmDgluDbiasKernel.dswiglu).parameters) != (
        required_backward | {"self"}
    ):
        raise RuntimeError("Unsupported cuDNN Frontend grouped dGLU kernel signature.")
    if BlockScaledMoEGroupedGemmGluHadamardKernel is not None and set(
        inspect.signature(BlockScaledMoEGroupedGemmGluHadamardKernel.swiglu_act).parameters
    ) != (required_forward | {"self"}):
        raise RuntimeError("Unsupported cuDNN Frontend grouped GLU-Hadamard kernel signature.")

    class _SiTUForwardKernel(BlockScaledMoEGroupedGemmGluBiasKernel):
        @cute.jit
        def swiglu_act(
            self, tCompute, acc_vec_up, acc_vec_gate, mProb
        ):  # pylint: disable=missing-function-docstring
            inv_beta1 = cutlass.Float32(1.0 / beta1)
            inv_beta2 = cutlass.Float32(1.0 / beta2)
            beta_product = cutlass.Float32(beta1 * beta2)
            for index in cutlass.range_constexpr(cute.size(tCompute)):
                gate = acc_vec_gate[index]
                up = acc_vec_up[index]
                gate_tanh = cute.math.tanh(gate * inv_beta1, fastmath=True)
                up_tanh = cute.math.tanh(up * inv_beta2, fastmath=True)
                if cutlass.const_expr(beta1 == 4.0):
                    reciprocal = cute.arch.rcp_approx(1.0 + gate_tanh * gate_tanh)
                    sigmoid = 0.5 + gate_tanh * reciprocal
                else:
                    sigmoid = cute.arch.rcp_approx(1.0 + cute.math.exp(-gate, fastmath=True))
                tCompute[index] = beta_product * gate_tanh * sigmoid * up_tanh * mProb

    class _SiTUBackwardKernel(BlockScaledMoEGroupedGemmDgluDbiasKernel):
        @cute.jit
        def dswiglu(
            self,
            acc_vec,
            ab1_vec_load,
            ab2_vec_load,
            mProb,
            beta_val,
            square_alpha,
            dprob_swiglu=None,
        ):  # pylint: disable=missing-function-docstring
            dgate = cute.make_rmem_tensor(acc_vec.shape, cutlass.Float32)
            dup = cute.make_rmem_tensor(acc_vec.shape, cutlass.Float32)
            inv_beta1 = cutlass.Float32(1.0 / beta1)
            inv_beta2 = cutlass.Float32(1.0 / beta2)
            for index in cutlass.range_constexpr(cute.size(acc_vec)):
                grad = acc_vec[index] * square_alpha
                gate = ab1_vec_load[index].to(self.acc_dtype) * beta_val
                up = ab2_vec_load[index].to(self.acc_dtype) * beta_val
                gate_tanh = cute.math.tanh(gate * inv_beta1, fastmath=True)
                up_tanh = cute.math.tanh(up * inv_beta2, fastmath=True)
                if cutlass.const_expr(beta1 == 4.0):
                    reciprocal = cute.arch.rcp_approx(1.0 + gate_tanh * gate_tanh)
                    sigmoid = 0.5 + gate_tanh * reciprocal
                    gate_grad = (1.0 - gate_tanh * gate_tanh) * (
                        0.5 + 2.0 * gate_tanh * reciprocal * reciprocal
                    )
                else:
                    sigmoid = cute.arch.rcp_approx(1.0 + cute.math.exp(-gate, fastmath=True))
                    gate_grad = (1.0 - gate_tanh * gate_tanh) * sigmoid
                    gate_grad += beta1 * gate_tanh * sigmoid * (1.0 - sigmoid)
                gate_value = beta1 * gate_tanh * sigmoid
                up_value = beta2 * up_tanh
                up_grad = 1.0 - up_tanh * up_tanh
                dgate[index] = grad * mProb * up_value * gate_grad
                dup[index] = grad * mProb * gate_value * up_grad
                if cutlass.const_expr(self.generate_dprob):
                    dprob_swiglu[index] = grad * gate_value * up_value
            dprob = None
            if cutlass.const_expr(self.generate_dprob):
                dprob = dprob_swiglu.load()
            return dgate.load(), dup.load(), dprob

    if BlockScaledMoEGroupedGemmGluHadamardKernel is not None:

        class _SiTUForwardHadamardKernel(BlockScaledMoEGroupedGemmGluHadamardKernel):
            @cute.jit
            def swiglu_act(
                self, tCompute, acc_vec_up, acc_vec_gate, mProb
            ):  # pylint: disable=missing-function-docstring
                inv_beta1 = cutlass.Float32(1.0 / beta1)
                inv_beta2 = cutlass.Float32(1.0 / beta2)
                beta_product = cutlass.Float32(beta1 * beta2)
                for index in cutlass.range_constexpr(cute.size(tCompute)):
                    gate = acc_vec_gate[index]
                    up = acc_vec_up[index]
                    gate_tanh = cute.math.tanh(gate * inv_beta1, fastmath=True)
                    up_tanh = cute.math.tanh(up * inv_beta2, fastmath=True)
                    if cutlass.const_expr(beta1 == 4.0):
                        reciprocal = cute.arch.rcp_approx(1.0 + gate_tanh * gate_tanh)
                        sigmoid = 0.5 + gate_tanh * reciprocal
                    else:
                        sigmoid = cute.arch.rcp_approx(1.0 + cute.math.exp(-gate, fastmath=True))
                    tCompute[index] = beta_product * gate_tanh * sigmoid * up_tanh * mProb

    glu_api.BlockScaledMoEGroupedGemmGluBiasKernel = _SiTUForwardKernel
    dglu_api.BlockScaledMoEGroupedGemmDgluDbiasKernel = _SiTUBackwardKernel
    glu_api._cache_of_GroupedGemmGluSm100Objects.clear()
    dglu_api._cache_of_GroupedGemmDgluSm100Objects.clear()
    if glu_hadamard_api is not None:
        glu_hadamard_api.BlockScaledMoEGroupedGemmGluHadamardKernel = _SiTUForwardHadamardKernel
        glu_hadamard_api._cache_of_GroupedGemmGluHadamardSm100Objects.clear()

    # CuTe DSL 4.4.2 needs the OperandMajorMode compatibility export above. TE may
    # have queried support before this hook ran, cached False, and skipped registering
    # its existing grouped-MLP fusion rule. Re-evaluate and register that TE rule now.
    if installed_operand_major_mode_compat:
        from transformer_engine.pytorch.ops.fused import grouped_mlp as te_grouped_mlp

        fused_op = te_grouped_mlp.GroupedMLP_CuTeGEMMGLU
        fused_op.is_supported.cache_clear()
        if fused_op.is_supported():
            te_grouped_mlp.register_forward_backward_fusion(te_grouped_mlp.fuse_ops, prepend=True)

    _GROUPED_KERNEL_PARAMETERS = (beta1, beta2)
    _GROUPED_KERNELS_INSTALLED = True


__all__ = [
    "CuTeDSLSiTUGLU",
    "ScaledSiTUGLU",
    "install_grouped_situ_glu_kernels",
    "make_scaled_situ_glu",
    "make_situ_glu",
    "situ_glu_reference",
]
