import torch
import torch_npu
import torch.nn.functional as F
from enum import Enum, IntEnum
import torch._dynamo as dynamo


FP32_EXPONENT_BIAS = 127
FP32_MIN_NORMAL = 2 ** (-FP32_EXPONENT_BIAS + 1)

# 替换 pow(2, x) 为更高效的位运算
# 原代码中的 2**shared_exp 可以优化为：
def fast_power_of_2(exp):
    return torch.exp2(exp)  # 或者使用位移操作


def _safe_lshift(x, bits, exp):
    if exp is None:
        return x * (2**bits)
    else:
        return x / (2 ** exp) * (2**bits)


def _safe_rshift(x, bits, exp):
    if exp is None:
        return x / (2**bits)
    else:
        return x / (2**bits) * (2 ** exp)


def _round_mantissa(A, bits, round, clamp=False):
    """
    Rounds mantissa to nearest bits depending on the rounding method 'round'
    Args:
      A     {PyTorch tensor} -- Input tensor
      round {str}            --  Rounding method
                                 "floor" rounds to the floor
                                 "nearest" rounds to ceil or floor, whichever is nearest
    Returns:
      A {PyTorch tensor} -- Tensor with mantissas rounded
    """

    if round == "dither":
        rand_A = torch.rand_like(A, requires_grad=False)
        A = torch.sign(A) * torch.floor(torch.abs(A) + rand_A)
    elif round == "floor":
        A = torch.sign(A) * torch.floor(torch.abs(A))
    elif round == "nearest":
        A = torch.sign(A) * torch.floor(torch.abs(A) + 0.5)
    elif round == "even":
        absA = torch.abs(A)
        # find 0.5, 2.5, 4.5 ...
        maskA = ((absA - 0.5) % 2 == torch.zeros_like(A)).type(A.dtype)
        A = torch.sign(A) * (torch.floor(absA + 0.5) - maskA)
    else:
        raise Exception("Unrecognized round method %s" % (round))

    # Clip values that cannot be expressed by the specified number of bits
    if clamp:
        max_mantissa = 2 ** (bits - 1) - 1
        A = torch.clamp(A, -max_mantissa, max_mantissa)
    return A


def _quantize_elemwise_core(A, bits, exp_bits, max_norm, round='nearest',
                            saturate_normals=False, allow_denorm=True):
    """ Core function used for element-wise quantization
    Arguments:
      A         {PyTorch tensor} -- A tensor to be quantized
      bits      {int}            -- Number of mantissa bits. Includes
                                    sign bit and implicit one for floats
      exp_bits  {int}            -- Number of exponent bits, 0 for ints
      max_norm  {float}          -- Largest representable normal number
      round     {str}            -- Rounding mode: (floor, nearest, even)
      saturate_normals {bool}    -- If True, normal numbers (i.e., not NaN/Inf)
                                    that exceed max norm are clamped.
                                    Must be True for correct MX conversion.
      allow_denorm     {bool}    -- If False, flush denorm numbers in the
                                    elem_format to zero.
    Returns:
      quantized tensor {PyTorch tensor} -- A tensor that has been quantized
    """

    out = A

    private_exp = torch.floor(torch.log2(
        torch.abs(A) + (A == 0).type(A.dtype)))

    # The minimum representable exponent for 8 exp bits is -126
    min_exp = -(2**(exp_bits-1)) + 2
    private_exp = private_exp.clip(min=min_exp)

    # Scale up so appropriate number of bits are in the integer portion of the number
    out = _safe_lshift(out, bits - 2, private_exp)

    out = _round_mantissa(out, bits, round, clamp=False)

    # Undo scaling
    out = _safe_rshift(out, bits - 2, private_exp)

    # Set values > max_norm to Inf if desired, else clamp them
    out = torch.clamp(out, min=-max_norm, max=max_norm)

    # handle Inf/NaN
    out[A == float("Inf")] = float("Inf")
    out[A == -float("Inf")] = -float("Inf")
    out[A == float("NaN")] = float("NaN")

    return out


def _shared_exponents(A, method="max", axes=None, ebits=0):
    """
    Get shared exponents for the passed matrix A.
    Args:
      A      {PyTorch tensor} -- Input tensor
      method {str}            -- Exponent selection method.
                                 "max" uses the max absolute value
                                 "none" uses an exponent for each value (i.e., no sharing)
      axes   {list(int)}      -- List of integers which specifies the axes across which
                                 shared exponents are calculated.
    Returns:
      shared_exp {PyTorch tensor} -- Tensor of shared exponents
    """

    if method == "max":
        if axes is None:
            shared_exp = torch.max(torch.abs(A))
        else:
            shared_exp = A
            for axis in axes:
                shared_exp, _ = torch.max(torch.abs(shared_exp), dim=axis, keepdim=True)
    elif method == "none":
        shared_exp = torch.abs(A)
    else:
        raise Exception("Unrecognized shared exponent selection method %s" % (method))

    # log2(shared_exp) and truncate to integer
    shared_exp = torch.floor(
        torch.log2(
            shared_exp + FP32_MIN_NORMAL * (shared_exp == 0).type(shared_exp.dtype)
        )
    )

    # Restrict to [-emax, emax] range
    if ebits > 0:
        emax = 2**(ebits-1) - 1
        #shared_exp = torch.clamp(shared_exp, -emax, emax)
        # Overflow to Inf
        shared_exp[shared_exp > emax] = float("NaN")
        # Underflows are set to -127 which causes them to be
        # flushed to 0 later
        shared_exp[shared_exp < -emax] = -emax

    return shared_exp


def _reshape_to_blocks(A, axes, block_size):
    if axes is None:
        raise Exception(
            "axes required in order to determine which "
            "dimension toapply block size to"
        )
    if block_size == 0:
        raise Exception("block_size == 0 in _reshape_to_blocks")

    # Fix axes to be positive and sort them
    axes = [(x + len(A.shape) if x < 0 else x) for x in axes]
    assert all(x >= 0 for x in axes)
    axes = sorted(axes)

    # Add extra dimension for tiles
    for i in range(len(axes)):
        axes[i] += i  # Shift axes due to added dimensions
        A = torch.unsqueeze(A, dim=axes[i] + 1)

    # Pad to block_size
    orig_shape = A.size()
    pad = []
    for i in range(len(orig_shape)):
        pad += [0, 0]

    do_padding = False
    for axis in axes:
        pre_pad_size = orig_shape[axis]
        if isinstance(pre_pad_size, torch.Tensor):
            pre_pad_size = int(pre_pad_size.value)
        # Don't pad if the axis is short enough to fit inside one tile
        if pre_pad_size % block_size == 0:
            pad[2 * axis] = 0
        else:
            pad[2 * axis] = block_size - pre_pad_size % block_size
            do_padding = True

    if do_padding:
        pad = list(reversed(pad))
        A = torch.nn.functional.pad(A, pad, mode="constant")

    def _reshape(shape, reshape_block_size):
        for axis in axes:
            # Reshape to tiles if axis length > reshape_block_size
            if shape[axis] >= reshape_block_size:
                assert shape[axis] % reshape_block_size == 0
                shape[axis + 1] = reshape_block_size
                shape[axis] = shape[axis] // reshape_block_size
            # Otherwise preserve length and insert a 1 into the shape
            else:
                shape[axis + 1] = shape[axis]
                shape[axis] = 1
        return shape

    # Reshape to tiles
    padded_shape = A.size()
    reshape = _reshape(list(padded_shape), block_size)

    A = A.view(reshape)
    return A, axes, orig_shape, padded_shape


def _undo_reshape_to_blocks(A, padded_shape, orig_shape, axes):
    # Undo tile reshaping
    A = A.view(padded_shape)
    # Undo padding
    if not list(padded_shape) == list(orig_shape):
        slices = [slice(0, x) for x in orig_shape]
        A = A[slices]
    for axis in reversed(axes):
        # Remove extra dimension
        A = torch.squeeze(A, dim=axis + 1)
    return A


def _quantize_mx(
    A,
    scale_bits,
    elem_format,    # can be None for no quantization
    shared_exp_method="max",
    axes=None,
    block_size=0,
    round="nearest",
    flush_fp32_subnorms=False,
):
    """Function used for MX* quantization
    """
    # Shortcut for no quantization
    if elem_format == None:
        return A

    assert(scale_bits > 0)

    # Make sure axes is a list of non-negative numbers
    axes = [axes] if type(axes) == int else axes
    axes = [x + A.ndim if x < 0 else x for x in axes]

    # ebits, mbits, emax, max_norm, _ = _get_format_params(elem_format)
    ebits, mbits, emax, max_norm = 4, 5, 8, 448.0

    # Perform tiling to the hardware vector size
    if block_size > 0:
        A, axes, orig_shape, padded_shape = _reshape_to_blocks(
            A, axes, block_size
        )

    ####################
    # Quantize
    ####################
    shared_exp_axes = [x + 1 for x in axes] if block_size > 0 else axes

    # Get shared exponents
    shared_exp = _shared_exponents(
        A, method=shared_exp_method, axes=shared_exp_axes, ebits=0,
    )

    # Flush subnormal FP32 inputs to zero
    if flush_fp32_subnorms:
        A = A * (shared_exp > -FP32_EXPONENT_BIAS).type(A.dtype)

    # Offset the max exponent by the largest representable exponent
    # in the element data format
    shared_exp = shared_exp - emax

    torch.npu.synchronize()
    shape = shared_exp.shape
    shared_exp = shared_exp.view(-1)
    scale_emax = 127
    shared_exp[shared_exp > scale_emax] = float("NaN")
    shared_exp[shared_exp < -scale_emax] = -scale_emax
    torch.npu.synchronize()
    shared_exp = shared_exp.view(shape)

    A = A / torch.exp2(shared_exp)  # 替代 A / (2**shared_exp)

    A = _quantize_elemwise_core(
            A, mbits, ebits, max_norm, round=round,
            allow_denorm=True, saturate_normals=True)

    A = A * torch.exp2(shared_exp)  # 替代 A * (2**shared_exp)

    # Undo tile reshaping
    if block_size:
        A = _undo_reshape_to_blocks(A, padded_shape, orig_shape, axes)

    return A


# 编译关键函数
# 预热编译，避免运行时编译延迟
def warmup_compilation():
    print("Warming up compilation...")
    with torch.no_grad():
        # 创建小规模测试数据来预热编译
        dummy_A = torch.randn(64, 64, device='npu')
        dummy_B = torch.randn(64, 64, device='npu')
        
        # 预热 _quantize_mx 编译
        _ = _quantize_mx(
            dummy_A,
            8,
            'fp8_e4m3',
            shared_exp_method="max",
            axes=1,
            block_size=16,
            round="nearest",
            flush_fp32_subnorms=False,
        )
        
        # 预热 _quantize_elemwise_core 编译
        _ = _quantize_elemwise_core(
            dummy_A,
            5, 4, 448.0,
            round='nearest',
            saturate_normals=False,
            allow_denorm=True
        )
        
        torch.npu.synchronize()
    print("Compilation warmup completed")

# 执行预热
warmup_compilation()

# 使用编译后的函数（如果需要的话）
# 注意：如果编译开销太大，可以考虑直接使用原函数
# _quantize_mx_compiled = torch.compile(_quantize_mx)
# _quantize_elemwise_core_compiled = torch.compile(_quantize_elemwise_core)


# Load data
A = torch.load("grad_output.pt", map_location='cpu').npu()
print(f"A_shape:{A.shape},grad_max:{torch.max(A)},grad_min:{torch.min(A)}")
B = torch.load("total_input.pt", map_location='cpu').npu()
print(f"B_shape:{B.shape},input_max:{torch.max(B)},input_min:{torch.min(B)}")

C = torch.matmul(A.t(), B)
print(f"C_shape:{C.shape},output_max:{torch.max(C)},output_min:{torch.min(C)}")

scale_bits = 8
elem_format = 'fp8_e4m3'

# 预热GPU
def warmup_gpu():
    with torch.no_grad():
        dummy = torch.randn(100, 100, device='npu')
        _ = torch.matmul(dummy, dummy)
        torch.npu.synchronize()

warmup_gpu()

# Use PyTorch profiler for performance analysis
import os
trace_dir = "./npu_trace"
os.makedirs(trace_dir, exist_ok=True)
with torch_npu.profiler.profile(
    activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
    schedule=torch_npu.profiler.schedule(wait=2, warmup=1, active=3, repeat=1),
    record_shapes=True,
    with_stack=True,
	on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(trace_dir)
) as prof:
    # 使用torch.no_grad()减少内存分配
    with torch.no_grad():
        # 批量处理量化操作以减少函数调用开销
        A_T = _quantize_mx(
            A.t(),
            scale_bits,
            elem_format,
            shared_exp_method="max",
            axes=1,
            block_size=16,
            round="nearest",
            flush_fp32_subnorms=False,
        )

        B = _quantize_mx(
            B,
            scale_bits,
            elem_format,
            shared_exp_method="max",
            axes=0,
            block_size=16,
            round="nearest",
            flush_fp32_subnorms=False,
        )

        # 使用torch.matmul的优化版本
        C_e4m3 = torch.matmul(A_T, B)
        
        # 确保计算完成
        torch.npu.synchronize()
# Print profiling results
# print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
import subprocess

msprof = "/usr/local/Ascend/ascend-toolkit/latest/tools/profiler/bin/msprof"
if os.path.exists(msprof):
    cmd = [
        msprof,
        "--application=./run_profiling.sh",
        f"--output={trace_dir}",
    ]
    # subprocess.run(cmd, check=True)
    # print(f"\n解析完成，请查看 {trace_dir}/summary.csv")
else:
    print("\n未找到 msprof.py，请确认 CANN 版本 >= 6.0 或手动使用 MindStudio Insight 打开 json")


print(f"C_shape:{C_e4m3.shape},output_max:{torch.max(C_e4m3)},output_min:{torch.min(C_e4m3)}")
print(torch.isnan(C).any())

mse_e4m3 = torch.mean((C - C_e4m3) ** 2)
max_err_e4m3 = torch.max(torch.abs(C - C_e4m3))
print(f"MSE: {mse_e4m3:.20f}")
print(f"Max Error: {max_err_e4m3:.20f}")
print(f"相对误差: {mse_e4m3 / torch.mean(C ** 2):.20f}")
