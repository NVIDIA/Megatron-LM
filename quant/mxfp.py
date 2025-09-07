import torch
# import torch_npu
from enum import Enum, IntEnum


FP32_EXPONENT_BIAS = 127
FP32_MIN_NORMAL = 2 ** (-FP32_EXPONENT_BIAS + 1)

# Enum for scalar data formats
class ElemFormat(Enum):
    int8 = 1
    int4 = 2
    int2 = 3
    fp8_e5m2 = 4
    fp8_e4m3 = 5
    fp6_e3m2 = 6
    fp6_e2m3 = 7
    fp4 = 8
    fp4_e2m1 = 8
    float16 = 9
    fp16 = 9
    bfloat16 = 10
    bf16 = 10

    @staticmethod
    def from_str(s):
        assert(s != None), "String elem_format == None"
        s = s.lower()
        if hasattr(ElemFormat, s):
            return getattr(ElemFormat, s)
        else:
            raise Exception("Undefined elem format", s)


def _get_min_norm(ebits):
    """ Valid for all float formats """
    emin = 2 - (2 ** (ebits - 1))
    return 0 if ebits == 0 else 2 ** emin


def _get_max_norm(ebits, mbits):
    """ Valid only for floats that define NaN """
    assert(ebits >= 5), "invalid for floats that don't define NaN"
    emax = 0 if ebits==0 else 2**(ebits - 1) - 1
    return 2**emax * float(2**(mbits-1) - 1) / 2**(mbits-2)


_FORMAT_CACHE = {}
def _get_format_params(fmt):
    """ Allowed formats:
        - intX:         2 <= X <= 32, assume sign-magnitude, 1.xxx representation
        - floatX/fpX:   16 <= X <= 28, assume top exp is used for NaN/Inf
        - bfloatX/bfX:  9 <= X <= 32
        - fp4,                  no NaN/Inf
        - fp6_e3m2/e2m3,        no NaN/Inf
        - fp8_e4m3/e5m2,        e5m2 normal NaN/Inf, e4m3 special behavior

        Returns:
          ebits: exponent bits
          mbits: mantissa bits: includes sign and implicit bits
          emax: max normal exponent
          max_norm: max normal number
          min_norm: min normal number
    """
    if type(fmt) is str:
        fmt = ElemFormat.from_str(fmt)

    if fmt in _FORMAT_CACHE:
        return _FORMAT_CACHE[fmt]

    if fmt == ElemFormat.int8:
        ebits, mbits = 0, 8
        emax = 0
    elif fmt == ElemFormat.int4:
        ebits, mbits = 0, 4
        emax = 0
    elif fmt == ElemFormat.int2:
        ebits, mbits = 0, 2
        emax = 0
    elif fmt == ElemFormat.fp8_e5m2:
        ebits, mbits = 5, 4
        emax = 2**(ebits - 1) - 1
    elif fmt == ElemFormat.fp8_e4m3:
        ebits, mbits = 4, 5
        emax = 2**(ebits - 1)
    elif fmt == ElemFormat.fp6_e3m2:
        ebits, mbits = 3, 4
        emax = 2**(ebits - 1)
    elif fmt == ElemFormat.fp6_e2m3:
        ebits, mbits = 2, 5
        emax = 2**(ebits - 1)
    elif fmt == ElemFormat.fp4:
        ebits, mbits = 2, 3
        emax = 2**(ebits - 1)
    elif fmt == ElemFormat.float16:
        ebits, mbits = 5, 12
        emax = 2**(ebits - 1) - 1
    elif fmt == ElemFormat.bfloat16:
        ebits, mbits = 8, 9
        emax = 2**(ebits - 1) - 1
    else:
        raise Exception("Unknown element format %s" % fmt)

    if fmt != ElemFormat.fp8_e4m3:
        max_norm = 2**emax * float(2**(mbits-1) - 1) / 2**(mbits-2)
    else:
        max_norm = 2**emax * 1.75  # FP8 has custom max_norm

    min_norm = _get_min_norm(ebits)

    _FORMAT_CACHE[fmt] = (ebits, mbits, emax, max_norm, min_norm)

    return ebits, mbits, emax, max_norm, min_norm


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
    A_is_sparse = A.is_sparse
    if A_is_sparse:
        if A.layout != torch.sparse_coo:
            raise NotImplementedError("Only COO layout sparse tensors are currently supported.")

        sparse_A = A.coalesce()
        A = sparse_A.values().clone()

    # Flush values < min_norm to zero if denorms are not allowed
    if not allow_denorm and exp_bits > 0:
        min_norm = _get_min_norm(exp_bits)
        out = (torch.abs(A) >= min_norm).type(A.dtype) * A
    else:
        out = A

    if exp_bits != 0:
        private_exp = torch.floor(torch.log2(
            torch.abs(A) + (A == 0).type(A.dtype)))

        # The minimum representable exponent for 8 exp bits is -126
        min_exp = -(2**(exp_bits-1)) + 2
        private_exp = private_exp.clip(min=min_exp)
    else:
        private_exp = None

    # Scale up so appropriate number of bits are in the integer portion of the number
    out = _safe_lshift(out, bits - 2, private_exp)

    out = _round_mantissa(out, bits, round, clamp=False)

    # Undo scaling
    out = _safe_rshift(out, bits - 2, private_exp)

    # Set values > max_norm to Inf if desired, else clamp them
    if saturate_normals or exp_bits == 0:
        out = torch.clamp(out, min=-max_norm, max=max_norm)
    else:
        out = torch.where((torch.abs(out) > max_norm),
                           torch.sign(out) * float("Inf"), out)

    # handle Inf/NaN
    # out[A == float("Inf")] = float("Inf")
    # out[A == -float("Inf")] = -float("Inf")
    # out[A == float("NaN")] = float("NaN")

    if A_is_sparse:
        output = torch.sparse_coo_tensor(sparse_A.indices(), output,
                sparse_A.size(), dtype=sparse_A.dtype, device=sparse_A.device,
                requires_grad=sparse_A.requires_grad)

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

    ebits, mbits, emax, max_norm, _ = _get_format_params(elem_format)

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

    scale_emax = 2**(scale_bits-1) - 1
    shared_exp[shared_exp > scale_emax] = float("NaN")
    shared_exp[shared_exp < -scale_emax] = -scale_emax

    A = A / (2**shared_exp)

    A = _quantize_elemwise_core(
            A, mbits, ebits, max_norm, round=round,
            allow_denorm=True, saturate_normals=True)

    A = A * (2**shared_exp)

    # Undo tile reshaping
    if block_size:
        A = _undo_reshape_to_blocks(A, padded_shape, orig_shape, axes)

    return A



def optimized_mxfp8_e4m3_matmul(A:torch.Tensor,B:torch.Tensor,block_size:int=32)->torch.Tensor:
    scale_bits = 8
    elem_format = 'fp8_e4m3'
    A = _quantize_mx(
        A,
        scale_bits,
        elem_format,
        shared_exp_method="max",
        axes=-1,
        block_size=16,
        round="nearest",
        flush_fp32_subnorms=False,
    )

    B = _quantize_mx(
        B,
        scale_bits,
        elem_format,
        shared_exp_method="max",
        axes=-2,
        block_size=16,
        round="nearest",
        flush_fp32_subnorms=False,
    )

    C = torch.matmul(A, B)
    return C



def optimized_mxfp8_e5m2_matmul(A:torch.Tensor,B:torch.Tensor,block_size:int=32)->torch.Tensor:
    scale_bits = 8
    elem_format = 'fp8_e5m2'
    A = _quantize_mx(
        A,
        scale_bits,
        elem_format,
        shared_exp_method="max",
        axes=-1,
        block_size=16,
        round="nearest",
        flush_fp32_subnorms=False,
    )

    B = _quantize_mx(
        B,
        scale_bits,
        elem_format,
        shared_exp_method="max",
        axes=-2,
        block_size=16,
        round="nearest",
        flush_fp32_subnorms=False,
    )
    C = torch.matmul(A, B)
    return C

def optimized_mxfp4_e2m1_matmul(A:torch.Tensor,B:torch.Tensor,block_size:int=32)->torch.Tensor:
    scale_bits = 8
    elem_format = 'fp4_e2m1'
    A = _quantize_mx(
        A,
        scale_bits,
        elem_format,
        shared_exp_method="max",
        axes=-1,
        block_size=16,
        round="nearest",
        flush_fp32_subnorms=False,
    )

    B = _quantize_mx(
        B,
        scale_bits,
        elem_format,
        shared_exp_method="max",
        axes=-2,
        block_size=16,
        round="nearest",
        flush_fp32_subnorms=False,
    )

    C = torch.matmul(A, B)
    return C

import torch
from torch.autograd import Function

class MXFPMatMul(Function):
    @staticmethod
    def forward(ctx, A: torch.Tensor, B: torch.Tensor,
                elem_format: str = 'fp8_e5m2', block_size: int = 32):
        ctx.save_for_backward(A, B)
        ctx.elem_format = elem_format
        ctx.block_size = block_size
        
        A_q = _quantize_mx(
            A, scale_bits=8, elem_format=elem_format,
            shared_exp_method="max", axes=-1, block_size=block_size,
            round="nearest", flush_fp32_subnorms=False
        )
        B_q = _quantize_mx(
            B, scale_bits=8, elem_format=elem_format,
            shared_exp_method="max", axes=-2, block_size=block_size,
            round="nearest", flush_fp32_subnorms=False
        )
        return torch.matmul(A_q, B_q)

    @staticmethod
    def backward(ctx, grad_output):
        A, B = ctx.saved_tensors
        grad_A = grad_B = None
        if ctx.needs_input_grad[0]:
            grad_A = torch.matmul(grad_output, B.transpose(-2, -1))
        if ctx.needs_input_grad[1]:
            grad_B = torch.matmul(A.transpose(-2, -1), grad_output)
        return grad_A, grad_B, None, None  # None对应elem_format和block_size

class MXFPBAddBmm(Function):
    @staticmethod
    def forward(ctx, input, batch1, batch2, beta=1.0, alpha=1.0,
                elem_format='fp8_e5m2', block_size=32):
        ctx.save_for_backward(input, batch1, batch2)
        ctx.beta, ctx.alpha = beta, alpha
        ctx.elem_format = elem_format
        ctx.block_size = block_size
        
        mm_out = MXFPMatMul.apply(batch1, batch2, elem_format, block_size)
        return beta * input + alpha * mm_out

    @staticmethod
    def backward(ctx, grad_output):
        input, batch1, batch2 = ctx.saved_tensors
        beta, alpha = ctx.beta, ctx.alpha
        
        grad_input = grad_batch1 = grad_batch2 = None
        if ctx.needs_input_grad[0]:
            grad_input = beta * grad_output
        if ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
            mm_grad = alpha * grad_output
            grad_batch1 = torch.matmul(mm_grad, batch2.transpose(-2, -1))
            grad_batch2 = torch.matmul(batch1.transpose(-2, -1), mm_grad)
        
        return grad_input, grad_batch1, grad_batch2, None, None, None, None

def mxfp_matmul(A, B, elem_format='fp8_e5m2', block_size=32):
    return MXFPMatMul.apply(A, B, elem_format, block_size)

def mxfp_baddbmm(input, batch1, batch2, beta=1.0, alpha=1.0,
                 elem_format='fp8_e5m2', block_size=32):
    return MXFPBAddBmm.apply(input, batch1, batch2, beta, alpha, elem_format, block_size)

# def mxfp_matmul(A:torch.Tensor,B:torch.Tensor,elem_format:str='fp8_e5m2',block_size:int=32)->torch.Tensor:
#     scale_bits = 8
#     # elem_format = 'fp8_e5m2'
#     A = _quantize_mx(
#         A,
#         scale_bits,
#         elem_format,
#         shared_exp_method="max",
#         axes=-1,
#         block_size=block_size,
#         round="nearest",
#         flush_fp32_subnorms=False,
#     )

#     B = _quantize_mx(
#         B,
#         scale_bits,
#         elem_format,
#         shared_exp_method="max",
#         axes=-2,
#         block_size=block_size,
#         round="nearest",
#         flush_fp32_subnorms=False,
#     )
#     C = torch.matmul(A, B)
#     return C

# def mxfp_baddbmm(input: torch.Tensor, batch1: torch.Tensor, batch2: torch.Tensor, *, beta: float=1.0, alpha: float=1.0,elem_format:str='fp8_e5m2',block_size:int=32 ) -> torch.Tensor: 
#     """
#     out=beta * input + alpha * batch1 @ batch2
#     """
#     mm_out = mxfp_matmul(A=batch1,B=batch2,elem_format=elem_format,block_size=block_size)
#     out = beta * input + alpha * mm_out
#     return out

if __name__ == '__main__':
    A = torch.load("grad_output.pt", map_location='cpu').cuda()
    print(f"A_shape:{A.shape},grad_max:{torch.max(A)},grad_min:{torch.min(A)}")
    B = torch.load("total_input.pt", map_location='cpu').cuda()
    print(f"B_shape:{B.shape},input_max:{torch.max(B)},input_min:{torch.min(B)}")
    A = A.unsqueeze(0).repeat(3, 1, 1)
    B = B.unsqueeze(0).repeat(3, 1, 1)
    C = torch.matmul(A.transpose(-2, -1), B)
    D = torch.baddbmm(C,A.transpose(-2,-1),B)
    print(f"C_shape:{C.shape},output_max:{torch.max(C)},output_min:{torch.min(C)}")
    C_e4m3 = mxfp_matmul(A.transpose(-2,-1),B,'fp8_e4m3')
    D_e4m3 = mxfp_baddbmm(C,A.transpose(-2,-1),B,elem_format='fp8_e4m3')
    print(f"C_shape:{C_e4m3.shape},output_max:{torch.max(C_e4m3)},output_min:{torch.min(C_e4m3)}")
    print(f"D_shape:{D_e4m3.shape},output_max:{torch.max(D_e4m3)},output_min:{torch.min(D_e4m3)}")
    print(torch.isnan(C).any())

    mse_e4m3 = torch.mean((C - C_e4m3) ** 2)
    max_err_e4m3 = torch.max(torch.abs(C - C_e4m3))
    print(f"MSE: {mse_e4m3:.20f}")
    print(f"Max Error: {max_err_e4m3:.20f}")
    print(f"相对误差: {mse_e4m3 / torch.mean(C ** 2):.20f}")
    
    b_mse_e4m3 = torch.mean((D - D_e4m3) ** 2)
    b_max_err_e4m3 = torch.max(torch.abs(D - D_e4m3))
    print(f"B_MSE: {b_mse_e4m3:.20f}")
    print(f"B Max Error: {b_max_err_e4m3:.20f}")
    


