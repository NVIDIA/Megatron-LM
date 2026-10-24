"""Two-level softmax fold shared by the PP attention kernels."""

import cutlass.cute as cute
from cutlass._mlir.dialects import cute as _cute_ir
from cutlass.cute.core import _to_mlir_address_space
from cutlass.cute.typing import Float32
from flash_attn.cute.utils import fmax


def view16(buf, off, n):
    """n-element view of `buf` at element offset `off`, tagged 16B-aligned."""
    base = buf.iterator + off
    swz = base.type.swizzle_type.attribute if base.type.is_swizzled else None
    addr = base.toint()
    cty = _cute_ir.ConstrainedIntType.get(16, type(addr).width)
    aaddr = _cute_ir.assume(cty, addr.ir_value())
    pty = _cute_ir.PtrType.get(base.type.value_type, _to_mlir_address_space(base.memspace), 16, swz)
    ptr = _cute_ir.inttoptr(pty, aaddr)
    ptr._dtype = buf.element_type
    return cute.make_tensor(ptr, cute.make_layout(n))


def vcopy(src, dst):
    """Vector copy through one universal copy atom (up to 128 bits per op)."""
    bits = min(128, cute.size(dst.shape) * src.element_type.width)
    atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), src.element_type, num_bits_per_copy=bits
    )
    cute.copy(atom, src, dst)


@cute.jit
def fold_stats(
    M: Float32, L: Float32, m_s: Float32, l_s: Float32, scale_log2: Float32
) -> tuple[Float32, Float32, Float32, Float32]:
    """Fold partial stats (m_s, l_s) into running (M, L)."""
    M_new = fmax(M, m_s)
    M_safe = M_new
    if M_new == -Float32.inf:
        M_safe = Float32(0.0)
    a = cute.math.exp2(scale_log2 * (M - M_safe), fastmath=True)
    b = cute.math.exp2(scale_log2 * (m_s - M_safe), fastmath=True)
    la, lb = cute.arch.mul_packed_f32x2((L, l_s), (a, b))
    L_new = la + lb
    return M_new, L_new, a, b


@cute.jit
def fold_pair(
    obar2: tuple[Float32, Float32], o2: tuple[Float32, Float32], a: Float32, b: Float32
) -> tuple[Float32, Float32]:
    """Obar' = Obar * a + O_s * b on a packed f32x2 pair (mul, mul, add; no fma)."""
    p_a = cute.arch.mul_packed_f32x2(obar2, (a, a))
    p_b = cute.arch.mul_packed_f32x2(o2, (b, b))
    return cute.arch.add_packed_f32x2(p_a, p_b)


@cute.jit
def normalize_scale(L: Float32) -> Float32:
    """1/L with the FA4 epilogue's zero-or-NaN guard; a true FP32 division."""
    l_is_zero_or_nan = L == 0.0 or L != L
    scale = Float32(1.0) / L if not l_is_zero_or_nan else Float32(0.0)
    return scale
