import numpy as np
from quant.qtype import QType
from torch import Tensor
import torch
from torch.autograd import Function

def to_HiFX(x, G: int = 64, N: int = 4) -> np.ndarray:
    x = np.array(x)
    Mi, Ni = x.shape[0],x.shape[1]
    Mcnt = np.ceil(Mi / G).astype(int)
    res = np.zeros((Mi,Ni))
    Ng = N - 2
    for i in range(Mcnt):
        for j in range(Ni):
            ori = x[i*G : i*G+G, j]        # 当前 64 长度向量
            S = np.ones(G)
            S[ori < 0] = -1
            S = S.T
            tmpG = np.abs(ori)

            # ---------- level-1 ----------
            EG = np.floor(np.log2(tmpG + 2**(-1000)))
            E16 = np.zeros(16)
            for k in range(16):
                E16[k] = np.max(EG[k*4 : k*4+4])

            E8 = np.zeros(8)
            for k in range(8):
                E8[k] = np.max(E16[k*2 : k*2+2])

            Emax = np.max(E8)
            E8_1 = Emax - 2                # [-127, 125]

            # ---------- level-2 ----------
            E1_8 = E8 - E8_1 - 1           # <= 1
            E1_8[E1_8 < 0] = 0             # [0, 1]

            # ---------- level-3 ----------
            E1_8x2 = np.zeros(16)
            for k in range(8):
                E1_8x2[k*2 : k*2+2] = E1_8[k]

            E1_16 = E16 - E1_8x2 - E8_1    # <= 1
            E1_16[E1_16 < 0] = 0           # [0, 1]

            # ---------- restore ----------
            E16G = E1_16 + E1_8x2 + E8_1   # fused 16 exp
            EG = np.zeros(G)
            for k in range(16):
                EG[k*4 : k*4+4] = E16G[k]

            in_grp = np.floor(tmpG * 2**(-EG + Ng) + 0.5) * 2.0**(-Ng)
            in_grp[in_grp >= 2] = 2 - 2**(-Ng)
            grp = S * in_grp * 2.0**EG
            res[i*G : i*G+G, j] = grp

    return res


class RoundHif8_dml(Function):
    @staticmethod
    def forward(ctx,
                x: torch.Tensor,
                max_exp: int,
                min_exp: int,
                Ec: int) -> torch.Tensor:
        x_tmp = x.clone().detach()
        E = torch.floor(torch.log2(torch.abs(x_tmp))).detach().float()
        D = torch.floor(torch.log2(torch.abs(E - Ec))) + 1
        D = torch.where(E != Ec, D, 0)

        x = torch.where(D <= 2,
                        torch.round(x * torch.exp2(-E + 3)) * torch.exp2(-3 + E),
                        x)
        x = torch.where((D > 2) & (D < 5),
                        torch.round(x * torch.exp2(-E + 5 - D)) * torch.exp2(D - 5 + E),
                        x)
        x = torch.where(D >= 5,
                        torch.round(x * torch.exp2(-E)) * torch.exp2(E),
                        x)

        over_value  = 1.25 * 2**(max_exp + Ec)
        down_value  = 1.5 * 2**(min_exp + Ec)
        x = torch.where(x_tmp >= over_value,  over_value,  x)
        x = torch.where(torch.abs(x_tmp) <= down_value, 0.0, x)
        x = torch.where(torch.isinf(x_tmp)|torch.isnan(x_tmp), x_tmp, x)   # 保持 NaN
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None

round_hif8_func_dml = RoundHif8_dml.apply
def any_to_hif8_dml(num: torch.Tensor,
                    Ec: int = 0,
                    dml = True) -> torch.Tensor:
    dtype = num.dtype
    num = num.float()
    if dml:
        max_exp = 15
        min_exp = -22
        num = round_hif8_func_dml(num, max_exp, min_exp, Ec)
    num = num.to(dtype)
    return num


fp_max_dict = {
    "fp16": 65504.0,
    "e4m3": 448.0,
    "e5m2": 57344.0,
    "hif8_7": 224.0,
    "hif8_15": 32768.0
}

def compute_scaling_factor_fp8(amax: torch.Tensor,
                               scale: torch.Tensor,
                               fp_max: float) -> torch.Tensor:
    sf = fp_max / amax
    sf = torch.where(amax > 0.0, sf, scale)
    sf = torch.where(torch.isfinite(amax), sf, scale)
    sf = torch.where(torch.isinf(sf),
                     torch.full_like(sf, torch.finfo(amax.dtype).max),
                     sf)
    return sf


@torch.no_grad()
def quant_hif8(x: Tensor, Q: QType=None, qdim: int=-1) -> Tensor:
    max_value = (2**15)*0.95
    min_value = 2**(-22)

    x_unsignedl = torch.abs(x)
    sign = torch.sign(x)

    x_unsigned = torch.clamp(x_unsignedl, min=min_value, max=max_value)

    if x.dtype == torch.float16:
        e = torch.floor(torch.log2(x_unsigned + 2**(-14)))
    else:
        e = torch.floor(torch.log2(x_unsigned + 2**(-45)))

    abse = e.abs()
    mant_bits = torch.zeros_like(abse)
    mant_bits[abse <= 15] = 1
    mant_bits[abse <= 7] = 2
    mant_bits[abse <= 3] = 3

    res = torch.floor(x_unsigned * 2.0**(-e + mant_bits) + 0.5) * 2.0**(e - mant_bits) * sign
    return res

# def hifp_matmul(A:torch.Tensor,B:torch.Tensor)->torch.Tensor:
    # A = quant_hif8(A)
    # # A = any_to_hif8_dml(A,Ec=15)
    # B = quant_hif8(B)
    # # B = any_to_hif8_dml(B,Ec=15)
    # C = torch.matmul(A,B)
    # return C
import torch
from torch.autograd import Function

class HIFPMatMul(Function):
    @staticmethod
    def forward(ctx, A: torch.Tensor, B: torch.Tensor,
                elem_format: str = 'fp8_e5m2', block_size: int = 32):
        ctx.save_for_backward(A, B)
        ctx.elem_format = elem_format
        ctx.block_size = block_size
        
        # A_q = _quantize_mx(
        #     A, scale_bits=8, elem_format=elem_format,
            # shared_exp_method="max", axes=-1, block_size=block_size,
            # round="nearest", flush_fp32_subnorms=False
        # )
        A_q = quant_hif8(A)
        # B_q = _quantize_mx(
            # B, scale_bits=8, elem_format=elem_format,
            # shared_exp_method="max", axes=-2, block_size=block_size,
            # round="nearest", flush_fp32_subnorms=False
        # )
        B_q = quant_hif8(B)
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

class HIFPBAddBmm(Function):
    @staticmethod
    def forward(ctx, input, batch1, batch2, beta=1.0, alpha=1.0):
        ctx.save_for_backward(input, batch1, batch2)
        ctx.beta, ctx.alpha = beta, alpha
        
        mm_out = HIFPMatMul.apply(batch1, batch2)
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

def hifp_matmul(A, B):
    return HIFPMatMul.apply(A, B)

def hifp_baddbmm(input, batch1, batch2, beta=1.0, alpha=1.0):
    return HIFPBAddBmm.apply(input, batch1, batch2, beta, alpha)

if __name__ == "__main__":
    A = torch.load("grad_output.pt", map_location='cpu').cuda()
    fp8 = quant_hif8(A)
    # fp8 = any_to_hif8_dml(A, Ec=15)

    print("origin_A:", A)
    print("hif8_A:", fp8)
    
    print(f"A_shape:{A.shape},grad_max:{torch.max(A)},grad_min:{torch.min(A)}")
    B = torch.load("total_input.pt", map_location='cpu').cuda()
    print(f"B_shape:{B.shape},input_max:{torch.max(B)},input_min:{torch.min(B)}")

    C_hifp8 = hifp_matmul(A.transpose(-2,-1),B)
    
    print(f"C_shape:{C_hifp8.shape},output_max:{torch.max(C_hifp8)},output_min:{torch.min(C_hifp8)}")
