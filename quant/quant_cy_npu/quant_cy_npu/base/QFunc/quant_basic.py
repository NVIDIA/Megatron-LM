import torch
from ..QType import QType
from torch import Tensor


@torch.no_grad()
def quant_py(x: Tensor, Q: QType, qdim: int) -> Tensor:
    # reshape x
    x = x.unflatten(qdim, (-1, Q.blk_outer_size, Q.blk_size))
    x_unsigned = torch.abs(x)
    sign = torch.sign(x)
    exp_offset = Q.exp_offset

    if Q.exp_bits==0:
        # compute initial shared exp
        max_inner = torch.max(x_unsigned, dim=qdim, keepdim=True)[0]
        shared_exp_grp = torch.floor(torch.log2(max_inner + 9.6e-7)) - Q.exp_max   # find max exp
        shared_exp_grp = torch.clip(shared_exp_grp, -Q.k_max-exp_offset, Q.k_max-exp_offset)

        if Q.do_carry:
            # if mantissa become 10.00, shift exp by 1
            raw_mant = x_unsigned / torch.exp2(shared_exp_grp)
            exp_bias = torch.floor(raw_mant + 0.5) - (2**Q.man_bits - 1)   # max: 1
            exp_bias = torch.clip(exp_bias, 0, 1)   # min: 0
            shared_exp_grp = shared_exp_grp + torch.max(exp_bias, dim=qdim, keepdim=True)[0]

        # constrain the smaller exp within range of (larger one - max_bias)
        shared_exp_blk = torch.max(shared_exp_grp, dim=qdim-1, keepdim=True)[0]
        shared_exp = shared_exp_grp.clip(shared_exp_blk - (2**Q.k_bits-1))
        shared_exp = shared_exp.clip(-Q.k_max-exp_offset, Q.k_max-exp_offset)

        # compute final value
        mant = x_unsigned / torch.exp2(shared_exp)
        mant = torch.floor(mant + 0.5)
        mant = mant.clip(0, 2**(Q.man_bits)-1)

        underflow_idx = (x_unsigned < (2-2**(-Q.man_bits))*(2**(-Q.k_max-exp_offset+1)))
        out = sign * mant * (2 ** shared_exp)
        out[underflow_idx] = 0

        nan_threshold = 2**(Q.k_max-exp_offset+Q.exp_max)*(2-1/2**Q.man_bits)
        nan_idx = torch.any(torch.isnan(x_unsigned), dim=qdim, keepdim=True) | torch.any(torch.isinf(x_unsigned), dim=qdim, keepdim=True) | (torch.max(x_unsigned>=nan_threshold, dim=qdim, keepdim=True)[0]).max(dim=qdim-1, keepdim=True)[0]

        nan_idx = torch.broadcast_to(nan_idx, out.shape)
        out[nan_idx] = torch.nan
    else:
        # compute shared bits that constrain the max exponential to exp_max
        # for example for fp4, x: 8 -> shared_exp: 3 -> after sub: 1
        max_inner = torch.max(x_unsigned, dim=qdim, keepdim=True)[0]
        if Q.do_carry:
            shared_exp_grp = torch.floor(torch.log2(max_inner + 9.6e-7) + 0.5) - Q.exp_max   # find max exp
        else:
            shared_exp_grp = torch.floor(torch.log2(max_inner + 9.6e-7)) - Q.exp_max   # find max exp
        shared_exp_grp = torch.clip(shared_exp_grp, -Q.k_max-exp_offset, Q.k_max-exp_offset)

        # constrain the smaller exp within range of larger one
        shared_exp_blk = torch.max(shared_exp_grp, dim=qdim-1, keepdim=True)[0]
        shared_exp = shared_exp_grp.clip(shared_exp_blk - (2**Q.k_bits-1))
        shared_exp = shared_exp.clip(-Q.k_max-exp_offset, Q.k_max-exp_offset)

        # compute private exp for each number
        # for example for fp4, x: 8 -> div shared_exp: 4 -> private_exp_before_bias: 2
        x_biased = x_unsigned / torch.exp2(shared_exp)
        private_exp_with_bias = torch.floor(torch.log2(x_biased + 9.6e-7))
        private_exp = torch.clip(private_exp_with_bias, Q.exp_min, Q.exp_max)

        # compute mantissa
        # for example for fp4: x: 14 -> div exp: 1.75 -> mant_lshifted: 3.5 -> mant_trunc: 3 -> mant: 1.5
        mant_lshifted = x_biased / (2 ** private_exp) * (2 ** Q.man_shift_bit) #1xpyzzzz
        mant_trunc = torch.floor(mant_lshifted + 0.5)  # 1xpyzzz -> 1x(rounded)
        fp_value = mant_trunc / (2 ** Q.man_shift_bit) * (2 ** private_exp)
        fp_value = torch.clip(fp_value, -Q.fp_val_max, Q.fp_val_max)

        ind_nan = shared_exp<-127
        fp_value[torch.broadcast_to(ind_nan, fp_value.shape)] = 0

        # dequantize it
        out = sign * (2 ** shared_exp) * fp_value
    out = out.flatten(qdim-2, qdim)
    return out