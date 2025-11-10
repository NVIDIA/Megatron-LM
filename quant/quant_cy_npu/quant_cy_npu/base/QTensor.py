import re
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Function
from torch.cuda.amp import custom_fwd, custom_bwd   # type: ignore
from functools import partial
from typing import Callable, Tuple, Optional, Union, List

from .QType import QType
# from .cusrc import npu_quant

from .QFunc.quant_basic import quant_py
# from .QFunc.int8 import quant_int8sym
# from .QFunc.hif8 import quant_hif8
# from .QFunc.nvf4 import quant_nvf4
# from .QFunc.hifx import quant_hifx


# type defs
NPU_FUNC_BUNDLE_T = Tuple[Optional[Callable], Optional[Callable], Optional[Callable]]

QFUNC_MAP: Tuple[Tuple[str, Callable], ...] = ((r'^int8sym$', quant_int8sym),
                                               (r'^hifx[0-9]*$', quant_hifx),
                                               (r'^hif8$', quant_hif8),
                                               (r'^nvf4$', quant_nvf4),
                                            )

NPU_KERNELS: Tuple[Tuple[str, NPU_FUNC_BUNDLE_T], ...] = (
                (r'^hifx[0-9]*$', (npu_quant.hifx_quant, npu_quant.hifx_quant_bf16, None)),
                (r'^mxfp4$', (npu_quant.mxfp4_quant, npu_quant.mxfp4_quant_bf16, None),),
                (r'^hif8$', (npu_quant.hif8_quant, npu_quant.hif8_quant_bf16, None)),
                (r'^mxfp8e4m3$', (npu_quant.mxfp8e4m3_quant, npu_quant.mxfp8e4m3_quant_bf16, None)),
                (r'^nvf4$', (npu_quant.nvf4_quant, npu_quant.nvf4_quant_bf16, None)),
            )


# npu quant function generator
def get_npu_func(x: Tensor, Q: QType) -> Optional[Callable[[Tensor], Tensor]]:
    def func_wrapper(x: Tensor, func: Callable, cvt_fp32: bool):
        if x.numel()<512:
            print('WARNING!!!!!!! X SIZE IS SMALLER THAN 512. MAY CAUSE ILLEGAL MEMORY OR WRONG RESULTS!!!')
        # transpose qdim to last
        if Q.q_dim!=-1 or Q.q_dim!=(len(x.shape)-1):
            x = x.transpose(Q.q_dim, -1).contiguous()

        # construct output tensor
        if cvt_fp32:
            x2 = x.to(torch.float32)
            out = torch.zeros_like(x, dtype=torch.float32, device=x.device)
        else:
            x2 = x
            out = torch.zeros_like(x, dtype=x.dtype, device=x.device)

        # run quant function
        if Q.desc[:4]=='hifx':
            func(x2, out, Q.man_bits)
        else:
            func(x2, out)

        # transpose back
        if Q.q_dim!=-1 or Q.q_dim!=(len(x.shape)-1):
            out = out.transpose(Q.q_dim, -1).contiguous()
        # convert dtype back
        if cvt_fp32:
            out = out.to(x.dtype)
        return out

    for reg_str, qfuncs in NPU_KERNELS:
        if re.match(reg_str, Q.desc):
            # print('Matched NPU kernel:', reg_str, Q.desc)
            idx: int = 0
            if x.dtype==torch.bfloat16:
                idx = 1
            elif x.dtype==torch.float16:
                idx = 2
            elif x.dtype==torch.float32:
                idx = 0

            cvt_fp32 = False
            if qfuncs[idx] is None:
                idx = 0
                cvt_fp32 = True

            f_sel = qfuncs[idx]
            # print('DTYPE IDX', idx)
            assert f_sel is not None
            return partial(func_wrapper, func=f_sel, cvt_fp32=cvt_fp32)
    return None


# pytorch quant function generator
def get_torch_func(x: Tensor, Q: QType, qdim: int) -> Callable[[Tensor], Tensor]:
    for reg_str, qfunc in QFUNC_MAP:
        if re.match(reg_str, Q.desc):
            return partial(qfunc, Q=Q, qdim=qdim)
    return partial(quant_py, Q=Q, qdim=qdim)


@torch.no_grad()
def quant_dequant_float(x: Tensor, Q: QType, force_py: bool=False, force_fp32: bool=False, **kwargs) -> Tensor:
    if Q.desc in ['fp16', 'bf16', 'fp32']:
        return x
    # torch.npu.synchronize()
    # convert to fp32 if forced
    dtype_ori = x.dtype
    if force_fp32:
        x = x.to(torch.float32)

    # pad to fit block size
    C = x.shape[Q.q_dim]
    blk_size_total = Q.blk_size * Q.blk_outer_size
    padC = (blk_size_total - C % blk_size_total) % blk_size_total
    qdim = Q.q_dim
    if qdim>=0:
        qdim = qdim - len(x.shape)
    if padC>0:
        pads = [0]* (-qdim * 2 - 1) + [padC]
        x = F.pad(x, pads, value=0.0)

    if (not force_py) and (f:=get_npu_func(x, Q)):
        # torch.npu.synchronize()
        out = f(x)
    else:
        out = get_torch_func(x, Q, qdim)(x)

    # slice and resize back
    if padC>0:
        slices = [slice(0, out.shape[i]) for i in range(len(x.shape)+qdim)] + [slice(0, C),]
        out = out[slices]

    if out.dtype!=dtype_ori:
        out = out.to(dtype_ori)
    return out


class QuantFunc(Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, x, Q, force_py, force_fp32):
        return quant_dequant_float(x, Q, force_py, force_fp32)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        return grad, None, None, None


def quant_func(x: Tensor, Q: QType, force_py: bool=False, force_fp32: bool=False) -> Tensor:
    return QuantFunc.apply(x, Q, force_py, force_fp32)   # type: ignore


class QuantSlideWindow(Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, x, Q, force_py, force_fp32, dim, win_size):
        if isinstance(win_size, (list, tuple)):
            if x.shape[-2]>win_size[0] and x.shape[-1]>win_size[1]:
                x[..., :-win_size[0], :-win_size[1]] = quant_dequant_float(x[..., :-win_size[0], :-win_size[1]], Q, force_py, force_fp32)
        elif dim==-1:
            if x.shape[-1]>win_size:
                x[..., :-win_size] = quant_dequant_float(x[..., :-win_size], Q, force_py, force_fp32)
        elif dim==-2:
            if x.shape[-2]>win_size:
                x[..., :-win_size, :] = quant_dequant_float(x[..., :-win_size, :], Q, force_py, force_fp32)
        else:
            raise NotImplementedError(f'QDIM only supports -1/-2, but got {dim}')
        return x

    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        return grad, None, None, None, None, None


def quant_slide_window(x: Tensor, Q: QType, force_py: bool=False, force_fp32: bool=False, \
                       qdim: int=-2, win_size: Union[int, List[int], Tuple[int,int]]=128) -> Tensor:
    if qdim>0:
        qdim = qdim - len(x.shape)
    return QuantSlideWindow.apply(x, Q, force_py, force_fp32, qdim, win_size)   # type: ignore