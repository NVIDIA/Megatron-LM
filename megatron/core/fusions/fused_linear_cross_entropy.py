# -*- coding: utf-8 -*-

# Code adapted from
# https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/fused_linear_cross_entropy.py

import functools
from functools import partial
from typing import Optional, Tuple
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from torch.distributed import DeviceMesh
from torch.distributed.tensor import Replicate, Shard, distribute_module
from torch.distributed.tensor.parallel import ParallelStyle
from torch.nn.parameter import Parameter
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Callable, Dict, Literal, Optional, Tuple
from typing import Optional
import torch
import triton
import triton.language as tl
import triton
import triton.language as tl
import triton.language.extra.libdevice as tldevice
from packaging import version

from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.tensor_parallel.layers import _initialize_affine_weight_gpu

@lru_cache(maxsize=None)
def check_pytorch_version(version_s: str = '2.4') -> bool:
    return version.parse(torch.__version__) >= version.parse(version_s)

@lru_cache(maxsize=None)
def get_available_device() -> str:
    try:
        return triton.runtime.driver.active.get_current_target().backend
    except BaseException:
        _cpu_device_warning()
        return 'cpu'
@lru_cache(maxsize=None)
def _check_platform() -> Literal['nvidia', 'amd', 'intel', 'musa', 'npu']:
    device = get_available_device()
    if device == 'cuda':
        return 'nvidia'
    elif device == 'hip':
        return 'amd'
    elif device == 'xpu':
        return 'intel'
    else:
        return device

device = get_available_device() if get_available_device() != 'hip' else 'cuda'
device_torch_lib = getattr(torch, device)
device_platform = _check_platform()

if check_pytorch_version('2.4'):
    device = 'cuda' if device == 'cpu' else device
    autocast_custom_fwd = functools.partial(torch.amp.custom_fwd, device_type=device)
    autocast_custom_bwd = functools.partial(torch.amp.custom_bwd, device_type=device)
    def custom_device_ctx(index: int):
        return device_torch_lib.device(index)
else:
    assert device == 'cuda', 'Only cuda device is supported for PyTorch version < 2.4.0.'
    autocast_custom_fwd = device_torch_lib.amp.custom_fwd
    autocast_custom_bwd = device_torch_lib.amp.custom_bwd
    def custom_device_ctx(index: int):
        return torch.cuda.device(index)


# For AMD GPUs, the triton backend is 'hip', while for Nvidia GPUs, the triton backend is 'cuda'.
# However, the torch backend is 'cuda' for both Nvidia and AMD GPUs.
# Therefore, we need to check the triton backend to determine the actual GPU vendor.
device = get_available_device() if get_available_device() != 'hip' else 'cuda'
device_torch_lib = getattr(torch, device)
device_platform = _check_platform()
is_amd = (device_platform == 'amd')
is_nvidia = (device_platform == 'nvidia')


# Nvidia Ampere or newer, haven't check AMD and intel yet.
is_tf32_supported = (is_nvidia and torch.cuda.get_device_capability(0)[0] >= 8)
is_gather_supported = hasattr(triton.language, 'gather')

if os.environ.get('FLA_USE_FAST_OPS', '0') == '1':
    div = tldevice.fast_dividef
    exp = tldevice.fast_expf
    exp2 = tldevice.exp2
    log = tldevice.fast_logf
    log2 = tldevice.fast_log2f
else:
    @triton.jit
    def div_normal(x, y):
        return x / y
    div = div_normal
    exp = tl.exp
    exp2 = tl.math.exp2
    log = tl.log
    log2 = tl.log2

def input_guard(
    fn: Callable[..., torch.Tensor]
) -> Callable[..., torch.Tensor]:
    """
    A decorator to make sure all input tensors are contiguous and set the device based on input tensors.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        contiguous_args = (i if not isinstance(i, torch.Tensor) else i.contiguous() for i in args)
        contiguous_kwargs = {k: (v if not isinstance(v, torch.Tensor) else v.contiguous()) for k, v in kwargs.items()}

        tensor = None
        for arg in args:
            if isinstance(arg, torch.Tensor):
                tensor = arg
                break
        if tensor is None:
            for value in kwargs.values():
                if isinstance(value, torch.Tensor):
                    tensor = value
                    break

        if tensor is not None:
            ctx = custom_device_ctx(tensor.device.index)
        else:
            ctx = contextlib.nullcontext()

        with ctx:
            return fn(*contiguous_args, **contiguous_kwargs)

    return wrapper


contiguous = input_guard

@triton.heuristics({
    'HAS_SCALE': lambda args: args['scale'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4, 8, 16, 32]
    ],
    key=['D']
)
@triton.jit
def logsumexp_fwd_kernel(
    x,
    z,
    scale,
    D: tl.constexpr,
    B: tl.constexpr,
    HAS_SCALE: tl.constexpr
):
    i_n, i_d = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    o_d = i_d * B + tl.arange(0, B)
    m_d = o_d < D

    b_x = tl.load(x + i_n * D + o_d, mask=m_d, other=-float('inf'))
    if HAS_SCALE:
        b_x = b_x * scale
    b_m = tl.max(b_x, 0)
    b_z = log(tl.sum(exp(b_x - b_m), 0)) + b_m
    tl.store(z + i_n * tl.cdiv(D, B) + i_d, b_z)

def logsumexp_fwd(
    x,
    scale: Optional[float] = None,
    dtype: Optional[torch.dtype] = None
):
    r"""
    Compute the logsumexp of the input tensor over the last dimension.

    Args:
        x (Tensor):
            The input tensor of any shape.
        scale (Optional[float]):
            The scale applied to the input tensor. Default: `None`.
        dtype (Optional[torch.dtype]):
            The data type of the output tensor. Default: `None`.
    Returns:
        Tensor: The logsumexp of the input tensor.
    """

    shape = x.shape
    x = x.view(-1, shape[-1])
    N, D = x.shape
    B = min(triton.next_power_of_2(D), 64 * 1024)
    ND = triton.cdiv(D, B)

    z = x.new_empty(N, ND, dtype=torch.float)
    logsumexp_fwd_kernel[(N, ND)](
        x=x,
        z=z,
        scale=scale,
        D=D,
        B=B
    )
    z = z.logsumexp(-1).view(*shape[:-1])
    if dtype is not None and dtype != torch.float:
        z = z.to(dtype)
    return z

try:
    from torch.distributed.tensor import DTensor
except (ImportError, AttributeError):
    DTensor = None

# The hard limit of TRITON_MAX_TENSOR_NUMEL is 1048576
# https://github.com/triton-lang/triton/blob/ba42a5c68fd0505f8c42f4202d53be0f8d9a5fe0/python/triton/language/core.py#L19
# However, setting limit as 65536 as in LayerNorm tutorial is faster because of less register spilling
# The optimal maximum block size depends on your hardware, your kernel, and your dtype
MAX_FUSED_SIZE = 65536 // 2
STATIC_WARPS = 32 if not is_amd else 16

@triton.jit
def cross_entropy_kernel(
    logits,
    lse,
    target,
    loss,
    total,
    ignore_index,
    label_smoothing: tl.constexpr,
    logit_scale: tl.constexpr,
    reduction: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr
):
    """
    This kernel computes both cross entropy loss and the gradient of the input.
    We only consider hard label + mean reduction for now.
    Please refer to https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html for the math.

    Args:
        logits:
            Pointer to logits tensor.
        lse:
            Pointer to logsumexp tensor.
        target: Pointer to target tensor.
        loss:
            Pointer to tensor to store the loss.
        V (int):
            The number of columns in the input tensor.
        total (int):
            The number of non-ignored classes.
        ignore_index (int):
            The index to ignore in the target.
        label_smoothing (float):
            The amount of smoothing when computing the loss, where 0.0 means no smoothing.
        reduction (str):
            The string for the reduction to apply
        BV (int):
            The block size for vocab.
    """

    # https://github.com/triton-lang/triton/issues/1058
    # If B*T*V is too large, i_n * stride will overflow out of int32, so we convert to int64
    i_n = tl.program_id(0).to(tl.int64)
    NV = tl.cdiv(V, BV)

    # 1. Load target first because if the target is ignore_index, we can return right away
    b_y = tl.load(target + i_n)

    # 2. locate the start index
    logits += i_n * V

    if b_y == ignore_index:
        # set all x as 0
        for i in range(0, V, BV):
            o_v = i + tl.arange(0, BV)
            tl.store(logits + o_v, 0.0, mask=o_v < V)
        return

    # Online softmax: 2 loads + 1 store (compared with 3 loads + 1 store for the safe softmax)
    # Refer to Algorithm 3 in the paper: https://arxiv.org/pdf/1805.02867

    # 3. [Online softmax] first pass: compute logsumexp
    # we did this in anouter kernel
    b_l = tl.load(logits + b_y) * logit_scale
    b_lse = tl.load(lse + i_n)

    # 4. Calculate the loss
    # loss = lse - logits_l
    b_loss = b_lse - b_l

    # Label smoothing is a general case of normal cross entropy
    # See the full derivation at https://github.com/linkedin/Liger-Kernel/pull/198#issue-2503665310
    b_z = 0.0
    eps = label_smoothing / V

    # We need tl.debug_barrier() as mentioned in
    # https://github.com/triton-lang/triton/blob/ba42a5c68fd0505f8c42f4202d53be0f8d9a5fe0/python/triton/ops/cross_entropy.py#L34
    tl.debug_barrier()

    # 5. [Online Softmax] Second pass: compute gradients
    # For 'mean' reduction, gradients are normalized by number of non-ignored elements
    # dx_y = (softmax(x_y) - 1) / N
    # dx_i = softmax(x_i) / N, i != y
    # For label smoothing:
    # dx_i = (softmax(x_y) - label_smoothing / V) / N, i != y
    # dx_y = (softmax(x_y) - label_smoothing / V - (1 - label_smoothing)) / N
    #      = dx_i - (1 - label_smoothing) / N
    for iv in range(0, NV):
        o_v = iv * BV + tl.arange(0, BV)
        b_logits = tl.load(logits + o_v, mask=o_v < V, other=float('-inf')) * logit_scale
        if label_smoothing > 0:
            # scale X beforehand to avoid overflow
            b_z += tl.sum(tl.where(o_v < V, -eps * b_logits, 0.0))
        b_p = (exp(b_logits - b_lse) - eps) * logit_scale
        if reduction == "mean":
            b_p = b_p / total
        tl.store(logits + o_v, b_p, mask=o_v < V)

        tl.debug_barrier()

    # Orginal loss = H(q, p),  with label smoothing regularization = H(q', p) and (label_smoothing / V) = eps
    # H(q', p) = (1 - label_smoothing) * H(q, p) + label_smoothing * H(u, p)
    #          = (1 - label_smoothing) * H(q, p) + eps * sum(logsoftmax(x_i))
    # By using m (global max of xi) and d (sum of e^(xi-m)), we can simplify as:
    #          = (1 - label_smoothing) * H(q, p) + (-sum(x_i * eps) + label_smoothing * (m + logd))
    # Refer to H(q', p) in section 7 of the paper:
    # https://arxiv.org/pdf/1512.00567
    # pytorch:
    # https://github.com/pytorch/pytorch/blob/2981534f54d49fa3a9755c9b0855e7929c2527f0/aten/src/ATen/native/LossNLL.cpp#L516
    # See full derivation at https://github.com/linkedin/Liger-Kernel/pull/198#issuecomment-2333753087
    if label_smoothing > 0:
        b_loss = b_loss * (1 - label_smoothing) + (b_z + label_smoothing * b_lse)

    # 6. Specially handle the i==y case where `dx_y = (softmax(x_y) - (1 - label_smoothing) / N`
    b_l = tl.load(logits + b_y)

    # Normalize the loss by the number of non-ignored elements if reduction is "mean"
    if reduction == 'mean':
        b_loss = b_loss / total
        b_l += (label_smoothing - 1) / total * logit_scale
    else:
        b_l += (label_smoothing - 1) * logit_scale

    tl.store(loss + i_n, b_loss)
    tl.store(logits + b_y, b_l)


@triton.jit
def elementwise_mul_kernel(
    x,
    g,
    N: tl.constexpr,
    B: tl.constexpr
):
    """
    This function multiplies each element of the tensor pointed by x with the value pointed by g.
    The multiplication is performed in-place on the tensor pointed by x.

    Parameters:
    x:
        Pointer to the input tensor.
    g:
        Pointer to the gradient output value.
    N (int):
        The number of columns in the input tensor.
    B (int):
        The block size for Triton operations.
    """

    # Get the program ID and convert it to int64 to avoid overflow
    i_x = tl.program_id(0).to(tl.int64)
    o_x = i_x * B + tl.arange(0, B)

    # Load the gradient output value
    b_g = tl.load(g)
    b_x = tl.load(x + o_x, mask=o_x < N)
    tl.store(x + o_x, b_x * b_g, mask=o_x < N)


def fused_linear_cross_entropy_forward(
    x: torch.Tensor,
    target: torch.LongTensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    logit_scale: float = 1.0,
    num_chunks: int = 8,
    reduction: str = "mean",
    use_l2warp: bool = False,
    l2_penalty_factor: float = 1e-4
):
    device = x.device
    # inputs have shape: [N, H]
    # materialized activations will have shape: [N, V]
    # the increase in memory = [N, V]
    # reduction can be achieved by partitioning the number of tokens N into smaller chunks.

    # ideally, we would like to achieve the same memory consumption as [N, H],
    # so the expected chunk size should be:
    # NC = ceil(V / H)
    # C = ceil(N / NC)
    # for ex: N = 4096*4, V = 32000, H = 4096 ==> NC = 8, C = ceil(N / NC) = 2048
    N, H, V = *x.shape, weight.shape[0]
    BV = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))
    # TODO: in real cases, we may need to limit the number of chunks NC to
    # ensure the precisions of accumulated gradients
    NC = min(num_chunks, triton.cdiv(V, H))
    C = triton.next_power_of_2(triton.cdiv(N, NC))
    NC = triton.cdiv(N, C)

    # [N, H]
    dx = torch.zeros_like(x, device=device)
    # [V, H]
    dw = torch.zeros_like(weight, device=device, dtype=torch.float) if weight is not None else None
    # [V]
    db = torch.zeros_like(bias, device=device, dtype=torch.float) if bias is not None else None
    # [N]
    loss = torch.zeros(N, device=device, dtype=torch.float)

    total = target.ne(ignore_index).sum().item()
    for ic in range(NC):
        start, end = ic * C, min((ic + 1) * C, N)
        # [C, N]
        c_x = x[start:end]
        # when doing matmul, use the original precision
        # [C, V]
        c_logits = F.linear(c_x, weight, bias)
        c_target = target[start:end]
        # [C]
        # keep lse in fp32 to maintain precision
        c_lse = logsumexp_fwd(c_logits, scale=logit_scale, dtype=torch.float)

        # unreduced loss
        c_loss = loss[start:end]
        if use_l2warp:
            c_maxx, c_ids = torch.max(c_logits, -1, keepdim=True)

        # Here we calculate the gradient of c_logits in place so we can save memory.
        cross_entropy_kernel[(c_logits.shape[0],)](
            logits=c_logits,
            lse=c_lse,
            target=c_target,
            loss=c_loss,
            total=total,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
            logit_scale=logit_scale,
            reduction=reduction,
            V=V,
            BV=BV,
            num_warps=STATIC_WARPS
        )
        if use_l2warp:
            # a. Calculate the L2 gradient w.r.t logits (g_logits_l2)
            g_logits_l2 = torch.zeros_like(c_logits)

            # Normalize factor by B*T, which is the 'total' variable here
            l2_factor = l2_penalty_factor / total if reduction == 'mean' else l2_penalty_factor
            penalty_grad = c_maxx * l2_factor
            g_logits_l2.scatter_(-1, c_ids, penalty_grad)

            # b. Backpropagate g_logits_l2 to get its effect on dx, dw, db
            # and add it to the main gradients.
            # Total_dx = CE_dx + L2_dx
            # Total_dw = CE_dw + L2_dw
            # Total_db = CE_db + L2_db
            if weight is not None:
                dw.add_(g_logits_l2.t() @ c_x)
            if bias is not None:
                db.add_(g_logits_l2.sum(0))
            # The dx contribution must be added to the final dx calculation
            dx_l2_contribution = torch.mm(g_logits_l2, weight)
        else:
            dx_l2_contribution = 0.0

        # gradient of logits is computed in-place by the above triton kernel and is of shape: C x V
        # thus dx should be of shape: C x H
        dx[start:end] = torch.mm(c_logits, weight) + dx_l2_contribution

        # keep dw in fp32 to maintain precision
        if weight is not None:
            dw += c_logits.t() @ c_x

        if bias is not None:
            torch.add(input=db, other=c_logits.sum(0), out=db)

    # loss = loss.mean()
    loss = loss.sum()
    if dw is not None:
        dw = dw.to(weight)
    if db is not None:
        db = db.to(bias)
    return loss, dx, dw, db


def fused_linear_cross_entropy_backward(
    do: torch.Tensor,
    dx: torch.Tensor,
    dw: torch.Tensor,
    db: torch.Tensor
):
    # If cross entropy is the last layer, do is 1.0. Skip the mul to save time
    if torch.ne(do, torch.tensor(1.0, device=do.device)):
        # We use a Triton kernel instead of a PyTorch operation because modifying inputs in-place
        # for gradient storage and backward multiple times causes anomalies with PyTorch but not with Triton.
        N, H = dx.shape
        B = min(MAX_FUSED_SIZE, triton.next_power_of_2(H))

        elementwise_mul_kernel[(triton.cdiv(N * H, B),)](
            x=dx,
            g=do,
            N=N*H,
            B=B,
            num_warps=STATIC_WARPS,
        )

        # handle dw
        if dw is not None:
            V, H = dw.shape
            elementwise_mul_kernel[(triton.cdiv(V * H, B),)](
                x=dw,
                g=do,
                N=V*H,
                B=B,
                num_warps=STATIC_WARPS,
            )

        if db is not None:
            V = db.shape[0]
            elementwise_mul_kernel[(triton.cdiv(V, B),)](
                x=db,
                g=do,
                N=V,
                B=B,
                num_warps=STATIC_WARPS,
            )
    return dx, dw, db


class FusedLinearCrossEntropyFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    def forward(
        ctx,
        x: torch.Tensor,
        target: torch.LongTensor,
        weight: torch.Tensor,
        bias: torch.Tensor = None,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
        logit_scale: float = 1.0,
        num_chunks: int = 8,
        reduction: str = "mean",
        use_l2warp: bool = False,
        l2_penalty_factor: float = 1e-4,
    ):
        """
        Fusing the last linear layer with cross-entropy loss
            Reference: https://github.com/mgmalek/efficient_cross_entropy

        Handle the forward and backward pass of the final linear layer via cross-entropy loss by avoiding
        the materialization of the large logits tensor. Since Cross Entropy Loss is the last layer, we can
        compute the gradient at the forward pass. By doing so, we don't have to store the x and target
        for the backward pass.

        x (torch.Tensor): [batch_size * seq_len, hidden_size]
        target (torch.LongTensor): [batch_size * seq_len]
            where each value is in [0, vocab_size).
        weight (torch.Tensor): [vocab_size, hidden_size]
            where `vocab_size` is the number of classes.
        bias (Optional[torch.Tensor]): [vocab_size]
            where `vocab_size` is the number of classes.
        ignore_index:
            the index to ignore in the target.
        label_smoothing:
            the amount of smoothing when computing the loss, where 0.0 means no smoothing.
        logit_scale: float = 1.0,
            A scaling factor applied to the logits. Default: 1.0
        num_chunks: int
            The number of chunks to split the input tensor into for processing.
            This can help optimize memory usage and computation speed.
            Default: 8
        reduction:
            Specifies the reduction to apply to the output: 'mean' | 'sum'.
            'mean': the weighted mean of the output is taken,
            'sum': the output will be summed.
            Default: 'mean'.
        use_l2warp: bool = False,
            Whether to use L2 regularization on the logits to prevent overconfidence.
            Default: False
        l2_penalty_factor: float = 1e-4,
        """
        loss, dx, dw, db = fused_linear_cross_entropy_forward(
            x,
            target,
            weight,
            bias,
            ignore_index,
            label_smoothing,
            logit_scale,
            num_chunks,
            reduction,
            use_l2warp,
            l2_penalty_factor
        )
        # downcast to dtype and store for backward
        ctx.save_for_backward(
            dx.detach(),
            dw.detach() if weight is not None else None,
            db.detach() if bias is not None else None,
        )
        return loss

    @staticmethod
    @input_guard
    def backward(ctx, do):
        dx, dw, db = ctx.saved_tensors
        dx, dw, db = fused_linear_cross_entropy_backward(do, dx, dw, db)
        return dx, None, dw, db, None, None, None, None, None, None, None


def fused_linear_cross_entropy_loss(
    x: torch.Tensor,
    target: torch.LongTensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    logit_scale: float = 1.0,
    num_chunks: int = 8,
    reduction: str = "mean",
    use_l2warp: bool = False,
    l2_penalty_factor: float = 1e-4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        x (torch.Tensor): [batch_size * seq_len, hidden_size]
        target (torch.LongTensor): [batch_size * seq_len]
            where each value is in [0, vocab_size).
        weight (torch.Tensor): [vocab_size, hidden_size]
            where `vocab_size` is the number of classes.
        bias (Optional[torch.Tensor]): [vocab_size]
            where `vocab_size` is the number of classes.
        ignore_index: int.
            If target == ignore_index, the loss is set to 0.0.
        label_smoothing: float
        logit_scale: float
            A scaling factor applied to the logits. Default: 1.0
        num_chunks: int
            The number of chunks to split the input tensor into for processing.
            This can help optimize memory usage and computation speed.
            Default: 8
        reduction:
            Specifies the reduction to apply to the output: 'mean' | 'sum'.
            'mean': the weighted mean of the output is taken,
            'sum': the output will be summed.
            Default: 'mean'.
    Returns:
        losses: [batch,], float
    """
    return FusedLinearCrossEntropyFunction.apply(
        x,
        target,
        weight,
        bias,
        ignore_index,
        label_smoothing,
        logit_scale,
        num_chunks,
        reduction,
        use_l2warp,
        l2_penalty_factor
    )


class FusedLinearCrossEntropyLoss(nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int, 
        config: ModelParallelConfig,
        init_method: Callable,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
        logit_scale: float = 1.0,
        num_chunks: int = 8,
        reduction: str = "mean",
        use_l2warp: bool = False,
        l2_penalty_factor: float = 1e-4,
    ):
        """
        Args:
            ignore_index: int.
                If target == ignore_index, the loss is set to 0.0.
            label_smoothing: float
            logit_scale: float
                A scaling factor applied to the logits. Default: 1.0
            num_chunks: int
                The number of chunks to split the input tensor into for processing.
                This can help optimize memory usage and computation speed.
                Default: 8
            reduction:
                Specifies the reduction to apply to the output: 'mean' | 'sum'.
                'mean': the weighted mean of the output is taken,
                'sum': the output will be summed.
                Default: 'mean'.
        """
        super().__init__()

        assert reduction in ["mean", "sum"], f"reduction: {reduction} is not supported"

        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.logit_scale = logit_scale
        self.num_chunks = num_chunks
        self.reduction = reduction
        self.use_l2warp = use_l2warp
        self.l2_penalty_factor = l2_penalty_factor

        # 假设没开tp
        self.weight = Parameter(
                torch.empty(
                    output_size,
                    input_size,
                    device=torch.cuda.current_device(),
                    dtype=config.params_dtype,
                )
            )
        _initialize_affine_weight_gpu(
            self.weight,
            init_method,
            partition_dim=0,
            stride=1,
            is_expert=False,
        )


    @torch.compiler.disable
    def forward(
        self,
        x: torch.Tensor,
        target: torch.LongTensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None
    ):
        """
        Args:
            x (torch.Tensor): [batch_size, seq_len, hidden_size]
            target (torch.LongTensor): [batch_size, seq_len]
                where each value is in [0, V).
            weight (torch.Tensor): [vocab_size, hidden_size]
                where `vocab_size` is the number of classes.
            bias (Optional[torch.Tensor]): [vocab_size]
                where `vocab_size` is the number of classes.
        Returns:
            loss
        """

        if weight is None:
            if self.weight is None:
                raise RuntimeError(
                    "weight was not supplied to ColumnParallelLinear forward pass "
                    "and skip_weight_param_allocation is True."
                )
            weight = self.weight

        loss = fused_linear_cross_entropy_loss(
            x.view(-1, x.shape[-1]),
            target.view(-1),
            weight=weight,
            bias=bias,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
            logit_scale=self.logit_scale,
            num_chunks=self.num_chunks,
            reduction=self.reduction,
            use_l2warp=self.use_l2warp,
            l2_penalty_factor=self.l2_penalty_factor
        )
        return loss

class LinearLossParallel(ParallelStyle):
    def __init__(
        self,
        *,
        sequence_dim: int = 1,
        use_local_output: bool = False,
    ):
        super().__init__()

        self.sequence_sharding = (Shard(sequence_dim),)
        self.use_local_output = use_local_output

    @staticmethod
    def _prepare_input_fn(sequence_sharding, mod, inputs, device_mesh):
        x, target, weight, bias = inputs

        if not isinstance(x, DTensor):
            # assume the input passed in already sharded on the sequence dim and create the DTensor
            x = DTensor.from_local(x, device_mesh, sequence_sharding)
        if x.placements != sequence_sharding:
            x = x.redistribute(placements=sequence_sharding, async_op=True)
        if not isinstance(target, DTensor):
            target = DTensor.from_local(target, device_mesh, [Replicate()])
        if target.placements != sequence_sharding:
            target = target.redistribute(placements=sequence_sharding, async_op=True)

        if not isinstance(weight, DTensor):
            weight = DTensor.from_local(weight, device_mesh, [Replicate()])
        if weight.placements != [Replicate()]:
            # we replicate the weight/bias in FLCE
            weight = weight.redistribute(placements=[Replicate()], async_op=True)

        if bias is not None and not isinstance(bias, DTensor):
            bias = DTensor.from_local(bias, device_mesh, [Replicate()])
        if bias is not None and bias.placements != [Replicate()]:
            bias = bias.redistribute(placements=[Replicate()], async_op=True)

        return x.to_local(), target.to_local(), weight.to_local(), bias.to_local() if bias is not None else bias

    @staticmethod
    def _prepare_output_fn(use_local_output, mod, outputs, device_mesh):
        return outputs.to_local() if use_local_output else outputs

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            partition_fn=None,
            input_fn=partial(self._prepare_input_fn, self.sequence_sharding),
            output_fn=partial(self._prepare_output_fn, self.use_local_output)
        )
