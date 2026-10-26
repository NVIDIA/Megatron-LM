import torch
import tilelang
import tilelang.language as T

import functools
from typing import Any, Callable, Dict, Optional, Tuple

from megatron.core.transformer.experimental_attention_variant.fused_loss import fused_qk_topk as tl_topk


def _is_equal(a, b):
    if isinstance(a, torch.Tensor):
        return a is b
    # Whitelist of types that are safe to compare by value for caching.
    if isinstance(a, (int, float, str, bool, type(None))) and isinstance(b, (int, float, str, bool, type(None))):
        return a == b
    # For other types, we cannot guarantee a cheap and safe comparison, so we fail the cache check.
    return False


def tensor_cache(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    """
    A decorator that caches the most recent result of a function with tensor inputs.

    This decorator will store the output of the decorated function for the most recent set of input tensors.
    If the function is called again with the same input tensors, it will return the cached result.


    Args:
        fn (Callable[..., torch.Tensor]):
            The function to be decorated. It should take tensor inputs and return tensor outputs.

    Returns:
        Callable[..., torch.Tensor]:
            A wrapped version of the input function with single-entry caching.
    """
    last_args: Optional[Tuple] = None
    last_kwargs: Optional[Dict] = None
    last_result: Any = None

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        nonlocal last_args, last_kwargs, last_result

        if last_args is not None and last_kwargs is not None:
            if len(args) == len(last_args) and len(kwargs) == len(last_kwargs):
                # For Tensors, check for object identity. For other types, check for equality.
                # Python caches small integers, so `is` works for them but not for large integers like 4096.
                if (
                    all(_is_equal(a, b) for a, b in zip(args, last_args))
                    and set(kwargs.keys()) == set(last_kwargs.keys())
                    and all(_is_equal(v, last_kwargs[k]) for k, v in kwargs.items())
                ):
                    return last_result

        result = fn(*args, **kwargs)
        last_args, last_kwargs, last_result = args, kwargs, result
        return result

    return wrapper


@tensor_cache
def cal_seq_idx_from_cu_seqlens(cu_seqlens: torch.LongTensor, seq_len: int):
    seq_idx = cu_seqlens.new_zeros(seq_len + 1)
    seq_idx.scatter_add_(0, cu_seqlens[1:].long(), torch.ones_like(seq_idx))
    seq_idx.cumsum_(0)
    return seq_idx[:-1]


@tensor_cache
def cal_seq_idx_for_q(cu_seqlens_qs: torch.LongTensor, cu_seqlens_qe: torch.LongTensor, seq_len: int) -> torch.IntTensor:
    seq_idx_for_q = torch.full((seq_len,), len(cu_seqlens_qs), dtype=torch.int32, device=cu_seqlens_qs.device)
    for i in range(len(cu_seqlens_qs)):
        seq_idx_for_q[cu_seqlens_qs[i] : cu_seqlens_qe[i]] = i
    return seq_idx_for_q


@tensor_cache
def cal_cu_seqlen_ks_for_q(
    cu_seqlens_qs: torch.LongTensor, cu_seqlens_qe: torch.LongTensor, cu_seqlens_ks: torch.LongTensor, seq_len: int
) -> torch.IntTensor:
    cu_seqlen_ks_for_each_q = torch.gather(
        input=torch.cat([cu_seqlens_ks, torch.full((1,), torch.iinfo(torch.int32).max, dtype=torch.int32, device=cu_seqlens_qs.device)]),
        dim=0,
        index=cal_seq_idx_for_q(cu_seqlens_qs=cu_seqlens_qs, cu_seqlens_qe=cu_seqlens_qe, seq_len=seq_len).long(),
    )
    return cu_seqlen_ks_for_each_q.int()


@tensor_cache
def cal_cu_seqlen_ke_for_q(
    cu_seqlens_qs: torch.LongTensor,
    cu_seqlens_qe: torch.LongTensor,
    cu_seqlens_ks: torch.LongTensor,
    cu_seqlens_ke: torch.LongTensor,
    q_start_idxs: torch.LongTensor,
    seq_len: int,
    kv_stride: int,
) -> torch.IntTensor:
    cu_seqlen_ke_for_each_q = torch.gather(
        input=torch.cat([cu_seqlens_ke, torch.zeros(1, dtype=torch.int32, device=cu_seqlens_qs.device)]),
        dim=0,
        index=cal_seq_idx_for_q(cu_seqlens_qs=cu_seqlens_qs, cu_seqlens_qe=cu_seqlens_qe, seq_len=seq_len).long(),
    )
    casual_cu_seqlen_ke_for_each_q = torch.zeros((seq_len,), dtype=torch.int32, device=cu_seqlens_qs.device)
    for i in range(len(cu_seqlens_qs)):
        casual_cu_seqlen_ke_for_each_q[cu_seqlens_qs[i] : cu_seqlens_qe[i]] = (
            torch.arange(
                q_start_idxs[i], q_start_idxs[i] + cu_seqlens_qe[i] - cu_seqlens_qs[i], dtype=torch.int32, device=cu_seqlens_qs.device
            )
            + 1
        ) // kv_stride + cu_seqlens_ks[i]
    cu_seqlen_ke_for_each_q = torch.minimum(casual_cu_seqlen_ke_for_each_q, cu_seqlen_ke_for_each_q)
    return cu_seqlen_ke_for_each_q.int()


@tensor_cache
def cal_ks_ke_from_cu_seqlen_qk(
    cu_seqlens_q: torch.LongTensor,
    cu_seqlens_k: torch.LongTensor = None,
    offs_q: torch.LongTensor = None,
    *,
    seq_len: int,
    kv_stride: int = 1,
    cp_rank: int = 0,
    cp_size: int = 1,
    balanced_cp=False,
):
    """
    seq_len: seq len per cp rank
    balanced cp slice assignment: 0 1 2 3 3 2 1 0
    """
    n_seq = len(cu_seqlens_q) - 1
    assert n_seq > 0
    assert cu_seqlens_q.shape == (n_seq + 1,)
    seq_idx = cal_seq_idx_from_cu_seqlens(cu_seqlens_q.long(), seq_len * cp_size)
    qs = cu_seqlens_q.gather(0, seq_idx)
    pos = torch.arange(len(qs), dtype=qs.dtype, device=qs.device) - qs
    if offs_q is not None:
        assert offs_q.shape == (n_seq,), offs_q.shape
        qoff = offs_q.gather(0, seq_idx)
        pos += qoff
    if cu_seqlens_k is None or cu_seqlens_k is cu_seqlens_q:
        ks = qs
    else:
        assert cu_seqlens_k.shape == (n_seq + 1,)
        ks = cu_seqlens_k.gather(0, seq_idx)
    ke = ks + (pos + 1) // kv_stride

    if cp_size == 1:
        pass
    elif balanced_cp:
        assert cp_size % 2 == 0, cp_size

        def f(x: torch.Tensor):
            chunks = x.chunk(cp_size * 2)
            return torch.cat(
                [
                    chunks[cp_rank],
                    chunks[cp_size - cp_rank - 1],
                ]
            )

        ks = f(ks)
        ke = f(ke)
    else:
        ks = ks.chunk(cp_size)[cp_rank]
        ke = ke.chunk(cp_size)[cp_rank]

    return ks, ke


def ceil_to_ue8m0(x: torch.Tensor):
    assert x.view(-1).amax().item() > 0
    return torch.pow(2.0, torch.ceil(torch.log2(x.abs())))

def per_custom_dims_cast_to_fp8(x: torch.Tensor, dims: Tuple[int], use_ue8m0: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    excluded_dims = tuple([i for i in range(x.dim()) if i not in set(dims)])
    x_amax = x.abs().float().amax(dim=excluded_dims, keepdim=True).clamp(1e-4)
    sf = x_amax / 448.0
    sf = ceil_to_ue8m0(sf) if use_ue8m0 else sf
    x_scaled = (x * (1.0 / sf)).to(torch.float8_e4m3fn)
    return x_scaled, sf.squeeze()






def ref_fp8_mqa_logits(q: torch.Tensor, kv: torch.Tensor, weights: torch.Tensor, cu_seqlen_ks: torch.Tensor, cu_seqlen_ke: torch.Tensor, topk: int):
    k = kv
    q = q.float()
    k = k.float()

    seq_len_kv = kv.shape[0]
    mask_lo = torch.arange(0, seq_len_kv, device="cuda")[None, :] >= cu_seqlen_ks[:, None]
    mask_hi = torch.arange(0, seq_len_kv, device="cuda")[None, :] < cu_seqlen_ke[:, None]
    mask = mask_lo & mask_hi

    score = torch.einsum("mhd,nd->hmn", q, k)
    logits = (score.relu() * weights.unsqueeze(-1).transpose(0, 1)).sum(dim=0)
    logits = logits.masked_fill(~mask, float("-inf"))

    cost = mask.sum()
    topk_logits, topk_indices = logits.topk(topk)
    return logits, cost, topk_indices, topk_logits

def display_error_message(msg):
    print(f"\033[31mWARNING: {msg}\033[0m")

def compute_correlation(a, b, label="tensor"):
    a, b = a.data.double(), b.data.double()
    norm_sum = (a * a + b * b).sum()
    if norm_sum == 0:
        display_error_message(f"{label} all zero")
        return 1
    correlation = 2 * (a * b).sum() / norm_sum
    return correlation

def validate_tensor_match(a, b, tolerance=1e-8, tensor_name="tensor", should_raise=True):
    a_finite = torch.isfinite(a)
    b_finite = torch.isfinite(b)
    if not torch.all(a_finite == b_finite):
        display_error_message(f"{tensor_name} Error: isfinite mask mismatch")
        if should_raise:
            assert False
    if not torch.isclose(
        a.masked_fill(a_finite, 0),
        b.masked_fill(b_finite, 0),
        rtol=0,
        atol=0,
        equal_nan=True,
    ).all():
        display_error_message(f"{tensor_name} Error: nonfinite value mismatch")
        if should_raise:
            assert False
    a = a.masked_fill(~a_finite, 0)
    b = b.masked_fill(~b_finite, 0)
    correlation = compute_correlation(a, b, tensor_name)
    difference = 1.0 - correlation
    if not (0 <= difference <= tolerance):
        display_error_message(f"{tensor_name} Error: {difference}")
        if should_raise:
            assert False
    return difference

pass_configs = {
    tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True, # logits computation needs
}


def convert_to_uint16(x):
    hval = T.Cast(T.float16, x)
    bits_uint = T.reinterpret(T.uint16, hval)
    bits_uint = T.if_then_else(x < 0, ~bits_uint & (0xFFFF), bits_uint | (0x8000))
    return bits_uint >> 8


def convert_to_uint32(x):
    bits_uint = T.reinterpret(T.uint32, x)
    bits_uint = T.if_then_else(
        x < 0,
        ~bits_uint & T.Cast(T.uint32, (0xFFFFFFFF)),
        bits_uint | T.Cast(T.uint32, (0x80000000)),
    )
    return bits_uint


def test_topk_selector(S=4096, SKV=16384, H=32, HKV=1, D=64, kv_stride=1, topk=2048, test_accuracy=True):
    torch.manual_seed(0)
    input = torch.randn(S, SKV, dtype=torch.float32).cuda()
    starts = torch.zeros(S, dtype=torch.int32).cuda()
    ends = torch.ones(S, dtype=torch.int32).cuda() * SKV

    q = torch.randn(S, H, D, device="cuda", dtype=torch.bfloat16).to(torch.bfloat16)
    kv = torch.randn(SKV, D, device="cuda", dtype=torch.bfloat16).to(torch.bfloat16)
    weights = torch.randn(S, H, device="cuda", dtype=torch.float32)
    p = (torch.randn(S, SKV, device="cuda", dtype=torch.float32) * 4).softmax(dim=-1)

    # TODO: use mask
    # ks, ke = generate_random_cu_seqlens(per_cp_seqlen=S, cp_size=4, cp_rank=3, kv_stride=kv_stride, average_q_len=2048)
    ks, ke = starts, ends

    print(f"{ks=}, {ke=}")

    q_fp8 = q.to(torch.float8_e4m3fn)
    kv_fp8, kv_scales = per_custom_dims_cast_to_fp8(kv, (0,), False)

    # topk_indexes, topk_logits, original_logits = tl_topk(
    #     q=q_fp8, kv=kv_fp8, weights=weights, cu_seqlen_ks=ks, cu_seqlen_ke=ke,
    #     starts=starts, ends=ends, topk=topk,
    #     debug=True, kv_scales=kv_scales, 
    # )
    topk_indexes, topk_logits, original_logits = tl_topk(
        q=q, kv=kv, weights=weights, cu_seqlen_ks=ks, cu_seqlen_ke=ke,
        starts=starts, ends=ends, topk=topk,
        debug=True, 
    )

    torch.cuda.synchronize()

    logits_ref, cost_ref, topk_indices_ref, topk_logits_ref = ref_fp8_mqa_logits(q=q, kv=kv, weights=weights, cu_seqlen_ks=ks, cu_seqlen_ke=ke, topk=topk)

    torch.cuda.synchronize()
    
    print(topk_indexes)

    print(topk_indices_ref)

    # # torch.testing.assert_close(original_logits, logits_ref)
    # acc_diff = validate_tensor_match(original_logits, logits_ref, tolerance=1e-14, tensor_name="logits", should_raise=False)
    # print(f"accuracy difference: {acc_diff}")

    # indexes_ref = fast_topk(input, topk)
    # print(indexes_ref)

    # Calculate intersection of out_ref and out_trt
    if test_accuracy:
        for i in range(S):
            ref_np = topk_indices_ref[i][torch.isfinite(topk_logits_ref[i])].cpu().to(torch.int32).numpy()

            trt_np = topk_indexes[i][( topk_indexes[i] >= 0) & (topk_indexes[i] < SKV)].cpu().to(torch.int32).numpy()

            # gt_len = (ke[i] - ks[i]).item()
            gt_len = ref_np.shape[0]

            set_ref = set(ref_np)
            set_trt = set(trt_np)
            intersection = set_ref & set_trt
            # print(set_ref - set_trt)
            print("selected/all:", len(intersection), "/", gt_len, "=", len(intersection) / gt_len)

    # Performance test with CUDA events

    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # Warmup
    for _ in range(5):
        _ = tl_topk(
            q=q_fp8, kv=kv_fp8, kv_scales=kv_scales, weights=weights, cu_seqlen_ks=ks, cu_seqlen_ke=ke,
            input=logits_ref, starts=starts, ends=ends, topk=topk
        )
        _ = ref_fp8_mqa_logits(q=q, kv=kv, weights=weights, cu_seqlen_ks=ks, cu_seqlen_ke=ke, topk=topk)
    torch.cuda.synchronize()

    n_iters = 20
    start_event.record()
    for _ in range(n_iters):
        _ = tl_topk(
            q=q_fp8, kv=kv_fp8, kv_scales=kv_scales, weights=weights, cu_seqlen_ks=ks, cu_seqlen_ke=ke,
            input=logits_ref, starts=starts, ends=ends, topk=topk
        )
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(f"Average tl_topk time: {elapsed_time_ms / n_iters:.3f} ms")

    # Torch topk time
    start_event.record()
    for _ in range(n_iters):
        # _ = torch.topk(input, topk, dim=-1)[1]
        _ = ref_fp8_mqa_logits(q=q, kv=kv, weights=weights, cu_seqlen_ks=ks, cu_seqlen_ke=ke, topk=topk)
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(f"Average torch.topk time: {elapsed_time_ms / n_iters:.3f} ms")


def run_regression_perf(batch=64, seq_len=32 * 1024, topk=2048):
    batch = 64
    seq_len = 32 * 1024
    topk = 2048
    torch.manual_seed(1)
    input = torch.randn(batch, seq_len, dtype=torch.float32).cuda()
    starts = torch.zeros(batch, dtype=torch.int32).cuda()
    ends = torch.ones(batch, dtype=torch.int32).cuda() * seq_len

    indexes = tl_topk(input, starts, ends, topk)

    indexes_ref = torch.topk(input, topk, dim=-1)[1]

    for i in range(batch):
        ref_np = indexes_ref[i].cpu().to(torch.int32).numpy()
        trt_np = indexes[i].cpu().to(torch.int32).numpy()

        set_ref = set(ref_np)
        set_trt = set(trt_np)
        intersection = set_ref & set_trt
        print("selected/all:", len(intersection), "/", len(set_ref), "=", len(intersection) / len(set_ref))

    from tilelang.profiler import do_bench

    def run_kernel_only():
        tl_topk(input, starts, ends, topk)

    return do_bench(run_kernel_only, warmup=10, rep=100, backend="cupti")


if __name__ == "__main__":
    test_topk_selector(S=2048+256, SKV=2048+256)
    # test_topk_selector(S=16384, SKV=16384)