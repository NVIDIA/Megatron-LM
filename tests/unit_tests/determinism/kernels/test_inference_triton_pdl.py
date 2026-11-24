# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Bit-exact check of the inference Triton kernels launched with programmatic dependent launch.

A kernel launched with PDL may start while the previous kernel on its stream is still running,
so it is correct only if every block executes ``griddepcontrol.wait`` before it touches memory.
Each kernel that opts in through ``megatron.core.inference.triton_pdl`` runs here after a
producer that lets its dependents launch at once and writes the kernel's input only after a long
spin, with the kernel's outputs filled with garbage beforehand, and its result is compared byte
for byte with the kernel launched without PDL on the same data, both eagerly and inside a CUDA
graph, under torch.inference_mode(). A kernel that read its input or wrote its output before
waiting would see the garbage or be overwritten by the producer. The NVLS symmetric-memory
collectives are checked the same way on every rank of a launch with two or more ranks (skipped
otherwise).
"""

import importlib
from typing import Callable, Optional, Sequence

import pytest
import torch

from megatron.core.inference import triton_pdl
from megatron.core.transformer import streamwise_residual_ops
from megatron.core.tuning import policy as tuning_policy
from tests.unit_tests.determinism.kernels.harness import bytes_equal, seeded

try:
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import gdc_launch_dependents

    HAVE_PDL_PRODUCER = True
except ImportError:
    HAVE_PDL_PRODUCER = False

pytestmark = pytest.mark.skipif(
    not (
        HAVE_PDL_PRODUCER
        and triton_pdl.HAVE_PDL
        and torch.cuda.is_available()
        and torch.version.hip is None
        and torch.cuda.get_device_capability()[0] >= 9
    ),
    reason="needs an NVIDIA GPU with programmatic dependent launch and Triton's gdc intrinsics",
)

# Spin iterations of the producer before it stores (about 0.2 ms per MB on current GPUs).
PRODUCER_SPIN = 4000
_NEVER = -123456789

if HAVE_PDL_PRODUCER:

    @triton.jit
    def _late_copy_kernel(src, dst, n, n_spin, never, BLOCK: tl.constexpr):
        gdc_launch_dependents()  # dependents may launch now, long before dst is written
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n
        x = tl.load(src + offsets, mask=mask, other=0)
        acc = x.to(tl.int32)
        for _ in tl.range(0, n_spin):
            acc = acc * 1103515245 + 12345
        # The store depends on the spin result (``never`` is never hit), so it stays after it.
        tl.store(dst + offsets, tl.where(acc == never, x ^ 1, x), mask=mask)


def _late_copy(src: torch.Tensor, dst: torch.Tensor) -> None:
    src_bytes, dst_bytes = src.view(torch.uint8).view(-1), dst.view(torch.uint8).view(-1)
    n = src_bytes.numel()
    _late_copy_kernel[(triton.cdiv(n, 1024),)](
        src_bytes, dst_bytes, n, PRODUCER_SPIN, _NEVER, BLOCK=1024
    )


def _fill_garbage(tensors: Sequence[torch.Tensor]) -> None:
    for tensor in tensors:
        tensor.view(torch.uint8).fill_(0xA5)


def _assert_same(outputs, references, what: str) -> None:
    torch.cuda.synchronize()
    assert len(outputs) == len(references), what
    for index, (out, ref) in enumerate(zip(outputs, references)):
        assert bytes_equal(out, ref), f"{what}: output {index} differs from the non-PDL launch"


def assert_pdl_launch_bit_exact(
    monkeypatch: pytest.MonkeyPatch,
    call: Callable[[], Sequence[torch.Tensor]],
    root: torch.Tensor,
    make_input: Callable[[int], torch.Tensor],
    outputs: Sequence[torch.Tensor] = (),
    reset: Optional[Callable[[], None]] = None,
    replays: int = 3,
    what: str = "kernel",
    group: Optional[torch.distributed.ProcessGroup] = None,
) -> None:
    """Assert ``call`` gives the same bytes launched with PDL after a late producer as without PDL.

    Args:
        monkeypatch: the test's monkeypatch fixture; it switches PDL off and on.
        call: launches the kernel(s) under test and returns the tensors to compare.
        root: the input buffer ``call`` reads; the producer writes ``make_input(i)`` into it.
        make_input: returns the input data of replay ``i``.
        outputs: buffers ``call`` writes in place; filled with garbage before every run.
        reset: restores in-place state (e.g. a recurrent state) before every run.
        replays: runs per mode (reference, eager, CUDA graph).
        what: label for failure messages.
        group: for a collective, its process group. Every rank runs the check in step, and the
            ranks barrier before and after each reset, so that no peer reads or writes this
            rank's buffers while they are reset.
    """
    reset = reset or (lambda: None)
    staging = torch.empty_like(root)

    def barrier():
        if group is not None:
            torch.cuda.synchronize()
            torch.distributed.barrier(group)

    def prepare(i: int, late: bool) -> None:
        barrier()
        staging.copy_(make_input(i))
        _fill_garbage((root, *outputs))
        reset()
        if not late:
            root.copy_(staging)
        barrier()

    with torch.inference_mode():  # PDL applies only under inference mode
        monkeypatch.setattr(triton_pdl, "_enabled", False)
        references = []
        for i in range(replays):
            prepare(i, late=False)
            references.append([out.clone() for out in call()])
        torch.cuda.synchronize()

        monkeypatch.setattr(triton_pdl, "_enabled", True)
        for i in range(replays):
            prepare(i, late=True)
            _late_copy(staging, root)
            _assert_same(call(), references[i], f"{what} (eager, replay {i})")

        prepare(0, late=True)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):  # compile and warm up outside the capture
            _late_copy(staging, root)
            call()
        torch.cuda.current_stream().wait_stream(stream)
        prepare(0, late=True)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _late_copy(staging, root)
            graph_outputs = call()
        for i in range(replays):
            prepare(i, late=True)
            graph.replay()
            _assert_same(graph_outputs, references[i], f"{what} (CUDA graph, replay {i})")


def seeded_randn(like: torch.Tensor, seed: int, scale: float = 1.0) -> torch.Tensor:
    """Random normal data with the shape and dtype of ``like``, from a seeded generator."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    data = torch.randn(like.shape, generator=generator, device="cuda", dtype=torch.float32)
    return (data * scale).to(like.dtype)


def _dev_scalar(value, dtype=torch.int32):
    return torch.tensor([value], dtype=dtype, device="cuda")


# --- Inference context and MoE kernels -------------------------------------------------------


def _zero_padding_rows():
    from megatron.core.inference.contexts.zero_padding_rows_kernel import zero_padding_rows

    rows = torch.empty(256, 2688, device="cuda", dtype=torch.bfloat16)
    count = _dev_scalar(186)

    def call():
        zero_padding_rows(rows, count)
        return [rows]

    return dict(call=call, root=rows, make_input=lambda i: seeded_randn(rows, i))


def _mask_routing_padding():
    from megatron.core.transformer.moe.inference_routing_mask_kernel import mask_routing_padding

    routing_map = torch.empty(256, 22, device="cuda", dtype=torch.int64)
    count = _dev_scalar(186)

    def make_input(i):
        generator = torch.Generator(device="cuda").manual_seed(i)
        return torch.randint(0, 512, routing_map.shape, generator=generator, device="cuda")

    def call():
        mask_routing_padding(routing_map, count)
        return [routing_map]

    return dict(call=call, root=routing_map, make_input=make_input)


def _sigmoid_topk_routing():
    from megatron.core.inference.moe.sigmoid_topk import sigmoid_topk_routing

    logits = torch.empty(744, 512, device="cuda", dtype=torch.float32)
    bias = seeded_randn(torch.empty(512, device="cuda"), 99, 0.1)

    def call():
        return list(sigmoid_topk_routing(logits, bias, 22, scaling_factor=2.5))

    return dict(call=call, root=logits, make_input=lambda i: seeded_randn(logits, i))


def _mxfp8_quantize_into():
    from megatron.core.inference.quantization.mxfp8_quantize import mxfp8_quantize_into

    # 744 rows and 21 scale columns: the kernel also zeroes padded scale rows and columns.
    rows, cols = 744, 672
    x = torch.empty(rows, cols, device="cuda", dtype=torch.bfloat16)
    data = torch.empty(rows, cols, device="cuda", dtype=torch.float8_e4m3fn)
    # Swizzled scale layout padded to 128-row tiles; the kernel writes the padding itself.
    scale = torch.empty(
        triton.cdiv(rows, 128) * 128 * triton.cdiv(cols, 128) * 4, device="cuda", dtype=torch.uint8
    )

    def call():
        mxfp8_quantize_into(x, data, scale)
        return [data, scale]

    return dict(
        call=call, root=x, make_input=lambda i: seeded_randn(x, i, 3.0), outputs=(data, scale)
    )


def _mxfp8_quantize_rows():
    from megatron.core.inference.quantization import mxfp8_quantize as q

    # The public mxfp8_quantize_rows allocates its scales with torch.zeros, a kernel that would
    # order the producer before the rows kernel and hide a missing wait, so launch the kernel
    # directly; reset zeroes the scales before the producer runs.
    rows, cols, live_rows = 2976, 672, 744
    x = torch.empty(rows, cols, device="cuda", dtype=torch.bfloat16)
    root = x[:live_rows]  # rows at or beyond the live count are never read
    valid_rows = _dev_scalar(live_rows)
    data = torch.empty(rows, cols, device="cuda", dtype=torch.float8_e4m3fn)
    scale_cols = cols // q.MXFP8_BLOCK_SIZE
    n_col_blocks = triton.cdiv(scale_cols, q.MXFP8_SCALE_COL_BLOCK)
    scale = torch.empty(
        triton.cdiv(rows, q.MXFP8_SCALE_ROW_BLOCK)
        * q.MXFP8_SCALE_ROW_BLOCK
        * n_col_blocks
        * q.MXFP8_SCALE_COL_BLOCK,
        device="cuda",
        dtype=torch.uint8,
    )
    block_k = triton.next_power_of_2(cols)
    num_programs = min(rows, 4096)

    def call():
        q._mxfp8_quant_swizzle_rows_kernel[(num_programs,)](
            data,
            scale,
            x,
            valid_rows,
            cols,
            n_col_blocks,
            REAL_GROUPS=scale_cols,
            BLOCK_K=block_k,
            BLOCK_GROUPS=block_k // q.MXFP8_BLOCK_SIZE,
            NUM_PROGRAMS=num_programs,
            **triton_pdl.pdl_kwargs(),
        )
        # Data rows at or beyond the live count stay uninitialized; all scales are compared.
        return [data[:live_rows], scale]

    return dict(
        call=call,
        root=root,
        make_input=lambda i: seeded_randn(root, i, 3.0),
        outputs=(data,),
        reset=scale.zero_,
    )


def _append_key_value_cache():
    from megatron.core.inference.contexts.fused_kv_append_kernel import (
        triton_append_key_value_cache,
    )

    tokens, heads, head_dim = 186, 2, 128
    memory = torch.empty(2, 2, 4, 256, heads, head_dim, device="cuda", dtype=torch.bfloat16)
    key = torch.empty(tokens, 1, heads, head_dim, device="cuda", dtype=torch.bfloat16)
    value = seeded_randn(key, 31)
    token = torch.arange(tokens, device="cuda")
    block, position = (token // 64).to(torch.int64), (token % 64 * 3).to(torch.int32)

    def call():
        triton_append_key_value_cache(1, key, value, memory, tokens, block, position, -1)
        return [memory]

    return dict(call=call, root=key, make_input=lambda i: seeded_randn(key, i), reset=memory.zero_)


# --- SSM decode kernels ------------------------------------------------------------------------


def _causal_conv1d_update():
    from megatron.core.ssm.ops.common.causal_conv1d_triton import causal_conv1d_update

    batch, dim, width = 186, 2048, 4
    x = torch.empty(batch, dim, device="cuda", dtype=torch.bfloat16)
    initial_state = seeded_randn(torch.empty(batch, dim, width - 1, dtype=torch.bfloat16), 7)
    state = torch.empty_like(initial_state)
    weight = seeded_randn(torch.empty(dim, width, dtype=torch.bfloat16), 8, 0.5)
    bias = seeded_randn(torch.empty(dim, dtype=torch.bfloat16), 9, 0.1)

    def call():
        return [causal_conv1d_update(x, state, weight, bias, True, None), state]

    return dict(
        call=call,
        root=x,
        make_input=lambda i: seeded_randn(x, i),
        reset=lambda: state.copy_(initial_state),
    )


def _fused_recurrent_update():
    from megatron.core.ssm.ops.gdp.fused_recurrent import fused_recurrent_gated_delta_rule_update

    batch, heads, value_heads, key_dim, value_dim = 64, 4, 8, 128, 128
    generator = torch.Generator(device="cuda").manual_seed(13)
    q = torch.empty(batch, 1, heads, key_dim, device="cuda", dtype=torch.bfloat16)
    k = seeded_randn(q, 11)
    v = seeded_randn(torch.empty(batch, 1, value_heads, value_dim, dtype=torch.bfloat16), 12)
    g = -torch.rand(batch, 1, value_heads, generator=generator, device="cuda")
    beta = torch.rand(batch, 1, value_heads, generator=generator, device="cuda").bfloat16()
    initial_state = seeded_randn(
        torch.empty(80, value_heads, key_dim, value_dim, dtype=torch.float32), 15, 0.1
    )
    state = torch.empty_like(initial_state)
    state_indices = torch.arange(batch, device="cuda", dtype=torch.int32)

    def call():
        out = fused_recurrent_gated_delta_rule_update(
            q,
            k,
            v,
            g=g,
            beta=beta,
            use_qk_l2norm_in_kernel=True,
            state=state,
            state_indices=state_indices,
        )[0]
        return [out, state]

    return dict(
        call=call,
        root=q,
        make_input=lambda i: seeded_randn(q, i),
        reset=lambda: state.copy_(initial_state),
    )


def _gdp_decode_prepare():
    from megatron.core.ssm.ops.gdp.prepare import gdp_decode_prepare

    householder, heads, groups, head_dim, state_dim, tokens = 2, 8, 2, 64, 64, 186
    conv_dim = heads * head_dim * householder + (householder + 1) * groups * state_dim
    packed = torch.empty(
        tokens, 1, conv_dim + (householder + 1) * heads, device="cuda", dtype=torch.bfloat16
    )
    a_log = seeded_randn(torch.empty(heads, dtype=torch.float32), 21)
    dt_bias = seeded_randn(torch.empty(heads, dtype=torch.bfloat16), 22)

    def call():
        x, ba = torch.split(packed, [conv_dim, (householder + 1) * heads], dim=-1)
        return list(
            gdp_decode_prepare(
                x,
                ba,
                A_log=a_log,
                dt_bias=dt_bias,
                num_householder=householder,
                num_heads=heads,
                num_groups=groups,
                head_dim=head_dim,
                state_dim=state_dim,
            )
        )

    return dict(call=call, root=packed, make_input=lambda i: seeded_randn(packed, i))


# --- vLLM MoE kernels --------------------------------------------------------------------------
#
# Each input (the producer's target) stays near 1 MiB: the producer writes 1 KiB per block, and
# most of a much larger input would be written before the kernel launches, hiding a missing wait.

_MOE_TOKENS, _MOE_TOPK, _MOE_EXPERTS, _MOE_LOCAL, _MOE_LOCAL_START = 256, 6, 128, 16, 16
_MOE_HIDDEN, _MOE_FFN, _MOE_VALID = 2688, 1856, 200


def _routing_map(seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    scores = torch.rand(_MOE_TOKENS, _MOE_EXPERTS, generator=generator, device="cuda")
    return scores.topk(_MOE_TOPK, dim=1).indices


def _vllm_fused_moe_module():
    # The package re-exports the vllm_fused_moe function under the submodule's name, so
    # ``from megatron.core.inference.moe import vllm_fused_moe`` binds the function.
    return importlib.import_module("megatron.core.inference.moe.vllm_fused_moe")


def _moe_align_parallel():
    m = _vllm_fused_moe_module()

    # Launched directly with persistent outputs: the wrapper's fallback buffers are fresh
    # torch.empty tensors whose unwritten tail would differ between runs.
    block_m = 16
    max_pairs = _MOE_TOKENS * _MOE_TOPK
    max_sorted = max_pairs + block_m * (_MOE_LOCAL + 1)
    routing_map = torch.empty(_MOE_TOKENS, _MOE_TOPK, device="cuda", dtype=torch.int64)
    valid = _dev_scalar(_MOE_VALID)
    sorted_ids = torch.empty(max_sorted, device="cuda", dtype=torch.int32)
    expert_ids = torch.empty(triton.cdiv(max_sorted, block_m), device="cuda", dtype=torch.int32)
    num_post_padded = torch.empty(1, device="cuda", dtype=torch.int32)
    num_ctas = max(
        1,
        min(
            triton.cdiv(max_pairs, m._PARALLEL_ALIGN_CHUNK),
            torch.cuda.get_device_properties(0).multi_processor_count,
        ),
    )

    def call():
        m._parallel_align_block_size_kernel[(num_ctas,)](
            routing_map,
            valid,
            sorted_ids,
            expert_ids,
            num_post_padded,
            _MOE_LOCAL_START,
            max_pairs,
            max_pairs,
            topk=_MOE_TOPK,
            num_local_experts=_MOE_LOCAL,
            BLOCK_SIZE_M=block_m,
            NUM_BINS=m._align_expert_bins(_MOE_LOCAL),
            FIRST_HIST_CHUNK=1024,
            HIST_CHUNK=1024,
            CHUNK=m._PARALLEL_ALIGN_CHUNK,
            num_warps=m._PARALLEL_ALIGN_NUM_WARPS,
            **triton_pdl.pdl_kwargs(),
        )
        return [sorted_ids, expert_ids, num_post_padded]

    return dict(
        call=call,
        root=routing_map,
        make_input=_routing_map,
        outputs=(sorted_ids, expert_ids, num_post_padded),
    )


def _fused_moe_gemm():
    m = _vllm_fused_moe_module()

    # FC1 with the fused squared ReLU, as in decode, on static expert tables.
    config, _ = m._get_launch_configs(
        _MOE_TOKENS, _MOE_LOCAL, _MOE_TOPK, _MOE_FFN, _MOE_HIDDEN, torch.device("cuda")
    )
    block_m = config["BLOCK_SIZE_M"]
    align_args = (_routing_map(3), block_m, _MOE_LOCAL, _MOE_LOCAL_START, _dev_scalar(_MOE_VALID))
    if block_m < 128 and _MOE_TOKENS * _MOE_TOPK <= m._PARALLEL_ALIGN_MAX_PAIRS:
        tables = m._moe_align_block_size_parallel(*align_args, _MOE_TOKENS * _MOE_TOPK)
    else:
        tables = m._moe_align_block_size_cuda_graphable(*align_args)
    sorted_ids, expert_ids, num_post_padded = [t.clone() for t in tables]
    generator = torch.Generator(device="cuda").manual_seed(61)
    probs = torch.rand(_MOE_TOKENS * _MOE_TOPK, generator=generator, device="cuda")
    a = torch.empty(_MOE_TOKENS, _MOE_HIDDEN, device="cuda", dtype=torch.bfloat16)
    weight = seeded_randn(
        torch.empty(_MOE_LOCAL, _MOE_FFN, _MOE_HIDDEN, dtype=torch.bfloat16), 62, 0.02
    )
    out = torch.empty(_MOE_TOKENS * _MOE_TOPK, _MOE_FFN, device="cuda", dtype=torch.bfloat16)
    num_pid_m = triton.cdiv(_MOE_TOKENS * _MOE_TOPK + block_m * _MOE_LOCAL, block_m)

    def call():
        m._invoke_fused_moe_kernel(
            a,
            weight,
            out,
            probs,
            sorted_ids,
            expert_ids,
            num_post_padded,
            mul_routed_weight=False,
            top_k=_MOE_TOPK,
            config=config,
            grid_size=num_pid_m * triton.cdiv(_MOE_FFN, config["BLOCK_SIZE_N"]),
            fuse_squared_relu=True,
        )
        return [out]

    return dict(call=call, root=a, make_input=lambda i: seeded_randn(a, i), outputs=(out,))


def _moe_sum_tiled():
    m = _vllm_fused_moe_module()

    routing, valid = _routing_map(3), _dev_scalar(_MOE_VALID)
    hidden = 384
    generator = torch.Generator(device="cuda").manual_seed(64)
    probs = torch.rand(_MOE_TOKENS, _MOE_TOPK, generator=generator, device="cuda")
    inp = torch.empty(_MOE_TOKENS * _MOE_TOPK, hidden, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(_MOE_TOKENS, hidden, device="cuda", dtype=torch.float32)

    def call():
        m._moe_sum(
            inp,
            probs,
            _MOE_TOKENS,
            _MOE_TOPK,
            hidden,
            valid,
            routing,
            _MOE_LOCAL_START,
            _MOE_LOCAL,
            out=out,
            apply_weights=False,
        )
        return [out[:_MOE_VALID]]

    return dict(call=call, root=inp, make_input=lambda i: seeded_randn(inp, i), outputs=(out,))


# --- Streamwise residual ops -------------------------------------------------------------------


def _streamwise_sigmoid(op):
    num_streams, width = 4, 1024
    hidden = torch.empty(186, num_streams * width, device="cuda", dtype=torch.bfloat16)
    read_logits = torch.zeros(128, device="cuda", dtype=torch.float32)
    read_logits[:num_streams] = torch.tensor([-0.7, -0.2, 0.3, 0.9], device="cuda")
    write_logits = torch.zeros(128, device="cuda", dtype=torch.float32)
    write_logits[:num_streams] = torch.tensor([0.4, -0.1, 0.2, -0.6], device="cuda")
    update = seeded_randn(torch.empty(186, width, dtype=torch.bfloat16), 41)

    if op == "read":

        def call():
            return [
                streamwise_residual_ops._streamwise_sigmoid_read_triton(
                    hidden, read_logits, num_streams, torch.bfloat16
                )
            ]

    else:

        def call():
            return [
                streamwise_residual_ops._streamwise_sigmoid_write_triton(
                    hidden, update, write_logits, num_streams, None, 0.0
                )
            ]

    return dict(call=call, root=hidden, make_input=lambda i: seeded_randn(hidden, i))


_SSM_CASES = {
    "causal_conv1d_update": _causal_conv1d_update,
    "fused_recurrent_gated_delta_rule_update": _fused_recurrent_update,
    "gdp_decode_prepare": _gdp_decode_prepare,
}
_STREAMWISE_CASES = {
    "streamwise_sigmoid_read": lambda: _streamwise_sigmoid("read"),
    "streamwise_sigmoid_write": lambda: _streamwise_sigmoid("write"),
}
_VLLM_MOE_CASES = {
    "moe_align_block_size_parallel": _moe_align_parallel,
    "fused_moe_gemm_fc1": _fused_moe_gemm,
    "moe_sum_tiled": _moe_sum_tiled,
}
_CASES = {
    "zero_padding_rows": _zero_padding_rows,
    "mask_routing_padding": _mask_routing_padding,
    "sigmoid_topk_routing": _sigmoid_topk_routing,
    "mxfp8_quantize_into": _mxfp8_quantize_into,
    "mxfp8_quantize_rows": _mxfp8_quantize_rows,
    "append_key_value_cache": _append_key_value_cache,
    **_VLLM_MOE_CASES,
    **_SSM_CASES,
    **_STREAMWISE_CASES,
}


@pytest.mark.parametrize("case", sorted(_CASES))
def test_pdl_launch_is_bit_exact(case, monkeypatch):
    """Launched with PDL right after a producer that writes its input late, each kernel waits
    for the producer and matches its non-PDL launch byte for byte (eager and CUDA graph)."""
    if case in _STREAMWISE_CASES and not streamwise_residual_ops.HAVE_STREAMWISE_TRITON:
        pytest.skip("streamwise residual Triton kernels unavailable")
    if case in _SSM_CASES:
        # Pin the SSM autotuner / workspace path like ``--deterministic-mode`` does.
        monkeypatch.setattr(tuning_policy, "_deterministic_override", True)
    seeded()
    assert_pdl_launch_bit_exact(monkeypatch, **_CASES[case](), what=case)


# --- Multi-rank symmetric-memory collectives -------------------------------------------------
#
# The NVLS collectives synchronise ranks through symmetric-memory barriers. Each rank launches
# its collective with PDL after a producer that writes the collective's input late; a missing
# wait would publish this rank's stale (garbage) data to its peers.

_LOCAL_TOKENS, _HIDDEN = 16, 2688
_SYMM_KEYS = ("pdl_test_0", "pdl_test_1", "pdl_test_2")


@pytest.fixture
def nvls_group():
    from megatron.core import parallel_state
    from megatron.core.inference.communication.torch_symm_triton import is_device_nvls_capable
    from megatron.core.inference.symmetric_memory import SymmetricMemoryManager
    from tests.unit_tests.test_utilities import Utils

    if Utils.world_size < 2:
        pytest.skip("multi-rank PDL cases need at least two ranks")
    if not is_device_nvls_capable(torch.device("cuda")):
        pytest.skip("NVLS multicast is unavailable")
    Utils.initialize_model_parallel(tensor_model_parallel_size=Utils.world_size)
    group = parallel_state.get_tensor_model_parallel_group()
    buffers = [
        SymmetricMemoryManager.get_buffer(key, process_group=group, size_mb=64)
        for key in _SYMM_KEYS
    ]
    try:
        if any(b.symm_mem_hdl is None for b in buffers):
            pytest.skip(f"symmetric memory unavailable: {buffers[0].init_failure_reason}")
        yield group, buffers
    finally:
        torch.cuda.synchronize()
        torch.distributed.barrier()
        for key in _SYMM_KEYS:
            SymmetricMemoryManager.destroy(key)
        Utils.destroy_model_parallel()


def _collective_case(name, group, buffers):
    from megatron.core.inference.communication import torch_symm_triton as st

    rank, world = torch.distributed.get_rank(group), torch.distributed.get_world_size(group)
    bf16 = torch.bfloat16

    def symm(index, shape, dtype=bf16):
        result = buffers[index].maybe_get_tensor(list(shape), dtype)
        return result["tensor"], result["handle"]

    def rank_input(like):
        return lambda i: seeded_randn(like, 1000 * rank + i)

    if name == "all_gather":
        out, hdl = symm(0, (_LOCAL_TOKENS * world, _HIDDEN))
        inp = torch.empty(_LOCAL_TOKENS, _HIDDEN, device="cuda", dtype=bf16)
        return dict(
            call=lambda: [st.multimem_all_gather(out, inp, hdl)],
            root=inp,
            make_input=rank_input(inp),
            outputs=(out,),
        )

    if name == "reduce_scatter":
        inp, hdl = symm(0, (_LOCAL_TOKENS * world, _HIDDEN))
        out = torch.empty(_LOCAL_TOKENS, _HIDDEN, device="cuda", dtype=bf16)
        return dict(
            call=lambda: [st.multimem_reduce_scatter(out, inp, hdl)],
            root=inp,
            make_input=rank_input(inp),
            outputs=(out,),
        )

    # Variable-count EP collectives: rank r holds 8 + 2r tokens (at most per_rank_max).
    counts = [8 + 2 * r for r in range(world)]
    per_rank_max = max(counts)
    offset = _dev_scalar(sum(counts[:rank]))
    ep_max = _dev_scalar(per_rank_max)
    local, total = counts[rank], sum(counts)

    if name == "reduce_scatter_v":
        inp, hdl = symm(0, (per_rank_max * world, _HIDDEN))
        out = torch.empty(local, _HIDDEN, device="cuda", dtype=bf16)

        def call():
            st.multimem_reduce_scatter_v(out, inp, hdl, offset, ep_max, per_rank_max)
            return [out]

        return dict(call=call, root=inp, make_input=rank_input(inp), outputs=(out,))

    if name == "all_gather_v_3tensor":
        topk = 8
        out_h, hdl_h = symm(0, (per_rank_max * world, _HIDDEN))
        out_r, hdl_r = symm(1, (per_rank_max * world, topk), torch.int64)
        out_p, hdl_p = symm(2, (per_rank_max * world, topk), torch.float32)
        hidden = torch.empty(local, _HIDDEN, device="cuda", dtype=bf16)
        generator = torch.Generator(device="cuda").manual_seed(90 + rank)
        routing = torch.randint(0, 128, (local, topk), generator=generator, device="cuda")
        probs = torch.rand(local, topk, generator=generator, device="cuda")

        def call():
            st.multimem_all_gatherv_3tensor(
                out_h,
                out_r,
                out_p,
                hidden,
                routing,
                probs,
                hdl_h,
                hdl_r,
                hdl_p,
                rank_token_offset=offset,
                ep_max_tokens=ep_max,
                per_rank_max_tokens=per_rank_max,
            )
            return [out_h[:total], out_r[:total], out_p[:total]]

        return dict(
            call=call, root=hidden, make_input=rank_input(hidden), outputs=(out_h, out_r, out_p)
        )

    raise ValueError(name)


@pytest.mark.parametrize(
    "case", ["all_gather", "reduce_scatter", "reduce_scatter_v", "all_gather_v_3tensor"]
)
def test_pdl_collective_launch_is_bit_exact(case, monkeypatch, nvls_group):
    """Every rank launches its NVLS collective with PDL right after a producer that writes the
    collective's input late; all ranks match the non-PDL launch byte for byte."""
    group, buffers = nvls_group
    seeded()
    assert_pdl_launch_bit_exact(
        monkeypatch, **_collective_case(case, group, buffers), what=case, group=group
    )
