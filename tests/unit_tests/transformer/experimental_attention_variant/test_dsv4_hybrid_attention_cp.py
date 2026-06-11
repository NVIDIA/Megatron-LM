# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import gc
import os
import statistics
from contextlib import contextmanager, nullcontext

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

import megatron.core.parallel_state as parallel_state
from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.experimental_attention_variant.csa_cp_utils import (
    DSV4_CP_PARTITION_CONTIGUOUS,
    DSV4_CP_PARTITION_TWO_CHUNK,
    all_gather_fixed_cp_tensor,
    build_cp_compressor_prep_compact_fused,
    build_cp_indexer_loss_indices_fused,
    build_cp_rank_major_compressed_metadata_fused,
    exchange_cp_boundary_hidden,
    build_global_compressed_cu_seqlens,
    local_q_cp_chunk_ranges,
    compute_cp_indexer_topk_logical_fused,
    repack_rank_major_compressed_to_seq_major_fused,
    thd_cp_local_row_indices,
)
from megatron.core.transformer.experimental_attention_variant.dsa_kernels import indexer_topk
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.experimental_attention_variant.test_dsv4_hybrid_attention import (
    _SEED,
    _build_attention,
    _make_config,
)
from tests.unit_tests.transformer.experimental_attention_variant.test_dsv4_hybrid_native_parity import (
    _DSV4_VARIANTS,
)


# ===========================================================================
# THD CP parity tests
# ===========================================================================


_DSV4_CP_PARITY_EPS = 1e-3
_DSV4_CP_GRAPH_FUSED_SIM_EPS = 1e-6
_DSV4_CP_GRAPH_FUSED_RTOL = 1e-6
# Fused backward kernels use atomic accumulation, so graph/eager accumulation
# order can differ. The similarity gates stay at 1e-6; assert_close only bounds
# local elementwise noise from the atomics.
_DSV4_CP_GRAPH_FUSED_BF16_ATOL = 1.0
_DSV4_CP_GRAPH_FUSED_FP32_ATOL = 2e-2
# Recent full-pass measurements put peak allocated delta scale at roughly
# 0.55-0.59 for CP2 and 0.30-0.32 for CP4 across ratio 0/4/128 and both
# partition modes. These limits leave room for allocator noise while still
# catching regressions that allocate full-size activation buffers on each CP
# rank.
_DSV4_CP_MEMORY_RATIO_LIMITS = {2: 0.60, 4: 0.35}
# Recent full-pass measurements put graph replay scale at roughly
# 0.67-0.77 for CP2 and 0.39-0.47 for CP4 across ratio 0/4/128 and both
# partition modes. These limits leave room for machine timing jitter while
# still catching regressions that reintroduce full-size work or slow unfused
# kernels into the CP layer path.
_DSV4_CP_GRAPH_TIME_RATIO_LIMITS = {2: 0.90, 4: 0.60}
_DSV4_CP_MEMORY_WARMUP_ITERS = 3
_DSV4_CP_GRAPH_TIMING_WARMUP_REPLAYS = 2
_DSV4_CP_GRAPH_TIMING_MEASURE_REPLAYS = 5
_DSV4_CP_TEST_VARIANT = "flash"


class _ReferenceCPGroup:
    def rank(self):
        return 0

    def size(self):
        return 1


# Padded total is 4096, divisible by CP2/CP4. Only the final segment has tail
# padding, so the intermediate sequence boundaries match the unpadded layout.
_DSV4_CP_RAGGED_SEG_LENS = (1, 127, 1000, 23, 129, 900, 55, 257, 800, 95, 509, 148)
_DSV4_CP_RAGGED_PADDED_SEG_LENS = (
    1,
    127,
    1000,
    23,
    129,
    900,
    55,
    257,
    800,
    95,
    509,
    200,
)
# Same shape, padded total, and max padded sequence length as
# _DSV4_CP_RAGGED_PADDED_SEG_LENS, but the padding is distributed across many
# sequences instead of only the tail. CUDA graph replay must tolerate this value
# change because CP kernels rebuild compressed-row metadata from device
# cu_seqlens_padded rather than from capture-time host sizes.
_DSV4_CP_REPLAY_PADDED_SEG_LENS = (
    8,
    128,
    1000,
    32,
    132,
    904,
    64,
    260,
    804,
    96,
    512,
    156,
)


def _make_thd_packed_seq_params(seg_lens, padded_seg_lens=None, device='cuda'):
    if padded_seg_lens is None:
        padded_seg_lens = seg_lens
    cu = torch.tensor([0] + list(torch.tensor(seg_lens).cumsum(0).tolist()), dtype=torch.int32, device=device)
    cu_padded = torch.tensor(
        [0] + list(torch.tensor(padded_seg_lens).cumsum(0).tolist()),
        dtype=torch.int32,
        device=device,
    )
    max_len = int(max(padded_seg_lens)) if padded_seg_lens else 0
    return PackedSeqParams(
        cu_seqlens_q=cu,
        cu_seqlens_q_padded=cu_padded,
        cu_seqlens_kv=cu,
        cu_seqlens_kv_padded=cu_padded,
        max_seqlen_q=max_len,
        max_seqlen_kv=max_len,
        qkv_format='thd',
    )


def _cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(
        a.flatten().double().unsqueeze(0), b.flatten().double().unsqueeze(0)
    ).item()


def _tensor_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double(), b.double()
    denom = (a * a + b * b).sum()
    return (2.0 * (a * b).sum() / denom).item() if denom else 1.0


def _assert_cp_tensor_match(actual: torch.Tensor, expected: torch.Tensor, label: str):
    assert actual.shape == expected.shape, (
        f"{label}: shape {tuple(actual.shape)} != {tuple(expected.shape)}"
    )
    assert torch.isfinite(actual).all(), f"{label}: actual has non-finite values"
    assert torch.isfinite(expected).all(), f"{label}: expected has non-finite values"

    diff = (actual - expected).abs()
    max_abs = diff.max().item() if diff.numel() else 0.0
    cosine_sim = _cosine_sim(actual, expected)
    tensor_sim = _tensor_sim(actual, expected)
    actual_norm = actual.double().norm().item()
    expected_norm = expected.double().norm().item()
    assert cosine_sim > 1 - _DSV4_CP_PARITY_EPS, (
        f"{label}: cosine_sim={cosine_sim:.10f}, "
        f"tensor_sim={tensor_sim:.10f}, max_abs={max_abs:.6e}, "
        f"actual_norm={actual_norm:.6e}, expected_norm={expected_norm:.6e}, "
        f"eps={_DSV4_CP_PARITY_EPS}"
    )
    assert tensor_sim > 1 - _DSV4_CP_PARITY_EPS, (
        f"{label}: tensor_sim={tensor_sim:.10f}, "
        f"cosine_sim={cosine_sim:.10f}, max_abs={max_abs:.6e}, "
        f"actual_norm={actual_norm:.6e}, expected_norm={expected_norm:.6e}, "
        f"eps={_DSV4_CP_PARITY_EPS}"
    )


def _assert_cp_graph_match(actual: torch.Tensor, expected: torch.Tensor, label: str):
    assert actual.shape == expected.shape, (
        f"{label}: shape {tuple(actual.shape)} != {tuple(expected.shape)}"
    )
    if torch.equal(actual, expected):
        return
    _assert_cp_tensor_match(actual, expected, label)


def _assert_cp_graph_bitwise_match(actual: torch.Tensor, expected: torch.Tensor, label: str):
    assert actual.shape == expected.shape, (
        f"{label}: shape {tuple(actual.shape)} != {tuple(expected.shape)}"
    )
    if torch.equal(actual, expected):
        return
    diff = (actual - expected).abs()
    max_abs = diff.max().item() if diff.numel() else 0.0
    cosine_sim = _cosine_sim(actual, expected)
    tensor_sim = _tensor_sim(actual, expected)
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else -1
    print(
        f"[rank{rank}] {label}: graph/eager not bitwise; max_abs={max_abs:.6e}, "
        f"cosine_sim={cosine_sim:.10f}, tensor_sim={tensor_sim:.10f}",
        flush=True,
    )
    raise AssertionError(
        f"{label}: graph/eager must be bitwise equal; max_abs={max_abs:.6e}, "
        f"cosine_sim={cosine_sim:.10f}, tensor_sim={tensor_sim:.10f}"
    )


def _assert_cp_graph_fused_backward_match(
    actual: torch.Tensor, expected: torch.Tensor, label: str
):
    assert actual.shape == expected.shape, (
        f"{label}: shape {tuple(actual.shape)} != {tuple(expected.shape)}"
    )
    assert actual.dtype == expected.dtype, f"{label}: dtype {actual.dtype} != {expected.dtype}"
    assert torch.isfinite(actual).all(), f"{label}: actual has non-finite values"
    assert torch.isfinite(expected).all(), f"{label}: expected has non-finite values"

    diff = (actual.float() - expected.float()).abs()
    max_abs = diff.max().item() if diff.numel() else 0.0
    cosine_sim = _cosine_sim(actual, expected)
    tensor_sim = _tensor_sim(actual, expected)
    assert cosine_sim > 1 - _DSV4_CP_GRAPH_FUSED_SIM_EPS, (
        f"{label}: cosine_sim={cosine_sim:.10f}, "
        f"max_abs={max_abs:.6e}, eps={_DSV4_CP_GRAPH_FUSED_SIM_EPS}"
    )
    assert tensor_sim > 1 - _DSV4_CP_GRAPH_FUSED_SIM_EPS, (
        f"{label}: tensor_sim={tensor_sim:.10f}, "
        f"max_abs={max_abs:.6e}, eps={_DSV4_CP_GRAPH_FUSED_SIM_EPS}"
    )

    if actual.dtype == torch.bfloat16:
        atol = _DSV4_CP_GRAPH_FUSED_BF16_ATOL
    elif actual.dtype == torch.float32:
        atol = _DSV4_CP_GRAPH_FUSED_FP32_ATOL
    else:
        raise AssertionError(f"{label}: unsupported dtype for fused graph close check: {actual.dtype}")
    try:
        torch.testing.assert_close(
            actual,
            expected,
            rtol=_DSV4_CP_GRAPH_FUSED_RTOL,
            atol=atol,
            msg=(
                f"{label}: fused graph/eager backward mismatch; "
                f"rtol={_DSV4_CP_GRAPH_FUSED_RTOL}, atol={atol}, "
                f"cosine_sim={cosine_sim:.10f}, tensor_sim={tensor_sim:.10f}, "
                f"max_abs={max_abs:.6e}"
            ),
        )
    except AssertionError:
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else -1
        print(
            f"[rank{rank}] {label}: fused backward close failed; "
            f"rtol={_DSV4_CP_GRAPH_FUSED_RTOL}, atol={atol}, "
            f"cosine_sim={cosine_sim:.10f}, tensor_sim={tensor_sim:.10f}, "
            f"max_abs={max_abs:.6e}",
            flush=True,
        )
        raise


@contextmanager
def _deterministic_torch_algorithms():
    old_enabled = torch.are_deterministic_algorithms_enabled()
    old_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(old_enabled, warn_only=old_warn_only)


def _make_dsv4_cp_config(
    *,
    context_parallel_size,
    dsa_indexer_loss_coeff=0.0,
    dsa_indexer_use_sparse_loss=True,
    apply_dsa_kernel_fusion=True,
    apply_rope_fusion=False,
    csa_cp_partition_mode=DSV4_CP_PARTITION_CONTIGUOUS,
):
    shape = _DSV4_VARIANTS[_DSV4_CP_TEST_VARIANT]
    return _make_config(
        hidden_size=shape["hidden_size"],
        num_attention_heads=shape["num_attention_heads"],
        v_head_dim=shape["v_head_dim"],
        qk_pos_emb_head_dim=shape["qk_pos_emb_head_dim"],
        q_lora_rank=shape["q_lora_rank"],
        o_groups=shape["o_groups"],
        o_lora_rank=shape["o_lora_rank"],
        csa_compress_ratios=[0, 4, 128, 4],
        csa_window_size=128,
        dsa_indexer_n_heads=64,
        dsa_indexer_head_dim=128,
        dsa_indexer_topk=shape["dsa_indexer_topk"],
        dsa_indexer_loss_coeff=dsa_indexer_loss_coeff,
        dsa_indexer_use_sparse_loss=dsa_indexer_use_sparse_loss,
        context_parallel_size=context_parallel_size,
        csa_dense_mode=False,
        csa_compress_rotary_base=shape["csa_compress_rotary_base"],
        layernorm_epsilon=1e-6,
        normalization="RMSNorm",
        qk_layernorm=True,
        layernorm_zero_centered_gamma=False,
        expert_model_parallel_size=1,
        apply_dsa_kernel_fusion=apply_dsa_kernel_fusion,
        apply_rope_fusion=apply_rope_fusion,
        csa_cp_partition_mode=csa_cp_partition_mode,
    )


def _copy_module_parameters(src, dst):
    src_params = dict(src.named_parameters())
    for name, param in dst.named_parameters():
        assert name in src_params
        param.data.copy_(src_params[name].data)
    return src_params


def _make_cp_partition_indices(partition_mode, padded_total_tokens, cp_size, device='cuda'):
    return tuple(
        thd_cp_local_row_indices(partition_mode, padded_total_tokens, cp_size, rank, device)
        for rank in range(cp_size)
    )


def _make_ragged_cp_case(partition_mode, cp_size, cp_rank):
    padded_total_tokens = sum(_DSV4_CP_RAGGED_PADDED_SEG_LENS)
    packed = _make_thd_packed_seq_params(
        _DSV4_CP_RAGGED_SEG_LENS,
        _DSV4_CP_RAGGED_PADDED_SEG_LENS,
    )
    partition_indices = _make_cp_partition_indices(partition_mode, padded_total_tokens, cp_size)
    return packed, padded_total_tokens, partition_indices[cp_rank]


def _make_hidden_and_grad(padded_total_tokens, hidden_size):
    hidden = torch.randn(
        padded_total_tokens,
        1,
        hidden_size,
        dtype=torch.bfloat16,
        device='cuda',
    )
    return hidden, torch.randn_like(hidden)


def _run_dsv4_attention_forward_backward(
    attn, hidden, grad, packed_seq_params, *, collect_result=True
):
    hidden.grad = None
    attn.zero_grad(set_to_none=True)
    output, _ = attn(hidden_states=hidden, attention_mask=None, packed_seq_params=packed_seq_params)
    output.backward(grad)
    if not collect_result:
        return None
    param_grads = {
        name: param.grad.detach().clone()
        for name, param in attn.named_parameters()
        if param.grad is not None
    }
    return output.detach().clone(), hidden.grad.detach().clone(), param_grads


def _capture_dsv4_attention_forward_backward(attn, static_hidden, static_grad, packed_seq_params):
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        for _ in range(3):
            _run_dsv4_attention_forward_backward(
                attn, static_hidden, static_grad, packed_seq_params, collect_result=False
            )
    torch.cuda.current_stream().wait_stream(warmup_stream)
    torch.cuda.synchronize()

    static_hidden.grad = None
    attn.zero_grad(set_to_none=True)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, capture_error_mode="thread_local"):
        graph_output, _ = attn(
            hidden_states=static_hidden,
            attention_mask=None,
            packed_seq_params=packed_seq_params,
        )
        graph_output.backward(static_grad)
    torch.cuda.synchronize()
    return graph, graph_output


def _zero_existing_grads(attn, hidden):
    if hidden.grad is not None:
        hidden.grad.zero_()
    for param in attn.parameters():
        if param.grad is not None:
            param.grad.zero_()


def _run_dsv4_attention_forward_backward_reuse_grads(attn, hidden, grad, packed_seq_params):
    _zero_existing_grads(attn, hidden)
    output, _ = attn(hidden_states=hidden, attention_mask=None, packed_seq_params=packed_seq_params)
    output.backward(grad)
    return output


def _max_int_across_world(value: int) -> int:
    tensor = torch.tensor(value, dtype=torch.int64, device='cuda')
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return int(tensor.item())


def _max_float_across_world(value: float) -> float:
    tensor = torch.tensor(value, dtype=torch.float64, device='cuda')
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def _measure_peak_allocated_delta(attn, hidden, grad, packed_seq_params) -> int:
    attn.train()
    for _ in range(_DSV4_CP_MEMORY_WARMUP_ITERS):
        _run_dsv4_attention_forward_backward_reuse_grads(attn, hidden, grad, packed_seq_params)
    torch.cuda.synchronize()
    _zero_existing_grads(attn, hidden)
    torch.cuda.reset_peak_memory_stats()
    memory_start = torch.cuda.memory_allocated()
    _run_dsv4_attention_forward_backward_reuse_grads(attn, hidden, grad, packed_seq_params)
    torch.cuda.synchronize()
    return _max_int_across_world(torch.cuda.max_memory_allocated() - memory_start)


def _measure_cuda_graph_time(attn, static_hidden, static_grad, packed_seq_params) -> float:
    attn.train()
    graph, _graph_output = _capture_dsv4_attention_forward_backward(
        attn, static_hidden, static_grad, packed_seq_params
    )
    for _ in range(_DSV4_CP_GRAPH_TIMING_WARMUP_REPLAYS):
        graph.replay()
    torch.cuda.synchronize()

    timings = []
    for _ in range(_DSV4_CP_GRAPH_TIMING_MEASURE_REPLAYS):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        graph.replay()
        end_event.record()
        torch.cuda.synchronize()
        timings.append(float(start_event.elapsed_time(end_event)))
    return _max_float_across_world(statistics.median(timings))


def _format_scale_report(metric, layer_number, cp_size, partition_mode, baseline, measured, scale, limit):
    return (
        f"DSv4 THD CP {metric} scale layer={layer_number} cp{cp_size}: "
        f"partition={partition_mode}, "
        f"cp1={baseline:.6f}, cp{cp_size}={measured:.6f}, "
        f"scale={scale:.6f}, limit={limit:.6f}"
    )


def _clear_cuda_test_state():
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not HAVE_TE, reason="transformer_engine not available")
class TestDSv4HybridAttentionTHDCP:
    """CP-sliced THD attention should match full-sequence THD reference output and gradients."""

    @pytest.fixture(scope='class', autouse=True, params=(2, 4), ids=lambda cp: f"cp{cp}")
    def setup_method(self, request):
        cp_size = request.param
        if Utils.world_size < cp_size:
            pytest.skip(f"THD CP path test requires at least {cp_size} distributed ranks")
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=cp_size,
        )
        torch.manual_seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)

        cls = request.cls
        cls.cp_size = cp_size
        cls.cp_rank = parallel_state.get_context_parallel_rank()
        cls.pg = ProcessGroupCollection.use_mpu_process_groups()
        # Tradeoff: reuse the current model-parallel groups and only disable CP for the
        # full-reference path to avoid extra group initialization. This is valid for
        # these CP-only tests; rebuild the reference groups if future tests add other
        # parallel dimensions.
        cls.ref_pg = ProcessGroupCollection.use_mpu_process_groups()
        cls.ref_pg.cp = _ReferenceCPGroup()
        cls._cp1_memory_delta_cache = {}
        cls._cp1_graph_time_cache = {}

        yield
        _clear_cuda_test_state()
        Utils.destroy_model_parallel()

    @pytest.fixture(autouse=True)
    def clear_cuda_test_case(self):
        _clear_cuda_test_state()
        yield
        _clear_cuda_test_state()

    def _measure_cp1_peak_allocated_delta(self, layer_number, packed, padded_tokens):
        cached = self._cp1_memory_delta_cache.get(layer_number)
        if cached is not None:
            return cached

        torch.manual_seed(_SEED + 900 + layer_number)
        model_parallel_cuda_manual_seed(_SEED + 900 + layer_number)
        config = _make_dsv4_cp_config(
            context_parallel_size=1,
            dsa_indexer_loss_coeff=1.0,
            dsa_indexer_use_sparse_loss=True,
            apply_rope_fusion=True,
        )
        attn = _build_attention(config, layer_number=layer_number, pg_collection=self.ref_pg).cuda()
        hidden_values, grad_values = _make_hidden_and_grad(padded_tokens, config.hidden_size)
        hidden = hidden_values.detach().clone().requires_grad_(True)
        grad = grad_values.detach().clone()
        measured = _measure_peak_allocated_delta(attn, hidden, grad, packed)

        del attn, hidden_values, grad_values, hidden, grad
        _clear_cuda_test_state()
        self._cp1_memory_delta_cache[layer_number] = measured
        return measured

    def _measure_cp1_cuda_graph_time(self, layer_number, packed, padded_tokens):
        cached = self._cp1_graph_time_cache.get(layer_number)
        if cached is not None:
            return cached

        torch.manual_seed(_SEED + 1000 + layer_number)
        model_parallel_cuda_manual_seed(_SEED + 1000 + layer_number)
        config = _make_dsv4_cp_config(
            context_parallel_size=1,
            dsa_indexer_loss_coeff=1.0,
            dsa_indexer_use_sparse_loss=True,
            apply_rope_fusion=True,
        )
        attn = _build_attention(config, layer_number=layer_number, pg_collection=self.ref_pg).cuda()
        hidden_values, grad_values = _make_hidden_and_grad(padded_tokens, config.hidden_size)
        hidden = hidden_values.detach().clone().requires_grad_(True)
        grad = grad_values.detach().clone()
        measured = _measure_cuda_graph_time(attn, hidden, grad, packed)

        del attn, hidden_values, grad_values, hidden, grad
        _clear_cuda_test_state()
        self._cp1_graph_time_cache[layer_number] = measured
        return measured

    @pytest.mark.parametrize(
        "partition_mode, uses_two_chunk",
        [
            (DSV4_CP_PARTITION_CONTIGUOUS, False),
            (DSV4_CP_PARTITION_TWO_CHUNK, True),
        ],
        ids=["contiguous", "two_chunk"],
    )
    def test_thd_cp_partition_mode_selects_expected_row_order(
        self, partition_mode, uses_two_chunk
    ):
        """DSv4 CP partition mode controls the attention-layer row order.

        Expected: ``contiguous`` keeps the original one-range CP partition,
        while ``two_chunk`` makes this rank own chunk ``rank`` and
        chunk ``2*cp_size-1-rank``. A failure here means the config knob could
        drift from the layout helpers even if lower-level utility tests pass.
        """
        config = _make_dsv4_cp_config(
            context_parallel_size=self.cp_size,
            csa_cp_partition_mode=partition_mode,
        )

        l_local = 16
        ranges = local_q_cp_chunk_ranges(
            config.csa_cp_partition_mode, l_local, self.cp_size, self.cp_rank
        )
        if uses_two_chunk:
            chunk_len = l_local // 2
            total_chunks = 2 * self.cp_size
            chunk_ids = (self.cp_rank, total_chunks - 1 - self.cp_rank)
            assert ranges == tuple(
                (chunk_id * chunk_len, (chunk_id + 1) * chunk_len) for chunk_id in chunk_ids
            )
        else:
            assert ranges == ((self.cp_rank * l_local, (self.cp_rank + 1) * l_local),)

    def test_thd_cp_partition_mode_rejects_unknown_value(self):
        """Unknown DSv4 CP partition modes fail before a forward pass.

        Expected: the attention layer rejects unsupported values instead of
        silently falling back to contiguous mode. A failure here could hide a
        misconfigured benchmark or test.
        """
        config = _make_dsv4_cp_config(
            context_parallel_size=self.cp_size,
            csa_cp_partition_mode="invalid_mode",
        )

        with pytest.raises(RuntimeError, match="Unsupported CSA CP partition mode"):
            local_q_cp_chunk_ranges(config.csa_cp_partition_mode, 16, self.cp_size, self.cp_rank)

    @pytest.mark.parametrize(
        "partition_mode",
        [DSV4_CP_PARTITION_CONTIGUOUS, DSV4_CP_PARTITION_TWO_CHUNK],
        ids=["contiguous", "two_chunk"],
    )
    @pytest.mark.parametrize(
        "layer_number",
        [1, 2, 3],
        ids=["ratio_0_window_only", "ratio_4_indexer", "ratio_128_compressor"],
    )
    def test_thd_cp_unfused_rope_is_rejected(self, partition_mode, layer_number):
        """THD CP rejects the unfused RoPE path for every DSv4 ratio and CP row order.

        Production THD CP requires fused RoPE because CP-local rows can start in
        the middle of a packed sequence; the unfused path is not implemented for
        that position reconstruction.
        """
        packed, padded_tokens, local_idx = _make_ragged_cp_case(
            partition_mode, self.cp_size, self.cp_rank
        )

        torch.manual_seed(_SEED + 300 + layer_number)
        model_parallel_cuda_manual_seed(_SEED + 300 + layer_number)
        config = _make_dsv4_cp_config(
            context_parallel_size=self.cp_size,
            dsa_indexer_loss_coeff=1.0,
            dsa_indexer_use_sparse_loss=True,
            apply_dsa_kernel_fusion=True,
            apply_rope_fusion=False,
            csa_cp_partition_mode=partition_mode,
        )
        attn = _build_attention(config, layer_number=layer_number, pg_collection=self.pg).cuda()

        full_hidden = torch.randn(
            padded_tokens, 1, config.hidden_size, dtype=torch.bfloat16, device='cuda'
        )
        local_hidden = full_hidden.index_select(0, local_idx).detach().clone().requires_grad_(True)

        with pytest.raises(
            RuntimeError,
            match="DSv4 THD CP requires apply_rope_fusion=True",
        ):
            attn(hidden_states=local_hidden, attention_mask=None, packed_seq_params=packed)

        del attn, full_hidden, local_hidden
        _clear_cuda_test_state()

    def test_thd_cp_left_boundary_exchange_forward_backward(self):
        """CP boundary exchange receives the previous rank's tail window.

        In backward, the current rank's tail tokens receive gradient when the
        next rank used those tokens as its left boundary.
        """
        d_window = 2
        local_len = 4
        width = 3
        # The expected tensors below assume a positive tail window fully inside
        # the local tensor, with non-empty feature rows.
        assert d_window > 0
        assert local_len >= d_window
        assert width > 0
        local_numel = local_len * width
        local_start = self.cp_rank * local_numel
        values = torch.arange(
            local_start,
            local_start + local_numel,
            device='cuda',
            dtype=torch.float32,
        ).reshape(local_len, width)
        local = values.detach().clone().requires_grad_(True)

        boundary = exchange_cp_boundary_hidden(
            local, [], d_window, DSV4_CP_PARTITION_CONTIGUOUS, self.pg.cp
        )
        if self.cp_rank == 0:
            # Rank 0 has no previous CP rank, so the fixed left boundary is zero-filled.
            expected_boundary = torch.zeros_like(boundary)
        else:
            # Nonzero ranks receive the previous rank's final d_window rows as their boundary.
            left_rank_start = (self.cp_rank - 1) * local_numel
            expected_boundary = torch.arange(
                left_rank_start + (local_len - d_window) * width,
                left_rank_start + local_numel,
                device='cuda',
                dtype=torch.float32,
            ).reshape(d_window, width)
        assert torch.equal(boundary, expected_boundary)

        boundary.sum().backward()
        # Local tokens receive no boundary-exchange grad unless the next rank reads them.
        expected_grad = torch.zeros_like(local)
        if self.cp_rank + 1 < self.cp_size:
            # The next rank uses this rank's tail as its left boundary, so boundary.sum()
            # contributes unit gradient to exactly those tail rows.
            expected_grad[-d_window:] = 1
        assert torch.equal(local.grad, expected_grad)

    @pytest.mark.parametrize(
        "layer_number",
        [1, 2, 3],
        ids=["ratio_0_window_only", "ratio_4_indexer", "ratio_128_compressor"],
    )
    @pytest.mark.parametrize(
        "partition_mode",
        [DSV4_CP_PARTITION_CONTIGUOUS, DSV4_CP_PARTITION_TWO_CHUNK],
        ids=["contiguous", "two_chunk"],
    )
    def test_thd_cp_matches_full_reference_forward_backward(self, layer_number, partition_mode):
        """CP path matches the full-sequence THD reference on ragged
        packed inputs using the DSv4 layer configuration.

        Verifies local CP output, hidden grad, and reduced parameter grads
        against the sliced full-reference tensors. The test body is shared by
        contiguous and two-chunk CP; only the partition mode and
        local row indices differ.
        """
        packed, padded_tokens, local_idx = _make_ragged_cp_case(
            partition_mode, self.cp_size, self.cp_rank
        )

        torch.manual_seed(_SEED + layer_number)
        model_parallel_cuda_manual_seed(_SEED + layer_number)
        config_cp = _make_dsv4_cp_config(
            context_parallel_size=self.cp_size,
            dsa_indexer_loss_coeff=1.0,
            dsa_indexer_use_sparse_loss=True,
            apply_rope_fusion=True,
            csa_cp_partition_mode=partition_mode,
        )
        config_ref = _make_dsv4_cp_config(
            context_parallel_size=1,
            dsa_indexer_loss_coeff=1.0,
            dsa_indexer_use_sparse_loss=True,
        )
        cp_attn = _build_attention(
            config_cp, layer_number=layer_number, pg_collection=self.pg
        ).cuda()
        ref_attn = _build_attention(
            config_ref, layer_number=layer_number, pg_collection=self.ref_pg
        ).cuda()
        _copy_module_parameters(cp_attn, ref_attn)

        full_hidden = torch.randn(
            padded_tokens, 1, config_cp.hidden_size, dtype=torch.bfloat16, device='cuda'
        )
        local_hidden = full_hidden.index_select(0, local_idx).detach().clone().requires_grad_(True)
        ref_hidden = full_hidden.detach().clone().requires_grad_(True)

        local_out, _ = cp_attn(
            hidden_states=local_hidden, attention_mask=None, packed_seq_params=packed
        )
        ref_out, _ = ref_attn(
            hidden_states=ref_hidden, attention_mask=None, packed_seq_params=packed
        )
        _assert_cp_tensor_match(
            local_out.detach(),
            ref_out.detach().index_select(0, local_idx),
            f"layer={layer_number}:{partition_mode}:output",
        )

        grad = torch.randn_like(ref_out)
        local_out.backward(grad.index_select(0, local_idx))
        ref_out.backward(grad)
        _assert_cp_tensor_match(
            local_hidden.grad.detach(),
            ref_hidden.grad.index_select(0, local_idx),
            f"layer={layer_number}:{partition_mode}:hidden_grad",
        )

        ref_params = dict(ref_attn.named_parameters())
        for name, param in cp_attn.named_parameters():
            ref_grad = ref_params[name].grad
            assert param.grad is not None, f"Missing CP grad for {name}"
            assert ref_grad is not None, f"Missing reference grad for {name}"
            grad_sum = param.grad.detach().clone()
            dist.all_reduce(grad_sum, group=self.pg.cp)
            _assert_cp_tensor_match(
                grad_sum, ref_grad, f"layer={layer_number}:{partition_mode}:param_grad:{name}"
            )

        del cp_attn, ref_attn, full_hidden, local_hidden, ref_hidden, local_out, ref_out, grad
        _clear_cuda_test_state()

    def test_thd_two_chunk_cp_indexer_inputs_match_full_reference(self):
        """Two-chunk CP indexer Q/weights and compressed K match full THD reference."""
        packed, padded_tokens, local_idx = _make_ragged_cp_case(
            DSV4_CP_PARTITION_TWO_CHUNK, self.cp_size, self.cp_rank
        )

        torch.manual_seed(_SEED + 400)
        model_parallel_cuda_manual_seed(_SEED + 400)
        config_cp = _make_dsv4_cp_config(
            context_parallel_size=self.cp_size,
            dsa_indexer_loss_coeff=1.0,
            dsa_indexer_use_sparse_loss=True,
            apply_rope_fusion=True,
            csa_cp_partition_mode=DSV4_CP_PARTITION_TWO_CHUNK,
        )
        config_ref = _make_dsv4_cp_config(
            context_parallel_size=1,
            dsa_indexer_loss_coeff=1.0,
            dsa_indexer_use_sparse_loss=True,
        )
        cp_attn = _build_attention(config_cp, layer_number=2, pg_collection=self.pg).cuda()
        ref_attn = _build_attention(config_ref, layer_number=2, pg_collection=self.ref_pg).cuda()
        _copy_module_parameters(cp_attn, ref_attn)

        full_hidden = torch.randn(
            padded_tokens, 1, config_cp.hidden_size, dtype=torch.bfloat16, device='cuda'
        )
        local_hidden = full_hidden.index_select(0, local_idx).detach().clone()
        ref_hidden = full_hidden.detach().clone()

        query_local, _, _, qr_local, _, _ = cp_attn.get_query_key_value_tensors(
            local_hidden, packed_seq_params=packed
        )
        query_ref, _, _, qr_ref, _, _ = ref_attn.get_query_key_value_tensors(
            ref_hidden, packed_seq_params=packed
        )

        core_cp = cp_attn.core_attention
        core_ref = ref_attn.core_attention
        cu_seqlens = packed.cu_seqlens_q_padded
        chunk_ranges = local_q_cp_chunk_ranges(
            DSV4_CP_PARTITION_TWO_CHUNK, padded_tokens // self.cp_size, self.cp_size, self.cp_rank
        )
        chunk_len = chunk_ranges[0][1] - chunk_ranges[0][0]
        d_comp = 8
        ratio = 4

        q_local, weights_local = core_cp.indexer._forward_thd_query_weights_cp(
            local_hidden.detach(),
            qr_local.detach(),
            cu_seqlens,
            int(packed.max_seqlen_q),
            chunk_ranges=chunk_ranges,
        )
        q_ref, k_ref, weights_ref, cu_ref = core_ref.indexer.forward_before_topk(
            ref_hidden.detach(), qr_ref.detach(), packed
        )
        _assert_cp_tensor_match(
            q_local.detach(),
            q_ref.squeeze(1).detach().index_select(0, local_idx),
            "two-chunk-indexer:q",
        )
        _assert_cp_tensor_match(
            weights_local.detach(),
            weights_ref.squeeze(1).detach().index_select(0, local_idx),
            "two-chunk-indexer:weights",
        )

        boundary_hidden = exchange_cp_boundary_hidden(
            local_hidden,
            config_cp.csa_compress_ratios,
            config_cp.csa_window_size,
            config_cp.csa_cp_partition_mode,
            self.pg.cp,
        )
        d_window = boundary_hidden.shape[0] // len(chunk_ranges)
        (
            hidden_compact,
            cu_compact,
            _seq_ids_local,
            comp_ids_local,
            _valid_local,
        ) = build_cp_compressor_prep_compact_fused(
            local_hidden,
            boundary_hidden,
            cu_seqlens,
            chunk_ranges,
            ratio,
            d_comp,
            d_window,
        )
        k_local, _ = core_cp.indexer.compressor._forward_thd(
            hidden_compact.detach(),
            cu_compact,
            max_seqlen_q=int(packed.max_seqlen_q),
            rope_positions=comp_ids_local,
        )
        k_rank_major = all_gather_fixed_cp_tensor(k_local.squeeze(1), self.pg.cp)
        seq_ids, comp_ids, valid = build_cp_rank_major_compressed_metadata_fused(
            cu_seqlens,
            chunk_ranges,
            self.cp_size,
            ratio,
            d_comp,
        )
        cu_compressed = build_global_compressed_cu_seqlens(cu_seqlens, ratio)
        k_seq_major, _rank_by_seq_major = repack_rank_major_compressed_to_seq_major_fused(
            k_rank_major,
            seq_ids,
            comp_ids,
            valid,
            cu_compressed,
            seq_major_rows=(padded_tokens // ratio),
        )
        actual_comp = int(cu_ref[-1].item())
        _assert_cp_tensor_match(
            k_seq_major[:actual_comp].detach(),
            k_ref.squeeze(1).detach()[:actual_comp],
            "two-chunk-indexer:k_seq_major",
        )

        topk_width = core_cp.indexer.index_topk
        max_seqlen_compressed_idx = int(packed.max_seqlen_q) // ratio
        cp_topk = compute_cp_indexer_topk_logical_fused(
            q_local,
            weights_local,
            k_seq_major,
            cu_seqlens,
            cu_compressed,
            chunk_ranges,
            ratio,
            topk_width,
            core_cp.indexer.softmax_scale,
            max_seqlen_q=int(packed.max_seqlen_q),
            max_seqlen_kv=max_seqlen_compressed_idx,
        )
        ref_topk, _ = indexer_topk(
            q_ref.squeeze(1),
            k_ref.squeeze(1),
            weights_ref.squeeze(1),
            topk=topk_width,
            ratio=ratio,
            indexer_softmax_scale=core_ref.indexer.softmax_scale,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_ref,
            max_seqlen_q=int(packed.max_seqlen_q),
            max_seqlen_kv=max_seqlen_compressed_idx,
        )
        ref_topk_local = ref_topk.index_select(0, local_idx)
        assert torch.equal(cp_topk.cpu(), ref_topk_local.cpu()), (
            "two-chunk-indexer:topk logical mismatch: "
            f"num_diff={(cp_topk != ref_topk_local).sum().item()}"
        )
        window_capacity = max(1, int(chunk_len) + d_window * (cu_seqlens.numel() - 1))
        shared_compressed_base = len(chunk_ranges) * window_capacity
        _loss_topk, cp_rank_major_topk = build_cp_indexer_loss_indices_fused(
            cu_seqlens,
            cu_compressed,
            chunk_ranges,
            d_window,
            core_cp.window_size,
            ratio,
            cp_topk,
            _rank_by_seq_major,
            shared_compressed_base=shared_compressed_base,
        )
        row_seq = torch.searchsorted(cu_seqlens, local_idx.to(cu_seqlens.dtype), right=True) - 1
        row_seq = row_seq.clamp(min=0, max=cu_seqlens.numel() - 2)
        seq_comp_start = cu_compressed.index_select(0, row_seq).unsqueeze(1)
        safe_topk = ref_topk_local.clamp(min=0).to(torch.long)
        safe_seq_major = (seq_comp_start.to(torch.long) + safe_topk).reshape(-1)
        expected_rank_major = _rank_by_seq_major.index_select(0, safe_seq_major).reshape_as(
            ref_topk_local
        )
        expected_rank_major = torch.where(
            ref_topk_local >= 0,
            expected_rank_major,
            torch.full_like(expected_rank_major, -1),
        )
        assert torch.equal(cp_rank_major_topk.cpu(), expected_rank_major.cpu()), (
            "two-chunk-indexer:rank-major topk mismatch: "
            f"num_diff={(cp_rank_major_topk != expected_rank_major).sum().item()}"
        )
        # Full layer parity covers indexer-loss backward. This test is scoped to
        # the two-chunk indexer input and top-k metadata contract.

        del (
            cp_attn,
            ref_attn,
            full_hidden,
            local_hidden,
            ref_hidden,
            query_local,
            qr_local,
            query_ref,
            qr_ref,
            q_local,
            weights_local,
            core_cp,
            core_ref,
            q_ref,
            k_ref,
            weights_ref,
            cu_ref,
            boundary_hidden,
            hidden_compact,
            cu_compact,
            _seq_ids_local,
            comp_ids_local,
            _valid_local,
            k_local,
            k_rank_major,
            seq_ids,
            comp_ids,
            valid,
            cu_compressed,
            k_seq_major,
            _rank_by_seq_major,
            cp_topk,
            ref_topk,
            ref_topk_local,
            _loss_topk,
            cp_rank_major_topk,
            row_seq,
            seq_comp_start,
            safe_topk,
            safe_seq_major,
            expected_rank_major,
        )

    @pytest.mark.parametrize(
        "layer_number",
        [1, 2, 3],
        ids=["ratio_0_window_only", "ratio_4_indexer", "ratio_128_compressor"],
    )
    @pytest.mark.parametrize("fused", [True, False])
    @pytest.mark.parametrize(
        "partition_mode",
        [DSV4_CP_PARTITION_CONTIGUOUS, DSV4_CP_PARTITION_TWO_CHUNK],
        ids=["contiguous", "two_chunk"],
    )
    def test_thd_cp_cuda_graph_matches_eager_forward_backward(
        self, layer_number, fused, partition_mode
    ):
        """CUDA graph replay matches eager THD CP forward/backward.

        Captures the DSv4 attention layer's CP-local forward and backward
        graph, replays it with fresh static-buffer contents, and compares the
        local output, hidden grad, and parameter grads against an eager module
        with identical weights. The unfused sparse-attn/indexer path is
        deterministic and must match bitwise for both forward and backward.
        Fused forward kernels are deterministic and must also match bitwise.
        Fused backward paths use strict similarity plus elementwise
        ``assert_close`` gates because their expected difference is atomic
        accumulation order.
        """
        context = nullcontext() if fused else _deterministic_torch_algorithms()
        mode = "fused" if fused else "unfused"
        with context:
            packed, padded_tokens, local_idx = _make_ragged_cp_case(
                partition_mode, self.cp_size, self.cp_rank
            )

            torch.manual_seed(_SEED + 700 + layer_number)
            model_parallel_cuda_manual_seed(_SEED + 700 + layer_number)
            config = _make_dsv4_cp_config(
                context_parallel_size=self.cp_size,
                dsa_indexer_loss_coeff=1.0,
                dsa_indexer_use_sparse_loss=True,
                apply_dsa_kernel_fusion=fused,
                apply_rope_fusion=True,
                csa_cp_partition_mode=partition_mode,
            )
            graph_attn = _build_attention(
                config, layer_number=layer_number, pg_collection=self.pg
            ).cuda()
            eager_attn = _build_attention(
                config, layer_number=layer_number, pg_collection=self.pg
            ).cuda()
            graph_attn.train()
            eager_attn.train()
            _copy_module_parameters(graph_attn, eager_attn)

            full_hidden = torch.randn(
                padded_tokens, 1, config.hidden_size, dtype=torch.bfloat16, device='cuda'
            )
            test_hidden = full_hidden.index_select(0, local_idx).detach().clone()
            test_grad = torch.randn_like(test_hidden)
            static_hidden = test_hidden.detach().clone().requires_grad_(True)
            eager_hidden = test_hidden.detach().clone().requires_grad_(True)
            static_grad = test_grad.detach().clone()

            graph, graph_output = _capture_dsv4_attention_forward_backward(
                graph_attn, static_hidden, static_grad, packed
            )
            with torch.no_grad():
                static_hidden.copy_(test_hidden)
                static_grad.copy_(test_grad)
                if static_hidden.grad is not None:
                    static_hidden.grad.zero_()
                for param in graph_attn.parameters():
                    if param.grad is not None:
                        param.grad.zero_()
            graph.replay()
            torch.cuda.synchronize()
            graph_out = graph_output.detach().clone()
            graph_hidden_grad = static_hidden.grad.detach().clone()
            graph_param_grads = {
                name: param.grad.detach().clone()
                for name, param in graph_attn.named_parameters()
                if param.grad is not None
            }

            eager_out, eager_hidden_grad, eager_param_grads = _run_dsv4_attention_forward_backward(
                eager_attn, eager_hidden, test_grad, packed
            )
            torch.cuda.synchronize()
            assert graph_param_grads.keys() == eager_param_grads.keys()
            output_label = f"layer={layer_number}:{mode}:{partition_mode}:output"
            _assert_cp_graph_bitwise_match(graph_out, eager_out, output_label)
            bwd_match_fn = (
                _assert_cp_graph_fused_backward_match
                if fused
                else _assert_cp_graph_bitwise_match
            )
            bwd_match_fn(
                graph_hidden_grad,
                eager_hidden_grad,
                f"layer={layer_number}:{mode}:{partition_mode}:hidden_grad",
            )
            for name, graph_grad in graph_param_grads.items():
                bwd_match_fn(
                    graph_grad,
                    eager_param_grads[name],
                    f"layer={layer_number}:{mode}:{partition_mode}:param_grad:{name}",
                )

            del graph, graph_output
            del graph_attn, eager_attn, full_hidden, test_hidden, test_grad, static_hidden
            del eager_hidden, static_grad, graph_out, graph_hidden_grad
            _clear_cuda_test_state()

    @pytest.mark.parametrize(
        "layer_number",
        [2, 3],
        ids=["ratio_4_indexer", "ratio_128_compressor"],
    )
    @pytest.mark.parametrize(
        "partition_mode",
        [DSV4_CP_PARTITION_CONTIGUOUS, DSV4_CP_PARTITION_TWO_CHUNK],
        ids=["contiguous", "two_chunk"],
    )
    def test_thd_cp_cuda_graph_replay_accepts_changed_padded_boundaries(
        self, layer_number, partition_mode
    ):
        """CUDA graph replay uses updated device cu_seqlens_padded values.

        The graph is captured with one padded THD layout, then replayed after
        overwriting the same PackedSeqParams metadata tensors with another
        layout whose tensor shapes, padded total, and max seqlen are unchanged.
        Expected: replay matches eager execution with the new metadata. A
        failure means a CP compressed-KV kernel baked capture-time padded
        boundaries or host-derived dynamic shape into the graph.
        """
        capture_packed = _make_thd_packed_seq_params(
            _DSV4_CP_RAGGED_SEG_LENS,
            _DSV4_CP_RAGGED_PADDED_SEG_LENS,
        )
        replay_packed = _make_thd_packed_seq_params(
            _DSV4_CP_RAGGED_SEG_LENS,
            _DSV4_CP_REPLAY_PADDED_SEG_LENS,
        )
        padded_tokens = sum(_DSV4_CP_RAGGED_PADDED_SEG_LENS)
        assert padded_tokens == sum(_DSV4_CP_REPLAY_PADDED_SEG_LENS)
        assert capture_packed.cu_seqlens_q.shape == replay_packed.cu_seqlens_q.shape
        assert (
            capture_packed.cu_seqlens_q_padded.shape
            == replay_packed.cu_seqlens_q_padded.shape
        )
        assert capture_packed.max_seqlen_q == replay_packed.max_seqlen_q
        assert capture_packed.max_seqlen_kv == replay_packed.max_seqlen_kv
        assert not torch.equal(
            capture_packed.cu_seqlens_q_padded, replay_packed.cu_seqlens_q_padded
        )
        local_idx = _make_cp_partition_indices(partition_mode, padded_tokens, self.cp_size)[
            self.cp_rank
        ]

        torch.manual_seed(_SEED + 1100 + layer_number)
        model_parallel_cuda_manual_seed(_SEED + 1100 + layer_number)
        config = _make_dsv4_cp_config(
            context_parallel_size=self.cp_size,
            dsa_indexer_loss_coeff=1.0,
            dsa_indexer_use_sparse_loss=True,
            apply_dsa_kernel_fusion=True,
            apply_rope_fusion=True,
            csa_cp_partition_mode=partition_mode,
        )
        graph_attn = _build_attention(
            config, layer_number=layer_number, pg_collection=self.pg
        ).cuda()
        eager_attn = _build_attention(
            config, layer_number=layer_number, pg_collection=self.pg
        ).cuda()
        graph_attn.train()
        eager_attn.train()
        _copy_module_parameters(graph_attn, eager_attn)

        capture_full_hidden = torch.randn(
            padded_tokens, 1, config.hidden_size, dtype=torch.bfloat16, device='cuda'
        )
        replay_full_hidden = torch.randn_like(capture_full_hidden)
        capture_hidden = capture_full_hidden.index_select(0, local_idx).detach().clone()
        replay_hidden = replay_full_hidden.index_select(0, local_idx).detach().clone()
        capture_grad = torch.randn_like(capture_hidden)
        replay_grad = torch.randn_like(replay_hidden)
        static_hidden = capture_hidden.detach().clone().requires_grad_(True)
        static_grad = capture_grad.detach().clone()

        graph, graph_output = _capture_dsv4_attention_forward_backward(
            graph_attn, static_hidden, static_grad, capture_packed
        )
        with torch.no_grad():
            capture_packed.cu_seqlens_q.copy_(replay_packed.cu_seqlens_q)
            capture_packed.cu_seqlens_kv.copy_(replay_packed.cu_seqlens_kv)
            capture_packed.cu_seqlens_q_padded.copy_(replay_packed.cu_seqlens_q_padded)
            capture_packed.cu_seqlens_kv_padded.copy_(replay_packed.cu_seqlens_kv_padded)
            static_hidden.copy_(replay_hidden)
            static_grad.copy_(replay_grad)
            if static_hidden.grad is not None:
                static_hidden.grad.zero_()
            for param in graph_attn.parameters():
                if param.grad is not None:
                    param.grad.zero_()
        graph.replay()
        torch.cuda.synchronize()
        graph_out = graph_output.detach().clone()
        graph_hidden_grad = static_hidden.grad.detach().clone()
        graph_param_grads = {
            name: param.grad.detach().clone()
            for name, param in graph_attn.named_parameters()
            if param.grad is not None
        }

        eager_hidden = replay_hidden.detach().clone().requires_grad_(True)
        eager_out, eager_hidden_grad, eager_param_grads = _run_dsv4_attention_forward_backward(
            eager_attn, eager_hidden, replay_grad, replay_packed
        )
        torch.cuda.synchronize()
        assert graph_param_grads.keys() == eager_param_grads.keys()
        _assert_cp_graph_bitwise_match(
            graph_out, eager_out, f"layer={layer_number}:metadata_replay:{partition_mode}:output"
        )
        _assert_cp_graph_fused_backward_match(
            graph_hidden_grad,
            eager_hidden_grad,
            f"layer={layer_number}:metadata_replay:{partition_mode}:hidden_grad",
        )
        for name, graph_grad in graph_param_grads.items():
            _assert_cp_graph_fused_backward_match(
                graph_grad,
                eager_param_grads[name],
                f"layer={layer_number}:metadata_replay:{partition_mode}:param_grad:{name}",
            )

        del graph, graph_output, graph_attn, eager_attn, capture_full_hidden, replay_full_hidden
        del capture_hidden, replay_hidden, capture_grad, replay_grad, static_hidden, static_grad
        del graph_out, graph_hidden_grad, eager_hidden
        _clear_cuda_test_state()

    @pytest.mark.parametrize(
        "layer_number",
        [1, 2, 3],
        ids=["ratio_0_window_only", "ratio_4_indexer", "ratio_128_compressor"],
    )
    @pytest.mark.parametrize(
        "partition_mode",
        [DSV4_CP_PARTITION_CONTIGUOUS, DSV4_CP_PARTITION_TWO_CHUNK],
        ids=["contiguous", "two_chunk"],
    )
    def test_thd_cp_peak_allocated_delta_scales_vs_cp1(self, layer_number, partition_mode):
        """CP-local THD forward/backward peak allocated delta scales down vs CP1.

        This is a black-box guard against implementations that pass parity but
        accidentally gather full hidden states or otherwise allocate global
        activation-sized buffers before the layer work.
        """
        packed, padded_tokens, local_idx = _make_ragged_cp_case(
            partition_mode, self.cp_size, self.cp_rank
        )

        ref_delta = self._measure_cp1_peak_allocated_delta(layer_number, packed, padded_tokens)

        torch.manual_seed(_SEED + 900 + layer_number)
        model_parallel_cuda_manual_seed(_SEED + 900 + layer_number)
        config_cp = _make_dsv4_cp_config(
            context_parallel_size=self.cp_size,
            dsa_indexer_loss_coeff=1.0,
            dsa_indexer_use_sparse_loss=True,
            apply_rope_fusion=True,
            csa_cp_partition_mode=partition_mode,
        )
        cp_attn = _build_attention(
            config_cp, layer_number=layer_number, pg_collection=self.pg
        ).cuda()
        full_hidden_values, full_grad_values = _make_hidden_and_grad(
            padded_tokens, config_cp.hidden_size
        )
        local_hidden = (
            full_hidden_values.index_select(0, local_idx).detach().clone().requires_grad_(True)
        )
        local_grad = full_grad_values.index_select(0, local_idx).detach().clone()
        del full_hidden_values, full_grad_values
        cp_delta = _measure_peak_allocated_delta(cp_attn, local_hidden, local_grad, packed)

        limit = _DSV4_CP_MEMORY_RATIO_LIMITS[self.cp_size]
        scale = float(cp_delta) / float(ref_delta)
        report = _format_scale_report(
            "memory_delta_mib",
            layer_number,
            self.cp_size,
            partition_mode,
            ref_delta / (1024.0**2),
            cp_delta / (1024.0**2),
            scale,
            limit,
        )
        print(report)
        assert scale <= limit, report

        del cp_attn, local_hidden, local_grad
        _clear_cuda_test_state()

    @pytest.mark.parametrize(
        "layer_number",
        [1, 2, 3],
        ids=["ratio_0_window_only", "ratio_4_indexer", "ratio_128_compressor"],
    )
    @pytest.mark.parametrize(
        "partition_mode",
        [DSV4_CP_PARTITION_CONTIGUOUS, DSV4_CP_PARTITION_TWO_CHUNK],
        ids=["contiguous", "two_chunk"],
    )
    def test_thd_cp_cuda_graph_time_scales_vs_cp1(self, layer_number, partition_mode):
        """CP-local CUDA graph replay time scales down vs CP1 on the same machine.

        This is a black-box performance guard: correctness and memory scaling
        are not enough if the implementation falls back to slow dynamic PyTorch
        work in the production graph path.
        """
        packed, padded_tokens, local_idx = _make_ragged_cp_case(
            partition_mode, self.cp_size, self.cp_rank
        )

        ref_time_ms = self._measure_cp1_cuda_graph_time(layer_number, packed, padded_tokens)

        torch.manual_seed(_SEED + 1000 + layer_number)
        model_parallel_cuda_manual_seed(_SEED + 1000 + layer_number)
        config_cp = _make_dsv4_cp_config(
            context_parallel_size=self.cp_size,
            dsa_indexer_loss_coeff=1.0,
            dsa_indexer_use_sparse_loss=True,
            apply_rope_fusion=True,
            csa_cp_partition_mode=partition_mode,
        )
        cp_attn = _build_attention(
            config_cp, layer_number=layer_number, pg_collection=self.pg
        ).cuda()
        full_hidden_values, full_grad_values = _make_hidden_and_grad(
            padded_tokens, config_cp.hidden_size
        )
        local_hidden = (
            full_hidden_values.index_select(0, local_idx).detach().clone().requires_grad_(True)
        )
        local_grad = full_grad_values.index_select(0, local_idx).detach().clone()
        del full_hidden_values, full_grad_values
        cp_time_ms = _measure_cuda_graph_time(cp_attn, local_hidden, local_grad, packed)

        limit = _DSV4_CP_GRAPH_TIME_RATIO_LIMITS[self.cp_size]
        scale = cp_time_ms / ref_time_ms
        report = _format_scale_report(
            "graph_time_ms",
            layer_number,
            self.cp_size,
            partition_mode,
            ref_time_ms,
            cp_time_ms,
            scale,
            limit,
        )
        print(report)
        assert scale <= limit, report

        del cp_attn, local_hidden, local_grad
        _clear_cuda_test_state()
