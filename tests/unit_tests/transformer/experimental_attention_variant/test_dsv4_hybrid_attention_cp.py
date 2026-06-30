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
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
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

# CUDA graph with fused kernels may not be deterministic against eager, so this
# path uses similarity plus rtol/atol gates. The unfused CUDA graph path is
# still checked for bitwise parity.
_DSV4_CP_GRAPH_FUSED_SIM_EPS = 1e-6
_DSV4_CP_GRAPH_FUSED_RTOL = 1e-6
_DSV4_CP_GRAPH_FUSED_BF16_ATOL = 1.0
_DSV4_CP_GRAPH_FUSED_FP32_ATOL = 2e-2

# Actual measured peak allocated delta scale is below these limits. They leave
# room for allocator noise while still catching regressions that allocate
# full-size activation buffers on each CP rank.
_DSV4_CP_MEMORY_RATIO_LIMITS = {2: 0.65, 4: 0.35}

# Actual measured graph replay scale is below these limits. They leave room for
# machine timing jitter while still catching regressions that reintroduce
# full-size work or slow dynamic operations into the CP layer path.
_DSV4_CP_GRAPH_TIME_RATIO_LIMITS = {2: 0.70, 4: 0.50}

_DSV4_CP_MEMORY_WARMUP_ITERS = 3
_DSV4_CP_GRAPH_TIMING_WARMUP_REPLAYS = 2
_DSV4_CP_GRAPH_TIMING_MEASURE_REPLAYS = 5
_DSV4_CP_TEST_VARIANT = "flash"

# Padded total is 4096, divisible by CP2/CP4. Only the final segment has tail
# padding, so the intermediate sequence boundaries match the unpadded layout.
_DSV4_CP_RAGGED_SEG_LENS = (1, 127, 1000, 23, 129, 900, 55, 257, 800, 95, 509, 148)
_DSV4_CP_RAGGED_PADDED_SEG_LENS = (1, 127, 1000, 23, 129, 900, 55, 257, 800, 95, 509, 200)
# Same shape, padded total, and max padded sequence length as
# _DSV4_CP_RAGGED_PADDED_SEG_LENS, but the padding is distributed across many
# sequences instead of only the tail. CUDA graph replay must tolerate this value
# change because the CP path rebuilds compressed-row metadata from device
# cu_seqlens_padded rather than from capture-time host sizes.
_DSV4_CP_REPLAY_PADDED_SEG_LENS = (8, 128, 1000, 32, 132, 904, 64, 260, 804, 96, 512, 156)


def _dsv4_cp_fused_kernels_available():
    """Return whether this host can run the fused DSv4 THD CP test path."""
    if not torch.cuda.is_available():
        return False
    try:
        sm_major = torch.cuda.get_device_capability()[0]
    except RuntimeError:
        return False
    if sm_major < 10:
        return False
    try:
        from cudnn import DSA  # noqa: F401
        from flash_mla import flash_mla_sparse_fwd  # noqa: F401
    except ImportError:
        return False
    return True


_DSV4_CP_FUSED_KERNELS_UNAVAILABLE_REASON = (
    "DSv4 fused DSA cases require SM100+, flash_mla, and cudnn.DSA"
)


class _ReferenceCPGroup:
    def rank(self):
        """Return the CP rank used by the CP1 reference path."""
        return 0

    def size(self):
        """Return the CP size used by the CP1 reference path."""
        return 1


def _make_thd_packed_seq_params(seg_lens, padded_seg_lens=None, device='cuda'):
    """Build THD PackedSeqParams from raw and padded segment lengths."""
    if padded_seg_lens is None:
        padded_seg_lens = seg_lens
    cu = torch.tensor(
        [0] + list(torch.tensor(seg_lens).cumsum(0).tolist()), dtype=torch.int32, device=device
    )
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
        cp_partition_mode='contiguous',
    )


def _cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Return cosine similarity between two tensors as a Python float."""
    return F.cosine_similarity(
        a.flatten().double().unsqueeze(0), b.flatten().double().unsqueeze(0)
    ).item()


def _tensor_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Return scale-invariant tensor similarity between two tensors."""
    a, b = a.double(), b.double()
    denom = (a * a + b * b).sum()
    return (2.0 * (a * b).sum() / denom).item() if denom else 1.0


def _assert_cp_tensor_match(actual: torch.Tensor, expected: torch.Tensor, label: str):
    """Assert CP output matches reference by similarity metrics."""
    assert (
        actual.shape == expected.shape
    ), f"{label}: shape {tuple(actual.shape)} != {tuple(expected.shape)}"
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


def _assert_cp_graph_bitwise_match(actual: torch.Tensor, expected: torch.Tensor, label: str):
    """Assert graph replay output is bitwise equal to eager output."""
    assert (
        actual.shape == expected.shape
    ), f"{label}: shape {tuple(actual.shape)} != {tuple(expected.shape)}"
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


def _assert_cp_graph_fused_grad_match(actual: torch.Tensor, expected: torch.Tensor, label: str):
    """Assert fused graph gradients match eager within fused-kernel tolerances."""
    assert (
        actual.shape == expected.shape
    ), f"{label}: shape {tuple(actual.shape)} != {tuple(expected.shape)}"
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
        raise AssertionError(
            f"{label}: unsupported dtype for fused graph close check: {actual.dtype}"
        )
    try:
        torch.testing.assert_close(
            actual,
            expected,
            rtol=_DSV4_CP_GRAPH_FUSED_RTOL,
            atol=atol,
            msg=(
                f"{label}: fused graph/eager gradient mismatch; "
                f"rtol={_DSV4_CP_GRAPH_FUSED_RTOL}, atol={atol}, "
                f"cosine_sim={cosine_sim:.10f}, tensor_sim={tensor_sim:.10f}, "
                f"max_abs={max_abs:.6e}"
            ),
        )
    except AssertionError:
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else -1
        print(
            f"[rank{rank}] {label}: fused gradient close failed; "
            f"rtol={_DSV4_CP_GRAPH_FUSED_RTOL}, atol={atol}, "
            f"cosine_sim={cosine_sim:.10f}, tensor_sim={tensor_sim:.10f}, "
            f"max_abs={max_abs:.6e}",
            flush=True,
        )
        raise


@contextmanager
def _deterministic_torch_algorithms():
    """Temporarily enable deterministic PyTorch algorithms."""
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
    dsa_indexer_loss_coeff=1.0,
    dsa_indexer_use_sparse_loss=True,
    apply_dsa_kernel_fusion=True,
    apply_rope_fusion=True,
):
    """Build the DSv4 flash attention config used by CP tests."""
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
        cp_partition_mode="contiguous" if context_parallel_size > 1 else "zigzag",
        sequence_packing_scheduler="dp_balanced" if context_parallel_size > 1 else None,
        csa_dense_mode=False,
        csa_compress_rotary_base=shape["csa_compress_rotary_base"],
        layernorm_epsilon=1e-6,
        normalization="RMSNorm",
        qk_layernorm=True,
        layernorm_zero_centered_gamma=False,
        expert_model_parallel_size=1,
        apply_dsa_kernel_fusion=apply_dsa_kernel_fusion,
        apply_rope_fusion=apply_rope_fusion,
    )


def _copy_module_parameters(src, dst):
    """Copy named parameters from ``src`` into ``dst``."""
    src_params = dict(src.named_parameters())
    for name, param in dst.named_parameters():
        assert name in src_params
        param.data.copy_(src_params[name].data)


def _make_ragged_cp_case(cp_size, cp_rank):
    """Build the ragged THD packed params and local rows for one CP rank."""
    padded_total_tokens = sum(_DSV4_CP_RAGGED_PADDED_SEG_LENS)
    packed = _make_thd_packed_seq_params(_DSV4_CP_RAGGED_SEG_LENS, _DSV4_CP_RAGGED_PADDED_SEG_LENS)
    local_rows = padded_total_tokens // cp_size
    local_indices = torch.arange(cp_rank * local_rows, (cp_rank + 1) * local_rows, device='cuda')
    return packed, padded_total_tokens, local_indices


def _make_hidden_and_grad(padded_total_tokens, hidden_size):
    """Create matching hidden states and output gradients for a layer run."""
    hidden = torch.randn(padded_total_tokens, 1, hidden_size, dtype=torch.bfloat16, device='cuda')
    return hidden, torch.randn_like(hidden)


def _run_dsv4_attention_forward_backward(
    attn, hidden, grad, packed_seq_params, *, collect_result=True
):
    """Run one eager DSv4 attention forward/backward pass."""
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
    """Capture a DSv4 attention forward/backward pass in a CUDA graph."""
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
            hidden_states=static_hidden, attention_mask=None, packed_seq_params=packed_seq_params
        )
        graph_output.backward(static_grad)
    torch.cuda.synchronize()
    return graph, graph_output


def _zero_existing_grads(attn, hidden):
    """Zero existing parameter and hidden gradients in place."""
    if hidden.grad is not None:
        hidden.grad.zero_()
    for param in attn.parameters():
        if param.grad is not None:
            param.grad.zero_()


def _run_dsv4_attention_forward_backward_reuse_grads(attn, hidden, grad, packed_seq_params):
    """Run eager forward/backward while reusing existing grad buffers."""
    _zero_existing_grads(attn, hidden)
    output, _ = attn(hidden_states=hidden, attention_mask=None, packed_seq_params=packed_seq_params)
    output.backward(grad)
    return output


def _max_int_across_world(value: int) -> int:
    """Return the max integer value across distributed ranks."""
    tensor = torch.tensor(value, dtype=torch.int64, device='cuda')
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return int(tensor.item())


def _max_float_across_world(value: float) -> float:
    """Return the max floating-point value across distributed ranks."""
    tensor = torch.tensor(value, dtype=torch.float64, device='cuda')
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def _measure_peak_allocated_delta(attn, hidden, grad, packed_seq_params) -> int:
    """Measure post-warmup peak allocated memory delta for one layer pass."""
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
    """Measure slowest-rank median CUDA graph replay time."""
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


def _format_scale_report(metric, layer_number, cp_size, baseline, measured, scale, limit):
    """Format a CP-vs-CP1 scale report line."""
    return (
        f"DSv4 THD CP {metric} scale layer={layer_number} cp{cp_size}: "
        f"cp1={baseline:.6f}, cp{cp_size}={measured:.6f}, "
        f"scale={scale:.6f}, limit={limit:.6f}"
    )


def _clear_cuda_test_state():
    """Synchronize and release cached CUDA memory between cases."""
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
        """Initialize model-parallel groups for each CP size."""
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
        cls.fused_kernels_available = _dsv4_cp_fused_kernels_available()
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
        """Clear CUDA state before and after each test case."""
        _clear_cuda_test_state()
        yield
        _clear_cuda_test_state()

    def _measure_cp1_peak_allocated_delta(self, layer_number, packed, padded_tokens):
        """Measure and cache CP1 peak memory for a layer."""
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
        """Measure and cache CP1 CUDA graph replay time for a layer."""
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
        "layer_number",
        [1, 2, 3],
        ids=["ratio_0_window_only", "ratio_4_indexer", "ratio_128_compressor"],
    )
    @pytest.mark.parametrize("apply_rope_fusion", [True, False], ids=["fused_rope", "unfused_rope"])
    def test_thd_cp_matches_full_reference_forward_backward(self, layer_number, apply_rope_fusion):
        """CP path matches the full-sequence THD reference on ragged
        packed inputs using the DSv4 layer configuration.

        Verifies local CP output, hidden grad, and reduced parameter grads
        against the sliced full-reference tensors.
        """
        packed, padded_tokens, local_idx = _make_ragged_cp_case(self.cp_size, self.cp_rank)

        torch.manual_seed(_SEED + layer_number)
        model_parallel_cuda_manual_seed(_SEED + layer_number)
        apply_dsa_kernel_fusion = self.fused_kernels_available
        config_cp = _make_dsv4_cp_config(
            context_parallel_size=self.cp_size,
            dsa_indexer_loss_coeff=1.0,
            dsa_indexer_use_sparse_loss=True,
            apply_dsa_kernel_fusion=apply_dsa_kernel_fusion,
            apply_rope_fusion=apply_rope_fusion,
        )
        config_ref = _make_dsv4_cp_config(
            context_parallel_size=1,
            dsa_indexer_loss_coeff=1.0,
            dsa_indexer_use_sparse_loss=True,
            # Numerical parity uses the unfused CP1 reference so failures point
            # at the CP path instead of non-CP fused RoPE behavior.
            apply_dsa_kernel_fusion=apply_dsa_kernel_fusion,
            apply_rope_fusion=False,
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
            f"layer={layer_number}:dsa={apply_dsa_kernel_fusion}:rope={apply_rope_fusion}:output",
        )

        grad = torch.randn_like(ref_out)
        local_out.backward(grad.index_select(0, local_idx))
        ref_out.backward(grad)
        _assert_cp_tensor_match(
            local_hidden.grad.detach(),
            ref_hidden.grad.index_select(0, local_idx),
            f"layer={layer_number}:dsa={apply_dsa_kernel_fusion}:rope={apply_rope_fusion}:hidden_grad",
        )

        ref_params = dict(ref_attn.named_parameters())
        for name, param in cp_attn.named_parameters():
            ref_grad = ref_params[name].grad
            assert param.grad is not None, f"Missing CP grad for {name}"
            assert ref_grad is not None, f"Missing reference grad for {name}"
            grad_sum = param.grad.detach().clone()
            dist.all_reduce(grad_sum, group=self.pg.cp)
            _assert_cp_tensor_match(
                grad_sum,
                ref_grad,
                f"layer={layer_number}:dsa={apply_dsa_kernel_fusion}:"
                f"rope={apply_rope_fusion}:param_grad:{name}",
            )

        del cp_attn, ref_attn, full_hidden, local_hidden, ref_hidden, local_out, ref_out, grad
        _clear_cuda_test_state()

    @pytest.mark.parametrize("apply_rope_fusion", [True, False], ids=["fused", "unfused"])
    def test_dynamic_cp_mla_up_proj_recompute_matches_eager(self, apply_rope_fusion):
        """Selective MLA recompute must retain the microbatch's dynamic CP group."""
        if apply_rope_fusion and not self.fused_kernels_available:
            pytest.skip(_DSV4_CP_FUSED_KERNELS_UNAVAILABLE_REASON)

        packed, padded_tokens, local_idx = _make_ragged_cp_case(self.cp_size, self.cp_rank)
        packed.local_cp_size = self.cp_size
        packed.cp_group = self.pg.cp

        torch.manual_seed(_SEED + 1300)
        model_parallel_cuda_manual_seed(_SEED + 1300)
        eager_config = _make_dsv4_cp_config(
            context_parallel_size=self.cp_size,
            dsa_indexer_loss_coeff=0.0,
            apply_dsa_kernel_fusion=False,
            apply_rope_fusion=apply_rope_fusion,
        )
        recompute_config = _make_dsv4_cp_config(
            context_parallel_size=self.cp_size,
            dsa_indexer_loss_coeff=0.0,
            apply_dsa_kernel_fusion=False,
            apply_rope_fusion=apply_rope_fusion,
        )
        recompute_config.recompute_granularity = "selective"
        recompute_config.recompute_modules = ["mla_up_proj"]

        eager_attn = _build_attention(
            eager_config, layer_number=2, pg_collection=self.ref_pg
        ).cuda()
        recompute_attn = _build_attention(
            recompute_config, layer_number=2, pg_collection=self.ref_pg
        ).cuda()
        eager_attn.train()
        recompute_attn.train()
        _copy_module_parameters(eager_attn, recompute_attn)

        full_hidden, full_grad = _make_hidden_and_grad(padded_tokens, eager_config.hidden_size)
        hidden = full_hidden.index_select(0, local_idx)
        grad = full_grad.index_select(0, local_idx)
        eager_result = _run_dsv4_attention_forward_backward(
            eager_attn, hidden.detach().clone().requires_grad_(True), grad, packed
        )
        recompute_result = _run_dsv4_attention_forward_backward(
            recompute_attn, hidden.detach().clone().requires_grad_(True), grad, packed
        )

        mode = f"dynamic_cp:rope_fused={apply_rope_fusion}"
        _assert_cp_tensor_match(recompute_result[0], eager_result[0], f"{mode}:output")
        _assert_cp_tensor_match(recompute_result[1], eager_result[1], f"{mode}:hidden_grad")
        assert recompute_result[2].keys() == eager_result[2].keys()
        for name, recompute_grad in recompute_result[2].items():
            _assert_cp_tensor_match(
                recompute_grad, eager_result[2][name], f"{mode}:param_grad:{name}"
            )

        del eager_attn, recompute_attn, full_hidden, full_grad, hidden, grad
        _clear_cuda_test_state()

    def test_thd_cp_ratio4_eval_matches_full_reference(self):
        """Ratio-4 inference lowers logical indexer top-k rows correctly."""
        packed, padded_tokens, local_idx = _make_ragged_cp_case(self.cp_size, self.cp_rank)

        torch.manual_seed(_SEED + 1202)
        model_parallel_cuda_manual_seed(_SEED + 1202)
        config_cp = _make_dsv4_cp_config(
            context_parallel_size=self.cp_size,
            apply_dsa_kernel_fusion=self.fused_kernels_available,
            apply_rope_fusion=True,
        )
        config_ref = _make_dsv4_cp_config(
            context_parallel_size=1,
            apply_dsa_kernel_fusion=self.fused_kernels_available,
            apply_rope_fusion=False,
        )
        cp_attn = _build_attention(config_cp, layer_number=2, pg_collection=self.pg).cuda().eval()
        ref_attn = (
            _build_attention(config_ref, layer_number=2, pg_collection=self.ref_pg).cuda().eval()
        )
        _copy_module_parameters(cp_attn, ref_attn)

        full_hidden = torch.randn(
            padded_tokens, 1, config_cp.hidden_size, dtype=torch.bfloat16, device='cuda'
        )
        local_hidden = full_hidden.index_select(0, local_idx)
        with torch.no_grad():
            local_out, _ = cp_attn(
                hidden_states=local_hidden, attention_mask=None, packed_seq_params=packed
            )
            ref_out, _ = ref_attn(
                hidden_states=full_hidden, attention_mask=None, packed_seq_params=packed
            )
        _assert_cp_tensor_match(
            local_out, ref_out.index_select(0, local_idx), "layer=2:eval:output"
        )

        del cp_attn, ref_attn, full_hidden, local_hidden, local_out, ref_out
        _clear_cuda_test_state()

    @pytest.mark.parametrize(
        "layer_number",
        [1, 2, 3],
        ids=["ratio_0_window_only", "ratio_4_indexer", "ratio_128_compressor"],
    )
    @pytest.mark.parametrize(
        ("dsa_fused", "rope_fused"),
        [(True, True), (False, True), (False, False)],
        ids=["fused", "unfused_dsa", "unfused_dsa_rope"],
    )
    def test_thd_cp_cuda_graph_matches_eager_forward_backward(
        self, layer_number, dsa_fused, rope_fused
    ):
        """CUDA graph replay matches eager THD CP forward/backward.

        Captures the DSv4 attention layer's CP-local forward and backward
        graph, replays it with fresh static-buffer contents, and compares the
        local output, hidden grad, and parameter grads against an eager module
        with identical weights. The unfused graph path is deterministic and
        must match bitwise for both forward and backward. Fused forward outputs
        are expected to match bitwise. Fused gradients use strict similarity
        plus elementwise ``assert_close`` gates because they may be
        nondeterministic against eager execution.
        """
        if (dsa_fused or rope_fused) and not self.fused_kernels_available:
            pytest.skip(_DSV4_CP_FUSED_KERNELS_UNAVAILABLE_REASON)

        context = nullcontext() if dsa_fused else _deterministic_torch_algorithms()
        mode = f"dsa_fused={dsa_fused}:rope_fused={rope_fused}"
        with context:
            packed, padded_tokens, local_idx = _make_ragged_cp_case(self.cp_size, self.cp_rank)

            torch.manual_seed(_SEED + 700 + layer_number)
            model_parallel_cuda_manual_seed(_SEED + 700 + layer_number)
            config = _make_dsv4_cp_config(
                context_parallel_size=self.cp_size,
                dsa_indexer_loss_coeff=1.0,
                dsa_indexer_use_sparse_loss=True,
                apply_dsa_kernel_fusion=dsa_fused,
                apply_rope_fusion=rope_fused,
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
            output_label = f"layer={layer_number}:{mode}:output"
            _assert_cp_graph_bitwise_match(graph_out, eager_out, output_label)
            bwd_match_fn = (
                _assert_cp_graph_fused_grad_match if dsa_fused else _assert_cp_graph_bitwise_match
            )
            bwd_match_fn(
                graph_hidden_grad, eager_hidden_grad, f"layer={layer_number}:{mode}:hidden_grad"
            )
            for name, graph_grad in graph_param_grads.items():
                bwd_match_fn(
                    graph_grad,
                    eager_param_grads[name],
                    f"layer={layer_number}:{mode}:param_grad:{name}",
                )

            del graph, graph_output
            del graph_attn, eager_attn, full_hidden, test_hidden, test_grad, static_hidden
            del eager_hidden, static_grad, graph_out, graph_hidden_grad
            _clear_cuda_test_state()

    @pytest.mark.parametrize(
        "layer_number", [2, 3], ids=["ratio_4_indexer", "ratio_128_compressor"]
    )
    def test_thd_cp_cuda_graph_replay_accepts_changed_padded_boundaries(self, layer_number):
        """CUDA graph replay uses updated device cu_seqlens_padded values.

        The graph is captured with one padded THD layout, then replayed after
        overwriting the same PackedSeqParams metadata tensors with another
        layout whose tensor shapes, padded total, and max seqlen are unchanged.
        Expected: replay matches eager execution with the new metadata. A
        failure means the CP path baked capture-time padded boundaries or a
        host-derived dynamic shape into the graph.
        """
        if not self.fused_kernels_available:
            pytest.skip(_DSV4_CP_FUSED_KERNELS_UNAVAILABLE_REASON)

        capture_packed = _make_thd_packed_seq_params(
            _DSV4_CP_RAGGED_SEG_LENS, _DSV4_CP_RAGGED_PADDED_SEG_LENS
        )
        replay_packed = _make_thd_packed_seq_params(
            _DSV4_CP_RAGGED_SEG_LENS, _DSV4_CP_REPLAY_PADDED_SEG_LENS
        )
        padded_tokens = sum(_DSV4_CP_RAGGED_PADDED_SEG_LENS)
        assert padded_tokens == sum(_DSV4_CP_REPLAY_PADDED_SEG_LENS)
        assert capture_packed.cu_seqlens_q.shape == replay_packed.cu_seqlens_q.shape
        assert capture_packed.cu_seqlens_q_padded.shape == replay_packed.cu_seqlens_q_padded.shape
        assert capture_packed.max_seqlen_q == replay_packed.max_seqlen_q
        assert capture_packed.max_seqlen_kv == replay_packed.max_seqlen_kv
        assert not torch.equal(
            capture_packed.cu_seqlens_q_padded, replay_packed.cu_seqlens_q_padded
        )
        local_rows = padded_tokens // self.cp_size
        local_idx = torch.arange(
            self.cp_rank * local_rows, (self.cp_rank + 1) * local_rows, device='cuda'
        )

        torch.manual_seed(_SEED + 1100 + layer_number)
        model_parallel_cuda_manual_seed(_SEED + 1100 + layer_number)
        config = _make_dsv4_cp_config(
            context_parallel_size=self.cp_size,
            dsa_indexer_loss_coeff=1.0,
            dsa_indexer_use_sparse_loss=True,
            apply_dsa_kernel_fusion=True,
            apply_rope_fusion=True,
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
            graph_out, eager_out, f"layer={layer_number}:metadata_replay:output"
        )
        _assert_cp_graph_fused_grad_match(
            graph_hidden_grad,
            eager_hidden_grad,
            f"layer={layer_number}:metadata_replay:hidden_grad",
        )
        for name, graph_grad in graph_param_grads.items():
            _assert_cp_graph_fused_grad_match(
                graph_grad,
                eager_param_grads[name],
                f"layer={layer_number}:metadata_replay:param_grad:{name}",
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
    def test_thd_cp_peak_allocated_delta_scales_vs_cp1(self, layer_number):
        """CP-local THD forward/backward peak allocated delta scales down vs CP1.

        This is a black-box guard against implementations that pass parity but
        accidentally gather full hidden states or otherwise allocate global
        activation-sized buffers before the layer work.
        """
        if not self.fused_kernels_available:
            pytest.skip(_DSV4_CP_FUSED_KERNELS_UNAVAILABLE_REASON)

        packed, padded_tokens, local_idx = _make_ragged_cp_case(self.cp_size, self.cp_rank)

        ref_delta = self._measure_cp1_peak_allocated_delta(layer_number, packed, padded_tokens)

        torch.manual_seed(_SEED + 900 + layer_number)
        model_parallel_cuda_manual_seed(_SEED + 900 + layer_number)
        config_cp = _make_dsv4_cp_config(
            context_parallel_size=self.cp_size,
            dsa_indexer_loss_coeff=1.0,
            dsa_indexer_use_sparse_loss=True,
            apply_rope_fusion=True,
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
    def test_thd_cp_cuda_graph_time_scales_vs_cp1(self, layer_number):
        """CP-local CUDA graph replay time scales down vs CP1 on the same machine.

        This is a black-box performance guard: correctness and memory scaling
        are not enough if the implementation falls back to slow dynamic PyTorch
        work in the production graph path.
        """
        if not self.fused_kernels_available:
            pytest.skip(_DSV4_CP_FUSED_KERNELS_UNAVAILABLE_REASON)

        packed, padded_tokens, local_idx = _make_ragged_cp_case(self.cp_size, self.cp_rank)

        ref_time_ms = self._measure_cp1_cuda_graph_time(layer_number, packed, padded_tokens)

        torch.manual_seed(_SEED + 1000 + layer_number)
        model_parallel_cuda_manual_seed(_SEED + 1000 + layer_number)
        config_cp = _make_dsv4_cp_config(
            context_parallel_size=self.cp_size,
            dsa_indexer_loss_coeff=1.0,
            dsa_indexer_use_sparse_loss=True,
            apply_rope_fusion=True,
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
            "graph_time_ms", layer_number, self.cp_size, ref_time_ms, cp_time_ms, scale, limit
        )
        print(report)
        assert scale <= limit, report

        del cp_attn, local_hidden, local_grad
        _clear_cuda_test_state()
