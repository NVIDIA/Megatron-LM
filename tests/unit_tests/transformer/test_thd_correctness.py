# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Compare THD format against SBHD format.

Test Strategy
-------------
1. Generate full (unsharded) data with deterministic seed on each rank.
2. Shard inputs for both SBHD and THD formats (zigzag CP, contiguous SP).
3. Forward pass through the same TransformerLayer.
4. Gather outputs back to full size (with gradient support).
5. Backward pass with format-specific grad_output handling.
6. Compare outputs and gradients with bitwise or similarity checks.

Check Levels
------------
- bitwise_all: B=1, forward + backward bitwise (MockCoreAttention)
- bitwise_fwd: B>1, forward bitwise, backward similarity (MockCoreAttention,
               THD padded to max_len so total tokens match SBHD)
- similarity:  All parallelism configs, real TE attention, similarity checks
"""

import os
from dataclasses import dataclass
from typing import List

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

from megatron.core import parallel_state
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer
from tests.unit_tests.test_utilities import Utils

# =============================================================================
# Constants
# =============================================================================

SIMILARITY_THRESHOLD = 0.999


# =============================================================================
# Test Cases
# =============================================================================


@dataclass
class TestCase:
    """Test case specification.

    check_level controls comparison strictness and attention implementation:
        "bitwise_all" - MockCoreAttention, forward + backward bitwise (B=1)
        "bitwise_fwd" - MockCoreAttention, forward bitwise, backward similarity
                        (B>1, THD padded to max_len to match SBHD total tokens)
        "similarity"  - Real TE attention, forward + backward similarity
    """

    name: str
    hidden_size: int
    num_heads: int
    num_kv_heads: int
    ffn_hidden_size: int
    seqlens: List[int]
    tp_size: int = 1
    cp_size: int = 1
    sp_enabled: bool = False
    check_level: str = "similarity"

    @property
    def use_mock_attention(self) -> bool:
        return self.check_level in ("bitwise_all", "bitwise_fwd")

    @property
    def forward_bitwise(self) -> bool:
        return self.check_level in ("bitwise_all", "bitwise_fwd")

    @property
    def backward_bitwise(self) -> bool:
        return self.check_level == "bitwise_all"

    @property
    def pad_thd_to_max(self) -> bool:
        """Pad each THD sequence to max_len so total tokens match SBHD."""
        return self.check_level == "bitwise_fwd"


# fmt: off
TEST_CASES = [
    # -------------------------------------------------------------------------
    # B=1: forward + backward bitwise (MockCoreAttention)
    # -------------------------------------------------------------------------
    #                   name              H    heads kv_h  ffn    seqlens                tp cp sp     check_level
    TestCase("b1_seq479_mha",           1024,  16,   16, 4096,   [479],                  1, 1, False, "bitwise_all"),
    TestCase("b1_seq1337_mha",          1024,  16,   16, 4096,   [1337],                 1, 1, False, "bitwise_all"),
    TestCase("b1_seq2047_gqa",          1024,  16,    4, 4096,   [2047],                 1, 1, False, "bitwise_all"),
    TestCase("b1_seq3891_gqa",          1024,  16,    4, 4096,   [3891],                 1, 1, False, "bitwise_all"),

    # -------------------------------------------------------------------------
    # B>1 single GPU: forward bitwise, backward similarity (MockCoreAttention)
    # THD is padded to max_len per sequence so TE GEMM sees the same M value
    # -------------------------------------------------------------------------
    TestCase("b2_uniform_513",          1024,  16,   16, 4096,   [513, 513],             1, 1, False, "bitwise_fwd"),
    TestCase("b3_uniform_997",          1024,  16,    4, 4096,   [997, 997, 997],        1, 1, False, "bitwise_fwd"),
    TestCase("varlen_pow2_512_1024",    1024,  16,   16, 4096,   [512, 1024],            1, 1, False, "bitwise_fwd"),
    TestCase("varlen_499_1031",         1024,  16,   16, 4096,   [499, 1031],            1, 1, False, "bitwise_fwd"),
    TestCase("varlen_263_751_503",      1024,  16,    4, 4096,   [263, 751, 503],        1, 1, False, "bitwise_fwd"),
    TestCase("varlen_mixed",            1024,  16,   16, 4096,   [1987, 523, 271, 1009], 1, 1, False, "bitwise_fwd"),
    TestCase("short_seqs",              1024,  16,   16, 4096,   [17, 31, 11],           1, 1, False, "bitwise_fwd"),

    # -------------------------------------------------------------------------
    # TP/CP/SP: similarity checks (TE Attention)
    # -------------------------------------------------------------------------
    # Llama-7B style (H=4096, 32 heads)
    TestCase("tp2_llama7b",             4096,  32,   32, 11008,  [503, 1019],            2, 1, False, "similarity"),
    TestCase("tp2_sp_llama7b",          4096,  32,   32, 11008,  [499, 1031, 773],       2, 1, True,  "similarity"),
    TestCase("cp2_llama7b",             4096,  32,   32, 11008,  [1021, 2039],           1, 2, False, "similarity"),
    TestCase("cp4_llama7b",             4096,  32,   32, 11008,  [1019, 509, 761],       1, 4, False, "similarity"),

    # Mixtral style (H=4096, 32 heads, 8 kv_heads)
    TestCase("tp2_mixtral",             4096,  32,    8, 14336,  [509, 1021],            2, 1, False, "similarity"),
    TestCase("tp4_sp_mixtral",          4096,  32,    8, 14336,  [1019, 503, 257],       4, 1, True,  "similarity"),

    # Llama-70B style (H=8192, 64 heads, 8 kv_heads)
    TestCase("tp2_cp2_llama70b",        8192,  64,    8, 28672,  [509, 1013],            2, 2, False, "similarity"),
    TestCase("tp2_cp2_sp_llama70b",     8192,  64,    8, 28672,  [2039, 1019, 509],      2, 2, True,  "similarity"),

    # Qwen3-235B style (H=4096, 64 heads, 4 kv_heads)
    TestCase("tp2_cp2_sp_qwen3",        4096,  64,    4, 12288,  [503, 1019],            2, 2, True,  "similarity"),
    TestCase("tp2_cp4_sp_qwen3",        4096,  64,    4, 12288,  [2039, 1013, 509],      2, 4, True,  "similarity"),

    # -------------------------------------------------------------------------
    # Edge cases
    # -------------------------------------------------------------------------
    TestCase("short_seqs_parallel",     1024,  16,    4, 4096,   [17, 31, 11],           2, 2, True,  "similarity"),
    TestCase("extreme_mixed",           4096,  32,    8, 14336,  [4093, 127, 257],       2, 2, True,  "similarity"),
]
# fmt: on


# =============================================================================
# Padding Helpers
# =============================================================================


def _round_up(value: int, divisor: int) -> int:
    return value if divisor <= 1 else (value + divisor - 1) // divisor * divisor


def compute_sbhd_padded_max_len(
    seqlens: List[int], cp_size: int, tp_size: int, sp_enabled: bool
) -> int:
    """Padded max_len for SBHD.

    Must be divisible by:
    - cp_size * 2 for zigzag CP sharding (if cp_size > 1)
    - tp_size for SP sharding along sequence dim (if sp_enabled)
    """
    divisor = 1
    if cp_size > 1:
        divisor *= cp_size * 2
    if sp_enabled:
        divisor *= tp_size
    return _round_up(max(seqlens), divisor)


def compute_thd_padded_seqlens(
    seqlens: List[int], cp_size: int, tp_size: int, sp_enabled: bool, pad_to_max: bool = False
) -> List[int]:
    """Padded per-sequence lengths for THD.

    When pad_to_max=True, each sequence is padded to max(seqlens) so that
    total THD tokens = max_len * B, matching SBHD. This ensures TE GEMM
    kernels see identical M dimensions for bitwise comparison.
    """
    cp_divisor = 2 * cp_size if cp_size > 1 else 1
    if pad_to_max:
        max_len = _round_up(max(seqlens), cp_divisor)
        padded = [max_len] * len(seqlens)
    else:
        padded = [_round_up(sl, cp_divisor) for sl in seqlens]
    if sp_enabled:
        remainder = sum(padded) % tp_size
        if remainder > 0:
            padded[-1] += tp_size - remainder
    return padded


# =============================================================================
# PackedSeqParams Helper
# =============================================================================


def make_packed_seq_params(
    seqlens: List[int],
    cp_size: int = 1,
    tp_size: int = 1,
    sp_enabled: bool = False,
    pad_to_max: bool = False,
) -> PackedSeqParams:
    """Create PackedSeqParams with cu_seqlens and cu_seqlens_padded."""

    def to_cu_seqlens(lens):
        cu = torch.zeros(len(lens) + 1, dtype=torch.int32)
        for i, l in enumerate(lens):
            cu[i + 1] = cu[i] + l
        return cu.cuda()

    padded = compute_thd_padded_seqlens(seqlens, cp_size, tp_size, sp_enabled, pad_to_max)
    return PackedSeqParams(
        cu_seqlens_q=to_cu_seqlens(seqlens),
        cu_seqlens_kv=to_cu_seqlens(seqlens),
        cu_seqlens_q_padded=to_cu_seqlens(padded),
        cu_seqlens_kv_padded=to_cu_seqlens(padded),
        max_seqlen_q=max(padded),
        max_seqlen_kv=max(padded),
        qkv_format='thd',
    )


# =============================================================================
# Mock Core Attention (for bitwise tests)
# =============================================================================


class MockCoreAttention(nn.Module):
    """Per-sequence unfused causal attention for bitwise comparison."""

    def __init__(
        self,
        config,
        layer_number,
        attn_mask_type,
        attention_type,
        attention_dropout=None,
        softmax_scale=None,
        cp_comm_type=None,
        pg_collection=None,
    ):
        super().__init__()
        self.num_q_heads = config.num_attention_heads
        self.num_kv_heads = config.num_query_groups
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.scale = 1.0 / (self.head_dim**0.5)
        self.num_rep = self.num_q_heads // self.num_kv_heads

    def _repeat_kv(self, x):
        """Repeat KV heads for GQA. [S, Hkv, D] -> [S, Hq, D]."""
        if self.num_rep == 1:
            return x
        S, Hkv, D = x.shape
        return x.unsqueeze(2).expand(S, Hkv, self.num_rep, D).reshape(S, self.num_q_heads, D)

    def _attention_single_seq(self, q, k, v):
        """Causal attention for one sequence."""
        S = q.shape[0]
        k, v = self._repeat_kv(k), self._repeat_kv(v)
        q, k, v = (x.transpose(0, 1).contiguous() for x in (q, k, v))
        q32, k32, v32 = q.float(), k.float(), v.float()
        scores = torch.matmul(q32, k32.transpose(-2, -1)) * self.scale
        mask = torch.triu(torch.ones(S, S, dtype=torch.bool, device=q.device), diagonal=1)
        scores.masked_fill_(mask, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v32)
        return out.transpose(0, 1).to(q.dtype).contiguous()

    def forward(
        self,
        query,
        key,
        value,
        attention_mask=None,
        attn_mask_type=None,
        attention_bias=None,
        packed_seq_params=None,
    ):
        if packed_seq_params is not None:
            # THD: [T, 1, H, D] -> [T, H, D]
            q = query.squeeze(1) if query.dim() == 4 else query
            k = key.squeeze(1) if key.dim() == 4 else key
            v = value.squeeze(1) if value.dim() == 4 else value

            cu_valid = packed_seq_params.cu_seqlens_q.cpu().tolist()
            cu_padded = packed_seq_params.cu_seqlens_q_padded.cpu().tolist()
            num_seqs = len(cu_valid) - 1

            outputs = []
            for i in range(num_seqs):
                out_seq = self._attention_single_seq(
                    q[cu_padded[i] : cu_padded[i + 1]],
                    k[cu_padded[i] : cu_padded[i + 1]],
                    v[cu_padded[i] : cu_padded[i + 1]],
                )
                outputs.append(out_seq)

            return torch.cat(outputs, dim=0)  # [T_padded, Hq, D]

        else:
            # SBHD: [S, B, H, D]
            S, B = query.shape[:2]
            outputs = [
                self._attention_single_seq(query[:, b], key[:, b], value[:, b]) for b in range(B)
            ]
            return torch.stack(outputs, dim=1).reshape(S, B, self.hidden_size)


# =============================================================================
# Layer Builder
# =============================================================================


def build_gpt_layer(
    hidden_size: int,
    num_heads: int,
    num_kv_heads: int,
    ffn_hidden_size: int,
    tp_size: int = 1,
    cp_size: int = 1,
    sp_enabled: bool = False,
    use_mock_attention: bool = False,
    deterministic: bool = False,
) -> TransformerLayer:
    """Build GPT TransformerLayer, optionally with MockCoreAttention."""
    config = TransformerConfig(
        num_layers=1,
        hidden_size=hidden_size,
        ffn_hidden_size=ffn_hidden_size,
        num_attention_heads=num_heads,
        num_query_groups=num_kv_heads,
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        autocast_dtype=torch.bfloat16,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        tensor_model_parallel_size=tp_size,
        context_parallel_size=cp_size,
        sequence_parallel=sp_enabled,
        cp_comm_type="p2p" if cp_size > 1 else None,
        deterministic_mode=deterministic,
    )
    spec = get_gpt_layer_with_transformer_engine_spec()
    if use_mock_attention:
        spec.submodules.self_attention.submodules.core_attention = MockCoreAttention
    layer = TransformerLayer(config, spec.submodules)
    layer.cuda()
    return layer


# =============================================================================
# Sharding: full -> local
# =============================================================================


def _zigzag_split(tensor, cp_rank, cp_size, dim=0):
    """Split tensor along dim using zigzag pattern for CP.

    For cp_size=2: rank0 gets chunks [0,3], rank1 gets chunks [1,2]
    For cp_size=4: rank0 gets [0,7], rank1 gets [1,6], rank2 gets [2,5], rank3 gets [3,4]
    """
    if cp_size <= 1:
        return tensor
    chunk_size = tensor.shape[dim] // (2 * cp_size)
    i0, i1 = cp_rank, 2 * cp_size - cp_rank - 1
    chunk0 = tensor.narrow(dim, i0 * chunk_size, chunk_size)
    chunk1 = tensor.narrow(dim, i1 * chunk_size, chunk_size)
    return torch.cat([chunk0, chunk1], dim=dim)


def shard_sbhd(tensor, cp_rank, cp_size, tp_rank, tp_size, sp_enabled):
    """Shard SBHD tensor: zigzag CP, then contiguous SP."""
    out = _zigzag_split(tensor, cp_rank, cp_size)
    if sp_enabled:
        seg = out.shape[0] // tp_size
        out = out.narrow(0, tp_rank * seg, seg)
    return out.contiguous()


def shard_thd(
    seq_data_list, seqlens, cp_rank, cp_size, tp_rank, tp_size, sp_enabled, H, pad_to_max=False
):
    """Shard per-sequence data into local THD [local_T, 1, H]."""
    padded = compute_thd_padded_seqlens(seqlens, cp_size, tp_size, sp_enabled, pad_to_max)

    chunks = []
    for data, sl, psl in zip(seq_data_list, seqlens, padded):
        if psl > sl:
            data = torch.cat([data, torch.zeros(psl - sl, H, dtype=data.dtype, device=data.device)])
        chunks.append(_zigzag_split(data, cp_rank, cp_size))

    packed = torch.cat(chunks, dim=0)
    if sp_enabled:
        seg = packed.shape[0] // tp_size
        packed = packed[tp_rank * seg : (tp_rank + 1) * seg]
    return packed.unsqueeze(1).contiguous()


# =============================================================================
# Gathering: local -> full (with backward support)
# =============================================================================


def _zigzag_merge(chunks: List[torch.Tensor], cp_size: int) -> torch.Tensor:
    """Reconstruct full sequence from per-rank zigzag chunks."""
    half = chunks[0].shape[0] // 2
    parts = [None] * (2 * cp_size)
    for r in range(cp_size):
        parts[r] = chunks[r][:half]
        parts[2 * cp_size - r - 1] = chunks[r][half:]
    return torch.cat(parts, dim=0)


def _strip_thd_padding(tensor, seqlens, padded_seqlens):
    """Remove per-sequence padding from THD tensor, keeping autograd."""
    total_valid = sum(seqlens)
    if tensor.shape[0] <= total_valid:
        return tensor
    offset, seqs = 0, []
    for sl, psl in zip(seqlens, padded_seqlens):
        seqs.append(tensor[offset : offset + sl])
        offset += psl
    return torch.cat(seqs, dim=0)


class _GatherSBHD(torch.autograd.Function):
    """Gather SBHD outputs from all ranks with gradient support."""

    @staticmethod
    def forward(ctx, local, cp_size, tp_size, sp_enabled):
        ctx.cp_size, ctx.tp_size, ctx.sp_enabled = cp_size, tp_size, sp_enabled
        ctx.cp_rank = parallel_state.get_context_parallel_rank() if cp_size > 1 else 0
        ctx.tp_rank = parallel_state.get_tensor_model_parallel_rank()

        out = local
        if sp_enabled:
            gathered = [torch.empty_like(out) for _ in range(tp_size)]
            dist.all_gather(
                gathered, out.contiguous(), group=parallel_state.get_tensor_model_parallel_group()
            )
            out = torch.cat(gathered, dim=0)
        if cp_size > 1:
            gathered = [torch.empty_like(out) for _ in range(cp_size)]
            dist.all_gather(
                gathered, out.contiguous(), group=parallel_state.get_context_parallel_group()
            )
            out = _zigzag_merge(gathered, cp_size)
        return out

    @staticmethod
    def backward(ctx, grad):
        out = grad
        if ctx.cp_size > 1:
            out = _zigzag_split(out, ctx.cp_rank, ctx.cp_size)
        if ctx.sp_enabled:
            seg = out.shape[0] // ctx.tp_size
            out = out[ctx.tp_rank * seg : (ctx.tp_rank + 1) * seg]
        return out.contiguous(), None, None, None


class _GatherTHD(torch.autograd.Function):
    """Gather THD outputs from all ranks with gradient support."""

    @staticmethod
    def forward(ctx, local, seqlens, cp_size, tp_size, sp_enabled, H, pad_to_max):
        ctx.seqlens, ctx.cp_size, ctx.tp_size, ctx.sp_enabled, ctx.H = (
            seqlens,
            cp_size,
            tp_size,
            sp_enabled,
            H,
        )
        ctx.cp_rank = parallel_state.get_context_parallel_rank() if cp_size > 1 else 0
        ctx.tp_rank = parallel_state.get_tensor_model_parallel_rank()
        ctx.padded = compute_thd_padded_seqlens(seqlens, cp_size, tp_size, sp_enabled, pad_to_max)

        out = local
        if sp_enabled:
            gathered = [torch.empty_like(out) for _ in range(tp_size)]
            dist.all_gather(
                gathered, out.contiguous(), group=parallel_state.get_tensor_model_parallel_group()
            )
            out = torch.cat(gathered, dim=0)

        if cp_size > 1:
            cp_group = parallel_state.get_context_parallel_group()
            local_lens = [p // cp_size for p in ctx.padded]
            offset, seqs = 0, []
            for i, ll in enumerate(local_lens):
                chunk = out[offset : offset + ll]
                gathered = [torch.empty_like(chunk) for _ in range(cp_size)]
                dist.all_gather(gathered, chunk.contiguous(), group=cp_group)
                seqs.append(_zigzag_merge(gathered, cp_size)[: seqlens[i]])
                offset += ll
            out = torch.cat(seqs, dim=0)
        else:
            out = _strip_thd_padding(out, seqlens, ctx.padded)
        return out

    @staticmethod
    def backward(ctx, grad):
        offset, chunks = 0, []
        for sl, psl in zip(ctx.seqlens, ctx.padded):
            g = grad[offset : offset + sl, 0, :]
            if psl > sl:
                g = torch.cat([g, torch.zeros(psl - sl, ctx.H, dtype=g.dtype, device=g.device)])
            chunks.append(_zigzag_split(g, ctx.cp_rank, ctx.cp_size))
            offset += sl

        packed = torch.cat(chunks, dim=0)
        if ctx.sp_enabled:
            seg = packed.shape[0] // ctx.tp_size
            packed = packed[ctx.tp_rank * seg : (ctx.tp_rank + 1) * seg]
        return packed.unsqueeze(1).contiguous(), None, None, None, None, None, None


def gather_sbhd(local, cp_size, tp_size, sp_enabled):
    if cp_size == 1 and not sp_enabled:
        return local
    return _GatherSBHD.apply(local, cp_size, tp_size, sp_enabled)


def gather_thd(local, seqlens, cp_size, tp_size, sp_enabled, H, pad_to_max=False):
    return _GatherTHD.apply(local, seqlens, cp_size, tp_size, sp_enabled, H, pad_to_max)


# =============================================================================
# Comparison Helpers
# =============================================================================


def _cosine_sim(a, b):
    return torch.nn.functional.cosine_similarity(
        a.flatten().float().unsqueeze(0), b.flatten().float().unsqueeze(0)
    ).item()


def _tensor_sim(a, b):
    a, b = a.double(), b.double()
    denom = (a * a + b * b).sum()
    return (2.0 * (a * b).sum() / denom).item() if denom else 1.0


def assert_close(name, a, b, bitwise):
    """Assert tensors match (bitwise or similarity)."""
    if bitwise:
        assert torch.equal(
            a, b
        ), f"{name}: NOT bitwise equal, max diff = {(a-b).abs().max().item()}"
    else:
        cs, ts = _cosine_sim(a, b), _tensor_sim(a, b)
        assert cs > SIMILARITY_THRESHOLD, f"{name}: cosine sim = {cs:.6f} < {SIMILARITY_THRESHOLD}"
        assert ts > SIMILARITY_THRESHOLD, f"{name}: tensor sim = {ts:.6f} < {SIMILARITY_THRESHOLD}"


# =============================================================================
# Test Function
# =============================================================================


@pytest.mark.parametrize("tc", TEST_CASES, ids=lambda tc: tc.name)
def test_thd_format(tc: TestCase):
    """Compare THD vs SBHD format outputs and gradients."""
    H, seqlens = tc.hidden_size, tc.seqlens
    tp_size, cp_size, sp = tc.tp_size, tc.cp_size, tc.sp_enabled
    B = len(seqlens)
    pad_to_max = tc.pad_thd_to_max

    # Deterministic mode for bitwise tests
    if tc.forward_bitwise or tc.backward_bitwise:
        os.environ["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] = "0"
        torch.use_deterministic_algorithms(True, warn_only=True)

    Utils.initialize_model_parallel(
        tensor_model_parallel_size=tp_size, context_parallel_size=cp_size
    )
    model_parallel_cuda_manual_seed(42)

    deterministic = tc.forward_bitwise or tc.backward_bitwise
    layer = build_gpt_layer(
        H,
        tc.num_heads,
        tc.num_kv_heads,
        tc.ffn_hidden_size,
        tp_size,
        cp_size,
        sp,
        tc.use_mock_attention,
        deterministic,
    )

    cp_rank = parallel_state.get_context_parallel_rank()
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    dp_rank = parallel_state.get_data_parallel_rank()

    # Generate data
    torch.manual_seed(42 + dp_rank)
    seq_data = [torch.randn(sl, H, dtype=torch.bfloat16).cuda() for sl in seqlens]
    torch.manual_seed(142 + dp_rank)
    grad_per_seq = [torch.randn(sl, H, dtype=torch.bfloat16).cuda() for sl in seqlens]

    # Prepare SBHD
    max_len = compute_sbhd_padded_max_len(seqlens, cp_size, tp_size, sp)
    full_sbhd = torch.zeros(max_len, B, H, dtype=torch.bfloat16, device='cuda')
    grad_sbhd = torch.zeros_like(full_sbhd)
    for b, sl in enumerate(seqlens):
        full_sbhd[:sl, b] = seq_data[b]
        grad_sbhd[:sl, b] = grad_per_seq[b]

    # Prepare THD grad (valid tokens only, gather_thd backward handles re-padding)
    grad_thd = torch.cat(grad_per_seq, dim=0).unsqueeze(1)

    # --- SBHD forward/backward ---
    local_sbhd = shard_sbhd(full_sbhd, cp_rank, cp_size, tp_rank, tp_size, sp)
    input_sbhd = local_sbhd.detach().clone().requires_grad_(True)
    out_sbhd, _ = layer(hidden_states=input_sbhd)
    gathered_sbhd = gather_sbhd(out_sbhd, cp_size, tp_size, sp)
    gathered_sbhd.backward(grad_sbhd)
    sbhd_grads = {n: p.grad.clone() for n, p in layer.named_parameters()}
    layer.zero_grad()

    # --- THD forward/backward ---
    local_thd = shard_thd(seq_data, seqlens, cp_rank, cp_size, tp_rank, tp_size, sp, H, pad_to_max)
    packed_seq_params = make_packed_seq_params(seqlens, cp_size, tp_size, sp, pad_to_max)
    input_thd = local_thd.detach().clone().requires_grad_(True)
    out_thd, _ = layer(hidden_states=input_thd, packed_seq_params=packed_seq_params)
    gathered_thd = gather_thd(out_thd, seqlens, cp_size, tp_size, sp, H, pad_to_max)
    gathered_thd.backward(grad_thd)
    thd_grads = {n: p.grad.clone() for n, p in layer.named_parameters()}

    # --- Gradient sync ---
    # Reduce across DP*CP group (each DP/CP rank sees different data/tokens)
    dp_cp_group = parallel_state.get_data_parallel_group(with_context_parallel=True)
    for n in sbhd_grads:
        dist.all_reduce(sbhd_grads[n], group=dp_cp_group)
        dist.all_reduce(thd_grads[n], group=dp_cp_group)
    # SP params also need reduction across TP group
    if sp:
        tp_group = parallel_state.get_tensor_model_parallel_group()
        for n, p in layer.named_parameters():
            if getattr(p, "sequence_parallel", False):
                dist.all_reduce(sbhd_grads[n], group=tp_group)
                dist.all_reduce(thd_grads[n], group=tp_group)

    # --- Forward comparison ---
    offset = 0
    for b, sl in enumerate(seqlens):
        assert_close(
            f"seq[{b}] output",
            gathered_sbhd[:sl, b].detach(),
            gathered_thd[offset : offset + sl, 0].detach(),
            tc.forward_bitwise,
        )
        offset += sl

    # --- Backward comparison ---
    for n in sbhd_grads:
        if n in thd_grads:
            assert_close(f"grad[{n}]", sbhd_grads[n], thd_grads[n], tc.backward_bitwise)

    # --- Cleanup ---
    Utils.destroy_model_parallel()
    if tc.forward_bitwise or tc.backward_bitwise:
        torch.use_deterministic_algorithms(False)
        os.environ.pop("NVTE_ALLOW_NONDETERMINISTIC_ALGO", None)
