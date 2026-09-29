# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CSA2 packed layout, compressor, indexer and sparse-loss contracts.

Independent Torch formulas check compression, candidate selection and teacher loss.
Per-sequence runs validate packed sharing; GPU cases use real fused kernels.
Published V4.1 operator formulas are covered in test_dsv41_native_parity.py.
"""

from itertools import accumulate
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.fp8_utils import get_fp8_context, is_float8tensor
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.models.common.embeddings.yarn_rotary_pos_embedding import YarnRotaryEmbedding
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_experimental_attention_variant_module_spec,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant import csa
from megatron.core.transformer.experimental_attention_variant.csa import (
    CompressedSparseAttentionSubmodules,
    CompressorSubmodules,
)
from megatron.core.transformer.experimental_attention_variant.csa2 import (
    CompressedSparseAttention2,
    CSA2Compressor,
    CSA2Indexer,
    CSA2IndexerSubmodules,
    CSA2State,
)
from megatron.core.transformer.experimental_attention_variant.csa_utils import (
    csa2_candidates,
    csa2_indexer,
    fused_compressor,
)
from megatron.core.transformer.experimental_attention_variant.csa_utils.csa2_candidates import (
    CSA2CandidateBlocks,
    candidate_blocks_from_scores,
)
from megatron.core.transformer.experimental_attention_variant.csa_utils.csa2_indexer import (
    fused_csa2_indexer_loss,
    prepare_csa2_indexer_inputs,
)
from megatron.core.transformer.experimental_attention_variant.csa_utils.thd_utils import (
    build_csa2_thd_layout,
)
from megatron.core.transformer.experimental_attention_variant.dsa import (
    DSAIndexerLossAutoScaler,
    DSAIndexerLossLoggingHelper,
)
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.experimental_attention_variant.test_dsv41 import _make_config
from tests.unit_tests.transformer.experimental_attention_variant.test_indexer_quantization import (
    _reference_quantize,
)


@pytest.fixture
def pg_collection():
    # Native indexer-loss tests below do not need CUDA or Transformer Engine.
    if not torch.cuda.is_available():
        pytest.skip("Megatron RoPE requires CUDA")
    if not HAVE_TE:
        pytest.skip("Transformer Engine is not installed")
    Utils.initialize_model_parallel()
    model_parallel_cuda_manual_seed(1234)
    yield ProcessGroupCollection.use_mpu_process_groups()
    Utils.destroy_model_parallel()


def _layer(pg_collection, ratio, dtype=torch.float32, **overrides):
    config_values = dict(
        num_layers=1,
        csa_compress_ratios=[ratio],
        csa2_kv_source_layers=[0] if ratio else [],
        csa2_index_source_layers=[0] if ratio else [],
        csa2_candidate_source_layer=None,
        csa2_candidate_topk_blocks=0,
        csa2_candidate_block_size=0,
        dsa_indexer_topk=2,
    )
    config_values.update(overrides)
    config = _make_config(params_dtype=dtype, **config_values)
    layer = build_module(
        get_experimental_attention_variant_module_spec(config),
        config=config,
        layer_number=1,
        pg_collection=pg_collection,
    ).cuda()
    # Nontrivial query magnitudes, gates, and sink expose normalization/softmax mistakes.
    with torch.no_grad():
        for name, param in layer.named_parameters():
            if "norm.weight" in name:
                param.uniform_(0.7, 1.3)
            else:
                param.normal_(std=0.2)
    return layer


@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("fp8_param", [False, True])
@pytest.mark.parametrize("packed", [False, True])
def test_compressor_and_indexer_preserve_bf16_under_fp8(pg_collection, ratio, fp8_param, packed):
    """Exercise real TE init/autocast: protected GEMMs stay BF16, Q remains FP8."""
    from transformer_engine.pytorch.fp8 import FP8GlobalStateManager

    config = _make_config(
        params_dtype=torch.bfloat16,
        use_cpu_initialization=False,
        num_layers=1,
        csa_compress_ratios=[ratio],
        csa2_kv_source_layers=[0],
        csa2_index_source_layers=[0],
        csa2_candidate_source_layer=None,
        csa2_candidate_topk_blocks=0,
        csa2_candidate_block_size=0,
        hidden_size=128,
        q_lora_rank=64,
        v_head_dim=128,
        dsa_indexer_n_heads=32,
        dsa_indexer_head_dim=128,
        fp8="hybrid",
        fp8_recipe="tensorwise",
        fp8_param=fp8_param,
    )
    with get_fp8_context(config, is_init=True):
        layer = build_module(
            get_experimental_attention_variant_module_spec(config),
            config=config,
            layer_number=1,
            pg_collection=pg_collection,
        ).cuda()
    compressor, indexer = layer.core_attention.compressor, layer.core_attention.indexer
    protected = {
        "wkv": compressor.linear_wkv,
        "wk": indexer.linear_wk,
        "weights": indexer.linear_weights_proj,
    }
    if ratio == 2:
        protected["wgate"] = compressor.linear_wgate
    for linear in protected.values():
        assert not is_float8tensor(linear.weight)
        assert linear.weight.dtype == torch.bfloat16
    assert is_float8tensor(indexer.linear_wq_b.weight) == fp8_param

    calls = {}

    def record_linear(name):
        def record(module, args, output):
            calls[name] = (FP8GlobalStateManager.is_fp8_enabled(), args[0].dtype, output[0].dtype)

        return record

    handles = [
        linear.register_forward_hook(record_linear(name))
        for name, linear in {**protected, "q": indexer.linear_wq_b}.items()
    ]
    x = torch.randn(64, 1, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    qr = torch.randn(64, 1, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    metadata = {}
    if packed:
        params, _, _ = _packed([32, 32], [32, 32], device="cuda")
        metadata["thd_layout"] = build_csa2_thd_layout(params, x.shape[0])
    try:
        with get_fp8_context(config):
            assert FP8GlobalStateManager.is_fp8_enabled()
            latent = compressor(x, thd_layout=metadata.get("thd_layout"))
            if packed:
                latent, metadata["compressed_layout"] = latent
            q, k, weights = indexer._project_inputs(
                x, qr, latent, layer.core_attention.rotary_pos_emb, **metadata
            )
            assert FP8GlobalStateManager.is_fp8_enabled(), "Must restore the outer FP8 context"
        sum(t.float().square().mean() for t in (latent, q, k, weights)).backward()
    finally:
        for handle in handles:
            handle.remove()
    assert calls == {
        **{name: (False, torch.bfloat16, torch.bfloat16) for name in protected},
        "q": (True, torch.bfloat16, torch.bfloat16),
    }
    for tensor in (x, qr, *(linear.weight for linear in protected.values())):
        assert tensor.grad is not None
        assert tensor.grad.dtype == torch.bfloat16
        assert torch.isfinite(tensor.grad).all()


# Cross-layer sharing and hierarchical candidate selection


# Integration with the existing TransformerBlock and HybridStack


# Native CPU indexer-loss supervision and shared autograd graphs.
# The teacher oracle normalizes joint global, window, and sink logits. Native
# projection/norm adapters retain the production compressor, indexer, RoPE, and loss.


def _groups():
    # A real singleton ProcessGroup supplies size/rank without starting collectives.
    groups = ProcessGroupCollection()
    groups.tp = groups.cp = torch.distributed.ProcessGroup(0, 1)
    return groups


def _math_core(sparse=False, per_token=False, coefficient=0.3):
    core = CompressedSparseAttention2.__new__(CompressedSparseAttention2)
    nn.Module.__init__(core)
    core.config = SimpleNamespace(
        dsa_indexer_loss_coeff=coefficient,
        dsa_indexer_use_sparse_loss=sparse,
        calculate_per_token_loss=per_token,
    )
    core.pg_collection = _groups()
    core.softmax_scale = 0.5
    core.attn_sink = nn.Parameter(torch.tensor([-1.0, 2.0, 4.0]))
    return core


def _inputs(dtype=torch.float32, candidates=False, empty=False):
    torch.manual_seed(102)
    length, batch, heads, dim = (1 if empty else 7), 2, 3, 4
    width = length // 2
    query = torch.randn(length, batch, heads, dim, dtype=dtype, requires_grad=True)
    local_kv = torch.randn(length, batch, dim, dtype=dtype, requires_grad=True)
    global_kv = torch.randn(width, batch, dim, dtype=dtype, requires_grad=True)
    raw_scores = torch.randn(batch, length, width, requires_grad=True)
    visible = torch.arange(width)[None, :] < torch.arange(1, length + 1)[:, None] // 2
    visible = visible.expand(batch, -1, -1).clone()
    if candidates:
        visible[0, 5:, 1] = False
        visible[1, 3:, 0] = False
    scores = raw_scores.masked_fill(~visible, -torch.inf)
    topk = scores.detach().topk(min(2, width), dim=-1).indices
    topk = topk.masked_fill(~scores.detach().gather(-1, topk).isfinite(), -1)
    window = torch.full((batch, length, 2), -1, dtype=torch.int64)
    for row in range(length):
        positions = torch.arange(max(row - 1, 0), row + 1)
        window[:, row, : positions.numel()] = positions
    return (query, local_kv, global_kv, window, topk, scores), raw_scores, visible


def _teacher_oracle(core, query, local_kv, global_kv, window, topk, scores):
    """Independent per-query KL, including window/sink mass before summing heads."""
    losses = []
    for batch in range(scores.shape[0]):
        for row in range(scores.shape[1]):
            positions = scores[batch, row].detach().isfinite().nonzero().flatten()
            if core.config.dsa_indexer_use_sparse_loss:
                positions = positions[torch.isin(positions, topk[batch, row])]
            if positions.numel() == 0:
                losses.append(scores[batch, row, :0].sum() * 0)
                continue
            q = query[row, batch].detach().float()
            selected = global_kv[positions, batch].detach().float()
            global_logits = (q @ selected.T) * core.softmax_scale
            local_positions = window[batch, row]
            local_positions = local_positions[local_positions >= 0]
            local = local_kv[local_positions, batch].detach().float()
            window_logits = (q @ local.T) * core.softmax_scale
            all_logits = torch.cat(
                (global_logits, window_logits, core.attn_sink.detach()[:, None]), dim=-1
            )
            teacher = all_logits.softmax(-1)[:, : positions.numel()].sum(0)
            teacher = teacher / teacher.sum()
            log_prediction = scores[batch, row, positions].float().log_softmax(-1)
            losses.append((teacher * (teacher.clamp_min(1e-10).log() - log_prediction)).sum())
    loss = torch.stack(losses).sum() * core.config.dsa_indexer_loss_coeff
    if not core.config.calculate_per_token_loss:
        loss = loss / (scores.shape[0] * scores.shape[1])
    return loss


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("per_token", [False, True])
@pytest.mark.parametrize("candidates", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_loss_and_score_gradients_match_joint_teacher(sparse, per_token, candidates, dtype):
    core = _math_core(sparse, per_token)
    args, raw_scores, visible = _inputs(dtype, candidates)
    scores_before = args[-1].detach().clone()
    reference_raw = raw_scores.detach().clone().requires_grad_()
    reference_scores = reference_raw.masked_fill(~visible, -torch.inf)
    expected = _teacher_oracle(core, *args[:-1], reference_scores)
    actual = core._compute_indexer_loss(*args)
    torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-6)
    torch.testing.assert_close(args[-1], scores_before)
    actual.backward()
    expected.backward()
    torch.testing.assert_close(raw_scores.grad, reference_raw.grad, atol=2e-7, rtol=2e-6)
    assert torch.isfinite(raw_scores.grad).all()
    assert raw_scores.grad.abs().sum() > 0
    assert torch.count_nonzero(raw_scores.grad.masked_select(~visible)) == 0
    assert all(tensor.grad is None for tensor in args[:3])
    assert core.attn_sink.grad is None


def _record_losses(monkeypatch):
    records = []

    def record(**kwargs):
        assert not kwargs["loss"].requires_grad
        records.append(kwargs)

    monkeypatch.setattr(DSAIndexerLossLoggingHelper, "save_loss_to_tracker", record)
    monkeypatch.setattr(DSAIndexerLossAutoScaler, "main_loss_backward_scale", None)
    return records


# Packed layout and padding-producer contracts.


def _params(valid, physical=None, max_seqlen=None):
    cu = torch.tensor(valid, dtype=torch.int32)
    padded = None if physical is None else torch.tensor(physical, dtype=torch.int32)
    lengths = cu.diff() if padded is None else padded.diff()
    if max_seqlen is None:
        max_seqlen = max(lengths.tolist(), default=0)
    return PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu.clone(),
        cu_seqlens_q_padded=padded,
        cu_seqlens_kv_padded=None if padded is None else padded.clone(),
        max_seqlen_q=max_seqlen,
        max_seqlen_kv=max_seqlen,
        pad_between_seqs=physical is not None,
    )


def _check_compression(layout, ratio):
    """Independent per-sequence enumeration checks addresses and validity."""
    compressed = layout.for_compression(ratio)
    valid = layout.valid_tokens.tolist()
    physical = layout.cu_seqlens_padded.tolist()
    source_rows, valid_groups, positions, seq_ids = [], [], [], []
    physical_prefix, valid_prefix = [0], [0]
    for seq_id, (start, end) in enumerate(zip(physical, physical[1:])):
        group_count = (end - start) // ratio
        seq_valid = 0
        for local_group in range(group_count):
            sources = list(range(start + ratio * local_group, start + ratio * (local_group + 1)))
            is_valid = all(valid[source] for source in sources)
            source_rows.append(sources)
            valid_groups.append(is_valid)
            positions.append(ratio * local_group if is_valid else 0)
            seq_ids.append(seq_id)
            seq_valid += is_valid
        physical_prefix.append(physical_prefix[-1] + group_count)
        valid_prefix.append(valid_prefix[-1] + seq_valid)
    expected_capacity = min(
        layout.total_tokens // ratio, (len(physical) - 1) * (layout.max_seqlen // ratio)
    )
    while len(source_rows) < expected_capacity:
        source_rows.append([0] * ratio)
        valid_groups.append(False)
        positions.append(0)
        seq_ids.append(-1)
    assert compressed.capacity == expected_capacity
    assert compressed.max_seqlen == layout.max_seqlen // ratio
    assert compressed.source_indices.shape == (compressed.capacity, ratio)
    assert compressed.source_indices.tolist() == source_rows
    assert compressed.valid_groups.tolist() == valid_groups
    assert compressed.position_ids.tolist() == positions
    assert compressed.sequence_ids.tolist() == seq_ids
    assert compressed.cu_seqlens_padded.tolist() == physical_prefix
    assert compressed.cu_seqlens.tolist() == valid_prefix
    assert compressed.cu_seqlens.dtype == layout.cu_seqlens.dtype
    return compressed


@pytest.mark.parametrize(
    "valid,physical,total,max_seqlen,expected_capacities",
    [
        pytest.param([0, 3], [0, 4], 64, 4, (4, 2), id="tail-exceeds-sequence-bound"),
        pytest.param([0, 1, 2], [0, 1, 2], 32, 1, (2, 0), id="max-shorter-than-r2"),
        pytest.param([0], [0], 16, 8, (0, 0), id="no-sequences-with-token-capacity"),
        pytest.param(
            [0, 2, 2, 2],
            [0, 2, 2, 2],
            64,
            4,
            (12, 6),
            id="metadata-capacity-includes-empty-entries",
        ),
    ],
)
def test_compressed_capacity_respects_dsv4_sequence_bound(
    valid, physical, total, max_seqlen, expected_capacities
):
    layout = build_csa2_thd_layout(_params(valid, physical, max_seqlen), total)
    for ratio, expected in zip((1, 2), expected_capacities):
        compressed = _check_compression(layout, ratio)
        assert compressed.capacity == expected
        assert compressed.capacity < total // ratio


@pytest.mark.parametrize(
    "valid,physical,total,max_seqlen,message",
    [
        ([1, 4], [1, 4], 4, 4, "start at zero"),
        ([0, 3, 2], [0, 3, 4], 4, 4, "monotonic"),
        ([0, 5], [0, 5], 4, 5, "exceeds physical token"),
        ([0, 4], [0, 3], 4, 4, "exceeds its physical segment"),
        ([0, 3], [0, 4], 4, 3, "exceeds max_seqlen"),
    ],
)
def test_invalid_prefix_values(valid, physical, total, max_seqlen, message):
    with pytest.raises(ValueError, match=message):
        build_csa2_thd_layout(_params(valid, physical, max_seqlen), total)


@pytest.mark.parametrize("mismatch", ["shape", "dtype", "values", "physical_values"])
def test_query_kv_metadata_must_agree(mismatch):
    params = _params([0, 2, 4], [0, 2, 4])
    if mismatch == "shape":
        params.cu_seqlens_kv = params.cu_seqlens_kv[:2]
    elif mismatch == "dtype":
        params.cu_seqlens_kv = params.cu_seqlens_kv.to(torch.int64)
    elif mismatch == "values":
        params.cu_seqlens_kv[1] = 1
    else:
        params.cu_seqlens_kv_padded[1] = 1
    with pytest.raises(ValueError, match="query/KV"):
        build_csa2_thd_layout(params, 4)


# Packed compressor/RoPE parity and shared DSv4 indexing regressions.


class _Linear(nn.Linear):
    """Ordinary PyTorch projection exposing Megatron's output/bias interface."""

    def __init__(self, input_size, output_size, config, **kwargs):
        super().__init__(input_size, output_size, bias=False, dtype=config.params_dtype)

    def forward(self, x):
        return super().forward(x), None


class _RMSNorm(nn.Module):
    """Native RMSNorm with FP32 statistics and activation-dtype output."""

    def __init__(self, hidden_size, eps, config):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=config.params_dtype))
        self.eps = eps

    def forward(self, x):
        return F.rms_norm(x.float(), (x.shape[-1],), self.weight.float(), self.eps).to(x.dtype)


def _config(dtype):
    return SimpleNamespace(
        fp8=None,
        fp4=None,
        fp8_param=False,
        fp4_param=False,
        attention_backend="unfused",
        dsa_kernel_backend="none",
        params_dtype=dtype,
        bf16=dtype == torch.bfloat16,
        hidden_size=6,
        q_lora_rank=4,
        v_head_dim=8,
        qk_pos_emb_head_dim=4,
        dsa_indexer_n_heads=2,
        dsa_indexer_head_dim=6,
        dsa_indexer_topk=2,
        attention_latent_norm_epsilon=1e-20,
        init_method=lambda weight: nn.init.normal_(weight, std=0.2),
        apply_rope_fusion=False,
        rotary_interleaved=False,
        multi_latent_attention=True,
        mrope_section=None,
    )


def _compressor(ratio, dtype, **overrides):
    config = _config(dtype)
    for name, value in overrides.items():
        setattr(config, name, value)
    module = CSA2Compressor(
        config=config,
        submodules=CompressorSubmodules(_Linear, _Linear, _RMSNorm),
        compress_ratio=ratio,
        pg_collection=_groups(),
    )
    with torch.no_grad():
        module.norm.weight.uniform_(0.7, 1.3)
    return module


def _packed(real_lengths, physical_lengths, *, tail=0, dummy_sequences=(), device="cpu"):
    """Build metadata and a producer mask; DSv4 attention includes logical dummy rows."""
    real_cu = torch.tensor([0, *accumulate(real_lengths)], dtype=torch.int32, device=device)
    physical_cu = torch.tensor([0, *accumulate(physical_lengths)], dtype=torch.int32, device=device)
    total = sum(physical_lengths) + tail
    valid = torch.zeros(total, dtype=torch.bool, device=device)
    producer_valid = valid.clone()
    offset = 0
    for sequence, (real, physical) in enumerate(zip(real_lengths, physical_lengths)):
        valid[offset : offset + real] = True
        if sequence not in dummy_sequences:
            producer_valid[offset : offset + real] = True
        offset += physical
    params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=real_cu,
        cu_seqlens_kv=real_cu,
        cu_seqlens_q_padded=physical_cu,
        cu_seqlens_kv_padded=physical_cu,
        max_seqlen_q=max(physical_lengths, default=0),
        max_seqlen_kv=max(physical_lengths, default=0),
        total_tokens=total,
        pad_between_seqs=True,
    )
    return params, (~producer_valid).unsqueeze(0), valid


def _reference_compressor(x, weights, ratio, physical_lengths, valid):
    """Project each complete physical group separately; padded groups have no derivative."""
    safe_x = x.masked_fill(~valid[:, None, None], 0)
    zero = safe_x.sum() * 0
    for weight in weights.values():
        zero = zero + weight.sum() * 0
    zero_row = zero.expand(1, weights["norm.weight"].shape[0]).to(x.dtype)
    outputs = []
    offset = 0
    for physical in physical_lengths:
        for position in range(0, physical - ratio + 1, ratio):
            start = offset + position
            if not valid[start : start + ratio].all():
                outputs.append(zero_row)
                continue
            tokens = x[start : start + ratio]
            if ratio == 1:
                latent = F.linear(tokens[0], weights["linear_wkv.weight"].to(x.dtype))
            else:
                projected = F.linear(tokens, weights["linear_wkv.weight"].to(x.dtype)).float()
                gate = F.linear(tokens, weights["linear_wgate.weight"].to(x.dtype)).float()
                # Each output channel has its own distribution over this sequence's group.
                latent = (projected * gate.softmax(dim=0)).sum(dim=0).to(x.dtype)
            xf = latent.float()
            normalized = xf / (xf.square().mean(dim=-1, keepdim=True) + 1e-20).sqrt()
            outputs.append((normalized * weights["norm.weight"].float()).to(x.dtype))
        offset += physical
    expected_capacity = min(
        x.shape[0] // ratio, len(physical_lengths) * (max(physical_lengths, default=0) // ratio)
    )
    outputs.extend([zero_row] * (expected_capacity - len(outputs)))
    if not outputs:
        return zero_row.unsqueeze(0)[:0]
    return torch.stack(outputs)


_PACK_CASES = [
    pytest.param([1, 3, 4, 5], [1, 3, 4, 5], 0, (), id="odd-sequences"),
    pytest.param([1, 3, 4], [4, 4, 6], 3, (), id="inter-sequence-and-tail-padding"),
    pytest.param([3, 4, 2], [4, 4, 2], 3, (2,), id="dummy-present-in-both-cu-arrays"),
    pytest.param([1, 1, 1], [3, 3, 3], 1, (), id="no-complete-r2-group"),
    pytest.param([0, 0], [4, 2], 2, (), id="entirely-padding"),
    pytest.param([1], [1], 0, (), id="zero-r2-capacity"),
    pytest.param([3], [4], 60, (), id="capacity-bounded-by-sequence-length"),
    pytest.param([1, 1], [1, 1], 30, (), id="capacity-zero-when-max-shorter-than-r2"),
    pytest.param([], [], 8, (), id="no-sequences-with-token-capacity"),
]


@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("real_lengths,physical_lengths,tail,dummies", _PACK_CASES)
@pytest.mark.parametrize("backend", ["native", "fused"])
def test_compressor_packed_outputs_and_gradients_match_independent_groups(
    ratio, dtype, real_lengths, physical_lengths, tail, dummies, backend, monkeypatch
):
    torch.manual_seed(320)
    device = "cuda" if backend == "fused" else "cpu"
    if backend == "fused":
        _require_csa2_compressor_kernels()
    params, _, valid = _packed(
        real_lengths, physical_lengths, tail=tail, dummy_sequences=dummies, device=device
    )
    module = _compressor(ratio, dtype).to(device)
    module.use_fused_compressor = backend == "fused"
    x = torch.randn(valid.numel(), 1, 6, device=device, dtype=dtype, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    weights = {
        name: parameter.detach().float().clone().requires_grad_()
        for name, parameter in module.named_parameters()
    }
    latent, layout = module(x, packed_seq_params=params)
    expected = _reference_compressor(reference_x, weights, ratio, physical_lengths, valid)
    tolerance = dict(atol=3e-6, rtol=3e-5)
    if dtype == torch.bfloat16:
        tolerance = dict(atol=2e-2, rtol=3e-2)
    expected_capacity = min(
        valid.numel() // ratio, len(physical_lengths) * (max(physical_lengths, default=0) // ratio)
    )
    assert latent.shape == (expected_capacity, 1, 8)
    assert latent.dtype == dtype
    torch.testing.assert_close(latent, expected, **tolerance)
    assert torch.count_nonzero(latent[~layout.valid_groups]) == 0

    # Multiple consumers exercise accumulation into the same compressor projections.
    probe_a = torch.randn_like(latent)
    probe_b = torch.randn_like(latent)
    actual_loss = (latent.float() * probe_a).sum() + (latent.float() * probe_b).sum() * 0.3
    expected_loss = (expected.float() * probe_a).sum() + (expected.float() * probe_b).sum() * 0.3
    actual_loss.backward()
    expected_loss.backward()
    torch.testing.assert_close(x.grad, reference_x.grad, **tolerance)
    assert torch.count_nonzero(x.grad[~valid]) == 0
    for name, parameter in module.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        torch.testing.assert_close(
            parameter.grad.float(),
            weights[name].grad,
            **tolerance,
            msg=lambda detail: f"{name}: {detail}",
        )


@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("backend", ["native", "fused"])
def test_nonfinite_padding_cannot_poison_compressor_forward_or_backward(ratio, dtype, backend):
    torch.manual_seed(321)
    device = "cuda" if backend == "fused" else "cpu"
    if backend == "fused":
        _require_csa2_compressor_kernels()
    params, _, valid = _packed([3, 1, 2], [4, 4, 2], tail=2, dummy_sequences=(2,), device=device)
    module = _compressor(ratio, dtype).to(device)
    module.use_fused_compressor = backend == "fused"
    x = torch.randn(valid.numel(), 1, 6, device=device, dtype=dtype)
    contaminated = x.clone()
    consumed = valid.clone()
    if ratio == 2:
        # Real trailing tokens also have no compressor representation. They must
        # not enter the Linear weight-gradient reduction with a zero upstream grad.
        consumed[2] = consumed[4] = False
    invalid_rows = (~consumed).nonzero().flatten()
    for index, row in enumerate(invalid_rows):
        contaminated[row] = [float("nan"), float("inf"), -float("inf")][index % 3]
    x.requires_grad_()
    contaminated.requires_grad_()
    expected, _ = module(x, packed_seq_params=params)
    actual, layout = module(contaminated, packed_seq_params=params)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert torch.isfinite(actual).all()
    assert torch.count_nonzero(actual[~layout.valid_groups]) == 0
    probe = torch.randn_like(actual)
    parameters = tuple(module.parameters())
    expected_grads = torch.autograd.grad((expected * probe).sum(), (x, *parameters))
    actual_grads = torch.autograd.grad((actual * probe).sum(), (contaminated, *parameters))
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        assert torch.isfinite(actual_grad).all()
        torch.testing.assert_close(actual_grad, expected_grad, atol=0, rtol=0)
    assert torch.count_nonzero(actual_grads[0][~consumed]) == 0


def _require_csa2_compressor_kernels():
    if not fused_compressor.fused_compressor_available():
        pytest.skip("CSA2 cuDNN compressor requires SM100+ and cuDNN frontend")


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fused_compressor_cuda_graph_replay_and_determinism(dtype):
    """Replay forward/backward with changed THD groups at fixed buffer capacity."""
    _require_csa2_compressor_kernels()
    torch.manual_seed(544)
    params, _, valid = _packed([3, 5, 0], [5, 6, 2], tail=3, device="cuda")
    layout = build_csa2_thd_layout(params, valid.numel()).for_compression(2)
    x = torch.randn(valid.numel(), 1, 64, device="cuda", dtype=dtype, requires_grad=True)
    token_cu = params.cu_seqlens_q_padded.clone()
    module = _compressor(2, dtype, hidden_size=64, v_head_dim=128).cuda()
    module.use_fused_compressor = True
    reference = _compressor(2, dtype, hidden_size=64, v_head_dim=128).cuda()
    reference.load_state_dict(module.state_dict())
    probe = torch.randn(layout.capacity, 1, 128, device="cuda", dtype=dtype)

    def run():
        output, _ = module._forward_thd(x, layout, token_cu)
        return output, torch.autograd.grad(output, (x, *module.parameters()), probe)

    # Compile both directions and initialize GEMM handles on a side stream before capture.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output, gradients = run()
    tolerance = dict(atol=4e-6, rtol=4e-5) if dtype == torch.float32 else dict(atol=2e-2, rtol=3e-2)
    for lengths, physical in [
        ([3, 5, 0], [5, 6, 2]),
        ([4, 2, 1], [4, 7, 2]),
        ([0, 0, 0], [5, 6, 2]),
    ]:
        next_params, _, next_valid = _packed(lengths, physical, tail=3, device="cuda")
        next_layout = build_csa2_thd_layout(next_params, next_valid.numel()).for_compression(2)
        with torch.no_grad():
            x.normal_()
            x[~next_valid] = float("nan")
            layout.source_indices.copy_(next_layout.source_indices)
            layout.valid_groups.copy_(next_layout.valid_groups)
            layout.cu_seqlens_padded.copy_(next_layout.cu_seqlens_padded)
            token_cu.copy_(next_params.cu_seqlens_q_padded)
        graph.replay()
        ref_x = x.detach().clone().requires_grad_()
        expected, _ = reference(ref_x, packed_seq_params=next_params)
        ref_grads = torch.autograd.grad(expected, (ref_x, *reference.parameters()), probe)
        torch.testing.assert_close(output, expected, **tolerance)
        for actual, ref in zip(gradients, ref_grads):
            torch.testing.assert_close(actual, ref, **tolerance)
        assert torch.count_nonzero(gradients[0][~next_valid]) == 0

    # The frontend still computes dAPE; strict deterministic mode must use native pooling.
    enabled, warn_only = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    try:
        torch.use_deterministic_algorithms(True)
        original_layout = build_csa2_thd_layout(params, valid.numel()).for_compression(2)
        with torch.no_grad():
            x.normal_()
            layout.source_indices.copy_(original_layout.source_indices)
            layout.valid_groups.copy_(original_layout.valid_groups)
        # Run the kernel region alone so the test does not depend on cuBLAS workspace settings.
        kv = torch.randn(32, 2, 128, device="cuda", dtype=dtype, requires_grad=True)
        gate = torch.randn_like(kv, requires_grad=True)
        probe = torch.randn(16, 2, 128, device="cuda", dtype=dtype)
        snapshots = []
        for _ in range(2):
            assert fused_compressor.maybe_pool_csa2_r2_fused(kv, gate, dtype) is None
            pooled = (
                (kv.float().reshape(16, 2, 2, 128) * gate.float().reshape(16, 2, 2, 128).softmax(1))
                .sum(1)
                .to(dtype)
            )
            grads = torch.autograd.grad(pooled, (kv, gate), probe)
            prepared = fused_compressor.maybe_prepare_csa2_r2_fused(x, layout)
            dx = torch.autograd.grad(prepared, x, torch.ones_like(prepared))[0]
            snapshots.append((pooled, *grads, dx))
        for first, second in zip(*snapshots):
            torch.testing.assert_close(first, second, atol=0, rtol=0)
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


def _rotary_module(config, use_yarn, cp_group):
    if use_yarn:
        return YarnRotaryEmbedding(
            config.qk_pos_emb_head_dim,
            rotary_base=160000,
            scaling_factor=16,
            original_max_position_embeddings=128,
            beta_fast=32,
            beta_slow=1,
            mscale=0,
            mscale_all_dim=0,
            cp_group=cp_group,
        )
    return RotaryEmbedding(config.qk_pos_emb_head_dim, 1.0, rotary_base=10000, cp_group=cp_group)


def _rope_reference(x, config, rotary, positions, valid):
    """Adjacent-pair rotation from the module's mathematical frequency table."""
    dim = config.qk_pos_emb_head_dim
    frequencies = rotary(64, packed_seq=True)
    if isinstance(frequencies, tuple):
        frequencies = frequencies[0]
    # Frequency generation is independent of packing. Apply explicit local positions here.
    angles = frequencies[positions, 0, 0, : dim // 2]
    shape = (x.shape[0], *([1] * (x.ndim - 2)), dim // 2)
    cos = angles.cos().to(x.dtype).reshape(shape)
    sin = angles.sin().to(x.dtype).reshape(shape)
    safe_x = x.masked_fill(~valid.reshape(-1, *([1] * (x.ndim - 1))), 0)
    even, odd = safe_x[..., -dim::2], safe_x[..., -dim + 1 :: 2]
    rotated = torch.stack((even * cos - odd * sin, odd * cos + even * sin), dim=-1).flatten(-2)
    return torch.cat((safe_x[..., :-dim], rotated), dim=-1)


class _CPUFrequencyTable(nn.Module):
    """Deterministic frequency inputs; the production RoPE operation remains unchanged."""

    def __init__(self, dim):
        super().__init__()
        self.register_buffer("frequencies", 160000 ** (-torch.arange(0, dim, 2).float() / dim))

    def forward(self, length, packed_seq=False):
        angles = torch.outer(
            torch.arange(length, device=self.frequencies.device).float(), self.frequencies
        )
        return torch.cat((angles, angles), dim=-1)[:, None, None, :]


# Complete THD attention and indexer supervision, compared to independent sequences.


def _packed_attention_cores(
    dtype=torch.float32,
    *,
    candidates=True,
    coefficient=0,
    sparse=False,
    per_token=False,
    **overrides,
):
    config_values = dict(
        params_dtype=dtype,
        csa2_candidate_source_layer=3 if candidates else None,
        csa2_candidate_topk_blocks=1 if candidates else 0,
        csa2_candidate_block_size=2 if candidates else 0,
        dsa_indexer_topk=2,
        dsa_indexer_loss_coeff=coefficient,
        dsa_indexer_use_sparse_loss=sparse,
        calculate_per_token_loss=per_token,
    )
    config_values.update(overrides)
    config = _make_config(**config_values)
    modules = CompressedSparseAttentionSubmodules(
        compressor=ModuleSpec(
            CSA2Compressor, submodules=CompressorSubmodules(_Linear, _Linear, _RMSNorm)
        ),
        indexer=ModuleSpec(
            CSA2Indexer, submodules=CSA2IndexerSubmodules(_Linear, _Linear, _RMSNorm, _Linear)
        ),
    )
    cores = nn.ModuleList(
        CompressedSparseAttention2(
            config,
            modules,
            layer_number=index + 1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
            pg_collection=_groups(),
            rotary_pos_emb=_CPUFrequencyTable(config.qk_pos_emb_head_dim),
        )
        for index in range(config.num_layers)
    )
    with torch.no_grad():
        for core in cores:
            core.attn_sink.uniform_(-1, 1)
    return cores


def _packed_core_inputs(cores, total):
    config = cores[0].config
    dtype = config.params_dtype
    return [
        (
            torch.randn(
                total,
                config.num_attention_heads,
                config.v_head_dim,
                dtype=dtype,
                requires_grad=True,
            ),
            torch.randn(total, 1, config.v_head_dim, dtype=dtype, requires_grad=True),
            torch.randn(total, 1, config.hidden_size, dtype=dtype, requires_grad=True),
            torch.randn(total, 1, config.q_lora_rank, dtype=dtype, requires_grad=True),
        )
        for _ in cores
    ]


def _run_packed_cores(cores, inputs, params):
    state = CSA2State()
    outputs = [
        core(
            # DSv4's QKV wrapper returns packed qr without the dummy batch axis.
            q,
            kv,
            kv,
            None,
            x=x,
            qr=qr.squeeze(1),
            packed_seq_params=params,
            csa2_state=state,
        )
        for core, (q, kv, x, qr) in zip(cores, inputs)
    ]
    return outputs, state


def _run_separate_cores(cores, inputs, real_lengths, physical_lengths):
    """Run independent SBHD forwards; no THD metadata/helpers enter this reference."""
    total = inputs[0][0].shape[0]
    contributions = [[] for _ in cores]
    start = 0
    for length, physical in zip(real_lengths, physical_lengths):
        if length:
            state = CSA2State()
            for index, (core, (q, kv, x, qr)) in enumerate(zip(cores, inputs)):
                selected = slice(start, start + length)
                output = core(
                    q[selected].unsqueeze(1),
                    kv[selected].unsqueeze(2),
                    kv[selected].unsqueeze(2),
                    None,
                    x=x[selected],
                    qr=qr[selected],
                    csa2_state=state,
                ).squeeze(1)
                contributions[index].append(F.pad(output, (0, 0, start, total - start - length)))
        start += physical
    return [sum(values) for values in contributions]


def _copy_core_inputs(inputs):
    return [tuple(tensor.detach().clone().requires_grad_() for tensor in row) for row in inputs]


def _assert_optional_gradients(actual, expected, *, dtype=torch.float32):
    if actual is None or expected is None:
        assert actual is expected
        return
    assert torch.isfinite(actual).all()
    tolerance = dict(atol=3e-6, rtol=3e-5)
    if dtype == torch.bfloat16:
        tolerance = dict(atol=2e-4, rtol=4e-2)
    torch.testing.assert_close(actual, expected, **tolerance)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("candidates", [False, True])
def test_thd_attention_stack_matches_separate_sequence_outputs_and_gradients(dtype, candidates):
    # This FP32 seed ties several Full1 indexer scores at zero. Packing must
    # retain the same smaller-position tie break despite the wider KV capacity.
    torch.manual_seed(326)
    cores = _packed_attention_cores(dtype, candidates=candidates)
    reference = _packed_attention_cores(dtype, candidates=candidates)
    reference.load_state_dict(cores.state_dict())
    real, physical, dummies = [1, 5, 7, 3], [3, 7, 8, 4], (3,)
    params, _, valid = _packed(real, physical, tail=3, dummy_sequences=dummies)
    inputs = _packed_core_inputs(cores, valid.numel())
    reference_inputs = _copy_core_inputs(inputs)
    outputs, state = _run_packed_cores(cores, inputs, params)
    expected = _run_separate_cores(reference, reference_inputs, real, physical)
    assert state.thd_layout is not None and state.compressed_layout is not None
    actual_loss = expected_loss = 0
    for output, ref_output in zip(outputs, expected):
        tolerance = (
            dict(atol=3e-6, rtol=3e-5) if dtype == torch.float32 else dict(atol=1e-2, rtol=2e-2)
        )
        torch.testing.assert_close(output, ref_output, **tolerance)
        assert torch.count_nonzero(output[~valid]) == 0
        probe = torch.randn_like(output)
        normalization = valid.sum() * output.shape[-1]
        actual_loss = actual_loss + (output.float() * probe).sum() / normalization
        expected_loss = expected_loss + (ref_output.float() * probe).sum() / normalization
    actual_loss.backward()
    expected_loss.backward()
    for row, reference_row in zip(inputs, reference_inputs):
        for tensor, reference_tensor in zip(row, reference_row):
            _assert_optional_gradients(tensor.grad, reference_tensor.grad, dtype=dtype)
            if tensor.grad is not None:
                assert torch.count_nonzero(tensor.grad[~valid]) == 0
    for parameter, reference_parameter in zip(cores.parameters(), reference.parameters()):
        _assert_optional_gradients(parameter.grad, reference_parameter.grad, dtype=dtype)


def test_thd_padding_and_other_sequences_cannot_contaminate_attention_or_gradients():
    torch.manual_seed(328)
    cores = _packed_attention_cores()
    real, physical, dummies = [1, 5, 7, 3], [3, 7, 8, 4], (3,)
    params, _, valid = _packed(real, physical, tail=3, dummy_sequences=dummies)
    inputs = _packed_core_inputs(cores, valid.numel())
    contaminated = _copy_core_inputs(inputs)
    with torch.no_grad():
        for row in contaminated:
            for tensor in row:
                tensor[~valid] = float("nan")
                tensor[10:17] = torch.randn_like(tensor[10:17]) * 10
    outputs, _ = _run_packed_cores(cores, inputs, params)
    changed, _ = _run_packed_cores(cores, contaminated, params)
    for output, changed_output in zip(outputs, changed):
        assert torch.isfinite(changed_output).all()
        torch.testing.assert_close(changed_output[3:8], output[3:8], atol=0, rtol=0)
        assert torch.count_nonzero(changed_output[~valid]) == 0
    actual_grads = torch.autograd.grad(
        sum(output[3:8].sum() for output in changed),
        [tensor for row in contaminated for tensor in row],
        allow_unused=True,
    )
    reference_grads = torch.autograd.grad(
        sum(output[3:8].sum() for output in outputs),
        [tensor for row in inputs for tensor in row],
        allow_unused=True,
    )
    for actual, expected in zip(actual_grads, reference_grads):
        _assert_optional_gradients(actual, expected)
        if actual is not None:
            assert torch.count_nonzero(actual[:3]) == 0
            assert torch.count_nonzero(actual[8:]) == 0


@pytest.mark.parametrize(
    "real,physical,tail,dummies",
    [
        pytest.param([], [], 0, (), id="zero-tokens"),
        pytest.param([0, 0], [3, 4], 3, (), id="only-padding"),
        pytest.param([1, 1], [1, 1], 2, (), id="no-complete-r2-groups"),
    ],
)
def test_thd_empty_groups_and_padding_have_finite_outputs_and_losses(
    monkeypatch, real, physical, tail, dummies
):
    logged = _record_losses(monkeypatch)
    cores = _packed_attention_cores(coefficient=0.3)
    params, _, valid = _packed(real, physical, tail=tail, dummy_sequences=dummies)
    inputs = _packed_core_inputs(cores, valid.numel())
    outputs, _ = _run_packed_cores(cores, inputs, params)
    for output in outputs:
        assert output.shape == (valid.numel(), 64)
        assert torch.isfinite(output).all()
        assert torch.count_nonzero(output[~valid]) == 0
    assert all(torch.isfinite(record["loss"]) and record["loss"] == 0 for record in logged)
    sum(output.float().sum() for output in outputs).backward()
    for row in inputs:
        for tensor in row:
            if tensor.grad is not None:
                assert torch.isfinite(tensor.grad).all()
                assert torch.count_nonzero(tensor.grad[~valid]) == 0
    for parameter in cores.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()


# Fused attention dispatch, shared gradients, and RoPE storage ownership.


@pytest.fixture
def reset_fused_indexer_loss_state(monkeypatch):
    monkeypatch.setattr(DSAIndexerLossAutoScaler, "main_loss_backward_scale", None)
    monkeypatch.setattr(DSAIndexerLossLoggingHelper, "tracker", {})


def _require_sparse_kernels():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("CSA2 sparse attention requires SM90+")
    # Match the existing V4 real-kernel tests: the CI FlashMLA build ships
    # SM100 sparse kernels only, although an SM90 build is supported by CSA.
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("CI FlashMLA sparse kernels require SM100+")
    pytest.importorskip("flash_mla")
    frontend = pytest.importorskip("cudnn")
    if not hasattr(frontend, "DSA"):
        pytest.skip("cuDNN Frontend DSA is unavailable")


def _sorted_valid_indices(scores, topk):
    """Independent selection oracle, with invalid slots after ascending valid keys."""
    ids = scores.argsort(dim=-1, descending=True, stable=True)[..., :topk]
    ids = ids.masked_fill(~scores.gather(-1, ids).isfinite(), torch.iinfo(torch.int32).max)
    ids = ids.sort(dim=-1).values
    return ids.masked_fill(ids == torch.iinfo(torch.int32).max, -1).int()


def _reference_fused_indexer_scores(
    indexer, q, k, weights, *, candidates=None, thd_layout=None, compressed_layout=None
):
    """Dense oracle for rounded-input selection, independent of fused dispatch/layout lowering."""
    if indexer.precision == "mxfp8":
        q_data, q_scale = _reference_quantize(q)
        k_data, k_scale = _reference_quantize(k)
        q = (q_data.float().unflatten(-1, (-1, 32)) * q_scale.unsqueeze(-1)).flatten(-2)
        k = (k_data.float().unflatten(-1, (-1, 32)) * k_scale.unsqueeze(-1)).flatten(-2)
    scale = indexer.head_dim**-0.5 * indexer.n_heads**-0.5
    w = (weights.float() * scale).to(weights.dtype).float()
    if thd_layout is None:
        dots = torch.einsum("sbhd,cbd->bshc", q.float(), k.float()).relu()
        scores = (dots * w.permute(1, 0, 2).unsqueeze(-1)).sum(-2)
        visible = torch.arange(1, q.shape[0] + 1, device=q.device) // indexer.compress_ratio
        valid = torch.arange(k.shape[0], device=k.device)[None, :] < visible[:, None]
    else:
        dots = torch.einsum("thd,cd->thc", q.float(), k.float()).relu()
        scores = (dots * w.unsqueeze(-1)).sum(-2)
        valid = (
            thd_layout.valid_tokens[:, None]
            & compressed_layout.valid_groups[None, :]
            & (thd_layout.sequence_ids[:, None] == compressed_layout.sequence_ids[None, :])
            & (
                compressed_layout.position_ids[None, :] + indexer.compress_ratio - 1
                <= thd_layout.position_ids[:, None]
            )
        )
    if candidates is not None:
        valid = valid & candidates.to_mask(
            k.shape[0], thd_layout=thd_layout, compressed_layout=compressed_layout
        )
    return scores.masked_fill(~valid, -torch.inf)


def _reference_fused_indexer(indexer, q, k, weights, **kwargs):
    scores = _reference_fused_indexer_scores(indexer, q, k, weights, **kwargs)
    return _sorted_valid_indices(scores, indexer.topk)


def _emulate_v4_indexer_topk_core(q, k, weights, topk, ratio, **kwargs):
    """CPU adapter for V4 core's prepared-layout and already-scaled weight contract."""
    assert q.is_contiguous() and k.is_contiguous() and weights.is_contiguous()
    assert q.dtype == k.dtype == weights.dtype == torch.bfloat16
    assert kwargs["deterministic"] is True
    assert kwargs["use_compact"] is True
    if kwargs["precision"] == "mxfp8":
        assert q.shape[-2:] == (64, 128) and weights.shape[-1] == 64
    w = weights.float()
    if "cu_seqlens_q" in kwargs:
        cu_q, cu_k = kwargs["cu_seqlens_q"], kwargs["cu_seqlens_kv"]
        assert cu_q.dtype == cu_k.dtype == torch.int32
        assert cu_q[-1] == q.shape[0] and cu_k[-1] == k.shape[0]
        assert cu_q.diff().max() <= kwargs["max_seqlen_q"]
        assert cu_k.diff().max() <= kwargs["max_seqlen_kv"]
        assert kwargs["max_seqlen_q"] <= kwargs["max_seqlen_kv"] * ratio
        scores = torch.full((q.shape[0], max(topk, kwargs["max_seqlen_kv"])), -torch.inf)
        for start, end, k_start, k_end in zip(cu_q[:-1], cu_q[1:], cu_k[:-1], cu_k[1:]):
            dots = torch.einsum(
                "thd,cd->thc", q[start:end].float(), k[k_start:k_end].float()
            ).relu()
            local_scores = (dots * w[start:end, :, None]).sum(-2)
            visible = torch.arange(1, end - start + 1) // ratio
            valid = torch.arange(k_end - k_start)[None, :] < visible[:, None]
            scores[start:end, : k_end - k_start] = local_scores.masked_fill(~valid, -torch.inf)
    else:
        assert q.shape[1] <= k.shape[1] * ratio
        dots = torch.einsum("bshd,bcd->bshc", q.float(), k.float()).relu()
        scores = (dots * w.unsqueeze(-1)).sum(-2)
        visible = torch.arange(1, q.shape[1] + 1) // ratio
        valid = torch.arange(k.shape[1])[None, :] < visible[:, None]
        scores = scores.masked_fill(~valid, -torch.inf)
        scores = F.pad(scores, (0, max(0, topk - k.shape[1])), value=-torch.inf)
    indices = _sorted_valid_indices(scores, topk).flip(-1).contiguous()
    return indices, (indices >= 0).sum(-1).int(), None, None


def _kernel_indexer(ratio, precision, heads, device):
    config = _config(torch.bfloat16)
    config.dsa_indexer_n_heads, config.dsa_indexer_head_dim = heads, 128
    config.dsa_indexer_topk = 7
    indexer = CSA2Indexer(
        config, CSA2IndexerSubmodules(_Linear, _Linear, _RMSNorm, _Linear), ratio, _groups()
    ).to(device)
    # Exercise the adapter independently of config validation and main attention.
    indexer.use_fused_kernels, indexer.precision = True, precision
    return indexer


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("ratio", [1, 2])
def test_bf16_indexer_rotates_q_and_k_in_fp32(monkeypatch, fused, packed, ratio):
    """RoPE must not reintroduce BF16 trig/product rounding before selection."""
    _require_candidate_kernels()
    torch.manual_seed(817)
    indexer = _kernel_indexer(ratio, "bf16", 32, "cuda")
    indexer.config.apply_rope_fusion = fused
    rotary = _rotary_module(indexer.config, False, indexer.cp_group)
    layout = compressed = None
    length = 12
    valid = torch.ones(length, device="cuda", dtype=torch.bool)
    positions = torch.arange(length, device="cuda")
    key_positions = torch.arange(length // ratio, device="cuda") * ratio
    key_valid = torch.ones(length // ratio, device="cuda", dtype=torch.bool)
    if packed:
        params, _, valid = _packed([3, 7], [4, 8], device="cuda")
        layout = build_csa2_thd_layout(params, length)
        compressed = layout.for_compression(ratio)
        positions, key_positions, key_valid = (
            layout.position_ids,
            compressed.position_ids,
            compressed.valid_groups,
        )
    x = torch.randn(length, 1, 6, device="cuda", dtype=torch.bfloat16)
    qr = torch.randn(length, 1, 4, device="cuda", dtype=torch.bfloat16)
    latent = torch.randn(key_positions.numel(), 1, 8, device="cuda", dtype=torch.bfloat16)
    q0 = indexer.linear_wq_b(qr.masked_fill(~valid[:, None, None], 0))[0].reshape(
        length, 1, 32, 128
    )
    k0 = indexer.k_norm(indexer.linear_wk(latent.masked_fill(~key_valid[:, None, None], 0))[0])
    expected_q = _rope_reference(q0.float(), indexer.config, rotary, positions, valid).bfloat16()
    expected_k = _rope_reference(
        k0.float(), indexer.config, rotary, key_positions, key_valid
    ).bfloat16()
    fused_dtypes = []
    apply_fused = csa._apply_fused_rope

    def record_fused(x, cos, sin, *args, **kwargs):
        fused_dtypes.append((x.dtype, cos.dtype, sin.dtype))
        return apply_fused(x, cos, sin, *args, **kwargs)

    monkeypatch.setattr(csa, "_apply_fused_rope", record_fused)
    actual_q, actual_k, _ = indexer._project_inputs(
        x, qr, latent, rotary, thd_layout=layout, compressed_layout=compressed
    )
    if packed:
        expected_q, expected_k = expected_q.squeeze(1), expected_k.squeeze(1)
    torch.testing.assert_close(actual_q, expected_q, atol=0, rtol=0)
    torch.testing.assert_close(actual_k, expected_k, atol=0, rtol=0)
    assert fused_dtypes == ([(torch.bfloat16, torch.float32, torch.float32)] * 2 if fused else [])


@pytest.mark.parametrize("backend", ["adapter", "real"])
@pytest.mark.parametrize("precision", ["bf16", "mxfp8"])
@pytest.mark.parametrize("heads", [32, 64])
@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize(
    "layout", ["sbhd", "thd", "empty", "short", "padding", "unused-capacity", "long-tail"]
)
def test_fused_indexer_topk_layout_and_precision(
    monkeypatch, backend, precision, heads, ratio, layout
):
    """r1/r2 odd tails, per-sequence padding, int64 metadata, H32 MXFP8 padding, and empty K."""
    real = backend == "real"
    if real:
        _require_sparse_kernels()
    device = "cuda" if real else "cpu"
    torch.manual_seed(412)
    indexer = _kernel_indexer(ratio, precision, heads, device)
    calls = []
    if not real:

        def adapter(q, k, w, **kwargs):
            calls.append(kwargs)
            if precision == "bf16" or heads == 64:
                assert q.data_ptr() == prepared.q.data_ptr(), "BSHD Q must be a view"
            if layout != "sbhd" or ratio == 1:
                assert k.data_ptr() == prepared.k.data_ptr(), "Only the odd r2 tail needs padded K"
            if precision == "mxfp8" and heads == 32:
                assert torch.count_nonzero(q[..., 32:, :]) == 0
                assert torch.count_nonzero(w[..., 32:]) == 0
            return _emulate_v4_indexer_topk_core(q, k, w, **kwargs)

        monkeypatch.setattr(
            "megatron.core.transformer.experimental_attention_variant.csa2._indexer_topk_core",
            adapter,
        )
    token_layout = compressed = None
    if layout == "sbhd":
        q = torch.randn(129, 2, heads, 128, device=device, dtype=torch.bfloat16)
        k = torch.randn(129 // ratio, 2, 128, device=device, dtype=torch.bfloat16)
        w = torch.randn(129, 2, heads, device=device, dtype=torch.bfloat16)
    else:
        lengths, padded, tail = {
            "thd": ([1, 0, 57, 129], [3, 0, 62, 133], 3),
            "empty": ([0, 0], [0, 0], 0),
            "short": ([1, 1], [1, 1], 0),
            "padding": ([0, 0], [3, 5], 3),
            "unused-capacity": ([1, 1], [1, 1], 5),
            "long-tail": ([3, 0, 5], [4, 0, 6], 31),
        }[layout]
        params, _, valid = _packed(lengths, padded, tail=tail, device=device)
        if layout == "unused-capacity":
            params.max_seqlen_q = params.max_seqlen_kv = 3
        for name in (
            "cu_seqlens_q",
            "cu_seqlens_kv",
            "cu_seqlens_q_padded",
            "cu_seqlens_kv_padded",
        ):
            setattr(params, name, getattr(params, name).long())
        token_layout = build_csa2_thd_layout(params, valid.numel())
        compressed = token_layout.for_compression(ratio)
        q = torch.randn(valid.numel(), heads, 128, device=device, dtype=torch.bfloat16)
        k = torch.randn(compressed.capacity, 128, device=device, dtype=torch.bfloat16)
        w = torch.randn(valid.numel(), heads, device=device, dtype=torch.bfloat16)
        q[~valid], w[~valid], k[~compressed.valid_groups] = 0, 0, 0
    q_before, k_before, w_before = q.clone(), k.clone(), w.clone()
    prepared = prepare_csa2_indexer_inputs(
        q, k, w, ratio, thd_layout=token_layout, compressed_layout=compressed
    )
    prepared_before = [t.clone() for t in (prepared.q, prepared.k, prepared.weights)]
    actual = indexer._fused_topk(prepared)
    if not real:
        # The CPU adapter checks precision dispatch, not the CUDA quantizer's arithmetic.
        indexer.precision = "bf16"
    expected = _reference_fused_indexer(
        indexer, q, k, w, thd_layout=token_layout, compressed_layout=compressed
    )
    torch.testing.assert_close(actual, expected)
    for original, before in ((q, q_before), (k, k_before), (w, w_before)):
        torch.testing.assert_close(original, before, rtol=0, atol=0)
    for tensor, before in zip((prepared.q, prepared.k, prepared.weights), prepared_before):
        torch.testing.assert_close(tensor, before, rtol=0, atol=0)
    assert actual.dtype == torch.int32 and actual.is_contiguous()
    if not real:
        assert len(calls) == (0 if q.shape[0] == 0 or k.shape[0] == 0 else 1)
    if real and precision == "mxfp8" and layout == "sbhd":
        indexer.precision = "bf16"
        unquantized = _reference_fused_indexer(indexer, q, k, w)
        assert not torch.equal(actual, unquantized), "MXFP8 must actually change quantized ranking"


@pytest.mark.parametrize("supervised", [False, True])
def test_fused_indexer_without_loss_avoids_dense_scores(monkeypatch, supervised):
    indexer = _kernel_indexer(2, "bf16", 32, "cpu")
    monkeypatch.setattr(
        "megatron.core.transformer.experimental_attention_variant.csa2._indexer_topk_core",
        _emulate_v4_indexer_topk_core,
    )

    def reject_dense(*args, **kwargs):
        pytest.fail(
            "Ordinary fused selection without supervision must not materialize dense scores"
        )

    monkeypatch.setattr(indexer, "_score_projected", reject_dense)
    k = indexer.project_keys(torch.randn(4, 2, 8, dtype=torch.bfloat16), _CPUFrequencyTable(4))
    k_flat = k.transpose(0, 1).reshape(-1, 128).contiguous()
    selected_inputs = []
    select = indexer._fused_topk

    def record_selection(prepared):
        assert not torch.is_grad_enabled()
        assert prepared.k is k_flat
        assert prepared.q.requires_grad == prepared.weights.requires_grad == supervised
        selected_inputs.append(prepared)
        return select(prepared)

    monkeypatch.setattr(indexer, "_fused_topk", record_selection)
    with torch.set_grad_enabled(supervised):
        indices, loss_inputs, candidate_scores = indexer.forward_with_scores(
            torch.randn(9, 2, 6, dtype=torch.bfloat16),
            torch.randn(9, 2, 4, dtype=torch.bfloat16),
            None,
            _CPUFrequencyTable(4),
            indexer_k=k,
            indexer_k_flat=k_flat,
            return_loss_inputs=supervised,
        )
    assert indices.shape == (2, 9, 4)
    assert candidate_scores is None
    assert len(selected_inputs) == 1
    assert loss_inputs is (selected_inputs[0] if supervised else None)


# Compact candidate generation. Keep the oracle independent of fused scoring,
# block reduction, radix selection, and packed address conversion.


def _reference_block_ids(scores, visible, topk_blocks, block_size):
    width = scores.shape[-1]
    blocks = (width + block_size - 1) // block_size
    output = torch.full(
        (*scores.shape[:-1], min(topk_blocks, blocks)), -1, device=scores.device, dtype=torch.int32
    )
    counts = torch.zeros(scores.shape[:-1], device=scores.device, dtype=torch.int32)
    if width == 0:
        return CSA2CandidateBlocks(output, counts, block_size)
    cols = torch.arange(width, device=scores.device)
    masked = scores.masked_fill(cols >= visible.unsqueeze(-1), -torch.inf)
    maxima = torch.stack(
        [masked[..., start : start + block_size].amax(-1) for start in range(0, width, block_size)],
        -1,
    )
    newest = (visible - 1) // block_size
    for block in range(blocks):
        maxima[..., block] = torch.where(newest == block, torch.inf, maxima[..., block])
    ordered = maxima.argsort(dim=-1, descending=True, stable=True)[..., :topk_blocks]
    valid = maxima.gather(-1, ordered) > -torch.inf
    ordered = ordered.masked_fill(~valid, width).sort(-1).values
    return CSA2CandidateBlocks(
        ordered.masked_fill(ordered == width, -1).int(), valid.sum(-1).int(), block_size
    )


def _reference_candidate_blocks(
    indexer, q, k, weights, *, selection_scores=None, thd_layout=None, compressed_layout=None
):
    scores = _reference_fused_indexer_scores(
        indexer, q, k, weights, thd_layout=thd_layout, compressed_layout=compressed_layout
    )
    block_size = indexer.config.csa2_candidate_block_size
    topk_blocks = indexer.config.csa2_candidate_topk_blocks
    if thd_layout is None:
        visible = ((torch.arange(q.shape[0], device=q.device) + 1) // indexer.compress_ratio).clamp(
            max=k.shape[0]
        )
        return _reference_block_ids(scores, visible, topk_blocks, block_size)
    width = min(topk_blocks, (compressed_layout.max_seqlen + block_size - 1) // block_size)
    ids = torch.full((q.shape[0], width), -1, device=q.device, dtype=torch.int32)
    lengths = torch.zeros(q.shape[0], device=q.device, dtype=torch.int32)
    for q_start, q_length, k_start, k_length in zip(
        thd_layout.cu_seqlens_padded[:-1].tolist(),
        thd_layout.cu_seqlens.diff().tolist(),
        compressed_layout.cu_seqlens_padded[:-1].tolist(),
        compressed_layout.cu_seqlens.diff().tolist(),
    ):
        local = scores[q_start : q_start + q_length, k_start : k_start + k_length]
        visible = (torch.arange(q_length, device=q.device) + 1) // indexer.compress_ratio
        selected = _reference_block_ids(local, visible, topk_blocks, block_size)
        ids[q_start : q_start + q_length, : selected.indices.shape[-1]] = selected.indices
        lengths[q_start : q_start + q_length] = selected.lengths
    return CSA2CandidateBlocks(ids, lengths, block_size)


def _require_candidate_kernels(precision="bf16"):
    major = 10 if precision == "mxfp8" else 9
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < major:
        pytest.skip(f"Fused {precision} candidates require SM{major}0+")
    pytest.importorskip("triton")
    pytest.importorskip("cudnn")
    if precision == "mxfp8":
        pytest.importorskip("transformer_engine")
    csa2_candidates.fused_sparse_attention._ensure_dsa_namespace()


def _reject_cuda_host_reads(monkeypatch):
    for method in ("tolist", "item", "cpu", "numpy", "__bool__", "__int__", "__index__"):
        original = getattr(torch.Tensor, method)

        def reject_host_read(tensor, *args, original=original, **kwargs):
            if tensor.is_cuda:
                pytest.fail("Candidate generation must not read CUDA tensor values on the host")
            return original(tensor, *args, **kwargs)

        monkeypatch.setattr(torch.Tensor, method, reject_host_read)


@pytest.mark.parametrize("backend", ["native", "real"])
@pytest.mark.parametrize(
    "block_size,topk,width", [(8, 3, 59), (3, 1, 37), (2, 8, 13), (8, 2048, 32771)]
)
def test_candidate_block_compaction_ties_newest_and_padding(backend, block_size, topk, width):
    if backend == "real":
        _require_candidate_kernels()
    device = "cuda" if backend == "real" else "cpu"
    torch.manual_seed(414)
    # Exact ties across the radix threshold, including negative weighted scores.
    scores = torch.randint(-3, 4, (5, width), device=device).float()
    visible = torch.tensor([0, 1, width // 2, width - 1, width], device=device, dtype=torch.int32)
    scores[4] = 0
    # Invisible high values must never enter the reduction. The latest block
    # has the worst ordinary score but must still be retained.
    scores[2, width // 2 - 1] = -100
    expected = _reference_block_ids(scores, visible, topk, block_size)
    if backend == "real":
        actual = csa2_candidates._select_blocks(scores, visible, topk, block_size)
    else:
        masked = scores.masked_fill(torch.arange(width)[None, :] >= visible[:, None], -torch.inf)
        actual = candidate_blocks_from_scores(masked, visible, topk, block_size)
    torch.testing.assert_close(actual.indices, expected.indices)
    torch.testing.assert_close(actual.lengths, expected.lengths)
    assert (actual.indices[0] == -1).all()
    assert (actual.indices[1:, :] == ((visible[1:] - 1) // block_size)[:, None]).any(-1).all()
    if backend == "real":
        again = csa2_candidates._select_blocks(
            F.pad(scores, (0, 19), value=1e10), visible, topk, block_size
        )
        torch.testing.assert_close(again.lengths, actual.lengths)
        torch.testing.assert_close(again.indices[:, : actual.indices.shape[-1]], actual.indices)
        assert (again.indices[:, actual.indices.shape[-1] :] == -1).all()


@pytest.mark.parametrize("shape", [(2, 0, 0), (2, 3, 0), (0, 7), (3, 0)])
def test_candidate_blocks_empty_native_storage(shape):
    scores = torch.empty(shape)
    selected = candidate_blocks_from_scores(
        scores, torch.zeros(shape[:-1], dtype=torch.int32), 3, 8
    )
    assert selected.indices.shape == (*shape[:-1], (shape[-1] + 7) // 8)
    assert torch.count_nonzero(selected.lengths) == 0
    assert selected.to_mask(shape[-1]).shape == shape


def test_fused_candidate_radix_rows_with_unaligned_block_counts():
    """Unaligned radix rows previously duplicated +inf and dropped a real winner."""
    _require_candidate_kernels()
    scores = torch.tensor(
        [[0.81289214, 0.430245, 0.707756, 1.1742834, -1.0] + [100.0] * 12] * 17, device="cuda"
    )
    visible = torch.full((17,), 5, dtype=torch.int32, device="cuda")
    actual = csa2_candidates._select_blocks(scores, visible, 3, 1)
    torch.testing.assert_close(
        actual.indices, torch.tensor([[0, 3, 4]] * 17, dtype=torch.int32, device="cuda")
    )
    torch.testing.assert_close(actual.lengths, torch.full_like(visible, 3))


@pytest.mark.parametrize("precision", ["bf16", "mxfp8"])
@pytest.mark.parametrize("heads", [32, 64])
@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("layout", ["sbhd", "thd", "long-tail"])
def test_fused_candidate_generation_matches_dense_oracle(
    monkeypatch, precision, heads, ratio, layout
):
    _require_candidate_kernels(precision)
    torch.manual_seed(415)
    indexer = _kernel_indexer(ratio, precision, heads, "cuda")
    indexer.config.csa2_candidate_topk_blocks = 3
    indexer.config.csa2_candidate_block_size = 8
    token_layout = compressed = None
    if layout == "sbhd":
        q = torch.randn(131, 2, heads, 128, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(131 // ratio, 2, 128, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(131, 2, heads, device="cuda", dtype=torch.bfloat16)
    else:
        params, _, valid = _packed(
            [1, 0, 43, 131],
            [3, 0, 48, 134],
            tail=511 if layout == "long-tail" else 7,
            device="cuda",
        )
        token_layout = build_csa2_thd_layout(params, valid.numel())
        compressed = token_layout.for_compression(ratio)
        q = torch.randn(valid.numel(), heads, 128, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(compressed.capacity, 128, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(valid.numel(), heads, device="cuda", dtype=torch.bfloat16)
        # Padding contains adversarial values, not a convenient all-zero input.
        q[~valid], w[~valid], k[~compressed.valid_groups] = 100, 100, 100
    snapshots = [x.clone() for x in (q, k, w)]
    expected = _reference_candidate_blocks(
        indexer, q, k, w, thd_layout=token_layout, compressed_layout=compressed
    )
    # Force chunks across sequences and compression/block boundaries. Bound the
    # score allocation by bytes as well as rows; neither depends on total Q.
    monkeypatch.setattr(csa2_candidates, "_QUERY_CHUNK_SIZE", 31)
    max_k = k.shape[0] if token_layout is None else compressed.max_seqlen
    monkeypatch.setattr(csa2_candidates, "_SCORE_CHUNK_MAX_BYTES", 17 * ((max_k + 3) // 4 * 4) * 4)
    scorer = csa2_candidates.fused_sparse_attention._DSA.indexer_forward_wrapper
    chunks = []
    chunk_metadata = []

    def bounded_scorer(chunk_q, all_k, chunk_w, **kwargs):
        assert chunk_q.shape[0] <= 17
        if precision == "bf16":
            assert all_k.data_ptr() == prepared.k.data_ptr()
            offset = sum(chunks) * heads * 128 * prepared.q.element_size()
            assert chunk_q.data_ptr() == prepared.q.data_ptr() + offset
        chunk_metadata.append(
            (
                kwargs["cu_seqlens_q"],
                kwargs["cu_seqlens_k"],
                kwargs["max_seqlen_q"],
                kwargs["max_seqlen_k"],
                all_k.shape[0],
            )
        )
        chunks.append(chunk_q.shape[0])
        return scorer(chunk_q, all_k, chunk_w, **kwargs)

    monkeypatch.setattr(
        csa2_candidates.fused_sparse_attention._DSA, "indexer_forward_wrapper", bounded_scorer
    )
    expected_topk = _reference_fused_indexer(
        indexer, q, k, w, thd_layout=token_layout, compressed_layout=compressed
    )
    prepared = prepare_csa2_indexer_inputs(
        q, k, w, ratio, thd_layout=token_layout, compressed_layout=compressed
    )
    prepared_before = [t.clone() for t in (prepared.q, prepared.k, prepared.weights)]
    with monkeypatch.context() as patch:
        _reject_cuda_host_reads(patch)
        actual_topk, actual = indexer._fused_topk_and_candidates(prepared)
    torch.testing.assert_close(actual_topk, expected_topk)
    torch.testing.assert_close(actual.indices, expected.indices)
    torch.testing.assert_close(actual.lengths, expected.lengths)
    assert len(chunks) > 1 and sum(chunks) == q.shape[0] * (2 if layout == "sbhd" else 1)
    for rows, (cu_q, cu_k, max_q, max_k, key_rows) in zip(chunks, chunk_metadata):
        assert cu_q[-1] == rows and cu_k[-1] == key_rows
        assert cu_q.diff().max() <= max_q and cu_k.diff().max() <= max_k
    for x, before in zip((q, k, w), snapshots):
        torch.testing.assert_close(x, before, atol=0, rtol=0)
    for x, before in zip((prepared.q, prepared.k, prepared.weights), prepared_before):
        torch.testing.assert_close(x, before, atol=0, rtol=0)
    assert actual.indices.is_contiguous() and not actual.indices.requires_grad
    mask = actual.to_mask(k.shape[0], thd_layout=token_layout, compressed_layout=compressed)
    if token_layout is not None:
        assert not mask[~token_layout.valid_tokens].any()
        assert not mask[:, ~compressed.valid_groups].any()
        assert not mask[
            token_layout.sequence_ids[:, None] != compressed.sequence_ids[None, :]
        ].any()
    if precision == "mxfp8" and layout == "sbhd":
        indexer.precision = "bf16"
        unquantized = _reference_candidate_blocks(indexer, q, k, w)
        assert not torch.equal(
            actual.indices, unquantized.indices
        ), "MXFP8 must change candidate ranking"


@pytest.mark.parametrize("precision", ["bf16", "mxfp8"])
@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("layout", ["empty", "padding", "short", "unused-capacity"])
def test_fused_candidate_generation_empty_and_unused_capacity(
    monkeypatch, precision, ratio, layout
):
    _require_candidate_kernels(precision)
    real, physical, tail = {
        "empty": ([0, 0], [0, 0], 0),
        "padding": ([0, 0], [3, 5], 3),
        "short": ([1, 1], [1, 1], 0),
        "unused-capacity": ([1, 1], [1, 1], 7),
    }[layout]
    params, _, valid = _packed(real, physical, tail=tail, device="cuda")
    if layout == "unused-capacity":
        params.max_seqlen_q = params.max_seqlen_kv = 4
    token_layout = build_csa2_thd_layout(params, valid.numel())
    compressed = token_layout.for_compression(ratio)
    indexer = _kernel_indexer(ratio, precision, 32, "cuda")
    indexer.config.csa2_candidate_topk_blocks = 2
    indexer.config.csa2_candidate_block_size = 2
    q = torch.randn(valid.numel(), 32, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(compressed.capacity, 128, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(valid.numel(), 32, device="cuda", dtype=torch.bfloat16)
    expected = _reference_candidate_blocks(
        indexer, q, k, w, thd_layout=token_layout, compressed_layout=compressed
    )
    with monkeypatch.context() as patch:
        _reject_cuda_host_reads(patch)
        actual = indexer._candidate_blocks(
            q, k, w, selection_scores=None, thd_layout=token_layout, compressed_layout=compressed
        )
    torch.testing.assert_close(actual.indices, expected.indices)
    torch.testing.assert_close(actual.lengths, expected.lengths)
    expected_topk = _reference_fused_indexer(
        indexer, q, k, w, candidates=actual, thd_layout=token_layout, compressed_layout=compressed
    )
    topk = indexer._fused_topk(
        prepare_csa2_indexer_inputs(q, k, w, ratio, actual, token_layout, compressed)
    )
    torch.testing.assert_close(topk, expected_topk)
    q.requires_grad_(), k.requires_grad_(), w.requires_grad_()
    loss = fused_csa2_indexer_loss(
        prepare_csa2_indexer_inputs(q, k, w, ratio, actual, token_layout, compressed),
        torch.zeros((q.shape[0], 64, 512), dtype=q.dtype, device=q.device),
        torch.zeros((q.shape[0], 512), dtype=q.dtype, device=q.device),
        torch.zeros((k.shape[0], 1, 512), dtype=q.dtype, device=q.device),
        torch.zeros(64, device=q.device),
        torch.full((q.shape[0], 1), -1, dtype=torch.int32, device=q.device),
        topk,
        512**-0.5,
        0.3,
        False,
    )
    assert loss == 0 and loss.requires_grad
    loss.backward()
    for t in (q, k, w):
        assert t.grad is not None and torch.count_nonzero(t.grad) == 0


@pytest.mark.parametrize("precision", ["bf16", "mxfp8"])
def test_fused_candidate_source_without_loss_avoids_dense_scores(monkeypatch, precision):
    _require_candidate_kernels(precision)
    indexer = _kernel_indexer(2, precision, 32, "cuda")
    indexer.config.csa2_candidate_topk_blocks = 2
    indexer.config.csa2_candidate_block_size = 8

    def reject_native(*args, **kwargs):
        pytest.fail("Fused candidate-source selection must not materialize native scores")

    monkeypatch.setattr(indexer, "_score_projected", reject_native)
    monkeypatch.setattr(indexer, "_fused_topk", reject_native)
    monkeypatch.setattr(indexer, "_candidate_blocks", reject_native)
    indices, scores, candidates = indexer.forward_with_scores(
        torch.randn(65, 2, 6, dtype=torch.bfloat16, device="cuda"),
        torch.randn(65, 2, 4, dtype=torch.bfloat16, device="cuda"),
        torch.randn(32, 2, 8, dtype=torch.bfloat16, device="cuda"),
        _CPUFrequencyTable(4).cuda(),
        return_candidates=True,
    )
    assert scores is None and indices.shape == (2, 65, 7)
    assert candidates.indices.shape == (2, 65, 2)


@pytest.mark.parametrize("mode", ["dense", "candidates", "sparse"])
@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("per_token", [False, True])
@pytest.mark.parametrize("layout", ["sbhd", "thd"])
def test_fused_indexer_loss_matches_independent_teacher(
    monkeypatch, mode, ratio, per_token, layout
):
    _require_candidate_kernels()
    torch.manual_seed(419)
    indexer = _kernel_indexer(ratio, "bf16", 32, "cuda")
    indexer.config.csa2_candidate_topk_blocks, indexer.config.csa2_candidate_block_size = 2, 3
    token_layout = compressed = None
    if layout == "sbhd":
        q = torch.randn(37, 2, 32, 128, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(37 // ratio, 2, 128, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(37, 2, 32, device="cuda", dtype=torch.bfloat16)
        teacher_q = torch.randn(37, 2, 64, 512, device="cuda", dtype=torch.bfloat16)
        teacher_k = torch.randn(37 // ratio, 2, 512, device="cuda", dtype=torch.bfloat16)
        local = torch.randn(37, 2, 512, device="cuda", dtype=torch.bfloat16)
        window = (
            torch.arange(37, device="cuda")[:, None] - torch.arange(4, device="cuda")[None, :]
        ).clamp_min(-1)
        window = window[None].expand(2, -1, -1).contiguous().int()
    else:
        params, _, valid = _packed([1, 0, 11, 37], [3, 0, 14, 40], tail=5, device="cuda")
        token_layout = build_csa2_thd_layout(params, valid.numel())
        compressed = token_layout.for_compression(ratio)
        q = torch.randn(valid.numel(), 32, 128, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(compressed.capacity, 128, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(valid.numel(), 32, device="cuda", dtype=torch.bfloat16)
        teacher_q = torch.randn(valid.numel(), 64, 512, device="cuda", dtype=torch.bfloat16)
        teacher_k = torch.randn(compressed.capacity, 1, 512, device="cuda", dtype=torch.bfloat16)
        local = torch.randn(valid.numel(), 512, device="cuda", dtype=torch.bfloat16)
        window = (
            torch.arange(valid.numel(), device="cuda")[:, None]
            - torch.arange(4, device="cuda")[None, :]
        )
        keep = (
            valid[:, None]
            & (window >= 0)
            & (torch.arange(4, device="cuda")[None, :] <= token_layout.position_ids[:, None])
        )
        window = window.masked_fill(~keep, -1).int()
        q[~valid], w[~valid], k[~compressed.valid_groups] = 100, 100, 100
    candidates = (
        None
        if mode == "dense"
        else _reference_candidate_blocks(
            indexer, q, k, w, thd_layout=token_layout, compressed_layout=compressed
        )
    )
    q, k, w = [t.requires_grad_() for t in (q, k, w)]
    ref_q, ref_k, ref_w = [t.detach().clone().requires_grad_() for t in (q, k, w)]
    kwargs = dict(candidates=candidates, thd_layout=token_layout, compressed_layout=compressed)
    scores = indexer._score_projected(ref_q, ref_k, ref_w, **kwargs)
    topk = _sorted_valid_indices(scores.detach(), 7)
    sink = torch.linspace(-7, 9, 64, device="cuda", requires_grad=True)
    if mode == "candidates":
        # Global teacher mass far below 1e-10 must still renormalize to one.
        with torch.no_grad():
            sink.add_(60)
    teacher_q.requires_grad_(), teacher_k.requires_grad_(), local.requires_grad_()
    core = SimpleNamespace(
        attn_sink=sink,
        softmax_scale=512**-0.5,
        config=SimpleNamespace(
            dsa_indexer_use_sparse_loss=mode == "sparse",
            dsa_indexer_loss_coeff=0.3,
            calculate_per_token_loss=per_token,
        ),
    )
    if layout == "sbhd":
        expected = _teacher_oracle(core, teacher_q, local, teacher_k, window, topk, scores)
    else:
        expected = _teacher_oracle(
            core,
            teacher_q.unsqueeze(1),
            local.unsqueeze(1),
            teacher_k,
            window.unsqueeze(0),
            topk.unsqueeze(0),
            scores.unsqueeze(0),
        )
        if not per_token:
            expected = expected * q.shape[0] / valid.sum()
    monkeypatch.setattr(csa2_indexer, "_QUERY_CHUNK_SIZE", 17)
    monkeypatch.setattr(
        CSA2CandidateBlocks, "to_mask", lambda *a, **kw: pytest.fail("dense candidate mask")
    )
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(
        lambda t: (saved.append(t.shape) or t), lambda t: t
    ):
        actual = fused_csa2_indexer_loss(
            prepare_csa2_indexer_inputs(q, k, w, ratio, candidates, token_layout, compressed),
            teacher_q,
            local,
            teacher_k,
            sink,
            window,
            topk,
            core.softmax_scale,
            0.3,
            mode == "sparse",
            per_token,
        )
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=2e-5)
    assert scores.shape not in saved
    assert len(saved) == 3, "Only the unit-loss dQ/dK/dW should survive forward"
    monkeypatch.setattr(
        csa2_indexer, "_score_chunk", lambda *a, **kw: pytest.fail("backward recomputed scores")
    )
    monkeypatch.setattr(
        csa2_indexer, "_target_chunk", lambda *a, **kw: pytest.fail("backward recomputed targets")
    )
    (actual * 0.7).backward()
    (expected * 0.7).backward()
    for value, ref in zip((q, k, w), (ref_q, ref_k, ref_w)):
        error = (value.grad.float() - ref.grad.float()).norm()
        assert error / ref.grad.float().norm().clamp_min(1e-10) < 0.01
        assert torch.isfinite(value.grad).all()
    assert all(t.grad is None for t in (teacher_q, teacher_k, local, sink))
    if layout == "thd":
        assert not q.grad[~valid].any() and not w.grad[~valid].any()
        assert not k.grad[~compressed.valid_groups].any()


@pytest.mark.parametrize("mode", ["dense", "candidates", "sparse"])
def test_fused_indexer_loss_spans_key_tiles_without_dense_saved_scores(mode):
    """Exercise H64 and reductions beyond 64/1024 slots, including backward accumulation."""
    _require_candidate_kernels()
    torch.manual_seed(420)
    length = 1099
    indexer = _kernel_indexer(1, "bf16", 64, "cuda")
    indexer.config.csa2_candidate_topk_blocks, indexer.config.csa2_candidate_block_size = 137, 8
    q = torch.randn(length, 1, 64, 128, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    k = torch.randn(length, 1, 128, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    w = torch.randn(length, 1, 64, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    candidates = (
        None
        if mode == "dense"
        else _reference_candidate_blocks(indexer, q.detach(), k.detach(), w.detach())
    )
    rq, rk, rw = [t.detach().clone().requires_grad_() for t in (q, k, w)]
    scores = indexer._score_projected(rq, rk, rw, candidates=candidates)
    topk = _sorted_valid_indices(scores.detach(), 1027)
    teacher_q = torch.randn(length, 1, 16, 64, dtype=q.dtype, device=q.device)
    teacher_k = torch.randn(length, 1, 64, dtype=q.dtype, device=q.device)
    local = torch.randn(length, 1, 64, dtype=q.dtype, device=q.device)
    window = (
        (torch.arange(length, device=q.device)[:, None] - torch.arange(4, device=q.device)[None, :])
        .clamp_min(-1)[None]
        .int()
    )
    core = _math_core(sparse=mode == "sparse").cuda()
    core.attn_sink = nn.Parameter(torch.linspace(-7, 9, 16, device=q.device))
    core.softmax_scale = 64**-0.5
    expected = core._compute_indexer_loss(teacher_q, local, teacher_k, window, topk, scores)
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(
        lambda t: (saved.append(t.shape) or t), lambda t: t
    ):
        actual = fused_csa2_indexer_loss(
            prepare_csa2_indexer_inputs(q, k, w, 1, candidates),
            teacher_q,
            local,
            teacher_k,
            core.attn_sink,
            window,
            topk,
            core.softmax_scale,
            0.3,
            mode == "sparse",
        )
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - before
    # Eager backward retains linear-size unit gradients instead of projections
    # and teacher activations. Account for those outputs plus bounded workspace.
    gradient_bytes = sum(t.numel() * t.element_size() for t in (q, k, w))
    assert peak < gradient_bytes + 8 * 1024 * 1024
    assert torch.Size((length, length)) not in saved and scores.shape not in saved
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=2e-5)
    actual.backward()
    expected.backward()
    for value, ref in zip((q, k, w), (rq, rk, rw)):
        error = (value.grad.float() - ref.grad.float()).norm()
        assert error / ref.grad.float().norm().clamp_min(1e-10) < 0.01
    print(f"indexer_loss_{mode}: forward_incremental_peak_bytes={peak}")
