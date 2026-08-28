# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for the experimental MLA latent P2P context-parallel implementation.

The numerical oracle is an independent standard-PyTorch port of the naive MLA path in
``deepseek-ai/DeepSeek-V3`` at commit
``9b4e9788e4a3a731f7567338ed15d3ec549ce03b``, file ``inference/model.py``.
The no-RoPE branch is pinned to ``moonshotai/Kimi-K3`` at commit
``c5d1dd4c428bd1ce8b88c5044f3b6ccde9e3b721``, file ``modeling_kimi_linear.py``.
It does not import MCore projection, RoPE, attention, CP, FA4, cuDNN, or TE code.
"""

from __future__ import annotations

import gc
import json
import math
import os
import traceback
import weakref
from contextlib import contextmanager
from dataclasses import dataclass, replace
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.context_parallel_layout.utils import finalize_packed_seq_params
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import multi_latent_attention as base_mla
from megatron.core.transformer.enums import AttnBackend, AttnMaskType
from megatron.core.transformer.experimental_attention_variant import mla_with_latent_cp
from megatron.core.transformer.experimental_attention_variant.mla_with_latent_cp import (
    backend as latent_cp_backend,
)
from megatron.core.transformer.experimental_attention_variant.mla_with_latent_cp import (
    cudnn_backend as latent_cp_cudnn_backend,
)
from megatron.core.transformer.experimental_attention_variant.mla_with_latent_cp import (
    mla_with_latent_cp as latent_cp_module,
)
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import MLATransformerConfig
from megatron.core.utils import init_method_normal, scaled_init_method_normal

latent_cp = mla_with_latent_cp

DEEPSEEK_V3_REFERENCE_REPO = "https://github.com/deepseek-ai/DeepSeek-V3"
DEEPSEEK_V3_REFERENCE_COMMIT = "9b4e9788e4a3a731f7567338ed15d3ec549ce03b"
DEEPSEEK_V3_REFERENCE_PATH = "inference/model.py"
KIMI_K3_REFERENCE_REPO = "https://huggingface.co/moonshotai/Kimi-K3"
KIMI_K3_REFERENCE_COMMIT = "c5d1dd4c428bd1ce8b88c5044f3b6ccde9e3b721"
KIMI_K3_REFERENCE_PATH = "modeling_kimi_linear.py"
EXPECTED_CUDNN_FRONTEND_SOURCE_REV = "0a14b7181d129d30e7bad34b8c3ed0a0c995e23d"
EXPECTED_QUALIFIED_BACKEND_CONFIGS: tuple[latent_cp.QualifiedBackendTuple, ...] = (
    (AttnBackend.fused, "1.22.1", "9.21.0", (9, 0)),
    (AttnBackend.fused, "1.26.0", "9.25.0", (10, 0)),
    (AttnBackend.flash, "4.0.0b11", "flash-attn-4==4.0.0b11", (10, 0)),
)
EXPECTED_QUALIFICATION_EPS: dict[latent_cp.QualifiedBackendTuple, float] = {
    (AttnBackend.fused, "1.22.1", "9.21.0", (9, 0)): 4.561878810305231e-05,
    (AttnBackend.fused, "1.26.0", "9.25.0", (10, 0)): 4.423665134356547e-05,
    (
        AttnBackend.flash,
        "4.0.0b11",
        "flash-attn-4==4.0.0b11",
        (10, 0),
    ): 4.3095951884009054e-05,
}

_PRODUCTION_HIDDEN = 7168
_PRODUCTION_HEADS = 96
_PRODUCTION_Q_LORA = 1536
_PRODUCTION_KV_LORA = 512
_QK_CONTENT = 128
_ROPE_DIM = 64
_VALUE_DIM = 128
_PRODUCTION_PACKED_LENGTHS = (128, 8, 8, 8, 8)
_PARITY_EPS_CEILING = 1e-3
_SEED = 7421


def _cosine_sim(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return F.cosine_similarity(
        actual.flatten().double().unsqueeze(0), expected.flatten().double().unsqueeze(0)
    ).item()


def _tensor_sim(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual = actual.double()
    expected = expected.double()
    denominator = (actual * actual + expected * expected).sum()
    if denominator == 0:
        return 1.0
    return (2.0 * (actual * expected).sum() / denominator).item()


def _measure_similarity(
    actual: torch.Tensor, expected: torch.Tensor, label: str
) -> dict[str, float]:
    assert actual.shape == expected.shape, (
        f"{label}: shape {tuple(actual.shape)} != {tuple(expected.shape)}"
    )
    assert torch.isfinite(actual).all(), f"{label}: actual is non-finite"
    assert torch.isfinite(expected).all(), f"{label}: expected is non-finite"
    if (
        torch.count_nonzero(actual).item() == 0
        or torch.count_nonzero(expected).item() == 0
    ):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, msg=label)
        return {"cosine": 1.0, "tensor_similarity": 1.0, "observed_error": 0.0}
    cosine = _cosine_sim(actual, expected)
    tensor = _tensor_sim(actual, expected)
    return {
        "cosine": cosine,
        "tensor_similarity": tensor,
        "observed_error": max(0.0, 1.0 - cosine, 1.0 - tensor),
    }


def _assert_similarity_metrics(
    metrics: dict[str, float], label: str, eps: float = _PARITY_EPS_CEILING
) -> None:
    cosine = metrics["cosine"]
    tensor = metrics["tensor_similarity"]
    assert cosine > 1 - eps, (
        f"{label}: cosine={cosine:.10f}, tensor={tensor:.10f}, eps={eps}"
    )
    assert tensor > 1 - eps, (
        f"{label}: tensor={tensor:.10f}, cosine={cosine:.10f}, eps={eps}"
    )


def _assert_similarity(
    actual: torch.Tensor,
    expected: torch.Tensor,
    label: str,
    eps: float = _PARITY_EPS_CEILING,
) -> dict[str, float]:
    metrics = _measure_similarity(actual, expected, label)
    _assert_similarity_metrics(metrics, label, eps)
    return metrics


def _cumulative(
    lengths: tuple[int, ...], device: str | torch.device = "cpu"
) -> torch.Tensor:
    values = [0]
    for length in lengths:
        values.append(values[-1] + length)
    return torch.tensor(values, dtype=torch.int32, device=device)


def _zigzag_global_indices(
    lengths: tuple[int, ...],
    cp_size: int,
    cp_rank: int,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    if cp_size == 1:
        return torch.arange(sum(lengths), dtype=torch.long, device=device)
    indices: list[int] = []
    offset = 0
    for length in lengths:
        assert length % (2 * cp_size) == 0
        chunk = length // (2 * cp_size)
        front = offset + cp_rank * chunk
        back = offset + (2 * cp_size - 1 - cp_rank) * chunk
        indices.extend(range(front, front + chunk))
        indices.extend(range(back, back + chunk))
        offset += length
    return torch.tensor(indices, dtype=torch.long, device=device)


def _sequence_parallel_slice(
    tensor: torch.Tensor, tp_group: dist.ProcessGroup
) -> torch.Tensor:
    """Return this TP rank's contiguous first-dimension shard."""

    tp_size = dist.get_world_size(tp_group)
    assert tensor.size(0) % tp_size == 0
    return torch.chunk(tensor, tp_size, dim=0)[dist.get_rank(tp_group)].contiguous()


def _zigzag_positions(
    lengths: tuple[int, ...], cp_size: int, cp_rank: int
) -> torch.Tensor:
    if cp_size == 1:
        return torch.cat([torch.arange(length, dtype=torch.long) for length in lengths])
    positions: list[int] = []
    for length in lengths:
        chunk = length // (2 * cp_size)
        positions.extend(range(cp_rank * chunk, (cp_rank + 1) * chunk))
        back = (2 * cp_size - 1 - cp_rank) * chunk
        positions.extend(range(back, back + chunk))
    return torch.tensor(positions, dtype=torch.long)


def _phase_global_rows(
    lengths: tuple[int, ...], cp_size: int, cp_rank: int, phase: latent_cp.PhaseSpec
) -> tuple[list[list[int]], list[list[int]]]:
    q_mapping = _zigzag_global_indices(lengths, cp_size, cp_rank)
    kv_mapping = _zigzag_global_indices(lengths, cp_size, phase.owner)
    q_rows = q_mapping.index_select(0, phase.q_indices.cpu())
    kv_rows = kv_mapping.index_select(0, phase.kv_indices.cpu())
    q_cu = phase.cu_seqlens_q.cpu().tolist()
    kv_cu = phase.cu_seqlens_kv.cpu().tolist()
    return (
        [q_rows[q_cu[i] : q_cu[i + 1]].tolist() for i in range(len(lengths))],
        [kv_rows[kv_cu[i] : kv_cu[i + 1]].tolist() for i in range(len(lengths))],
    )


def _make_config(
    *,
    tp_size: int = 1,
    cp_size: int = 1,
    backend: AttnBackend = AttnBackend.fused,
    rope_type: str = "rope",
    production_shape: bool = False,
    attention_output_gate: bool = False,
    gate_granularity: str = "elementwise",
    no_rope: bool = False,
    dynamic_cp: bool = False,
) -> MLATransformerConfig:
    hidden_size = _PRODUCTION_HIDDEN if production_shape else 32
    heads = _PRODUCTION_HEADS if production_shape else 4
    q_lora = _PRODUCTION_Q_LORA if production_shape else 16
    kv_lora = _PRODUCTION_KV_LORA if production_shape else 16
    yarn = rope_type == "yarn"
    return MLATransformerConfig(
        num_layers=1,
        hidden_size=hidden_size,
        num_attention_heads=heads,
        num_query_groups=heads,
        kv_channels=128,
        multi_latent_attention=True,
        mla_latent_cp=True,
        attention_output_gate=attention_output_gate,
        gated_attention_proj_granularity=gate_granularity,
        q_lora_rank=q_lora,
        kv_lora_rank=kv_lora,
        qk_head_dim=_QK_CONTENT,
        qk_pos_emb_head_dim=_ROPE_DIM,
        v_head_dim=_VALUE_DIM,
        add_bias_linear=False,
        bf16=True,
        fp16=False,
        params_dtype=torch.bfloat16,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        layernorm_epsilon=1e-6,
        normalization="RMSNorm",
        qk_layernorm=True,
        layernorm_zero_centered_gamma=False,
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=1,
        context_parallel_size=cp_size,
        dynamic_context_parallel=dynamic_cp,
        min_dynamic_context_parallel_size=1,
        expert_model_parallel_size=1,
        sequence_parallel=tp_size > 1,
        cp_comm_type="p2p",
        cp_partition_mode="zigzag",
        apply_rope_fusion=False,
        rope_type=rope_type,
        no_rope_freq=[1] if no_rope else None,
        rotary_percent=1.0,
        rotary_scaling_factor=40.0 if yarn else 1.0,
        mscale=1.0,
        mscale_all_dim=1.0,
        rotary_base=10000,
        original_max_position_embeddings=64 if yarn else 4096,
        beta_fast=32,
        beta_slow=1,
        rotary_interleaved=False,
        recompute_granularity=None,
        recompute_modules=[],
        fine_grained_activation_offloading=False,
        gradient_accumulation_fusion=False,
        fp8=None,
        fp4=None,
        cache_mla_latents=False,
        cuda_graph_impl="none",
        enable_cuda_graph=False,
        external_cuda_graph=False,
        use_cpu_initialization=False,
        perform_initialization=True,
        batch_invariant_mode=False,
        symmetric_ar_type=None,
        disable_parameter_transpose_cache=False,
        init_model_with_meta_device=False,
        delay_wgrad_compute=False,
        tp_comm_overlap=False,
        attention_backend=backend,
        init_method=init_method_normal(0.02),
        output_layer_init_method=scaled_init_method_normal(0.02, 1, multiplier=2.0),
    )


def _base_mla_spec():
    return get_gpt_layer_local_spec(
        num_experts=None,
        moe_grouped_gemm=False,
        qk_layernorm=True,
        multi_latent_attention=True,
        normalization="RMSNorm",
    ).submodules.self_attention


def _build_layer(
    config: MLATransformerConfig,
    pg: ProcessGroupCollection,
    backend_adapter: latent_cp.DirectAttentionAdapter | None = None,
):
    spec = latent_cp.make_mla_with_latent_cp_spec(_base_mla_spec())
    if backend_adapter is None:
        return build_module(
            spec, config=config, layer_number=1, cp_comm_type="p2p", pg_collection=pg
        )
    runtime = (
        config.attention_backend,
        "test-only",
        "test-only",
        torch.cuda.get_device_capability(),
    )
    with mock.patch.object(
        latent_cp_module,
        "_qualified_backend_adapter",
        return_value=(backend_adapter, runtime),
    ):
        return build_module(
            spec, config=config, layer_number=1, cp_comm_type="p2p", pg_collection=pg
        )


def _build_legacy_cp_layer(config: MLATransformerConfig, pg: ProcessGroupCollection):
    from megatron.core.extensions.transformer_engine import TEDotProductAttention

    base = _base_mla_spec()
    spec = replace(
        base,
        params=dict(base.params),
        metainfo=dict(base.metainfo),
        submodules=replace(base.submodules, core_attention=TEDotProductAttention),
    )
    return build_module(
        spec, config=config, layer_number=1, cp_comm_type="p2p", pg_collection=pg
    )


def _make_packed(
    lengths: tuple[int, ...],
    *,
    device: str | torch.device,
    cp_group=None,
    local_cp_size: int | None = None,
    total_tokens: int | None = None,
) -> PackedSeqParams:
    cu = _cumulative(lengths, device)
    return PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu.clone(),
        max_seqlen_q=max(lengths),
        max_seqlen_kv=max(lengths),
        cp_group=cp_group,
        local_cp_size=local_cp_size,
        total_tokens=total_tokens,
        cp_partition_mode="zigzag",
        pad_between_seqs=False,
    )


@contextmanager
def _model_parallel(tp_size: int, cp_size: int, *, dynamic_cp: bool = False):
    from tests.unit_tests.test_utilities import Utils

    required = tp_size * cp_size
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if Utils.world_size < required or Utils.world_size % required:
        pytest.skip(f"test requires a world size divisible by TP{tp_size}xCP{cp_size}")
    dynamic_kwargs = (
        {"dynamic_context_parallel": True, "min_dynamic_context_parallel_size": 1}
        if dynamic_cp
        else {}
    )
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=1,
        context_parallel_size=cp_size,
        **dynamic_kwargs,
    )
    model_parallel_cuda_manual_seed(_SEED)
    try:
        yield ProcessGroupCollection.use_mpu_process_groups(
            required_pgs=["tp", "cp", "tp_cp"]
        )
    finally:
        Utils.destroy_model_parallel()


@contextmanager
def _rank_common_assertions():
    """Fail rank-local assertions collectively so peers never enter later collectives alone."""

    local_failure = None
    try:
        yield
    except AssertionError:
        local_failure = traceback.format_exc()
    failures = [None] * dist.get_world_size()
    dist.all_gather_object(failures, local_failure)
    formatted = [
        f"rank {rank}:\n{failure}"
        for rank, failure in enumerate(failures)
        if failure is not None
    ]
    if formatted:
        pytest.fail(
            "rank-local assertion failure(s):\n" + "\n".join(formatted), pytrace=False
        )


# Pinned from DeepSeek-V3 inference/model.py. Standard PyTorch only.
def _find_correction_dim(
    rotations: float, dim: int, base: float, original_max_positions: int
) -> float:
    return (
        dim
        * math.log(original_max_positions / (rotations * 2 * math.pi))
        / (2 * math.log(base))
    )


def _find_correction_range(
    low_rot: float, high_rot: float, dim: int, base: float, original_max_positions: int
) -> tuple[int, int]:
    low = math.floor(_find_correction_dim(low_rot, dim, base, original_max_positions))
    high = math.ceil(_find_correction_dim(high_rot, dim, base, original_max_positions))
    return max(low, 0), min(high, dim - 1)


def _linear_ramp(low: int, high: int, dim: int, device: torch.device) -> torch.Tensor:
    if low == high:
        high += 0.001
    ramp = (torch.arange(dim, dtype=torch.float32, device=device) - low) / (high - low)
    return ramp.clamp(0, 1)


def _official_freqs_cis(
    *,
    dim: int,
    max_seq_len: int,
    base: float,
    factor: float,
    original_max_positions: int,
    beta_fast: float,
    beta_slow: float,
    device: torch.device,
) -> torch.Tensor:
    frequencies = 1.0 / (
        base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
    )
    if factor > 1:
        low, high = _find_correction_range(
            beta_fast, beta_slow, dim, base, original_max_positions
        )
        smooth = 1 - _linear_ramp(low, high, dim // 2, device)
        frequencies = frequencies / factor * (1 - smooth) + frequencies * smooth
    positions = torch.arange(max_seq_len, dtype=torch.float32, device=device)
    phase = torch.outer(positions, frequencies)
    return torch.polar(torch.ones_like(phase), phase)


def _official_apply_rotary(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype
    complex_x = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    frequencies = freqs_cis.view(freqs_cis.size(0), 1, freqs_cis.size(1))
    return torch.view_as_real(complex_x * frequencies).flatten(-2).to(dtype)


class NaiveMLA(nn.Module):
    """Independent official-naive MLA formula for a packed batch."""

    def __init__(self, config: MLATransformerConfig, device: torch.device):
        super().__init__()
        factory = {"device": device, "dtype": torch.bfloat16}
        q_head_dim = config.qk_head_dim + config.qk_pos_emb_head_dim
        kv_head_dim = config.qk_head_dim + config.v_head_dim
        self.wq_a = nn.Linear(
            config.hidden_size, config.q_lora_rank, bias=False, **factory
        )
        self.q_norm = nn.RMSNorm(
            config.q_lora_rank, eps=config.layernorm_epsilon, **factory
        )
        self.wq_b = nn.Linear(
            config.q_lora_rank,
            config.num_attention_heads * q_head_dim,
            bias=False,
            **factory,
        )
        self.wkv_a = nn.Linear(
            config.hidden_size,
            config.kv_lora_rank + config.qk_pos_emb_head_dim,
            bias=False,
            **factory,
        )
        self.kv_norm = nn.RMSNorm(
            config.kv_lora_rank, eps=config.layernorm_epsilon, **factory
        )
        self.wkv_b = nn.Linear(
            config.kv_lora_rank,
            config.num_attention_heads * kv_head_dim,
            bias=False,
            **factory,
        )
        self.wo = nn.Linear(
            config.num_attention_heads * config.v_head_dim,
            config.hidden_size,
            bias=False,
            **factory,
        )
        self.gate = None
        if config.attention_output_gate:
            gate_size = (
                config.num_attention_heads * config.v_head_dim
                if config.gated_attention_proj_granularity == "elementwise"
                else config.num_attention_heads
            )
            self.gate = nn.Linear(config.hidden_size, gate_size, bias=False, **factory)
        self.config = config
        self.q_head_dim = q_head_dim
        self.use_rope = not bool(config.no_rope_freq and config.no_rope_freq[0])
        scale = 1.0
        if self.use_rope and config.rotary_scaling_factor > 1:
            # Pinned DeepSeek-V3 inference/model.py uses its sole ``mscale`` field.
            scale = 0.1 * config.mscale * math.log(config.rotary_scaling_factor) + 1.0
        self.softmax_scale = scale * scale / math.sqrt(q_head_dim)

    def forward(
        self, hidden_states: torch.Tensor, packed_lengths: tuple[int, ...]
    ) -> torch.Tensor:
        x = hidden_states.squeeze(1)
        q_latent = self.q_norm(self.wq_a(x))
        query = self.wq_b(q_latent).view(
            x.size(0), self.config.num_attention_heads, self.q_head_dim
        )
        q_content, q_rope = torch.split(
            query, [self.config.qk_head_dim, self.config.qk_pos_emb_head_dim], dim=-1
        )

        kv_down = self.wkv_a(x)
        latent, k_rope = torch.split(
            kv_down, [self.config.kv_lora_rank, self.config.qk_pos_emb_head_dim], dim=-1
        )
        latent = self.kv_norm(latent)
        expanded = self.wkv_b(latent).view(
            x.size(0),
            self.config.num_attention_heads,
            self.config.qk_head_dim + self.config.v_head_dim,
        )
        k_content, value = torch.split(
            expanded, [self.config.qk_head_dim, self.config.v_head_dim], dim=-1
        )

        frequencies = None
        if self.use_rope:
            frequencies = _official_freqs_cis(
                dim=self.config.qk_pos_emb_head_dim,
                max_seq_len=max(packed_lengths),
                base=self.config.rotary_base,
                factor=self.config.rotary_scaling_factor,
                original_max_positions=self.config.original_max_position_embeddings,
                beta_fast=self.config.beta_fast,
                beta_slow=self.config.beta_slow,
                device=x.device,
            )
        outputs: list[torch.Tensor] = []
        offset = 0
        for length in packed_lengths:
            token_slice = slice(offset, offset + length)
            if self.use_rope:
                q_pe = _official_apply_rotary(q_rope[token_slice], frequencies[:length])
                k_pe = _official_apply_rotary(
                    k_rope[token_slice].unsqueeze(1), frequencies[:length]
                )
            else:
                # Pinned Kimi-K3 MLA keeps the positional-width Q/K branches but
                # concatenates them without constructing or applying RoPE.
                q_pe = q_rope[token_slice]
                k_pe = k_rope[token_slice].unsqueeze(1)
            q_seq = torch.cat((q_content[token_slice], q_pe), dim=-1)
            k_seq = torch.cat(
                (
                    k_content[token_slice],
                    k_pe.expand(-1, self.config.num_attention_heads, -1),
                ),
                dim=-1,
            )
            v_seq = value[token_slice]
            scores = torch.einsum("qhd,khd->hqk", q_seq.float(), k_seq.float())
            scores = scores * self.softmax_scale
            causal = torch.triu(
                torch.ones(length, length, dtype=torch.bool, device=x.device),
                diagonal=1,
            )
            scores = scores.masked_fill(causal.unsqueeze(0), -torch.inf)
            probabilities = torch.softmax(scores, dim=-1).to(v_seq.dtype)
            output = torch.einsum("hqk,khd->qhd", probabilities, v_seq)
            outputs.append(output.to(torch.bfloat16))
            offset += length
        merged = torch.cat(outputs, dim=0).reshape(x.size(0), -1)
        if self.gate is not None:
            gate = torch.sigmoid(self.gate(x).float()).to(merged.dtype)
            if self.config.gated_attention_proj_granularity == "headwise":
                merged = merged.view(x.size(0), self.config.num_attention_heads, -1)
                merged = (merged * gate.unsqueeze(-1)).reshape(x.size(0), -1)
            else:
                merged = merged * gate
        return self.wo(merged).unsqueeze(1)


def _parameter_map(config: MLATransformerConfig) -> dict[str, tuple[str, int | None]]:
    mapping = {
        "wq_a.weight": ("linear_q_down_proj.weight", 0),
        "q_norm.weight": ("q_layernorm.weight", None),
        "wq_b.weight": ("linear_q_up_proj.weight", 0),
        "wkv_a.weight": ("linear_kv_down_proj.weight", 0),
        "kv_norm.weight": ("kv_layernorm.weight", None),
        "wkv_b.weight": ("linear_kv_up_proj.weight", 0),
        "wo.weight": ("linear_proj.weight", 1),
    }
    if config.attention_output_gate:
        mapping["gate.weight"] = ("linear_gate.weight", 0)
    return mapping


def _copy_reference_parameters(
    reference: NaiveMLA,
    real_layer: latent_cp.MLAWithLatentCP,
    pg: ProcessGroupCollection,
) -> None:
    reference_params = dict(reference.named_parameters())
    real_params = dict(real_layer.named_parameters())
    parameter_map = _parameter_map(reference.config)
    assert set(reference_params) == set(parameter_map)
    assert set(real_params) == {real_name for real_name, _ in parameter_map.values()}
    tp_size = dist.get_world_size(pg.tp)
    tp_rank = dist.get_rank(pg.tp)
    for reference_name, (real_name, shard_dim) in parameter_map.items():
        source = reference_params[reference_name]
        if shard_dim is None:
            expected = source
        else:
            assert source.size(shard_dim) % tp_size == 0
            expected = torch.chunk(source, tp_size, dim=shard_dim)[tp_rank]
        destination = real_params[real_name]
        assert destination.shape == expected.shape
        destination.data.copy_(expected.data)

    for reference_name, (real_name, shard_dim) in parameter_map.items():
        real_parameter = real_params[real_name]
        gathered = [torch.empty_like(real_parameter) for _ in range(tp_size)]
        dist.all_gather(gathered, real_parameter, group=pg.tp)
        if shard_dim is None:
            assert all(torch.equal(gathered[0], other) for other in gathered[1:])
            reconstructed = gathered[0]
        else:
            reconstructed = torch.cat(gathered, dim=shard_dim)
        assert torch.equal(reconstructed, reference_params[reference_name])


def _reconstruct_real_parameter_gradients(
    real_layer: latent_cp.MLAWithLatentCP,
    pg: ProcessGroupCollection,
    *,
    cp_group: dist.ProcessGroup | None = None,
) -> dict[str, torch.Tensor]:
    real_params = dict(real_layer.named_parameters())
    reconstructed: dict[str, torch.Tensor] = {}
    parameter_map = _parameter_map(real_layer.config)
    tp_size = dist.get_world_size(pg.tp)
    cp_group = pg.cp if cp_group is None else cp_group
    for reference_name, (real_name, shard_dim) in parameter_map.items():
        parameter = real_params[real_name]
        assert parameter.grad is not None, f"missing real gradient {real_name}"
        grad = parameter.grad.detach().float().clone()
        if dist.get_world_size(cp_group) > 1:
            dist.all_reduce(grad, group=cp_group)
        if shard_dim is None:
            if tp_size > 1:
                dist.all_reduce(grad, group=pg.tp)
            reconstructed[reference_name] = grad
        else:
            shards = [torch.empty_like(grad) for _ in range(tp_size)]
            dist.all_gather(shards, grad, group=pg.tp)
            reconstructed[reference_name] = torch.cat(shards, dim=shard_dim)
    return reconstructed


@dataclass(frozen=True)
class _SavedTensorRecord:
    shape: tuple[int, ...]
    numel: int
    dtype: torch.dtype
    tensor_class: str
    state_class: str = "unclassified"


class _PackedSavedTensor:
    __slots__ = ("tensor", "__weakref__")

    def __init__(self, tensor: torch.Tensor) -> None:
        self.tensor = tensor


class _SavedTensorRecorder:
    """Enumerate tensors still physically retained by the outer autograd graph."""

    def __init__(self) -> None:
        self._packed: list[
            tuple[_SavedTensorRecord, weakref.ReferenceType[_PackedSavedTensor]]
        ] = []

    def pack(self, tensor: torch.Tensor) -> _PackedSavedTensor:
        record = _SavedTensorRecord(
            shape=tuple(tensor.shape),
            numel=tensor.numel(),
            dtype=tensor.dtype,
            tensor_class=f"{type(tensor).__module__}.{type(tensor).__qualname__}",
        )
        holder = _PackedSavedTensor(tensor)
        self._packed.append((record, weakref.ref(holder)))
        return holder

    @staticmethod
    def unpack(holder: _PackedSavedTensor) -> torch.Tensor:
        return holder.tensor

    @property
    def records(self) -> list[_SavedTensorRecord]:
        return [record for record, holder in self._packed if holder() is not None]


class _SaveExpandedKV(torch.autograd.Function):
    """Identity sentinel that exposes exact THD K/V saves when not checkpointed."""

    @staticmethod
    def forward(ctx, key, value):
        ctx.save_for_backward(key, value)
        return key, value

    @staticmethod
    def backward(ctx, grad_key, grad_value):
        del ctx
        return grad_key, grad_value


def _classify_saved_attention_state(
    records: list[_SavedTensorRecord],
    *,
    expected_query_shapes: list[tuple[int, ...]],
    expected_latent_shapes: list[tuple[int, ...]],
    heads: int,
) -> list[_SavedTensorRecord]:
    remaining_queries = list(expected_query_shapes)
    remaining_latents = list(expected_latent_shapes)
    classified: list[_SavedTensorRecord] = []
    for record in records:
        shape = record.shape
        if record.dtype == torch.bfloat16 and shape in remaining_queries:
            remaining_queries.remove(shape)
            state_class = "checkpoint_query_input"
        elif record.dtype == torch.bfloat16 and shape in remaining_latents:
            remaining_latents.remove(shape)
            state_class = "checkpoint_latent_input"
        elif (
            record.dtype == torch.bfloat16
            and len(shape) == 3
            and shape[1:] == (heads, _VALUE_DIM)
        ):
            state_class = "expanded_value"
        elif len(shape) == 3 and shape[1:] == (heads, _QK_CONTENT + _ROPE_DIM):
            state_class = "expanded_key_or_uncheckpointed_query"
        elif (
            record.dtype == torch.float32
            and len(shape) == 3
            and shape[1:] == (heads, _VALUE_DIM)
        ):
            state_class = "partial_output_or_merge_state"
        elif record.dtype == torch.float32 and len(shape) == 2 and shape[1] == heads:
            state_class = "partial_lse_or_merge_state"
        else:
            state_class = "projection_or_autograd_auxiliary"
        classified.append(replace(record, state_class=state_class))
    if remaining_queries or remaining_latents:
        raise AssertionError(
            "outer saved state missed checkpoint inputs: "
            f"queries={remaining_queries}, latents={remaining_latents}"
        )
    return classified


class _TorchPackedAttentionAdapter:
    """Standard-PyTorch phase backend for mechanics tests, never the parity oracle."""

    def __init__(self):
        self.prepare_calls = 0
        self.forward_calls = 0
        self.raw_output_dtypes: list[torch.dtype] = []
        self.expanded_refs: list[weakref.ReferenceType] = []
        self.partial_refs: list[weakref.ReferenceType] = []
        self.partial_lse_refs: list[weakref.ReferenceType] = []

    def prepare(self, **_kwargs) -> None:
        self.prepare_calls += 1

    def forward_phase(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_q: torch.Tensor,
        cu_kv: torch.Tensor,
        max_q: int,
        max_kv: int,
        causal: bool,
        scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del max_q, max_kv
        self.forward_calls += 1
        k, v = _SaveExpandedKV.apply(k, v)
        self.expanded_refs.extend((weakref.ref(k), weakref.ref(v)))
        outputs: list[torch.Tensor] = []
        stats: list[torch.Tensor] = []
        q_offsets = cu_q.cpu().tolist()
        kv_offsets = cu_kv.cpu().tolist()
        for index in range(len(q_offsets) - 1):
            q_seq = q[q_offsets[index] : q_offsets[index + 1]]
            k_seq = k[kv_offsets[index] : kv_offsets[index + 1]]
            v_seq = v[kv_offsets[index] : kv_offsets[index + 1]]
            scores = torch.einsum("qhd,khd->hqk", q_seq.float(), k_seq.float()) * scale
            if causal:
                assert q_seq.size(0) == k_seq.size(0)
                mask = torch.triu(
                    torch.ones(
                        q_seq.size(0), k_seq.size(0), dtype=torch.bool, device=q.device
                    ),
                    diagonal=1,
                )
                scores = scores.masked_fill(mask.unsqueeze(0), -torch.inf)
            stats.append(torch.logsumexp(scores, dim=-1).transpose(0, 1))
            probabilities = torch.softmax(scores, dim=-1).to(v_seq.dtype)
            outputs.append(torch.einsum("hqk,khd->qhd", probabilities, v_seq))
        raw = torch.cat(outputs, dim=0).to(torch.bfloat16)
        canonical = raw.float()
        lse = torch.cat(stats, dim=0).float()
        self.raw_output_dtypes.append(raw.dtype)
        self.partial_refs.append(weakref.ref(canonical))
        self.partial_lse_refs.append(weakref.ref(lse))
        return canonical, lse


def _independent_torch_phase_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_q: torch.Tensor,
    cu_kv: torch.Tensor,
    *,
    causal: bool,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Standard-PyTorch ragged phase oracle with the official BF16 probability cast."""

    outputs: list[torch.Tensor] = []
    stats: list[torch.Tensor] = []
    q_offsets = cu_q.cpu().tolist()
    kv_offsets = cu_kv.cpu().tolist()
    for index in range(len(q_offsets) - 1):
        q_seq = q[q_offsets[index] : q_offsets[index + 1]]
        k_seq = k[kv_offsets[index] : kv_offsets[index + 1]]
        v_seq = v[kv_offsets[index] : kv_offsets[index + 1]]
        scores = torch.einsum("qhd,khd->hqk", q_seq.float(), k_seq.float()) * scale
        if causal:
            assert q_seq.size(0) == k_seq.size(0)
            mask = torch.triu(
                torch.ones(
                    q_seq.size(0), k_seq.size(0), dtype=torch.bool, device=q.device
                ),
                diagonal=1,
            )
            scores = scores.masked_fill(mask.unsqueeze(0), -torch.inf)
        stats.append(torch.logsumexp(scores, dim=-1).transpose(0, 1))
        probabilities = torch.softmax(scores, dim=-1).to(torch.bfloat16)
        outputs.append(torch.einsum("hqk,khd->qhd", probabilities, v_seq))
    return torch.cat(outputs).to(torch.bfloat16).float(), torch.cat(stats).float()


def _qualified_real_backend_runtime_or_skip(
    backend: AttnBackend,
) -> latent_cp.QualifiedBackendTuple:
    try:
        runtime = latent_cp_backend._runtime_backend_tuple(backend)
    except latent_cp.BackendNotQualifiedError as error:
        pytest.skip(
            "unable to resolve the installed latent-CP backend tuple before adapter "
            f"construction: {error}; exact qualified tuples are "
            f"{latent_cp.QUALIFIED_BACKEND_CONFIGS!r}"
        )
    if runtime not in latent_cp.QUALIFIED_BACKEND_CONFIGS:
        pytest.skip(
            f"installed latent-CP backend tuple {runtime!r} is not exactly qualified; "
            f"exact qualified tuples are {latent_cp.QUALIFIED_BACKEND_CONFIGS!r}"
        )
    return runtime


@pytest.mark.parametrize("cp_size", [1, 2, 4])
def test_phase_plan_exact_causal_pair_coverage(cp_size: int):
    lengths = (16, 8)
    cu = _cumulative(lengths)
    expected_pairs: set[tuple[int, int, int]] = set()
    offset = 0
    for sequence, length in enumerate(lengths):
        expected_pairs.update(
            (sequence, offset + query, offset + key)
            for query in range(length)
            for key in range(query + 1)
        )
        offset += length

    observed: list[tuple[int, int, int]] = []
    for rank in range(cp_size):
        local_tokens = sum(lengths) // cp_size
        layout = latent_cp.build_zigzag_layout(cu, local_tokens, cp_size, rank)
        for derived_cu in (layout.cu_full, layout.cu_half):
            assert derived_cu.dtype == torch.int32
            assert derived_cu.is_contiguous()
        assert [phase.owner for phase in layout.phases] == [
            (rank - phase) % cp_size for phase in range(cp_size)
        ]
        assert layout.cu_full.tolist() == [0] + [
            sum(lengths[: index + 1]) // cp_size for index in range(len(lengths))
        ]
        half_divisor = cp_size if cp_size == 1 else 2 * cp_size
        assert layout.cu_half.tolist() == [0] + [
            sum(lengths[: index + 1]) // half_divisor for index in range(len(lengths))
        ]
        expected_kinds = [
            "diagonal" if phase == 0 else "lower" if phase <= rank else "upper"
            for phase in range(cp_size)
        ]
        assert [phase.kind for phase in layout.phases] == expected_kinds
        for phase in layout.phases:
            for phase_cu in (phase.cu_seqlens_q, phase.cu_seqlens_kv):
                assert phase_cu.dtype == torch.int32
                assert phase_cu.is_contiguous()
            q_sequences, kv_sequences = _phase_global_rows(
                lengths, cp_size, rank, phase
            )
            for sequence, (q_rows, kv_rows) in enumerate(
                zip(q_sequences, kv_sequences)
            ):
                for q_position, q_global in enumerate(q_rows):
                    for k_position, k_global in enumerate(kv_rows):
                        if phase.causal and k_position > q_position:
                            continue
                        observed.append((sequence, q_global, k_global))
    assert len(observed) == len(set(observed))
    assert set(observed) == expected_pairs


def test_cp1_planner_and_transport_are_exact_no_ring_degenerations():
    lengths = (7, 5)
    layout = latent_cp.build_zigzag_layout(
        _cumulative(lengths), local_tokens=sum(lengths), cp_size=1, cp_rank=0
    )
    assert layout.cu_full.tolist() == [0, 7, 12]
    assert layout.cu_half.tolist() == layout.cu_full.tolist()
    assert torch.equal(layout.front_indices, torch.arange(sum(lengths)))
    assert layout.back_indices.numel() == 0
    assert len(layout.phases) == 1
    phase = layout.phases[0]
    assert (phase.phase, phase.owner, phase.kind, phase.causal) == (
        0,
        0,
        "diagonal",
        True,
    )
    assert phase.q_indices is layout.front_indices
    assert phase.kv_indices is layout.front_indices

    cp_group = object()
    payload = torch.randn(sum(lengths), 11, requires_grad=True)
    with (
        mock.patch.object(latent_cp.dist, "get_process_group_ranks", return_value=[17]),
        mock.patch.object(latent_cp.dist, "get_rank", return_value=0),
        mock.patch.object(latent_cp.dist, "get_world_size", return_value=1),
        mock.patch.object(
            latent_cp._LatentRingExchange,
            "apply",
            side_effect=AssertionError("CP=1 must not launch P2P"),
        ) as exchange,
    ):
        leases = list(
            latent_cp.P2PRingTransport(cp_group).iter_payloads(payload, layout.phases)
        )
    assert len(leases) == 1
    assert leases[0].owner == 0
    assert leases[0].tensor is payload
    exchange.assert_not_called()
    leases[0].tensor.sum().backward()
    torch.testing.assert_close(payload.grad, torch.ones_like(payload), rtol=0, atol=0)


@pytest.mark.parametrize("rope_type", ["rope", "yarn"])
@pytest.mark.parametrize("cp_size", [2, 4])
def test_independent_packed_zigzag_global_positions(rope_type: str, cp_size: int):
    lengths = (16, 8)
    dim = 64
    factor = 40.0 if rope_type == "yarn" else 1.0
    frequencies = _official_freqs_cis(
        dim=dim,
        max_seq_len=max(lengths),
        base=10000,
        factor=factor,
        original_max_positions=8,
        beta_fast=32,
        beta_slow=1,
        device=torch.device("cpu"),
    )
    full = torch.arange(sum(lengths) * dim, dtype=torch.float32).reshape(
        sum(lengths), 1, dim
    )
    rotated_full: list[torch.Tensor] = []
    offset = 0
    for length in lengths:
        rotated_full.append(
            _official_apply_rotary(full[offset : offset + length], frequencies[:length])
        )
        offset += length
    rotated_full_tensor = torch.cat(rotated_full)
    for rank in range(cp_size):
        global_indices = _zigzag_global_indices(lengths, cp_size, rank)
        positions = _zigzag_positions(lengths, cp_size, rank)
        local = full.index_select(0, global_indices)
        expected = rotated_full_tensor.index_select(0, global_indices)
        actual_parts: list[torch.Tensor] = []
        local_offset = 0
        for length in lengths:
            local_length = length // cp_size
            local_seq = local[local_offset : local_offset + local_length]
            local_positions = positions[local_offset : local_offset + local_length]
            actual_parts.append(
                _official_apply_rotary(
                    local_seq, frequencies.index_select(0, local_positions)
                )
            )
            local_offset += local_length
        torch.testing.assert_close(torch.cat(actual_parts), expected)


def test_merge_matches_direct_softmax_and_gradients():
    torch.manual_seed(17)
    logits_a = torch.randn(5, 3, 7, dtype=torch.float32, requires_grad=True)
    logits_b = torch.randn(5, 3, 4, dtype=torch.float32, requires_grad=True)
    value_a = torch.randn(7, 3, 6, dtype=torch.float32, requires_grad=True)
    value_b = torch.randn(4, 3, 6, dtype=torch.float32, requires_grad=True)
    probs_a = torch.softmax(logits_a, dim=-1)
    probs_b = torch.softmax(logits_b, dim=-1)
    output_a = torch.einsum("qhk,khd->qhd", probs_a, value_a)
    output_b = torch.einsum("qhk,khd->qhd", probs_b, value_b)
    lse_a = torch.logsumexp(logits_a, dim=-1)
    lse_b = torch.logsumexp(logits_b, dim=-1)
    merged, merged_lse = latent_cp.merge_attention_partials(
        output_a, lse_a, output_b, lse_b
    )

    logits_direct = torch.cat((logits_a, logits_b), dim=-1)
    values_direct = torch.cat((value_a, value_b), dim=0)
    direct = torch.einsum(
        "qhk,khd->qhd", torch.softmax(logits_direct, dim=-1), values_direct
    )
    direct_lse = torch.logsumexp(logits_direct, dim=-1)
    torch.testing.assert_close(merged, direct, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(merged_lse, direct_lse, rtol=2e-6, atol=2e-6)

    upstream = torch.randn_like(merged)
    merged.backward(upstream, retain_graph=True)
    merged_grads = [
        tensor.grad.detach().clone()
        for tensor in (logits_a, logits_b, value_a, value_b)
    ]
    for tensor in (logits_a, logits_b, value_a, value_b):
        tensor.grad = None
    direct.backward(upstream)
    for actual, tensor in zip(merged_grads, (logits_a, logits_b, value_a, value_b)):
        torch.testing.assert_close(actual, tensor.grad, rtol=3e-6, atol=3e-6)


def test_cudnn_selective_recompute_matches_native_projection_gradients():
    """The selective path must replay projection math, not attention forward."""

    class FakeAdapter:
        def __init__(self):
            self.forward_calls = 0
            self.backward_calls = 0

        def _execute_forward(self, q, k, v, *_args):
            self.forward_calls += 1
            return q + k + v, torch.zeros(q.shape[:2], dtype=torch.float32)

        def _execute_backward(self, q, k, v, output, grad_output, stats, *_args):
            del output, stats
            self.backward_calls += 1
            return (
                grad_output.to(q.dtype),
                grad_output.to(k.dtype),
                grad_output.to(v.dtype),
            )

    torch.manual_seed(51)
    query = torch.randn(5, 1, 2, requires_grad=True)
    payload = torch.randn(5, 2, requires_grad=True)
    weight = torch.randn(4, 2, requires_grad=True)
    query_ref = query.detach().clone().requires_grad_(True)
    payload_ref = payload.detach().clone().requires_grad_(True)
    weight_ref = weight.detach().clone().requires_grad_(True)
    cu = torch.tensor([0, 5], dtype=torch.int32)
    indices = torch.arange(5)
    phase = latent_cp.PhaseSpec(
        phase=0,
        owner=0,
        kind="diagonal",
        q_indices=indices,
        kv_indices=indices,
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu,
        max_seqlen_q=5,
        max_seqlen_kv=5,
        causal=True,
    )
    replay_calls = 0

    def expand_phase_kv(latent, _phase):
        nonlocal replay_calls
        replay_calls += 1
        expanded = F.linear(latent, weight).view(5, 1, 4)
        return expanded[..., :2], expanded[..., 2:]

    adapter = FakeAdapter()
    output, lse = latent_cp_cudnn_backend._CudnnRecomputedPhaseFunction.apply(
        query,
        payload,
        weight,
        phase,
        1.0,
        adapter,
        expand_phase_kv,
    )
    upstream = torch.randn_like(output).to(torch.bfloat16).float()
    (output * upstream).sum().backward()

    expanded_ref = F.linear(payload_ref, weight_ref).view(5, 1, 4)
    reference = query_ref + expanded_ref[..., :2] + expanded_ref[..., 2:]
    (reference * upstream).sum().backward()
    torch.testing.assert_close(output, reference)
    torch.testing.assert_close(lse, torch.zeros_like(lse))
    torch.testing.assert_close(query.grad, query_ref.grad)
    torch.testing.assert_close(payload.grad, payload_ref.grad)
    torch.testing.assert_close(weight.grad, weight_ref.grad)
    assert replay_calls == 2
    assert adapter.forward_calls == 1
    assert adapter.backward_calls == 1


def test_merge_scatter_and_cudnn_proxy_extremes():
    output = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    lse = torch.tensor([[0.0, -10000.0, -torch.inf], [80.0, -80.0, 0.0]])
    output_version = output._version
    lse_version = lse._version
    scattered, scattered_lse = latent_cp.scatter_upper_phase(
        output, lse, torch.tensor([1, 3]), 4
    )
    assert output._version == output_version and lse._version == lse_version
    assert torch.equal(scattered[[1, 3]], output)
    assert torch.count_nonzero(scattered[[0, 2]]) == 0
    assert torch.equal(scattered_lse[[1, 3]], lse)
    assert torch.isneginf(scattered_lse[[0, 2]]).all()

    other_output = torch.full_like(scattered, 2.0)
    other_lse = torch.tensor(
        [
            [-torch.inf, -torch.inf, -torch.inf],
            [0.0, -10001.0, -80.0],
            [-torch.inf, -torch.inf, -torch.inf],
            [-80.0, 80.0, 0.0],
        ],
        dtype=torch.float32,
    )
    merged, merged_lse = latent_cp.merge_attention_partials(
        scattered, scattered_lse, other_output, other_lse
    )
    assert torch.isfinite(merged).all()
    assert not torch.isnan(merged_lse).any()

    threshold = math.sqrt(torch.finfo(torch.float32).tiny)
    grad = torch.tensor(
        [
            [[1.0, 2.0, 3.0, 4.0]],
            [[0.0, 0.0, 0.0, 0.0]],
            [[math.sqrt(threshold) / 4, 0.0, 0.0, 0.0]],
            [[torch.inf, 0.0, 0.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    partial = torch.randn_like(grad)
    grad_lse = torch.tensor([[2.5], [3.0], [1.0], [4.0]], dtype=torch.float32)
    corrected, returned_grad = latent_cp.cudnn_backward_proxy(partial, grad, grad_lse)
    torch.testing.assert_close(returned_grad, grad)
    safe_dot = torch.sum(grad[0] * corrected[0], dim=-1)
    expected_dot = torch.sum(grad[0] * partial[0], dim=-1) - grad_lse[0]
    torch.testing.assert_close(safe_dot, expected_dot, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(corrected[1], partial[1])
    torch.testing.assert_close(corrected[2], partial[2])
    torch.testing.assert_close(corrected[3], partial[3])
    assert torch.isfinite(corrected[:3]).all()


def test_cudnn_proxy_bf16_boundary_diagnostic():
    """Expose the real BF16 o/dO dot-product residual hidden by the FP32 identity test."""

    torch.manual_seed(_SEED)
    rows, heads, width = 64, 8, _VALUE_DIM
    partial = torch.randn(rows, heads, width).to(torch.bfloat16).float()
    other = torch.randn_like(partial).to(torch.bfloat16).float()
    lse_a = torch.randn(rows, heads)
    lse_b = torch.randn(rows, heads)
    merged_lse = torch.logaddexp(lse_a, lse_b)
    weight_a = torch.exp(lse_a - merged_lse)
    weight_b = torch.exp(lse_b - merged_lse)
    global_output = partial * weight_a.unsqueeze(-1) + other * weight_b.unsqueeze(-1)
    upstream = torch.randn_like(global_output).to(torch.bfloat16).float()
    phase_grad = weight_a.unsqueeze(-1) * upstream
    grad_lse = torch.sum(phase_grad * (partial - global_output), dim=-1)

    current_output, current_grad = latent_cp.cudnn_backward_proxy(
        partial, phase_grad, grad_lse
    )
    quantized_grad = current_grad.to(torch.bfloat16).float()
    current_dot = torch.sum(
        quantized_grad * current_output.to(torch.bfloat16).float(), dim=-1
    )

    partial_bf16 = partial.to(torch.bfloat16).float()
    norm2 = torch.sum(quantized_grad * quantized_grad, dim=-1)
    quantized_target = torch.sum(quantized_grad * partial_bf16, dim=-1) - grad_lse
    quantized_output = (
        (partial_bf16 - (grad_lse / norm2).unsqueeze(-1) * quantized_grad)
        .to(torch.bfloat16)
        .float()
    )
    pivot_index = quantized_grad.abs().argmax(dim=-1, keepdim=True)
    pivot = quantized_grad.gather(-1, pivot_index)
    assert torch.count_nonzero(pivot) == pivot.numel()
    for _ in range(2):
        residual = quantized_target - torch.sum(
            quantized_grad * quantized_output, dim=-1
        )
        pivot_value = quantized_output.gather(-1, pivot_index)
        quantized_output = (
            quantized_output.scatter(
                -1, pivot_index, pivot_value + residual.unsqueeze(-1) / pivot
            )
            .to(torch.bfloat16)
            .float()
        )
    quantized_dot = torch.sum(quantized_grad * quantized_output, dim=-1)

    global_proxy = global_output.to(torch.bfloat16).float()
    global_dot = torch.sum(quantized_grad * global_proxy, dim=-1)
    real_target = torch.sum(phase_grad * global_output, dim=-1)

    def residual_summary(
        actual: torch.Tensor, target: torch.Tensor
    ) -> dict[str, float]:
        residual = (actual - target).abs()
        return {
            "mean": float(residual.mean()),
            "p99": float(torch.quantile(residual, 0.99)),
            "max": float(residual.max()),
        }

    evidence = {
        "boundary_target": {
            "current": residual_summary(current_dot, quantized_target),
            "o_global": residual_summary(global_dot, quantized_target),
            "quantized_do_aware": residual_summary(quantized_dot, quantized_target),
        },
        "event": "mla_latent_cp_cudnn_proxy_bf16_diagnostic",
        "real_target": {
            "current": residual_summary(current_dot, real_target),
            "o_global": residual_summary(global_dot, real_target),
            "quantized_do_aware": residual_summary(quantized_dot, real_target),
        },
        "seed": _SEED,
        "shape": [rows, heads, width],
    }
    print(json.dumps(evidence, sort_keys=True, separators=(",", ":")), flush=True)
    assert evidence["real_target"]["current"]["mean"] > 0.0
    assert (
        evidence["real_target"]["o_global"]["mean"]
        < evidence["real_target"]["current"]["mean"]
    )
    assert (
        evidence["boundary_target"]["quantized_do_aware"]["mean"]
        < evidence["boundary_target"]["current"]["mean"]
    )


def test_spec_factory_is_non_mutating_and_bypasses_core_attention_wrapper():
    base = _base_mla_spec()
    original_module = base.module
    original_core = base.submodules.core_attention
    original_params = dict(base.params)
    result = latent_cp.make_mla_with_latent_cp_spec(base)
    assert base.module is original_module
    assert base.submodules.core_attention is original_core
    assert base.params == original_params
    assert result is not base
    assert result.submodules is not base.submodules
    assert result.params is not base.params
    assert result.module is latent_cp.MLAWithLatentCP
    assert result.submodules.core_attention is IdentityOp
    assert result.submodules.q_layernorm is latent_cp._build_local_latent_norm
    assert result.submodules.kv_layernorm is latent_cp._build_local_latent_norm


def test_factory_rejects_unsupported_projection_and_mask_specs():
    base = _base_mla_spec()
    outer = get_gpt_layer_local_spec(
        num_experts=None,
        moe_grouped_gemm=False,
        qk_layernorm=True,
        multi_latent_attention=True,
        normalization="RMSNorm",
    )
    with pytest.raises(ValueError, match="base_mla_spec"):
        latent_cp.make_mla_with_latent_cp_spec(outer)
    with pytest.raises(ValueError, match="norms"):
        latent_cp.make_mla_with_latent_cp_spec(
            replace(base, submodules=replace(base.submodules, q_layernorm=IdentityOp))
        )
    with pytest.raises(ValueError, match="fused MLA down"):
        latent_cp.make_mla_with_latent_cp_spec(
            replace(
                base, submodules=replace(base.submodules, linear_qkv_down_proj=object())
            )
        )
    with pytest.raises(ValueError, match="causal"):
        latent_cp.make_mla_with_latent_cp_spec(
            replace(base, params={"attn_mask_type": AttnMaskType.no_mask})
        )


def test_mla_latent_cp_config_validation_is_fail_closed():
    from megatron.core.transformer.transformer_config import TransformerConfig

    assert TransformerConfig.__dataclass_fields__["mla_latent_cp"].default is False
    config = _make_config()
    assert config.mla_latent_cp is True
    with pytest.raises(ValueError, match="multi_latent_attention"):
        replace(config, multi_latent_attention=False)
    with pytest.raises(ValueError, match="gated_delta_net"):
        replace(config, experimental_attention_variant="dsa")
    with pytest.raises(ValueError, match="zigzag"):
        replace(config, cp_partition_mode="contiguous")
    with pytest.raises(ValueError, match="MTP"):
        replace(config, mtp_num_layers=1)
    with pytest.raises(ValueError, match="attention_backend"):
        replace(config, attention_backend=AttnBackend.auto)


def test_feature_configures_gpt_decoder_without_mutation():
    from megatron.core.transformer.transformer_block import (
        TransformerBlock,
        TransformerBlockSubmodules,
    )

    base_layer = get_gpt_layer_local_spec(
        num_experts=None,
        moe_grouped_gemm=False,
        qk_layernorm=True,
        multi_latent_attention=True,
        normalization="RMSNorm",
    )
    untouched_layer = replace(
        base_layer,
        submodules=replace(
            base_layer.submodules, self_attention=ModuleSpec(module=object)
        ),
    )
    block_spec = TransformerBlockSubmodules(layer_specs=[base_layer, untouched_layer])
    configured = latent_cp.configure_mla_latent_cp_decoder(block_spec)

    assert configured is not block_spec
    assert configured.layer_specs[0] is not base_layer
    assert (
        configured.layer_specs[0].submodules.self_attention.module
        is latent_cp.MLAWithLatentCP
    )
    assert configured.layer_specs[1] is untouched_layer
    assert base_layer.submodules.self_attention.module is base_mla.MLASelfAttention

    gated_attention = latent_cp.get_mla_with_latent_cp_spec()
    assert gated_attention.module is latent_cp.MLAWithLatentCP
    assert gated_attention.metainfo == {"fuse_input_layernorm": False}
    assert gated_attention.submodules.linear_gate is latent_cp.ColumnParallelLinear


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_feature_configures_hybrid_stack_without_mutation():
    from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
    from megatron.core.transformer.attention import SelfAttention

    base_attention = hybrid_stack_spec.submodules.attention_layer
    assert base_attention.submodules.self_attention.module is SelfAttention
    latent_stack = latent_cp.configure_mla_latent_cp_hybrid_stack(hybrid_stack_spec)
    assert latent_stack is not hybrid_stack_spec
    assert latent_stack.submodules is not hybrid_stack_spec.submodules
    assert latent_stack.module is hybrid_stack_spec.module
    assert (
        latent_stack.submodules.attention_layer.submodules.self_attention.module
        is latent_cp.MLAWithLatentCP
    )
    assert hybrid_stack_spec.submodules.attention_layer is base_attention
    assert base_attention.submodules.self_attention.module is SelfAttention


def test_qualification_constants_are_exact_and_fail_closed():
    assert latent_cp.CUDNN_FRONTEND_SOURCE_REV == EXPECTED_CUDNN_FRONTEND_SOURCE_REV
    assert latent_cp.QUALIFIED_BACKEND_CONFIGS == EXPECTED_QUALIFIED_BACKEND_CONFIGS
    assert isinstance(latent_cp.QUALIFIED_BACKEND_CONFIGS, tuple)
    assert len(latent_cp.QUALIFIED_BACKEND_CONFIGS) == 3
    assert tuple(EXPECTED_QUALIFICATION_EPS) == EXPECTED_QUALIFIED_BACKEND_CONFIGS
    assert set(EXPECTED_QUALIFICATION_EPS) == set(EXPECTED_QUALIFIED_BACKEND_CONFIGS)
    assert len(EXPECTED_QUALIFICATION_EPS) == 3
    assert all(
        0 < eps <= _PARITY_EPS_CEILING for eps in EXPECTED_QUALIFICATION_EPS.values()
    )
    for entry in latent_cp.QUALIFIED_BACKEND_CONFIGS:
        backend, package, runtime, capability = entry
        assert backend in (AttnBackend.fused, AttnBackend.flash)
        assert package and "*" not in package
        assert runtime and "*" not in runtime
        assert (
            isinstance(capability, tuple)
            and len(capability) == 2
            and all(isinstance(value, int) for value in capability)
        )

    candidate = (
        AttnBackend.flash,
        "unqualified-test",
        "flash-attn-4==unqualified-test",
        (10, 0),
    )
    assert candidate not in latent_cp.QUALIFIED_BACKEND_CONFIGS
    with (
        mock.patch.object(
            latent_cp_backend, "_runtime_backend_tuple", return_value=candidate
        ),
        mock.patch.object(latent_cp_backend, "_shared_cudnn_adapter") as cudnn_factory,
        mock.patch.object(latent_cp_backend, "FA4Adapter") as flash_factory,
        pytest.raises(latent_cp.BackendNotQualifiedError),
    ):
        latent_cp._qualified_backend_adapter(AttnBackend.flash)
    cudnn_factory.assert_not_called()
    flash_factory.assert_not_called()


def test_qualified_backend_dispatch_selects_only_the_direct_adapter():
    for runtime in EXPECTED_QUALIFIED_BACKEND_CONFIGS:
        backend = runtime[0]
        cudnn_adapter = object()
        flash_adapter = object()
        with (
            mock.patch.object(
                latent_cp_backend, "_shared_cudnn_adapter", return_value=cudnn_adapter
            ) as cudnn_factory,
            mock.patch.object(
                latent_cp_backend, "FA4Adapter", return_value=flash_adapter
            ) as flash_factory,
        ):
            adapter, identity = latent_cp._qualified_backend_adapter(backend, runtime)
        assert identity == runtime
        if backend is AttnBackend.fused:
            assert adapter is cudnn_adapter
            cudnn_factory.assert_called_once_with(runtime)
            flash_factory.assert_not_called()
        else:
            assert adapter is flash_adapter
            flash_factory.assert_called_once_with()
            cudnn_factory.assert_not_called()


def test_real_backend_gate_skips_unqualified_before_adapter_construction():
    unqualified = (AttnBackend.fused, "installed-but-unqualified", "9.99.0", (9, 0))
    with (
        mock.patch.object(
            latent_cp_backend, "_runtime_backend_tuple", return_value=unqualified
        ),
        mock.patch.object(
            latent_cp_backend, "_qualified_backend_adapter"
        ) as adapter_factory,
        mock.patch.object(latent_cp_backend, "_shared_cudnn_adapter") as cudnn_factory,
        mock.patch.object(
            latent_cp_cudnn_backend, "CudnnFusedAttentionAdapter"
        ) as graph_factory,
        mock.patch.object(latent_cp_backend, "FA4Adapter") as flash_factory,
        pytest.raises(pytest.skip.Exception, match="not exactly qualified"),
    ):
        _qualified_real_backend_runtime_or_skip(AttnBackend.fused)
    adapter_factory.assert_not_called()
    cudnn_factory.assert_not_called()
    graph_factory.assert_not_called()
    flash_factory.assert_not_called()

    for runtime in EXPECTED_QUALIFIED_BACKEND_CONFIGS:
        with (
            mock.patch.object(
                latent_cp_backend, "_runtime_backend_tuple", return_value=runtime
            ),
            mock.patch.object(
                latent_cp_backend, "_qualified_backend_adapter"
            ) as adapter_factory,
        ):
            assert _qualified_real_backend_runtime_or_skip(runtime[0]) == runtime
        adapter_factory.assert_not_called()


def test_fa4_adapter_uses_only_public_varlen_contract(monkeypatch):
    calls = []
    layout = latent_cp.build_zigzag_layout(
        _cumulative((8, 4)), local_tokens=6, cp_size=2, cp_rank=1
    )
    phase = layout.phases[1]
    assert phase.kind == "lower"
    cu_q = phase.cu_seqlens_q
    cu_kv = phase.cu_seqlens_kv

    def public_varlen(q, k, v, **kwargs):
        assert kwargs["cu_seqlens_q"] is cu_q
        assert kwargs["cu_seqlens_k"] is cu_kv
        assert cu_q.dtype == cu_kv.dtype == torch.int32
        assert cu_q.is_contiguous() and cu_kv.is_contiguous()
        calls.append((q, k, v, kwargs))
        return torch.zeros(
            q.size(0), q.size(1), v.size(2), dtype=torch.bfloat16
        ), torch.zeros(q.size(1), q.size(0), dtype=torch.float32)

    fake_module = SimpleNamespace(flash_attn_varlen_func=public_varlen)
    real_import = latent_cp.importlib.import_module

    def import_module(name):
        return fake_module if name == "flash_attn.cute" else real_import(name)

    monkeypatch.setattr(latent_cp.importlib, "import_module", import_module)
    adapter = latent_cp.FA4Adapter()
    q = torch.randn(int(cu_q[-1].item()), 3, 192, dtype=torch.bfloat16)
    k = torch.randn(int(cu_kv[-1].item()), 3, 192, dtype=torch.bfloat16)
    v = torch.randn(int(cu_kv[-1].item()), 3, 128, dtype=torch.bfloat16)
    output, lse = adapter.forward_phase(
        q,
        k,
        v,
        cu_q,
        cu_kv,
        phase.max_seqlen_q,
        phase.max_seqlen_kv,
        phase.causal,
        0.125,
    )
    assert output.dtype == torch.float32 and output.shape == (6, 3, 128)
    assert lse.dtype == torch.float32 and lse.shape == (6, 3)
    assert len(calls) == 1
    kwargs = calls[0][3]
    assert kwargs["cu_seqlens_q"] is cu_q
    assert kwargs["cu_seqlens_k"] is cu_kv
    assert {
        key: value for key, value in kwargs.items() if not key.startswith("cu_")
    } == {
        "max_seqlen_q": 4,
        "max_seqlen_k": 2,
        "softmax_scale": 0.125,
        "causal": False,
        "return_lse": True,
    }


def test_fa4_adapter_rejects_invalid_metadata_without_conversion(monkeypatch):
    calls = []

    def public_varlen(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("invalid metadata reached the public FA4 call")

    fake_module = SimpleNamespace(flash_attn_varlen_func=public_varlen)
    real_import = latent_cp.importlib.import_module

    def import_module(name):
        return fake_module if name == "flash_attn.cute" else real_import(name)

    monkeypatch.setattr(latent_cp.importlib, "import_module", import_module)
    adapter = latent_cp.FA4Adapter()
    q = torch.randn(6, 3, 192, dtype=torch.bfloat16)
    k = torch.randn(3, 3, 192, dtype=torch.bfloat16)
    v = torch.randn(3, 3, 128, dtype=torch.bfloat16)
    cu_q = torch.tensor([0, 2, 6], dtype=torch.int32)
    cu_kv = torch.tensor([0, 1, 3], dtype=torch.int32)

    for bad_q, bad_kv in ((cu_q.to(torch.int64), cu_kv), (cu_q, cu_kv.to(torch.int64))):
        with pytest.raises(ValueError, match="dtype torch.int32"):
            adapter.forward_phase(q, k, v, bad_q, bad_kv, 4, 2, False, 0.125)

    noncontiguous_q = torch.tensor([0, -1, 2, -1, 6], dtype=torch.int32)[::2]
    noncontiguous_kv = torch.tensor([0, -1, 1, -1, 3], dtype=torch.int32)[::2]
    assert not noncontiguous_q.is_contiguous()
    assert not noncontiguous_kv.is_contiguous()
    for bad_q, bad_kv in ((noncontiguous_q, cu_kv), (cu_q, noncontiguous_kv)):
        with pytest.raises(ValueError, match="must be contiguous"):
            adapter.forward_phase(q, k, v, bad_q, bad_kv, 4, 2, False, 0.125)

    meta_q = torch.empty(cu_q.shape, dtype=torch.int32, device="meta")
    meta_kv = torch.empty(cu_kv.shape, dtype=torch.int32, device="meta")
    for bad_q, bad_kv in ((meta_q, cu_kv), (cu_q, meta_kv)):
        with pytest.raises(ValueError, match="colocated with their Q/K tensors"):
            adapter.forward_phase(q, k, v, bad_q, bad_kv, 4, 2, False, 0.125)

    assert calls == []


def test_cudnn_canonical_rank4_ragged_metadata_contract():
    class FakeTensor:
        def __init__(self, graph, uid=None, dim=None, stride=None, data_type=None):
            self.graph = graph
            self.uid = uid
            self.dim = dim
            self.stride = stride
            self.data_type = data_type
            self.output = False
            self.ragged_offset = None
            if uid is not None:
                graph.tensors[uid] = self

        def set_uid(self, uid):
            self.uid = uid
            self.graph.tensors[uid] = self
            return self

        def set_output(self, output):
            self.output = output
            return self

        def set_dim(self, dim):
            self.dim = tuple(dim)
            return self

        def set_stride(self, stride):
            self.stride = tuple(stride)
            return self

        def set_data_type(self, data_type):
            self.data_type = data_type
            return self

        def set_ragged_offset(self, offset):
            self.ragged_offset = offset
            return self

    class FakeGraph:
        def __init__(self):
            self.tensors = {}
            self.sdpa_kwargs = None
            self.sdpa_backward_kwargs = None

        def tensor(self, *, uid, dim, stride, data_type):
            return FakeTensor(self, uid, tuple(dim), tuple(stride), data_type)

        def sdpa(self, **kwargs):
            self.sdpa_kwargs = kwargs
            return FakeTensor(self), FakeTensor(self)

        def sdpa_backward(self, **kwargs):
            self.sdpa_backward_kwargs = kwargs
            return FakeTensor(self), FakeTensor(self), FakeTensor(self)

    data_type = SimpleNamespace(
        BFLOAT16="BFLOAT16", FLOAT="FLOAT", INT32="INT32", INT64="INT64"
    )
    adapter = object.__new__(latent_cp.CudnnFusedAttentionAdapter)
    adapter.cudnn = SimpleNamespace(
        data_type=data_type, diagonal_alignment=SimpleNamespace(TOP_LEFT="TOP_LEFT")
    )
    graphs = []

    def new_graph(_key):
        graph = FakeGraph()
        graphs.append(graph)
        return graph

    adapter._new_graph = new_graph
    adapter._build_graph = lambda _graph: None
    key = latent_cp._CudnnPlanKey(
        process_id=1,
        device_index=0,
        frontend_version="frontend",
        runtime_version="runtime",
        dtype=torch.bfloat16,
        sm=(9, 0),
        batch=3,
        heads=2,
        qk_dim=192,
        v_dim=128,
        max_q=4,
        max_kv=4,
        capacity_q=64,
        capacity_kv=64,
        causal=False,
        scale=0.125,
    )
    forward_graph = adapter._build_forward_graph(key)
    backward_graph = adapter._build_backward_graph(key)
    assert graphs == [forward_graph, backward_graph]
    uid = latent_cp._CudnnUid
    descriptor_contract = {
        uid.SEQ_Q: ((3, 1, 1, 1), data_type.INT32),
        uid.SEQ_KV: ((3, 1, 1, 1), data_type.INT32),
        uid.Q_OFFSET: ((4, 1, 1, 1), data_type.INT64),
        uid.K_OFFSET: ((4, 1, 1, 1), data_type.INT64),
        uid.V_OFFSET: ((4, 1, 1, 1), data_type.INT64),
        uid.O_OFFSET: ((4, 1, 1, 1), data_type.INT64),
        uid.STATS_OFFSET: ((4, 1, 1, 1), data_type.INT64),
    }
    for graph in (forward_graph, backward_graph):
        for metadata_uid, (shape, dtype) in descriptor_contract.items():
            descriptor = graph.tensors[int(metadata_uid)]
            assert descriptor.dim == shape
            assert descriptor.stride == (1, 1, 1, 1)
            assert descriptor.data_type == dtype

    assert (
        forward_graph.sdpa_kwargs["seq_len_q"] is forward_graph.tensors[int(uid.SEQ_Q)]
    )
    assert (
        forward_graph.sdpa_kwargs["seq_len_kv"]
        is forward_graph.tensors[int(uid.SEQ_KV)]
    )
    assert (
        backward_graph.sdpa_backward_kwargs["seq_len_q"]
        is backward_graph.tensors[int(uid.SEQ_Q)]
    )
    assert (
        backward_graph.sdpa_backward_kwargs["seq_len_kv"]
        is backward_graph.tensors[int(uid.SEQ_KV)]
    )

    forward_attachments = {
        uid.Q: uid.Q_OFFSET,
        uid.K: uid.K_OFFSET,
        uid.V: uid.V_OFFSET,
        uid.O: uid.O_OFFSET,
        uid.STATS: uid.STATS_OFFSET,
    }
    backward_attachments = {
        uid.Q: uid.Q_OFFSET,
        uid.K: uid.K_OFFSET,
        uid.V: uid.V_OFFSET,
        uid.O: uid.O_OFFSET,
        uid.DO: uid.O_OFFSET,
        uid.STATS: uid.STATS_OFFSET,
        uid.DQ: uid.Q_OFFSET,
        uid.DK: uid.K_OFFSET,
        uid.DV: uid.V_OFFSET,
    }
    for graph, attachments in (
        (forward_graph, forward_attachments),
        (backward_graph, backward_attachments),
    ):
        for tensor_uid, offset_uid in attachments.items():
            assert (
                graph.tensors[int(tensor_uid)].ragged_offset
                is graph.tensors[int(offset_uid)]
            )

    metadata = adapter._metadata(
        torch.tensor([0, 2, 6, 7], dtype=torch.int32),
        torch.tensor([0, 1, 5, 9], dtype=torch.int32),
        heads=2,
        qk_dim=192,
        v_dim=128,
    )
    expected_buffers = {
        uid.SEQ_Q: (torch.int32, (3, 1, 1, 1), [2, 4, 1]),
        uid.SEQ_KV: (torch.int32, (3, 1, 1, 1), [1, 4, 4]),
        uid.Q_OFFSET: (torch.int64, (4, 1, 1, 1), [0, 768, 2304, 2688]),
        uid.K_OFFSET: (torch.int64, (4, 1, 1, 1), [0, 384, 1920, 3456]),
        uid.V_OFFSET: (torch.int64, (4, 1, 1, 1), [0, 256, 1280, 2304]),
        uid.O_OFFSET: (torch.int64, (4, 1, 1, 1), [0, 512, 1536, 1792]),
        uid.STATS_OFFSET: (torch.int64, (4, 1, 1, 1), [0, 4, 12, 14]),
    }
    bound_metadata = {
        int(metadata_uid): tensor for metadata_uid, tensor in metadata.items()
    }
    assert set(metadata) == set(expected_buffers)
    assert set(bound_metadata) == {
        int(metadata_uid) for metadata_uid in expected_buffers
    }
    for metadata_uid, (dtype, shape, values) in expected_buffers.items():
        buffer = bound_metadata[int(metadata_uid)]
        assert buffer.dtype == dtype
        assert tuple(buffer.shape) == shape
        assert tuple(buffer.stride()) == (1, 1, 1, 1)
        assert buffer.is_contiguous()
        assert torch.equal(buffer.flatten(), torch.tensor(values, dtype=dtype))


def test_cudnn_phase_execution_requires_prepared_plan():
    adapter = object.__new__(latent_cp.CudnnFusedAttentionAdapter)
    adapter._execution_lock = mock.MagicMock()
    adapter._plans = {}
    adapter._prepare_plan = mock.Mock(
        side_effect=AssertionError("phase execution must not build")
    )
    key = mock.sentinel.plan_key

    with pytest.raises(
        latent_cp.BackendPlanNotSupportedError, match="was not prepared"
    ):
        adapter._get_prepared_plan(key)
    adapter._prepare_plan.assert_not_called()

    prepared = mock.sentinel.plan
    adapter._plans[key] = prepared
    assert adapter._get_prepared_plan(key) is prepared


def test_cudnn_prepare_caches_phase_metadata_binding():
    adapter = object.__new__(latent_cp.CudnnFusedAttentionAdapter)
    adapter.device_index = 0
    adapter._execution_lock = latent_cp_cudnn_backend.threading.RLock()
    adapter._bindings = latent_cp_cudnn_backend.OrderedDict()
    plan_key = mock.sentinel.plan_key
    plan = mock.sentinel.plan
    metadata = {mock.sentinel.uid: mock.sentinel.tensor}
    adapter._plan_key_from_metadata = mock.Mock(return_value=plan_key)
    adapter._prepare_plan = mock.Mock(return_value=plan)
    adapter._metadata = mock.Mock(return_value=metadata)

    cu_q = torch.tensor([0, 2, 6], dtype=torch.int32)
    cu_kv = torch.tensor([0, 1, 3], dtype=torch.int32)
    phase = latent_cp.PhaseSpec(
        phase=0,
        owner=0,
        kind="lower",
        q_indices=torch.arange(6),
        kv_indices=torch.arange(3),
        cu_seqlens_q=cu_q,
        cu_seqlens_kv=cu_kv,
        max_seqlen_q=4,
        max_seqlen_kv=2,
        causal=False,
    )
    kwargs = {
        "num_heads": 3,
        "qk_dim": 192,
        "v_dim": 128,
        "phases": (phase,),
        "scale": 0.125,
    }
    adapter.prepare(**kwargs)
    adapter.prepare(**kwargs)

    adapter._plan_key_from_metadata.assert_called_once()
    adapter._prepare_plan.assert_called_once_with(plan_key)
    adapter._metadata.assert_called_once_with(cu_q, cu_kv, 3, 192, 128)
    binding_key = adapter._binding_key(
        cu_q=cu_q,
        cu_kv=cu_kv,
        dtype=torch.bfloat16,
        heads=3,
        qk_dim=192,
        v_dim=128,
        max_q=4,
        max_kv=2,
        causal=False,
        scale=0.125,
    )
    binding = adapter._require_binding(binding_key)
    assert binding.plan is plan
    assert binding.metadata is metadata
    assert binding.cu_q is cu_q and binding.cu_kv is cu_kv
    assert binding.total_q == 6 and binding.total_kv == 3


def test_cudnn_aligned_staging_and_workspace_reuse(monkeypatch):
    aligned = torch.randn(64, 3, 8)
    assert latent_cp_cudnn_backend._pad_token_rows(aligned, 64) is aligned

    smaller = torch.randn(17, 3, 8)
    padded = latent_cp_cudnn_backend._pad_token_rows(smaller, 64)
    assert padded.shape == (64, 3, 8)
    assert padded.is_contiguous()
    torch.testing.assert_close(padded[:17], smaller)

    class FakeGraph:
        def __init__(self, size):
            self.size = size

        def get_workspace_size(self):
            return self.size

    plan = latent_cp_cudnn_backend._CudnnPlan(
        forward_graph=FakeGraph(13),
        backward_graph=FakeGraph(17),
        key=mock.sentinel.plan_key,
    )
    adapter = object.__new__(latent_cp.CudnnFusedAttentionAdapter)
    adapter.device_index = 0
    adapter._execution_lock = latent_cp_cudnn_backend.threading.RLock()
    adapter._workspaces = {}
    stream = SimpleNamespace(cuda_stream=101)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _device: stream)
    forward = adapter._workspace(plan, backward=False, device=torch.device("cpu"))
    assert (
        adapter._workspace(plan, backward=False, device=torch.device("cpu")) is forward
    )
    backward = adapter._workspace(plan, backward=True, device=torch.device("cpu"))
    assert backward is not forward
    assert forward.numel() == 13 and backward.numel() == 17

    stream.cuda_stream = 202
    other_stream = adapter._workspace(plan, backward=False, device=torch.device("cpu"))
    assert other_stream is not forward


def test_shared_cudnn_adapter_is_process_device_runtime_scoped(monkeypatch):
    created = []

    class FakeAdapter:
        def __init__(self, identity, device_index):
            self.identity = identity
            self.device_index = device_index
            created.append(self)

    monkeypatch.setattr(
        latent_cp_cudnn_backend, "CudnnFusedAttentionAdapter", FakeAdapter
    )
    identity_a = (AttnBackend.fused, "frontend-a", "runtime-a", (9, 0))
    identity_b = (AttnBackend.fused, "frontend-b", "runtime-b", (9, 0))
    with mock.patch.object(latent_cp.torch.cuda, "current_device", return_value=701):
        first = latent_cp_cudnn_backend._shared_cudnn_adapter(identity_a)
        second = latent_cp_cudnn_backend._shared_cudnn_adapter(identity_a)
        third = latent_cp_cudnn_backend._shared_cudnn_adapter(identity_b)
    with mock.patch.object(latent_cp.torch.cuda, "current_device", return_value=702):
        fourth = latent_cp_cudnn_backend._shared_cudnn_adapter(identity_a)
    assert first is second
    assert first is not third and first is not fourth
    assert len(created) == 3


@pytest.mark.parametrize("gradient_accumulation_fusion", [False, True])
def test_explicit_output_projection_uses_only_injected_group_and_matches_reference(
    gradient_accumulation_fusion, monkeypatch
):
    tp_group = object()

    class Projection(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(7, 4, dtype=torch.bfloat16))
            self.register_parameter("bias", None)
            self.input_is_parallel = True
            self.skip_bias_add = True
            self.sequence_parallel = True
            self.explicit_expert_comm = False
            self.gradient_accumulation_fusion = gradient_accumulation_fusion
            self.tp_group = tp_group

        def forward(self, _input):
            raise AssertionError("RowParallelLinear.forward must not be called")

    torch.manual_seed(_SEED)
    projection = Projection()
    harness = SimpleNamespace(
        linear_proj=projection,
        pg_collection=SimpleNamespace(tp=tp_group),
        config=SimpleNamespace(
            cpu_offloading=False, _cpu_offloading_context=None, sequence_parallel=True
        ),
    )
    actual_input = torch.randn(6, 1, 4, dtype=torch.bfloat16, requires_grad=True)
    reference_input = actual_input.detach().clone().requires_grad_(True)
    reference_weight = projection.weight.detach().clone().requires_grad_(True)
    peer_output = torch.randn(6, 1, 7, dtype=torch.bfloat16)
    upstream = torch.randn(3, 1, 7, dtype=torch.bfloat16)
    linear_calls = []
    reduction_calls = []

    def explicit_linear(
        *,
        input,
        weight,
        bias,
        gradient_accumulation_fusion,
        allreduce_dgrad,
        sequence_parallel,
        grad_output_buffer,
        wgrad_deferral_limit,
        tp_group,
    ):
        linear_calls.append(
            {
                "group": tp_group,
                "gradient_accumulation_fusion": gradient_accumulation_fusion,
                "allreduce_dgrad": allreduce_dgrad,
                "sequence_parallel": sequence_parallel,
                "grad_output_buffer": grad_output_buffer,
                "wgrad_deferral_limit": wgrad_deferral_limit,
            }
        )
        assert bias is None
        return F.linear(input, weight)

    def explicit_reduce(input_, group=None):
        reduction_calls.append(group)
        return torch.chunk(input_ + peer_output, 2, dim=0)[0].contiguous()

    state_keys = tuple(projection.state_dict())
    weight_object = projection.weight
    with (
        mock.patch.object(
            latent_cp.mcore_tp,
            "linear_with_grad_accumulation_and_async_allreduce",
            side_effect=explicit_linear,
        ),
        mock.patch.object(
            latent_cp.mcore_tp,
            "reduce_scatter_to_sequence_parallel_region",
            side_effect=explicit_reduce,
        ),
        mock.patch.object(
            latent_cp.mcore_tp,
            "reduce_from_tensor_model_parallel_region",
            side_effect=AssertionError("non-SP TP reduction is forbidden"),
        ),
        mock.patch.object(
            parallel_state,
            "get_tensor_model_parallel_group",
            side_effect=AssertionError("default TP lookup is forbidden"),
        ),
    ):
        actual, bias = latent_cp.MLAWithLatentCP._explicit_output_projection(
            harness, actual_input
        )
        reference = torch.chunk(
            F.linear(reference_input, reference_weight) + peer_output, 2, dim=0
        )[0].contiguous()
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        assert actual.dtype == projection.weight.dtype == torch.bfloat16
        assert bias is None
        actual.backward(upstream)
        reference.backward(upstream)
        projection.weight.requires_grad_(False)
        with pytest.raises(ValueError, match="frozen linear_proj"):
            latent_cp.MLAWithLatentCP._explicit_output_projection(
                harness, actual_input.detach()
            )
        projection.weight.requires_grad_(True)
        harness.config._cpu_offloading_context = object()
        with pytest.raises(ValueError, match="CPU offloading"):
            latent_cp.MLAWithLatentCP._explicit_output_projection(
                harness, actual_input.detach()
            )

    assert linear_calls == [
        {
            "group": tp_group,
            "gradient_accumulation_fusion": gradient_accumulation_fusion,
            "allreduce_dgrad": False,
            "sequence_parallel": False,
            "grad_output_buffer": None,
            "wgrad_deferral_limit": 0,
        }
    ]
    assert reduction_calls == [tp_group]
    torch.testing.assert_close(actual_input.grad, reference_input.grad, rtol=0, atol=0)
    torch.testing.assert_close(
        weight_object.grad, reference_weight.grad, rtol=0, atol=0
    )
    assert projection.weight is weight_object
    assert tuple(projection.state_dict()) == state_keys == ("weight",)


@pytest.mark.parametrize(
    ("attribute", "value", "message"),
    [
        ("multi_latent_attention", False, "multi_latent_attention"),
        ("qk_layernorm", False, "layer norms"),
        ("add_bias_linear", True, "biases"),
        ("rotary_percent", 0.5, "partial rotary"),
        ("num_query_groups", 2, "Hq=Hkv"),
        ("qk_head_dim", 64, "dimensions"),
        ("qk_pos_emb_head_dim", 32, "dimensions"),
        ("v_head_dim", 64, "dimensions"),
        ("rope_type", "none", "rope and yarn"),
        ("apply_rope_fusion", True, "fused RoPE"),
        ("attention_dropout", 0.1, "dropout"),
        ("cache_mla_latents", True, "caching"),
        ("fine_grained_activation_offloading", True, "offload"),
        ("cpu_offloading", True, "CPU offloading"),
        ("attention_backend", AttnBackend.unfused, "attention_backend"),
    ],
)
def test_initial_config_negative_validation(monkeypatch, attribute, value, message):
    config = _make_config()
    object.__setattr__(config, attribute, value)
    dummy = SimpleNamespace(
        config=config,
        pg_collection=SimpleNamespace(tp=object(), cp=object()),
        attn_mask_type=AttnMaskType.causal,
        _cp_comm_type="p2p",
        use_rope=True,
    )
    monkeypatch.setattr(latent_cp.dist, "get_world_size", lambda _group: 1)
    with pytest.raises(ValueError, match=message):
        latent_cp.MLAWithLatentCP._validate_initial_config(dummy)


def test_inactive_default_recompute_modules_are_accepted(monkeypatch):
    config = _make_config()
    object.__setattr__(config, "recompute_granularity", None)
    # TransformerConfig normalizes an unspecified list to this inert default.
    object.__setattr__(config, "recompute_modules", ["core_attn"])
    dummy = SimpleNamespace(
        config=config,
        pg_collection=SimpleNamespace(tp=object(), cp=object()),
        attn_mask_type=AttnMaskType.causal,
        _cp_comm_type="p2p",
        use_rope=True,
    )
    monkeypatch.setattr(latent_cp.dist, "get_world_size", lambda _group: 1)

    latent_cp.MLAWithLatentCP._validate_initial_config(dummy)


def test_tp2_requires_sequence_parallel(monkeypatch):
    config = _make_config(tp_size=2)
    object.__setattr__(config, "sequence_parallel", False)
    tp_group = object()
    cp_group = object()
    dummy = SimpleNamespace(
        config=config,
        pg_collection=SimpleNamespace(tp=tp_group, cp=cp_group),
        attn_mask_type=AttnMaskType.causal,
        _cp_comm_type="p2p",
        use_rope=True,
    )
    monkeypatch.setattr(
        latent_cp.dist, "get_world_size", lambda group: 2 if group is tp_group else 1
    )
    with pytest.raises(ValueError, match="TP>1 requires sequence_parallel=True"):
        latent_cp.MLAWithLatentCP._validate_initial_config(dummy)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda config: (
            object.__setattr__(config, "bf16", False),
            object.__setattr__(config, "fp16", True),
        ),
        lambda config: object.__setattr__(config, "fp8", "e4m3"),
        lambda config: object.__setattr__(config, "fp4", "nvfp4"),
        lambda config: (
            object.__setattr__(config, "recompute_granularity", "selective"),
            object.__setattr__(config, "recompute_modules", ["mla_up_proj"]),
        ),
        lambda config: object.__setattr__(config, "_cpu_offloading_context", object()),
        lambda config: object.__setattr__(config, "cuda_graph_impl", "local"),
    ],
)
def test_precision_recompute_and_graph_negative_validation(monkeypatch, mutator):
    config = _make_config()
    mutator(config)
    dummy = SimpleNamespace(
        config=config,
        pg_collection=SimpleNamespace(tp=object(), cp=object()),
        attn_mask_type=AttnMaskType.causal,
        _cp_comm_type="p2p",
        use_rope=True,
    )
    monkeypatch.setattr(latent_cp.dist, "get_world_size", lambda _group: 1)
    with pytest.raises(ValueError):
        latent_cp.MLAWithLatentCP._validate_initial_config(dummy)


def test_cp_mode_missing_groups_and_mtp_fail_early(monkeypatch):
    config = _make_config(cp_size=2)
    dummy = SimpleNamespace(
        config=config,
        pg_collection=SimpleNamespace(tp=object(), cp=object()),
        attn_mask_type=AttnMaskType.causal,
        _cp_comm_type="a2a",
        use_rope=True,
    )
    monkeypatch.setattr(
        latent_cp.dist,
        "get_world_size",
        lambda group: 2 if group is dummy.pg_collection.cp else 1,
    )
    with pytest.raises(ValueError, match="p2p"):
        latent_cp.MLAWithLatentCP._validate_initial_config(dummy)
    with pytest.raises(ValueError, match="ProcessGroupCollection"):
        latent_cp.MLAWithLatentCP(
            config, _base_mla_spec().submodules, 1, pg_collection=None
        )
    for missing_group in (
        SimpleNamespace(tp=None, cp=object()),
        SimpleNamespace(tp=object(), cp=None),
    ):
        with pytest.raises(ValueError, match="non-null TP and CP"):
            latent_cp.MLAWithLatentCP(
                config, _base_mla_spec().submodules, 1, pg_collection=missing_group
            )
    fake_pg = SimpleNamespace(tp=object(), cp=object())
    with pytest.raises(ValueError, match="MTP"):
        latent_cp.MLAWithLatentCP(
            config, None, 1, pg_collection=fake_pg, is_mtp_layer=True
        )


@pytest.mark.parametrize("cp_size", [1, 2, 4])
def test_planner_rejects_invalid_global_metadata(cp_size: int):
    if cp_size == 1:
        layout = latent_cp.build_zigzag_layout(
            torch.tensor([0, 7], dtype=torch.int32), 7, cp_size, 0
        )
        assert layout.local_tokens == 7
    else:
        with pytest.raises(ValueError, match="divisible"):
            latent_cp.build_zigzag_layout(
                torch.tensor([0, 7], dtype=torch.int32), 7, cp_size, 0
            )
    with pytest.raises(ValueError, match="empty"):
        latent_cp.build_zigzag_layout(
            torch.tensor([0, 0], dtype=torch.int32), 0, cp_size, 0
        )
    valid_length = 2 * cp_size
    with pytest.raises(ValueError, match="hidden token count"):
        latent_cp.build_zigzag_layout(
            torch.tensor([0, valid_length], dtype=torch.int32), 999, cp_size, 0
        )


def test_layout_adapter_rejects_invalid_format_and_layout(monkeypatch):
    adapter = latent_cp.AlreadyZigZagTHDAdapter()
    hidden = mock.Mock()
    cp_group = object()
    monkeypatch.setattr(latent_cp.dist, "get_world_size", lambda _group: 1)
    monkeypatch.setattr(latent_cp.dist, "get_rank", lambda _group: 0)
    base = SimpleNamespace(qkv_format="thd", cp_partition_mode="zigzag")
    for field, value, message in (
        ("qkv_format", "sbhd", "THD"),
        ("cp_partition_mode", "contiguous", "zigzag"),
    ):
        metadata = SimpleNamespace(**vars(base))
        setattr(metadata, field, value)
        with pytest.raises(ValueError, match=message):
            adapter.prepare(hidden, metadata, cp_group)


def test_block_preprocess_dispatches_only_to_latent_cp_layers():
    hidden = torch.empty(1, 1, 1)
    packed = PackedSeqParams()
    latent_cp.preprocess_mla_latent_cp(nn.Sequential(nn.Identity()), hidden, None)

    first = object.__new__(latent_cp.MLAWithLatentCP)
    second = object.__new__(latent_cp.MLAWithLatentCP)
    nn.Module.__init__(first)
    nn.Module.__init__(second)
    prepared = (mock.sentinel.cp_group, mock.sentinel.layout)
    first._microbatch_layout = mock.Mock(return_value=prepared)
    first._preprocess_backend = mock.Mock()
    second._preprocess_backend = mock.Mock()
    latent_cp.preprocess_mla_latent_cp(
        nn.Sequential(first, nn.Identity(), second), hidden, packed
    )
    first._microbatch_layout.assert_called_once_with(hidden, packed)
    first._preprocess_backend.assert_called_once_with(
        hidden, packed, prepared_layout=prepared
    )
    second._preprocess_backend.assert_called_once_with(
        hidden, packed, prepared_layout=prepared
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_metadata_and_forward_negative_validation():
    with _model_parallel(1, 1) as pg:
        config = _make_config()
        layer = (
            _build_layer(config, pg, _TorchPackedAttentionAdapter())
            .cuda()
            .bfloat16()
            .train()
        )
        hidden = torch.randn(
            8, 1, config.hidden_size, dtype=torch.bfloat16, device="cuda"
        )
        good = _make_packed((8,), device="cuda", cp_group=pg.cp)
        adapter = latent_cp.AlreadyZigZagTHDAdapter()
        assert adapter.prepare(hidden, good, pg.cp).local_tokens == 8

        mismatched = replace(
            good,
            cu_seqlens_kv=torch.tensor([0, 4, 8], dtype=torch.int32, device="cuda"),
        )
        padded = replace(
            good,
            cu_seqlens_q_padded=torch.tensor([0, 7], dtype=torch.int32, device="cuda"),
        )
        noncontiguous_base = torch.tensor(
            [0, 99, 8, 99], dtype=torch.int32, device="cuda"
        )
        noncontiguous = replace(
            good,
            cu_seqlens_q=noncontiguous_base[::2],
            cu_seqlens_kv=noncontiguous_base[::2],
        )
        cpu_metadata = replace(
            good,
            cu_seqlens_q=torch.tensor([0, 8], dtype=torch.int32),
            cu_seqlens_kv=torch.tensor([0, 8], dtype=torch.int32),
        )
        invalid_max = replace(good, max_seqlen_q=0, max_seqlen_kv=0)
        for metadata, message in (
            (mismatched, "equal Q/KV"),
            (padded, "padding"),
            (noncontiguous, "contiguous"),
            (cpu_metadata, "CUDA"),
            (invalid_max, "positive Python"),
        ):
            with pytest.raises(ValueError, match=message):
                adapter.prepare(hidden, metadata, pg.cp)

        with pytest.raises(ValueError, match="explicit attention masks"):
            layer(
                hidden,
                torch.ones(1, 1, 8, 8, dtype=torch.bool, device="cuda"),
                packed_seq_params=good,
            )
        with pytest.raises(ValueError, match="cross attention"):
            layer(hidden, None, key_value_states=hidden, packed_seq_params=good)
        with pytest.raises(ValueError, match="inference"):
            layer(hidden, None, inference_context=object(), packed_seq_params=good)
        with pytest.raises(ValueError, match="BF16"):
            layer(hidden.float(), None, packed_seq_params=good)
        with pytest.raises(ValueError, match="singleton batch"):
            layer(hidden.expand(-1, 2, -1), None, packed_seq_params=good)
        with pytest.raises(ValueError, match="external position"):
            layer(
                hidden,
                None,
                position_ids=torch.arange(8, device="cuda"),
                packed_seq_params=good,
            )
        layer.eval()
        with pytest.raises(ValueError, match="training-only"):
            layer(hidden, None, packed_seq_params=good)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("cp_size", [2, 4])
def test_ring_forward_reverse_backward_payload_bytes_and_explicit_group(
    cp_size, monkeypatch
):
    with _model_parallel(1, cp_size) as pg:
        cp_rank = dist.get_rank(pg.cp)
        lengths = (8 * cp_size,)
        layout = latent_cp.build_zigzag_layout(
            _cumulative(lengths, "cuda"), 8, cp_size, cp_rank
        )
        payload = torch.full(
            (8, _PRODUCTION_KV_LORA + _ROPE_DIM),
            float(cp_rank),
            dtype=torch.bfloat16,
            device="cuda",
            requires_grad=True,
        )
        real_p2p_op = latent_cp.dist.P2POp
        real_batch = latent_cp.dist.batch_isend_irecv
        p2p_records = []
        batch_calls = []
        returned_proxies = []
        wait_count = 0

        def p2p_op(op, tensor, peer, group=None, tag=0):
            p2p_records.append((op, tensor.numel(), tensor.element_size(), peer, group))
            return real_p2p_op(op, tensor, peer, group=group, tag=tag)

        class WorkProxy:
            def __init__(self, work):
                self.work = work
                self.waited = False

            def wait(self):
                nonlocal wait_count
                assert not self.waited
                self.waited = True
                wait_count += 1
                return self.work.wait()

        def batch(operations):
            assert all(proxy.waited for proxy in returned_proxies)
            batch_calls.append(tuple(operations))
            proxies = [WorkProxy(work) for work in real_batch(operations)]
            returned_proxies.extend(proxies)
            return proxies

        monkeypatch.setattr(latent_cp.dist, "P2POp", p2p_op)
        monkeypatch.setattr(latent_cp.dist, "batch_isend_irecv", batch)
        leases = list(
            latent_cp.P2PRingTransport(pg.cp).iter_payloads(payload, layout.phases)
        )
        assert [lease.owner for lease in leases] == [
            (cp_rank - phase) % cp_size for phase in range(cp_size)
        ]
        for lease in leases:
            assert torch.equal(
                lease.tensor, torch.full_like(lease.tensor, float(lease.owner))
            )
        weights = [float(cp_rank * cp_size + phase + 1) for phase in range(cp_size)]
        loss = sum(
            weight * lease.tensor.float().sum()
            for weight, lease in zip(weights, leases)
        )
        loss.backward()
        expected = sum(
            float(query_rank * cp_size + ((query_rank - cp_rank) % cp_size) + 1)
            for query_rank in range(cp_size)
        )
        torch.testing.assert_close(
            payload.grad, torch.full_like(payload.grad, expected)
        )
        latent_elements = 8 * (_PRODUCTION_KV_LORA + _ROPE_DIM)
        full_elements = 8 * _PRODUCTION_HEADS * (192 + _VALUE_DIM)
        assert latent_elements < full_elements
        assert p2p_records
        assert all(record[1] == latent_elements for record in p2p_records)
        assert all(record[4] is pg.cp for record in p2p_records)
        assert len(batch_calls) == 2 * (cp_size - 1)
        assert len(p2p_records) == 2 * len(batch_calls)
        assert returned_proxies and all(proxy.waited for proxy in returned_proxies)
        assert wait_count == len(returned_proxies)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("rope_type", ["rope", "yarn"])
def test_finalized_metadata_projection_groups_rope_and_checkpoint_lifetime(
    rope_type, monkeypatch
):
    with _model_parallel(1, 2) as pg:
        config = _make_config(cp_size=2, rope_type=rope_type)
        backend = _TorchPackedAttentionAdapter()
        layer = _build_layer(config, pg, backend).cuda().bfloat16().train()
        lengths = (16, 8)
        local_tokens = sum(lengths) // 2
        packed = _make_packed(lengths, device="cuda", total_tokens=local_tokens)
        packed = finalize_packed_seq_params(packed)
        assert packed.cp_group is pg.cp
        hidden = torch.randn(
            local_tokens,
            1,
            config.hidden_size,
            dtype=torch.bfloat16,
            device="cuda",
            requires_grad=True,
        )
        layout = layer._layout_adapter.prepare(hidden, packed, pg.cp)
        assert layout.cu_global is packed.cu_seqlens_q

        rope_calls = []
        real_rope = latent_cp.apply_rotary_pos_emb

        def rope_spy(*args, **kwargs):
            rope_calls.append(kwargs)
            return real_rope(*args, **kwargs)

        real_gather = latent_cp.tp_mappings.gather_from_tensor_model_parallel_region
        with (
            mock.patch.object(
                latent_cp.tp_mappings,
                "gather_from_tensor_model_parallel_region",
                wraps=real_gather,
            ) as gather_spy,
            mock.patch.object(
                parallel_state,
                "get_tensor_model_parallel_group",
                side_effect=AssertionError("default TP group lookup is forbidden"),
            ),
            mock.patch.object(
                latent_cp_module, "apply_rotary_pos_emb", side_effect=rope_spy
            ),
        ):
            query, payload = layer._project_query_and_payload(hidden, packed, layout)
        assert query.shape == (local_tokens, config.num_attention_heads, 192)
        assert payload.shape == (local_tokens, config.kv_lora_rank + _ROPE_DIM)
        assert (
            len(gather_spy.call_args_list) == 0
        )  # TP1 projections are already complete.
        assert len(rope_calls) == 2
        assert all(call["cu_seqlens"] is layout.cu_global for call in rope_calls)
        assert all(call["max_seqlen"] == layout.max_global for call in rope_calls)
        assert all(call["cp_group"] is pg.cp for call in rope_calls)

        up_projection_calls = 0

        def count_up_projection(_module, _inputs, _output):
            nonlocal up_projection_calls
            up_projection_calls += 1

        expected_query_shapes = [
            (
                phase.q_indices.numel(),
                layer.num_attention_heads_per_partition,
                _QK_CONTENT + _ROPE_DIM,
            )
            for phase in layout.phases
        ]
        # Checkpoint receives the full ring lease; phase KV slicing happens after up-projection.
        expected_latent_shapes = [
            (layout.local_tokens, config.kv_lora_rank + _ROPE_DIM)
            for _phase in layout.phases
        ]
        retained = _SavedTensorRecorder()
        handle = layer.linear_kv_up_proj.register_forward_hook(count_up_projection)
        try:
            with torch.autograd.graph.saved_tensors_hooks(
                retained.pack, retained.unpack
            ):
                output, bias = layer(hidden, None, packed_seq_params=packed)
            with _rank_common_assertions():
                retained_state = _classify_saved_attention_state(
                    retained.records,
                    expected_query_shapes=expected_query_shapes,
                    expected_latent_shapes=expected_latent_shapes,
                    heads=layer.num_attention_heads_per_partition,
                )
                assert all(
                    record.numel == math.prod(record.shape) for record in retained_state
                )
                assert all(
                    record.tensor_class.endswith((".Tensor", ".Parameter"))
                    for record in retained_state
                )
                retained_classes = [record.state_class for record in retained_state]
                assert retained_classes.count("checkpoint_query_input") == len(
                    layout.phases
                )
                assert retained_classes.count("checkpoint_latent_input") == len(
                    layout.phases
                )
                assert "partial_output_or_merge_state" in retained_classes
                assert "partial_lse_or_merge_state" in retained_classes
                assert "expanded_value" not in retained_classes
                assert "expanded_key_or_uncheckpointed_query" not in retained_classes
                assert bias is None
                assert output.dtype == torch.bfloat16
                assert backend.forward_calls == 2
                assert backend.raw_output_dtypes == [torch.bfloat16, torch.bfloat16]
                assert up_projection_calls == 2
                torch.cuda.synchronize()
                gc.collect()
                assert all(reference() is None for reference in backend.expanded_refs)
                assert any(
                    reference() is not None for reference in backend.partial_refs
                )
                assert any(
                    reference() is not None for reference in backend.partial_lse_refs
                )
            output.backward(torch.randn_like(output))
            assert up_projection_calls == 4
            assert backend.forward_calls == 4
        finally:
            handle.remove()

        def run_without_checkpoint(function, *args, **kwargs):
            assert kwargs.pop("use_reentrant") is False
            assert kwargs.pop("preserve_rng_state") is False
            assert not kwargs
            return function(*args)

        direct_backend = _TorchPackedAttentionAdapter()
        layer._backend_adapter = direct_backend
        uncheckpointed = _SavedTensorRecorder()
        direct_hidden = hidden.detach().clone().requires_grad_(True)
        with (
            mock.patch.object(
                latent_cp_module, "checkpoint", side_effect=run_without_checkpoint
            ),
            torch.autograd.graph.saved_tensors_hooks(
                uncheckpointed.pack, uncheckpointed.unpack
            ),
        ):
            direct_output, direct_bias = layer(
                direct_hidden, None, packed_seq_params=packed
            )
        assert direct_bias is None
        uncheckpointed_state = _classify_saved_attention_state(
            uncheckpointed.records,
            expected_query_shapes=[],
            expected_latent_shapes=[],
            heads=layer.num_attention_heads_per_partition,
        )
        uncheckpointed_classes = {record.state_class for record in uncheckpointed_state}
        assert "expanded_value" in uncheckpointed_classes
        assert "expanded_key_or_uncheckpointed_query" in uncheckpointed_classes
        del direct_output, direct_hidden


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_kimi_no_rope_skips_rotary_construction_and_application():
    with _model_parallel(1, 1) as pg:
        config = _make_config(cp_size=1, rope_type="yarn", no_rope=True)
        with (
            mock.patch.object(
                base_mla.RotaryEmbedding,
                "__init__",
                side_effect=AssertionError(
                    "no-RoPE must not initialize RotaryEmbedding"
                ),
            ) as rotary_init,
            mock.patch.object(
                base_mla.YarnRotaryEmbedding,
                "__init__",
                side_effect=AssertionError(
                    "no-RoPE must not initialize YarnRotaryEmbedding"
                ),
            ) as yarn_init,
        ):
            layer = _build_layer(config, pg).cuda().bfloat16().train()
        rotary_init.assert_not_called()
        yarn_init.assert_not_called()
        assert layer.rotary_pos_emb is None
        assert not layer.use_rope
        assert layer.softmax_scale == pytest.approx(
            1.0 / math.sqrt(_QK_CONTENT + _ROPE_DIM)
        )

        lengths = (7, 5)
        hidden = torch.randn(
            sum(lengths), 1, config.hidden_size, dtype=torch.bfloat16, device="cuda"
        )
        packed = _make_packed(lengths, device="cuda", cp_group=pg.cp)
        layout = layer._layout_adapter.prepare(hidden, packed, pg.cp)
        with mock.patch.object(
            latent_cp_module,
            "apply_rotary_pos_emb",
            side_effect=AssertionError("no-RoPE must not apply rotary embeddings"),
        ) as apply_rope:
            query, payload = layer._project_query_and_payload(hidden, packed, layout)
        apply_rope.assert_not_called()
        assert query.shape == (sum(lengths), config.num_attention_heads, 192)
        assert payload.shape == (sum(lengths), config.kv_lora_rank + _ROPE_DIM)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_tp2_sequence_parallel_payload_and_explicit_mappings():
    with _model_parallel(2, 2) as pg:
        config = _make_config(tp_size=2, cp_size=2)
        assert config.sequence_parallel
        layer = _build_layer(config, pg).cuda().bfloat16().train()
        assert config.sequence_parallel
        for norm in (layer.q_layernorm, layer.kv_layernorm):
            assert all(
                getattr(parameter, "sequence_parallel", False)
                for parameter in norm.parameters()
            )
        lengths = (16, 8)
        local_tokens = sum(lengths) // 2
        packed = _make_packed(lengths, device="cuda", cp_group=pg.cp)
        full_local_hidden = torch.randn(
            local_tokens, 1, config.hidden_size, dtype=torch.bfloat16, device="cuda"
        )
        tp_source = dist.get_process_group_ranks(pg.tp)[0]
        dist.broadcast(full_local_hidden, src=tp_source, group=pg.tp)
        hidden = _sequence_parallel_slice(full_local_hidden, pg.tp)
        layout = layer._layout_adapter.prepare(
            hidden, packed, pg.cp, tp_group=pg.tp, sequence_parallel=True
        )
        assert layout.local_tokens == local_tokens

        real_last_dim_gather = (
            latent_cp.tp_mappings.gather_from_tensor_model_parallel_region
        )
        real_sequence_scatter = (
            latent_cp.tp_mappings.scatter_to_sequence_parallel_region
        )
        sequence_scatter_shapes: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        latent_outputs: list[torch.Tensor] = []

        def explicit_sequence_scatter(input_: torch.Tensor, group=None) -> torch.Tensor:
            assert group is pg.tp
            output = real_sequence_scatter(input_, group=group)
            sequence_scatter_shapes.append((tuple(input_.shape), tuple(output.shape)))
            return output

        def capture_latent(_module, _inputs, output):
            latent_outputs.append(output)

        norm_handle = layer.kv_layernorm.register_forward_hook(capture_latent)
        try:
            with (
                mock.patch.object(
                    latent_cp.tp_mappings,
                    "gather_from_tensor_model_parallel_region",
                    wraps=real_last_dim_gather,
                ) as last_dim_gather_spy,
                mock.patch.object(
                    latent_cp.tp_mappings,
                    "scatter_to_sequence_parallel_region",
                    side_effect=explicit_sequence_scatter,
                ) as sequence_scatter_spy,
                mock.patch.object(
                    latent_cp.tp_mappings,
                    "copy_to_tensor_model_parallel_region",
                    side_effect=AssertionError("non-SP TP copy is forbidden"),
                ),
                mock.patch.object(
                    parallel_state,
                    "get_tensor_model_parallel_group",
                    side_effect=AssertionError("default TP group lookup is forbidden"),
                ),
            ):
                query, payload = layer._project_query_and_payload(
                    hidden, packed, layout
                )
        finally:
            norm_handle.remove()

        assert len(last_dim_gather_spy.call_args_list) == 2
        assert all(
            call.kwargs["group"] is pg.tp for call in last_dim_gather_spy.call_args_list
        )
        assert sequence_scatter_spy.call_count == 3
        assert len(sequence_scatter_shapes) == 3
        assert all(before[0] == local_tokens for before, _ in sequence_scatter_shapes)
        assert all(
            after[0] == local_tokens // 2 for _, after in sequence_scatter_shapes
        )
        assert len(latent_outputs) == 1
        latent = latent_outputs[0]
        assert query.shape == (
            local_tokens,
            config.num_attention_heads // 2,
            _QK_CONTENT + _ROPE_DIM,
        )
        assert payload.shape == (local_tokens // 2, config.kv_lora_rank + _ROPE_DIM)
        assert latent.shape == (local_tokens // 2, config.kv_lora_rank)
        assert torch.equal(payload[:, : config.kv_lora_rank], latent)
        payloads = [torch.empty_like(payload) for _ in range(2)]
        dist.all_gather(payloads, payload, group=pg.tp)
        reconstructed_payload = torch.cat(payloads, dim=0)
        assert reconstructed_payload.shape == (
            local_tokens,
            config.kv_lora_rank + _ROPE_DIM,
        )

        phase = layout.phases[0]
        q_phase = query.index_select(0, phase.q_indices)
        payload.retain_grad()
        real_sequence_gather = (
            latent_cp.tp_mappings.gather_from_sequence_parallel_region
        )
        phase_latent_inputs: list[tuple[tuple[int, ...], bool]] = []

        def capture_phase_latent(_module, inputs):
            phase_latent_inputs.append(
                (tuple(inputs[0].shape), inputs[0].is_contiguous())
            )

        phase_latent_handle = layer.linear_kv_up_proj.register_forward_pre_hook(
            capture_phase_latent
        )
        with (
            mock.patch.object(
                latent_cp.tp_mappings,
                "gather_from_sequence_parallel_region",
                wraps=real_sequence_gather,
            ) as sequence_gather_spy,
            mock.patch.object(
                parallel_state,
                "get_tensor_model_parallel_group",
                side_effect=AssertionError("default TP group lookup is forbidden"),
            ),
        ):
            try:
                output, lse = layer._phase_attention(
                    q_phase, payload, phase, _TorchPackedAttentionAdapter()
                )
                (output.square().mean() + lse.square().mean()).backward()
            finally:
                phase_latent_handle.remove()
        assert phase_latent_inputs == [((payload.size(0), config.kv_lora_rank), True)]
        sequence_gather_spy.assert_called_once()
        assert sequence_gather_spy.call_args.args[0].is_contiguous()
        assert sequence_gather_spy.call_args.kwargs == {
            "tensor_parallel_output_grad": True,
            "group": pg.tp,
        }
        assert payload.grad is not None
        assert torch.isfinite(payload.grad).all()

        for name in (
            "linear_q_down_proj",
            "linear_q_up_proj",
            "linear_kv_down_proj",
            "linear_kv_up_proj",
            "linear_proj",
        ):
            assert getattr(layer, name).tp_group is pg.tp


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize(
    ("local_cp_size", "lengths"),
    [(1, (7, 5)), (2, (16, 8))],
    ids=("dynamic-cp1-odd-packed", "dynamic-cp2-zigzag"),
)
def test_dynamic_cp1_cp2_full_chain_parity_without_static_group_mutation(
    local_cp_size: int, lengths: tuple[int, ...]
):
    with _model_parallel(2, 2, dynamic_cp=True) as pg:
        torch.manual_seed(_SEED + 40 + local_cp_size)
        torch.cuda.manual_seed_all(_SEED + 40 + local_cp_size)
        model_parallel_cuda_manual_seed(_SEED + 40 + local_cp_size)
        config = _make_config(tp_size=2, cp_size=2, dynamic_cp=True)
        phase_backend = _TorchPackedAttentionAdapter()
        layer = _build_layer(config, pg, phase_backend).cuda().bfloat16().train()
        static_cp_group = layer.pg_collection.cp
        effective_cp_group = parallel_state.get_dynamic_data_context_parallel_groups(
            group_size=local_cp_size
        )
        assert dist.get_world_size(effective_cp_group) == local_cp_size

        reference = NaiveMLA(config, torch.device("cuda")).train()
        _copy_reference_parameters(reference, layer, pg)
        total_tokens = sum(lengths)
        full_hidden = torch.randn(
            total_tokens, 1, config.hidden_size, dtype=torch.bfloat16, device="cuda"
        )
        full_upstream = torch.randn_like(full_hidden)
        tp_cp_source = dist.get_process_group_ranks(pg.tp_cp)[0]
        dist.broadcast(full_hidden, src=tp_cp_source, group=pg.tp_cp)
        dist.broadcast(full_upstream, src=tp_cp_source, group=pg.tp_cp)

        effective_cp_rank = dist.get_rank(effective_cp_group)
        local_indices = _zigzag_global_indices(
            lengths, local_cp_size, effective_cp_rank, device="cuda"
        )
        cp_local_hidden = full_hidden.index_select(0, local_indices)
        local_hidden = (
            _sequence_parallel_slice(cp_local_hidden, pg.tp)
            .detach()
            .clone()
            .requires_grad_(True)
        )
        reference_hidden = full_hidden.detach().clone().requires_grad_(True)
        packed = _make_packed(
            lengths,
            device="cuda",
            cp_group=effective_cp_group,
            local_cp_size=local_cp_size,
        )

        latent_cp.preprocess_mla_latent_cp(layer, local_hidden, packed)
        with _forbid_default_process_group_resolvers():
            real_output, bias = layer(local_hidden, None, packed_seq_params=packed)
        assert bias is None
        assert layer.pg_collection.cp is static_cp_group
        assert phase_backend.forward_calls == local_cp_size

        reference_output = reference(reference_hidden, lengths)
        expected_output = _sequence_parallel_slice(
            reference_output.index_select(0, local_indices), pg.tp
        )
        _assert_similarity(
            real_output.detach(), expected_output.detach(), "dynamic CP output"
        )

        local_upstream = _sequence_parallel_slice(
            full_upstream.index_select(0, local_indices), pg.tp
        )
        with _forbid_default_process_group_resolvers():
            real_output.backward(local_upstream)
        reference_output.backward(full_upstream)
        expected_hidden_grad = _sequence_parallel_slice(
            reference_hidden.grad.index_select(0, local_indices), pg.tp
        )
        _assert_similarity(
            local_hidden.grad, expected_hidden_grad, "dynamic CP input gradient"
        )
        assert phase_backend.forward_calls == 2 * local_cp_size

        reconstructed = _reconstruct_real_parameter_gradients(
            layer, pg, cp_group=effective_cp_group
        )
        reference_params = dict(reference.named_parameters())
        assert (
            set(reconstructed) == set(reference_params) == set(_parameter_map(config))
        )
        for name in sorted(reference_params):
            assert reference_params[name].grad is not None
            _assert_similarity(
                reconstructed[name],
                reference_params[name].grad.detach().float(),
                f"dynamic CP parameter gradient {name}",
            )
        assert layer.pg_collection.cp is static_cp_group


@contextmanager
def _forbid_default_process_group_resolvers():
    def fail_default_lookup(*_args, **_kwargs):
        raise AssertionError("default process-group lookup is forbidden")

    with (
        mock.patch.object(
            parallel_state,
            "get_tensor_model_parallel_group",
            side_effect=fail_default_lookup,
        ),
        mock.patch.object(
            parallel_state,
            "get_context_parallel_group",
            side_effect=fail_default_lookup,
        ),
        mock.patch.object(
            parallel_state,
            "get_tensor_and_context_parallel_group",
            side_effect=fail_default_lookup,
        ),
    ):
        yield


def _aggregate_and_emit_metrics(
    *,
    event: str,
    metadata: dict[str, object],
    local_metrics: dict[str, dict[str, float]],
    pg: ProcessGroupCollection,
    device: torch.device,
) -> tuple[dict[str, dict[str, float]], float]:
    metrics: dict[str, dict[str, float]] = {}
    for label in sorted(local_metrics):
        local = local_metrics[label]
        minima = torch.tensor(
            [local["cosine"], local["tensor_similarity"]],
            dtype=torch.float64,
            device=device,
        )
        if dist.get_world_size(pg.tp_cp) > 1:
            dist.all_reduce(minima, op=dist.ReduceOp.MIN, group=pg.tp_cp)
        cosine, tensor_similarity = (float(value) for value in minima.tolist())
        metrics[label] = {
            "cosine": cosine,
            "tensor_similarity": tensor_similarity,
            "observed_error": max(0.0, 1.0 - cosine, 1.0 - tensor_similarity),
        }

    max_observed_error = max(metric["observed_error"] for metric in metrics.values())
    candidate_eps = max(1e-5, 2.0 * max_observed_error)
    group_rank = dist.get_rank(pg.tp_cp)
    group_ranks = dist.get_process_group_ranks(pg.tp_cp)
    if group_ranks[group_rank] == 0:
        evidence = {
            "candidate_eps": candidate_eps,
            "event": event,
            "max_observed_error": max_observed_error,
            "metrics": metrics,
            **metadata,
        }
        print(json.dumps(evidence, sort_keys=True, separators=(",", ":")), flush=True)
    return metrics, candidate_eps


def _assert_emitted_metrics(
    metrics: dict[str, dict[str, float]],
    candidate_eps: float,
    *,
    eps: float = _PARITY_EPS_CEILING,
) -> None:
    assert 0 < eps <= _PARITY_EPS_CEILING
    for label in sorted(metrics):
        _assert_similarity_metrics(metrics[label], label, eps)
    assert candidate_eps <= eps, (
        f"candidate eps {candidate_eps:.10g} exceeds qualified eps {eps:.10g}"
    )


def _cudnn_phase_diagnostic_setup() -> tuple[
    latent_cp.QualifiedBackendTuple,
    latent_cp.CudnnFusedAttentionAdapter,
    latent_cp.ZigZagLayout,
    float,
]:
    runtime = _qualified_real_backend_runtime_or_skip(AttnBackend.fused)
    adapter, adapter_runtime = latent_cp._qualified_backend_adapter(
        AttnBackend.fused, runtime
    )
    assert adapter_runtime == runtime
    assert isinstance(adapter, latent_cp.CudnnFusedAttentionAdapter)
    layout = latent_cp.build_zigzag_layout(
        _cumulative((32,), "cuda"), local_tokens=16, cp_size=2, cp_rank=1
    )
    scale = 1.0 / math.sqrt(_QK_CONTENT + _ROPE_DIM)
    adapter.prepare(
        num_heads=_PRODUCTION_HEADS // 2,
        qk_dim=_QK_CONTENT + _ROPE_DIM,
        v_dim=_VALUE_DIM,
        phases=layout.phases,
        scale=scale,
    )
    return runtime, adapter, layout, scale


def _run_cudnn_do_only_phase_diagnostic() -> None:
    with _model_parallel(1, 1) as pg:
        torch.manual_seed(_SEED + 20)
        torch.cuda.manual_seed_all(_SEED + 20)
        runtime, adapter, layout, scale = _cudnn_phase_diagnostic_setup()
        phase = layout.phases[0]
        heads = _PRODUCTION_HEADS // 2
        q_tokens = int(phase.cu_seqlens_q[-1].item())
        kv_tokens = int(phase.cu_seqlens_kv[-1].item())
        q = torch.randn(
            q_tokens,
            heads,
            _QK_CONTENT + _ROPE_DIM,
            dtype=torch.bfloat16,
            device="cuda",
            requires_grad=True,
        )
        k = torch.randn_like(q, requires_grad=True)
        assert k.size(0) == kv_tokens
        v = torch.randn(
            kv_tokens,
            heads,
            _VALUE_DIM,
            dtype=torch.bfloat16,
            device="cuda",
            requires_grad=True,
        )
        q_ref = q.detach().clone().requires_grad_(True)
        k_ref = k.detach().clone().requires_grad_(True)
        v_ref = v.detach().clone().requires_grad_(True)

        with _forbid_default_process_group_resolvers():
            output, lse = adapter.forward_phase(
                q,
                k,
                v,
                phase.cu_seqlens_q,
                phase.cu_seqlens_kv,
                phase.max_seqlen_q,
                phase.max_seqlen_kv,
                phase.causal,
                scale,
            )
        reference_output, reference_lse = _independent_torch_phase_attention(
            q_ref,
            k_ref,
            v_ref,
            phase.cu_seqlens_q,
            phase.cu_seqlens_kv,
            causal=phase.causal,
            scale=scale,
        )
        upstream = torch.randn_like(output).to(torch.bfloat16).float()
        with (
            _forbid_default_process_group_resolvers(),
            mock.patch.object(
                latent_cp_cudnn_backend,
                "cudnn_backward_proxy",
                wraps=latent_cp_cudnn_backend.cudnn_backward_proxy,
            ) as proxy_spy,
        ):
            output.backward(upstream)
        proxy_spy.assert_called_once()
        assert torch.count_nonzero(proxy_spy.call_args.args[2]) == 0
        reference_output.backward(upstream)
        local_metrics = {
            "dk": _measure_similarity(k.grad, k_ref.grad, "dO-only dK"),
            "dq": _measure_similarity(q.grad, q_ref.grad, "dO-only dQ"),
            "dv": _measure_similarity(v.grad, v_ref.grad, "dO-only dV"),
            "lse": _measure_similarity(lse.detach(), reference_lse.detach(), "LSE"),
            "output": _measure_similarity(
                output.detach(), reference_output.detach(), "output"
            ),
        }
        metrics, candidate_eps = _aggregate_and_emit_metrics(
            event="mla_latent_cp_cudnn_do_only_diagnostic",
            metadata={
                "phase_kind": phase.kind,
                "runtime_tuple": [
                    runtime[0].name,
                    runtime[1],
                    runtime[2],
                    list(runtime[3]),
                ],
                "seed": _SEED + 20,
                "shape": [q_tokens, kv_tokens, heads, 192, _VALUE_DIM],
            },
            local_metrics=local_metrics,
            pg=pg,
            device=q.device,
        )
        _assert_emitted_metrics(
            metrics, candidate_eps, eps=EXPECTED_QUALIFICATION_EPS[runtime]
        )


def _run_cudnn_two_phase_merge_diagnostic() -> None:
    with _model_parallel(1, 1) as pg:
        torch.manual_seed(_SEED + 21)
        torch.cuda.manual_seed_all(_SEED + 21)
        runtime, adapter, layout, scale = _cudnn_phase_diagnostic_setup()
        diagonal, lower = layout.phases
        assert diagonal.kind == "diagonal" and lower.kind == "lower"
        heads = _PRODUCTION_HEADS // 2
        q_tokens = int(diagonal.cu_seqlens_q[-1].item())
        diagonal_kv_tokens = int(diagonal.cu_seqlens_kv[-1].item())
        lower_kv_tokens = int(lower.cu_seqlens_kv[-1].item())
        q_diagonal = torch.randn(
            q_tokens,
            heads,
            _QK_CONTENT + _ROPE_DIM,
            dtype=torch.bfloat16,
            device="cuda",
            requires_grad=True,
        )
        q_lower = q_diagonal.detach().clone().requires_grad_(True)
        k_diagonal = torch.randn_like(q_diagonal, requires_grad=True)
        assert k_diagonal.size(0) == diagonal_kv_tokens
        v_diagonal = torch.randn(
            diagonal_kv_tokens,
            heads,
            _VALUE_DIM,
            dtype=torch.bfloat16,
            device="cuda",
            requires_grad=True,
        )
        k_lower = torch.randn(
            lower_kv_tokens,
            heads,
            _QK_CONTENT + _ROPE_DIM,
            dtype=torch.bfloat16,
            device="cuda",
            requires_grad=True,
        )
        v_lower = torch.randn(
            lower_kv_tokens,
            heads,
            _VALUE_DIM,
            dtype=torch.bfloat16,
            device="cuda",
            requires_grad=True,
        )
        actual_inputs = (q_diagonal, k_diagonal, v_diagonal, q_lower, k_lower, v_lower)
        reference_inputs = tuple(
            tensor.detach().clone().requires_grad_(True) for tensor in actual_inputs
        )
        (
            q_diagonal_ref,
            k_diagonal_ref,
            v_diagonal_ref,
            q_lower_ref,
            k_lower_ref,
            v_lower_ref,
        ) = reference_inputs

        with _forbid_default_process_group_resolvers():
            diagonal_output, diagonal_lse = adapter.forward_phase(
                q_diagonal,
                k_diagonal,
                v_diagonal,
                diagonal.cu_seqlens_q,
                diagonal.cu_seqlens_kv,
                diagonal.max_seqlen_q,
                diagonal.max_seqlen_kv,
                diagonal.causal,
                scale,
            )
            lower_output, lower_lse = adapter.forward_phase(
                q_lower,
                k_lower,
                v_lower,
                lower.cu_seqlens_q,
                lower.cu_seqlens_kv,
                lower.max_seqlen_q,
                lower.max_seqlen_kv,
                lower.causal,
                scale,
            )
        diagonal_reference, diagonal_lse_reference = _independent_torch_phase_attention(
            q_diagonal_ref,
            k_diagonal_ref,
            v_diagonal_ref,
            diagonal.cu_seqlens_q,
            diagonal.cu_seqlens_kv,
            causal=diagonal.causal,
            scale=scale,
        )
        lower_reference, lower_lse_reference = _independent_torch_phase_attention(
            q_lower_ref,
            k_lower_ref,
            v_lower_ref,
            lower.cu_seqlens_q,
            lower.cu_seqlens_kv,
            causal=lower.causal,
            scale=scale,
        )
        merged_output, merged_lse = latent_cp.merge_attention_partials(
            diagonal_output, diagonal_lse, lower_output, lower_lse
        )
        reference_merged_lse = torch.logaddexp(
            diagonal_lse_reference, lower_lse_reference
        )
        diagonal_weight = torch.exp(diagonal_lse_reference - reference_merged_lse)
        lower_weight = torch.exp(lower_lse_reference - reference_merged_lse)
        reference_merged_output = diagonal_reference * diagonal_weight.unsqueeze(
            -1
        ) + lower_reference * lower_weight.unsqueeze(-1)
        upstream = torch.randn_like(merged_output).to(torch.bfloat16).float()
        with (
            _forbid_default_process_group_resolvers(),
            mock.patch.object(
                latent_cp_cudnn_backend,
                "cudnn_backward_proxy",
                wraps=latent_cp_cudnn_backend.cudnn_backward_proxy,
            ) as proxy_spy,
        ):
            merged_output.backward(upstream)
        assert proxy_spy.call_count == 2
        reachable_lse_grads = [call.args[2] for call in proxy_spy.call_args_list]
        assert all(torch.isfinite(grad).all() for grad in reachable_lse_grads)
        assert any(torch.count_nonzero(grad) > 0 for grad in reachable_lse_grads)
        reference_merged_output.backward(upstream)

        labels = (
            "dq_diagonal",
            "dk_diagonal",
            "dv_diagonal",
            "dq_lower",
            "dk_lower",
            "dv_lower",
        )
        local_metrics = {
            label: _measure_similarity(actual.grad, expected.grad, label)
            for label, actual, expected in zip(
                labels, actual_inputs, reference_inputs, strict=True
            )
        }
        local_metrics["dq_sum"] = _measure_similarity(
            q_diagonal.grad + q_lower.grad,
            q_diagonal_ref.grad + q_lower_ref.grad,
            "summed dQ",
        )
        local_metrics.update(
            {
                "diagonal_lse": _measure_similarity(
                    diagonal_lse.detach(),
                    diagonal_lse_reference.detach(),
                    "diagonal LSE",
                ),
                "diagonal_output": _measure_similarity(
                    diagonal_output.detach(),
                    diagonal_reference.detach(),
                    "diagonal output",
                ),
                "lower_lse": _measure_similarity(
                    lower_lse.detach(), lower_lse_reference.detach(), "lower LSE"
                ),
                "lower_output": _measure_similarity(
                    lower_output.detach(), lower_reference.detach(), "lower output"
                ),
                "merged_lse": _measure_similarity(
                    merged_lse.detach(), reference_merged_lse.detach(), "merged LSE"
                ),
                "merged_output": _measure_similarity(
                    merged_output.detach(),
                    reference_merged_output.detach(),
                    "merged output",
                ),
            }
        )
        metrics, candidate_eps = _aggregate_and_emit_metrics(
            event="mla_latent_cp_cudnn_two_phase_merge_diagnostic",
            metadata={
                "phase_kinds": [diagonal.kind, lower.kind],
                "runtime_tuple": [
                    runtime[0].name,
                    runtime[1],
                    runtime[2],
                    list(runtime[3]),
                ],
                "seed": _SEED + 21,
                "shape": [
                    q_tokens,
                    diagonal_kv_tokens,
                    lower_kv_tokens,
                    heads,
                    192,
                    _VALUE_DIM,
                ],
            },
            local_metrics=local_metrics,
            pg=pg,
            device=q_diagonal.device,
        )
        _assert_emitted_metrics(
            metrics, candidate_eps, eps=EXPECTED_QUALIFICATION_EPS[runtime]
        )


def _run_production_parity(
    backend: AttnBackend,
    rope_type: str,
    *,
    torch_phase_backend: bool = False,
    attention_output_gate: bool = False,
    gate_granularity: str = "elementwise",
    no_rope: bool = False,
) -> None:
    runtime: latent_cp.QualifiedBackendTuple | None = None
    assertion_eps = _PARITY_EPS_CEILING
    if not torch_phase_backend:
        if backend is AttnBackend.flash and torch.cuda.get_device_capability()[0] < 10:
            pytest.skip("FA4 latent-CP qualification is Blackwell-only")
        runtime = _qualified_real_backend_runtime_or_skip(backend)
        assert runtime[0] is backend
        assertion_eps = EXPECTED_QUALIFICATION_EPS[runtime]

    with _model_parallel(2, 2) as pg:
        torch.manual_seed(_SEED)
        torch.cuda.manual_seed_all(_SEED)
        model_parallel_cuda_manual_seed(_SEED)
        config = _make_config(
            tp_size=2,
            cp_size=2,
            backend=backend,
            rope_type=rope_type,
            production_shape=True,
            attention_output_gate=attention_output_gate,
            gate_granularity=gate_granularity,
            no_rope=no_rope,
        )
        phase_backend = _TorchPackedAttentionAdapter() if torch_phase_backend else None
        layer = _build_layer(config, pg, phase_backend).cuda().bfloat16().train()
        assert (layer.linear_gate is not None) is attention_output_gate
        assert (layer.rotary_pos_emb is None) is no_rope
        if torch_phase_backend:
            assert layer._backend_adapter is phase_backend
            evidence_event = (
                "mla_latent_cp_torch_feature_parity"
                if attention_output_gate or no_rope
                else "mla_latent_cp_torch_full_chain_diagnostic"
            )
            evidence_backend = "torch_packed_attention"
            runtime_payload: list[object] = [
                "standard_pytorch",
                torch.__version__,
                "BF16 phase output with FP32 LSE",
            ]
        else:
            assert runtime is not None
            assert layer._backend_runtime_tuple == runtime
            assert layer._backend_adapter is not None
            evidence_event = "mla_latent_cp_parity"
            evidence_backend = backend.name
            runtime_payload = [
                runtime[0].name,
                runtime[1],
                runtime[2],
                list(runtime[3]),
            ]

        torch.manual_seed(_SEED + 1)
        torch.cuda.manual_seed_all(_SEED + 1)
        reference = NaiveMLA(config, torch.device("cuda")).train()
        _copy_reference_parameters(reference, layer, pg)

        total_tokens = sum(_PRODUCTION_PACKED_LENGTHS)
        full_hidden = torch.randn(
            total_tokens, 1, config.hidden_size, dtype=torch.bfloat16, device="cuda"
        )
        full_upstream = torch.randn_like(full_hidden)
        tp_cp_source = dist.get_process_group_ranks(pg.tp_cp)[0]
        dist.broadcast(full_hidden, src=tp_cp_source, group=pg.tp_cp)
        dist.broadcast(full_upstream, src=tp_cp_source, group=pg.tp_cp)
        cp_rank = dist.get_rank(pg.cp)
        local_indices = _zigzag_global_indices(
            _PRODUCTION_PACKED_LENGTHS, 2, cp_rank, "cuda"
        )
        cp_local_hidden = full_hidden.index_select(0, local_indices)
        local_hidden = (
            _sequence_parallel_slice(cp_local_hidden, pg.tp)
            .detach()
            .clone()
            .requires_grad_(True)
        )
        reference_hidden = full_hidden.detach().clone().requires_grad_(True)
        packed = _make_packed(_PRODUCTION_PACKED_LENGTHS, device="cuda", cp_group=pg.cp)

        with torch.no_grad():
            layout = layer._layout_adapter.prepare(
                local_hidden, packed, pg.cp, tp_group=pg.tp, sequence_parallel=True
            )
            _, production_payload = layer._project_query_and_payload(
                local_hidden, packed, layout
            )
            assert production_payload.shape == (
                local_hidden.size(0),
                _PRODUCTION_KV_LORA + _ROPE_DIM,
            )
            tp_payloads = [torch.empty_like(production_payload) for _ in range(2)]
            dist.all_gather(tp_payloads, production_payload, group=pg.tp)
            assert torch.cat(tp_payloads, dim=0).size(0) == cp_local_hidden.size(0)
        del production_payload, tp_payloads

        tp_inputs = [torch.empty_like(local_hidden) for _ in range(2)]
        dist.all_gather(tp_inputs, local_hidden, group=pg.tp)
        assert torch.equal(torch.cat(tp_inputs, dim=0), cp_local_hidden)
        del tp_inputs

        latent_cp.preprocess_mla_latent_cp(layer, local_hidden, packed)
        with _forbid_default_process_group_resolvers():
            real_output, bias = layer(local_hidden, None, packed_seq_params=packed)
        assert bias is None
        if not torch_phase_backend:
            assert layer._backend_runtime_tuple == runtime
            assert layer._backend_adapter is not None
        reference_output = reference(reference_hidden, _PRODUCTION_PACKED_LENGTHS)
        expected_local_output = _sequence_parallel_slice(
            reference_output.index_select(0, local_indices), pg.tp
        )
        parity_metrics = {
            "output": _measure_similarity(
                real_output.detach(), expected_local_output.detach(), "output"
            )
        }

        local_upstream = _sequence_parallel_slice(
            full_upstream.index_select(0, local_indices), pg.tp
        )
        with _forbid_default_process_group_resolvers():
            real_output.backward(local_upstream)
        reference_output.backward(full_upstream)
        expected_hidden_grad = _sequence_parallel_slice(
            reference_hidden.grad.index_select(0, local_indices), pg.tp
        )
        parity_metrics["input_gradient"] = _measure_similarity(
            local_hidden.grad, expected_hidden_grad, "input gradient"
        )

        reconstructed = _reconstruct_real_parameter_gradients(layer, pg)
        reference_params = dict(reference.named_parameters())
        parameter_map = _parameter_map(config)
        assert set(reconstructed) == set(reference_params) == set(parameter_map)
        for name in sorted(reference_params):
            assert reference_params[name].grad is not None
            label = f"parameter_gradient/{name}"
            parity_metrics[label] = _measure_similarity(
                reconstructed[name],
                reference_params[name].grad.detach().float(),
                f"parameter gradient {name}",
            )
        assert len(parity_metrics) == 2 + len(parameter_map)
        metrics, candidate_eps = _aggregate_and_emit_metrics(
            event=evidence_event,
            metadata={
                "backend": evidence_backend,
                "attention_output_gate": attention_output_gate,
                "gate_granularity": gate_granularity if attention_output_gate else None,
                "no_rope": no_rope,
                "rope": rope_type,
                "runtime_tuple": runtime_payload,
                "qualified_eps": assertion_eps,
                "seed": _SEED,
            },
            local_metrics=parity_metrics,
            pg=pg,
            device=local_hidden.device,
        )
        _assert_emitted_metrics(metrics, candidate_eps, eps=assertion_eps)


def _run_legacy_full_kv_cp_parity(rope_type: str) -> None:
    runtime = _qualified_real_backend_runtime_or_skip(AttnBackend.fused)
    assertion_eps = EXPECTED_QUALIFICATION_EPS[runtime]
    with _model_parallel(1, 2) as pg:
        torch.manual_seed(_SEED + 30)
        torch.cuda.manual_seed_all(_SEED + 30)
        model_parallel_cuda_manual_seed(_SEED + 30)
        config = _make_config(
            tp_size=1,
            cp_size=2,
            backend=AttnBackend.fused,
            rope_type=rope_type,
            production_shape=True,
        )
        latent_layer = _build_layer(config, pg).cuda().bfloat16().train()
        legacy_layer = _build_legacy_cp_layer(config, pg).cuda().bfloat16().train()
        incompatible = legacy_layer.load_state_dict(
            latent_layer.state_dict(), strict=True
        )
        assert incompatible.missing_keys == []
        assert incompatible.unexpected_keys == []

        total_tokens = sum(_PRODUCTION_PACKED_LENGTHS)
        full_hidden = torch.randn(
            total_tokens, 1, config.hidden_size, dtype=torch.bfloat16, device="cuda"
        )
        full_upstream = torch.randn_like(full_hidden)
        tp_cp_source = dist.get_process_group_ranks(pg.tp_cp)[0]
        dist.broadcast(full_hidden, src=tp_cp_source, group=pg.tp_cp)
        dist.broadcast(full_upstream, src=tp_cp_source, group=pg.tp_cp)
        local_indices = _zigzag_global_indices(
            _PRODUCTION_PACKED_LENGTHS, 2, dist.get_rank(pg.cp), "cuda"
        )
        cp_local_hidden = full_hidden.index_select(0, local_indices)
        local_hidden = _sequence_parallel_slice(cp_local_hidden, pg.tp)
        latent_hidden = local_hidden.detach().clone().requires_grad_(True)
        legacy_hidden = local_hidden.detach().clone().requires_grad_(True)
        local_upstream = _sequence_parallel_slice(
            full_upstream.index_select(0, local_indices), pg.tp
        )
        latent_packed = _make_packed(
            _PRODUCTION_PACKED_LENGTHS, device="cuda", cp_group=pg.cp
        )
        legacy_packed = _make_packed(
            _PRODUCTION_PACKED_LENGTHS, device="cuda", cp_group=pg.cp
        )

        latent_cp.preprocess_mla_latent_cp(latent_layer, latent_hidden, latent_packed)
        with _forbid_default_process_group_resolvers():
            latent_output, latent_bias = latent_layer(
                latent_hidden, None, packed_seq_params=latent_packed
            )
        legacy_output, legacy_bias = legacy_layer(
            legacy_hidden, None, packed_seq_params=legacy_packed
        )
        assert latent_bias is None
        assert legacy_bias is None
        parity_metrics = {
            "output": _measure_similarity(
                latent_output.detach(),
                legacy_output.detach(),
                "latent CP vs legacy full-KV CP output",
            )
        }
        with _forbid_default_process_group_resolvers():
            latent_output.backward(local_upstream)
        legacy_output.backward(local_upstream)
        assert latent_hidden.grad is not None
        assert legacy_hidden.grad is not None
        parity_metrics["input_gradient"] = _measure_similarity(
            latent_hidden.grad,
            legacy_hidden.grad,
            "latent CP vs legacy full-KV CP input gradient",
        )

        latent_gradients = _reconstruct_real_parameter_gradients(latent_layer, pg)
        legacy_gradients = _reconstruct_real_parameter_gradients(legacy_layer, pg)
        parameter_map = _parameter_map(config)
        assert set(latent_gradients) == set(legacy_gradients) == set(parameter_map)
        for name in sorted(latent_gradients):
            label = f"parameter_gradient/{name}"
            parity_metrics[label] = _measure_similarity(
                latent_gradients[name],
                legacy_gradients[name],
                f"latent CP vs legacy full-KV CP parameter gradient {name}",
            )
        assert len(parity_metrics) == 2 + len(parameter_map)

        metrics, candidate_eps = _aggregate_and_emit_metrics(
            event="mla_latent_cp_legacy_full_kv_cp_parity",
            metadata={
                "backend": AttnBackend.fused.name,
                "legacy_path": "MLASelfAttention+TEDotProductAttention",
                "rope": rope_type,
                "qualified_eps": assertion_eps,
                "runtime_tuple": [
                    runtime[0].name,
                    runtime[1],
                    runtime[2],
                    list(runtime[3]),
                ],
                "seed": _SEED + 30,
            },
            local_metrics=parity_metrics,
            pg=pg,
            device=latent_hidden.device,
        )
        _assert_emitted_metrics(metrics, candidate_eps, eps=assertion_eps)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cudnn_do_only_phase_backward_diagnostic():
    _run_cudnn_do_only_phase_diagnostic()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cudnn_two_phase_merged_backward_diagnostic():
    _run_cudnn_two_phase_merge_diagnostic()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_torch_phase_tp2_cp2_production_shape_full_chain_diagnostic():
    _run_production_parity(AttnBackend.fused, "rope", torch_phase_backend=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize(
    ("gate_granularity", "no_rope"),
    [("elementwise", False), ("headwise", False), ("headwise", True)],
)
def test_torch_phase_output_gate_and_kimi_no_rope_parity(
    gate_granularity: str, no_rope: bool
):
    _run_production_parity(
        AttnBackend.fused,
        "rope",
        torch_phase_backend=True,
        attention_output_gate=True,
        gate_granularity=gate_granularity,
        no_rope=no_rope,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("rope_type", ["rope", "yarn"])
def test_fused_cudnn_tp2_cp2_production_shape_parity(rope_type: str):
    _run_production_parity(AttnBackend.fused, rope_type)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("rope_type", ["rope", "yarn"])
def test_fa4_tp2_cp2_production_shape_parity_blackwell(rope_type: str):
    _run_production_parity(AttnBackend.flash, rope_type)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("rope_type", ["rope", "yarn"])
def test_mla_latent_cp_matches_legacy_full_kv_cp(rope_type: str):
    """Compare forward and backward with the legacy TE full-KV CP wrapper."""

    _run_legacy_full_kv_cp_parity(rope_type)
