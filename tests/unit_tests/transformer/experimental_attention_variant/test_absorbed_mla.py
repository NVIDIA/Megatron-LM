# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import random
from types import SimpleNamespace
from typing import List, Optional, Tuple

import pytest
import torch
import torch.distributed as dist

from megatron.core import parallel_state
from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant.absorbed_mla import (
    AbsorbedMLASelfAttention,
    AbsorbedMLASelfAttentionSubmodules,
)
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.multi_latent_attention import (
    MLASelfAttention,
    MLASelfAttentionSubmodules,
)
from megatron.core.utils import init_method_normal, scaled_init_method_normal
from tests.unit_tests.test_utilities import Utils


class MockCoreAttention(torch.nn.Module):
    """Mock core attention for testing MLA computation flow."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.softmax_scale = kwargs["softmax_scale"]
        self.k_channels = kwargs["k_channels"]
        self.v_channels = kwargs["v_channels"]
        self.pg_collection = kwargs["pg_collection"]

    def forward(
        self, q, k, v, *args, packed_seq_params: Optional[PackedSeqParams] = None, **kwargs
    ):
        """Mock forward pass."""
        if packed_seq_params is None:
            return self._forward_standard(q, k, v)
        else:
            return self._forward_thd(q, k, v, packed_seq_params)

    def _forward_standard(self, q, k, v):
        """Standard forward for [s, b, n, d] format."""
        sq, b, n = q.shape[:3]
        dtype = q.dtype
        if v is None:
            # Absorbed MLA
            assert q.shape[-1] == self.k_channels
            assert k.shape == (sq, b, 1, self.k_channels)
            v = k[..., : self.v_channels]
            k = k.expand(-1, -1, n, -1)
            v = v.expand(-1, -1, n, -1)
        else:
            # Standard MLA
            assert k.shape == q.shape
            assert v.shape[:-1] == q.shape[:-1]

        q = q.permute(1, 2, 0, 3).contiguous()
        k = k.permute(1, 2, 3, 0).contiguous()
        v = v.permute(1, 2, 0, 3).contiguous()

        q = q.view(b * n, q.size(-2), q.size(-1)).float()
        k = k.view(b * n, k.size(-2), k.size(-1)).float()
        v = v.view(b * n, v.size(-2), v.size(-1)).float()

        score = torch.bmm(q, k) * self.softmax_scale
        score = torch.nn.functional.softmax(score, dim=-1, dtype=torch.float32)
        out = torch.bmm(score, v)
        out = out.to(dtype)
        out = out.permute(1, 0, 2)
        out = out.reshape(sq, b, -1)

        return out

    def _forward_thd(self, q, k, v, packed_seq_params):
        """Forward for THD packed sequence format."""
        cu_seqlens = packed_seq_params.cu_seqlens_q
        num_seqs = len(cu_seqlens) - 1

        sq, n = q.shape[:2]
        dtype = q.dtype
        if v is None:
            # Absorbed MLA
            assert q.shape[-1] == self.k_channels
            assert k.shape == (sq, 1, self.k_channels)
            v = k[..., : self.v_channels]
            k = k.expand(-1, n, -1)
            v = v.expand(-1, n, -1)
        else:
            # Standard MLA
            assert k.shape == q.shape
            assert v.shape[:-1] == q.shape[:-1]

        out_list = []
        for i in range(num_seqs):
            start = cu_seqlens[i] // self.pg_collection.cp.size()
            end = cu_seqlens[i + 1] // self.pg_collection.cp.size()
            q_seq = q[start:end]
            k_seq = k[start:end]
            v_seq = v[start:end]

            q_seq = q_seq.permute(1, 0, 2).contiguous().float()
            k_seq = k_seq.permute(1, 2, 0).contiguous().float()
            v_seq = v_seq.permute(1, 0, 2).contiguous().float()

            score = torch.bmm(q_seq, k_seq) * self.softmax_scale
            score = torch.nn.functional.softmax(score, dim=-1, dtype=torch.float32)
            out = torch.bmm(score, v_seq)
            out = out.to(dtype)
            out = out.permute(1, 0, 2).contiguous()
            out = out.reshape(out.shape[0], -1)
            out_list.append(out)

        return torch.cat(out_list, dim=0)


def get_mock_mla_config(
    tensor_model_parallel_size: int, context_parallel_size: int
) -> SimpleNamespace:
    """Create test config with all attributes used in MLA."""
    return SimpleNamespace(
        multi_latent_attention=True,
        hidden_size=7168,
        num_attention_heads=128,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_head_dim=128,
        qk_pos_emb_head_dim=64,
        v_head_dim=128,
        add_bias_linear=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        layernorm_epsilon=1e-5,
        normalization="RMSNorm",
        layernorm_zero_centered_gamma=False,
        expert_model_parallel_size=1,
        tensor_model_parallel_size=tensor_model_parallel_size,
        sequence_parallel=tensor_model_parallel_size > 1,
        context_parallel_size=context_parallel_size,
        apply_rope_fusion=False,
        rope_type="yarn",
        rotary_scaling_factor=40,
        mscale=1.0,
        mscale_all_dim=1.0,
        rotary_base=10000,
        original_max_position_embeddings=4096,
        beta_fast=32,
        beta_slow=1,
        rotary_interleaved=False,
        recompute_granularity=None,
        fine_grained_activation_offloading=False,
        gradient_accumulation_fusion=False,
        fp8=False,
        fp4=False,
        init_method=init_method_normal(0.02),
        output_layer_init_method=scaled_init_method_normal(0.02, 61, multiplier=2.0),
        kv_channels=56,
        num_query_groups=128,
        batch_invariant_mode=False,
        cache_mla_latents=False,
        use_cpu_initialization=False,
        perform_initialization=True,
        symmetric_ar_type=None,
        disable_parameter_transpose_cache=False,
        init_model_with_meta_device=False,
        delay_wgrad_compute=False,
        tp_comm_overlap=False,
        experimental_attention_variant=None,
    )


def get_absorbed_mla_submodules(
    down_proj_use_column_parallel: bool, qk_layernorm: bool, rms_norm: bool
) -> AbsorbedMLASelfAttentionSubmodules:
    """Get submodules for AbsorbedMLASelfAttention testing."""
    backend = TESpecProvider()
    linear_q_down_proj = (
        backend.column_parallel_linear() if down_proj_use_column_parallel else backend.linear()
    )
    linear_kv_down_proj = (
        backend.column_parallel_linear() if down_proj_use_column_parallel else backend.linear()
    )
    qk_norm = backend.layer_norm(rms_norm=rms_norm, for_qk=True) if qk_layernorm else IdentityOp
    return AbsorbedMLASelfAttentionSubmodules(
        linear_q_proj=backend.column_parallel_linear(),
        linear_q_down_proj=linear_q_down_proj,
        linear_q_up_proj=backend.column_parallel_linear(),
        linear_kv_down_proj=linear_kv_down_proj,
        linear_k_up_proj=backend.column_parallel_linear(),
        linear_v_up_proj=backend.column_parallel_linear(),
        core_attention=MockCoreAttention,
        linear_proj=backend.row_parallel_linear(),
        q_layernorm=qk_norm,
        kv_layernorm=qk_norm,
    )


def get_mla_submodules(
    down_proj_use_column_parallel: bool, qk_layernorm: bool, rms_norm: bool
) -> MLASelfAttentionSubmodules:
    """Get submodules for AbsorbedMLASelfAttention testing."""
    backend = TESpecProvider()
    linear_q_down_proj = (
        backend.column_parallel_linear() if down_proj_use_column_parallel else backend.linear()
    )
    linear_kv_down_proj = (
        backend.column_parallel_linear() if down_proj_use_column_parallel else backend.linear()
    )
    qk_norm = backend.layer_norm(rms_norm=rms_norm, for_qk=True) if qk_layernorm else IdentityOp
    return MLASelfAttentionSubmodules(
        linear_q_proj=backend.column_parallel_linear(),
        linear_q_down_proj=linear_q_down_proj,
        linear_q_up_proj=backend.column_parallel_linear(),
        linear_kv_down_proj=linear_kv_down_proj,
        linear_kv_up_proj=backend.column_parallel_linear(),
        core_attention=MockCoreAttention,
        linear_proj=backend.row_parallel_linear(),
        q_layernorm=qk_norm,
        kv_layernorm=qk_norm,
    )


@pytest.mark.parametrize("tp_cp", [[1, 1], [2, 1], [1, 2], [2, 2]])
@pytest.mark.parametrize("qkv_format", ['sbhd', 'thd'])
@pytest.mark.parametrize("down_proj_use_column_parallel", [False, True])
def test_functionality(tp_cp: List[int], qkv_format: str, down_proj_use_column_parallel: bool):
    """Test that AbsorbedMLASelfAttention is equivalent to standard MLA."""
    tp_size, cp_size = tp_cp
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=tp_size, context_parallel_size=cp_size
    )
    model_parallel_cuda_manual_seed(123)

    # Create model
    config = get_mock_mla_config(tensor_model_parallel_size=tp_size, context_parallel_size=cp_size)
    absorbed_submodules = get_absorbed_mla_submodules(
        down_proj_use_column_parallel=down_proj_use_column_parallel,
        qk_layernorm=True,
        rms_norm=True,
    )
    standard_submodules = get_mla_submodules(
        down_proj_use_column_parallel=down_proj_use_column_parallel,
        qk_layernorm=True,
        rms_norm=True,
    )
    absorbed_mla = AbsorbedMLASelfAttention(
        config=config,
        submodules=absorbed_submodules,
        layer_number=0,
        attn_mask_type=AttnMaskType.causal,
        cp_comm_type="all_gather" if cp_size > 1 else None,
        pg_collection=None,
    ).cuda()
    standard_mla = MLASelfAttention(
        config=config,
        submodules=standard_submodules,
        layer_number=0,
        attn_mask_type=AttnMaskType.causal,
        cp_comm_type="all_gather" if cp_size > 1 else None,
        pg_collection=None,
    ).cuda()

    state_dict = standard_mla.state_dict()
    absorbed_mla.load_state_dict(state_dict)

    # Prepare random data
    if qkv_format == 'thd':
        # Create random seqlens
        num_seqs, min_len, max_len = 3, 128, 1024
        divisor = tp_size * cp_size * 2
        random.seed(42)
        seqlens = [random.randint(min_len, max_len) // divisor * divisor for _ in range(num_seqs)]
        # Create cumulative sequence lengths
        cu_seqlens = [0]
        for length in seqlens:
            cu_seqlens.append(cu_seqlens[-1] + length)
        total_tokens = cu_seqlens[-1]
        cu_seqlens = torch.IntTensor(cu_seqlens).cuda()
        max_seqlen = max(seqlens)
        # Create packed sequence parameters
        packed_seq_params = PackedSeqParams(
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_q_padded=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            cu_seqlens_kv_padded=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_kv=max_seqlen,
            qkv_format='thd',
        )
        hidden_states = torch.randn(
            (total_tokens // tp_size // cp_size, 1, config.hidden_size),
            dtype=torch.bfloat16,
            device='cuda',
        )
        grads = torch.randn_like(hidden_states)
    else:
        seqlen = 1024 // tp_size // cp_size
        hidden_states = torch.randn((seqlen, 3, 7168), dtype=torch.bfloat16, device='cuda')
        grads = torch.randn_like(hidden_states)
        packed_seq_params = None

    # Forward & Backward
    for name, param in absorbed_mla.named_parameters():
        if param.grad is not None:
            param.grad.zero_()
    absorbed_outputs, _ = absorbed_mla(
        hidden_states, attention_mask=None, packed_seq_params=packed_seq_params
    )
    absorbed_outputs.backward(grads)

    for name, param in standard_mla.named_parameters():
        if param.grad is not None:
            param.grad.zero_()
    standard_outputs, _ = standard_mla(
        hidden_states, attention_mask=None, packed_seq_params=packed_seq_params
    )
    standard_outputs.backward(grads)

    # Compute cosine similarity
    absorbed_flat = absorbed_outputs.flatten().float()
    standard_flat = standard_outputs.flatten().float()
    cosine_sim = torch.nn.functional.cosine_similarity(
        absorbed_flat.unsqueeze(0), standard_flat.unsqueeze(0)
    ).item()
    assert cosine_sim > 0.9999, f"output cosine similarity = {cosine_sim} < 0.9999"
    torch.testing.assert_close(absorbed_outputs, standard_outputs, atol=5e-3, rtol=5e-3)

    for name, param in absorbed_mla.named_parameters():
        assert param.grad is not None
    for name, param in standard_mla.named_parameters():
        assert param.grad is not None

    # Compare gradients with cosine similarity
    absorbed_grads = dict(absorbed_mla.named_parameters())
    standard_grads = dict(standard_mla.named_parameters())

    # Map parameter names between absorbed and standard MLA
    # Most parameters have the same name, except for K/V up proj
    for name, param in standard_grads.items():
        if 'linear_kv_up_proj' in name:
            # Special handling: combine k and v up proj grads from absorbed_mla
            k_name = name.replace('linear_kv_up_proj', 'linear_k_up_proj')
            v_name = name.replace('linear_kv_up_proj', 'linear_v_up_proj')

            k_grad = absorbed_grads[k_name].grad
            v_grad = absorbed_grads[v_name].grad

            # Combine k and v grads (interleaved by head)
            # k_grad: [n * qk_head_dim, kv_lora_rank]
            # v_grad: [n * v_head_dim, kv_lora_rank]
            # combined: [n * (qk_head_dim + v_head_dim), kv_lora_rank]
            n_heads = absorbed_mla.num_attention_heads_per_partition
            qk_head_dim = absorbed_mla.config.qk_head_dim
            v_head_dim = absorbed_mla.config.v_head_dim
            kv_lora_rank = absorbed_mla.config.kv_lora_rank

            k_grad_3d = k_grad.view(n_heads, qk_head_dim, kv_lora_rank)
            v_grad_3d = v_grad.view(n_heads, v_head_dim, kv_lora_rank)
            combined_grad_3d = torch.cat([k_grad_3d, v_grad_3d], dim=1)
            combined_grad = combined_grad_3d.view(-1, kv_lora_rank)

            absorbed_grad_flat = combined_grad.flatten().float()
            standard_grad_flat = param.grad.flatten().float()

            cos_sim = torch.nn.functional.cosine_similarity(
                absorbed_grad_flat.unsqueeze(0), standard_grad_flat.unsqueeze(0)
            ).item()
            assert cos_sim > 0.9999, f"name: {name}, cosine similarity = {cos_sim} < 0.9999"
            torch.testing.assert_close(absorbed_grad, standard_grad, atol=1e-1, rtol=1e-2)
        else:
            absorbed_grad = absorbed_grads[name].grad
            standard_grad = param.grad

            absorbed_grad_flat = absorbed_grad.flatten().float()
            standard_grad_flat = standard_grad.flatten().float()

            cos_sim = torch.nn.functional.cosine_similarity(
                absorbed_grad_flat.unsqueeze(0), standard_grad_flat.unsqueeze(0)
            ).item()
            assert cos_sim > 0.9999, f"name: {name}, cosine similarity = {cos_sim} < 0.9999"
            torch.testing.assert_close(absorbed_grad, standard_grad, atol=1e-1, rtol=1e-2)

    Utils.destroy_model_parallel()
