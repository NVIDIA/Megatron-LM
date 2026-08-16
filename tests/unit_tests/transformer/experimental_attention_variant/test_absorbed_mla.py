# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import random
from typing import List, Optional, Tuple

import pytest
import torch
import torch.distributed as dist

from megatron.core import parallel_state
from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
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
from megatron.core.transformer.transformer_config import MLATransformerConfig
from megatron.core.utils import init_method_normal, scaled_init_method_normal
from tests.unit_tests.test_utilities import Utils


class MockCoreAttention(torch.nn.Module):
    """Mock core attention for testing MLA computation flow."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.softmax_scale = kwargs.get("softmax_scale")
        self.k_channels = kwargs.get("k_channels")
        self.v_channels = kwargs.get("v_channels")
        self.pg_collection = kwargs.get("pg_collection")

    def forward(
        self, q, k, v=None, *args, packed_seq_params: Optional[PackedSeqParams] = None, **kwargs
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


class FakeCPGroup:
    """Minimal context-parallel group for local RoPE caller tests."""

    def __init__(self, size=1, rank=0):
        self._size = size
        self._rank = rank

    def size(self):
        return self._size

    def rank(self):
        return self._rank


def get_mock_mla_config(
    tensor_model_parallel_size: int,
    context_parallel_size: int,
    sequence_parallel: bool,
    recompute_mla_up_proj: bool,
    apply_rope_fusion: bool = False,
    rope_type: str = "yarn",
) -> MLATransformerConfig:
    """Create test config with all attributes used in MLA."""
    return MLATransformerConfig(
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
        sequence_parallel=tensor_model_parallel_size > 1 and sequence_parallel,
        context_parallel_size=context_parallel_size,
        apply_rope_fusion=apply_rope_fusion,
        rope_type=rope_type,
        rotary_scaling_factor=40,
        mscale=1.0,
        mscale_all_dim=1.0,
        rotary_base=10000,
        original_max_position_embeddings=4096,
        beta_fast=32,
        beta_slow=1,
        rotary_interleaved=False,
        recompute_granularity="selective" if recompute_mla_up_proj else None,
        recompute_modules=["mla_up_proj"] if recompute_mla_up_proj else [],
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
        softmax_scale=None,
    )


def get_absorbed_mla_submodules(
    down_proj_use_column_parallel: bool,
    qk_layernorm: bool,
    rms_norm: bool,
    combined_kv_up_projection: bool,
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
    if combined_kv_up_projection:
        kv_up_projections = {"linear_kv_up_proj": backend.column_parallel_linear()}
    else:
        kv_up_projections = {
            "linear_k_up_proj": backend.column_parallel_linear(),
            "linear_v_up_proj": backend.column_parallel_linear(),
        }

    return AbsorbedMLASelfAttentionSubmodules(
        linear_q_proj=backend.column_parallel_linear(),
        linear_q_down_proj=linear_q_down_proj,
        linear_q_up_proj=backend.column_parallel_linear(),
        linear_kv_down_proj=linear_kv_down_proj,
        core_attention=MockCoreAttention,
        linear_proj=backend.row_parallel_linear(),
        q_layernorm=qk_norm,
        kv_layernorm=qk_norm,
        **kv_up_projections,
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


# TODO: Consider using get_gpt_layer_with_transformer_engine_spec from
#       megatron.core.models.gpt.gpt_layer_specs to simplify submodule setup and cover real specs.
# TODO: Add test case to cover TP > 1 but SP = False.


def _run_functionality(
    tp_cp_sp: List,
    qkv_format: str,
    down_proj_use_column_parallel: bool,
    recompute_mla_up_proj: bool,
    combined_kv_up_projection: bool,
    apply_rope_fusion: bool = False,
    rope_type: str = "yarn",
    check_hidden_grad: bool = False,
):
    """Test that AbsorbedMLASelfAttention is equivalent to standard MLA."""
    tp_size, cp_size, sp = tp_cp_sp
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=tp_size, context_parallel_size=cp_size
    )
    model_parallel_cuda_manual_seed(123)

    # Create model
    config = get_mock_mla_config(
        tensor_model_parallel_size=tp_size,
        context_parallel_size=cp_size,
        sequence_parallel=sp,
        recompute_mla_up_proj=recompute_mla_up_proj,
        apply_rope_fusion=apply_rope_fusion,
        rope_type=rope_type,
    )
    absorbed_submodules = get_absorbed_mla_submodules(
        down_proj_use_column_parallel=down_proj_use_column_parallel,
        qk_layernorm=True,
        rms_norm=True,
        combined_kv_up_projection=combined_kv_up_projection,
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
            (total_tokens // cp_size // (tp_size if sp else 1), 1, config.hidden_size),
            dtype=torch.bfloat16,
            device='cuda',
        )
        grads = torch.randn_like(hidden_states)
    else:
        # When SP is enabled, sequence is sharded across TP ranks
        # When SP is disabled, each TP rank has the full sequence
        seqlen = 1024 // cp_size // (tp_size if sp else 1)
        hidden_states = torch.randn((seqlen, 3, 7168), dtype=torch.bfloat16, device='cuda')
        grads = torch.randn_like(hidden_states)
        packed_seq_params = None

    absorbed_hidden_states = hidden_states.detach().requires_grad_(check_hidden_grad)
    standard_hidden_states = hidden_states.detach().clone().requires_grad_(check_hidden_grad)

    # Forward & Backward
    for name, param in absorbed_mla.named_parameters():
        if param.grad is not None:
            param.grad.zero_()
    absorbed_outputs, _ = absorbed_mla(
        absorbed_hidden_states, attention_mask=None, packed_seq_params=packed_seq_params
    )
    absorbed_outputs.backward(grads)

    for name, param in standard_mla.named_parameters():
        if param.grad is not None:
            param.grad.zero_()
    standard_outputs, _ = standard_mla(
        standard_hidden_states, attention_mask=None, packed_seq_params=packed_seq_params
    )
    standard_outputs.backward(grads)

    def _calculate_tensor_similarity(x, y):
        x, y = x.data.double(), y.data.double()
        denominator = (x * x + y * y).sum()
        if denominator == 0:
            return 1
        sim = 2 * (x * y).sum() / denominator
        return sim

    # Compute cosine similarity
    absorbed_flat = absorbed_outputs.flatten().float()
    standard_flat = standard_outputs.flatten().float()
    cosine_sim = torch.nn.functional.cosine_similarity(
        absorbed_flat.unsqueeze(0), standard_flat.unsqueeze(0)
    ).item()
    assert cosine_sim > 0.9999, f"output cosine similarity = {cosine_sim} < 0.9999"
    assert _calculate_tensor_similarity(absorbed_outputs, standard_outputs) > 0.9999
    torch.testing.assert_close(absorbed_outputs, standard_outputs, atol=5e-3, rtol=5e-3)
    if check_hidden_grad:
        torch.testing.assert_close(
            absorbed_hidden_states.grad, standard_hidden_states.grad, atol=5e-3, rtol=5e-3
        )

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
        if 'linear_kv_up_proj' in name and not combined_kv_up_projection:
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
            assert _calculate_tensor_similarity(combined_grad, param.grad) > 0.9999
        else:
            absorbed_grad = absorbed_grads[name].grad
            standard_grad = param.grad

            absorbed_grad_flat = absorbed_grad.flatten().float()
            standard_grad_flat = standard_grad.flatten().float()

            cos_sim = torch.nn.functional.cosine_similarity(
                absorbed_grad_flat.unsqueeze(0), standard_grad_flat.unsqueeze(0)
            ).item()
            assert cos_sim > 0.9999, f"name: {name}, cosine similarity = {cos_sim} < 0.9999"
            assert _calculate_tensor_similarity(absorbed_grad, standard_grad) > 0.9999

    Utils.destroy_model_parallel()


@pytest.mark.parametrize("tp_cp_sp", [[1, 1, False], [2, 1, True], [1, 2, False], [2, 2, True]])
@pytest.mark.parametrize("qkv_format", ['sbhd', 'thd'])
@pytest.mark.parametrize("down_proj_use_column_parallel", [False, True])
@pytest.mark.parametrize("recompute_mla_up_proj", [False, True])
@pytest.mark.parametrize("combined_kv_up_projection", [True, False])
def test_functionality(
    tp_cp_sp: List,
    qkv_format: str,
    down_proj_use_column_parallel: bool,
    recompute_mla_up_proj: bool,
    combined_kv_up_projection: bool,
):
    _run_functionality(
        tp_cp_sp,
        qkv_format,
        down_proj_use_column_parallel,
        recompute_mla_up_proj,
        combined_kv_up_projection,
    )


@pytest.mark.parametrize("qkv_format", ['sbhd', 'thd'])
def test_standard_rope_fusion_functionality(qkv_format):
    """Absorbed MLA's fused packing must match standard MLA end to end."""
    _run_functionality(
        [1, 1, False],
        qkv_format,
        down_proj_use_column_parallel=False,
        recompute_mla_up_proj=False,
        combined_kv_up_projection=True,
        apply_rope_fusion=True,
        rope_type="rope",
        check_hidden_grad=True,
    )


@pytest.mark.parametrize("attention_type", ["standard", "absorbed"])
@pytest.mark.parametrize(("qkv_format", "cp_size"), [("sbhd", 1), ("thd", 1), ("thd", 2)])
def test_standard_rope_fused_unfused_parity(attention_type, qkv_format, cp_size):
    """Standard RoPE fusion must preserve module outputs and all gradients."""
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=1)
    model_parallel_cuda_manual_seed(123)

    configs = [
        get_mock_mla_config(
            tensor_model_parallel_size=1,
            context_parallel_size=cp_size,
            sequence_parallel=False,
            recompute_mla_up_proj=False,
            apply_rope_fusion=apply_rope_fusion,
            rope_type="rope",
        )
        for apply_rope_fusion in (False, True)
    ]

    if attention_type == "absorbed":
        attention_cls = AbsorbedMLASelfAttention
        submodules = [
            get_absorbed_mla_submodules(
                down_proj_use_column_parallel=False,
                qk_layernorm=True,
                rms_norm=True,
                combined_kv_up_projection=True,
            )
            for _ in configs
        ]
    else:
        attention_cls = MLASelfAttention
        submodules = [
            get_mla_submodules(
                down_proj_use_column_parallel=False, qk_layernorm=True, rms_norm=True
            )
            for _ in configs
        ]

    cp_group = FakeCPGroup(size=cp_size, rank=cp_size - 1)
    attentions = [
        attention_cls(
            config=config,
            submodules=module_spec,
            layer_number=0,
            attn_mask_type=AttnMaskType.causal,
            cp_comm_type="all_gather" if cp_size > 1 else None,
            pg_collection=ProcessGroupCollection(tp=None, cp=cp_group),
        ).cuda()
        for config, module_spec in zip(configs, submodules)
    ]
    attentions[1].load_state_dict(attentions[0].state_dict())

    if qkv_format == "thd":
        seqlens = [96, 160, 64]
        cu_seqlens = torch.tensor([0, 96, 256, 320], dtype=torch.int32, device="cuda")
        packed_seq_params = PackedSeqParams(
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_q_padded=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            cu_seqlens_kv_padded=cu_seqlens,
            max_seqlen_q=max(seqlens),
            max_seqlen_kv=max(seqlens),
            qkv_format="thd",
        )
        hidden_states = torch.randn(
            (sum(seqlens) // cp_size, 1, configs[0].hidden_size),
            dtype=torch.bfloat16,
            device="cuda",
        )
    else:
        packed_seq_params = None
        hidden_states = torch.randn(
            (320, 2, configs[0].hidden_size), dtype=torch.bfloat16, device="cuda"
        )

    inputs = [hidden_states.detach().clone().requires_grad_(True) for _ in attentions]
    output_grad = torch.randn_like(hidden_states)
    outputs = []
    for attention, input_tensor in zip(attentions, inputs):
        output, _ = attention(
            input_tensor, attention_mask=None, packed_seq_params=packed_seq_params
        )
        output.backward(output_grad)
        outputs.append(output)

    torch.testing.assert_close(outputs[1], outputs[0], atol=5e-3, rtol=5e-3)
    torch.testing.assert_close(inputs[1].grad, inputs[0].grad, atol=5e-3, rtol=5e-3)

    unfused_parameters = dict(attentions[0].named_parameters())
    fused_parameters = dict(attentions[1].named_parameters())
    assert fused_parameters.keys() == unfused_parameters.keys()
    for name in unfused_parameters:
        unfused_grad = unfused_parameters[name].grad
        fused_grad = fused_parameters[name].grad
        assert unfused_grad is not None, f"unfused parameter {name} has no gradient"
        assert fused_grad is not None, f"fused parameter {name} has no gradient"
        fused_grad_fp64 = fused_grad.double()
        unfused_grad_fp64 = unfused_grad.double()
        denominator = (fused_grad_fp64.square() + unfused_grad_fp64.square()).sum()
        similarity = (
            1.0
            if denominator == 0
            else (2 * fused_grad_fp64 * unfused_grad_fp64).sum() / denominator
        )
        assert similarity > 0.9999, f"parameter {name} gradient similarity = {similarity}"

    Utils.destroy_model_parallel()
