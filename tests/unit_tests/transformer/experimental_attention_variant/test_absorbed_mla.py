# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import random
from types import SimpleNamespace
from typing import List, Optional, Tuple

import numpy as np
import pytest
import torch
import torch.distributed as dist
from transformer_engine.pytorch.fp8 import check_fp8_support, check_nvfp4_support

from megatron.core import parallel_state
from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
from megatron.core.fp4_utils import get_fp4_context
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant import (
    absorbed_mla as absorbed_mla_module,
)
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

fp8_available, reason_for_no_fp8 = check_fp8_support()
nvfp4_available, reason_for_no_nvfp4 = check_nvfp4_support()


# Inlined from tests.unit_tests.determinism.utils rather than imported: that module pulls
# in torch.testing._internal.common_utils, whose import calls
# torch.backends.disable_global_flags() and breaks later tests in this pytest session that
# assign torch.backends.* flags (e.g. test_te_layers_batch_invariant.py).
def capture_rng_state() -> dict:
    """Snapshot every RNG that the framework consumes during a fwd+bwd pass."""
    from megatron.core.tensor_parallel.random import get_cuda_rng_tracker

    return {
        "random": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(),
        "mpu_tracker": get_cuda_rng_tracker().get_states(),
    }


def restore_rng_state(state: dict) -> None:
    """Inverse of ``capture_rng_state``."""
    from megatron.core.tensor_parallel.random import get_cuda_rng_tracker

    random.setstate(state["random"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state(state["torch_cuda"])
    get_cuda_rng_tracker().set_states(state["mpu_tracker"])


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


def get_mock_mla_config(
    tensor_model_parallel_size: int,
    context_parallel_size: int,
    qk_layernorm: bool,
    recompute_mla_up_proj: bool = False,
    fp8: Optional[str] = None,
    fp8_recipe: str = "delayed",
    fp4: Optional[str] = None,
    fp4_recipe: str = "nvfp4",
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
        qk_layernorm=qk_layernorm,
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
        recompute_granularity="selective" if recompute_mla_up_proj else None,
        recompute_modules=["mla_up_proj"] if recompute_mla_up_proj else [],
        fine_grained_activation_offloading=False,
        gradient_accumulation_fusion=False,
        fp8=fp8 if fp8 else False,
        fp8_recipe=fp8_recipe,
        fp4=fp4,
        fp4_recipe=fp4_recipe,
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
        linear_kv_up_proj=backend.column_parallel_linear(),
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


def test_checkpointed_attention_forward_captures_metadata(monkeypatch):
    """Optional metadata should stay in the closure instead of checkpoint tensor args."""

    packed_seq_params = PackedSeqParams(qkv_format='thd')
    checkpoint_args = None

    def fake_checkpoint(run_function, distribute_saved_activations, *args):
        nonlocal checkpoint_args
        del distribute_saved_activations
        checkpoint_args = args
        assert all(torch.is_tensor(arg) for arg in args)
        return run_function(*args)

    class CoreAttention(torch.nn.Module):
        def forward(self, query, key, *, value, attention_mask, **kwargs):
            del query, key, value, attention_mask
            assert kwargs["packed_seq_params"] is packed_seq_params
            assert kwargs["position_ids"] is None
            return kwargs["x"]

    dummy_attention = type(
        "DummyAttention",
        (),
        {"attn_mask_type": AttnMaskType.causal, "core_attention": CoreAttention()},
    )()

    monkeypatch.setattr(absorbed_mla_module.tensor_parallel, "checkpoint", fake_checkpoint)

    hidden_states = torch.randn(4, 1, 8)
    output = AbsorbedMLASelfAttention._checkpointed_attention_forward(
        dummy_attention,
        q_absorbed=torch.randn(4, 1, 2, 8),
        k_compressed=torch.randn(4, 1, 1, 8),
        hidden_states=hidden_states,
        q_compressed=torch.randn(4, 1, 8),
        attention_mask=torch.empty(1),
        up_v_weight=torch.randn(2, 4, 4),
        position_ids=None,
        packed_seq_params=packed_seq_params,
    )

    assert checkpoint_args is not None
    assert all(arg is not packed_seq_params for arg in checkpoint_args)
    assert all(arg is not None for arg in checkpoint_args)
    assert output is hidden_states


def test_restore_packed_thd_batch_dim_when_core_output_is_2d():
    """Packed-THD absorbed MLA should restore a missing singleton batch dim."""
    hidden_states = torch.empty(7, 1, 16)
    core_attn_out = torch.empty(7, 16)
    packed_seq_params = PackedSeqParams(qkv_format='thd')

    restored = absorbed_mla_module._restore_packed_thd_batch_dim(
        core_attn_out, hidden_states, packed_seq_params
    )

    assert restored.shape == (7, 1, 16)


def test_restore_packed_thd_batch_dim_keeps_already_normalized_output():
    """Packed-THD absorbed MLA should keep an already restored batch dim."""
    hidden_states = torch.empty(7, 1, 16)
    core_attn_out = torch.empty(7, 1, 16)
    packed_seq_params = PackedSeqParams(qkv_format='thd')

    restored = absorbed_mla_module._restore_packed_thd_batch_dim(
        core_attn_out, hidden_states, packed_seq_params
    )

    assert restored is core_attn_out
    assert restored.shape == hidden_states.shape


def test_absorbed_v_up_projection_applies_when_core_did_not_consume_weight():
    """Absorbed MLA should apply V-up when core attention returns latent channels."""
    torch.manual_seed(123)
    num_heads, kv_lora_rank, v_head_dim = 2, 3, 3
    core_attn_out = torch.randn(5, 1, num_heads * kv_lora_rank)
    v_up_weight = torch.randn(num_heads, v_head_dim, kv_lora_rank)

    projected = absorbed_mla_module._apply_absorbed_v_up_projection(
        core_attn_out,
        v_up_weight,
        num_attention_heads_per_partition=num_heads,
        kv_lora_rank=kv_lora_rank,
        v_head_dim=v_head_dim,
        core_consumed_v_up_projection=False,
    )
    expected = core_attn_out.view(5, 1, num_heads, kv_lora_rank)
    expected = torch.einsum("...nc,ndc->...nd", expected, v_up_weight)
    expected = expected.contiguous().view(5, 1, -1)

    torch.testing.assert_close(projected, expected, rtol=0, atol=0)


def test_absorbed_v_up_projection_skips_when_core_consumed_weight():
    """Absorbed MLA should not reapply V-up when core attention already consumed it."""
    num_heads, kv_lora_rank, v_head_dim = 2, 3, 3
    core_attn_out = torch.randn(5, 1, num_heads * v_head_dim)
    v_up_weight = torch.randn(num_heads, v_head_dim, kv_lora_rank)

    projected = absorbed_mla_module._apply_absorbed_v_up_projection(
        core_attn_out,
        v_up_weight,
        num_attention_heads_per_partition=num_heads,
        kv_lora_rank=kv_lora_rank,
        v_head_dim=v_head_dim,
        core_consumed_v_up_projection=True,
    )

    assert projected is core_attn_out


def test_load_from_state_dict_backwards_compatible_with_split_kv_up_projection(monkeypatch):
    """Pre-refactor split K/V up-projection checkpoints load into the combined layout."""

    dummy_attention = object.__new__(AbsorbedMLASelfAttention)
    dummy_attention.num_attention_heads_per_partition = 2
    dummy_attention.config = SimpleNamespace(qk_head_dim=2, v_head_dim=3, kv_lora_rank=4)

    prefix = "self_attention."
    k_weight = torch.arange(2 * 2 * 4, dtype=torch.float32).view(2 * 2, 4)
    v_weight = torch.arange(2 * 3 * 4, dtype=torch.float32).view(2 * 3, 4)
    state_dict = {
        f"{prefix}linear_k_up_proj.weight": k_weight.clone(),
        f"{prefix}linear_v_up_proj.weight": v_weight.clone(),
        f"{prefix}linear_k_up_proj._extra_state": torch.empty(0),
        f"{prefix}linear_v_up_proj._extra_state": torch.empty(0),
    }
    captured_state_dict = {}

    def fake_super_load(self, state_dict, *args, **kwargs):
        del self, args, kwargs
        captured_state_dict.update(state_dict)

    monkeypatch.setattr(absorbed_mla_module.Attention, "_load_from_state_dict", fake_super_load)

    AbsorbedMLASelfAttention._load_from_state_dict(
        dummy_attention, state_dict, prefix, {}, True, [], [], []
    )

    expected_weight = (
        torch.cat((k_weight.view(2, 2, 4), v_weight.view(2, 3, 4)), dim=1)
        .contiguous()
        .view(2 * (2 + 3), 4)
    )
    torch.testing.assert_close(
        captured_state_dict[f"{prefix}linear_kv_up_proj.weight"], expected_weight
    )
    assert f"{prefix}linear_k_up_proj.weight" not in captured_state_dict
    assert f"{prefix}linear_v_up_proj.weight" not in captured_state_dict
    assert f"{prefix}linear_kv_up_proj._extra_state" in captured_state_dict
    assert f"{prefix}linear_k_up_proj._extra_state" not in captured_state_dict
    assert f"{prefix}linear_v_up_proj._extra_state" not in captured_state_dict


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
    qk_layernorm = True
    config = get_mock_mla_config(
        tensor_model_parallel_size=tp_size, context_parallel_size=cp_size, qk_layernorm=qk_layernorm
    )
    absorbed_submodules = get_absorbed_mla_submodules(
        down_proj_use_column_parallel=down_proj_use_column_parallel,
        qk_layernorm=qk_layernorm,
        rms_norm=True,
    )
    standard_submodules = get_mla_submodules(
        down_proj_use_column_parallel=down_proj_use_column_parallel,
        qk_layernorm=qk_layernorm,
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

    for name, param in absorbed_mla.named_parameters():
        assert param.grad is not None
    for name, param in standard_mla.named_parameters():
        assert param.grad is not None

    # Compare gradients with cosine similarity
    absorbed_grads = dict(absorbed_mla.named_parameters())
    standard_grads = dict(standard_mla.named_parameters())

    for name, param in standard_grads.items():
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


# Hopper = SM 9.0, Blackwell = SM 10.0+. mxfp8 needs Blackwell.
_IS_BLACKWELL = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10

_QUANT_RECIPES = [
    pytest.param(
        {"fp8": "hybrid", "fp8_recipe": "tensorwise"},
        id="fp8-tensorwise",
        marks=pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8),
    ),
    pytest.param(
        {"fp8": "hybrid", "fp8_recipe": "mxfp8"},
        id="fp8-mxfp8",
        marks=pytest.mark.skipif(not fp8_available or not _IS_BLACKWELL, reason="needs Blackwell"),
    ),
    pytest.param(
        {"fp8": "hybrid", "fp8_recipe": "blockwise"},
        id="fp8-blockwise",
        marks=pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8),
    ),
    pytest.param(
        {"fp4": "e2m1", "fp4_recipe": "nvfp4"},
        id="fp4-nvfp4",
        marks=pytest.mark.skipif(not nvfp4_available, reason=reason_for_no_nvfp4),
    ),
]


def _build_absorbed_mla(config, state_dict=None):
    """Build an AbsorbedMLASelfAttention, optionally loading a shared state dict."""
    model = AbsorbedMLASelfAttention(
        config=config,
        submodules=get_absorbed_mla_submodules(
            down_proj_use_column_parallel=False, qk_layernorm=True, rms_norm=True
        ),
        layer_number=0,
        attn_mask_type=AttnMaskType.causal,
        cp_comm_type=None,
        pg_collection=None,
    ).cuda()
    if state_dict is not None:
        model.load_state_dict(state_dict)
    return model


@pytest.mark.parametrize("quant_overrides", _QUANT_RECIPES)
@pytest.mark.parametrize("qkv_format", ['sbhd', 'thd'])
def test_quantized_up_proj_recompute_parity(quant_overrides: dict, qkv_format: str):
    """`mla_up_proj` recompute must be an exact replay under FP8 and FP4.

    The absorbed up-projection recompute used to be blocked for FP8/FP4. It is
    allowed now, so pin the contract: running the same inputs through two
    identically-initialized modules — one recomputing the up projection, one not —
    must produce bitwise-identical outputs and parameter gradients. Any drift means
    the replay used a different quantization scale or a stale weight.

    A bf16 reference run guards against the parity check passing vacuously, that is,
    both modules silently falling back to bf16.
    """
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=1)
    model_parallel_cuda_manual_seed(123)

    def make_config(recompute: bool) -> MLATransformerConfig:
        return get_mock_mla_config(
            tensor_model_parallel_size=1,
            context_parallel_size=1,
            qk_layernorm=True,
            recompute_mla_up_proj=recompute,
            **quant_overrides,
        )

    baseline = _build_absorbed_mla(make_config(False))
    recomputed = _build_absorbed_mla(make_config(True), state_dict=baseline.state_dict())
    assert recomputed.recompute_up_proj and not baseline.recompute_up_proj

    # Quantized GEMMs need the token dimension aligned; keep every sequence a multiple
    # of 128, which satisfies both the FP8 (16) and FP4 (32) alignment requirements.
    if qkv_format == 'thd':
        random.seed(42)
        seqlens = [random.randint(1, 8) * 128 for _ in range(3)]
        cu_seqlens = [0]
        for length in seqlens:
            cu_seqlens.append(cu_seqlens[-1] + length)
        total_tokens = cu_seqlens[-1]
        packed_seq_params = PackedSeqParams(
            cu_seqlens_q=torch.IntTensor(cu_seqlens).cuda(),
            cu_seqlens_q_padded=torch.IntTensor(cu_seqlens).cuda(),
            cu_seqlens_kv=torch.IntTensor(cu_seqlens).cuda(),
            cu_seqlens_kv_padded=torch.IntTensor(cu_seqlens).cuda(),
            max_seqlen_q=max(seqlens),
            max_seqlen_kv=max(seqlens),
            qkv_format='thd',
        )
        hidden_states = torch.randn(
            (total_tokens, 1, baseline.config.hidden_size), dtype=torch.bfloat16, device='cuda'
        )
    else:
        packed_seq_params = None
        hidden_states = torch.randn(
            (1024, 2, baseline.config.hidden_size), dtype=torch.bfloat16, device='cuda'
        )
    grads = torch.randn_like(hidden_states)

    def quantization_context(config):
        # Mirrors the dispatch in TransformerBlock.
        return get_fp8_context(config) if config.fp8 else get_fp4_context(config)

    def run(model):
        with quantization_context(model.config):
            output, _ = model(
                hidden_states, attention_mask=None, packed_seq_params=packed_seq_params
            )
        output.backward(grads)
        return output, {name: param.grad for name, param in model.named_parameters()}

    # NVFP4 draws randomness in the backward (stochastic rounding), so the two runs have
    # to start from the same RNG state to be comparable at all. Same ritual as
    # BitExactRunner._two_runs.
    rng_state = capture_rng_state()
    baseline_output, baseline_grads = run(baseline)
    restore_rng_state(rng_state)
    recomputed_output, recomputed_grads = run(recomputed)

    torch.testing.assert_close(recomputed_output, baseline_output, atol=0, rtol=0)

    assert recomputed_grads.keys() == baseline_grads.keys()
    for name, baseline_grad in baseline_grads.items():
        assert baseline_grad is not None, f"{name} has no gradient"
        recomputed_grad = recomputed_grads[name]
        assert recomputed_grad is not None, f"{name} has no gradient with up-proj recompute"
        torch.testing.assert_close(
            recomputed_grad, baseline_grad, atol=0, rtol=0, msg=lambda m, n=name: f"{n}: {m}"
        )

    # The assertions above hold trivially if the recipe never engaged, since two bf16
    # replays also match bitwise. Run the same weights through a bf16 module: the
    # quantized output must track it closely (the math is right) yet differ from it
    # (the values really went through quantization).
    bf16_model = _build_absorbed_mla(
        get_mock_mla_config(
            tensor_model_parallel_size=1, context_parallel_size=1, qk_layernorm=True
        )
    )
    # Copy parameters directly rather than via state_dict, which also carries the
    # quantization `_extra_state` blobs that a bf16 module has no use for.
    baseline_params = dict(baseline.named_parameters())
    with torch.no_grad():
        for name, param in bf16_model.named_parameters():
            param.copy_(baseline_params[name])
        bf16_output, _ = bf16_model(
            hidden_states, attention_mask=None, packed_seq_params=packed_seq_params
        )

    assert not torch.equal(
        recomputed_output, bf16_output
    ), "quantized output is bitwise equal to bf16, so the recipe never engaged"
    # The bound stays loose on purpose: it only has to catch a quantized path that
    # produces garbage, and it must hold across recipes and architectures whose
    # quantization error differs by an order of magnitude — hence the separate, coarser
    # bound for 4-bit elements. Accuracy itself is pinned by the exact comparisons above
    # and by test_functionality.
    threshold = 0.9 if quant_overrides.get("fp4") else 0.99
    cosine_sim = torch.nn.functional.cosine_similarity(
        recomputed_output.flatten().float().unsqueeze(0), bf16_output.flatten().float().unsqueeze(0)
    ).item()
    assert (
        cosine_sim > threshold
    ), f"{'FP8' if quant_overrides.get('fp8') else 'FP4'} quantized output diverges from bf16: cosine similarity = {cosine_sim}"

    Utils.destroy_model_parallel()
