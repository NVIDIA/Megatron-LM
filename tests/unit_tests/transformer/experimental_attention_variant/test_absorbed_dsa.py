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
from megatron.core.transformer.experimental_attention_variant.dsa import (
    DSAIndexer,
    DSAIndexerSubmodules,
    DSAttention,
    DSAttentionSubmodules,
)
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.multi_latent_attention import (
    MLASelfAttention,
    MLASelfAttentionSubmodules,
)
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.utils import init_method_normal, scaled_init_method_normal
from tests.unit_tests.test_utilities import Utils

SEQ_LEN = 4096
MBS = 1
HIDDEN = 7168


def get_mock_mla_config(
    tensor_model_parallel_size: int, context_parallel_size: int
) -> SimpleNamespace:
    """Create test config with all attributes used in MLA."""
    return SimpleNamespace(
        multi_latent_attention=True,
        hidden_size=HIDDEN,
        num_attention_heads=128,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_head_dim=128,
        qk_pos_emb_head_dim=64,
        v_head_dim=128,
        num_layers=1,
        dsa_indexer_n_heads=64,
        dsa_indexer_head_dim=128,
        dsa_indexer_topk=2048,
        dsa_indexer_loss_coeff=100000.0,
        dsa_indexer_use_sparse_loss=True,
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
        original_max_position_embeddings=SEQ_LEN,
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
    core_attention = ModuleSpec(
        module=DSAttention,
        submodules=DSAttentionSubmodules(
            indexer=ModuleSpec(
                module=DSAIndexer,
                submodules=DSAIndexerSubmodules(
                    linear_wq_b=backend.linear(),
                    linear_wk=backend.linear(),
                    k_norm=backend.layer_norm(rms_norm=False, for_qk=True),
                    linear_weights_proj=backend.linear(),
                ),
            )
        ),
    )
    return AbsorbedMLASelfAttentionSubmodules(
        linear_q_proj=backend.column_parallel_linear(),
        linear_q_down_proj=linear_q_down_proj,
        linear_q_up_proj=backend.column_parallel_linear(),
        linear_kv_down_proj=linear_kv_down_proj,
        linear_k_up_proj=backend.column_parallel_linear(),
        linear_v_up_proj=backend.column_parallel_linear(),
        core_attention=core_attention,
        linear_proj=backend.row_parallel_linear(),
        q_layernorm=qk_norm,
        kv_layernorm=qk_norm,
    )


@pytest.mark.parametrize("tp_cp", [[1, 1]])
@pytest.mark.parametrize("qkv_format", ['sbhd'])
@pytest.mark.parametrize("down_proj_use_column_parallel", [False])
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
    absorbed_mla = AbsorbedMLASelfAttention(
        config=config,
        submodules=absorbed_submodules,
        layer_number=0,
        attn_mask_type=AttnMaskType.causal,
        cp_comm_type="all_gather" if cp_size > 1 else None,
        pg_collection=None,
    ).cuda()

    seqlen = SEQ_LEN // tp_size // cp_size
    hidden_states = torch.randn((seqlen, MBS, HIDDEN), dtype=torch.bfloat16, device='cuda')
    grads = torch.randn_like(hidden_states)
    packed_seq_params = None

    # Fused version
    absorbed_mla.core_attention.force_unfused_dsa = False
    for name, param in absorbed_mla.named_parameters():
        if param.grad is not None:
            param.grad.zero_()
    absorbed_outputs, _ = absorbed_mla(
        hidden_states, attention_mask=None, packed_seq_params=packed_seq_params
    )
    absorbed_outputs.backward(grads)
    results_1 = {}
    results_1['absorbed_outputs'] = absorbed_outputs.clone()
    for name, param in absorbed_mla.named_parameters():
        results_1[name] = param.grad.clone()

    # Unfused version
    absorbed_mla.core_attention.force_unfused_dsa = True
    for name, param in absorbed_mla.named_parameters():
        if param.grad is not None:
            param.grad.zero_()
    absorbed_outputs, _ = absorbed_mla(
        hidden_states, attention_mask=None, packed_seq_params=packed_seq_params
    )
    absorbed_outputs.backward(grads)
    results_2 = {}
    results_2['absorbed_outputs'] = absorbed_outputs.clone()
    for name, param in absorbed_mla.named_parameters():
        results_2[name] = param.grad.clone()

    for name in results_1:
        t1 = results_1[name].flatten().float().unsqueeze(0)
        t2 = results_2[name].flatten().float().unsqueeze(0)
        cosine_sim = torch.nn.functional.cosine_similarity(t1, t2).item()
        assert cosine_sim > 0.9999, f"{name} cosine similarity: {cosine_sim} < 0.9999"
        print(f"{name} cosine similarity: {cosine_sim}")

    Utils.destroy_model_parallel()


@pytest.mark.parametrize("tp_cp", [[1, 1]])
@pytest.mark.parametrize("qkv_format", ['sbhd'])
@pytest.mark.parametrize("down_proj_use_column_parallel", [False])
def test_perf(tp_cp: List[int], qkv_format: str, down_proj_use_column_parallel: bool):
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
    absorbed_mla = AbsorbedMLASelfAttention(
        config=config,
        submodules=absorbed_submodules,
        layer_number=0,
        attn_mask_type=AttnMaskType.causal,
        cp_comm_type="all_gather" if cp_size > 1 else None,
        pg_collection=None,
    ).cuda()

    seqlen = SEQ_LEN // tp_size // cp_size
    hidden_states = torch.randn((seqlen, MBS, HIDDEN), dtype=torch.bfloat16, device='cuda')
    grads = torch.randn_like(hidden_states)
    packed_seq_params = None

    # absorbed_mla.core_attention.force_unfused_dsa = True
    # Forward & Backward
    for _ in range(10):
        for name, param in absorbed_mla.named_parameters():
            if param.grad is not None:
                param.grad.zero_()
        absorbed_outputs, _ = absorbed_mla(
            hidden_states, attention_mask=None, packed_seq_params=packed_seq_params
        )
        absorbed_outputs.backward(grads)

    print(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"CUDA memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    print(f"CUDA max memory allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    print(f"CUDA max memory reserved: {torch.cuda.max_memory_reserved() / 1024**3:.2f} GB")

    Utils.destroy_model_parallel()


if __name__ == "__main__":

    # test_functionality(tp_cp=[1, 1], qkv_format='sbhd', down_proj_use_column_parallel=False)

    test_perf(tp_cp=[1, 1], qkv_format='sbhd', down_proj_use_column_parallel=False)
