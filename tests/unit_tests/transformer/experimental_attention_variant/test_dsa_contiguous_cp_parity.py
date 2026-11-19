# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""A GLM-5.3-Flash DSA attention layer gives the same result in the zigzag and contiguous CP layouts.

Under ``cp_partition_mode='contiguous'`` (one global layout for a KDA + DSA hybrid) the
absorbed-MLA DSA layer converts its down-projected latents to zigzag, runs the zigzag DSA core,
and converts its output back. Every operation outside the conversions is per token, so the layer's
output and input gradient, mapped to global token positions, must equal those of the zigzag
layout. In particular the DSA indexer reads its keys, head weights and KPool gate from the layer's
hidden states and its queries from ``q_compressed``: both have to reach it in the same (zigzag)
row order.

The layer is the real hybrid ``dsa_layer`` attention of GLM-5.3-Flash at the official widths (NoPE,
KPool selection), on packed THD input of two documents, with CP 2 or 4.
"""

import os

import pytest
import torch

from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_glm53_stack_spec
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.experimental_attention_variant import dsa_layout
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_config import MLATransformerConfig
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")

_HIDDEN, _HEADS, _Q_RANK, _KV_RANK, _HEAD_DIM = 4096, 64, 1536, 512, 256
_INDEX_HEADS, _INDEX_DIM, _POOL, _TOPK = 32, 128, 4, 16
_EPS = 1e-5
# Two documents; lengths divisible by 2 * cp for CP 2 and 4.
_CU_SEQLENS = (0, 96, 256)


def _config(cp_size, backend, kpool_scorer):
    return MLATransformerConfig(
        num_layers=1,
        hidden_size=_HIDDEN,
        num_attention_heads=_HEADS,
        kv_channels=_HEAD_DIM,
        q_lora_rank=_Q_RANK,
        kv_lora_rank=_KV_RANK,
        qk_head_dim=_HEAD_DIM,
        qk_pos_emb_head_dim=0,
        v_head_dim=_HEAD_DIM,
        qk_layernorm=True,
        normalization='RMSNorm',
        layernorm_epsilon=_EPS,
        attention_latent_norm_epsilon=_EPS,
        rmsnorm_cast_before_weight=True,
        add_bias_linear=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        use_cpu_initialization=True,
        gradient_accumulation_fusion=False,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        attention_backend=AttnBackend.unfused if backend == 'none' else AttnBackend.auto,
        dsa_kernel_backend=backend,
        experimental_attention_variant='dsa',
        dsa_indexer_n_heads=_INDEX_HEADS,
        dsa_indexer_head_dim=_INDEX_DIM,
        dsa_indexer_topk=_TOPK,
        dsa_indexer_kpool=_POOL,
        dsa_indexer_kpool_use_quantization=False,
        dsa_indexer_kpool_always_select_tail=True,
        dsa_indexer_kpool_key_dtype_compression=True,
        dsa_indexer_kpool_scorer_backend=kpool_scorer,
        dsa_indexer_hf_scoring=True,
        dsa_indexer_loss_coeff=0.0,
        dsa_indexer_k_norm_epsilon=1e-6,
        dsa_indexer_scoring_relu=True,
        dsa_indexer_rotate_activation=False,
        dsa_indexer_qk_proj_use_quantization=False,
        dsa_indexer_weights_proj_use_quantization=False,
        dsa_indexer_weights_proj_output_dtype='bf16',
        mla_proj_use_quantization=False,
        rope_type='rope',
        rotary_scaling_factor=1.0,
        mscale_all_dim=0.0,
        kda_two_stage_gates=True,
        kda_beta_sigmoid_in_projection_dtype=True,
        enable_hyper_connections=True,
        mhc_norm_eps_inside_sqrt=True,
        mhc_normalize_before_projection=True,
        mhc_keep_mappings_in_fp32=False,
        mhc_keep_pre_in_fp32=True,
        mhc_learned_output_contract=False,
        context_parallel_size=cp_size,
        cp_comm_type='allgather',
        cp_partition_mode='contiguous',
        sequence_packing_scheduler='dp_balanced',
        max_seqlen_per_dp_cp_rank=_CU_SEQLENS[-1] // cp_size,
    )


def _packed(config, cp_group, mode):
    from megatron.training.utils.packed_seq_utils import prepare_packed_seq_params

    cu = torch.tensor(_CU_SEQLENS, dtype=torch.int32, device='cuda')
    longest = max(b - a for a, b in zip(_CU_SEQLENS[:-1], _CU_SEQLENS[1:]))
    params = PackedSeqParams(
        qkv_format='thd',
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu,
        cu_seqlens_q_padded=cu.clone(),
        cu_seqlens_kv_padded=cu.clone(),
        max_seqlen_q=longest,
        max_seqlen_kv=longest,
    )
    params.cp_group = cp_group
    params = prepare_packed_seq_params(params, config)
    params.cp_partition_mode = mode
    return params


def _local_positions(mode, cp_size, cp_rank):
    total = _CU_SEQLENS[-1]
    if mode == 'contiguous':
        rows = total // cp_size
        return torch.arange(cp_rank * rows, (cp_rank + 1) * rows, device='cuda')
    return dsa_layout.build_packed_allgather_cp_local_positions_from_host(
        list(_CU_SEQLENS), cp_size, cp_rank, torch.device('cuda')
    )


def _to_global(local, positions, cp_group):
    gathered_values = [torch.empty_like(local) for _ in range(cp_group.size())]
    gathered_positions = [torch.empty_like(positions) for _ in range(cp_group.size())]
    torch.distributed.all_gather(gathered_values, local.contiguous(), group=cp_group)
    torch.distributed.all_gather(gathered_positions, positions.contiguous(), group=cp_group)
    result = torch.empty(
        (_CU_SEQLENS[-1],) + tuple(local.shape[1:]), dtype=local.dtype, device=local.device
    )
    result[torch.cat(gathered_positions)] = torch.cat(gathered_values)
    return result


@pytest.mark.parametrize(('backend', 'kpool_scorer'), [('none', 'torch'), ('cudnn', 'cudnn')])
@pytest.mark.parametrize('cp_size', [2, 4])
def test_dsa_layer_contiguous_cp_matches_zigzag(cp_size, backend, kpool_scorer):
    world = int(os.environ.get('WORLD_SIZE', '1'))
    if world < cp_size:
        pytest.skip(f'needs {cp_size} ranks')
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=cp_size)
    try:
        model_parallel_cuda_manual_seed(5353)
        torch.manual_seed(5353)
        config = _config(cp_size, backend, kpool_scorer)
        groups = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])
        spec = hybrid_glm53_stack_spec(config).submodules.dsa_layer.submodules.self_attention
        layer = build_module(
            spec, config=config, layer_number=1, pg_collection=groups, cp_comm_type='allgather'
        ).cuda()
        cp_group, cp_rank = groups.cp, groups.cp.rank()
        # One global input, identical on every rank.
        generator = torch.Generator(device='cuda').manual_seed(7)
        hidden = torch.randn(
            _CU_SEQLENS[-1], 1, _HIDDEN, device='cuda', dtype=torch.bfloat16, generator=generator
        )
        grad_out = torch.randn(
            _CU_SEQLENS[-1], 1, _HIDDEN, device='cuda', dtype=torch.bfloat16, generator=generator
        )
        results = {}
        for mode in ('zigzag', 'contiguous'):
            positions = _local_positions(mode, cp_size, cp_rank)
            local = hidden[positions].detach().clone().requires_grad_(True)
            out, bias = layer(
                local, attention_mask=None, packed_seq_params=_packed(config, cp_group, mode)
            )
            assert bias is None
            out.backward(grad_out[positions])
            results[mode] = (
                _to_global(out.detach(), positions, cp_group),
                _to_global(local.grad.detach(), positions, cp_group),
            )
            layer.zero_grad(set_to_none=True)
    finally:
        Utils.destroy_model_parallel()

    (out_zz, grad_zz), (out_ct, grad_ct) = results['zigzag'], results['contiguous']
    assert torch.isfinite(out_zz).all() and torch.isfinite(grad_zz).all()
    torch.testing.assert_close(out_ct, out_zz, rtol=0, atol=0)
    torch.testing.assert_close(grad_ct, grad_zz, rtol=2e-2, atol=2e-3)
