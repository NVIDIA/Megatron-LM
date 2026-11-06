# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Single small native-Core MoE vs fused MegaMoE, one configuration per job."""
import json
import os
import time
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

pytestmark = [
    pytest.mark.gpus(1, min_architecture="blackwell"),
    pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA'),
]


def test_core_owned_nonfused_moe_matches_native_fused(tmp_path):
    from vllm.model_executor.layers.fused_moe.prepare_megamoe import prepare_megamoe_inputs
    from vllm.utils.deep_gemm import _import_deep_gemm

    from megatron.core.models.gpt.gpt_layer_specs import get_mlp_module_spec
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.lite.model.deepseek_v41.lite.components.config import core_groups
    from megatron.lite.model.deepseek_v41.vllm.experts import packed_scales, quantize_activation
    from megatron.lite.model.deepseek_v41.vllm.linear import linear
    from megatron.lite.model.deepseek_v41.vllm.moe import MegaMoE
    from megatron.lite.model.deepseek_v41.vllm.routing import route
    from megatron.lite.primitive.parallel import init_parallel
    from megatron.lite.primitive.quantization.block_fp8 import quantize_block_fp8
    from megatron.lite.primitive.quantization.mxfp4 import quantize_mxfp4
    from megatron.lite.runtime.contracts import ParallelConfig

    dist.init_process_group(
        'nccl', init_method=f'file://{tmp_path / "group"}', rank=0, world_size=1
    )
    ps = init_parallel(ParallelConfig())
    model_parallel_cuda_manual_seed(
        73,
        tp_rank=0,
        ep_rank=0,
        etp_rank=0,
        gtp_remat_rank=0,
        egtp_remat_rank=0,
        gtp_remat_world_size=1,
        egtp_remat_world_size=1,
    )
    h, i, e, k, n = 1024, 512, 16, 6, 7
    cfg = TransformerConfig(
        num_layers=1,
        hidden_size=h,
        num_attention_heads=8,
        ffn_hidden_size=i,
        num_moe_experts=e,
        moe_ffn_hidden_size=i,
        moe_shared_expert_intermediate_size=i,
        moe_router_topk=k,
        moe_router_score_function='sqrtsoftplus',
        moe_router_pre_softmax=False,
        moe_router_topk_scaling_factor=2.5,
        moe_router_enable_expert_bias=True,
        moe_router_load_balancing_type='none',
        moe_grouped_gemm=True,
        gated_linear_unit=True,
        activation_func=torch.nn.functional.silu,
        activation_func_clamp_value=10.0,
        add_bias_linear=False,
        params_dtype=torch.bfloat16,
        bf16=True,
        perform_initialization=False,
    )
    spec = get_mlp_module_spec(use_te=True, num_experts=e, moe_grouped_gemm=True)
    model = MegaMoE(
        cfg,
        submodules=spec.keywords['submodules'],
        layer_number=1,
        pg_collection=core_groups(ps, SimpleNamespace(cp_size=1)),
        hash_moe_layer_threshold=0,
    )
    torch.manual_seed(73)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_((torch.randn_like(parameter.float()) * 0.02).to(parameter.dtype))
    x = torch.randn(n, 1, h, device='cuda', dtype=torch.bfloat16) * 0.3
    dg = _import_deep_gemm()
    dg.use_deterministic_algorithms(True)

    def fp4(fc):
        items = [quantize_mxfp4(getattr(fc, f'weight{index}').detach()) for index in range(e)]
        codes = torch.stack([item[0] for item in items])
        sf = torch.stack([item[1] for item in items]).view(torch.uint8)
        sf = (sf.to(torch.int32) << 23).view(torch.float32)
        return codes, dg.transform_sf_into_required_layout(
            sf, codes.shape[1], codes.shape[2] * 2, (1, 32), e
        )

    def fp8(weight):
        codes, sf = quantize_block_fp8(weight.detach(), (32, 32), scale_format='e8m0')
        raw = (sf.view(torch.uint8).repeat_interleave(32, 0).to(torch.int32) << 23).view(
            torch.float32
        )
        return codes, packed_scales(raw)

    weights = dg.transform_weights_for_mega_moe(
        fp4(model.experts.linear_fc1), fp4(model.experts.linear_fc2), activation='swiglu'
    )
    shared = dg.transform_weights_for_mega_moe(
        fp8(model.shared_experts.linear_fc1.weight),
        fp8(model.shared_experts.linear_fc2.weight),
        activation='swiglu',
    )
    buffer = dg.get_symm_buffer_for_mega_moe(
        dist.group.WORLD,
        e,
        2048,
        k,
        h,
        i,
        num_shared_experts=1,
        mma_type='fp8xfp4',
        activation='swiglu',
    )
    try:
        with torch.no_grad():
            scores, ids = route(
                linear(x.squeeze(1), model.router.weight, fp8=False), model.router, None
            )
            block = dg.get_block_m_for_mega_moe(
                1, e, buffer.num_max_tokens_per_rank, n, k, 'fp8xfp4'
            )
            prepare_megamoe_inputs(
                x.squeeze(1),
                scores,
                ids,
                buffer.x[:n],
                buffer.x_sf[:n],
                buffer.topk_idx[:n],
                buffer.topk_weights[:n],
                shared_x_sf=buffer.shared_l1_acts_sf,
                shared_block_m=block,
            )
            codes, scale = quantize_activation(x.squeeze(1))
            assert torch.equal(codes.view(torch.uint8), buffer.x[:n].view(torch.uint8))
            assert torch.equal(packed_scales(scale), buffer.x_sf[:n])
            reference = torch.empty_like(x.squeeze(1))
            dg.fp8_fp4_mega_moe(
                reference,
                *weights,
                buffer,
                shared_l1_weights=shared[0],
                shared_l2_weights=shared[1],
                activation='swiglu',
                activation_clamp=10.0,
                fast_math=True,
            )
            torch.cuda.synchronize()
            started = time.monotonic()
            output, bias = model(x)
            torch.cuda.synchronize()
            elapsed = time.monotonic() - started
            actual, expected = output.squeeze(1).view(torch.int16), reference.view(torch.int16)
            different = (actual != expected).nonzero()
            receipt = {
                'M': os.environ.get('W10_SOURCE_M'),
                'scope': 'random H1024 I512 E16 EP1 seven rows; native Core parameter owners; forward only, not C5',
                'different_elements': different.shape[0],
                'first_difference': different[:1].tolist(),
                'first_forward_seconds': elapsed,
                'peak_GiB': torch.cuda.max_memory_allocated() / 2**30,
            }
            (tmp_path / 'moe-result.json').write_text(json.dumps(receipt, indent=2))
            print(json.dumps(receipt), flush=True)
            assert bias is None and not different.numel(), receipt
    finally:
        buffer.destroy()
        dist.destroy_process_group()
