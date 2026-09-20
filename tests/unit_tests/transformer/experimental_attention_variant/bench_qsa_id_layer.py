# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""One-layer QSA selected-ID length gate; intentionally no dense reference."""

import json
import os
import time

import torch

from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_qsa_module_spec_for_backend,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.spec_utils import build_module
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.experimental_attention_variant.test_attention_variant_qsa import (
    _make_config,
)


def main():
    seq_len = int(os.environ['QSA_BENCH_SEQ_LEN'])
    torch.cuda.set_per_process_memory_fraction(
        float(os.environ.get('QSA_BENCH_MEM_FRACTION', '0.7'))
    )
    os.environ['NVTE_FLASH_ATTN'] = '0'
    os.environ['NVTE_FUSED_ATTN'] = '0'
    os.environ['NVTE_UNFUSED_ATTN'] = '1'
    Utils.initialize_model_parallel(1, 1)
    try:
        torch.manual_seed(137)
        config = _make_config(
            hidden_size=2560,
            num_attention_heads=24,
            num_query_groups=2,
            kv_channels=256,
            qsa_indexer_n_heads=4,
            qsa_indexer_kv_heads=1,
            qsa_indexer_head_dim=128,
            qsa_indexer_budget=2048,
            qsa_indexer_compress_ratio=4,
            mrope_section=[11, 11, 10],
        )
        attention = (
            build_module(get_qsa_module_spec_for_backend(config), config=config, layer_number=1)
            .cuda()
            .eval()
        )
        attention.core_attention.sparse_backend = 'id_sparse'
        cu = torch.tensor([0, seq_len], dtype=torch.int32, device='cuda')
        packed = PackedSeqParams(
            qkv_format='thd',
            cu_seqlens_q=cu,
            cu_seqlens_kv=cu,
            cu_seqlens_q_padded=cu,
            cu_seqlens_kv_padded=cu,
            max_seqlen_q=seq_len,
            max_seqlen_kv=seq_len,
        )
        hidden = torch.randn(
            seq_len, 1, config.hidden_size, dtype=torch.bfloat16, device='cuda'
        ).requires_grad_()
        frequencies = torch.randn(seq_len, 1, 1, 64, dtype=torch.float32, device='cuda')
        torch.cuda.synchronize()
        initial_allocated = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        with torch.no_grad():
            selection = attention.indexer(hidden, frequencies, packed, output_format='ids')
        torch.cuda.synchronize()
        route_seconds = time.perf_counter() - t0
        route_peak = torch.cuda.max_memory_allocated()
        route_shape = list(selection.selected_ids.shape)
        is_sparse = not selection.all_selected
        del selection
        torch.cuda.reset_peak_memory_stats()
        t1 = time.perf_counter()
        output, _ = attention(
            hidden, attention_mask=None, rotary_pos_emb=frequencies, packed_seq_params=packed
        )
        torch.cuda.synchronize()
        forward_seconds = time.perf_counter() - t1
        forward_peak = torch.cuda.max_memory_allocated()
        t2 = time.perf_counter()
        loss = output.float().square().mean()
        loss.backward()
        torch.cuda.synchronize()
        backward_seconds = time.perf_counter() - t2
        parameter_gradients = {
            name: parameter.grad
            for name, parameter in attention.named_parameters()
            if parameter.grad is not None
        }
        missing_parameter_gradients = [
            name for name, parameter in attention.named_parameters() if parameter.grad is None
        ]
        print(
            json.dumps(
                {
                    'seq_len': seq_len,
                    'geometry': 'BF16 Hq24 Hkv2 D256 K512 R4 packed single doc mRoPE64',
                    'route_shape': route_shape,
                    'is_sparse': is_sparse,
                    'route_seconds': route_seconds,
                    'forward_seconds': forward_seconds,
                    'backward_seconds': backward_seconds,
                    'initial_allocated_gib': initial_allocated / (1024**3),
                    'route_peak_gib': route_peak / (1024**3),
                    'forward_peak_gib': forward_peak / (1024**3),
                    'peak_allocated_gib': torch.cuda.max_memory_allocated() / (1024**3),
                    'peak_reserved_gib': torch.cuda.max_memory_reserved() / (1024**3),
                    'finite_output': bool(torch.isfinite(output).all().item()),
                    'finite_hidden_grad': bool(torch.isfinite(hidden.grad).all().item()),
                    'nonzero_hidden_grad': int(torch.count_nonzero(hidden.grad).item()),
                    'finite_parameter_gradients': all(
                        bool(torch.isfinite(gradient).all().item())
                        for gradient in parameter_gradients.values()
                    ),
                    'nonzero_parameter_gradient_tensors': sum(
                        bool(torch.count_nonzero(gradient).item())
                        for gradient in parameter_gradients.values()
                    ),
                    'parameter_gradient_tensors': len(parameter_gradients),
                    'missing_parameter_gradients': missing_parameter_gradients,
                }
            ),
            flush=True,
        )
    finally:
        Utils.destroy_model_parallel()


if __name__ == '__main__':
    main()
