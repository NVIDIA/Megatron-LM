# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Packed DSv4 CP parity against per-document SBHD; run on at least 4 GPUs."""

from copy import copy
from dataclasses import replace

import pytest
import torch
import torch.distributed as dist

from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.experimental_attention_variant.test_dsv4_hybrid_attention import (
    HAVE_HADAMARD,
    _build_attention,
    _make_config,
)

pytestmark = pytest.mark.launch_on_gb200


class _CP1:
    @staticmethod
    def size():
        return 1

    @staticmethod
    def rank():
        return 0


def _assert_match(actual, expected):
    assert actual.shape == expected.shape
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    # BF16 GEMMs change their reduction tiling when CP changes the token count.
    # Bound both relative RMS error and normalized direction, including tiny grads.
    delta = (actual.float() - expected.float()).square().sum()
    energy = expected.float().square().sum().clamp_min(1e-12)
    assert delta / energy < 2e-3
    torch.testing.assert_close(actual, expected, atol=0.06, rtol=0.06)


def _reference_per_document(model, hidden, physical, real):
    """Use the existing native SBHD backend without packed layout or CP helpers."""
    outputs = []
    for i, (start, end) in enumerate(zip(physical, physical[1:])):
        length = real[i + 1] - real[i]
        if length:
            output, _ = model(hidden[start : start + length], attention_mask=None)
            outputs.append(output)
        if start + length < end:
            # Keep padding gradients explicitly zero in the reference input.
            outputs.append(hidden[start + length : end] * 0)
    return torch.cat(outputs, dim=0)


@pytest.mark.skipif(
    not (torch.cuda.is_available() and HAVE_TE and HAVE_HADAMARD),
    reason="needs CUDA, TE and the real Hadamard kernel",
)
@pytest.mark.parametrize("cp_size", [2, 4])
@pytest.mark.parametrize(
    "ratio,sparse,coeff,recompute",
    [
        (0, True, 0.0, False),
        (4, True, 0.2, True),
        (4, False, 0.0, False),
        (4, False, 0.2, False),
        (128, True, 0.0, True),
    ],
)
def test_packed_cp_matches_full_attention_and_gradients(cp_size, ratio, sparse, coeff, recompute):
    if Utils.world_size < cp_size:
        pytest.skip(f"requires {cp_size} ranks")
    pytest.importorskip("flash_mla")
    pytest.importorskip("cudnn.deepseek_sparse_attention")
    # CI builds FlashMLA with FLASH_MLA_DISABLE_SM90=1.
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("Fused CSA CP tests require the SM100 kernels included in the CI image")
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, context_parallel_size=cp_size
    )
    try:
        pg = ProcessGroupCollection.use_mpu_process_groups()
        ref_pg = copy(pg)
        ref_pg.cp = _CP1()
        torch.manual_seed(198)
        model_parallel_cuda_manual_seed(198)
        # A rank boundary cuts a document and a ratio-4 group. Real lengths
        # exclude internal padding; a repeated prefix exercises empty documents.
        physical_cu = [0, 133, 133, 1157, 2048]
        real_cu = [0, 129, 129, 1141, 2016]
        cfg = _make_config(
            num_layers=1,
            hidden_size=256,
            num_attention_heads=64,
            v_head_dim=512,
            qk_pos_emb_head_dim=64,
            q_lora_rank=128,
            csa_compress_ratios=[ratio],
            csa_window_size=128,
            dsa_indexer_n_heads=64,
            dsa_indexer_head_dim=128,
            dsa_indexer_topk=512,
            dsa_indexer_loss_coeff=coeff,
            dsa_indexer_use_sparse_loss=sparse,
            dsa_kernel_backend="cudnn",
            context_parallel_size=cp_size,
            attention_cp_layout="contiguous",
            linear_cp_layout="contiguous",
            qk_layernorm=True,
            apply_rope_fusion=True,
            recompute_granularity="selective" if recompute else None,
            recompute_modules=["mla_up_proj"] if recompute else [],
        )
        # Keep RoPE arithmetic identical. Sum each document's auxiliary loss
        # with one global divisor, matching the packed mean over real tokens.
        ref_cfg = replace(
            cfg,
            context_parallel_size=1,
            dsa_kernel_backend="none",
            calculate_per_token_loss=True,
            dsa_indexer_loss_coeff=coeff / real_cu[-1],
        )
        model = _build_attention(cfg, 1, pg).cuda()
        reference = _build_attention(ref_cfg, 1, ref_pg).cuda()
        reference.load_state_dict(model.state_dict())
        physical = torch.tensor(physical_cu, dtype=torch.int32, device="cuda")
        real = torch.tensor(real_cu, dtype=torch.int32, device="cuda")
        packed = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=real,
            cu_seqlens_kv=real,
            cu_seqlens_q_padded=physical,
            cu_seqlens_kv_padded=physical,
            max_seqlen_q=1024,
            max_seqlen_kv=1024,
        )
        whole = torch.randn(2048, 1, 256, dtype=torch.bfloat16, device="cuda")
        count = whole.shape[0] // cp_size
        rows = slice(pg.cp.rank() * count, (pg.cp.rank() + 1) * count)
        local = whole[rows].clone().requires_grad_()
        whole = whole.requires_grad_()
        output, _ = model(local, attention_mask=None, packed_seq_params=packed)
        expected = _reference_per_document(reference, whole, physical_cu, real_cu)
        _assert_match(output, expected[rows])
        grad = torch.randn_like(expected)
        output.backward(grad[rows])
        expected.backward(grad)
        _assert_match(local.grad, whole.grad[rows])
        ref_params = dict(reference.named_parameters())
        for name, param in model.named_parameters():
            assert param.grad is not None, name
            # Avoid extra BF16 rounding when summing CP rank contributions.
            total_grad = param.grad.detach().to(dtype=torch.float32, copy=True)
            dist.all_reduce(total_grad, group=pg.cp)
            try:
                _assert_match(total_grad, ref_params[name].grad.float())
            except AssertionError as error:
                raise AssertionError(f"Parameter gradient mismatch: {name}\n{error}") from error
            if ratio == 4 and coeff == 0 and ".indexer." in name:
                assert torch.count_nonzero(total_grad) == 0
    finally:
        Utils.destroy_model_parallel()
