# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CP1/CP2 parity for opt-in mixed-document QSA routing with padded THD metadata."""

import hashlib
import os
from pathlib import Path

import pytest
import torch

from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_qsa_module_spec_for_backend,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.ssm.mamba_context_parallel import reconstruct_tensor_cp, split_tensor_cp
from megatron.core.transformer.spec_utils import build_module
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.experimental_attention_variant.test_attention_variant_qsa import (
    _make_config,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_qsa_mixed_padded_empty_documents_cp1_cp2_dense_id_parity():
    world = int(os.environ.get("WORLD_SIZE", "1"))
    assert world in (1, 2)
    Utils.initialize_model_parallel(1, 1, context_parallel_size=world)
    torch.cuda.set_per_process_memory_fraction(0.1)
    os.environ["NVTE_FLASH_ATTN"] = "0"
    os.environ["NVTE_FUSED_ATTN"] = "0"
    os.environ["NVTE_UNFUSED_ATTN"] = "1"
    try:
        torch.manual_seed(137)
        config = _make_config(context_parallel_size=world, mrope_section=[2, 1, 1])
        attention = (
            build_module(get_qsa_module_spec_for_backend(config), config=config, layer_number=1)
            .cuda()
            .eval()
        )
        cu = torch.tensor([0, 0, 94, 94, 108], dtype=torch.int32, device="cuda")
        cu_padded = torch.tensor([0, 0, 96, 96, 112], dtype=torch.int32, device="cuda")
        packed = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu,
            cu_seqlens_kv=cu,
            cu_seqlens_q_padded=cu_padded,
            cu_seqlens_kv_padded=cu_padded,
            max_seqlen_q=96,
            max_seqlen_kv=96,
        )
        torch.manual_seed(151)
        global_hidden = torch.randn(112, 1, config.hidden_size, dtype=torch.bfloat16, device="cuda")
        global_freqs = torch.randn(112, 1, 1, 8, device="cuda")
        upstream = torch.randn_like(global_hidden)
        hidden_base = split_tensor_cp(global_hidden, packed, dim=0)
        freqs = split_tensor_cp(global_freqs, packed, dim=0)
        upstream_local = split_tensor_cp(upstream, packed, dim=0)

        def run(backend):
            attention.core_attention.sparse_backend = backend
            hidden = hidden_base.detach().clone().requires_grad_()
            output, _ = attention(
                hidden, attention_mask=None, rotary_pos_emb=freqs, packed_seq_params=packed
            )
            selection = attention.core_attention._selection
            assert selection.all_selected is False
            assert selection.doc_ids[0].unique().numel() == 2
            grads = torch.autograd.grad(
                output, (hidden, attention.linear_qkv.weight), upstream_local
            )
            return output.detach(), tuple(grad.detach() for grad in grads)

        dense, ids = run("dense_masked"), run("id_sparse")
        torch.testing.assert_close(ids[0].float(), dense[0].float(), atol=6e-2, rtol=6e-2)
        for actual, expected in zip(ids[1], dense[1]):
            torch.testing.assert_close(actual.float(), expected.float(), atol=8e-2, rtol=8e-2)

        save_dir = os.environ.get("QSA_MIXED_CP_SAVE_DIR")
        if save_dir:
            directory = Path(save_dir)
            directory.mkdir(parents=True, exist_ok=True)
            rank = int(os.environ.get("RANK", "0"))
            digest = hashlib.sha256(
                attention.linear_qkv.weight.detach()
                .contiguous()
                .cpu()
                .view(torch.uint8)
                .numpy()
                .tobytes()
            ).hexdigest()
            for backend, result in (("dense", dense), ("id", ids)):
                torch.save(
                    {
                        "output": reconstruct_tensor_cp(result[0], packed, dim=0).cpu(),
                        "hidden_grad": reconstruct_tensor_cp(result[1][0], packed, dim=0).cpu(),
                        "weight_grad": result[1][1].cpu(),
                        "weight_sha": digest,
                    },
                    directory / f"cp{world}-{backend}-rank{rank}.pt",
                )
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.skipif(
    not os.environ.get("QSA_MIXED_CP_SAVE_DIR"), reason="requires saved parity tensors"
)
def test_qsa_mixed_padded_empty_documents_saved_cp1_cp2_parity():
    directory = Path(os.environ["QSA_MIXED_CP_SAVE_DIR"])
    for backend in ("dense", "id"):
        cp1 = torch.load(
            directory / f"cp1-{backend}-rank0.pt", map_location="cpu", weights_only=True
        )
        cp2 = [
            torch.load(
                directory / f"cp2-{backend}-rank{rank}.pt", map_location="cpu", weights_only=True
            )
            for rank in (0, 1)
        ]
        assert cp1["weight_sha"] == cp2[0]["weight_sha"] == cp2[1]["weight_sha"]
        for rank in (0, 1):
            torch.testing.assert_close(cp2[rank]["output"], cp1["output"], atol=0, rtol=0)
            torch.testing.assert_close(
                cp2[rank]["hidden_grad"].float(), cp1["hidden_grad"].float(), atol=6e-2, rtol=6e-2
            )
        torch.testing.assert_close(
            (cp2[0]["weight_grad"] + cp2[1]["weight_grad"]).float(),
            cp1["weight_grad"].float(),
            atol=8e-2,
            rtol=8e-2,
        )
