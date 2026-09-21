# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Real-weight smoke of the production Magi protocol on a Truncated-M3 checkpoint (bf16).

Optional: ``MINIMAX_M3_TRUNCATED_DIR`` must point at a text-only slice of ``MiniMaxAI/MiniMax-M3``
(embed + first layers + norm + head, HF-module spelling). A packed two-document batch must yield the
same logits as the two documents run one at a time: Magi's padding, dispatch and cu_seqlens handling
must not let information cross document boundaries.
"""

from __future__ import annotations

import os

import pytest
import torch

pytestmark = [pytest.mark.gpus(1, min_architecture="blackwell"), pytest.mark.optional]
DEV = "cuda"
CKPT = os.environ.get("MINIMAX_M3_TRUNCATED_DIR", "")
DOC_LENGTHS = (384, 320)


@pytest.mark.skipif(not CKPT, reason="set MINIMAX_M3_TRUNCATED_DIR")
def test_truncated_m3_packed_logits_match_individual_documents():
    from megatron.lite.model.minimax_m3.lite import protocol as P
    from megatron.lite.primitive.kernels import magi_msa
    from megatron.lite.runtime.contracts import ParallelConfig
    from megatron.lite.runtime.contracts.data import PackedBatch

    torch.cuda.set_device(0)
    magi_msa.ensure_single_process_group()
    cfg = P.build_model_config(CKPT)
    bundle = P.build_model(cfg, impl_cfg=P.ImplConfig(parallel=ParallelConfig(), optimizer=None, magi_chunk_size=512))
    model = bundle.chunks[0]
    P.load_hf_weights(model, CKPT, cfg, bundle.parallel_state)
    model.eval()

    generator = torch.Generator(device=DEV).manual_seed(20260918)
    documents = [torch.randint(0, cfg.vocab_size, (length,), device=DEV, generator=generator) for length in DOC_LENGTHS]

    def forward_docs(docs):
        batch = PackedBatch(input_ids=torch.cat(docs), labels=None, seq_lens=torch.tensor([d.numel() for d in docs], device=DEV))
        with torch.no_grad():
            output = bundle.forward_step(model, batch)
            unpacked = P.unpack_forward_output(model, batch, output["logits"])
        return [piece.detach().cpu() for piece in unpacked.unbind()]

    individual = [forward_docs([doc])[0] for doc in documents]
    packed = forward_docs(documents)
    for i, (got, want) in enumerate(zip(packed, individual, strict=True)):
        cosine = torch.nn.functional.cosine_similarity(got.float(), want.float(), dim=-1).mean().item()
        mean_abs = (got.float() - want.float()).abs().mean().item()
        print(f"truncated_m3_packed_vs_individual doc={i} len={DOC_LENGTHS[i]} cos={cosine:.6f} mean_abs={mean_abs:.3e}")
        assert torch.isfinite(got.float()).all()
        assert cosine >= 0.9999 and mean_abs < 2e-2
