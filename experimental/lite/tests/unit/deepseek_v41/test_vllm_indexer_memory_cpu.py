# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import sys
from types import SimpleNamespace

import torch


def test_indexer_diagnostics_are_lazy_and_preserve_topk(monkeypatch):
    from torch.utils._python_dispatch import TorchDispatchMode

    from megatron.lite.model.deepseek_v41.vllm import compressed

    index_topk = compressed.index_topk

    logits = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    q = torch.zeros(4, 32, 128, dtype=torch.bfloat16)
    w = torch.ones(4, 32)
    keys = (torch.zeros(4, 64, dtype=torch.uint8), torch.zeros(4, 4, dtype=torch.uint8))

    def topk(values, starts, ends, out, *args):
        assert torch.equal(values, logits[ends.long() - 1])
        out[:, :2] = values.topk(2).indices

    mocks = {
        'vllm.model_executor.kernels.batch_invariant': dict(top_k_per_row_prefill_index_order=topk),
        'vllm.model_executor.layers.sparse_attn_indexer': dict(as_fp32_logits=lambda x, _: x),
        'vllm.models.deepseek_v41.common.ops': dict(
            fused_indexer_q_rope_quant=lambda *a, **kw: ((q.view(torch.uint8), w), w)
        ),
        'vllm.utils.deep_gemm': dict(
            fp8_fp4_mqa_logits=lambda q, k, w, starts, ends, width: logits[ends.long() - 1],
            mqa_logits_mx_inputs=lambda q, s, w: (s, w),
        ),
    }
    for name, values in mocks.items():
        monkeypatch.setitem(sys.modules, name, SimpleNamespace(**values))

    class MaskedMaterialization(TorchDispatchMode):
        count = 0

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func == torch.ops.aten.index.Tensor and any(
                t is not None and t.dtype == torch.bool for t in args[1]
            ):
                self.count += 1
            return func(*args, **(kwargs or {}))

    events = {}
    with MaskedMaterialization() as quiet:
        actual = index_topk(q, w, keys, None, 1, None)
    with MaskedMaterialization() as observed:
        expected = index_topk(q, w, keys, None, 1, events.__setitem__)
    assert quiet.count == 0 and observed.count == 1
    assert torch.equal(actual, expected)
    monkeypatch.setattr(compressed, 'INDEXER_LOGITS_BYTES', 32)
    chunks = []

    def candidates(values, starts, ends, *, offset):
        chunks.append((offset, values.shape[0]))

    with MaskedMaterialization() as bounded:
        chunked = index_topk(q, w, keys, None, 1, None, candidates)
    assert bounded.count == 0 and chunks == [(0, 2), (2, 2)]
    assert torch.equal(chunked, expected)
    index_topk(q, w, keys, None, 1, events.__setitem__)
    assert torch.equal(events['indexer_logits_valid'], logits[torch.tril(torch.ones(4, 4)).bool()])
