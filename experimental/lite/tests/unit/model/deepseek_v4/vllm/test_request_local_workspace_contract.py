from __future__ import annotations

import inspect


def test_attention_gathers_workspace_inside_request_loop() -> None:
    from megatron.lite.model.deepseek_v4.vllm.primitive.attention.module import (
        VLLMAttention,
    )

    source = inspect.getsource(VLLMAttention._forward_training_attention)
    assert "request_workspace = workspace.index_select" not in source
    assert source.index("for seq_idx in range") < source.index(
        "local_workspace = workspace.index_select"
    )
    assert "torch.unique" not in source
    assert "torch.searchsorted" not in source


def test_indexer_uses_loaded_deterministic_topk_in_bi_mode() -> None:
    from megatron.lite.model.deepseek_v4.vllm.primitive.attention.runtime import (
        _top_k_per_row_prefill,
        official_indexer_topk,
    )

    wrapper = inspect.getsource(_top_k_per_row_prefill)
    indexer = inspect.getsource(official_indexer_topk)
    assert "envs.VLLM_BATCH_INVARIANT" in wrapper
    assert 'os.environ.get("DS4_BI_TOPK_LIB")' in wrapper
    assert "torch.ops.load_library(library)" in wrapper
    assert "torch.ops.ds4_bi.top_k_per_row_prefill" in wrapper
    assert "requires the loaded DS4 deterministic" in wrapper
    assert "_top_k_per_row_prefill(" in indexer
    assert "ops.top_k_per_row_prefill(" not in indexer
