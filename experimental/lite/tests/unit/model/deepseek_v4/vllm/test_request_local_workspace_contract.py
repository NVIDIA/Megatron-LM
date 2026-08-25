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
