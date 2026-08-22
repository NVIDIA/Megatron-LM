from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from megatron.lite.model.deepseek_v4.config import DeepseekV4Config
from megatron.lite.model.deepseek_v4.vllm.primitive.attention import runtime


def _config(**overrides) -> DeepseekV4Config:
    values = dict(
        head_dim=512,
        qk_rope_head_dim=64,
        num_attention_heads=64,
        sliding_window=128,
        compress_ratios=[1, 1, 4, 128],
        num_hidden_layers=4,
        max_position_embeddings=512,
    )
    values.update(overrides)
    return DeepseekV4Config(**values)


def test_packed_blocks_keep_requests_isolated() -> None:
    blocks = runtime._packed_blocks([5, 9], 4, device=torch.device("cpu"))
    assert blocks.block_table.tolist() == [[0, 1, -1], [2, 3, 4]]
    assert blocks.slot_mapping.tolist() == list(range(5)) + list(range(8, 17))


def test_packed_blocks_write_only_complete_compressor_groups() -> None:
    blocks = runtime._packed_blocks(
        [5, 9], 64, device=torch.device("cpu"), write_every=4
    )
    assert blocks.block_table.tolist() == [[0], [1]]
    assert blocks.slot_mapping.tolist() == [
        -1,
        -1,
        -1,
        0,
        -1,
        -1,
        -1,
        -1,
        64,
        -1,
        -1,
        -1,
        65,
        -1,
    ]


def test_builders_share_rope_by_ratio(monkeypatch) -> None:
    get_config = Mock(return_value=SimpleNamespace())
    build_rope = Mock(
        side_effect=lambda *_args, compress_ratio, **_kwargs: torch.full(
            (512, 128), compress_ratio, dtype=torch.float32
        )
    )
    monkeypatch.setattr(runtime, "get_config", get_config)
    monkeypatch.setattr(runtime, "_build_rope", build_rope)

    builders = runtime.build_attention_metadata_builders(
        "/model", _config(), (0, 1, 2, 3), torch.device("cpu")
    )

    get_config.assert_called_once_with("/model", trust_remote_code=True)
    assert build_rope.call_count == 3
    assert builders[0].cos_sin_cache is builders[1].cos_sin_cache
    assert builders[2].ratio == 4
    assert builders[3].ratio == 128


@pytest.mark.parametrize(
    ("layer_idx", "rows", "expected_tokens"),
    [(0, 37, 0), (2, 37, 64), (3, 5000, 5120)],
)
def test_training_metadata_has_only_local_packed_state(
    layer_idx: int, rows: int, expected_tokens: int
) -> None:
    packed = object()
    positions = torch.arange(rows, dtype=torch.int64)
    builder = runtime.AttentionMetadataBuilder(
        _config(),
        layer_idx=layer_idx,
        cos_sin_cache=torch.zeros(8192, 128, dtype=torch.float32),
    )
    metadata = builder.build(positions, packed)

    assert metadata.positions is positions
    assert metadata.packed_seq_params is packed
    assert not hasattr(metadata, "swa_cache")
    assert not hasattr(metadata, "kv_workspace")
    if expected_tokens == 0:
        assert metadata.compressor_metadata is None
    else:
        compressor = metadata.compressor_metadata
        assert compressor is not None
        assert compressor.token_to_req_indices.shape == (expected_tokens,)
        assert compressor.token_to_req_indices.count_nonzero().item() == 0


def _compressor_metadata(tokens: int = 4):
    return runtime.DS4CompressorMetadata(
        state_cache=torch.zeros(1, 4, 512, dtype=torch.float32),
        state_slot_mapping=torch.arange(tokens, dtype=torch.int64),
        state_block_table=torch.tensor([[0]], dtype=torch.int32),
        state_block_size=4,
        token_to_req_indices=torch.zeros(tokens, dtype=torch.int32),
        k_cache=torch.zeros(1, 64, 132, dtype=torch.uint8),
        k_slot_mapping=torch.full((tokens,), -1, dtype=torch.int64),
        cos_sin_cache=torch.zeros(16, 128, dtype=torch.float32),
        rms_norm_eps=1e-6,
        rope_head_dim=64,
    )


def test_compressor_calls_official_operations(monkeypatch) -> None:
    from vllm.models.deepseek_v4.common.ops import fused_compress_quant_cache

    save = Mock()
    compress = Mock()
    monkeypatch.setattr(runtime, "save_partial_states", save)
    monkeypatch.setattr(
        fused_compress_quant_cache, "compress_norm_rope_store_triton", compress
    )
    metadata = _compressor_metadata()
    runtime.compressor_operation(
        kv_score=torch.zeros(4, 512, dtype=torch.float32),
        positions=torch.arange(4, dtype=torch.int64),
        ape=torch.zeros(4, 256, dtype=torch.float32),
        norm_weight=torch.ones(128, dtype=torch.bfloat16),
        compress_ratio=4,
        head_dim=128,
        metadata=metadata,
    )

    save.assert_called_once()
    assert save.call_args.kwargs["state_cache"] is metadata.state_cache
    assert save.call_args.kwargs["state_width"] == 256
    compress.assert_called_once()
    assert compress.call_args.kwargs["kv_cache"] is metadata.k_cache
    assert compress.call_args.kwargs["token_stride"] == 128
