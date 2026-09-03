# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from megatron.core.models.audio.audio_feature_config import (
    NemoAudioFeatureConfig,
    NemoTransformerAudioTokenEstimator,
)
from megatron.core.models.audio.nemo_audio_checkpoint import (
    _split_audio_configs,
    resolve_nemo_audio_configs_from_args,
    write_nemo_audio_configs_to_checkpoint_dir,
)
from megatron.core.models.audio.nemo_rope_transformer_encoder import (
    FeatureStacking,
    RopeTransformerEncoder,
    RopeTransformerEncoderConfig,
)
from megatron.core.models.audio.nemo_transformer_audio_model import (
    NemoTransformerAudioConfig,
    NemoTransformerAudioModel,
)


def _rope_config(**overrides) -> NemoTransformerAudioConfig:
    values = {
        "n_mels": 8,
        "d_model": 16,
        "n_heads": 2,
        "n_layers": 1,
        "drop_rate": 0.0,
        "pre_encode": "feature_stacking",
        "subsampling_factor": 2,
        "qk_norm": True,
        "architecture": "rope_transformer",
        "self_attention_model": "rope",
        "rotary_fraction": 0.5,
    }
    values.update(overrides)
    return NemoTransformerAudioConfig(**values)


def test_rope_feature_stacking_packed_matches_dense_valid_tokens():
    stacking = FeatureStacking(subsampling_factor=4, feat_in=3, feat_out=5).eval()
    features = torch.randn(2, 3, 12)
    lengths = torch.tensor([12, 9], dtype=torch.int64)
    features[1, :, 9:] = 0

    dense, dense_lengths = stacking(features, lengths)
    packed, packed_lengths = stacking.forward_packed(features, lengths)
    valid = torch.arange(dense.shape[1]).unsqueeze(0) < dense_lengths.unsqueeze(1)

    torch.testing.assert_close(packed_lengths, dense_lengths)
    torch.testing.assert_close(packed, dense[valid])


def test_rope_feature_stacking_accepts_already_packed_frames():
    stacking = FeatureStacking(subsampling_factor=4, feat_in=3, feat_out=5).eval()
    clips = [torch.randn(12, 3), torch.randn(9, 3)]
    lengths = torch.tensor([12, 9], dtype=torch.int64)
    dense_input = torch.nn.utils.rnn.pad_sequence(clips, batch_first=True).transpose(1, 2)
    dense, dense_lengths = stacking(dense_input, lengths)

    packed, packed_lengths = stacking.forward_packed(torch.cat(clips), lengths)
    valid = torch.arange(dense.shape[1]).unsqueeze(0) < dense_lengths.unsqueeze(1)

    torch.testing.assert_close(packed_lengths, dense_lengths)
    torch.testing.assert_close(packed, dense[valid])


def test_rope_encoder_builds_one_mixed_policy_object_without_reordering_tokens():
    class RecordingBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inputs = []
            self.cu_seqlens = []
            self.policies = []

        def forward_packed(self, x, cu_seqlens, max_seqlen, *, is_causal, attention_policies):
            self.inputs.append(x)
            self.cu_seqlens.append(cu_seqlens)
            self.policies.append(attention_policies)
            return x

    config = RopeTransformerEncoderConfig(
        n_mels=3,
        d_model=8,
        n_heads=2,
        n_layers=2,
        drop_rate=0.0,
        qkv_bias=False,
        qk_norm=False,
        ff_expansion=2.0,
        pre_block_norm=True,
        subsampling_factor=2,
        rope_base=10000.0,
        rotary_fraction=1.0,
        left_context=3,
    )
    encoder = RopeTransformerEncoder(config).eval()
    recording_block = RecordingBlock()
    encoder.layers = torch.nn.ModuleList([recording_block, recording_block])
    audio = torch.randn(2, 3, 6)
    lengths = torch.tensor([6, 4], dtype=torch.int64)
    expected_packed, _ = encoder.pre_encode.forward_packed(audio, lengths)
    expected_packed = encoder.embed_norm(expected_packed)

    _, output_lengths = encoder.forward_packed(
        audio, lengths, thd_sequence_is_causal=torch.tensor([True, False], dtype=torch.bool)
    )

    assert output_lengths.tolist() == [3, 2]
    assert len(recording_block.policies) == 2
    assert recording_block.policies[0] is recording_block.policies[1]
    policies = recording_block.policies[0].policies
    assert [policy["mask_type"] for policy in policies] == ["padding", "padding_causal"]
    assert [policy["window_size"] for policy in policies] == [(-1, -1), (3, 0)]
    torch.testing.assert_close(policies[0]["sequence_ids"], torch.tensor([1]))
    torch.testing.assert_close(policies[1]["sequence_ids"], torch.tensor([0]))
    torch.testing.assert_close(recording_block.inputs[0], expected_packed)
    torch.testing.assert_close(
        recording_block.cu_seqlens[0], torch.tensor([0, 3, 5], dtype=torch.int32)
    )


def test_rope_transformer_requires_transformer_engine_attention():
    with pytest.raises(ValueError, match="requires Transformer Engine"):
        NemoTransformerAudioModel(_rope_config(attn_impl="sdpa"))


def test_rope_transformer_advertises_dynamic_causal_packed_attention():
    config = _rope_config()
    model = NemoTransformerAudioModel(config).eval()

    assert config.attn_impl == "te"
    assert model.supports_dynamic_causal_mask
    assert model.supports_packed_forward


def test_rope_transformer_forwards_final_grouped_policy_api(monkeypatch):
    init_kwargs = []
    forward_kwargs = []

    class FakeTEDotProductAttention(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            init_kwargs.append(kwargs)

        def forward(self, query, *args, **kwargs):
            forward_kwargs.append(kwargs)
            return query

    te_root = ModuleType("transformer_engine")
    te_pytorch = ModuleType("transformer_engine.pytorch")
    te_pytorch.DotProductAttention = FakeTEDotProductAttention
    te_root.pytorch = te_pytorch
    monkeypatch.setitem(sys.modules, "transformer_engine", te_root)
    monkeypatch.setitem(sys.modules, "transformer_engine.pytorch", te_pytorch)

    model = NemoTransformerAudioModel(_rope_config(left_context=3)).eval()
    attention = model.encoder.layers[0].attn
    query = torch.randn(4, 2, 8)
    cu_seqlens = torch.tensor([0, 4], dtype=torch.int32)

    attention._run_packed_te_attention(query, query, query, cu_seqlens, 4, is_causal=False)
    attention._run_packed_te_attention(query, query, query, cu_seqlens, 4, is_causal=True)
    mixed_policy = SimpleNamespace(
        policies=[
            {"sequence_ids": torch.tensor([0]), "mask_type": "padding", "window_size": (-1, -1)},
            {
                "sequence_ids": torch.tensor([1]),
                "mask_type": "padding_causal",
                "window_size": (3, 0),
            },
        ]
    )
    attention._run_packed_te_attention(
        query, query, query, cu_seqlens, 4, is_causal=False, attention_policies=mixed_policy
    )

    assert len(init_kwargs) == 1
    assert init_kwargs[0]["attn_mask_type"] == "padding"
    assert forward_kwargs[0]["attn_mask_type"] == "padding"
    assert forward_kwargs[0]["window_size"] == (-1, -1)
    assert forward_kwargs[1]["attn_mask_type"] == "padding_causal"
    assert forward_kwargs[1]["window_size"] == (3, 0)
    assert "attn_mask_type" not in forward_kwargs[2]
    assert "window_size" not in forward_kwargs[2]
    assert forward_kwargs[2]["thd_attention_policies"] is mixed_policy.policies
    assert forward_kwargs[2]["thd_attention_policy_dispatch"] == "grouped"


def test_rope_left_context_preserves_checkpoint_state_dict():
    unlimited = NemoTransformerAudioModel(_rope_config())
    windowed = NemoTransformerAudioModel(_rope_config(left_context=3))

    windowed.load_state_dict(unlimited.state_dict(), strict=True)
    assert windowed.encoder.layers[0].attn.causal_window_size == (3, 0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_rope_mixed_windowed_packed_te_matches_sdpa():
    pytest.importorskip("transformer_engine.pytorch")
    from megatron.core.models.audio import nemo_rope_transformer_encoder as rope_encoder

    config = rope_encoder.RopeTransformerEncoderConfig(
        n_mels=8,
        d_model=16,
        n_heads=2,
        n_layers=1,
        drop_rate=0.0,
        qkv_bias=False,
        qk_norm=False,
        ff_expansion=4.0,
        pre_block_norm=True,
        subsampling_factor=2,
        rope_base=10000.0,
        rotary_fraction=0.5,
        left_context=2,
    )
    rope = rope_encoder.RotaryPositionalEncoding(d_k=8, rotary_fraction=0.5).cuda()
    attention = rope_encoder.MultiHeadAttention(config, rope).cuda().bfloat16().eval()
    lengths = torch.tensor([6, 4], dtype=torch.int32, device="cuda")
    cu_seqlens = torch.nn.functional.pad(torch.cumsum(lengths, dim=0, dtype=torch.int32), (1, 0))
    hidden = torch.randn(10, 16, dtype=torch.bfloat16, device="cuda")
    rope.extend_pe(6, hidden.device, hidden.dtype)
    attention_policies = rope_encoder._PackedAttentionPolicies(
        policies=[
            {
                "sequence_ids": torch.tensor([0], device="cuda"),
                "mask_type": "padding",
                "window_size": (-1, -1),
            },
            {
                "sequence_ids": torch.tensor([1], device="cuda"),
                "mask_type": "padding_causal",
                "window_size": (2, 0),
            },
        ]
    )

    with torch.no_grad():
        actual = attention.forward_packed(
            hidden, cu_seqlens, max_seqlen=6, is_causal=False, attention_policies=attention_policies
        )

        q_weight, k_weight, v_weight = attention.w_qkv.weight.chunk(3, dim=0)
        query = torch.nn.functional.linear(hidden, q_weight).view(10, 2, 8)
        key = torch.nn.functional.linear(hidden, k_weight).view(10, 2, 8)
        value = torch.nn.functional.linear(hidden, v_weight).view(10, 2, 8)
        query, key = rope.forward_packed(query, key, cu_seqlens)
        expected_chunks = []
        for sequence_id, (start, end) in enumerate(
            zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist())
        ):
            sequence_length = end - start
            attention_mask = None
            if sequence_id == 1:
                positions = torch.arange(sequence_length, device="cuda")
                attention_mask = (positions.unsqueeze(1) >= positions.unsqueeze(0)) & (
                    positions.unsqueeze(1) - positions.unsqueeze(0) <= 2
                )
            expected_chunks.append(
                torch.nn.functional.scaled_dot_product_attention(
                    query[start:end].transpose(0, 1).unsqueeze(0),
                    key[start:end].transpose(0, 1).unsqueeze(0),
                    value[start:end].transpose(0, 1).unsqueeze(0),
                    attn_mask=attention_mask,
                    dropout_p=0.0,
                )
                .squeeze(0)
                .transpose(0, 1)
            )
        expected = attention.out_proj(torch.cat(expected_chunks).reshape(10, 16))

    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


def test_rope_archive_config_is_detected_from_nemo_fields():
    encoder_config, _ = _split_audio_configs(
        {
            "encoder": {
                "_target_": "nemo.collections.asr.modules.transformer_encoder.TransformerEncoder",
                "feat_in": 128,
                "d_model": 1280,
                "n_heads": 16,
                "n_layers": 32,
                "drop_rate": 0.0,
                "qkv_bias": False,
                "qk_norm": True,
                "self_attention_model": "rope",
                "subsampling_factor": 8,
            },
            "preprocessor": {
                "_target_": "nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor",
                "features": 128,
            },
        },
        "fake.nemo",
    )

    assert encoder_config.n_mels == 128
    assert encoder_config.architecture == "rope_transformer"
    assert encoder_config.pre_encode == "feature_stacking"
    assert encoder_config.attn_impl == "te"


@pytest.mark.parametrize(("causal_mode", "expected"), [("causal", True), ("offline", False)])
def test_resolve_rope_runtime_causal_mode(tmp_path, causal_mode, expected):
    encoder_path, preprocessor_path = write_nemo_audio_configs_to_checkpoint_dir(
        tmp_path, _rope_config(), NemoAudioFeatureConfig(features=8)
    )
    args = SimpleNamespace(
        load_audio_from=None,
        nemo_transformer_audio_config=str(encoder_path),
        nemo_transformer_audio_attn_impl=None,
        nemo_transformer_audio_left_context=128,
        nemo_transformer_audio_causal_mode=causal_mode,
        nemo_audio_preprocessor_config=str(preprocessor_path),
        recompute_audio=False,
    )

    resolved, _ = resolve_nemo_audio_configs_from_args(args)

    assert resolved.causal_mask is expected
    assert resolved.left_context == 128


def test_feature_stacking_token_estimator_uses_per_sample_ceil():
    estimator = NemoTransformerAudioTokenEstimator(
        encoder_time_stride=8, pre_encode="feature_stacking"
    )

    assert estimator.estimate(num_frames=9, padded_num_frames=64) == 2
