# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import os
import tarfile
from types import SimpleNamespace

import pytest
import torch

from megatron.core.models.audio.audio_feature_config import NemoTransformerAudioTokenEstimator
from megatron.core.models.audio.nemo_audio_checkpoint import (
    CHECKPOINT_NEMO_AUDIO_PREPROCESSOR_CONFIG_NAME,
    CHECKPOINT_NEMO_TRANSFORMER_AUDIO_CONFIG_NAME,
    extract_nemo_archive,
    has_nemo_audio_configs_in_checkpoint_dir,
    load_nemo_transformer_audio_weights,
    nemo_audio_configs_from_archive,
    nemo_audio_configs_from_checkpoint_dir,
    nemo_audio_configs_from_path,
    read_nemo_config,
    resolve_nemo_audio_configs_from_args,
    write_nemo_audio_configs_from_args_to_checkpoint_dir,
    write_nemo_audio_configs_to_checkpoint_dir,
)
from megatron.core.models.audio.nemo_transformer_audio_model import (
    NemoTransformerAudioConfig,
    NemoTransformerAudioModel,
)

_MIN_CFG = NemoTransformerAudioConfig(
    n_mels=8, d_model=16, n_heads=2, n_layers=1, pre_encode="conv", nan_debug=False
)


def _write_fake_nemo_archive(
    tmp_path,
    *,
    encoder_state: dict,
    encoder_target: str = "nemo.collections.asr.modules.transformer_encoder.TransformerEncoder",
    preproc_target: str = "nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor",
    encoder_overrides: dict | None = None,
    preproc_overrides: dict | None = None,
) -> str:
    """Build a minimal .nemo archive that mirrors NeMo's SaveRestoreConnector layout."""
    omegaconf = pytest.importorskip("omegaconf")
    from omegaconf import OmegaConf

    encoder_dict = {
        "_target_": encoder_target,
        "n_mels": _MIN_CFG.n_mels,
        "d_model": _MIN_CFG.d_model,
        "n_heads": _MIN_CFG.n_heads,
        "n_layers": _MIN_CFG.n_layers,
        "drop_rate": _MIN_CFG.drop_rate,
        "qkv_bias": _MIN_CFG.qkv_bias,
        "pre_encode": _MIN_CFG.pre_encode,
        "qk_norm": _MIN_CFG.qk_norm,
        "subsampling_factor": _MIN_CFG.subsampling_factor,
    }
    encoder_dict.update(encoder_overrides or {})

    preproc_dict = {
        "_target_": preproc_target,
        "sample_rate": 16000,
        "window_size": 0.025,
        "window_stride": 0.01,
        "window": "hann",
        "features": _MIN_CFG.n_mels,
        "n_fft": 512,
        "normalize": "per_feature",
        "preemph": 0.97,
        "log": True,
        "dither": 0.0,
        "pad_to": 0,
    }
    preproc_dict.update(preproc_overrides or {})

    cfg = OmegaConf.create({"model": {"encoder": encoder_dict, "preprocessor": preproc_dict}})

    work = tmp_path / "build"
    work.mkdir()
    cfg_path = work / "model_config.yaml"
    OmegaConf.save(cfg, cfg_path)

    # Prefix all encoder tensors with ``encoder.`` to match NeMo's flat state dict layout.
    full_state = {f"encoder.{k}": v for k, v in encoder_state.items()}
    # Sprinkle in some non-encoder keys to confirm the loader filters them out.
    full_state["decoder.linear.weight"] = torch.randn(2, 2)
    full_state["preprocessor.featurizer.window"] = torch.ones(400)

    wts_path = work / "model_weights.ckpt"
    torch.save(full_state, wts_path)

    nemo_path = tmp_path / "fake.nemo"
    with tarfile.open(nemo_path, "w:") as tar:
        tar.add(str(cfg_path), arcname="./model_config.yaml")
        tar.add(str(wts_path), arcname="./model_weights.ckpt")
    return str(nemo_path)


class TestNemoTransformerAudioModel:
    def test_forward_shapes_conv_subsample(self):
        model = NemoTransformerAudioModel(_MIN_CFG)

        input_features = torch.randn(2, 40, _MIN_CFG.n_mels)
        attention_mask = torch.tensor([[True] * 40, [True] * 28 + [False] * 12], dtype=torch.bool)

        hidden_states, output_mask = model(input_features, attention_mask)

        assert hidden_states.shape == torch.Size([2, 10, _MIN_CFG.d_model])
        assert output_mask.shape == torch.Size([2, 10])
        assert output_mask[0].all()
        assert output_mask[1].sum().item() == 7

    def test_encoder_stride_frames_keep_dummy_audio_nonempty(self):
        model = NemoTransformerAudioModel(_MIN_CFG)
        dummy_frames = _MIN_CFG.encoder_time_stride

        input_features = torch.zeros(1, dummy_frames, _MIN_CFG.n_mels)
        attention_mask = torch.ones(1, dummy_frames, dtype=torch.bool)

        hidden_states, output_mask = model(input_features, attention_mask)

        assert hidden_states.shape == torch.Size([1, 1, _MIN_CFG.d_model])
        assert output_mask.shape == torch.Size([1, 1])
        assert output_mask.sum().item() == 1

    def test_forward_shapes_windowed_causal_sdpa(self):
        cfg = NemoTransformerAudioConfig(
            **{**_MIN_CFG.__dict__, "attn_impl": "sdpa", "causal_mask": True, "left_context": 2}
        )
        model = NemoTransformerAudioModel(cfg)

        input_features = torch.randn(2, 40, cfg.n_mels)
        attention_mask = torch.tensor([[True] * 40, [True] * 28 + [False] * 12], dtype=torch.bool)

        hidden_states, output_mask = model(input_features, attention_mask)

        assert hidden_states.shape == torch.Size([2, 10, cfg.d_model])
        assert output_mask.shape == torch.Size([2, 10])
        assert output_mask[0].all()
        assert output_mask[1].sum().item() == 7

    def test_forward_packed_matches_dense_valid_tokens_sdpa(self):
        cfg = NemoTransformerAudioConfig(
            **{**_MIN_CFG.__dict__, "attn_impl": "sdpa", "drop_rate": 0.0}
        )
        model = NemoTransformerAudioModel(cfg)
        model.eval()

        input_features = torch.randn(2, 40, cfg.n_mels)
        attention_mask = torch.tensor([[True] * 40, [True] * 28 + [False] * 12], dtype=torch.bool)

        dense_hidden, dense_mask = model(input_features, attention_mask)
        packed_hidden = model.forward_packed(input_features, attention_mask)

        torch.testing.assert_close(packed_hidden.lengths, dense_mask.sum(dim=-1).to(torch.int32))
        torch.testing.assert_close(packed_hidden.embeddings, dense_hidden[dense_mask])


class TestNemoArchiveReaders:
    def test_read_nemo_config_returns_dict(self, tmp_path):
        pytest.importorskip("omegaconf")
        ref_model = NemoTransformerAudioModel(_MIN_CFG)
        nemo_path = _write_fake_nemo_archive(tmp_path, encoder_state=ref_model.encoder.state_dict())

        cfg = read_nemo_config(nemo_path)
        assert "encoder" in cfg
        assert "preprocessor" in cfg
        assert cfg["encoder"]["d_model"] == _MIN_CFG.d_model

    def test_configs_from_path_skip_state_dict(self, tmp_path):
        pytest.importorskip("omegaconf")
        ref_model = NemoTransformerAudioModel(_MIN_CFG)
        nemo_path = _write_fake_nemo_archive(tmp_path, encoder_state=ref_model.encoder.state_dict())

        encoder_cfg, preproc_cfg = nemo_audio_configs_from_path(nemo_path)
        assert encoder_cfg.n_mels == _MIN_CFG.n_mels
        assert encoder_cfg.pre_encode == _MIN_CFG.pre_encode
        assert preproc_cfg.features == _MIN_CFG.n_mels
        assert preproc_cfg.sample_rate == 16000

    def test_configs_from_path_accepts_flex_transformer_encoder_target(self, tmp_path):
        pytest.importorskip("omegaconf")
        ref_model = NemoTransformerAudioModel(_MIN_CFG)
        nemo_path = _write_fake_nemo_archive(
            tmp_path,
            encoder_state=ref_model.encoder.state_dict(),
            encoder_target="nemo.collections.asr.modules.transformer_encoder_flex.TransformerEncoder",
        )

        encoder_cfg, preproc_cfg = nemo_audio_configs_from_path(nemo_path)
        assert encoder_cfg.d_model == _MIN_CFG.d_model
        assert preproc_cfg.features == _MIN_CFG.n_mels

    def test_configs_from_path_maps_nemo_causal_attn_mode(self, tmp_path):
        pytest.importorskip("omegaconf")
        ref_model = NemoTransformerAudioModel(_MIN_CFG)
        nemo_path = _write_fake_nemo_archive(
            tmp_path,
            encoder_state=ref_model.encoder.state_dict(),
            encoder_overrides={"attn_mode": "causal"},
        )

        encoder_cfg, _ = nemo_audio_configs_from_path(nemo_path)

        assert encoder_cfg.causal_mask is True

    def test_left_context_requires_causal_mask(self):
        cfg = NemoTransformerAudioConfig(
            **{**_MIN_CFG.__dict__, "causal_mask": False, "left_context": 2}
        )

        with pytest.raises(ValueError, match="left_context requires causal_mask"):
            NemoTransformerAudioModel(cfg)

    def test_windowed_sdpa_attention_builds_local_causal_mask(self, monkeypatch):
        from megatron.core.models.audio import nemo_transformer_encoder

        captured = {}

        def fake_sdpa(query, key, value, attn_mask=None, is_causal=False, dropout_p=0.0):
            captured["attn_mask"] = attn_mask.detach().cpu()
            captured["is_causal"] = is_causal
            return torch.zeros_like(query)

        monkeypatch.setattr(nemo_transformer_encoder, "scaled_dot_product_attention", fake_sdpa)
        mha = nemo_transformer_encoder.MultiHeadAttentionWithSDPA(
            dim_in=4, dim_out=4, num_heads=1, dropout=0.0, causal_mask=True, left_context=2
        )
        x = torch.randn(1, 5, 4)
        pad_mask = torch.ones(1, 1, 1, 5, dtype=torch.bool)

        mha(x, attn_mask=pad_mask)

        assert captured["is_causal"] is False
        assert captured["attn_mask"].shape == torch.Size([1, 1, 5, 5])
        assert captured["attn_mask"][0, 0].tolist() == [
            [True, False, False, False, False],
            [True, True, False, False, False],
            [True, True, True, False, False],
            [False, True, True, True, False],
            [False, False, True, True, True],
        ]

    def test_te_attention_receives_causal_window_size(self, monkeypatch):
        from megatron.core.models.audio import nemo_transformer_encoder

        captured = {}

        class FakeTEDotProductAttention(torch.nn.Module):
            def __init__(self, **kwargs):
                super().__init__()
                captured.update(kwargs)

            def forward(self, q, k, v, *args, **kwargs):
                return torch.zeros_like(q)

        monkeypatch.setattr(
            nemo_transformer_encoder,
            "_get_te_dot_product_attention",
            lambda: FakeTEDotProductAttention,
        )

        nemo_transformer_encoder.MultiHeadAttentionWithTE(
            dim_in=4, dim_out=4, num_heads=1, dropout=0.0, causal_mask=True, left_context=3
        )

        assert captured["window_size"] == (3, 0)

    def test_flash_attention_receives_causal_window_size(self, monkeypatch):
        from megatron.core.models.audio import nemo_transformer_encoder

        captured = {}

        def fake_flash_attn_func(query, key, value, **kwargs):
            captured.update(kwargs)
            return torch.zeros_like(query)

        monkeypatch.setattr(nemo_transformer_encoder, "_flash_attn_func", fake_flash_attn_func)
        mha = nemo_transformer_encoder.MultiHeadAttentionWithFA(
            dim_in=4, dim_out=4, num_heads=1, dropout=0.0, causal_mask=True, left_context=4
        )

        mha(torch.randn(1, 5, 4))

        assert captured["causal"] is True
        assert captured["window_size"] == (4, 0)

    def test_recompute_audio_checkpoints_transformer_layers(self, monkeypatch):
        from megatron.core.models.audio import nemo_transformer_encoder

        calls = []

        def fake_checkpoint(function, *args, **kwargs):
            calls.append(kwargs)
            return function(*args)

        monkeypatch.setattr(nemo_transformer_encoder, "checkpoint", fake_checkpoint)
        cfg = NemoTransformerAudioConfig(
            **{**_MIN_CFG.__dict__, "attn_impl": "sdpa", "n_layers": 2, "recompute_layers": True}
        )
        model = NemoTransformerAudioModel(cfg)
        model.train()

        input_features = torch.randn(2, 40, cfg.n_mels)
        attention_mask = torch.ones(2, 40, dtype=torch.bool)
        hidden_states, _ = model(input_features, attention_mask)
        hidden_states.sum().backward()

        assert len(calls) == cfg.n_layers
        assert all(call["use_reentrant"] is False for call in calls)

    def test_checkpoint_local_audio_configs_roundtrip(self, tmp_path):
        from megatron.core.models.audio.audio_feature_config import NemoAudioFeatureConfig

        encoder_cfg = NemoTransformerAudioConfig(**{**_MIN_CFG.__dict__, "attn_impl": "te"})
        preproc_cfg = NemoAudioFeatureConfig(features=_MIN_CFG.n_mels, sample_rate=16000)
        ckpt_dir = tmp_path / "iter_0001000"

        encoder_path, preproc_path = write_nemo_audio_configs_to_checkpoint_dir(
            ckpt_dir, encoder_cfg, preproc_cfg
        )

        assert encoder_path.name == CHECKPOINT_NEMO_TRANSFORMER_AUDIO_CONFIG_NAME
        assert preproc_path.name == CHECKPOINT_NEMO_AUDIO_PREPROCESSOR_CONFIG_NAME
        assert has_nemo_audio_configs_in_checkpoint_dir(ckpt_dir)

        loaded_encoder_cfg, loaded_preproc_cfg = nemo_audio_configs_from_checkpoint_dir(ckpt_dir)
        assert loaded_encoder_cfg == encoder_cfg
        assert loaded_preproc_cfg == preproc_cfg

    def test_write_checkpoint_local_audio_configs_from_nemo_args(self, tmp_path):
        pytest.importorskip("omegaconf")
        ref_model = NemoTransformerAudioModel(_MIN_CFG)
        nemo_path = _write_fake_nemo_archive(tmp_path, encoder_state=ref_model.encoder.state_dict())
        ckpt_dir = tmp_path / "iter_0002000"
        args = SimpleNamespace(
            audio_model_type="nemo_transformer",
            load_audio_from=nemo_path,
            nemo_transformer_audio_config=None,
            nemo_transformer_audio_attn_impl="te",
            nemo_audio_preprocessor_config=None,
            recompute_audio=True,
        )

        write_nemo_audio_configs_from_args_to_checkpoint_dir(args, ckpt_dir)

        encoder_cfg, preproc_cfg = nemo_audio_configs_from_checkpoint_dir(ckpt_dir)
        assert encoder_cfg.d_model == _MIN_CFG.d_model
        assert encoder_cfg.attn_impl == "te"
        assert encoder_cfg.recompute_layers is True
        assert preproc_cfg.features == _MIN_CFG.n_mels

    def test_resolve_nemo_audio_configs_from_json_args(self, tmp_path):
        from megatron.core.models.audio.audio_feature_config import NemoAudioFeatureConfig

        encoder_cfg = NemoTransformerAudioConfig(**{**_MIN_CFG.__dict__, "attn_impl": "sdpa"})
        preproc_cfg = NemoAudioFeatureConfig(features=_MIN_CFG.n_mels, sample_rate=8000)
        encoder_path, preproc_path = write_nemo_audio_configs_to_checkpoint_dir(
            tmp_path, encoder_cfg, preproc_cfg
        )
        args = SimpleNamespace(
            load_audio_from=None,
            nemo_transformer_audio_config=str(encoder_path),
            nemo_transformer_audio_attn_impl="te",
            nemo_audio_preprocessor_config=str(preproc_path),
            recompute_audio=True,
        )

        resolved_encoder_cfg, resolved_preproc_cfg = resolve_nemo_audio_configs_from_args(args)

        assert resolved_encoder_cfg.attn_impl == "te"
        assert resolved_encoder_cfg.recompute_layers is True
        assert resolved_encoder_cfg.d_model == encoder_cfg.d_model
        assert resolved_preproc_cfg == preproc_cfg

    def test_extract_archive_returns_full_state(self, tmp_path):
        pytest.importorskip("omegaconf")
        ref_model = NemoTransformerAudioModel(_MIN_CFG)
        nemo_path = _write_fake_nemo_archive(tmp_path, encoder_state=ref_model.encoder.state_dict())

        cfg, state = extract_nemo_archive(nemo_path)
        assert cfg["encoder"]["n_layers"] == _MIN_CFG.n_layers
        encoder_keys = [k for k in state if k.startswith("encoder.")]
        assert encoder_keys, "expected encoder.* tensors in full state dict"
        assert "decoder.linear.weight" in state  # non-encoder key preserved here

    def test_configs_from_archive_strips_encoder_prefix(self, tmp_path):
        pytest.importorskip("omegaconf")
        ref_model = NemoTransformerAudioModel(_MIN_CFG)
        nemo_path = _write_fake_nemo_archive(tmp_path, encoder_state=ref_model.encoder.state_dict())

        _, _, encoder_state = nemo_audio_configs_from_archive(nemo_path)
        assert encoder_state, "expected non-empty encoder state dict"
        assert all(not k.startswith("encoder.") for k in encoder_state)
        # And the keys must be loadable into a fresh encoder.
        fresh = NemoTransformerAudioModel(_MIN_CFG)
        missing, unexpected = fresh.encoder.load_state_dict(encoder_state, strict=False)
        missing = [k for k in missing if "_extra_state" not in k]
        unexpected = [k for k in unexpected if "_extra_state" not in k]
        assert not missing
        assert not unexpected

    def test_unknown_encoder_target_rejected(self, tmp_path):
        pytest.importorskip("omegaconf")
        ref_model = NemoTransformerAudioModel(_MIN_CFG)
        nemo_path = _write_fake_nemo_archive(
            tmp_path,
            encoder_state=ref_model.encoder.state_dict(),
            encoder_target="some.other.Encoder",
        )

        with pytest.raises(ValueError, match="not a known transformer"):
            nemo_audio_configs_from_path(nemo_path)


class TestNemoTransformerAudioCheckpoint:
    def test_load_weights_into_audio_model(self, tmp_path):
        pytest.importorskip("omegaconf")

        torch.manual_seed(0)
        ref_model = NemoTransformerAudioModel(_MIN_CFG)
        ref_state = {k: v.clone() for k, v in ref_model.encoder.state_dict().items()}
        nemo_path = _write_fake_nemo_archive(tmp_path, encoder_state=ref_state)

        fresh = NemoTransformerAudioModel(_MIN_CFG)
        missing, unexpected = load_nemo_transformer_audio_weights(fresh, nemo_path)

        assert not missing
        assert not unexpected
        for k, v in ref_state.items():
            assert torch.equal(fresh.encoder.state_dict()[k], v)

    def test_config_mismatch_raises(self, tmp_path):
        pytest.importorskip("omegaconf")
        ref_model = NemoTransformerAudioModel(_MIN_CFG)
        nemo_path = _write_fake_nemo_archive(
            tmp_path,
            encoder_state=ref_model.encoder.state_dict(),
            encoder_overrides={"d_model": _MIN_CFG.d_model + 8},
        )

        # Model-side config disagrees with archive's d_model.
        fresh = NemoTransformerAudioModel(_MIN_CFG)
        with pytest.raises(ValueError, match="d_model"):
            load_nemo_transformer_audio_weights(fresh, nemo_path)

    def test_non_nemo_file_rejected(self, tmp_path):
        bad = tmp_path / "weights.pt"
        torch.save({"x": torch.zeros(1)}, bad)
        fresh = NemoTransformerAudioModel(_MIN_CFG)
        with pytest.raises(ValueError, match="\\.nemo"):
            load_nemo_transformer_audio_weights(fresh, str(bad))

    def test_te_qk_norm_checkpoint_keys_load(self, tmp_path, monkeypatch):
        pytest.importorskip("omegaconf")
        from megatron.core.models.audio import nemo_transformer_encoder

        class FakeTEDotProductAttention(torch.nn.Module):
            def __init__(self, **kwargs):
                super().__init__()

            def forward(self, q, k, v, *args, **kwargs):
                return torch.zeros_like(q)

        monkeypatch.setattr(
            nemo_transformer_encoder,
            "_get_te_dot_product_attention",
            lambda: FakeTEDotProductAttention,
        )

        cfg = NemoTransformerAudioConfig(
            **{**_MIN_CFG.__dict__, "qk_norm": True, "attn_impl": "sdpa"}
        )
        ref_model = NemoTransformerAudioModel(cfg)
        nemo_path = _write_fake_nemo_archive(
            tmp_path,
            encoder_state=ref_model.encoder.state_dict(),
            encoder_overrides={"qk_norm": True},
        )

        te_cfg = NemoTransformerAudioConfig(
            **{**_MIN_CFG.__dict__, "qk_norm": True, "attn_impl": "te"}
        )
        fresh = NemoTransformerAudioModel(te_cfg)
        missing, unexpected = load_nemo_transformer_audio_weights(fresh, nemo_path)

        assert not missing
        assert not unexpected
        assert "layers.0.mha.q_norm.weight" in fresh.encoder.state_dict()
        assert "layers.0.mha.k_norm.weight" in fresh.encoder.state_dict()


class TestNemoTransformerAudioTokenEstimator:
    def test_matches_encoder_floor_then_projection_ceil_formula(self):
        est = NemoTransformerAudioTokenEstimator(
            stack_factor=2, encoder_time_stride=4, pre_encode="conv"
        )
        assert est.estimate(40) == 5
        assert est.estimate(39) == 5
        assert est.estimate(38) == 5
        assert est.estimate(37) == 5
        assert est.estimate(36) == 5
        assert est.estimate(35) == 4

    def test_matches_nemo_conv_subsample_output_lengths(self):
        model = NemoTransformerAudioModel(_MIN_CFG)
        estimator = NemoTransformerAudioTokenEstimator(
            stack_factor=1,
            encoder_time_stride=_MIN_CFG.encoder_time_stride,
            pre_encode=_MIN_CFG.pre_encode,
        )

        for num_frames in range(1, 17):
            input_features = torch.randn(1, num_frames, _MIN_CFG.n_mels)
            attention_mask = torch.ones(1, num_frames, dtype=torch.bool)
            _, output_mask = model(input_features, attention_mask)

            assert estimator.estimate(num_frames) == output_mask.sum().item()

    def test_matches_nemo_stacking_subsample_per_sample_ceil_lengths(self):
        cfg = NemoTransformerAudioConfig(
            **{**_MIN_CFG.__dict__, "pre_encode": "stacking", "subsampling_factor": 8}
        )
        model = NemoTransformerAudioModel(cfg)
        estimator = NemoTransformerAudioTokenEstimator(
            stack_factor=1, encoder_time_stride=cfg.encoder_time_stride, pre_encode=cfg.pre_encode
        )

        input_lengths = torch.tensor([293, 300], dtype=torch.long)
        padded_num_frames = 302
        input_features = torch.randn(2, padded_num_frames, cfg.n_mels)
        attention_mask = torch.arange(padded_num_frames).unsqueeze(0) < input_lengths.unsqueeze(1)
        _, output_mask = model(input_features, attention_mask)

        expected = [estimator.estimate(int(num_frames)) for num_frames in input_lengths.tolist()]
        assert expected == output_mask.sum(dim=-1).tolist()

    def test_stacking_estimate_is_independent_of_batch_padded_width(self):
        est = NemoTransformerAudioTokenEstimator(
            stack_factor=2, encoder_time_stride=8, pre_encode="stacking"
        )

        assert est.estimate(293) == est.estimate(293, padded_num_frames=293)
        assert est.estimate(293) == 19
        assert est.estimate(293, padded_num_frames=302) == 19

    def test_stacking_lengths_cover_partial_tail_frames(self):
        cfg = NemoTransformerAudioConfig(
            **{**_MIN_CFG.__dict__, "pre_encode": "stacking", "subsampling_factor": 8}
        )
        model = NemoTransformerAudioModel(cfg)
        estimator = NemoTransformerAudioTokenEstimator(
            stack_factor=1, encoder_time_stride=cfg.encoder_time_stride, pre_encode=cfg.pre_encode
        )

        input_lengths = torch.tensor([2597, 546, 2331, 1876], dtype=torch.long)
        padded_num_frames = 3000
        input_features = torch.randn(len(input_lengths), padded_num_frames, cfg.n_mels)
        attention_mask = torch.arange(padded_num_frames).unsqueeze(0) < input_lengths.unsqueeze(1)
        _, output_mask = model(input_features, attention_mask)

        expected = [estimator.estimate(int(num_frames)) for num_frames in input_lengths.tolist()]
        assert expected == [325, 69, 292, 235]
        assert expected == output_mask.sum(dim=-1).tolist()
