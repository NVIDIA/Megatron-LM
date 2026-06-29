# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
from packaging.version import Version

import megatron.core.transformer.transformer_config as transformer_config_module
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.transformer.transformer_config import TransformerConfig


def _sequence_packing_config(**overrides):
    kwargs = {
        "num_layers": 1,
        "hidden_size": 16,
        "num_attention_heads": 1,
        "sequence_packing_scheduler": "dp_balanced",
        "max_seqlen_per_dp_cp_rank": 128,
        "moe_token_dispatcher_type": "alltoall",
    }
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


def test_sequence_packing_scheduler_is_disabled_by_default():
    config = TransformerConfig(num_layers=1, hidden_size=16, num_attention_heads=1)

    assert config.sequence_packing_scheduler is None
    assert config.variable_seq_lengths is False


def test_sequence_packing_scheduler_name_is_validated_before_te(monkeypatch):
    monkeypatch.setattr(
        transformer_config_module,
        "is_te_min_version",
        lambda *_args, **_kwargs: pytest.fail("TE validation should not run"),
    )

    with pytest.raises(ValueError, match="Unsupported sequence packing scheduler"):
        _sequence_packing_config(
            sequence_packing_scheduler="unsupported",
            max_seqlen_per_dp_cp_rank=None,
            variable_seq_lengths=True,
        )


@pytest.mark.parametrize("max_seqlen_per_dp_cp_rank", [None, 0, -1, 1.5, True])
def test_sequence_packing_scheduler_requires_positive_max_seqlen(
    monkeypatch, max_seqlen_per_dp_cp_rank
):
    monkeypatch.setattr(
        transformer_config_module,
        "is_te_min_version",
        lambda *_args, **_kwargs: pytest.fail("TE validation should not run"),
    )

    with pytest.raises(ValueError, match="max_seqlen_per_dp_cp_rank must be a positive integer"):
        _sequence_packing_config(max_seqlen_per_dp_cp_rank=max_seqlen_per_dp_cp_rank)


def test_model_parallel_config_enforces_base_scheduler_invariants():
    config = ModelParallelConfig(
        sequence_packing_scheduler="dp_balanced", max_seqlen_per_dp_cp_rank=128
    )

    assert config.variable_seq_lengths is True


def test_sequence_packing_scheduler_requires_alltoall_dispatcher(monkeypatch):
    monkeypatch.setattr(
        transformer_config_module,
        "is_te_min_version",
        lambda *_args, **_kwargs: pytest.fail("TE validation should not run"),
    )

    with pytest.raises(ValueError, match="moe_token_dispatcher_type='alltoall'"):
        _sequence_packing_config(moe_token_dispatcher_type="allgather", variable_seq_lengths=True)


def test_sequence_packing_scheduler_requires_supported_te(monkeypatch):
    monkeypatch.setattr(transformer_config_module, "HAVE_PACKAGING", True)
    monkeypatch.setattr(transformer_config_module, "is_te_min_version", lambda *_args: False)
    monkeypatch.setattr(transformer_config_module, "get_te_version", lambda: Version("2.8.0"))

    with pytest.raises(ValueError, match="requires Transformer Engine >= 2.9.0"):
        _sequence_packing_config()


def test_sequence_packing_scheduler_requires_packaging(monkeypatch):
    monkeypatch.setattr(transformer_config_module, "HAVE_PACKAGING", False)

    with pytest.raises(ImportError, match="packaging is not installed"):
        _sequence_packing_config()


def test_sequence_packing_scheduler_allows_supported_te_development_version(monkeypatch):
    monkeypatch.setattr(transformer_config_module, "HAVE_PACKAGING", True)
    monkeypatch.setattr(transformer_config_module, "is_te_min_version", lambda *_args: False)
    monkeypatch.setattr(
        transformer_config_module, "get_te_version", lambda: Version("2.9.0.dev0+5b3092a")
    )

    config = _sequence_packing_config()

    assert config.variable_seq_lengths is True


def test_sequence_packing_scheduler_enables_variable_sequence_lengths(monkeypatch):
    monkeypatch.setattr(transformer_config_module, "HAVE_PACKAGING", True)
    monkeypatch.setattr(transformer_config_module, "is_te_min_version", lambda *_args: True)

    config = _sequence_packing_config()

    assert config.sequence_packing_scheduler == "dp_balanced"
    assert config.variable_seq_lengths is True
