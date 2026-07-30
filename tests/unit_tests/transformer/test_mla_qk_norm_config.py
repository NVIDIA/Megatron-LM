# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from argparse import ArgumentParser
from types import SimpleNamespace

import pytest

import megatron.core.transformer.mla_qk_norm_config as qk_norm_config
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mla_qk_norm_config import QKNormConfigResolver
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import MLATransformerConfig
from megatron.training.arguments import _add_mla_args
from tests.unit_tests.transformer.experimental_attention_variant.dsa_native_parity_utils import (
    run_absorbed_mla_dsa_parity,
)


class _Linear:
    pass


class _FusedNormLinear:
    pass


class _Norm:
    pass


class _Backend:
    @staticmethod
    def layer_norm(**_kwargs):
        return _Norm

    @staticmethod
    def column_parallel_linear():
        return _Linear

    @staticmethod
    def column_parallel_layer_norm_linear():
        return _FusedNormLinear


def _make_config(*, q_lora_rank=32, attention_latent_norm_epsilon=1.0e-6):
    return MLATransformerConfig(
        num_layers=1,
        hidden_size=128,
        num_attention_heads=1,
        q_lora_rank=q_lora_rank,
        qk_layernorm=True,
        layernorm_epsilon=1.0e-5,
        attention_latent_norm_epsilon=attention_latent_norm_epsilon,
    )


def _make_submodules(**overrides):
    values = dict(
        linear_q_proj=None,
        linear_q_up_proj=None,
        linear_kv_up_proj=None,
        q_layernorm=IdentityOp,
        kv_layernorm=IdentityOp,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.fixture(autouse=True)
def _mock_backend(monkeypatch):
    monkeypatch.setattr(qk_norm_config, "get_backend", lambda _impl: _Backend())


@pytest.mark.parametrize("attention_latent_norm_epsilon", [None, 1.0e-6])
def test_attention_latent_norm_epsilon_default(attention_latent_norm_epsilon):
    config = _make_config(attention_latent_norm_epsilon=attention_latent_norm_epsilon)
    expected = (
        config.layernorm_epsilon
        if attention_latent_norm_epsilon is None
        else attention_latent_norm_epsilon
    )

    assert config.attention_latent_norm_epsilon == expected


def test_attention_latent_norm_epsilon_argument():
    parser = ArgumentParser()
    _add_mla_args(parser)

    args = parser.parse_args(["--attention-latent-norm-epsilon", "1e-6"])

    assert args.attention_latent_norm_epsilon == pytest.approx(1.0e-6)


@pytest.mark.parametrize(
    ("q_lora_rank", "q_projection"), [(None, "linear_q_proj"), (32, "linear_q_up_proj")]
)
def test_fused_norm_projections_carry_attention_latent_epsilon(q_lora_rank, q_projection):
    config = _make_config(q_lora_rank=q_lora_rank)
    submodules = _make_submodules(
        linear_kv_up_proj=ModuleSpec(module=_FusedNormLinear, params={"custom": True})
    )

    resolved = QKNormConfigResolver(config, submodules).resolve()

    q_spec = resolved[q_projection]
    assert isinstance(q_spec, ModuleSpec)
    assert q_spec.module is _FusedNormLinear
    assert q_spec.params["eps"] == config.attention_latent_norm_epsilon

    kv_spec = resolved["linear_kv_up_proj"]
    assert isinstance(kv_spec, ModuleSpec)
    assert kv_spec.module is _FusedNormLinear
    assert kv_spec.params == {"custom": True, "eps": config.attention_latent_norm_epsilon}
    assert submodules.linear_kv_up_proj.params == {"custom": True}


def test_dsa_attention_latent_norm_epsilon_matches_native():
    """Exercise distinct global, attention latent, and indexer norm epsilons in DSA."""
    run_absorbed_mla_dsa_parity(
        kernel_backend="none",
        seqlen=64,
        attention_backend=AttnBackend.unfused,
        calculate_per_token_loss=True,
        use_sparse_loss=False,
        num_iterations=1,
    )
