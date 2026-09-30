# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Exercise the public Triton policy from CLI/YAML parsing through framework initialization."""

import sys
from argparse import ArgumentParser, Namespace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from megatron.core import tuning
from megatron.core.transformer import TransformerConfig
from megatron.core.tuning import AutotunePolicy
from megatron.training import initialize as initialization
from megatron.training.argument_utils import (
    _triton_autotune_config_from_args,
    core_transformer_config_from_args,
)
from megatron.training.arguments import add_megatron_arguments, parse_args
from megatron.training.yaml_arguments import core_transformer_config_from_yaml, load_yaml


@pytest.fixture
def parser():
    """Use the real training parser, including generated TransformerConfig arguments."""
    return add_megatron_arguments(ArgumentParser())


@pytest.fixture
def install_policy(monkeypatch):
    """Observe installation without changing the process-wide policy."""
    install = Mock()
    monkeypatch.setattr(tuning, 'install_from_config', install)
    return install


def test_cli_policy_reaches_transformer_config(parser, install_policy):
    args = parser.parse_args(
        [
            '--num-layers',
            '1',
            '--hidden-size',
            '128',
            '--num-attention-heads',
            '4',
            '--deterministic-mode',
            '--no-rope-fusion',
            '--triton-autotune-mode',
            'pinned',
            '--triton-autotune-modules',
            'mamba_ssm',
            'my_kernels',
            '--triton-autotune-table-path',
            '/tmp/tuned',
            '/tmp/backup',
            '--triton-autotune-on-miss',
            'error',
            '--triton-autotune-verify-every',
            '10',
            '--triton-autotune-verify-strict',
            '--triton-autotune-enumerate',
            '--triton-autotune-chaos',
            '--triton-autotune-block-sizes',
            'BLOCK_C=512',
            'BLOCK_S=1',
        ]
    )
    args.params_dtype = torch.float32

    config = core_transformer_config_from_args(args)

    expected = AutotunePolicy(
        mode='pinned',
        modules=('mamba_ssm', 'my_kernels'),
        table_path=('/tmp/tuned', '/tmp/backup'),
        on_miss='error',
        verify_every=10,
        verify_strict=True,
        enumerate_autotuners=True,
        chaos=True,
        block_sizes=(('BLOCK_C', 512), ('BLOCK_S', 1)),
    )
    assert config.triton_autotune == expected
    install_policy.assert_called_once_with(expected, deterministic=True)


def test_cli_defaults_preserve_implicit_policy(parser):
    assert _triton_autotune_config_from_args(parser.parse_args([])) is None
    args = parser.parse_args(['--triton-autotune-record-path', '/tmp/record'])
    policy = _triton_autotune_config_from_args(args)
    assert policy.record_path == '/tmp/record'
    assert policy.resolve(deterministic=True).mode == 'record'


def test_python_policy_reaches_config_and_survives_argument_conversion(parser, install_policy):
    policy = AutotunePolicy(mode='auto', modules=('my_kernels',))
    config = TransformerConfig(
        num_layers=1,
        hidden_size=128,
        num_attention_heads=4,
        deterministic_mode=True,
        triton_autotune=policy,
    )
    assert config.triton_autotune is policy
    install_policy.assert_called_once_with(policy, deterministic=True)
    args = parser.parse_args([])
    args.triton_autotune = policy
    assert _triton_autotune_config_from_args(args) is policy


@pytest.mark.parametrize('value', ['BLOCK_C', 'BLOCK_C=zero', 'BLOCK_C=0', 'OTHER=32'])
def test_cli_rejects_invalid_block_overrides(parser, value):
    with pytest.raises(SystemExit):
        parser.parse_args(['--triton-autotune-block-sizes', value])


def test_training_installs_cli_policy_before_distributed_init(parser, install_policy, monkeypatch):
    args = parser.parse_args(['--deterministic-mode', '--triton-autotune-modules', 'my_kernels'])
    monkeypatch.setattr(initialization, 'get_args', lambda: args)
    monkeypatch.setattr(initialization, 'setup_logging', lambda: None)
    monkeypatch.setattr(initialization, 'initialize_rerun_state_machine', Mock())

    initialization.initialize_megatron(allow_no_cuda=True, skip_mpu_initialization=True)

    install_policy.assert_called_once_with(
        AutotunePolicy(modules=('my_kernels',)), deterministic=True
    )


@pytest.mark.parametrize('as_namespace', [False, True])
def test_yaml_policy_builds_typed_config(as_namespace, install_policy):
    values = {'mode': 'pinned', 'modules': ['my_kernels'], 'block_sizes': {'BLOCK_C': 512}}
    if as_namespace:
        values['block_sizes'] = SimpleNamespace(**values['block_sizes'])
        values = SimpleNamespace(**values)
    language_model = Namespace(
        **vars(TransformerConfig(num_layers=1, hidden_size=128, num_attention_heads=4))
    )
    language_model.triton_autotune = values
    language_model.params_dtype = torch.float32
    language_model.activation_func = 'gelu'
    language_model.embedding_init_method = 'xavier_uniform'
    args = SimpleNamespace(language_model=language_model, model_parallel=SimpleNamespace())
    install_policy.reset_mock()

    config = core_transformer_config_from_yaml(args)

    expected = AutotunePolicy(mode='pinned', modules=('my_kernels',), block_sizes={'BLOCK_C': 512})
    assert config.triton_autotune == expected
    install_policy.assert_called_once_with(expected, deterministic=False)


def test_cli_policy_cannot_be_silently_replaced_by_yaml(monkeypatch):
    monkeypatch.setattr(
        sys, 'argv', ['test', '--yaml-cfg', 'unused.yaml', '--triton-autotune-mode', 'auto']
    )
    with pytest.raises(ValueError, match='Triton autotune CLI arguments cannot be combined'):
        parse_args()


def test_yaml_loader_preserves_nested_block_overrides(tmp_path):
    config_path = tmp_path / 'autotune.yaml'
    config_path.write_text(
        'language_model:\n'
        '  triton_autotune:\n'
        '    mode: pinned\n'
        '    modules: [mamba_ssm, my_kernels]\n'
        '    table_path: [/tmp/tuned]\n'
        '    block_sizes:\n'
        '      BLOCK_C: 512\n'
    )

    args = load_yaml(config_path)
    policy = _triton_autotune_config_from_args(args.language_model)

    assert policy == AutotunePolicy(
        mode='pinned',
        modules=('mamba_ssm', 'my_kernels'),
        table_path=('/tmp/tuned',),
        block_sizes={'BLOCK_C': 512},
    )


def test_yaml_policy_nulls_fall_back_and_mistyped_values_raise():
    nulls = SimpleNamespace(
        triton_autotune={'mode': 'pinned', 'table_path': None, 'verify_every': None}
    )
    assert _triton_autotune_config_from_args(nulls) == AutotunePolicy(mode='pinned')
    quoted = SimpleNamespace(triton_autotune={'mode': 'pinned', 'chaos': 'false'})
    with pytest.raises(TypeError, match='chaos must be a bool'):
        _triton_autotune_config_from_args(quoted)
    unknown = SimpleNamespace(triton_autotune={'enumerate': True})
    with pytest.raises(TypeError, match='Unknown AutotunePolicy option'):
        _triton_autotune_config_from_args(unknown)


def test_cli_config_invariant_list(parser):
    empty = parser.parse_args(['--triton-autotune-config-invariant'])
    assert _triton_autotune_config_from_args(empty).config_invariant == ()
    named = parser.parse_args(['--triton-autotune-config-invariant', 'pkg.mod.kernel'])
    assert _triton_autotune_config_from_args(named).config_invariant == ('pkg.mod.kernel',)
