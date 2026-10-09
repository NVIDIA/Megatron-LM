# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Exercise the public Triton policy from CLI/YAML parsing through training initialization."""

import sys
from argparse import ArgumentParser
from dataclasses import fields
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from megatron.core import tuning
from megatron.core.transformer import TransformerConfig
from megatron.core.tuning import AutotunePolicy
from megatron.training import initialize as initialization
from megatron.training.argument_utils import _triton_autotune_config_from_args
from megatron.training.arguments import add_megatron_arguments, parse_args, validate_args
from megatron.training.yaml_arguments import load_yaml


@pytest.fixture
def parser():
    """Use the real training parser, including generated TransformerConfig arguments."""
    return add_megatron_arguments(ArgumentParser())


@pytest.fixture
def install_policy(monkeypatch):
    """Observe installation without changing the process-wide policy."""
    install = Mock()
    monkeypatch.setattr(tuning, 'install', install)
    return install


def _yaml_args(section):
    """Arguments as load_yaml returns them, with an optional triton_autotune section."""
    if section is None:
        return SimpleNamespace(yaml_cfg='config.yaml')
    return SimpleNamespace(yaml_cfg='config.yaml', triton_autotune=section)


def _validated_args(monkeypatch, *flags):
    monkeypatch.setattr(sys, 'argv', ['test_autotune_config.py', *flags])
    args = parse_args()
    # parse_args reads WORLD_SIZE from the environment; pin it so the derived parallel sizes
    # do not depend on how the test was launched.
    args.world_size = 1
    args.num_layers = 2
    args.hidden_size = 128
    args.num_attention_heads = 4
    args.max_position_embeddings = 1024
    args.seq_length = 1024
    args.micro_batch_size = 1
    args.global_batch_size = 1
    args.train_iters = 1
    args.lr = 1e-4
    args.tokenizer_type = 'NullTokenizer'
    args.vocab_size = 1024
    return validate_args(args)


def test_cli_flags_build_the_policy(parser):
    args = parser.parse_args(
        [
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
            '--triton-autotune-enumerate-autotuners',
            '--triton-autotune-chaos',
            '--triton-autotune-block-sizes',
            'BLOCK_C=512',
            'BLOCK_S=1',
        ]
    )

    assert _triton_autotune_config_from_args(args) == AutotunePolicy(
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


def test_cli_defaults_preserve_implicit_policy(parser):
    args = parser.parse_args([])
    # Every field has an argument, generated from the dataclass with its default.
    assert all(hasattr(args, f'triton_autotune_{item.name}') for item in fields(AutotunePolicy))
    assert _triton_autotune_config_from_args(args) == AutotunePolicy()
    args = parser.parse_args(['--triton-autotune-record-path', '/tmp/record'])
    policy = _triton_autotune_config_from_args(args)
    assert policy.record_path == '/tmp/record'
    assert policy.resolve(deterministic=True).mode == 'record'


def test_validated_arguments_build_the_policy_without_storing_it(monkeypatch):
    args = _validated_args(monkeypatch, '--triton-autotune-modules', 'my_kernels')
    assert not hasattr(args, 'triton_autotune')
    assert _triton_autotune_config_from_args(args) == AutotunePolicy(modules=('my_kernels',))


def test_model_config_does_not_install_a_policy(install_policy):
    TransformerConfig(num_layers=1, hidden_size=128, num_attention_heads=4, deterministic_mode=True)
    install_policy.assert_not_called()


@pytest.mark.parametrize('value', ['BLOCK_C', 'BLOCK_C=zero', 'BLOCK_C=0', 'OTHER=32'])
def test_cli_rejects_invalid_block_overrides(parser, value):
    with pytest.raises(SystemExit):
        parser.parse_args(['--triton-autotune-block-sizes', value])


def test_training_installs_the_policy_before_distributed_init(
    parser, install_policy, monkeypatch, run_config
):
    # Parsed but not validated, as in callers that build their own arguments.
    args = parser.parse_args(['--deterministic-mode', '--triton-autotune-modules', 'my_kernels'])
    monkeypatch.setattr(initialization, 'get_args', lambda: args)
    monkeypatch.setattr(initialization, 'setup_logging', lambda: None)
    monkeypatch.setattr(initialization, 'initialize_rerun_state_machine', Mock())

    initialization.initialize_megatron(allow_no_cuda=True, skip_mpu_initialization=True)

    install_policy.assert_called_once_with(
        AutotunePolicy(modules=('my_kernels',)), deterministic=True
    )


@pytest.mark.parametrize('as_namespace', [False, True])
def test_yaml_section_builds_typed_policy(as_namespace):
    values = {'mode': 'pinned', 'modules': ['my_kernels'], 'block_sizes': {'BLOCK_C': 512}}
    if as_namespace:
        values['block_sizes'] = SimpleNamespace(**values['block_sizes'])
        values = SimpleNamespace(**values)

    policy = _triton_autotune_config_from_args(_yaml_args(values))

    assert policy == AutotunePolicy(
        mode='pinned', modules=('my_kernels',), block_sizes={'BLOCK_C': 512}
    )


def test_cli_policy_cannot_be_silently_replaced_by_yaml(monkeypatch):
    monkeypatch.setattr(
        sys, 'argv', ['test', '--yaml-cfg', 'unused.yaml', '--triton-autotune-mode', 'auto']
    )
    with pytest.raises(ValueError, match='Triton autotune CLI arguments cannot be combined'):
        parse_args()


def test_yaml_loader_preserves_nested_block_overrides(tmp_path):
    config_path = tmp_path / 'autotune.yaml'
    config_path.write_text(
        'triton_autotune:\n'
        '  mode: pinned\n'
        '  modules: [mamba_ssm, my_kernels]\n'
        '  table_path: [/tmp/tuned]\n'
        '  block_sizes:\n'
        '    BLOCK_C: 512\n'
    )

    policy = _triton_autotune_config_from_args(load_yaml(config_path))

    assert policy == AutotunePolicy(
        mode='pinned',
        modules=('mamba_ssm', 'my_kernels'),
        table_path=('/tmp/tuned',),
        block_sizes={'BLOCK_C': 512},
    )


def test_yaml_policy_nulls_fall_back_and_mistyped_values_raise():
    assert _triton_autotune_config_from_args(_yaml_args(None)) == AutotunePolicy()
    nulls = _yaml_args({'mode': 'pinned', 'table_path': None, 'verify_every': None})
    assert _triton_autotune_config_from_args(nulls) == AutotunePolicy(mode='pinned')
    quoted = _yaml_args({'mode': 'pinned', 'chaos': 'false'})
    with pytest.raises(TypeError, match='chaos must be a bool'):
        _triton_autotune_config_from_args(quoted)
    unknown = _yaml_args({'enumerate': True})
    with pytest.raises(TypeError, match='Unknown AutotunePolicy option'):
        _triton_autotune_config_from_args(unknown)


def test_cli_config_invariant_list(parser):
    empty = parser.parse_args(['--triton-autotune-config-invariant'])
    assert _triton_autotune_config_from_args(empty).config_invariant == ()
    named = parser.parse_args(['--triton-autotune-config-invariant', 'pkg.mod.kernel'])
    assert _triton_autotune_config_from_args(named).config_invariant == ('pkg.mod.kernel',)
