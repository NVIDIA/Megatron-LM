"""Tests for config construction before training runtime initialization."""

from argparse import ArgumentParser, Namespace
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from megatron.core.tokenizers.utils.build_tokenizer import build_tokenizer
from megatron.core.transformer import TransformerConfig
from megatron.training import arguments, global_vars
from megatron.training.argument_utils import (
    gpt_config_from_args,
    hybrid_config_from_args,
    resolve_tokenizer_vocab_size,
)
from megatron.training.config.training_config import TokenizerConfig


@pytest.fixture
def isolated_globals(monkeypatch):
    """Avoid changing services owned by the distributed test harness."""
    for name in ("_GLOBAL_ARGS", "_GLOBAL_TOKENIZER"):
        monkeypatch.setattr(global_vars, name, None)


def _runtime_args():
    return Namespace(
        rank=0,
        global_batch_size=8,
        micro_batch_size=2,
        data_parallel_size=2,
        gtp_weight_remat_size=1,
        decrease_batch_size_if_needed=False,
        step_batch_size_schedule=None,
        seq_length=32,
        padded_vocab_size=None,
        enable_experimental=False,
        exit_signal_handler=False,
        exit_signal_handler_for_training=False,
        disable_jit_fuser=False,
    )


@pytest.mark.parametrize("experimental", [False, True])
def test_parse_only_prepares_args(monkeypatch, isolated_globals, experimental):
    args = Namespace(use_checkpoint_args=False, yaml_cfg=None, enable_experimental=experimental)
    monkeypatch.setattr(arguments, "parse_args", Mock(return_value=args))
    monkeypatch.setattr(arguments, "validate_args", Mock())
    services = Mock()
    experimental_flag = Mock()
    monkeypatch.setattr(global_vars, "initialize_runtime_services", services)
    monkeypatch.setattr(arguments, "set_experimental_flag", experimental_flag)

    assert arguments.parse_and_validate_args() is args
    assert global_vars.get_args() is args
    services.assert_not_called()
    if experimental:
        experimental_flag.assert_called_once_with(True)
    else:
        experimental_flag.assert_not_called()


def test_args_only_bootstrap_registers_args_and_constructs_services(monkeypatch, isolated_globals):
    args = _runtime_args()
    initialize = Mock()
    monkeypatch.setattr(global_vars, "initialize_runtime_services", initialize)
    global_vars.set_global_variables(args)
    assert global_vars.get_args() is args
    initialize.assert_called_once_with(args, build_tokenizer=True)
    with pytest.raises(AssertionError, match="already initialized"):
        global_vars.set_global_variables(args)


def test_parse_restores_checkpoint_args_before_validation_and_config(monkeypatch, isolated_globals):
    from megatron.training import checkpointing

    args = Namespace(
        use_checkpoint_args=True,
        yaml_cfg=None,
        enable_experimental=False,
        load="resume",
        pretrained_checkpoint="pretrained",
        non_persistent_ckpt_type=None,
        padded_vocab_size=None,
    )
    events = []
    monkeypatch.setattr(arguments, "parse_args", Mock(return_value=args))

    def restore(received_args, load_arg="load"):
        assert received_args is args
        events.append(load_arg)
        if load_arg == "load":
            received_args.padded_vocab_size = 512

    def validate(received_args, defaults):
        assert received_args.padded_vocab_size == 512
        events.append("validate")

    monkeypatch.setattr(checkpointing, "load_args_from_checkpoint", restore)
    monkeypatch.setattr(arguments, "validate_args", validate)
    runtime = Mock()
    monkeypatch.setattr(global_vars, "initialize_runtime_services", runtime)

    assert arguments.parse_and_validate_args() is args
    assert events == ["pretrained_checkpoint", "load", "validate"]
    assert global_vars.get_args().padded_vocab_size == 512
    runtime.assert_not_called()


@pytest.mark.parametrize("adapter", [gpt_config_from_args, hybrid_config_from_args])
@pytest.mark.parametrize("checkpoint_vocab", [None, 256])
def test_vocabulary_resolves_after_config_construction(
    monkeypatch, isolated_globals, adapter, checkpoint_vocab
):
    parser = ArgumentParser()
    arguments.add_megatron_arguments(parser)
    args = parser.parse_args([])
    args.padded_vocab_size = checkpoint_vocab
    args.vocab_size = None
    model_cfg = adapter(
        args,
        config=TransformerConfig(num_layers=2, hidden_size=128, num_attention_heads=4),
        vocab_size_from_tokenizer=True,
    )
    assert model_cfg.vocab_size == checkpoint_vocab
    cfg = SimpleNamespace(model=model_cfg, tokenizer=TokenizerConfig())
    global_vars.set_args(args)

    def initialize_services(received_args):
        # Config construction has finished without creating a tokenizer.
        assert cfg.model.vocab_size == checkpoint_vocab
        assert received_args is args
        if received_args.padded_vocab_size is None:
            received_args.padded_vocab_size = 128

    initialize = Mock(side_effect=initialize_services)
    monkeypatch.setattr(global_vars, "initialize_runtime_services", initialize)
    global_vars.initialize_runtime_services(args)
    resolve_tokenizer_vocab_size(cfg, args.padded_vocab_size)

    assert model_cfg.vocab_size == (128 if checkpoint_vocab is None else checkpoint_vocab)
    assert model_cfg.should_pad_vocab is False
    assert cfg.tokenizer.padded_vocab_size == model_cfg.vocab_size
    initialize.assert_called_once_with(args)


def test_runtime_service_order_and_microbatch_inputs(monkeypatch, isolated_globals):
    args = _runtime_args()
    global_vars.set_args(args)
    calls = []
    microbatches = Mock(side_effect=lambda **kwargs: calls.append("microbatches"))
    monkeypatch.setattr(global_vars, "init_num_microbatches_calculator", microbatches)

    for name in (
        "_build_tokenizer",
        "_set_tensorboard_writer",
        "_set_wandb_writer",
        "_set_one_logger",
        "_set_adlr_autoresume",
        "_set_timers",
        "_set_energy_monitor",
        "_set_telemetry",
    ):

        def record(received_args, service=name):
            assert received_args is args
            calls.append(service)

        monkeypatch.setattr(global_vars, name, record)

    global_vars.initialize_runtime_services(args)
    assert calls == [
        "microbatches",
        "_build_tokenizer",
        "_set_tensorboard_writer",
        "_set_wandb_writer",
        "_set_one_logger",
        "_set_adlr_autoresume",
        "_set_timers",
        "_set_energy_monitor",
        "_set_telemetry",
    ]
    microbatches.assert_called_once_with(
        rank=0,
        global_batch_size=8,
        micro_batch_size=2,
        data_parallel_size=2,
        decrease_batch_size_if_needed=False,
        step_batch_size_schedule=None,
        seq_length=32,
    )


def test_custom_model_config_does_not_receive_vocabulary(monkeypatch, isolated_globals):
    custom_model = SimpleNamespace()
    cfg = SimpleNamespace(model=custom_model, tokenizer=TokenizerConfig())
    resolve_tokenizer_vocab_size(cfg, 128)
    assert vars(custom_model) == {}
    assert cfg.tokenizer.padded_vocab_size == 128


@pytest.mark.parametrize("adapter", [gpt_config_from_args, hybrid_config_from_args])
@pytest.mark.parametrize("pad_vocab,checkpoint_vocab", [(True, None), (True, 512), (False, None)])
def test_config_first_matches_legacy_tokenizer_and_model_inputs(
    monkeypatch, isolated_globals, adapter, pad_vocab, checkpoint_vocab
):
    parser = ArgumentParser()
    arguments.add_megatron_arguments(parser)
    args = parser.parse_args([])
    args.rank = 0
    args.tokenizer_type = "NullTokenizer"
    args.vocab_size = 133
    args.pad_vocab_size = pad_vocab
    args.padded_vocab_size = checkpoint_vocab
    args.tensor_model_parallel_size = 2
    transformer = TransformerConfig(num_layers=2, hidden_size=128, num_attention_heads=4)

    legacy_args = deepcopy(args)
    legacy_tokenizer = build_tokenizer(legacy_args)
    legacy_model = adapter(legacy_args, config=deepcopy(transformer))

    model = adapter(args, config=deepcopy(transformer), vocab_size_from_tokenizer=True)
    cfg = SimpleNamespace(model=model, tokenizer=TokenizerConfig())
    global_vars.set_args(args)

    def initialize(received_args):
        global_vars._build_tokenizer(received_args)

    monkeypatch.setattr(global_vars, "initialize_runtime_services", initialize)
    global_vars.initialize_runtime_services(args)
    resolve_tokenizer_vocab_size(cfg, args.padded_vocab_size)
    assert model.as_dict() == legacy_model.as_dict()
    assert cfg.tokenizer.padded_vocab_size == legacy_args.padded_vocab_size
    assert global_vars.get_tokenizer().vocab_size == legacy_tokenizer.vocab_size


def test_explicit_model_vocabulary_is_not_overwritten(monkeypatch, isolated_globals):
    from megatron.training.models import GPTModelConfig

    args = _runtime_args()
    args.padded_vocab_size = 128
    global_vars.set_args(args)
    monkeypatch.setattr(global_vars, "initialize_runtime_services", Mock())
    model = GPTModelConfig(
        transformer=TransformerConfig(num_layers=2, hidden_size=128, num_attention_heads=4),
        vocab_size=512,
        should_pad_vocab=False,
    )
    resolve_tokenizer_vocab_size(
        SimpleNamespace(model=model, tokenizer=TokenizerConfig()), args.padded_vocab_size
    )
    assert model.vocab_size == 512
    assert model.should_pad_vocab is False


@pytest.mark.parametrize("adapter", [gpt_config_from_args, hybrid_config_from_args])
def test_missing_tokenizer_vocabulary_fails_before_model_construction(adapter):
    parser = ArgumentParser()
    arguments.add_megatron_arguments(parser)
    args = parser.parse_args([])
    model = adapter(
        args,
        config=TransformerConfig(num_layers=2, hidden_size=128, num_attention_heads=4),
        vocab_size_from_tokenizer=True,
    )
    cfg = SimpleNamespace(model=model, tokenizer=TokenizerConfig())
    with pytest.raises(ValueError, match="vocabulary must be resolved"):
        resolve_tokenizer_vocab_size(cfg, None)


@pytest.mark.parametrize("adapter", [gpt_config_from_args, hybrid_config_from_args])
@pytest.mark.parametrize("raw_vocab", [None, 133])
def test_tokenizer_vocabulary_is_authoritative_when_padding(
    monkeypatch, isolated_globals, adapter, raw_vocab
):
    """Tokenizer metadata/special tokens may differ from an optional CLI size."""
    from megatron.core.tokenizers import MegatronTokenizer

    parser = ArgumentParser()
    arguments.add_megatron_arguments(parser)
    args = parser.parse_args([])
    args.rank = 0
    args.tokenizer_type = "HuggingFaceTokenizer"
    args.vocab_size = raw_vocab
    args.tensor_model_parallel_size = 2
    args.make_vocab_size_divisible_by = 128
    tokenizer = SimpleNamespace(vocab_size=261)
    factory = Mock(return_value=tokenizer)
    monkeypatch.setattr(MegatronTokenizer, "from_pretrained", factory)
    model = adapter(
        args,
        config=TransformerConfig(num_layers=2, hidden_size=128, num_attention_heads=4),
        vocab_size_from_tokenizer=True,
    )
    assert model.vocab_size is None
    factory.assert_not_called()
    cfg = SimpleNamespace(model=model, tokenizer=TokenizerConfig())
    global_vars.set_args(args)
    monkeypatch.setattr(global_vars, "initialize_runtime_services", global_vars._build_tokenizer)

    global_vars.initialize_runtime_services(args)
    resolve_tokenizer_vocab_size(cfg, args.padded_vocab_size)
    assert model.vocab_size == cfg.tokenizer.padded_vocab_size == 512
    assert model.should_pad_vocab is False
    assert global_vars.get_tokenizer() is tokenizer
    factory.assert_called_once()


@pytest.mark.parametrize("adapter", [gpt_config_from_args, hybrid_config_from_args])
@pytest.mark.parametrize("cli_vocab,expected", [(None, 512), (768, 768)])
def test_checkpoint_vocabulary_precedence_survives_runtime_setup(
    monkeypatch, isolated_globals, adapter, cli_vocab, expected
):
    """Exercise real checkpoint-argument restoration, not a prefilled namespace."""
    from megatron.training import checkpointing

    parser = ArgumentParser()
    arguments.add_megatron_arguments(parser)
    args = parser.parse_args([])
    args.rank = 0
    args.load = "checkpoint"
    args.tokenizer_type = "NullTokenizer"
    args.vocab_size = 133
    args.padded_vocab_size = cli_vocab
    args.tensor_model_parallel_size = 2
    checkpoint_args = Namespace(padded_vocab_size=512)
    state = {"args": checkpoint_args, "iteration": 17, "checkpoint_version": 3.0}
    monkeypatch.setattr(
        checkpointing,
        "_load_base_checkpoint",
        Mock(return_value=(state, "checkpoint", False, None)),
    )

    checkpointing.load_args_from_checkpoint(args)
    assert args.iteration == 17
    model = adapter(
        args,
        config=TransformerConfig(num_layers=2, hidden_size=128, num_attention_heads=4),
        vocab_size_from_tokenizer=True,
    )
    assert model.vocab_size == expected
    cfg = SimpleNamespace(model=model, tokenizer=TokenizerConfig())
    global_vars.set_args(args)
    monkeypatch.setattr(global_vars, "initialize_runtime_services", global_vars._build_tokenizer)
    global_vars.initialize_runtime_services(args)
    resolve_tokenizer_vocab_size(cfg, args.padded_vocab_size)

    assert model.vocab_size == cfg.tokenizer.padded_vocab_size == expected
    assert model.should_pad_vocab is False
    assert global_vars.get_tokenizer().vocab_size == 133


@pytest.mark.parametrize("adapter", [gpt_config_from_args, hybrid_config_from_args])
def test_known_unpadded_vocabulary_needs_no_tokenizer_for_config(adapter):
    parser = ArgumentParser()
    arguments.add_megatron_arguments(parser)
    args = parser.parse_args([])
    args.vocab_size = 133
    args.pad_vocab_size = False
    model = adapter(
        args,
        config=TransformerConfig(num_layers=2, hidden_size=128, num_attention_heads=4),
        vocab_size_from_tokenizer=True,
    )
    assert model.vocab_size == 133
    # Preserve the existing model-builder padding policy for raw vocabulary.
    assert model.should_pad_vocab is True


def test_vocabulary_resolution_neither_reads_globals_nor_initializes_services(monkeypatch):
    monkeypatch.setattr(
        global_vars, "get_args", Mock(side_effect=AssertionError("global args read"))
    )
    initialize = Mock(side_effect=AssertionError("runtime initialization"))
    monkeypatch.setattr(global_vars, "initialize_runtime_services", initialize)
    cfg = SimpleNamespace(model=None, tokenizer=TokenizerConfig())
    resolve_tokenizer_vocab_size(cfg, 256)
    assert cfg.tokenizer.padded_vocab_size == 256
    initialize.assert_not_called()
