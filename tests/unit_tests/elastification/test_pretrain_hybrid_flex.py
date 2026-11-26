# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for optional ModelOpt integration in the Flextron entry point."""

import builtins
import importlib
import sys
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest

from megatron.training.state import TrainState


def test_missing_modelopt_plugin_uses_stock_hybrid_builder(monkeypatch, request):
    """A dependency-free config adapter must not make ModelOpt appear installed."""
    module_name = "megatron.elastification.pretrain_hybrid_flex"
    original_import = builtins.__import__

    def import_without_modelopt_plugin(name, *args, **kwargs):
        if name == "modelopt.torch.distill.plugins.megatron":
            raise ModuleNotFoundError("No module named 'modelopt'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_modelopt_plugin)
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    request.addfinalizer(lambda: sys.modules.pop(module_name, None))
    module = importlib.import_module(module_name)
    assert module.has_nvidia_modelopt is False

    args = Namespace(
        use_legacy_models=False,
        spec=["stock.module", "stock_spec"],
        max_position_embeddings=16,
        hybrid_layer_pattern=":M*:",
        n_loops_distribution="1:1,4:1",
        inference_n_loops=2,
        fp16_lm_cross_entropy=False,
        untie_embeddings_and_output_weights=False,
        position_embedding_type="rope",
        rotary_percent=1.0,
        rotary_base=10000,
    )
    model = Mock(decoder=SimpleNamespace(num_layers_per_pipeline_rank=0))
    stock_spec = object()
    run_config = SimpleNamespace(tokenizer=SimpleNamespace(padded_vocab_size=128))

    with (
        patch.object(module, "get_args", return_value=args),
        patch.object(module, "get_run_config", return_value=run_config),
        patch.object(module, "core_transformer_config_from_args", return_value=object()),
        patch.object(module, "import_module", return_value=stock_spec) as import_spec,
        patch.object(module, "HybridModel", return_value=model) as build_stock,
        patch.object(module, "_build_modelopt_stage") as build_modelopt,
        patch("megatron.elastification.flextron_utils.setup_flextron_model"),
        patch("megatron.elastification.flextron_utils.inject_flextron_forward_logic"),
    ):
        assert module.model_provider() is model

    import_spec.assert_called_once_with(args.spec)
    assert build_stock.call_args.kwargs["hybrid_stack_spec"] is stock_spec
    assert build_stock.call_args.kwargs["vocab_size"] == 128
    assert build_stock.call_args.kwargs["n_loops_distribution"] == "1:1,4:1"
    assert build_stock.call_args.kwargs["inference_n_loops"] == 2
    build_modelopt.assert_not_called()


def test_modelopt_stage_binds_the_tokenizer_vocab_size():
    """The stage config is translated after tokenizer init, so it takes the run's vocab size."""
    module = importlib.import_module("megatron.elastification.pretrain_hybrid_flex")
    built = object()
    builder = Mock()
    builder.build_model.return_value = built
    model_config = SimpleNamespace(
        vocab_size=None, pre_wrap_hooks=[], get_builder_cls=lambda: lambda config: builder
    )
    run_config = SimpleNamespace(
        tokenizer=SimpleNamespace(padded_vocab_size=128), checkpoint=None, distillation=None
    )

    with (
        patch.object(module, "get_run_config", return_value=run_config),
        patch.object(module, "hybrid_config_from_args"),
        patch.object(module, "apply_modelopt_model_config", return_value=model_config, create=True),
        patch.object(module, "modelopt_setup_hooks", return_value=[], create=True),
        patch.object(module, "distillation_hooks", return_value=[], create=True) as kd_hooks,
    ):
        assert module._build_modelopt_stage(Namespace(), True, True, None, object()) is built

    assert model_config.vocab_size == 128
    kd_hooks.assert_called_once_with(model_config, None)


@pytest.mark.parametrize("pattern", ["M*", ":M*:"])
def test_flextron_forward_step_keeps_recurrence_step_across_microbatches(monkeypatch, pattern):
    module = importlib.import_module("megatron.elastification.pretrain_hybrid_flex")
    args = Namespace(hybrid_layer_pattern=pattern, original_model_sample_prob=1.0)
    train_state = TrainState(iteration=11)
    stimer = MagicMock()
    stimer.return_value = stimer
    monkeypatch.setattr(module, "get_args", lambda: args)
    monkeypatch.setattr(module, "get_train_state", lambda: train_state)
    monkeypatch.setattr(module, "get_timers", Mock)
    monkeypatch.setattr(module, "get_batch", lambda _: (None,) * 7)
    monkeypatch.setattr(module, "get_grad_acc_based_random_choice", lambda **kwargs: 0.0)
    monkeypatch.setattr(module, "_logged_params_norm", True)
    monkeypatch.setattr(module, "stimer", stimer)
    model = Mock(training=True)

    module.forward_step(None, model)
    module.forward_step(None, model)
    train_state.iteration += 1
    module.forward_step(None, model)

    assert all(call.kwargs["budget"] == 1.0 for call in model.call_args_list)
    if pattern == ":M*:":
        assert [call.kwargs["recurrence_step"] for call in model.call_args_list] == [11, 11, 12]
    else:
        assert all("recurrence_step" not in call.kwargs for call in model.call_args_list)
