# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for optional ModelOpt integration in the Flextron entry point."""

import builtins
import importlib
import sys
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import Mock, patch


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
        padded_vocab_size=128,
        max_position_embeddings=16,
        hybrid_layer_pattern="M*",
        fp16_lm_cross_entropy=False,
        untie_embeddings_and_output_weights=False,
        position_embedding_type="rope",
        rotary_percent=1.0,
        rotary_base=10000,
    )
    model = Mock(decoder=SimpleNamespace(num_layers_per_pipeline_rank=0))
    stock_spec = object()

    with (
        patch.object(module, "get_args", return_value=args),
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
    build_modelopt.assert_not_called()
