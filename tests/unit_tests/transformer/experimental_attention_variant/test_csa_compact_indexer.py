# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Construction failures and model-local compact workspace ownership."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from megatron.core.ops.attention.csa.backends import select_compact_indexer
from megatron.core.transformer.experimental_attention_variant.dsv4_layer_config import (
    get_compact_indexer_workspace_registry,
)


def test_bf16_does_not_import_compact_dependencies():
    with patch("megatron.core.ops.attention.csa.backends.require", side_effect=AssertionError):
        assert select_compact_indexer("bf16", workspaces={}) is None


def test_mxfp8_rejects_unsupported_gpu_before_loading_kernels():
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.get_device_capability", return_value=(9, 0)),
        patch("megatron.core.ops.attention.csa.backends.require", side_effect=AssertionError),
        pytest.raises(RuntimeError, match="SM100"),
    ):
        select_compact_indexer("mxfp8", workspaces={})


def test_compact_workspace_registry_survives_layer_copies_but_is_model_local():
    model = SimpleNamespace()
    registry = get_compact_indexer_workspace_registry(model)
    layer = deepcopy(model)
    assert get_compact_indexer_workspace_registry(layer) is registry
    assert get_compact_indexer_workspace_registry(SimpleNamespace()) is not registry
