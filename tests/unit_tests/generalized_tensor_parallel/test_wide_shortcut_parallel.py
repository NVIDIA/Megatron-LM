# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Wide ShortcutMoE forward/backward parity with real GTP/EGTP sharding."""

import pytest
import torch

from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols
from megatron.core.tensor_parallel.gtp_api import HAVE_GTP

if not HAVE_GTP:
    pytest.skip("GTP requires TransformerEngine >= 2.19", allow_module_level=True)

from tests.unit_tests.ssm.test_wide_shortcut_parallel import _run_wide_shortcut_gtp_parity


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize(
    "compute_symbol", [Symbols.ATTENTION, Symbols.MAMBA], ids=["attention", "mamba"]
)
def test_wide_shortcut_gtp_egtp_matches_unsharded_forward_backward(compute_symbol):
    _run_wide_shortcut_gtp_parity(compute_symbol)
