# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Blockwise FP8 gather ON/OFF parity for GDP-AdamW and MoE-Muon, without GTP."""

import pytest

from megatron.core.optimizer import HAVE_EMERGING_OPTIMIZERS
from megatron.core.utils import is_te_min_version
from tests.unit_tests.fp8_gdp_test_utils import (
    FLA_SEQUENCE_PACKING_REASON,
    HAVE_GDP_DEPS,
    run_gdp_gather_parity,
)
from tests.unit_tests.test_fp8_param import fp8_available, reason_for_no_fp8

pytestmark = pytest.mark.launch_on_gb200


class TestMuonBlockwiseFP8ParamGather:
    @pytest.mark.skipif(not is_te_min_version("2.3.0.dev0"), reason="TE 2.3.0.dev0 is required")
    @pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
    @pytest.mark.skipif(
        not HAVE_EMERGING_OPTIMIZERS, reason="emerging-optimizers package is required"
    )
    @pytest.mark.skipif(
        not HAVE_GDP_DEPS,
        reason=(
            FLA_SEQUENCE_PACKING_REASON or "GDP requires mamba-ssm, einops, FLA, and causal-conv1d"
        ),
    )
    @pytest.mark.parametrize("overlap", [False, True])
    @pytest.mark.parametrize("use_layer_wise_param_layout", [False, True])
    def test_gdp_adamw_moe_muon_blockwise_param_gather(
        self, overlap, use_layer_wise_param_layout, monkeypatch
    ):
        """Blockwise gather ON/OFF must preserve GDP-AdamW + grouped-MoE Muon losses.

        Both layouts use native Adam FP8 transport and implicit Muon gradient reuse without
        the MXFP8 flag. Check the BF16 Muon router and fused convolution's actual parameter
        reads; the first post-update loss guards master initialization from high-precision values.
        """
        run_gdp_gather_parity("blockwise", overlap, use_layer_wise_param_layout, monkeypatch)
