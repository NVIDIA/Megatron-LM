# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""GDP AdamW + MoE Muon MXFP8 parameter-gather parity under ordinary DDP."""

import pytest

from megatron.core.optimizer import HAVE_EMERGING_OPTIMIZERS
from megatron.training.utils import get_device_arch_version
from tests.unit_tests.fp8_gdp_test_utils import (
    FLA_SEQUENCE_PACKING_REASON,
    HAVE_GDP_DEPS,
    run_gdp_gather_parity,
)
from tests.unit_tests.test_fp8_param import fp8_available, reason_for_no_fp8


class TestMuonMXFP8ParamGather:
    """Validate both LayerWise layouts without weight rematerialization."""

    @pytest.mark.launch_on_gb200
    @pytest.mark.skipif(
        get_device_arch_version() < 10, reason="MXFP8 is supported since Blackwell architecture"
    )
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
    def test_gdp_adamw_moe_muon_mxfp8_param_gather(
        self, overlap, use_layer_wise_param_layout, monkeypatch
    ):
        """MXFP8 gather ON/OFF must agree for both layouts and overlap modes.

        Compact keeps scalar Adam inside LayerWise; padded retains a separate Adam DistOpt.
        Real GDP exclusions and grouped-MoE weights exercise mixed ownership and DP sync.
        Disable GTP for ON/OFF parity and check the first post-update loss for master initialization.
        """
        run_gdp_gather_parity("mxfp8", overlap, use_layer_wise_param_layout, monkeypatch)
