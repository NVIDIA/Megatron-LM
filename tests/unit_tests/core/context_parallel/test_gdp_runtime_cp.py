# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Runtime GDP head geometry and forward-local context-parallel views."""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.parallel_state import get_tensor_model_parallel_group
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm import gated_delta_product as gdp_module
from megatron.core.ssm.gated_delta_product import (
    GatedDeltaProductMixer,
    GatedDeltaProductMixerSubmodules,
)
from megatron.core.ssm.gdp_context_parallel import GDPContextParallel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.launch_on_gb200


class _FakeGroup:
    def __init__(self, rank: int, size: int):
        self._rank = rank
        self._size = size

    def rank(self) -> int:
        return self._rank

    def size(self) -> int:
        return self._size


@pytest.mark.parametrize("build_cp_size", [1, 4])
def test_chunkwise_cp_keeps_heads_when_cp_exceeds_head_count(monkeypatch, build_cp_size):
    """Chunkwise construction and runtime CP4 must accept only two TP-local heads."""
    # Exercise the actual constructor without requiring the external recurrence libraries.
    monkeypatch.setattr(gdp_module, "HAVE_MAMBA_SSM", True)
    monkeypatch.setattr(gdp_module, "HAVE_FLA", True)
    monkeypatch.setattr(gdp_module, "HAVE_FLA_GDP_CP", True)
    monkeypatch.setattr(gdp_module, "FLAGatedDeltaProductCPBackend", object)
    monkeypatch.setattr(
        gdp_module, "chunk_gated_delta_product", lambda **kwargs: None, raising=False
    )
    monkeypatch.setattr(gdp_module, "check_fla_sequence_packing_support", lambda: (True, None))

    class _Projection(torch.nn.Linear):
        def __init__(self, in_features, out_features, **kwargs):
            super().__init__(in_features, out_features)

    Utils.initialize_model_parallel()
    try:
        model_parallel_cuda_manual_seed(123)
        config = TransformerConfig(
            num_layers=1,
            hidden_size=32,
            num_attention_heads=2,
            mamba_num_heads=2,
            mamba_head_dim=16,
            mamba_num_groups=1,
            mamba_state_dim=8,
            linear_cp_mode="chunkwise",
            linear_cp_layout="contiguous",
            context_parallel_size=build_cp_size,
            dynamic_context_parallel=True,
            calculate_per_token_loss=True,
            max_seqlen_per_dp_cp_rank=128,
        )
        build_group = _FakeGroup(rank=0, size=build_cp_size)
        mixer = GatedDeltaProductMixer(
            config,
            GatedDeltaProductMixerSubmodules(in_proj=_Projection, out_proj=_Projection),
            d_model=32,
            rmsnorm=False,
            pg_collection=ProcessGroupCollection(
                tp=get_tensor_model_parallel_group(), cp=build_group
            ),
        )
        base_helper = mixer.cp
        assert (base_helper is None) == (build_cp_size > 1)
        assert mixer.nheads_local_cp == 2
        assert mixer.d_inner_local_cp == 32
        assert mixer.ngroups_local_cp == 1

        for cp_size in (1, 4, 2, 1):
            group = _FakeGroup(rank=0, size=cp_size)
            runtime = mixer._resolve_runtime_cp(
                PackedSeqParams(local_cp_size=cp_size, cp_group=group)
            )
            assert runtime.group is group
            assert runtime.chunkwise == (cp_size > 1)
            assert (runtime.d_inner, runtime.nheads, runtime.ngroups) == (32, 2, 1)
            if cp_size > 1:
                assert runtime.helper is None
            else:
                assert runtime.helper.cp_group is group
                assert runtime.helper.conv1d_cp1 is mixer.conv1d
                assert runtime.helper.dt_bias_cp1 is mixer.dt_bias
                # The head-divisibility restriction still applies to headwise CP.
                with pytest.raises(
                    AssertionError,
                    match="TP-local head count 2 must be divisible by runtime CP size 4",
                ):
                    runtime.helper.for_context_parallel_group(_FakeGroup(rank=0, size=4))
            assert mixer.cp is base_helper
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.parametrize("linear_cp_mode", ["headwise", "chunkwise"])
def test_runtime_cp_resolution_is_forward_local(linear_cp_mode):
    """Sequential CP1/2/4 resolution must not alter the mixer or base helper."""
    static_group = _FakeGroup(rank=0, size=1)
    runtime_groups = {size: _FakeGroup(rank=0, size=size) for size in (1, 2, 4)}
    conv = torch.nn.Conv1d(128, 128, kernel_size=4, groups=128)
    helper = GDPContextParallel(
        cp_group=static_group,
        d_inner_local_tp=64,
        nheads_local_tp=4,
        ngroups_local_tp=4,
        d_state=16,
        num_householder=1,
        headdim=16,
        conv1d_cp1=conv,
        dt_bias_cp1=torch.randn(4),
        A_log_cp1=torch.randn(4),
        D_cp1=None,
        D_has_hdim=False,
    )
    mixer = GatedDeltaProductMixer.__new__(GatedDeltaProductMixer)
    mixer.pg_collection = SimpleNamespace(cp=static_group)
    mixer.cp = helper
    mixer.config = SimpleNamespace(linear_cp_mode=linear_cp_mode)
    mixer.chunkwise_cp_backend = object() if linear_cp_mode == "chunkwise" else None
    mixer.d_inner_local_tp = 64
    mixer.nheads_local_tp = 4
    mixer.ngroups_local_tp = 4

    for cp_size in (1, 2, 4, 2, 1):
        runtime = mixer._resolve_runtime_cp(
            PackedSeqParams(local_cp_size=cp_size, cp_group=runtime_groups[cp_size])
        )
        assert runtime.group is runtime_groups[cp_size]
        assert runtime.chunkwise is (linear_cp_mode == "chunkwise" and cp_size > 1)
        if runtime.chunkwise:
            assert runtime.helper is None
        else:
            assert runtime.helper.cp_size == cp_size
        expected_heads = 4 if runtime.chunkwise else 4 // cp_size
        assert runtime.nheads == expected_heads

        assert mixer.cp is helper
        assert helper.cp_group is static_group
        assert helper.cp_size == 1
