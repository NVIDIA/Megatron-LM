# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Reject scheduler-selected head geometries before any rank starts training."""

from types import SimpleNamespace

import pytest

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm import gated_delta_product, mamba_mixer
from megatron.core.ssm.gated_delta_net import common, gdn, gdn2, kda
from megatron.core.ssm.kda_layer_config import KDALayerConfig
from megatron.core.ssm.utils import _validate_dynamic_cp_head_geometry
from megatron.core.transformer import TransformerConfig

pytestmark = pytest.mark.launch_on_gb200


def _group(size):
    return SimpleNamespace(size=lambda: size, rank=lambda: 0)


@pytest.mark.parametrize(
    "mixer_module,mixer_cls",
    [
        (mamba_mixer, mamba_mixer.MambaMixer),
        (gated_delta_product, gated_delta_product.GatedDeltaProductMixer),
        (gdn, gdn.GatedDeltaNet),
        (gdn2, gdn2.GatedDeltaNet2),
    ],
)
def test_dynamic_cp_rejects_incompatible_heads_before_parameter_creation(
    monkeypatch, mixer_module, mixer_cls
):
    """A CP1 build must reject CP4 at construction, even if CP2 would fit."""
    monkeypatch.setattr(mamba_mixer, "HAVE_MAMBA_SSM", True)
    monkeypatch.setattr(mamba_mixer, "HAVE_EINOPS", True)
    monkeypatch.setattr(gated_delta_product, "HAVE_MAMBA_SSM", True)
    monkeypatch.setattr(gated_delta_product, "HAVE_FLA", True)
    monkeypatch.setattr(
        gated_delta_product, "check_fla_sequence_packing_support", lambda: (True, "")
    )
    monkeypatch.setattr(gated_delta_product, "chunk_gated_delta_product", None, raising=False)
    monkeypatch.setattr(common, "HAVE_FLA", True)

    def no_parameter_creation(*args, **kwargs):
        pytest.fail("parameter construction happened before the dynamic CP head check")

    monkeypatch.setattr(mixer_module, "build_module", no_parameter_creation, raising=False)
    monkeypatch.setattr(common, "build_module", no_parameter_creation)
    config = TransformerConfig(
        num_layers=1,
        hidden_size=32,
        num_attention_heads=2,
        mamba_num_heads=2,
        mamba_head_dim=16,
        mamba_num_groups=1,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        dynamic_context_parallel=True,
        calculate_per_token_loss=True,
        max_seqlen_per_dp_cp_rank=128,
    )
    kwargs = dict(
        config=config,
        submodules=None,
        pg_collection=ProcessGroupCollection(tp=_group(1), cp=_group(1), dp_cp=_group(4)),
    )
    if mixer_cls in (mamba_mixer.MambaMixer, gated_delta_product.GatedDeltaProductMixer):
        kwargs["d_model"] = config.hidden_size
    with pytest.raises(ValueError, match=r"runtime CP size 4.*heads=2"):
        mixer_cls(**kwargs)


@pytest.mark.parametrize(
    "total_size,min_size,heads,groups", [(4, 1, 4, 1), (4, 2, 8, 2), (6, 2, 12, 1), (8, 4, 8, 2)]
)
def test_dynamic_cp_accepts_all_scheduler_enabled_head_sizes(total_size, min_size, heads, groups):
    config = SimpleNamespace(
        dynamic_context_parallel=True, min_dynamic_context_parallel_size=min_size
    )
    _validate_dynamic_cp_head_geometry(
        config,
        ProcessGroupCollection(dp_cp=_group(total_size)),
        head_counts={"heads": heads},
        group_count=groups,
    )


def test_dynamic_cp_checks_full_non_power_of_two_group():
    config = SimpleNamespace(dynamic_context_parallel=True, min_dynamic_context_parallel_size=2)
    with pytest.raises(ValueError, match=r"runtime CP size 6"):
        _validate_dynamic_cp_head_geometry(
            config, ProcessGroupCollection(dp_cp=_group(6)), head_counts={"heads": 8}
        )


def test_static_cp_does_not_require_dynamic_groups():
    _validate_dynamic_cp_head_geometry(
        SimpleNamespace(dynamic_context_parallel=False),
        ProcessGroupCollection(),
        head_counts={"heads": 2},
    )


@pytest.mark.parametrize("heads", [(2, 4), (4, 2)])
def test_runtime_gdn_rejects_empty_or_fractional_head_shards(heads):
    mixer = SimpleNamespace(num_k_heads_local_tp=heads[0], num_v_heads_local_tp=heads[1])
    with pytest.raises(ValueError, match="runtime CP size 4"):
        common._GDNBase._validate_runtime_cp_size(mixer, 4)


def test_dynamic_cp_validates_only_injected_scheduler_sizes():
    """A scheduler restricted to CP1/CP2 cannot select the incompatible CP4 size."""
    config = SimpleNamespace(dynamic_context_parallel=True, min_dynamic_context_parallel_size=1)
    _validate_dynamic_cp_head_geometry(
        config,
        ProcessGroupCollection(dp_cp=_group(4), dynamic_cp={1: _group(1), 2: _group(2)}),
        head_counts={"heads": 2},
        group_count=1,
    )


@pytest.mark.parametrize("build_cp_size,dynamic_cp", [(4, False), (1, True), (4, True)])
def test_chunkwise_kda_keeps_heads_when_cp_exceeds_head_count(
    monkeypatch, build_cp_size, dynamic_cp
):
    """Preserve main's chunkwise KDA geometry at static and scheduler-selected CP4."""
    monkeypatch.setattr(common, "HAVE_FLA", True)
    monkeypatch.setattr(kda, "HAVE_FLA", True)
    monkeypatch.setattr(kda, "HAVE_FLA_KDA", True)
    monkeypatch.setattr(kda, "chunk_kda", lambda *args, **kwargs: None)

    def first_projection(*args, **kwargs):
        # Stop before allocating parameters: reaching the first projection proves
        # both the static and runtime CP head-sharding guards were bypassed.
        raise RuntimeError("reached first KDA projection")

    monkeypatch.setattr(common, "build_module", first_projection)
    config = KDALayerConfig(
        num_layers=1,
        hidden_size=32,
        num_attention_heads=2,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        context_parallel_size=build_cp_size,
        linear_cp_mode="chunkwise",
        linear_cp_layout="contiguous",
        dynamic_context_parallel=dynamic_cp,
        calculate_per_token_loss=True,
        max_seqlen_per_dp_cp_rank=128,
    )
    groups = ProcessGroupCollection(tp=_group(1), cp=_group(build_cp_size), dp_cp=_group(4))
    with pytest.raises(RuntimeError, match="reached first KDA projection"):
        kda.KimiDeltaAttention(
            config=config, submodules=SimpleNamespace(in_proj=None), pg_collection=groups
        )
