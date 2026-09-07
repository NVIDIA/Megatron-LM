# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Regression tests for GDP tensor-parallel checkpoint resharding."""

import importlib.util
import inspect
import sys
from collections import defaultdict
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch

from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.ssm import gated_delta_product as gdp_module
from megatron.core.ssm.gated_delta_product import (
    GatedDeltaProductMixer,
    _get_in_proj_checkpoint_split_layout,
    _split_tensor_factory,
)


class _FakeProcessGroup:
    def __init__(self, rank, size):
        self._rank = rank
        self._size = size

    def rank(self):
        return self._rank

    def size(self):
        return self._size


def _load_gdp_module_with_fake_rmsnorm(monkeypatch):
    """Load GDP with a concrete RMSNorm base when mamba-ssm is unavailable."""
    for package_name in ("mamba_ssm", "mamba_ssm.ops", "mamba_ssm.ops.triton"):
        package = ModuleType(package_name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, package_name, package)

    layernorm_gated = ModuleType("mamba_ssm.ops.triton.layernorm_gated")
    layernorm_gated.RMSNorm = torch.nn.Module
    monkeypatch.setitem(sys.modules, layernorm_gated.__name__, layernorm_gated)

    module_name = "megatron.core.ssm._gated_delta_product_with_fake_rmsnorm"
    spec = importlib.util.spec_from_file_location(module_name, gdp_module.__file__)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def test_constructor_drops_unused_mamba_compatibility_arguments():
    """GDP dimensions come from TransformerConfig, not ignored constructor overrides."""
    parameters = inspect.signature(GatedDeltaProductMixer.__init__).parameters
    removed_parameters = {
        "expand",
        "dt_init",
        "dt_scale",
        "use_mem_eff_path",
        "d_state",
        "headdim",
        "ngroups",
    }
    assert parameters.keys().isdisjoint(removed_parameters)


def test_gdp_rmsnorm_checkpoint_wrap_uses_explicit_process_groups(monkeypatch):
    """Custom TP/DP+CP groups must not fall back to the global MPU groups."""
    gdp_with_rmsnorm = _load_gdp_module_with_fake_rmsnorm(monkeypatch)
    tp_group = _FakeProcessGroup(rank=2, size=4)
    dp_cp_group = _FakeProcessGroup(rank=5, size=8)
    norm = SimpleNamespace(tp_group=tp_group)
    norm.state_dict = lambda *args, **kwargs: {"weight": torch.arange(2)}

    with (
        patch("torch.distributed.is_initialized", return_value=True),
        patch(
            "megatron.core.parallel_state.get_tensor_model_parallel_group",
            side_effect=AssertionError("global TP group fallback"),
        ),
        patch(
            "megatron.core.parallel_state.get_data_parallel_group",
            side_effect=AssertionError("global DP+CP group fallback"),
        ),
    ):
        sharded_weight = gdp_with_rmsnorm.ExtendedRMSNorm.sharded_state_dict(
            norm, prefix="norm.", metadata={"dp_cp_group": dp_cp_group}
        )["norm.weight"]

    assert sharded_weight.global_offset == (4,)
    assert sharded_weight.global_shape == (8,)
    assert sharded_weight.replica_id == (0, 0, 5)


def test_gdp_checkpoint_threads_explicit_groups_to_all_wrappers():
    """GDP-owned parameters, conv1d, and child modules use the owning topology."""
    tp_group = object()
    dp_cp_group = object()
    in_proj_dim = 8
    conv_dim = 4

    in_proj = SimpleNamespace(weight=torch.empty(in_proj_dim, 1))
    conv1d = SimpleNamespace(
        state_dict=lambda *args, **kwargs: {"weight": torch.empty(conv_dim, 1, 1)}
    )
    mixer = SimpleNamespace(
        pg_collection=SimpleNamespace(tp=tp_group),
        in_proj=in_proj,
        d_inner_local_tp=2,
        num_householder=1,
        ngroups_local_tp=1,
        d_state=1,
        nheads_local_tp=1,
    )
    mixer._save_to_state_dict = lambda state_dict, *args, **kwargs: state_dict.update(
        A_log=torch.empty(1)
    )
    mixer.named_children = lambda: (("in_proj", in_proj), ("conv1d", conv1d))

    checkpoint_wrap_calls = []
    child_wrap_calls = []

    def checkpoint_wrap(state_dict, prefix, *args, **kwargs):
        checkpoint_wrap_calls.append(kwargs)
        return {
            f"{prefix}{name}": SimpleNamespace(data=tensor) for name, tensor in state_dict.items()
        }

    def child_wrap(module, prefix, *args, **kwargs):
        child_wrap_calls.append(kwargs)
        return {f"{prefix}weight": SimpleNamespace(data=module.weight)}

    with (
        patch(
            "megatron.core.ssm.gated_delta_product.make_sharded_tensors_for_checkpoint",
            side_effect=checkpoint_wrap,
        ),
        patch(
            "megatron.core.ssm.gated_delta_product.sharded_state_dict_default",
            side_effect=child_wrap,
        ),
        patch(
            "megatron.core.ssm.gated_delta_product._split_tensor_factory",
            side_effect=lambda tensor, *args, **kwargs: tensor,
        ),
        patch(
            "megatron.core.parallel_state.get_data_parallel_group",
            side_effect=AssertionError("global DP+CP group fallback"),
        ),
    ):
        GatedDeltaProductMixer.sharded_state_dict(
            mixer, prefix="mixer.", metadata={"dp_cp_group": dp_cp_group}
        )

    assert checkpoint_wrap_calls == [
        {
            "tensor_parallel_layers_axis_map": {"A_log": 0, "dt_bias": 0, "D": 0},
            "sharded_offsets": (),
            "tp_group": tp_group,
            "dp_cp_group": dp_cp_group,
        },
        {"tp_group": tp_group, "dp_cp_group": dp_cp_group},
    ]
    assert child_wrap_calls == [{"tp_group": tp_group}]


def test_householder_components_reshard_tp2_to_tp1_in_semantic_order():
    """Each householder copy must gather across TP ranks before copies are concatenated."""
    num_householder = 3
    local_sections, names = _get_in_proj_checkpoint_split_layout(
        d_inner_local_tp=2,
        group_state_local_tp=1,
        nheads_local_tp=1,
        num_householder=num_householder,
    )

    # Local layout is [z, V0, V1, V2, K0, K1, K2, Q, b0, b1, b2, a].
    rank_data = [
        torch.tensor([0, 1, 10, 11, 20, 21, 30, 31, 40, 50, 60, 70, 80, 90, 100, 110]),
        torch.tensor([2, 3, 12, 13, 22, 23, 32, 33, 41, 51, 61, 71, 81, 91, 101, 111]),
    ]

    checkpoint_chunks = defaultdict(list)
    for tp_rank, local_data in enumerate(rank_data):
        sharded_tensor = ShardedTensor.from_rank_offsets(
            "in_proj.weight", local_data, (0, tp_rank, 2)
        )
        factory = _split_tensor_factory(sharded_tensor, local_sections, names, split_dim=0)
        for chunk in factory.build():
            checkpoint_chunks[chunk.key].append(chunk.data)

    # Simulate DCP assembling every semantic checkpoint key for a TP=1 load.
    assembled_checkpoint = {
        key: torch.cat(chunks, dim=0) for key, chunks in checkpoint_chunks.items()
    }

    global_sections, global_names = _get_in_proj_checkpoint_split_layout(
        d_inner_local_tp=4,
        group_state_local_tp=2,
        nheads_local_tp=2,
        num_householder=num_householder,
    )
    target_tensor = ShardedTensor.from_rank_offsets(
        "in_proj.weight", torch.empty(sum(global_sections), dtype=torch.int64), (0, 0, 1)
    )
    target_factory = _split_tensor_factory(
        target_tensor, global_sections, global_names, split_dim=0
    )
    loaded_chunks = [assembled_checkpoint[chunk.key] for chunk in target_factory.build()]
    reloaded = target_factory.merge_fn(loaded_chunks)

    expected = torch.tensor(
        [
            0,
            1,
            2,
            3,
            10,
            11,
            12,
            13,
            20,
            21,
            22,
            23,
            30,
            31,
            32,
            33,
            40,
            41,
            50,
            51,
            60,
            61,
            70,
            71,
            80,
            81,
            90,
            91,
            100,
            101,
            110,
            111,
        ]
    )
    torch.testing.assert_close(reloaded, expected)
