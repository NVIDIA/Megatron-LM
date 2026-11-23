# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adapted from https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/53c20f81cb575f19af34ff0ae1749fb13b07999b/tests/unit_tests/models/test_gtp_conversion.py
# Source license: Apache-2.0.

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from megatron.bridge.models.conversion.gtp import (
    _gather_gtp_weight,
    _get_mapping_shape,
    _slice_gtp_weight,
)
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge, WeightConversionTask
from megatron.bridge.models.conversion.param_mapping import (
    ColumnParallelMapping,
    DirectMapping,
    GatedMLPMapping,
    RowParallelMapping,
)


class _Bridge(MegatronModelBridge):
    def mapping_registry(self):
        return MegatronMappingRegistry()


def _shard(shape, *, rank=1, size=2, padding=0):
    weight = torch.nn.Parameter(torch.zeros(shape))
    weight.is_gtp_weight_remat = True
    weight.group = Mock()
    weight.group.rank.return_value = rank
    weight.group.size.return_value = size
    weight.pad_length = padding
    return weight


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("padding", [0, 1, 3])
def test_import_slices_rows_and_zero_pads(rank, padding):
    param = _shard((3, 4), rank=rank, padding=padding)
    logical = torch.arange((6 - padding) * 4).reshape(6 - padding, 4).float()
    actual = _slice_gtp_weight(logical, param)
    expected = torch.cat((logical, torch.zeros(padding, 4)))[rank * 3 : (rank + 1) * 3]
    assert torch.equal(actual, expected)
    assert actual.is_contiguous()
    assert _get_mapping_shape(param) == logical.shape


def test_import_rejects_wrong_logical_shape():
    with pytest.raises(ValueError, match="TP-local GTP weight shape"):
        _slice_gtp_weight(torch.ones(3, 4), _shard((3, 4)))


def test_non_gtp_conversion_keeps_tensor_identity():
    param = torch.nn.Parameter(torch.ones(3, 4))
    source = torch.zeros_like(param)
    assert _slice_gtp_weight(source, param) is source
    assert _gather_gtp_weight(param) is param
    assert _gather_gtp_weight(None) is None
    assert _get_mapping_shape(param) == param.shape


def test_local_hf_views_reject_unrepresentable_gtp_sharding():
    param = _shard((3, 4))
    param.data = param.data.bfloat16()
    with pytest.raises(ValueError, match="cannot represent GTP sharding"):
        DirectMapping("weight", "hf.weight").local_hf_params(
            param, global_param_name="weight", megatron_module=torch.nn.Module()
        )


def test_export_gathers_rows_and_trims_padding(monkeypatch):
    param = _shard((3, 4), padding=1)
    logical_padded = torch.arange(24).reshape(6, 4).float()

    def gather(output, local, *, group):
        assert group is param.group
        assert type(local) is torch.Tensor
        assert local.shape == param.shape
        output.copy_(logical_padded)

    monkeypatch.setattr(torch.distributed, "all_gather_into_tensor", gather)
    assert torch.equal(_gather_gtp_weight(param), logical_padded[:5])


@pytest.mark.parametrize(
    "mapping_type", [ColumnParallelMapping, RowParallelMapping, GatedMLPMapping]
)
def test_tp_scatter_uses_shape_before_gtp_sharding(mapping_type, monkeypatch):
    monkeypatch.setattr(
        "megatron.bridge.models.conversion.param_mapping.get_pg_size", lambda group: group.size()
    )
    monkeypatch.setattr(
        "megatron.bridge.models.conversion.param_mapping.get_pg_rank", lambda group: group.rank()
    )
    module = torch.nn.Module()
    module.weight = _shard((3, 4), padding=0)
    if mapping_type is GatedMLPMapping:
        mapping = mapping_type("weight", gate="gate", up="up")
        source = {"gate": torch.ones(6, 4), "up": torch.full((6, 4), 2.0)}
    else:
        mapping = mapping_type("weight", "hf.weight")
        source = torch.ones(12, 4) if mapping_type is ColumnParallelMapping else torch.ones(6, 8)
    group = Mock()
    group.size.return_value = 2
    group.rank.return_value = 0
    mapping._tp_group = group

    def scatter(splits, shape, dtype, device):
        assert shape == torch.Size((6, 4))
        assert splits[0].shape == shape
        return splits[0]

    mapping.scatter_to_tp_ranks = scatter
    result = mapping.hf_to_megatron(source, module)
    assert result.shape == (6, 4)


@pytest.mark.parametrize("streaming", [False, True])
def test_hf_import_loads_local_gtp_shard(monkeypatch, streaming):
    bridge = _Bridge()
    module = torch.nn.Module()
    module.weight = _shard((3, 4), rank=1, padding=1)
    source = torch.arange(20).reshape(5, 4).float()
    mapping = Mock()
    mapping.hf_param = "hf.weight"
    mapping.is_grouped_export = False
    mapping.hf_to_megatron.return_value = source
    task = WeightConversionTask(
        param_name="weight",
        global_param_name="weight",
        mapping=mapping,
        megatron_module=module,
        param_weight=module.weight,
        vp_stage=0,
    )
    hf = SimpleNamespace(state={"hf.weight": source}, model_name_or_path="toy")
    monkeypatch.setattr(bridge, "build_conversion_tasks", lambda *args: [task])
    monkeypatch.setattr(bridge, "finalize_hf_import", lambda *args: None)
    if streaming:
        [result] = bridge.stream_weights_hf_to_megatron(hf, [module], [task])
        actual = result.weight
    else:
        bridge.load_weights_hf_to_megatron(hf, [module])
        actual = module.weight
    expected = torch.cat((source[3:], torch.zeros(1, 4)))
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("copy_error", [False, True])
def test_native_fp8_hf_import_preserves_gtp_storage_context(monkeypatch, copy_error):
    """TE copy must run inside Core's temporary native-FP8 class restoration."""
    import megatron.core.tensor_parallel.gtp_api as gtp_api

    bridge = _Bridge()
    module = torch.nn.Module()
    module.weight = _shard((3, 4), rank=1, padding=1)
    module.weight._gtp_native_fp8 = True
    source = torch.arange(20).reshape(5, 4).float()
    events = []
    original_copy = torch.nn.Parameter.copy_

    @contextmanager
    def native_load_context(owner):
        assert owner is module
        events.append("enter")
        try:
            yield
        finally:
            events.append("exit")

    def copy_with_te_check(param, value):
        assert events == ["enter"], "Native FP8 copy ran outside Core's load context"
        events.append("copy")
        if copy_error:
            raise RuntimeError("TE copy failed")
        return original_copy(param, value)

    monkeypatch.setattr(gtp_api, "gtp_native_fp8_load_context", native_load_context, raising=False)
    monkeypatch.setattr(torch.nn.Parameter, "copy_", copy_with_te_check)
    mapping = Mock(hf_param="hf.weight", is_grouped_export=False)
    mapping.hf_to_megatron.return_value = source
    task = WeightConversionTask(
        "weight", "weight", mapping, megatron_module=module, param_weight=module.weight
    )
    monkeypatch.setattr(bridge, "build_conversion_tasks", lambda *args: [task])
    monkeypatch.setattr(bridge, "finalize_hf_import", lambda *args: None)

    hf = SimpleNamespace(state={"hf.weight": source}, model_name_or_path="toy")
    if copy_error:
        with pytest.raises(RuntimeError, match="TE copy failed"):
            bridge.load_weights_hf_to_megatron(hf, [module])
    else:
        bridge.load_weights_hf_to_megatron(hf, [module])

    assert events == ["enter", "copy", "exit"]
    assert module.weight.is_gtp_weight_remat
    expected = torch.zeros(3, 4) if copy_error else torch.cat((source[3:], torch.zeros(1, 4)))
    assert torch.equal(module.weight, expected)


@pytest.mark.parametrize("tp_size", [1, 2])
def test_quantized_export_gathers_gtp_before_computing_values_and_scales(monkeypatch, tp_size):
    bridge = _Bridge()
    model = torch.nn.Module()
    model.config = SimpleNamespace(share_embeddings_and_output_weights=False)
    model.weight = _shard((3, 4), padding=2)
    logical = torch.arange(16).reshape(4, 4).float()
    monkeypatch.setattr(
        torch.distributed,
        "all_gather_into_tensor",
        lambda output, local, **kwargs: output.copy_(torch.cat((logical, torch.zeros(2, 4)))),
    )
    monkeypatch.setattr(
        "megatron.bridge.models.conversion.param_mapping.get_pg_size", lambda group: tp_size
    )
    monkeypatch.setattr(
        "megatron.bridge.models.conversion.quant_bridge.unwrap_model", lambda models: models
    )
    mapping = ColumnParallelMapping("weight", "hf.weight")
    monkeypatch.setattr(mapping, "broadcast_from_pp_rank", lambda weight, **kwargs: weight)
    monkeypatch.setattr(
        mapping,
        "gather_from_tp_ranks",
        lambda weight: [weight + rank * 32 for rank in range(tp_size)],
    )
    task = WeightConversionTask(
        "weight", "weight", mapping, megatron_module=model, param_weight=model.weight
    )

    def quantize(weight, block_size):
        assert weight.shape == (4, 4), "Quantization received a stored GTP shard"
        assert block_size == (2, 2)
        return weight.to(torch.int8), weight.reshape(2, 2, 2, 2).amax(dim=(1, 3))

    exported = dict(
        bridge.stream_weights_megatron_to_hf_quant(
            [model],
            SimpleNamespace(state={}),
            lambda name: True,
            quantize,
            quant_block_size=(2, 2),
            conversion_tasks=[task],
            show_progress=False,
        )
    )
    expected_weights = torch.cat([logical + rank * 32 for rank in range(tp_size)]).to(torch.int8)
    expected_scales = torch.cat(
        [torch.tensor([[5.0, 7.0], [13.0, 15.0]]) + rank * 32 for rank in range(tp_size)]
    )
    assert torch.equal(exported["hf.weight"], expected_weights)
    assert torch.equal(exported["hf.weight_scale_inv"], expected_scales)


@pytest.mark.parametrize("remote_pp", [False, True])
def test_raw_fp8_detection_rejects_gtp_on_every_pp_rank(monkeypatch, remote_pp):
    bridge = _Bridge()
    model = torch.nn.Module()
    model.weight = _shard((4, 4), rank=0)
    model.weight.get_metadata = lambda: {
        "is_2D_scaled": True,
        "rowwise_data": torch.zeros(4, 4, dtype=torch.uint8),
        "rowwise_scale_inv": torch.ones(2, 2),
    }
    path = "megatron.bridge.models.conversion.model_bridge"
    monkeypatch.setattr(f"{path}.persistent_buffers", lambda module: [])
    monkeypatch.setattr(f"{path}._megatron_local_name_to_global", lambda *args: "weight")
    monkeypatch.setattr(f"{path}.get_pg_size", lambda group: 2)

    def gather(flags, local, **kwargs):
        if not remote_pp:
            assert local == {"weight": -1}
        flags[:] = [{"weight": -1}, {}]

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    with pytest.raises(ValueError, match="raw FP8 export does not support GTP"):
        bridge._detect_fp8_params(
            [] if remote_pp else [model], SimpleNamespace(), ["weight"], None, "rowwise_scale_inv"
        )
