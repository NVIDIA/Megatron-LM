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
# Adapted from https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/53c20f81cb575f19af34ff0ae1749fb13b07999b/tests/unit_tests/models/test_conversion_utils.py
# Source license: Apache-2.0.
# Follow-up fixes from https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/b57951ad292e43209f1cc7d4b5500768365a20ac/tests/unit_tests/models/test_conversion_utils.py

from collections import namedtuple

import pytest

from megatron.bridge.models.conversion.utils import (
    carry_rope_into_rope_parameters,
    conform_config_to_reference,
    mcore_to_hf_window_size,
    remove_non_pickleables,
)


class TestCarryRopeIntoRopeParameters:
    """The legacy RoPE keys ``megatron_to_hf_config`` emits must reach transformers 5 ``rope_parameters``."""

    REFERENCE = {
        "hidden_size": 64,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0},
    }

    def _carry(self, emitted, reference=None):
        return carry_rope_into_rope_parameters(
            emitted, conform_config_to_reference(emitted, reference or self.REFERENCE)
        )

    def test_projection_alone_backfills_the_reference_rope(self):
        assert conform_config_to_reference({"rope_theta": 500000.0}, self.REFERENCE)[
            "rope_parameters"
        ] == {"rope_type": "default", "rope_theta": 10000.0}

    def test_rope_theta_overrides_the_reference(self):
        assert self._carry({"rope_theta": 500000.0})["rope_parameters"] == {
            "rope_type": "default",
            "rope_theta": 500000.0,
        }

    def test_scaling_of_another_type_replaces_the_reference_scaling(self):
        reference = {
            "rope_parameters": {
                "rope_type": "yarn",
                "factor": 4.0,
                "beta_fast": 32,
                "original_max_position_embeddings": 4096,
                "rope_theta": 10000.0,
                "partial_rotary_factor": 0.5,
            }
        }
        emitted = {"rope_theta": 10000.0, "rope_scaling": {"type": "linear", "factor": 2.0}}

        assert self._carry(emitted, reference)["rope_parameters"] == {
            "rope_type": "linear",
            "factor": 2.0,
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.5,
        }

    def test_scaling_of_the_same_type_updates_only_changed_values(self):
        reference = {
            "rope_parameters": {
                "rope_type": "yarn",
                "factor": 4,
                "beta_fast": 32,
                "rope_theta": 10000,
            }
        }
        emitted = {
            "rope_theta": 10000.0,
            "rope_scaling": {"rope_type": "yarn", "factor": 8.0, "beta_fast": 32.0},
        }

        assert self._carry(emitted, reference)["rope_parameters"] == {
            "rope_type": "yarn",
            "factor": 8.0,
            "beta_fast": 32,
            "rope_theta": 10000,
        }

    def test_equal_values_keep_the_reference_rope_parameters_exactly(self):
        reference = {"rope_parameters": {"rope_type": "yarn", "factor": 4, "rope_theta": 10000}}
        emitted = {
            "rope_theta": 10000.0,
            "rope_scaling": {"rope_type": "yarn", "factor": 4.0, "truncate": True},
        }
        conformed = conform_config_to_reference(emitted, reference)

        assert carry_rope_into_rope_parameters(emitted, conformed) is conformed

    def test_default_scaling_drops_the_reference_scaling(self):
        reference = {
            "rope_parameters": {
                "rope_type": "llama3",
                "factor": 8.0,
                "low_freq_factor": 1.0,
                "high_freq_factor": 4.0,
                "original_max_position_embeddings": 8192,
                "rope_theta": 500000.0,
            }
        }
        emitted = {"rope_theta": 500000.0, "rope_scaling": {"rope_type": "default"}}

        assert self._carry(emitted, reference)["rope_parameters"] == {
            "rope_type": "default",
            "rope_theta": 500000.0,
        }

    def test_partial_rotary_factor_overrides_the_reference(self):
        reference = {
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10000.0,
                "partial_rotary_factor": 0.5,
            }
        }

        assert self._carry({"rope_theta": 10000.0, "partial_rotary_factor": 1.0}, reference)[
            "rope_parameters"
        ] == {"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 1.0}

    def test_missing_partial_rotary_factor_reads_as_one(self):
        emitted = {"rope_theta": 10000.0, "partial_rotary_factor": 1.0}
        conformed = conform_config_to_reference(emitted, self.REFERENCE)

        assert carry_rope_into_rope_parameters(emitted, conformed) is conformed
        assert self._carry({"rope_theta": 10000.0, "partial_rotary_factor": 0.25})[
            "rope_parameters"
        ] == {"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25}

    def test_top_level_partial_rotary_factor_is_left_to_the_projection(self):
        reference = {
            "partial_rotary_factor": 0.5,
            "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0},
        }

        carried = self._carry({"rope_theta": 10000.0, "partial_rotary_factor": 0.25}, reference)

        assert carried["partial_rotary_factor"] == 0.25
        assert carried["rope_parameters"] == {"rope_type": "default", "rope_theta": 10000.0}

    def test_model_keys_such_as_mrope_section_are_kept(self):
        reference = {
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 1e6,
                "mrope_section": [16, 24, 24],
            }
        }

        assert self._carry({"rope_theta": 5e6}, reference)["rope_parameters"] == {
            "rope_type": "default",
            "rope_theta": 5e6,
            "mrope_section": [16, 24, 24],
        }

    def test_nested_or_missing_rope_parameters_are_left_unchanged(self):
        nested = {
            "rope_parameters": {"full_attention": {"rope_type": "default", "rope_theta": 1e6}}
        }
        legacy = {"rope_theta": 10000.0}

        assert carry_rope_into_rope_parameters({"rope_theta": 5.0}, nested) is nested
        assert carry_rope_into_rope_parameters({"rope_theta": 5.0}, legacy) is legacy


class MetadataList(list):
    pass


class MetadataTuple(tuple):
    pass


MetadataPair = namedtuple("MetadataPair", ["head_dim", "callback"])


def test_remove_non_pickleables_preserves_torch_size():
    import torch

    original = torch.Size([8, 16])
    cleaned = remove_non_pickleables(original)
    assert type(cleaned) is torch.Size
    assert cleaned == original


@pytest.mark.parametrize("source_is_bf16", [True, False])
def test_weight_comparison_preserves_differences_between_fp32_and_bf16(source_is_bf16):
    from io import StringIO
    from types import SimpleNamespace

    import torch
    from rich.console import Console

    from megatron.bridge.models.conversion.utils import weights_verification_table

    bf16 = {"equal": torch.tensor([1.0], dtype=torch.bfloat16)}
    bf16["different"] = bf16["equal"].clone()
    fp32 = {"equal": torch.tensor([1.0]), "different": torch.tensor([1.001])}
    # Rounding the FP32 value down to BF16 would incorrectly hide this difference.
    source, exported = (bf16, fp32) if source_is_bf16 else (fp32, bf16)
    bridge = SimpleNamespace(
        hf_pretrained=SimpleNamespace(state=source),
        export_hf_weights=lambda model, show_progress: iter(exported.items()),
    )
    output = StringIO()
    Console(file=output, width=120, color_system=None).print(weights_verification_table(bridge, []))
    assert output.getvalue().count("✅") == 1
    assert output.getvalue().count("❌") == 1


@pytest.mark.parametrize(
    ("window_size", "expected"), [(None, None), (2048, 2048), ((2047, 0), 2048), ([2047, 0], 2048)]
)
def test_mcore_to_hf_window_size(window_size, expected):
    assert mcore_to_hf_window_size(window_size) == expected


def test_mcore_to_hf_window_size_rejects_malformed_pair():
    with pytest.raises(ValueError, match="two-element MCore window"):
        mcore_to_hf_window_size([2047])


def test_remove_non_pickleables_reads_raw_config_attributes():
    class HeterogeneousConfig:
        def __init__(self):
            self.num_key_value_heads = 8
            self.callback = lambda: None

        def __getattribute__(self, name):
            if name == "num_key_value_heads":
                raise RuntimeError("attribute must be read from the per-layer config")
            return super().__getattribute__(name)

    original = HeterogeneousConfig()

    cleaned = remove_non_pickleables(original)

    assert vars(cleaned)["num_key_value_heads"] == 8
    assert cleaned.callback is None
    assert vars(original)["callback"] is not None


@pytest.mark.parametrize("max_depth", [2, 3])
def test_remove_non_pickleables_serializes_nested_peft_hooks_without_mutation(max_depth):
    import pickle
    from types import SimpleNamespace

    hook = lambda model: model
    original = SimpleNamespace(
        num_attention_heads=4,
        num_query_groups=2,
        kv_channels=8,
        _pre_wrap_hooks=[hook],
        _megatron_bridge_setup_pre_wrap_hooks={"peft": hook},
        safe_metadata={"layers": [1, 2, 3]},
    )
    cleaned = remove_non_pickleables(original, max_depth=max_depth)
    restored = pickle.loads(pickle.dumps(cleaned))
    assert restored.num_attention_heads == 4
    assert restored.num_query_groups == 2
    assert restored.kv_channels == 8
    assert restored.safe_metadata == original.safe_metadata
    assert original._pre_wrap_hooks[0] is hook
    assert original._megatron_bridge_setup_pre_wrap_hooks["peft"] is hook
    assert "_pre_wrap_hooks" not in vars(cleaned)
    assert "_megatron_bridge_setup_pre_wrap_hooks" not in vars(cleaned)


@pytest.mark.parametrize("max_depth", [2, 3])
def test_remove_non_pickleables_preserves_mapping_metadata(max_depth):
    import pickle
    from collections import OrderedDict, defaultdict
    from types import SimpleNamespace

    hook = lambda: None
    registry = OrderedDict([(0, hook), (1, 8)])
    registry.description = "head dimensions"
    original = SimpleNamespace(registry=registry, defaults=defaultdict(int, head_dim=8))
    cleaned = remove_non_pickleables(original, max_depth=max_depth)
    restored = pickle.loads(pickle.dumps(cleaned))
    assert isinstance(restored.registry, OrderedDict)
    assert list(restored.registry.items()) == [(0, None), (1, 8)]
    assert restored.registry.description == registry.description
    assert restored.defaults.default_factory is int
    assert restored.defaults["head_dim"] == 8
    assert original.registry[0] is hook


def test_remove_non_pickleables_rejects_unsafe_subtree_without_losing_metadata():
    from types import SimpleNamespace

    hook = lambda: None
    original = SimpleNamespace(nested={"metadata": {"head_dim": 8, "callback": hook}})
    with pytest.raises(TypeError, match="attribute 'nested'.*max_depth=2"):
        remove_non_pickleables(original, max_depth=2)
    assert original.nested["metadata"] == {"head_dim": 8, "callback": hook}
    cleaned = remove_non_pickleables(original, max_depth=3)
    assert cleaned.nested["metadata"] == {"head_dim": 8, "callback": None}


@pytest.mark.parametrize("max_depth", [2, 3])
def test_remove_non_pickleables_cleans_defaultdict_factory(max_depth):
    import pickle
    from collections import defaultdict
    from types import SimpleNamespace

    factory = lambda: 0
    original = SimpleNamespace(defaults=defaultdict(factory, head_dim=8))
    cleaned = remove_non_pickleables(original, max_depth=max_depth)
    restored = pickle.loads(pickle.dumps(cleaned))
    assert isinstance(restored.defaults, defaultdict)
    assert restored.defaults.default_factory is None
    assert restored.defaults["head_dim"] == 8
    assert original.defaults.default_factory is factory
    assert original.defaults["missing"] == 0


@pytest.mark.parametrize("max_depth", [2, 3])
@pytest.mark.parametrize("sequence_type", [MetadataList, MetadataTuple])
def test_remove_non_pickleables_preserves_sequence_subclass_metadata(max_depth, sequence_type):
    import pickle
    from types import SimpleNamespace

    hook = lambda: None
    values = sequence_type([1, 2, hook])
    values.head_dim = 8
    values.callback = hook
    original = SimpleNamespace(values=values)
    cleaned = remove_non_pickleables(original, max_depth=max_depth)
    restored = pickle.loads(pickle.dumps(cleaned))
    assert type(restored.values) is sequence_type
    assert list(restored.values) == [1, 2, None]
    assert restored.values.head_dim == 8
    assert restored.values.callback is None
    assert original.values[2] is hook
    assert original.values.callback is hook


@pytest.mark.parametrize("max_depth", [2, 3])
def test_remove_non_pickleables_preserves_namedtuple_fields(max_depth):
    import pickle
    from types import SimpleNamespace

    hook = lambda: None
    original = SimpleNamespace(values=MetadataPair(8, hook))
    cleaned = remove_non_pickleables(original, max_depth=max_depth)
    restored = pickle.loads(pickle.dumps(cleaned))
    assert type(restored.values) is MetadataPair
    assert restored.values.head_dim == 8
    assert restored.values.callback is None
    assert original.values.callback is hook
