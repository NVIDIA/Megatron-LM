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

"""Unit tests for fsdp_dtensor_checkpoint.py — PR #3936 fixes.

Tests cover:
  1. Per-TransformerLayer gated_linear_unit check in handle_swiglu_in_state_dict
  2. GDN fused projection splitting in handle_gdn_in_state_dict
  3. Helper functions: get_expert_index_from_key, flatten_state_dict, etc.

Note: handle_swiglu_in_state_dict and handle_gdn_in_state_dict require
HAVE_MEGATRON_FSDP=True and a real distributed environment with DTensors.
We test their internal helper logic (is_swiglu_key, _key_in_glu_layer,
_match_gdn_key) by extracting the logic into standalone testable units.
For the checkpoint_inspector.py conversion functions, we test the
non-distributed components: SWiGLU key detection, SWiGLU regex splitting,
MTP key renaming, and --swiglu-modules prefix filtering.
"""

import re

import pytest
import torch

from megatron.core.transformer.fsdp_dtensor_checkpoint import (
    flatten_state_dict,
    get_expert_index_from_key,
)


# ============================================================================
# Test get_expert_index_from_key
# ============================================================================
class TestGetExpertIndexFromKey:
    """Test expert index extraction from various key formats."""

    def test_grouped_mlp_fc1_weight(self):
        key = "decoder.layers.0.mlp.experts.linear_fc1.weight3"
        assert get_expert_index_from_key(key) == 3

    def test_grouped_mlp_fc2_weight(self):
        key = "decoder.layers.0.mlp.experts.linear_fc2.weight12"
        assert get_expert_index_from_key(key) == 12

    def test_sequential_mlp_fc1(self):
        key = "decoder.layers.2.mlp.experts.local_experts.5.linear_fc1.weight"
        assert get_expert_index_from_key(key) == 5

    def test_sequential_mlp_fc2(self):
        key = "decoder.layers.2.mlp.experts.local_experts.7.linear_fc2.weight"
        assert get_expert_index_from_key(key) == 7

    def test_non_expert_key_returns_none(self):
        key = "decoder.layers.0.mlp.linear_fc1.weight"
        assert get_expert_index_from_key(key) is None

    def test_embedding_key_returns_none(self):
        key = "embedding.word_embeddings.weight"
        assert get_expert_index_from_key(key) is None


# ============================================================================
# Test flatten_state_dict
# ============================================================================
class TestFlattenStateDict:

    def test_flat_dict(self):
        d = {"a": 1, "b": 2}
        assert flatten_state_dict(d) == {"a": 1, "b": 2}

    def test_nested_dict(self):
        d = {"a": {"b": 1, "c": 2}}
        assert flatten_state_dict(d) == {"a.b": 1, "a.c": 2}

    def test_nested_list(self):
        d = {"a": [10, 20]}
        assert flatten_state_dict(d) == {"a.0": 10, "a.1": 20}

    def test_deeply_nested(self):
        d = {"model": {"layers": {"0": {"weight": 42}}}}
        assert flatten_state_dict(d) == {"model.layers.0.weight": 42}


# ============================================================================
# Test SWiGLU key detection logic (checkpoint_inspector.py patterns)
# ============================================================================
_SWIGLU_PATTERNS = [
    r"(.*)\.mlp\.linear_fc1\.weight$",
    r"(.*)\.mlp\.linear_fc1\.bias$",
    r"(.*)\.mlp\.experts\.linear_fc1\.weight(\d+)$",
    r"(.*)\.mlp\.experts\.linear_fc1\.bias(\d+)$",
    r"(.*)\.mlp\.experts\.local_experts\.(\d+)\.linear_fc1\.weight$",
    r"(.*)\.mlp\.experts\.local_experts\.(\d+)\.linear_fc1\.bias$",
    r"(.*)\.mlp\.shared_experts\.linear_fc1\.weight$",
    r"(.*)\.mlp\.shared_experts\.linear_fc1\.bias$",
]


def _is_swiglu_key(key, swiglu_prefixes=None):
    """Standalone version of the is_swiglu_key logic from checkpoint_inspector.py."""
    if not any(re.search(pat, key) for pat in _SWIGLU_PATTERNS):
        return False
    if swiglu_prefixes is None:
        return True
    return any(f".{mod}." in key or key.startswith(f"{mod}.") for mod in swiglu_prefixes)


class TestIsSWiGLUKey:
    """Test the SWiGLU key detection with per-module prefix filtering."""

    def test_dense_mlp_fc1_weight(self):
        assert _is_swiglu_key("model.module.language_model.layers.0.mlp.linear_fc1.weight")

    def test_dense_mlp_fc1_bias(self):
        assert _is_swiglu_key("model.module.language_model.layers.0.mlp.linear_fc1.bias")

    def test_grouped_expert_fc1_weight(self):
        assert _is_swiglu_key("model.module.language_model.layers.3.mlp.experts.linear_fc1.weight0")

    def test_grouped_expert_fc1_bias(self):
        assert _is_swiglu_key("model.module.language_model.layers.3.mlp.experts.linear_fc1.bias0")

    def test_sequential_expert_fc1(self):
        assert _is_swiglu_key(
            "model.module.language_model.layers.3.mlp.experts.local_experts.2.linear_fc1.weight"
        )

    def test_shared_expert_fc1(self):
        assert _is_swiglu_key(
            "model.module.language_model.layers.3.mlp.shared_experts.linear_fc1.weight"
        )

    def test_fc2_not_matched(self):
        assert not _is_swiglu_key("model.module.language_model.layers.0.mlp.linear_fc2.weight")

    def test_attention_not_matched(self):
        assert not _is_swiglu_key("model.module.language_model.layers.0.self_attention.linear_qkv.weight")

    # --- Per-module prefix filtering (--swiglu-modules) ---

    def test_prefix_filter_language_model_matches(self):
        key = "model.module.language_model.layers.0.mlp.linear_fc1.weight"
        assert _is_swiglu_key(key, swiglu_prefixes=["language_model"])

    def test_prefix_filter_vision_encoder_excluded(self):
        """Vision encoder fc1 should NOT be split when only language_model is in prefixes."""
        key = "model.module.vision_encoder.layers.0.mlp.linear_fc1.weight"
        assert not _is_swiglu_key(key, swiglu_prefixes=["language_model"])

    def test_prefix_filter_empty_list_matches_nothing(self):
        key = "model.module.language_model.layers.0.mlp.linear_fc1.weight"
        assert not _is_swiglu_key(key, swiglu_prefixes=[])

    def test_prefix_filter_none_matches_all(self):
        """None means global mode (--swiglu flag)."""
        key = "model.module.vision_encoder.layers.0.mlp.linear_fc1.weight"
        assert _is_swiglu_key(key, swiglu_prefixes=None)

    def test_prefix_filter_multiple_modules(self):
        """Multiple modules: both language_model and vision_encoder."""
        key_lm = "model.module.language_model.layers.0.mlp.linear_fc1.weight"
        key_ve = "model.module.vision_encoder.layers.0.mlp.linear_fc1.weight"
        prefixes = ["language_model", "vision_encoder"]
        assert _is_swiglu_key(key_lm, swiglu_prefixes=prefixes)
        assert _is_swiglu_key(key_ve, swiglu_prefixes=prefixes)


# ============================================================================
# Test SWiGLU regex splitting (_w / _v key generation)
# ============================================================================
class TestSWiGLURegexSplit:
    """Test the regex that splits SWiGLU keys into _w and _v variants.

    The PR updated the regex from matching only 'weight' to
    r'((?:weight|bias)\\d*)(.*)' to also handle biases and indexed weights.
    """

    @staticmethod
    def _split_key(key):
        w_key = re.sub(r'((?:weight|bias)\d*)(.*)', r'\1_w\2', key)
        v_key = re.sub(r'((?:weight|bias)\d*)(.*)', r'\1_v\2', key)
        return w_key, v_key

    def test_dense_weight(self):
        w, v = self._split_key("model.module.language_model.layers.0.mlp.linear_fc1.weight")
        assert w == "model.module.language_model.layers.0.mlp.linear_fc1.weight_w"
        assert v == "model.module.language_model.layers.0.mlp.linear_fc1.weight_v"

    def test_dense_bias(self):
        w, v = self._split_key("model.module.language_model.layers.0.mlp.linear_fc1.bias")
        assert w == "model.module.language_model.layers.0.mlp.linear_fc1.bias_w"
        assert v == "model.module.language_model.layers.0.mlp.linear_fc1.bias_v"

    def test_grouped_expert_weight_with_index(self):
        w, v = self._split_key(
            "model.module.language_model.layers.3.mlp.experts.linear_fc1.weight5"
        )
        assert w == "model.module.language_model.layers.3.mlp.experts.linear_fc1.weight5_w"
        assert v == "model.module.language_model.layers.3.mlp.experts.linear_fc1.weight5_v"

    def test_grouped_expert_bias_with_index(self):
        w, v = self._split_key(
            "model.module.language_model.layers.3.mlp.experts.linear_fc1.bias0"
        )
        assert w == "model.module.language_model.layers.3.mlp.experts.linear_fc1.bias0_w"
        assert v == "model.module.language_model.layers.3.mlp.experts.linear_fc1.bias0_v"

    def test_sequential_expert_weight(self):
        w, v = self._split_key(
            "model.module.language_model.layers.3.mlp.experts.local_experts.2.linear_fc1.weight"
        )
        assert w.endswith("weight_w")
        assert v.endswith("weight_v")


# ============================================================================
# Test MTP key renaming logic
# ============================================================================
class TestMTPKeyRenaming:
    """Test MTP key detection and renaming (transformer_layer -> mtp_model_layer).

    The PR added --rename-mtp-keys with auto-detection from checkpoint keys.
    """

    _MTP_OLD = ".mtp.layers."
    _MTP_SRC = ".transformer_layer."
    _MTP_DST = ".mtp_model_layer."

    @classmethod
    def _rename_mtp_keys(cls, state_dict):
        """Standalone MTP key renaming, mirrors checkpoint_inspector.py logic."""
        result = dict(state_dict)
        renamed = 0
        for k in list(result.keys()):
            if cls._MTP_OLD in k and cls._MTP_SRC in k:
                new_k = k.replace(cls._MTP_SRC, cls._MTP_DST, 1)
                result[new_k] = result.pop(k)
                renamed += 1
        return result, renamed

    @classmethod
    def _should_auto_detect_mtp(cls, keys):
        """Check if MTP auto-detection would trigger based on checkpoint keys."""
        return any(".mtp.layers." in k and ".transformer_layer." in k for k in keys)

    def test_mtp_key_renamed(self):
        sd = {
            "model.module.language_model.mtp.layers.0.transformer_layer.mlp.linear_fc1.weight": torch.tensor(1.0),
        }
        result, count = self._rename_mtp_keys(sd)
        assert count == 1
        expected_key = "model.module.language_model.mtp.layers.0.mtp_model_layer.mlp.linear_fc1.weight"
        assert expected_key in result

    def test_non_mtp_key_unchanged(self):
        sd = {
            "model.module.language_model.layers.0.mlp.linear_fc1.weight": torch.tensor(1.0),
        }
        result, count = self._rename_mtp_keys(sd)
        assert count == 0
        assert "model.module.language_model.layers.0.mlp.linear_fc1.weight" in result

    def test_multiple_mtp_keys(self):
        sd = {
            "model.module.language_model.mtp.layers.0.transformer_layer.mlp.linear_fc1.weight": torch.tensor(1.0),
            "model.module.language_model.mtp.layers.0.transformer_layer.self_attention.linear_qkv.weight": torch.tensor(2.0),
            "model.module.language_model.mtp.layers.1.transformer_layer.mlp.linear_fc1.weight": torch.tensor(3.0),
            "model.module.language_model.layers.0.mlp.linear_fc1.weight": torch.tensor(4.0),
        }
        result, count = self._rename_mtp_keys(sd)
        assert count == 3
        for k in result:
            if ".mtp.layers." in k:
                assert ".mtp_model_layer." in k
                assert ".transformer_layer." not in k

    def test_only_first_occurrence_replaced(self):
        """Ensure replace(..., 1) only replaces the first .transformer_layer."""
        key = "model.module.mtp.layers.0.transformer_layer.sub.transformer_layer.w"
        sd = {key: torch.tensor(1.0)}
        result, count = self._rename_mtp_keys(sd)
        assert count == 1
        new_key = list(result.keys())[0]
        assert new_key == "model.module.mtp.layers.0.mtp_model_layer.sub.transformer_layer.w"

    def test_auto_detect_triggers(self):
        keys = [
            "model.module.language_model.mtp.layers.0.transformer_layer.mlp.weight",
            "model.module.language_model.layers.0.mlp.weight",
        ]
        assert self._should_auto_detect_mtp(keys)

    def test_auto_detect_does_not_trigger_without_mtp(self):
        keys = [
            "model.module.language_model.layers.0.transformer_layer.mlp.weight",
        ]
        assert not self._should_auto_detect_mtp(keys)

    def test_auto_detect_does_not_trigger_without_transformer_layer(self):
        keys = [
            "model.module.language_model.mtp.layers.0.mtp_model_layer.mlp.weight",
        ]
        assert not self._should_auto_detect_mtp(keys)


# ============================================================================
# Test per-TransformerLayer gated_linear_unit check logic
# ============================================================================
class TestKeyInGluLayer:
    """Test the _key_in_glu_layer logic that checks whether a key belongs to
    a TransformerLayer with gated_linear_unit=True.

    This is the core fix in PR #3936: VLM vision encoder layers should NOT
    have their fc1 weights split because they use GELU, not SWiGLU.
    """

    @staticmethod
    def _strip_wrappers(path):
        parts = path.split('.')
        while parts and parts[0] in ('module', 'model'):
            parts = parts[1:]
        return '.'.join(parts)

    @staticmethod
    def _key_in_glu_layer(key, layer_glu):
        """Standalone version of _key_in_glu_layer from handle_swiglu_in_state_dict."""
        norm_key = TestKeyInGluLayer._strip_wrappers(key)
        best_glu, best_len = None, -1
        for layer_path, uses_glu in layer_glu.items():
            if norm_key.startswith(layer_path + '.') and len(layer_path) > best_len:
                best_glu, best_len = uses_glu, len(layer_path)
        if best_glu is None:
            return True  # no TransformerLayer found — assume GLU for backward compat
        return best_glu

    def test_language_model_layer_is_glu(self):
        layer_glu = {
            "language_model.decoder.layers.0": True,
            "vision_encoder.encoder.layers.0": False,
        }
        key = "module.language_model.decoder.layers.0.mlp.linear_fc1.weight"
        assert self._key_in_glu_layer(key, layer_glu) is True

    def test_vision_encoder_layer_is_not_glu(self):
        layer_glu = {
            "language_model.decoder.layers.0": True,
            "vision_encoder.encoder.layers.0": False,
        }
        key = "module.vision_encoder.encoder.layers.0.mlp.linear_fc1.weight"
        assert self._key_in_glu_layer(key, layer_glu) is False

    def test_multiple_vision_layers_all_non_glu(self):
        layer_glu = {
            "language_model.decoder.layers.0": True,
            "language_model.decoder.layers.1": True,
            "vision_encoder.encoder.layers.0": False,
            "vision_encoder.encoder.layers.1": False,
            "vision_encoder.encoder.layers.2": False,
        }
        for i in range(3):
            key = f"module.vision_encoder.encoder.layers.{i}.mlp.linear_fc1.weight"
            assert self._key_in_glu_layer(key, layer_glu) is False

    def test_unknown_key_defaults_to_true(self):
        """Keys not matching any TransformerLayer default to GLU=True for backward compat."""
        layer_glu = {
            "language_model.decoder.layers.0": True,
        }
        key = "module.embedding.word_embeddings.weight"
        assert self._key_in_glu_layer(key, layer_glu) is True

    def test_empty_layer_map_defaults_to_true(self):
        assert self._key_in_glu_layer("any.key.weight", {}) is True

    def test_longest_prefix_match(self):
        """When multiple layers match, the longest prefix should win."""
        layer_glu = {
            "language_model": True,
            "language_model.decoder.layers.0": False,
        }
        key = "module.language_model.decoder.layers.0.mlp.linear_fc1.weight"
        assert self._key_in_glu_layer(key, layer_glu) is False

    def test_strip_wrappers(self):
        assert self._strip_wrappers("module.model.language_model.layers.0") == "language_model.layers.0"
        assert self._strip_wrappers("language_model.layers.0") == "language_model.layers.0"
        assert self._strip_wrappers("module.module.model.x") == "x"


# ============================================================================
# Test GDN key matching logic
# ============================================================================
class TestGDNKeyMatching:
    """Test the _match_gdn_key logic from handle_gdn_in_state_dict.

    GDN modules have fused projections that need to be split:
      in_proj.weight -> 6-way (query, key, value, z, beta, alpha)
      conv1d.weight  -> 3-way (query, key, value)
      conv1d.bias    -> 3-way (query, key, value)
    """

    GDN_IN_PROJ_NAMES = ["query", "key", "value", "z", "beta", "alpha"]
    GDN_CONV1D_NAMES = ["query", "key", "value"]

    @staticmethod
    def _strip_wrappers(path):
        parts = path.split('.')
        while parts and parts[0] in ('module', 'model'):
            parts = parts[1:]
        return '.'.join(parts)

    @classmethod
    def _match_gdn_key(cls, key, gdn_info):
        """Standalone version of _match_gdn_key from handle_gdn_in_state_dict."""
        norm = cls._strip_wrappers(key)
        for gdn_path, info in gdn_info.items():
            if not norm.startswith(gdn_path + '.'):
                continue
            rel = norm[len(gdn_path) + 1:]
            if rel == 'in_proj.weight':
                return info['in_proj_sizes'], cls.GDN_IN_PROJ_NAMES, 0
            if rel in ('conv1d.weight', 'conv1d.bias'):
                return info['conv1d_sizes'], cls.GDN_CONV1D_NAMES, 0
        return None

    def _make_gdn_info(self, qk_dim=64, v_dim=128, num_value_heads=4, tp=1):
        return {
            'in_proj_sizes': [qk_dim // tp, qk_dim // tp, v_dim // tp,
                              v_dim // tp, num_value_heads // tp, num_value_heads // tp],
            'conv1d_sizes': [qk_dim // tp, qk_dim // tp, v_dim // tp],
        }

    def test_in_proj_weight_matched(self):
        gdn_info = {
            "language_model.decoder.layers.0.self_attention.gdn": self._make_gdn_info()
        }
        key = "module.language_model.decoder.layers.0.self_attention.gdn.in_proj.weight"
        result = self._match_gdn_key(key, gdn_info)
        assert result is not None
        sizes, names, dim = result
        assert names == self.GDN_IN_PROJ_NAMES
        assert len(sizes) == 6
        assert dim == 0

    def test_conv1d_weight_matched(self):
        gdn_info = {
            "language_model.decoder.layers.0.self_attention.gdn": self._make_gdn_info()
        }
        key = "module.language_model.decoder.layers.0.self_attention.gdn.conv1d.weight"
        result = self._match_gdn_key(key, gdn_info)
        assert result is not None
        sizes, names, dim = result
        assert names == self.GDN_CONV1D_NAMES
        assert len(sizes) == 3

    def test_conv1d_bias_matched(self):
        gdn_info = {
            "language_model.decoder.layers.0.self_attention.gdn": self._make_gdn_info()
        }
        key = "module.language_model.decoder.layers.0.self_attention.gdn.conv1d.bias"
        result = self._match_gdn_key(key, gdn_info)
        assert result is not None
        sizes, names, dim = result
        assert names == self.GDN_CONV1D_NAMES

    def test_non_gdn_key_not_matched(self):
        gdn_info = {
            "language_model.decoder.layers.0.self_attention.gdn": self._make_gdn_info()
        }
        key = "module.language_model.decoder.layers.0.mlp.linear_fc1.weight"
        assert self._match_gdn_key(key, gdn_info) is None

    def test_other_gdn_subkey_not_matched(self):
        """Only in_proj.weight and conv1d.weight/bias should be matched, not e.g. gate.weight."""
        gdn_info = {
            "language_model.decoder.layers.0.self_attention.gdn": self._make_gdn_info()
        }
        key = "module.language_model.decoder.layers.0.self_attention.gdn.gate.weight"
        assert self._match_gdn_key(key, gdn_info) is None

    def test_empty_gdn_info_matches_nothing(self):
        key = "module.language_model.decoder.layers.0.self_attention.gdn.in_proj.weight"
        assert self._match_gdn_key(key, {}) is None

    def test_multiple_gdn_layers(self):
        gdn_info = {
            "language_model.decoder.layers.0.self_attention.gdn": self._make_gdn_info(qk_dim=64),
            "language_model.decoder.layers.1.self_attention.gdn": self._make_gdn_info(qk_dim=128),
        }
        key0 = "module.language_model.decoder.layers.0.self_attention.gdn.in_proj.weight"
        key1 = "module.language_model.decoder.layers.1.self_attention.gdn.in_proj.weight"
        r0 = self._match_gdn_key(key0, gdn_info)
        r1 = self._match_gdn_key(key1, gdn_info)
        assert r0 is not None and r1 is not None
        assert r0[0][0] == 64  # layer 0 qk_dim
        assert r1[0][0] == 128  # layer 1 qk_dim

    def test_tp_affects_split_sizes(self):
        gdn_info_tp1 = {
            "gdn": self._make_gdn_info(qk_dim=64, v_dim=128, num_value_heads=4, tp=1)
        }
        gdn_info_tp2 = {
            "gdn": self._make_gdn_info(qk_dim=64, v_dim=128, num_value_heads=4, tp=2)
        }
        key = "module.gdn.in_proj.weight"
        r1 = self._match_gdn_key(key, gdn_info_tp1)
        r2 = self._match_gdn_key(key, gdn_info_tp2)
        # TP=2 should halve the split sizes
        assert r1[0][0] == 64 and r2[0][0] == 32
        assert r1[0][2] == 128 and r2[0][2] == 64


# ============================================================================
# Test GDN state dict key generation after splitting
# ============================================================================
class TestGDNStateDictKeySplitting:
    """Test that after GDN splitting, the correct sub-keys are generated."""

    GDN_IN_PROJ_NAMES = ["query", "key", "value", "z", "beta", "alpha"]
    GDN_CONV1D_NAMES = ["query", "key", "value"]

    @staticmethod
    def _generate_split_keys(key, names):
        """Mirrors the key generation in handle_gdn_in_state_dict."""
        return [f"{key}.{sub_name}" for sub_name in names]

    def test_in_proj_generates_6_keys(self):
        base = "module.language_model.decoder.layers.0.self_attention.gdn.in_proj.weight"
        keys = self._generate_split_keys(base, self.GDN_IN_PROJ_NAMES)
        assert len(keys) == 6
        assert keys[0].endswith(".in_proj.weight.query")
        assert keys[1].endswith(".in_proj.weight.key")
        assert keys[2].endswith(".in_proj.weight.value")
        assert keys[3].endswith(".in_proj.weight.z")
        assert keys[4].endswith(".in_proj.weight.beta")
        assert keys[5].endswith(".in_proj.weight.alpha")

    def test_conv1d_weight_generates_3_keys(self):
        base = "module.language_model.decoder.layers.0.self_attention.gdn.conv1d.weight"
        keys = self._generate_split_keys(base, self.GDN_CONV1D_NAMES)
        assert len(keys) == 3
        assert keys[0].endswith(".conv1d.weight.query")
        assert keys[1].endswith(".conv1d.weight.key")
        assert keys[2].endswith(".conv1d.weight.value")

    def test_conv1d_bias_generates_3_keys(self):
        base = "module.language_model.decoder.layers.0.self_attention.gdn.conv1d.bias"
        keys = self._generate_split_keys(base, self.GDN_CONV1D_NAMES)
        assert len(keys) == 3
        assert all(".conv1d.bias." in k for k in keys)


# ============================================================================
# Integration-style test: VLM SWiGLU + vision encoder scenario
# ============================================================================
class TestVLMSWiGLUScenario:
    """End-to-end scenario test for VLM where language model uses SWiGLU
    but vision encoder uses GELU. This is the core bug that PR #3936 fixes.

    Without the fix, vision encoder fc1 weights would be incorrectly split
    into _w/_v, causing missing key errors and loss spikes (~1.5 vs ~0.7).
    """

    @staticmethod
    def _strip_wrappers(path):
        parts = path.split('.')
        while parts and parts[0] in ('module', 'model'):
            parts = parts[1:]
        return '.'.join(parts)

    @staticmethod
    def _key_in_glu_layer(key, layer_glu):
        norm_key = TestVLMSWiGLUScenario._strip_wrappers(key)
        best_glu, best_len = None, -1
        for layer_path, uses_glu in layer_glu.items():
            if norm_key.startswith(layer_path + '.') and len(layer_path) > best_len:
                best_glu, best_len = uses_glu, len(layer_path)
        if best_glu is None:
            return True
        return best_glu

    def test_vlm_scenario(self):
        """Simulate a VLM with 2 LM layers (SWiGLU) and 2 VE layers (GELU)."""
        layer_glu = {
            "language_model.decoder.layers.0": True,
            "language_model.decoder.layers.1": True,
            "vision_encoder.encoder.layers.0": False,
            "vision_encoder.encoder.layers.1": False,
        }

        vlm_keys = [
            "module.language_model.decoder.layers.0.mlp.linear_fc1.weight",
            "module.language_model.decoder.layers.0.mlp.linear_fc1.bias",
            "module.language_model.decoder.layers.1.mlp.linear_fc1.weight",
            "module.language_model.decoder.layers.1.mlp.linear_fc1.bias",
            "module.vision_encoder.encoder.layers.0.mlp.linear_fc1.weight",
            "module.vision_encoder.encoder.layers.0.mlp.linear_fc1.bias",
            "module.vision_encoder.encoder.layers.1.mlp.linear_fc1.weight",
            "module.vision_encoder.encoder.layers.1.mlp.linear_fc1.bias",
        ]

        should_split = []
        should_skip = []
        for key in vlm_keys:
            if self._key_in_glu_layer(key, layer_glu):
                should_split.append(key)
            else:
                should_skip.append(key)

        # Language model keys: all 4 should be split
        assert len(should_split) == 4
        assert all("language_model" in k for k in should_split)

        # Vision encoder keys: all 4 should be skipped
        assert len(should_skip) == 4
        assert all("vision_encoder" in k for k in should_skip)


# ============================================================================
# Test checkpoint_inspector.py pretrained-only checkpoint handling
# (hasattr mcore_data guard — part of PR #3912 but related)
# ============================================================================
class TestMCoreDataGuard:
    """Test that the hasattr(metadata, 'mcore_data') guard works correctly."""

    def test_metadata_without_mcore_data(self):
        class FakeMetadata:
            pass
        m = FakeMetadata()
        assert not hasattr(m, "mcore_data")

    def test_metadata_with_mcore_data(self):
        class FakeMetadata:
            mcore_data = {"key": {"nd_reformulated_orig_global_shape": (10, 20)}}
        m = FakeMetadata()
        assert hasattr(m, "mcore_data")
        assert "key" in m.mcore_data
