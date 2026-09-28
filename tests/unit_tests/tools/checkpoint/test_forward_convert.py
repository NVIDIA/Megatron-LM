# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for the torch_dist -> fsdp_dtensor forward converter helpers.

Covers the two transforms ``convert_checkpoint`` needs for a hybrid
Gated-DeltaProduct checkpoint: merging the per-section ``in_proj``/``conv1d``
sub-keys back into the fused parameters ``named_parameters()`` exposes, and
reconciling pattern-position layer numbering with ModuleList numbering. Pure
key/tensor logic -- no distributed environment, no GPU.
"""

import os
import sys
from types import SimpleNamespace

import pytest
import torch

_INSPECTOR_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..", "tools", "checkpoint"
)
sys.path.insert(0, _INSPECTOR_DIR)

from checkpoint_inspector import (  # noqa: E402  (import after sys.path tweak)
    _LAYER_INDEX_RE,
    _gdp_split_names,
    _hybrid_layer_index_map,
    _merge_deltaproduct_projections,
    _renumber_hybrid_layers,
    _renumber_param_group_map,
)


def _split_keys(prefix, num_householder, sections, is_conv, trailing=()):
    """Build the per-section sub-keys a torch_dist store holds, without using the
    reverse transform -- this file must exercise the merge on its own."""
    names = _gdp_split_names(num_householder, is_conv=is_conv)
    out, offset = {}, 0
    for name, size in zip(names, sections):
        shape = (size, 1, 4) if is_conv else (size, 3)
        out[f"{prefix}.{name}"] = torch.arange(
            offset, offset + size, dtype=torch.float32
        ).reshape(-1, *([1] * (len(shape) - 1))).expand(shape).contiguous()
        offset += size
    return out


def _gdp_args(num_householder=3, heads=24, head_dim=64, groups=24, state=128):
    """Checkpoint args for a Gated DeltaProduct mixer (nm4-style defaults)."""
    return SimpleNamespace(
        gdp_num_householder=num_householder,
        mamba_num_heads=heads,
        mamba_head_dim=head_dim,
        mamba_num_groups=groups,
        mamba_state_dim=state,
    )


def _gdp_widths(args):
    d_inner = args.mamba_num_heads * args.mamba_head_dim
    gds = args.mamba_num_groups * args.mamba_state_dim
    m = args.gdp_num_householder
    in_proj = d_inner * (1 + m) + gds * (m + 1) + args.mamba_num_heads * (m + 1)
    conv = d_inner * m + gds * (m + 1)
    return in_proj, conv


class TestGdpSplitNames:
    """The concatenation order is the contract between the two directions."""

    def test_in_proj_order(self):
        assert _gdp_split_names(3, is_conv=False) == [
            "z", "V0", "V1", "V2", "K0", "K1", "K2", "Q", "b0", "b1", "b2", "a"
        ]

    def test_conv_order(self):
        assert _gdp_split_names(3, is_conv=True) == [
            "V0", "V1", "V2", "K0", "K1", "K2", "Q"
        ]

    def test_single_householder(self):
        assert _gdp_split_names(1, is_conv=False) == ["z", "V0", "K0", "Q", "b0", "a"]



class TestMergeRefusesStackedBlocks:
    def test_unindexed_layer_keys_raise(self):
        """Without an explicit layer index the leading axis is the layer axis, not the
        section axis, so concatenating would splice layers together."""
        args = _gdp_args(num_householder=1, heads=2, head_dim=4, groups=2, state=4)
        in_proj_w, _ = _gdp_widths(args)
        base = "decoder.layers.mixer.in_proj.weight"   # stacked: no .{idx}.
        names = _gdp_split_names(1, is_conv=False)
        tensors = {f"{base}.{n}": torch.zeros(3, 2) for n in names}
        with pytest.raises(NotImplementedError, match="no explicit layer index"):
            _merge_deltaproduct_projections(tensors)



class TestHybridLayerRenumbering:
    """Pattern-position numbering (sparse) vs ModuleList numbering (dense 0..n-1)."""

    # A fused compute+moe block is keyed at its first pattern slot and eats the next one,
    # so indices 6 and 8 are absent here.
    SPARSE = [0, 1, 2, 3, 4, 5, 7, 9]

    def _keys(self, indices, prefix="language_model.decoder.layers."):
        return {f"{prefix}{i}.mixer.out_proj.weight": torch.zeros(1) for i in indices}

    def test_dense_namespace_is_untouched(self):
        keys = self._keys(range(6))
        assert _hybrid_layer_index_map(keys) == {}
        out, n = _renumber_hybrid_layers(keys, {})
        assert n == 0 and out == keys

    def test_sparse_namespace_becomes_dense_preserving_order(self):
        keys = self._keys(self.SPARSE)
        index_map = _hybrid_layer_index_map(keys)
        prefix = "language_model.decoder.layers."
        assert index_map == {prefix: {old: new for new, old in enumerate(self.SPARSE)}}
        out, n = _renumber_hybrid_layers(keys, index_map)
        assert n == len(self.SPARSE)
        got = sorted(
            int(k[len(prefix):].split(".")[0]) for k in out
        )
        assert got == list(range(len(self.SPARSE)))

    def test_namespaces_are_independent(self):
        """A sparse decoder must not drag an already-dense vision stack with it."""
        keys = {**self._keys(self.SPARSE),
                **self._keys(range(4), prefix="vision_model.decoder.layers.")}
        index_map = _hybrid_layer_index_map(keys)
        assert set(index_map) == {"language_model.decoder.layers."}
        out, _ = _renumber_hybrid_layers(keys, index_map)
        for i in range(4):
            assert f"vision_model.decoder.layers.{i}.mixer.out_proj.weight" in out

    def test_optimizer_keys_follow_the_same_map(self):
        prefix = "optimizer.state.module.module.module.language_model.decoder.layers."
        keys = {**self._keys(self.SPARSE), **self._keys(self.SPARSE, prefix=prefix)}
        index_map = _hybrid_layer_index_map(keys)
        assert len(index_map) == 2
        out, n = _renumber_hybrid_layers(keys, index_map)
        assert n == 2 * len(self.SPARSE)
        assert f"{prefix}7.mixer.out_proj.weight" in out   # old index 9 -> new 7

    def test_inner_nesting_uses_the_innermost_layers_segment(self):
        keys = {
            "language_model.mtp.layers.0.mtp_model_layer.layers.0.x.weight": torch.zeros(1)
        }
        assert _hybrid_layer_index_map(keys) == {}



class TestRenumberingGuards:
    def test_dense_namespaces_may_disagree(self):
        """A layer frozen out of the optimizer state leaves the two namespaces with
        different index sets, but nothing is renumbered, so it must not abort."""
        model = {f"model.module.decoder.layers.{i}.a.weight": torch.zeros(1) for i in range(4)}
        opt = {f"optimizer.state.module.module.decoder.layers.{i}.a.weight": torch.zeros(1)
               for i in (0, 1, 2)}
        assert _hybrid_layer_index_map({**model, **opt}) == {}

    def test_disagreeing_namespaces_raise(self):
        """A layer present for the model but absent from the optimizer must not be
        renumbered independently."""
        model = {f"model.module.language_model.decoder.layers.{i}.a.weight": torch.zeros(1)
                 for i in [0, 1, 3]}
        opt = {f"optimizer.state.module.module.language_model.decoder.layers.{i}.a.weight":
               torch.zeros(1) for i in [0, 3]}
        with pytest.raises(NotImplementedError, match="disagree on which layers exist"):
            _hybrid_layer_index_map({**model, **opt})

    def test_agreeing_namespaces_are_fine(self):
        idx = [0, 1, 3]
        model = {f"model.module.language_model.decoder.layers.{i}.a.weight": torch.zeros(1)
                 for i in idx}
        opt = {f"optimizer.state.module.module.language_model.decoder.layers.{i}.a.weight":
               torch.zeros(1) for i in idx}
        index_map = _hybrid_layer_index_map({**model, **opt})
        assert len(index_map) == 2
        assert all(m == {0: 0, 1: 1, 3: 2} for m in index_map.values())



class TestParamGroupRenumberingDoesNotAlias:
    def test_descending_collision_is_safe(self):
        """New indices are <= old ones, so an in-place rename could clobber an entry
        that has not been visited yet."""
        prefix = "model.module.decoder.layers."
        index_map = {prefix: {7: 6, 9: 7}}
        pgm = {f"{prefix}9.a.weight": "group-for-9", f"{prefix}7.a.weight": "group-for-7"}
        assert _renumber_param_group_map(pgm, index_map) == {
            f"{prefix}7.a.weight": "group-for-9",
            f"{prefix}6.a.weight": "group-for-7",
        }

    def test_keys_outside_the_map_are_untouched(self):
        prefix = "model.module.decoder.layers."
        index_map = {prefix: {3: 1}}
        pgm = {f"{prefix}3.a.weight": "g", "model.module.embedding.word_embeddings.weight": "e"}
        out = _renumber_param_group_map(pgm, index_map)
        assert out == {f"{prefix}1.a.weight": "g",
                       "model.module.embedding.word_embeddings.weight": "e"}




class TestMergeDeltaProductProjections:
    """The merge is the forward direction's own transform; build its input directly."""

    ARGS = _gdp_args()

    def _sections(self, is_conv):
        a = self.ARGS
        d_inner, gds, m = a.mamba_num_heads * a.mamba_head_dim, a.mamba_num_groups * a.mamba_state_dim, a.gdp_num_householder
        if is_conv:
            return [d_inner] * m + [gds] * m + [gds]
        return ([d_inner] + [d_inner] * m + [gds] * m + [gds]
                + [a.mamba_num_heads] * m + [a.mamba_num_heads])

    def test_in_proj_sections_concatenate_in_mcore_order(self):
        pre = "model.module.decoder.layers.0.mixer.in_proj.weight"
        sections = self._sections(is_conv=False)
        tensors = _split_keys(pre, self.ARGS.gdp_num_householder, sections, is_conv=False)
        out, n, _merged = _merge_deltaproduct_projections(tensors)
        assert n == 1 and set(out) == {pre}
        fused = out[pre]
        assert fused.shape[0] == sum(sections)
        # each section must land at its own offset, in _gdp_split_names order
        offset = 0
        for name, size in zip(_gdp_split_names(self.ARGS.gdp_num_householder, False), sections):
            assert torch.equal(fused[offset], torch.full((3,), float(offset))), name
            offset += size

    def test_conv1d_merges_and_keeps_its_trailing_dims(self):
        pre = "model.module.decoder.layers.0.mixer.conv1d.weight"
        sections = self._sections(is_conv=True)
        out, n, _merged = _merge_deltaproduct_projections(
            _split_keys(pre, self.ARGS.gdp_num_householder, sections, is_conv=True)
        )
        assert n == 1 and out[pre].shape == (sum(sections), 1, 4)

    def test_optimizer_state_subkeys_merge_independently(self):
        base = "optimizer.state.module.module.decoder.layers.0.mixer.in_proj.weight"
        sections = self._sections(is_conv=False)
        split = _split_keys(base, self.ARGS.gdp_num_householder, sections, is_conv=False)
        out, n, _merged = _merge_deltaproduct_projections({f"{k}.exp_avg": v for k, v in split.items()})
        assert n == 1 and f"{base}.exp_avg" in out

    def test_incomplete_section_set_raises(self):
        pre = "model.module.decoder.layers.0.mixer.in_proj.weight"
        tensors = _split_keys(pre, self.ARGS.gdp_num_householder, self._sections(False), is_conv=False)
        del tensors[f"{pre}.a"]
        with pytest.raises(NotImplementedError, match="Refusing to guess"):
            _merge_deltaproduct_projections(tensors)


class TestMergeLeavesOtherMixersAlone:
    """Mamba-2 and Gated-DeltaNet also leave a bare ``.z`` sub-key, which the forward
    converter keeps un-merged."""

    def test_mamba_sections_pass_through(self):
        pre = "model.module.decoder.layers.0.mixer.in_proj.weight"
        keys = {f"{pre}.{n}": torch.zeros(4, 2) for n in ("z", "x", "B", "C", "dt")}
        out, n, _merged = _merge_deltaproduct_projections(dict(keys))
        assert n == 0 and set(out) == set(keys)

    def test_gdn_self_attention_sections_pass_through(self):
        pre = "model.module.decoder.layers.0.self_attention.in_proj.weight"
        keys = {f"{pre}.{n}": torch.zeros(2, 2)
                for n in ("query", "key", "value", "z", "beta", "alpha")}
        out, n, _merged = _merge_deltaproduct_projections(dict(keys))
        assert n == 0 and set(out) == set(keys)


class TestNamespaceGroupingUsesConfiguredPrefixes:
    """The model/optimizer agreement guard must keep working when the caller overrides
    --output-model-weight-prefix; deriving the grouping from a regex of the default
    spellings silently stops reconciling the two namespaces."""

    OPT = "optimizer.state.module.module.module"

    def _keys(self, model_prefix):
        model = {f"{model_prefix}.decoder.layers.{i}.a.weight": torch.zeros(1) for i in (0, 1, 3)}
        opt = {f"{self.OPT}.decoder.layers.{i}.a.weight": torch.zeros(1) for i in (0, 3)}
        return {**model, **opt}

    @pytest.mark.parametrize("model_prefix", ["model.module.module", "mymodel", "a.b.c"])
    def test_disagreement_is_caught_for_any_prefix(self, model_prefix):
        with pytest.raises(NotImplementedError, match="disagree on which layers exist"):
            _hybrid_layer_index_map(self._keys(model_prefix), (model_prefix, self.OPT))

    def test_agreeing_namespaces_still_renumber_together(self):
        mp = "mymodel"
        keys = {f"{mp}.decoder.layers.{i}.a.weight": torch.zeros(1) for i in (0, 1, 3)}
        keys.update({f"{self.OPT}.decoder.layers.{i}.a.weight": torch.zeros(1) for i in (0, 1, 3)})
        m = _hybrid_layer_index_map(keys, (mp, self.OPT))
        assert len(m) == 2 and all(v == {0: 0, 1: 1, 3: 2} for v in m.values())


class TestParamGroupFixupOnlyTouchesMergedKeys:
    """Re-matching the regex would also catch Mamba-2 / GatedDeltaNet ".z" keys that the
    merge deliberately leaves alone, deleting their param-group entries while the state
    dict still holds them -- which aborts the conversion."""

    def test_merge_reports_exactly_the_keys_it_consumed(self):
        args = _gdp_args()
        d = args.mamba_num_heads * args.mamba_head_dim
        g = args.mamba_num_groups * args.mamba_state_dim
        n, m = args.mamba_num_heads, args.gdp_num_householder
        sections = [d] + [d] * m + [g] * m + [g] + [n] * m + [n]
        pre = "model.module.decoder.layers.0.mixer.in_proj.weight"
        tensors = _split_keys(pre, m, sections, is_conv=False)
        # plus foreign sub-keys the merge must not claim
        tensors["model.module.decoder.layers.1.mixer.in_proj.weight.z"] = torch.zeros(4, 3)
        tensors["model.module.decoder.layers.2.self_attention.in_proj.weight.z"] = torch.zeros(4, 3)
        out, n_merged, merged = _merge_deltaproduct_projections(tensors)
        assert n_merged == 1
        assert all(k.startswith(pre + ".") for k in merged), merged
        assert set(merged.values()) == {pre}
        assert "model.module.decoder.layers.1.mixer.in_proj.weight.z" in out
        assert "model.module.decoder.layers.2.self_attention.in_proj.weight.z" in out
