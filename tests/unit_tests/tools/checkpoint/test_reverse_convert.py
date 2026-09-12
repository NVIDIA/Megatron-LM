# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for the fsdp_dtensor -> torch_dist reverse converter helpers.

These exercise the pure key/tensor transforms of
``tools/checkpoint/checkpoint_inspector.py`` (the inverse of
``convert_checkpoint``) without a distributed environment: prefix stripping,
optimizer-key reversal, SwiGLU merge, expert re-stacking, layer stacking, MTP
rename, homogeneity detection, and common-state unflattening.
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
    _assert_supported_scope,
    _is_keep_fp32_key,
    _layers_are_homogeneous,
    _merge_swiglu,
    _model_param_dtype,
    _output_group_id,
    _rebuild_param_groups_from_meta,
    _restack_experts,
    _reverse_mtp_keys,
    _reverse_optimizer_state_key,
    _split_gdn_projections,
    _split_mamba_projections,
    _stack_layers,
    _strip_fsdp_model_prefix,
    _unflatten,
)


class TestStripModelPrefix:
    def test_default_prefix(self):
        assert (
            _strip_fsdp_model_prefix("model.module.decoder.layers.0.mlp.linear_fc1.weight")
            == "decoder.layers.0.mlp.linear_fc1.weight"
        )

    def test_deeper_wrapper_run(self):
        assert (
            _strip_fsdp_model_prefix("model.module.module.embedding.word_embeddings.weight")
            == "embedding.word_embeddings.weight"
        )

    def test_bare_model_prefix(self):
        assert _strip_fsdp_model_prefix("model.output_layer.weight") == "output_layer.weight"

    def test_non_model_key_returns_none(self):
        assert _strip_fsdp_model_prefix("optimizer.state.module.module.module.x.exp_avg") is None
        assert _strip_fsdp_model_prefix("checkpoint_version") is None


class TestReverseOptimizerKey:
    def test_exp_avg(self):
        fsdp = "optimizer.state.module.module.module.decoder.layers.0.mlp.linear_fc2.weight.exp_avg"
        assert (
            _reverse_optimizer_state_key(fsdp)
            == "optimizer.state.exp_avg.decoder.layers.0.mlp.linear_fc2.weight"
        )

    def test_exp_avg_sq(self):
        fsdp = "optimizer.state.module.module.module.embedding.word_embeddings.weight.exp_avg_sq"
        assert (
            _reverse_optimizer_state_key(fsdp)
            == "optimizer.state.exp_avg_sq.embedding.word_embeddings.weight"
        )

    def test_step(self):
        fsdp = "optimizer.state.module.module.module.output_layer.weight.step"
        assert _reverse_optimizer_state_key(fsdp) == "optimizer.state.step.output_layer.weight"

    def test_wrapper_depth_agnostic(self):
        # Any run of ``module.`` wrappers is stripped.
        fsdp = "optimizer.state.module.decoder.final_norm.weight.exp_avg"
        assert (
            _reverse_optimizer_state_key(fsdp)
            == "optimizer.state.exp_avg.decoder.final_norm.weight"
        )


class TestReverseMTP:
    def test_top_level_mtp(self):
        out, n = _reverse_mtp_keys(
            {"mtp.layers.0.mtp_model_layer.mlp.linear_fc1.weight": torch.zeros(1)}
        )
        assert n == 1
        assert "mtp.layers.0.transformer_layer.mlp.linear_fc1.weight" in out

    def test_nested_mtp(self):
        out, n = _reverse_mtp_keys(
            {"language_model.mtp.layers.1.mtp_model_layer.self_attention.w": torch.zeros(1)}
        )
        assert n == 1
        assert "language_model.mtp.layers.1.transformer_layer.self_attention.w" in out

    def test_non_mtp_untouched(self):
        out, n = _reverse_mtp_keys({"decoder.layers.0.mlp.linear_fc1.weight": torch.zeros(1)})
        assert n == 0
        assert "decoder.layers.0.mlp.linear_fc1.weight" in out


class TestMergeSwiglu:
    def test_dense_weight_merge(self):
        w = torch.arange(6.0).reshape(3, 2)
        v = torch.arange(6.0, 12.0).reshape(3, 2)
        out, n = _merge_swiglu(
            {
                "decoder.layers.0.mlp.linear_fc1.weight_w": w,
                "decoder.layers.0.mlp.linear_fc1.weight_v": v,
            }
        )
        assert n == 1
        merged = out["decoder.layers.0.mlp.linear_fc1.weight"]
        assert torch.equal(merged, torch.cat([w, v], dim=0))
        assert "decoder.layers.0.mlp.linear_fc1.weight_w" not in out

    def test_optimizer_swiglu_merge(self):
        w = torch.ones(2, 2)
        v = torch.zeros(2, 2)
        key_w = "optimizer.state.exp_avg.decoder.layers.0.mlp.linear_fc1.weight_w"
        key_v = "optimizer.state.exp_avg.decoder.layers.0.mlp.linear_fc1.weight_v"
        out, n = _merge_swiglu({key_w: w, key_v: v})
        assert n == 1
        assert "optimizer.state.exp_avg.decoder.layers.0.mlp.linear_fc1.weight" in out

    def test_indexed_expert_weight(self):
        out, n = _merge_swiglu(
            {
                "decoder.layers.0.mlp.experts.linear_fc1.weight3_w": torch.ones(1, 2),
                "decoder.layers.0.mlp.experts.linear_fc1.weight3_v": torch.ones(1, 2),
            }
        )
        assert n == 1
        assert "decoder.layers.0.mlp.experts.linear_fc1.weight3" in out

    def test_module_scope_filter(self):
        keys = {
            "language_model.decoder.layers.0.mlp.linear_fc1.weight_w": torch.ones(1, 1),
            "language_model.decoder.layers.0.mlp.linear_fc1.weight_v": torch.ones(1, 1),
            "vision_model.decoder.layers.0.mlp.linear_fc1.weight_w": torch.ones(1, 1),
            "vision_model.decoder.layers.0.mlp.linear_fc1.weight_v": torch.ones(1, 1),
        }
        out, n = _merge_swiglu(keys, swiglu_modules=["language_model"])
        assert n == 1
        assert "language_model.decoder.layers.0.mlp.linear_fc1.weight" in out
        assert "vision_model.decoder.layers.0.mlp.linear_fc1.weight_w" in out

    def test_fc2_not_merged(self):
        out, n = _merge_swiglu({"decoder.layers.0.mlp.linear_fc2.weight": torch.ones(2, 2)})
        assert n == 0


class TestRestackExperts:
    def test_stack_grouped_experts(self):
        t = {
            "decoder.layers.0.mlp.experts.linear_fc1.weight0": torch.zeros(4, 2),
            "decoder.layers.0.mlp.experts.linear_fc1.weight1": torch.ones(4, 2),
        }
        out, n = _restack_experts(t)
        assert n == 1
        key = "decoder.layers.0.mlp.experts.experts.linear_fc1.weight"
        assert out[key].shape == (2, 4, 2)
        assert torch.equal(out[key][1], torch.ones(4, 2))

    def test_optimizer_experts(self):
        t = {
            "optimizer.state.exp_avg.decoder.layers.0.mlp.experts.linear_fc2.weight0": torch.zeros(
                2
            ),
            "optimizer.state.exp_avg.decoder.layers.0.mlp.experts.linear_fc2.weight1": torch.ones(
                2
            ),
        }
        out, n = _restack_experts(t)
        assert n == 1
        assert (
            "optimizer.state.exp_avg.decoder.layers.0.mlp.experts.experts.linear_fc2.weight" in out
        )

    def test_non_contiguous_raises(self):
        t = {
            "decoder.layers.0.mlp.experts.linear_fc1.weight0": torch.zeros(1),
            "decoder.layers.0.mlp.experts.linear_fc1.weight2": torch.zeros(1),
        }
        with pytest.raises(AssertionError):
            _restack_experts(t)

    def test_shared_experts_untouched(self):
        t = {"decoder.layers.0.mlp.shared_experts.linear_fc1.weight": torch.zeros(2)}
        out, n = _restack_experts(t)
        assert n == 0
        assert "decoder.layers.0.mlp.shared_experts.linear_fc1.weight" in out

    def test_stack_non_grouped_local_experts(self):
        # SequentialMLP (no --moe-grouped-gemm) stores experts per local index;
        # mcore's sharded_state_dict still re-stacks them into the grouped key.
        t = {
            "decoder.layers.0.mlp.experts.local_experts.0.linear_fc1.weight": torch.zeros(4, 2),
            "decoder.layers.0.mlp.experts.local_experts.1.linear_fc1.weight": torch.ones(4, 2),
        }
        out, n = _restack_experts(t)
        assert n == 1
        key = "decoder.layers.0.mlp.experts.experts.linear_fc1.weight"
        assert out[key].shape == (2, 4, 2)
        assert torch.equal(out[key][1], torch.ones(4, 2))

    def test_local_experts_optimizer(self):
        t = {
            "optimizer.state.exp_avg.decoder.layers.3.mlp.experts.local_experts.0.linear_fc2.weight": torch.zeros(
                2
            ),  # noqa: E501
            "optimizer.state.exp_avg.decoder.layers.3.mlp.experts.local_experts.1.linear_fc2.weight": torch.ones(
                2
            ),  # noqa: E501
        }
        out, n = _restack_experts(t)
        assert n == 1
        assert (
            "optimizer.state.exp_avg.decoder.layers.3.mlp.experts.experts.linear_fc2.weight" in out
        )

    def test_shared_experts_not_treated_as_local(self):
        # shared_experts must not match the local_experts pattern.
        t = {
            "decoder.layers.0.mlp.shared_experts.local_experts.0.linear_fc1.weight": torch.zeros(2)
        }
        out, n = _restack_experts(t)
        assert n == 0


class TestSplitGdnProjections:
    @staticmethod
    def _args():
        from types import SimpleNamespace

        # qk_dim = 4*64 = 256, v_dim = 8*64 = 512, num_value_heads = 8
        return SimpleNamespace(
            experimental_attention_variant="gated_delta_net",
            linear_num_key_heads=4,
            linear_key_head_dim=64,
            linear_num_value_heads=8,
            linear_value_head_dim=64,
        )

    def test_in_proj_split_names_sizes_and_order(self):
        qk, v, nvh = 256, 512, 8
        rows = 2 * qk + 2 * v + 2 * nvh  # 1552
        key = "decoder.layers.0.self_attention.in_proj.weight"
        blob = torch.arange(rows * 3).float().reshape(rows, 3)
        out, n = _split_gdn_projections({key: blob}, self._args())
        assert n == 1
        assert key not in out
        for name, size in [
            ("query", qk),
            ("key", qk),
            ("value", v),
            ("z", v),
            ("beta", nvh),
            ("alpha", nvh),
        ]:
            assert out[f"{key}.{name}"].shape == (size, 3)
        # concatenating the parts in factory order reproduces the fused blob.
        cat = torch.cat(
            [out[f"{key}.{name}"] for name in ["query", "key", "value", "z", "beta", "alpha"]],
            dim=0,
        )
        assert torch.equal(cat, blob)

    def test_conv1d_split(self):
        qk, v = 256, 512
        key = "decoder.layers.0.self_attention.conv1d.weight"
        out, n = _split_gdn_projections({key: torch.zeros(2 * qk + v, 1, 4)}, self._args())
        assert n == 1
        assert out[f"{key}.query"].shape == (qk, 1, 4)
        assert out[f"{key}.value"].shape == (v, 1, 4)

    def test_optimizer_state_split(self):
        key = "optimizer.state.exp_avg.decoder.layers.0.self_attention.in_proj.weight"
        out, n = _split_gdn_projections({key: torch.zeros(1552, 3)}, self._args())
        assert n == 1
        assert f"{key}.alpha" in out and f"{key}.query" in out

    def test_stacked_block_splits_second_dim(self):
        # A homogeneous (all-GDN) block stacks layers on axis 0, so the projection
        # dim is axis 1 and must be split there.
        key = "decoder.layers.self_attention.in_proj.weight"  # no explicit layer index
        out, n = _split_gdn_projections({key: torch.zeros(3, 1552, 5)}, self._args())
        assert n == 1
        assert out[f"{key}.query"].shape == (3, 256, 5)

    def test_noop_without_gdn_variant(self):
        from types import SimpleNamespace

        key = "decoder.layers.0.self_attention.in_proj.weight"
        out, n = _split_gdn_projections({key: torch.zeros(1552, 3)}, SimpleNamespace())
        assert n == 0 and key in out

    def test_noop_when_args_none(self):
        key = "decoder.layers.0.self_attention.in_proj.weight"
        out, n = _split_gdn_projections({key: torch.zeros(1552, 3)}, None)
        assert n == 0 and key in out


class TestStackLayers:
    def test_dense_stack(self):
        t = {
            "decoder.layers.0.mlp.linear_fc2.weight": torch.zeros(2, 3),
            "decoder.layers.1.mlp.linear_fc2.weight": torch.ones(2, 3),
            "embedding.word_embeddings.weight": torch.zeros(5, 3),
        }
        out, n = _stack_layers(t)
        assert n == 1
        assert out["decoder.layers.mlp.linear_fc2.weight"].shape == (2, 2, 3)
        # non-layer keys pass through unchanged
        assert out["embedding.word_embeddings.weight"].shape == (5, 3)

    def test_optimizer_and_model_stack_together(self):
        t = {
            "decoder.layers.0.w": torch.zeros(2),
            "decoder.layers.1.w": torch.zeros(2),
            "optimizer.state.exp_avg.decoder.layers.0.w": torch.zeros(2),
            "optimizer.state.exp_avg.decoder.layers.1.w": torch.zeros(2),
        }
        out, n = _stack_layers(t)
        assert n == 2
        assert out["decoder.layers.w"].shape == (2, 2)
        assert out["optimizer.state.exp_avg.decoder.layers.w"].shape == (2, 2)

    def test_heterogeneous_layers_raise(self):
        # 'a' present in both layers, 'b' only in layer 0 -> heterogeneous.
        t = {
            "decoder.layers.0.a": torch.zeros(1),
            "decoder.layers.1.a": torch.zeros(1),
            "decoder.layers.0.b": torch.zeros(1),
        }
        with pytest.raises(ValueError, match="Heterogeneous"):
            _stack_layers(t)


class TestLayersAreHomogeneous:
    def _uniform(self, per_layer_suffixes, n_layers, extra=()):
        keys = list(extra)
        for i in range(n_layers):
            for s in per_layer_suffixes:
                keys.append(f"decoder.layers.{i}.{s}")
        return keys

    def test_plain_dense_multi_layer_is_homogeneous(self):
        keys = self._uniform(
            ["self_attention.linear_qkv.weight", "mlp.linear_fc1.weight"],
            4,
            extra=["embedding.word_embeddings.weight"],
        )
        assert _layers_are_homogeneous(keys)

    def test_uniform_all_moe_is_homogeneous(self):
        # Every layer is MoE (moe_layer_freq == 1) -> mcore stacks the block.
        keys = self._uniform(
            [
                "self_attention.linear_qkv.weight",
                "mlp.router.weight",
                "mlp.experts.experts.linear_fc1.weight",
                "mlp.experts.experts.linear_fc2.weight",
            ],
            8,
        )
        assert _layers_are_homogeneous(keys)

    def test_interleaved_moe_and_dense_is_non_homogeneous(self):
        keys = [
            "decoder.layers.0.mlp.linear_fc1.weight",  # dense layer
            "decoder.layers.1.mlp.experts.experts.linear_fc1.weight",  # MoE layer
        ]
        assert not _layers_are_homogeneous(keys)

    def test_interleaved_linear_attention_is_non_homogeneous(self):
        keys = [
            "decoder.layers.0.self_attention.linear_qkv.weight",
            "decoder.layers.2.self_attention.in_proj.weight",  # GDN layer differs
            "decoder.layers.2.self_attention.conv1d.weight",
        ]
        assert not _layers_are_homogeneous(keys)

    def test_mtp_layers_do_not_block_decoder_stacking(self):
        keys = self._uniform(
            ["mlp.experts.experts.linear_fc1.weight"],
            4,
            extra=["mtp.layers.0.transformer_layer.mlp.linear_fc1.weight"],
        )
        assert _layers_are_homogeneous(keys)

    def test_no_per_layer_keys_returns_false(self):
        assert not _layers_are_homogeneous(["embedding.word_embeddings.weight"])


class TestUnflatten:
    def test_scalars_and_namespace_leaf(self):
        from types import SimpleNamespace

        ns = SimpleNamespace(x=1)
        flat = {
            "args": ns,
            "checkpoint_version": 3.0,
            "iteration": 100,
            "optimizer.param_groups.0.lr": 0.001,
            "optimizer.param_groups.0.weight_decay": 0.1,
        }
        out = _unflatten(flat)
        assert out["args"] is ns
        assert out["checkpoint_version"] == 3.0
        assert out["optimizer"]["param_groups"][0]["lr"] == 0.001
        assert out["optimizer"]["param_groups"][0]["weight_decay"] == 0.1

    def test_int_keyed_dict_becomes_list(self):
        out = _unflatten({"g.0": "a", "g.1": "b", "g.2": "c"})
        assert out["g"] == ["a", "b", "c"]


class TestRebuildParamGroupsFromMeta:
    _PREFIX = "optimizer.param_to_group_meta."

    def _meta(self, fqn, wd_mult, weight_decay, lr=1e-4, expert=False):
        # one flat entry per attribute, mirroring the fsdp on-disk layout
        attrs = {
            "wd_mult": wd_mult,
            "lr_mult": 1.0,
            "is_expert_parallel": expert,
            "is_decoupled_lr": False,
            "lr": lr,
            "weight_decay": weight_decay,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
            "step": 25,
        }
        return {f"{self._PREFIX}module.module.module.{fqn}.{a}": v for a, v in attrs.items()}

    def test_groups_by_identifier_tuple(self):
        flat = {}
        # two decay params, one no-decay param, one expert param -> 3 distinct groups
        flat.update(self._meta("decoder.layers.0.self_attention.linear_qkv.weight", 1.0, 0.1))
        flat.update(self._meta("decoder.layers.1.self_attention.linear_qkv.weight", 1.0, 0.1))
        flat.update(self._meta("decoder.layers.0.self_attention.linear_qkv.bias", 0.0, 0.0))
        flat.update(
            self._meta("decoder.layers.0.mlp.experts.linear_fc1.weight0", 1.0, 0.1, expert=True)
        )

        groups = _rebuild_param_groups_from_meta(flat, self._PREFIX)
        idents = {
            (g["wd_mult"], g["lr_mult"], g["is_expert_parallel"], g["is_decoupled_lr"])
            for g in groups
        }
        assert len(groups) == 3
        assert idents == {
            (1.0, 1.0, False, False),
            (0.0, 1.0, False, False),
            (1.0, 1.0, True, False),
        }
        # full hyperparameters are carried; params are contiguous integer indices
        decay = next(g for g in groups if g["wd_mult"] == 1.0 and not g["is_expert_parallel"])
        assert decay["weight_decay"] == 0.1 and decay["betas"] == (0.9, 0.999)
        assert decay["step"] == 25 and "params" in decay
        allparams = sorted(i for g in groups for i in g["params"])
        assert allparams == list(range(4))  # one index per parameter, no gaps/dupes


class TestModelParamDtype:
    def test_bf16_fp16_fp32_and_none(self):
        from types import SimpleNamespace

        assert _model_param_dtype(SimpleNamespace(bf16=True, fp16=False)) == torch.bfloat16
        assert _model_param_dtype(SimpleNamespace(bf16=False, fp16=True)) == torch.float16
        assert _model_param_dtype(SimpleNamespace(bf16=False, fp16=False)) == torch.float32
        assert _model_param_dtype(None) is None


class TestSupportedScope:
    """The scope fence raises on architectures the converter cannot invert."""

    def test_unindexed_stacked_layer_buffer_raises(self):
        with pytest.raises(NotImplementedError, match="un-indexed"):
            _assert_supported_scope({"decoder.layers.norm.weight": None})

    def test_mamba_and_gdn_conv1d_are_not_fenced(self):
        # Mamba's fused ``mixer.conv1d_weight`` and GatedDeltaNet's dotted
        # ``self_attention.conv1d.weight`` are both handled by dedicated splits,
        # so neither may trip the fence.
        _assert_supported_scope(
            {
                "decoder.layers.0.mixer.conv1d_weight": None,
                "decoder.layers.0.mixer.conv1d_bias": None,
                "decoder.layers.0.mixer.in_proj.weight": None,
                "decoder.layers.0.self_attention.conv1d.weight": None,
                "optimizer.state.exp_avg.decoder.layers.0.mixer.conv1d_weight": None,
            }
        )

    def test_indexed_layers_and_plain_keys_pass(self):
        _assert_supported_scope(
            {
                "embedding.word_embeddings.weight": None,
                "decoder.layers.0.self_attention.linear_qkv.weight": None,
                "decoder.final_layernorm.weight": None,
                "mtp.layers.0.transformer_layer.self_attention.linear_qkv.weight": None,
                "optimizer.state.exp_avg.decoder.layers.3.mlp.linear_fc1.weight": None,
            }
        )


class TestKeepFp32Keys:
    """Persistent fp32 buffers (e.g. router.expert_bias) are protected from downcast."""

    def test_expert_bias_is_kept(self):
        assert _is_keep_fp32_key("decoder.layers.0.mlp.router.expert_bias")
        assert _is_keep_fp32_key("decoder.layers.mlp.router.expert_bias")  # stacked form

    def test_ordinary_weights_are_not_kept(self):
        assert not _is_keep_fp32_key("decoder.layers.0.mlp.linear_fc1.weight")
        assert not _is_keep_fp32_key("decoder.layers.0.mlp.router.weight")
        assert not _is_keep_fp32_key("output_layer.weight")

    def test_substring_expert_bias_not_falsely_matched(self):
        # only a whole trailing ``.expert_bias`` segment matches, not a substring.
        assert not _is_keep_fp32_key("decoder.layers.0.mlp.router.expert_bias_extra")


class TestSplitMambaProjections:
    """Reverse split of fused Mamba-2 in_proj/conv1d into named factory sub-keys."""

    # d_state=16, ngroups=2, head_dim=8 -> nheads=4, d_inner=32, gds=32.
    # in_proj width = 2*32 + 2*32 + 4 = 132 ; conv width = 32 + 2*32 = 96.
    ARGS = SimpleNamespace(mamba_state_dim=16, mamba_num_groups=2, mamba_head_dim=8)

    def test_in_proj_split_names_sizes_and_reproduce(self):
        blob = torch.randn(132, 24)
        out, n = _split_mamba_projections(
            {"decoder.layers.0.mixer.in_proj.weight": blob}, self.ARGS
        )
        assert n == 1
        base = "decoder.layers.0.mixer.in_proj.weight"
        names = ("z", "x", "B", "C", "dt")
        assert set(out) == {f"{base}.{s}" for s in names}
        assert [out[f"{base}.{s}"].shape[0] for s in names] == [32, 32, 32, 32, 4]
        # concatenation reproduces the original fused blob (order z,x,B,C,dt)
        assert torch.equal(torch.cat([out[f"{base}.{s}"] for s in names], dim=0), blob)

    def test_conv1d_weight_renamed_and_split(self):
        out, n = _split_mamba_projections(
            {"decoder.layers.1.mixer.conv1d_weight": torch.randn(96, 1, 4)}, self.ARGS
        )
        assert n == 1
        # conv1d_weight -> conv1d.weight (dotted on-disk key), split x/B/C.
        base = "decoder.layers.1.mixer.conv1d.weight"
        assert set(out) == {f"{base}.{s}" for s in ("x", "B", "C")}
        assert [out[f"{base}.{s}"].shape[0] for s in ("x", "B", "C")] == [32, 32, 32]

    def test_conv1d_bias_renamed_and_split(self):
        out, n = _split_mamba_projections(
            {"decoder.layers.0.mixer.conv1d_bias": torch.randn(96)}, self.ARGS
        )
        assert n == 1
        assert set(out) == {f"decoder.layers.0.mixer.conv1d.bias.{s}" for s in ("x", "B", "C")}

    def test_optimizer_state_split(self):
        key = "optimizer.state.exp_avg.decoder.layers.0.mixer.in_proj.weight"
        out, n = _split_mamba_projections({key: torch.randn(132, 24)}, self.ARGS)
        assert n == 1
        assert f"{key}.dt" in out and out[f"{key}.dt"].shape[0] == 4

    def test_stacked_block_splits_second_dim(self):
        # Homogeneous all-Mamba block: leading num-layers axis, split dim 1.
        out, n = _split_mamba_projections(
            {"decoder.layers.mixer.in_proj.weight": torch.randn(3, 132, 24)}, self.ARGS
        )
        assert n == 1
        assert out["decoder.layers.mixer.in_proj.weight.z"].shape == (3, 32, 24)

    def test_passthrough_and_noop_keys_untouched(self):
        src = {
            "decoder.layers.0.mixer.A_log": torch.randn(4),
            "decoder.layers.0.mixer.out_proj.weight": torch.randn(24, 32),
            "decoder.layers.0.mlp.linear_fc1.weight": torch.randn(8, 8),
        }
        out, n = _split_mamba_projections(dict(src), self.ARGS)
        assert n == 0 and set(out) == set(src)

    def test_missing_args_raises(self):
        with pytest.raises(NotImplementedError, match="Mamba"):
            _split_mamba_projections(
                {"decoder.layers.0.mixer.in_proj.weight": torch.randn(132, 8)}, None
            )


class TestOutputGroupId:
    """Sharded-convert grouping: keys of one transform group share a group id."""

    def test_swiglu_halves_share_group(self):
        w = "decoder.layers.0.mlp.linear_fc1.weight_w"
        v = "decoder.layers.0.mlp.linear_fc1.weight_v"
        assert _output_group_id(w, False) == _output_group_id(v, False)

    def test_grouped_experts_share_group(self):
        g0 = "decoder.layers.0.mlp.experts.linear_fc1.weight0"
        g3 = "decoder.layers.0.mlp.experts.linear_fc1.weight3"
        assert _output_group_id(g0, False) == _output_group_id(g3, False)

    def test_sequential_experts_share_group(self):
        e0 = "decoder.layers.0.mlp.experts.local_experts.0.linear_fc1.weight"
        e5 = "decoder.layers.0.mlp.experts.local_experts.5.linear_fc1.weight"
        assert _output_group_id(e0, False) == _output_group_id(e5, False)

    def test_layer_index_grouped_only_when_stacking(self):
        l0 = "decoder.layers.0.mlp.linear_fc1.weight"
        l1 = "decoder.layers.1.mlp.linear_fc1.weight"
        # homogeneous/stacked -> all layers of a param share a group (co-resident to stack)
        assert _output_group_id(l0, True) == _output_group_id(l1, True)
        # per-layer -> each layer is its own group (finer sharding)
        assert _output_group_id(l0, False) != _output_group_id(l1, False)

    def test_distinct_params_distinct_groups(self):
        a = "decoder.layers.0.self_attention.linear_qkv.weight"
        b = "decoder.layers.0.mlp.linear_fc2.weight"
        assert _output_group_id(a, True) != _output_group_id(b, True)
