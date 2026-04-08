# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from unittest.mock import MagicMock, patch

import pytest

from megatron.core.transformer.enums import AttnMaskType, LayerType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules
from megatron.core.transformer.transformer_layer import TransformerLayer

# ---------------------------------------------------------------------------
# Helpers: fake backend and config builders
# ---------------------------------------------------------------------------


class _FakeLinear:
    pass


class _FakeColumnParallelLinear:
    pass


class _FakeRowParallelLinear:
    pass


class _FakeLayerNormColumnParallelLinear:
    pass


class _FakeLayerNorm:
    pass


class _FakeQKNorm:
    pass


class _FakeCoreAttention:
    pass


def _make_backend(fuse_layernorm=True):
    """Return a mock BackendSpecProvider with deterministic return values."""
    backend = MagicMock()
    backend.linear.return_value = _FakeLinear
    backend.column_parallel_linear.return_value = _FakeColumnParallelLinear
    backend.row_parallel_linear.return_value = _FakeRowParallelLinear
    backend.column_parallel_layer_norm_linear.return_value = _FakeLayerNormColumnParallelLinear
    backend.fuse_layernorm_and_linear.return_value = fuse_layernorm
    backend.core_attention.return_value = _FakeCoreAttention

    def _layer_norm(rms_norm=False, for_qk=False):
        return _FakeQKNorm if for_qk else _FakeLayerNorm

    backend.layer_norm.side_effect = _layer_norm
    return backend


def _make_config(**overrides):
    """Return a mock TransformerConfig with sane defaults."""
    defaults = dict(
        num_layers=4,
        normalization="RMSNorm",
        qk_layernorm=False,
        multi_latent_attention=False,
        qk_l2_norm=False,
        transformer_impl="transformer_engine",
        use_kitchen=False,
        experimental_attention_variant=None,
        linear_attention_freq=None,
        moe_layer_freq=1,
        num_moe_experts=None,
        moe_grouped_gemm=False,
        moe_use_legacy_grouped_gemm=False,
        use_te_activation_func=False,
        pipeline_model_parallel_size=1,
        pipeline_model_parallel_layout=None,
        use_kitchen_attention=False,
        kitchen_attention_backend="sdpa",
        fallback_to_eager_attn=False,
    )
    defaults.update(overrides)
    cfg = MagicMock()
    for k, v in defaults.items():
        setattr(cfg, k, v)
    return cfg


# ===================================================================
# Tests for is_linear_attention_variant
# ===================================================================


class TestIsLinearAttentionVariant:
    @staticmethod
    def _fn(variant):
        from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
            is_linear_attention_variant,
        )

        return is_linear_attention_variant(variant)

    @pytest.mark.parametrize(
        "variant, expected",
        [("gated_delta_net", True), ("dsa", False), (None, False), ("some_unknown_variant", False)],
    )
    def test_variants(self, variant, expected):
        """Validate linear-attention variant classification across supported and unsupported names."""
        assert self._fn(variant) is expected


# ===================================================================
# Tests for get_moe_layer_pattern
# ===================================================================


class TestGetMoeLayerPattern:
    @staticmethod
    def _fn(config):
        from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
            get_moe_layer_pattern,
        )

        return get_moe_layer_pattern(config)

    @pytest.mark.parametrize(
        "num_layers, freq, expected",
        [(4, 1, [1, 1, 1, 1]), (6, 2, [1, 0, 1, 0, 1, 0]), (6, 3, [1, 0, 0, 1, 0, 0])],
    )
    def test_int_freq(self, num_layers, freq, expected):
        """Verify integer moe_layer_freq is expanded into the expected per-layer MoE pattern."""
        cfg = _make_config(num_layers=num_layers, moe_layer_freq=freq)
        assert self._fn(cfg) == expected

    def test_list_freq(self):
        """Verify an explicit list pattern is used as-is."""
        pattern = [1, 0, 1, 0]
        cfg = _make_config(num_layers=4, moe_layer_freq=pattern)
        assert self._fn(cfg) == pattern

    def test_list_freq_wrong_length_raises(self):
        """Verify a list with mismatched length fails fast."""
        cfg = _make_config(num_layers=4, moe_layer_freq=[1, 0])
        with pytest.raises(AssertionError, match="Invalid length"):
            self._fn(cfg)

    def test_invalid_type_raises(self):
        """Verify unsupported moe_layer_freq types raise ValueError."""
        cfg = _make_config(num_layers=4, moe_layer_freq="bad")
        with pytest.raises(ValueError, match="Invalid moe_layer_freq"):
            self._fn(cfg)


# ===================================================================
# Tests for get_linear_attention_pattern
# ===================================================================


class TestGetLinearAttentionPattern:
    @staticmethod
    def _fn(config):
        from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
            get_linear_attention_pattern,
        )

        return get_linear_attention_pattern(config)

    @pytest.mark.parametrize(
        "num_layers, freq, expected",
        [
            # Every 4th layer (1-indexed) is SDPA (0), the rest are LA (1)
            (8, 4, [1, 1, 1, 0, 1, 1, 1, 0]),
            (4, 2, [1, 0, 1, 0]),
            (3, 1, [0, 0, 0]),
        ],
    )
    def test_int_freq(self, num_layers, freq, expected):
        """Verify integer linear_attention_freq is expanded into the expected LA/SDPA pattern."""
        cfg = _make_config(num_layers=num_layers, linear_attention_freq=freq)
        assert self._fn(cfg) == expected

    def test_list_freq(self):
        """Verify an explicit linear-attention pattern list is used directly."""
        pattern = [1, 0, 1, 0]
        cfg = _make_config(num_layers=4, linear_attention_freq=pattern)
        assert self._fn(cfg) == pattern

    def test_list_freq_wrong_length_raises(self):
        """Verify list length validation for linear_attention_freq."""
        cfg = _make_config(num_layers=4, linear_attention_freq=[1, 0, 1])
        with pytest.raises(AssertionError, match="Invalid length"):
            self._fn(cfg)

    def test_none_for_non_linear_variant(self):
        """Verify non-linear variants default to all-standard attention when freq is None."""
        cfg = _make_config(
            num_layers=4, linear_attention_freq=None, experimental_attention_variant="dsa"
        )
        assert self._fn(cfg) == [0, 0, 0, 0]

    def test_none_for_linear_variant_raises(self):
        """Verify linear variants require linear_attention_freq to be explicitly set."""
        cfg = _make_config(
            num_layers=4,
            linear_attention_freq=None,
            experimental_attention_variant="gated_delta_net",
        )
        with pytest.raises(ValueError, match="linear_attention_freq is None"):
            self._fn(cfg)

    def test_invalid_type_raises(self):
        """Verify unsupported linear_attention_freq types raise ValueError."""
        cfg = _make_config(num_layers=4, linear_attention_freq=3.14)
        with pytest.raises(ValueError, match="Invalid linear_attention_freq"):
            self._fn(cfg)


# ===================================================================
# Tests for get_gated_delta_net_module_spec
# ===================================================================


class TestGetGatedDeltaNetModuleSpec:
    def test_returns_correct_module_spec(self):
        """Verify the top-level module spec targets GatedDeltaNet with expected metainfo."""
        from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
            get_gated_delta_net_module_spec,
        )
        from megatron.core.ssm.gated_delta_net import GatedDeltaNet

        backend = _make_backend()
        cfg = _make_config(normalization="RMSNorm")
        spec = get_gated_delta_net_module_spec(cfg, backend=backend)

        assert isinstance(spec, ModuleSpec)
        assert spec.module is GatedDeltaNet
        assert spec.metainfo == {"fuse_input_layernorm": True}

    def test_submodules_use_backend_modules(self):
        """Verify backend-provided projection/norm modules are wired into submodules."""
        from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
            get_gated_delta_net_module_spec,
        )

        backend = _make_backend()
        cfg = _make_config(normalization="RMSNorm")
        spec = get_gated_delta_net_module_spec(cfg, backend=backend)

        subs = spec.submodules
        assert subs.in_proj == _FakeLayerNormColumnParallelLinear
        assert subs.out_proj == _FakeRowParallelLinear
        backend.layer_norm.assert_any_call(rms_norm=True, for_qk=False)

    def test_layer_norm_normalization(self):
        """Verify LayerNorm mode passes rms_norm=False to backend.layer_norm."""
        from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
            get_gated_delta_net_module_spec,
        )

        backend = _make_backend()
        cfg = _make_config(normalization="LayerNorm")
        get_gated_delta_net_module_spec(cfg, backend=backend)
        backend.layer_norm.assert_any_call(rms_norm=False, for_qk=False)

    def test_backend_auto_resolved_when_none(self):
        """Verify backend is auto-resolved when caller does not pass one."""
        from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
            get_gated_delta_net_module_spec,
        )

        cfg = _make_config(normalization="RMSNorm")
        with patch(
            "megatron.core.models.gpt.experimental_attention_variant_module_specs"
            "._get_backend_spec_provider",
            return_value=_make_backend(),
        ):
            spec = get_gated_delta_net_module_spec(cfg, backend=None)
            assert isinstance(spec, ModuleSpec)


# ===================================================================
# Tests for get_dsa_module_spec_for_backend
# ===================================================================


class TestGetDsaModuleSpec:
    def _call(self, cfg=None, backend=None):
        from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
            get_dsa_module_spec_for_backend,
        )

        if cfg is None:
            cfg = _make_config(multi_latent_attention=True, qk_l2_norm=False, qk_layernorm=True)
        if backend is None:
            backend = _make_backend()
        return get_dsa_module_spec_for_backend(cfg, backend=backend)

    def test_requires_multi_latent_attention(self):
        """Verify DSA path rejects configs without MLA enabled."""
        from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
            get_dsa_module_spec_for_backend,
        )

        cfg = _make_config(multi_latent_attention=False, qk_l2_norm=False)
        with pytest.raises(AssertionError, match="only MLA supports"):
            get_dsa_module_spec_for_backend(cfg, backend=_make_backend())

    def test_rejects_qk_l2_norm(self):
        """Verify unsupported qk_l2_norm setting is rejected for DSA+MLA."""
        from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
            get_dsa_module_spec_for_backend,
        )

        cfg = _make_config(multi_latent_attention=True, qk_l2_norm=True)
        with pytest.raises(AssertionError, match="qk_l2_norm is not supported"):
            get_dsa_module_spec_for_backend(cfg, backend=_make_backend())

    def test_returns_mla_self_attention_spec(self):
        """Verify the returned attention module is MLA self-attention with causal mask."""
        from megatron.core.transformer.multi_latent_attention import MLASelfAttention

        spec = self._call()
        assert spec.module is MLASelfAttention
        assert spec.params == {"attn_mask_type": AttnMaskType.causal}
        assert spec.metainfo == {"fuse_input_layernorm": False}

    def test_core_attention_is_dsa(self):
        """Verify MLA core_attention is wrapped with DSAttention."""
        from megatron.core.transformer.experimental_attention_variant.dsa import DSAttention

        spec = self._call()
        core = spec.submodules.core_attention
        assert core.module is DSAttention

    def test_dsa_indexer_structure(self):
        """Verify DSA indexer wiring uses expected backend linear/norm modules."""
        from megatron.core.transformer.experimental_attention_variant.dsa import DSAIndexer

        spec = self._call()
        indexer = spec.submodules.core_attention.submodules.indexer
        assert indexer.module is DSAIndexer
        subs = indexer.submodules
        assert subs.linear_wq_b == _FakeLinear
        assert subs.linear_wk == _FakeLinear
        assert subs.k_norm == _FakeQKNorm
        assert subs.linear_weights_proj == _FakeLinear

    @pytest.mark.parametrize("normalization", ["RMSNorm", "LayerNorm"])
    def test_qk_layernorm_enabled(self, normalization):
        """Verify q/kv layernorm uses backend.layer_norm(rms_norm=..., for_qk=True)."""
        backend = _make_backend()
        cfg = _make_config(
            multi_latent_attention=True,
            qk_l2_norm=False,
            qk_layernorm=True,
            normalization=normalization,
        )
        spec = self._call(cfg=cfg, backend=backend)
        expected_rms = normalization == "RMSNorm"
        assert spec.submodules.q_layernorm == _FakeQKNorm
        assert spec.submodules.kv_layernorm == _FakeQKNorm
        # Both point to the same qk_norm object
        assert spec.submodules.q_layernorm is spec.submodules.kv_layernorm
        backend.layer_norm.assert_any_call(rms_norm=expected_rms, for_qk=True)

    def test_qk_layernorm_disabled(self):
        """Verify q/kv layernorm becomes IdentityOp, skipping backend.layer_norm for qk."""
        backend = _make_backend()
        cfg = _make_config(multi_latent_attention=True, qk_l2_norm=False, qk_layernorm=False)
        spec = self._call(cfg=cfg, backend=backend)
        assert spec.submodules.q_layernorm is IdentityOp
        assert spec.submodules.kv_layernorm is IdentityOp
        # backend.layer_norm is still called for the indexer k_norm (for_qk=True at line 94),
        # but NOT for the outer qk_norm (line 105-107 takes the else branch).
        # Exactly one for_qk=True call should exist (from the indexer, not from qk_norm).
        qk_calls = [c for c in backend.layer_norm.call_args_list if c.kwargs.get("for_qk")]
        assert (
            len(qk_calls) == 1
        ), f"Expected 1 for_qk=True call (indexer only), got {len(qk_calls)}"

    def test_linear_projections(self):
        """Verify Q/KV projection slots and backend.column_parallel_linear call count."""
        backend = _make_backend()
        cfg = _make_config(multi_latent_attention=True, qk_l2_norm=False, qk_layernorm=True)
        spec = self._call(cfg=cfg, backend=backend)
        subs = spec.submodules
        assert subs.linear_q_proj == _FakeColumnParallelLinear
        assert subs.linear_q_down_proj == _FakeLinear
        assert subs.linear_q_up_proj == _FakeColumnParallelLinear
        assert subs.linear_kv_down_proj == _FakeLinear
        assert subs.linear_kv_up_proj == _FakeColumnParallelLinear
        assert subs.linear_proj == _FakeRowParallelLinear
        # column_parallel_linear() is called exactly 3 times (q_proj, q_up_proj, kv_up_proj)
        assert backend.column_parallel_linear.call_count == 3
        assert backend.row_parallel_linear.call_count == 1


# ===================================================================
# Tests for get_experimental_attention_variant_module_spec
# ===================================================================


class TestGetExperimentalAttentionVariantModuleSpec:
    MODULE = "megatron.core.models.gpt.experimental_attention_variant_module_specs"

    @pytest.mark.parametrize(
        "variant, target_fn",
        [
            ("gated_delta_net", "get_gated_delta_net_module_spec"),
            ("dsa", "get_dsa_module_spec_for_backend"),
        ],
    )
    def test_dispatches_to_variant_handler(self, variant, target_fn):
        """Verify dispatcher routes each variant name to its corresponding builder function."""
        backend = _make_backend()
        cfg = _make_config(experimental_attention_variant=variant, normalization="RMSNorm")
        with patch(f"{self.MODULE}.{target_fn}") as mock_fn:
            mock_fn.return_value = ModuleSpec(module=MagicMock)
            from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
                get_experimental_attention_variant_module_spec,
            )

            result = get_experimental_attention_variant_module_spec(cfg, backend=backend)
            mock_fn.assert_called_once_with(config=cfg, backend=backend)
            assert result is mock_fn.return_value

    def test_invalid_variant_raises(self):
        """Verify unknown variant names raise a clear ValueError."""
        cfg = _make_config(experimental_attention_variant="unknown")
        with pytest.raises(ValueError, match="Invalid experimental attention variant"):
            from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
                get_experimental_attention_variant_module_spec,
            )

            get_experimental_attention_variant_module_spec(cfg, backend=_make_backend())


# ===================================================================
# Tests for get_transformer_layer_with_experimental_attention_variant_spec
# ===================================================================


class TestGetTransformerLayerWithExperimentalAttentionVariantSpec:
    MODULE = "megatron.core.models.gpt.experimental_attention_variant_module_specs"

    def _make_attention_spec(self, fuse_input_layernorm=True):
        """Construct a mock attention spec with configurable fuse metadata."""
        return ModuleSpec(module=MagicMock, metainfo={"fuse_input_layernorm": fuse_input_layernorm})

    def _make_mlp_spec(self, fuse_pre_mlp_layernorm=True):
        """Construct a mock MLP spec with configurable fuse metadata."""
        return ModuleSpec(
            module=MagicMock, metainfo={"fuse_pre_mlp_layernorm": fuse_pre_mlp_layernorm}
        )

    def test_all_experimental_no_moe(self):
        """Verify all layers use experimental attention and dense MLP when no MoE is configured."""
        from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
            get_transformer_layer_with_experimental_attention_variant_spec,
        )

        cfg = _make_config(
            num_layers=4,
            experimental_attention_variant="dsa",
            num_moe_experts=None,
            normalization="RMSNorm",
        )
        backend = _make_backend()
        attn_spec = self._make_attention_spec(fuse_input_layernorm=False)
        mlp_spec = self._make_mlp_spec(fuse_pre_mlp_layernorm=True)

        with (
            patch(
                f"{self.MODULE}.get_experimental_attention_variant_module_spec",
                return_value=attn_spec,
            ),
            patch(f"{self.MODULE}._get_dense_mlp_module_spec", return_value=mlp_spec),
        ):
            specs = get_transformer_layer_with_experimental_attention_variant_spec(
                cfg, backend=backend
            )

        assert len(specs) == 4
        for s in specs:
            # Each layer should share the same selected module specs in this setup.
            assert s.module is TransformerLayer
            assert s.submodules.self_attention is attn_spec
            assert s.submodules.mlp is mlp_spec

    def test_hybrid_attention_pattern(self):
        """Verify attention alternates between experimental and standard specs per pattern."""
        from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
            get_transformer_layer_with_experimental_attention_variant_spec,
        )

        cfg = _make_config(
            num_layers=4,
            experimental_attention_variant="gated_delta_net",
            linear_attention_freq=2,
            num_moe_experts=None,
            normalization="RMSNorm",
        )
        backend = _make_backend()
        exp_attn_spec = self._make_attention_spec(fuse_input_layernorm=True)
        std_attn_spec = self._make_attention_spec(fuse_input_layernorm=False)
        mlp_spec = self._make_mlp_spec(fuse_pre_mlp_layernorm=True)

        with (
            patch(
                f"{self.MODULE}.get_experimental_attention_variant_module_spec",
                return_value=exp_attn_spec,
            ),
            patch(f"{self.MODULE}._get_self_attention_module_spec", return_value=std_attn_spec),
            patch(f"{self.MODULE}._get_dense_mlp_module_spec", return_value=mlp_spec),
        ):
            specs = get_transformer_layer_with_experimental_attention_variant_spec(
                cfg, backend=backend
            )

        assert len(specs) == 4
        # Pattern for linear_attention_freq=2: [1, 0, 1, 0]
        assert specs[0].submodules.self_attention is exp_attn_spec
        assert specs[1].submodules.self_attention is std_attn_spec
        assert specs[2].submodules.self_attention is exp_attn_spec
        assert specs[3].submodules.self_attention is std_attn_spec

    def test_hybrid_moe_pattern(self):
        """Verify MLP alternates between MoE and dense specs per moe_layer_freq pattern."""
        from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
            get_transformer_layer_with_experimental_attention_variant_spec,
        )

        cfg = _make_config(
            num_layers=4,
            experimental_attention_variant="dsa",
            num_moe_experts=8,
            moe_layer_freq=2,
            normalization="RMSNorm",
        )
        backend = _make_backend()
        attn_spec = self._make_attention_spec(fuse_input_layernorm=False)
        moe_spec = self._make_mlp_spec(fuse_pre_mlp_layernorm=False)
        dense_spec = self._make_mlp_spec(fuse_pre_mlp_layernorm=True)

        with (
            patch(
                f"{self.MODULE}.get_experimental_attention_variant_module_spec",
                return_value=attn_spec,
            ),
            patch(f"{self.MODULE}._get_moe_module_spec", return_value=moe_spec),
            patch(f"{self.MODULE}._get_dense_mlp_module_spec", return_value=dense_spec),
        ):
            specs = get_transformer_layer_with_experimental_attention_variant_spec(
                cfg, backend=backend
            )

        # moe_layer_freq=2 -> [1, 0, 1, 0]
        assert specs[0].submodules.mlp is moe_spec
        assert specs[1].submodules.mlp is dense_spec
        assert specs[2].submodules.mlp is moe_spec
        assert specs[3].submodules.mlp is dense_spec


# ===================================================================
# Tests for get_transformer_block_with_experimental_attention_variant_spec
# ===================================================================


class TestGetTransformerBlockWithExperimentalAttentionVariantSpec:
    MODULE = "megatron.core.models.gpt.experimental_attention_variant_module_specs"

    @pytest.mark.parametrize(
        "num_layers,pp_size,vp_stage,pp_rank,use_layout,offset,num_layers_to_build,layout_ids,expected_ids",
        [
            # no pipeline split
            (4, 1, None, None, False, 0, 4, None, [0, 1, 2, 3]),
            # pp split (rank 1 gets [4,5,6,7])
            (8, 2, None, 1, False, 4, 4, None, [4, 5, 6, 7]),
            # vpp + pp split (example stage)
            (8, 2, 1, 0, False, 2, 2, None, [2, 3]),
            # explicit pipeline layout wins over offset/num_layers
            (8, 2, 0, 0, True, None, None, [0, 3, 5], [0, 3, 5]),
        ],
    )
    def test_get_transformer_block_with_experimental_attention_variant_spec(
        self,
        num_layers,
        pp_size,
        vp_stage,
        pp_rank,
        use_layout,
        offset,
        num_layers_to_build,
        layout_ids,
        expected_ids,
    ):
        """Verify transformer block layer slicing and vp/pp argument forwarding."""
        from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
            get_transformer_block_with_experimental_attention_variant_spec,
        )

        mock_layout = MagicMock() if use_layout else None
        if mock_layout is not None:
            # When layout is provided, it should fully control local layer selection.
            mock_layout.get_layer_id_list.return_value = layout_ids

        cfg = _make_config(
            num_layers=num_layers,
            pipeline_model_parallel_size=pp_size,
            pipeline_model_parallel_layout=mock_layout,
            normalization="RMSNorm",
        )
        backend = _make_backend()
        fake_layer_specs = [
            ModuleSpec(module=TransformerLayer, submodules=MagicMock()) for _ in range(num_layers)
        ]

        with (
            patch(f"{self.MODULE}._get_backend_spec_provider", return_value=backend),
            patch(
                f"{self.MODULE}.get_transformer_layer_with_experimental_attention_variant_spec",
                return_value=fake_layer_specs,
            ),
        ):
            if use_layout:
                result = get_transformer_block_with_experimental_attention_variant_spec(
                    cfg, vp_stage=vp_stage, pp_rank=pp_rank
                )
                mock_layout.get_layer_id_list.assert_called_once_with(
                    layer_type=LayerType.decoder, vp_stage=vp_stage, pp_rank=pp_rank
                )
            else:
                # Without explicit layout, slicing comes from offset + num_layers_to_build.
                with (
                    patch(
                        f"{self.MODULE}.get_transformer_layer_offset", return_value=offset
                    ) as mock_offset,
                    patch(
                        f"{self.MODULE}.get_num_layers_to_build", return_value=num_layers_to_build
                    ) as mock_num_layers,
                ):
                    result = get_transformer_block_with_experimental_attention_variant_spec(
                        cfg, vp_stage=vp_stage, pp_rank=pp_rank
                    )
                mock_offset.assert_called_once_with(cfg, vp_stage=vp_stage, pp_rank=pp_rank)
                mock_num_layers.assert_called_once_with(cfg, vp_stage=vp_stage, pp_rank=pp_rank)

        assert isinstance(result, TransformerBlockSubmodules)
        assert result.layer_specs == [fake_layer_specs[i] for i in expected_ids]
