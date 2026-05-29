# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_dsa_module_spec_for_backend,
    get_experimental_attention_variant_module_spec,
)
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant.absorbed_mla import (
    AbsorbedMLASelfAttention,
)
from megatron.core.transformer.experimental_attention_variant.dsa import DSAIndexer, DSAttention
from megatron.core.transformer.transformer_config import MLATransformerConfig
from tests.unit_tests.test_utilities import Utils


@pytest.mark.internal
class TestDSAModuleSpecDispatch:
    """Tests for get_dsa_module_spec_for_backend and get_experimental_attention_variant_module_spec."""

    @pytest.fixture(scope='class', autouse=True)
    def setup_method(self):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        yield
        Utils.destroy_model_parallel()

    def _make_dsa_config(self, **kwargs):
        config_kwargs = dict(
            num_layers=2,
            hidden_size=256,
            num_attention_heads=16,
            use_cpu_initialization=True,
            bf16=True,
            params_dtype=torch.bfloat16,
            q_lora_rank=64,
            kv_lora_rank=64,
            qk_head_dim=64,
            qk_pos_emb_head_dim=32,
            v_head_dim=64,
            rope_type='rope',
            rotary_base=10000,
            rotary_percent=1.0,
            dsa_indexer_n_heads=8,
            dsa_indexer_head_dim=64,
            dsa_indexer_topk=32,
        )
        config_kwargs.update(kwargs)
        return MLATransformerConfig(**config_kwargs)

    def test_get_experimental_attention_variant_module_spec_dsa(self):
        """get_experimental_attention_variant_module_spec dispatches to DSA for variant='dsa'."""
        config = self._make_dsa_config(experimental_attention_variant="dsa")
        spec = get_experimental_attention_variant_module_spec(config)
        assert spec.module == AbsorbedMLASelfAttention
        assert spec.submodules.core_attention.module == DSAttention

    def test_get_dsa_module_spec_for_backend(self):
        """get_dsa_module_spec_for_backend returns the correct full spec structure."""
        from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider

        config = self._make_dsa_config()
        backend = TESpecProvider()
        spec = get_dsa_module_spec_for_backend(config, backend=backend)
        assert spec.module == AbsorbedMLASelfAttention
        assert spec.submodules.core_attention.module == DSAttention
        assert spec.submodules.core_attention.submodules.indexer.module == DSAIndexer
        assert spec.params["attn_mask_type"] == AttnMaskType.causal

    def test_get_dsa_module_spec_requires_mla(self):
        """get_dsa_module_spec_for_backend rejects configs without MLA."""
        from megatron.core.transformer import TransformerConfig as _TransformerConfig

        config = _TransformerConfig(num_layers=2, hidden_size=256, num_attention_heads=4)
        with pytest.raises(AssertionError, match="only MLA supports sparse attention"):
            get_dsa_module_spec_for_backend(config, backend=None)

    def test_get_dsa_module_spec_rejects_qk_l2_norm(self):
        """get_dsa_module_spec_for_backend rejects configs with qk_l2_norm=True."""
        config = self._make_dsa_config(qk_l2_norm=True)
        with pytest.raises(AssertionError, match="qk_l2_norm is not supported"):
            get_dsa_module_spec_for_backend(config, backend=None)
