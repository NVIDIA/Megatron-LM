from argparse import Namespace
from unittest import TestCase
from unittest.mock import patch

import gpt_builders as gb
import megatron.core.models.engram.engram_model as engram_model

from megatron.core.models.engram.engram_module import EngramConfig
from megatron.core.models.engram.plugin import (
    apply_engram_to_transformer_layer_spec,
    build_engram_config_from_args,
)
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules


def _build_test_args(**overrides):
    args = {
        "yaml_cfg": None,
        "use_legacy_models": False,
        "spec": None,
        "transformer_impl": "local",
        "experimental_attention_variant": None,
        "num_experts": None,
        "heterogeneous_layers_config_path": None,
        "normalization": "LayerNorm",
        "qk_l2_norm": False,
        "mtp_num_layers": None,
        "padded_vocab_size": 128,
        "max_position_embeddings": 32,
        "fp16_lm_cross_entropy": False,
        "untie_embeddings_and_output_weights": False,
        "position_embedding_type": "learned_absolute",
        "rotary_percent": 1.0,
        "rotary_base": 10000,
        "use_rope_scaling": False,
        "tokenizer_model": "test-tokenizer",
        "use_engram": True,
        "engram_vocab_size": [31, 37],
        "engram_max_ngram_size": 3,
        "engram_n_embed_per_ngram": 8,
        "engram_n_head_per_ngram": 2,
        "engram_layer_ids": [1, 3],
        "engram_pad_id": 0,
        "engram_seed": 17,
        "engram_kernel_size": 5,
        "engram_hc_mult": 2,
        "engram_tokenizer_name_or_path": None,
    }
    args.update(overrides)
    return Namespace(**args)


class TestEngramPlugin(TestCase):
    def test_build_engram_config_from_args_uses_overrides(self):
        args = _build_test_args(
            tokenizer_model="fallback-tokenizer",
            engram_tokenizer_name_or_path="explicit-tokenizer",
            engram_layer_ids=[2],
        )

        config = build_engram_config_from_args(args)

        self.assertEqual(
            config,
            EngramConfig(
                engram_vocab_size=[31, 37],
                max_ngram_size=3,
                n_embed_per_ngram=8,
                n_head_per_ngram=2,
                engram_layer_ids=[2],
                pad_id=0,
                seed=17,
                kernel_size=5,
                hc_mult=2,
                tokenizer_name_or_path="explicit-tokenizer",
            ),
        )

    def test_apply_engram_to_transformer_layer_spec_wraps_module_and_block_specs(self):
        engram_config = EngramConfig(
            engram_vocab_size=[31, 37],
            max_ngram_size=3,
            n_embed_per_ngram=8,
            n_head_per_ngram=2,
            engram_layer_ids=[1],
            pad_id=0,
            tokenizer_name_or_path="test-tokenizer",
        )
        layer_spec = ModuleSpec(module=TransformerLayer, submodules=TransformerLayerSubmodules())

        wrapped_layer_spec = apply_engram_to_transformer_layer_spec(layer_spec, engram_config)
        wrapped_block_spec = apply_engram_to_transformer_layer_spec(
            TransformerBlockSubmodules(layer_specs=[layer_spec, layer_spec], layer_norm=object()),
            engram_config,
        )

        self.assertIs(wrapped_layer_spec.module, engram_model.EngramTransformerLayer)
        self.assertEqual(wrapped_layer_spec.params["engram_config"], engram_config)
        self.assertTrue(wrapped_layer_spec.params["engram_vocab_size_across_layers"][1])
        self.assertTrue(
            all(
                layer.module is engram_model.EngramTransformerLayer
                for layer in wrapped_block_spec.layer_specs
            )
        )

    def test_gpt_builder_uses_engram_model_when_flag_enabled(self):
        args = _build_test_args()
        layer_spec = ModuleSpec(module=TransformerLayer, submodules=TransformerLayerSubmodules())
        captured_kwargs = {}

        class FakeEngramGPTModel:
            def __init__(self, **kwargs):
                captured_kwargs.update(kwargs)

        with patch.object(gb, "core_transformer_config_from_args", lambda args: "config"), patch.object(
            gb, "_get_transformer_layer_spec", lambda use_te, config: layer_spec
        ), patch.object(engram_model, "EngramGPTModel", FakeEngramGPTModel):
            gb.gpt_builder(args, pre_process=True, post_process=True)

        self.assertEqual(captured_kwargs["config"], "config")
        self.assertEqual(captured_kwargs["engram_config"], build_engram_config_from_args(args))
        self.assertIs(
            captured_kwargs["transformer_layer_spec"].module, engram_model.EngramTransformerLayer
        )
