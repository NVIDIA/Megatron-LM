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
# Adapted from https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/0f06205924d4e19b0723aa8c33e34a1959ca17fe/tests/unit_tests/models/qwen_vl/test_qwen35_vl_builder.py
# Source license: Apache-2.0.

"""Unit tests for the Qwen3.5-VL builder-backed model config and builder."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import yaml

from megatron.bridge.models.conversion.checkpoint import load_model_config
from megatron.bridge.models.qwen_vl.modeling_qwen35.hybrid_model import Qwen35VLHybridLanguageModel
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.attention import Qwen3VLSelfAttention
from megatron.bridge.models.qwen_vl.qwen35_vl_builder import Qwen35VLModelBuilder
from megatron.bridge.models.qwen_vl.qwen35_vl_config import (
    Qwen35VLModelConfig,
    Qwen35VLTransformerConfig,
)
from megatron.core.dist_checkpointing.core import CheckpointingConfig, save_config
from megatron.core.dist_checkpointing.mapping import ShardedTensor
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.training.config.utils import serialize_value
from megatron.training.config.yaml_utils import safe_yaml_representers

_BUILDER_MODULE = "megatron.bridge.models.qwen_vl.qwen35_vl_builder"


def _model_config(**overrides) -> Qwen35VLModelConfig:
    """Map a small Qwen3.5 MoE vision-language HF config to its builder config."""
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeConfig

    from megatron.bridge.models.qwen_vl.qwen35_vl_bridge import Qwen35VLMoEBridge

    hf_config = Qwen3_5MoeConfig(
        text_config=dict(
            hidden_size=64,
            num_hidden_layers=8,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
            vocab_size=128,
            moe_intermediate_size=32,
            shared_expert_intermediate_size=32,
            num_experts=4,
            num_experts_per_tok=2,
            mtp_num_hidden_layers=1,
            dtype="bfloat16",
        ),
        vision_config=dict(
            depth=2, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64
        ),
    )
    model_config = Qwen35VLMoEBridge().hf_config_to_model_config(hf_config)
    for name, value in overrides.items():
        setattr(model_config, name, value)
    return model_config


class _FakeGroup:
    def __init__(self, rank: int, size: int):
        self._rank = rank
        self._size = size

    def rank(self) -> int:
        return self._rank

    def size(self) -> int:
        return self._size


def _pg_collection(pp_rank: int = 0, pp_size: int = 1) -> SimpleNamespace:
    return SimpleNamespace(pp=_FakeGroup(pp_rank, pp_size))


@pytest.fixture
def build_calls(monkeypatch):
    """Replace the model class, recording the arguments the builder passes."""
    model = Mock()
    model_class = Mock(return_value=model)
    monkeypatch.setattr(f"{_BUILDER_MODULE}.Qwen35VLHybridModel", model_class)
    return dict(model=model, model_class=model_class)


class TestQwen35VLModelConfig:
    def test_defaults(self):
        config = Qwen35VLModelConfig(
            transformer=Qwen35VLTransformerConfig(
                num_layers=4, hidden_size=64, num_attention_heads=4
            )
        )

        assert config.position_embedding_type == "mrope"
        assert config.scatter_embedding_sequence_parallel is False
        assert config.vision_config is None
        assert config.special_token_ids == {"images": 248056}
        assert Qwen35VLModelConfig.modality_keys == {"images": "qwen_visual"}
        assert config.vision_recompute_granularity == "inherit"
        assert config.vision_cuda_graph_impl == "none"
        assert config.transformer.apply_rotary_pos_emb_in_fp32 is False
        assert config.transformer.spatial_merge_size == 2
        assert config.transformer.vision_dp_when_cp is False
        assert config.get_builder_cls() is Qwen35VLModelBuilder

    def test_flat_overrides_route_to_their_owner(self):
        config = _model_config()

        config.freeze_vision_model = True
        config.vision_dp_when_cp = True
        config.tensor_model_parallel_size = 2

        assert "freeze_vision_model" not in vars(config.transformer)
        assert config.freeze_vision_model is True
        assert config.transformer.vision_dp_when_cp is True
        assert config.transformer.tensor_model_parallel_size == 2
        # The providers carried these as undeclared attributes; the builder config rejects them.
        for name in ("head_dim", "hf_text_config", "audio_token_id"):
            with pytest.raises(AttributeError):
                setattr(config, name, None)

    @pytest.mark.parametrize(("context_parallel_size", "per_token_loss"), [(1, False), (2, True)])
    def test_finalize_computes_per_token_loss_for_context_parallelism(
        self, context_parallel_size, per_token_loss
    ):
        config = _model_config(context_parallel_size=context_parallel_size)

        config.finalize()

        assert config.calculate_per_token_loss is per_token_loss

    def test_serialization_round_trip(self, tmp_path):
        config = _model_config(
            freeze_vision_model=True,
            vision_recompute_granularity="full",
            vision_recompute_method="block",
            vision_recompute_num_layers=2,
            vision_dp_when_cp=True,
        )
        serialized = serialize_value(config)
        with safe_yaml_representers():
            (tmp_path / "run_config.yaml").write_text(yaml.safe_dump({"model": serialized}))

        save_config(
            CheckpointingConfig(sharded_backend="torch_dist", sharded_backend_version=1), tmp_path
        )
        restored, _ = load_model_config(str(tmp_path))

        assert isinstance(restored, Qwen35VLModelConfig)
        assert isinstance(restored.transformer, Qwen35VLTransformerConfig)
        assert restored.get_builder_cls() is Qwen35VLModelBuilder
        assert type(restored.vision_config) is type(config.vision_config)
        assert restored.vision_config.to_dict() == config.vision_config.to_dict()
        for name in (
            "freeze_vision_model",
            "vision_recompute_granularity",
            "vision_recompute_method",
            "vision_recompute_num_layers",
            "vision_dp_when_cp",
            "image_token_id",
            "mrope_section",
            "experimental_attention_variant",
            "linear_attention_freq",
            "num_moe_experts",
            "mtp_num_layers",
            "attention_softmax_in_fp32",
            "activation_func",
            "params_dtype",
            "num_layers",
            "hybrid_layer_pattern",
            "hybrid_stack_spec",
        ):
            assert getattr(restored, name) == getattr(config, name), name
        assert (
            yaml.safe_load((tmp_path / "run_config.yaml").read_text())["model"]["_builder_"]
            == config.builder
        )


class TestQwen35VLModelBuilder:
    def test_build_model_combines_language_and_vision(self, build_calls):
        config = _model_config()

        model = Qwen35VLModelBuilder(config).build_model(
            _pg_collection(pp_rank=1, pp_size=2),
            pre_process=False,
            post_process=True,
            vp_stage=None,
        )

        assert model is build_calls["model"]
        kwargs = build_calls["model_class"].call_args.kwargs
        assert kwargs["language_transformer_config"] is not config.transformer
        assert kwargs["language_transformer_config"].num_layers == config.num_layers
        assert kwargs["model_config"] is not config
        assert kwargs["language_model_kwargs"]["hybrid_layer_pattern"].startswith(
            config.hybrid_layer_pattern
        )
        assert kwargs["vision_transformer_config"] is config.vision_config
        assert config.vision_config.dtype == config.params_dtype
        assert (kwargs["pre_process"], kwargs["post_process"], kwargs["vp_stage"]) == (
            False,
            True,
            None,
        )
        assert "mtp_block_spec" not in kwargs
        # Attention layers take mRoPE; the GDN mixer keeps Core's module. HybridModel builds MTP from these specs.
        stack = kwargs["language_transformer_layer_spec"].submodules
        assert stack.attention_layer.submodules.self_attention.module is Qwen3VLSelfAttention
        assert stack.gdn_layer.submodules.self_attention.module is not Qwen3VLSelfAttention
        build_calls["model"].freeze.assert_not_called()
        # Without a Float16Module wrapper the vision encoder would stay in fp32.
        build_calls["model"].vision_model.to.assert_called_once_with(dtype=config.params_dtype)

    def test_build_model_skips_cast_without_vision_encoder(self, build_calls):
        build_calls["model"].vision_model = None

        model = Qwen35VLModelBuilder(_model_config()).build_model(
            _pg_collection(pp_rank=1, pp_size=2), pre_process=False, post_process=True
        )

        assert model.vision_model is None

    @pytest.mark.parametrize(
        ("pp_rank", "pp_size", "vp_stage", "vp_size", "pre_process", "post_process"),
        [
            (0, 1, None, None, True, True),
            (0, 2, None, None, True, False),
            (1, 2, None, None, False, True),
            (0, 2, 1, 2, False, False),
            (1, 2, 1, 2, False, True),
        ],
    )
    def test_build_model_derives_stage_flags(
        self,
        build_calls,
        monkeypatch,
        pp_rank,
        pp_size,
        vp_stage,
        vp_size,
        pre_process,
        post_process,
    ):
        # Megatron-Core reads stage ranks from the process group only once distributed is initialized.
        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
        config = _model_config(
            pipeline_model_parallel_size=pp_size,
            virtual_pipeline_model_parallel_size=vp_size,
            pipeline_dtype=torch.bfloat16,
        )
        original_pattern = config.hybrid_layer_pattern

        model = Qwen35VLModelBuilder(config).build_model(
            _pg_collection(pp_rank, pp_size), vp_stage=vp_stage
        )

        kwargs = build_calls["model_class"].call_args.kwargs
        assert (kwargs["pre_process"], kwargs["post_process"]) == (pre_process, post_process)
        pattern = kwargs["language_model_kwargs"]["hybrid_layer_pattern"]
        main, mtp = pattern.split("/", 1)
        segments = main.split("|")
        assert len(segments) == pp_size * (vp_size or 1)
        assert all(len(segment) == config.num_layers // len(segments) for segment in segments)
        assert "".join(segments) == original_pattern
        assert mtp == config.mtp_hybrid_override_pattern
        assert config.hybrid_layer_pattern == original_pattern
        if vp_stage is not None:
            assert model.vp_stage == vp_stage

    def test_build_model_freezes_requested_modules(self, build_calls):
        config = _model_config(freeze_language_model=True, freeze_vision_projection=True)

        Qwen35VLModelBuilder(config).build_model(
            _pg_collection(), pre_process=True, post_process=True
        )

        build_calls["model"].freeze.assert_called_once_with(
            freeze_language_model=True, freeze_vision_model=False, freeze_vision_projection=True
        )

    def test_build_model_rejects_other_configs(self, build_calls):
        from megatron.bridge.models.transformer_config import TransformerConfig
        from megatron.training.models.gpt import GPTModelConfig

        text_config = GPTModelConfig(
            transformer=TransformerConfig(num_layers=2, hidden_size=64, num_attention_heads=4),
            vocab_size=128,
        )
        with pytest.raises(TypeError, match="Qwen35VLModelConfig"):
            Qwen35VLModelBuilder(text_config).build_model(_pg_collection())

    def test_build_model_requires_vision_config(self, build_calls):
        with pytest.raises(ValueError, match="vision_config"):
            Qwen35VLModelBuilder(_model_config(vision_config=None)).build_model(_pg_collection())

    def test_build_model_rejects_custom_layer_spec(self, build_calls):
        config = _model_config()
        config.transformer_layer_spec = ModuleSpec(module=torch.nn.Identity)

        with pytest.raises(ValueError, match="transformer_layer_spec"):
            Qwen35VLModelBuilder(config).build_model(_pg_collection())


@pytest.mark.parametrize("pre_process", [False, True])
def test_vl_tied_mtp_checkpoint_uses_embedding_on_last_pipeline_stage(pre_process):
    model = Qwen35VLHybridLanguageModel.__new__(Qwen35VLHybridLanguageModel)
    torch.nn.Module.__init__(model)
    model.mtp_process = True
    model.pre_process = pre_process
    model.post_process = True
    embedding_key = "language_model.embedding.word_embeddings.weight"
    output_key = "language_model.output_layer.weight"
    embedding = ShardedTensor.from_rank_offsets(embedding_key, torch.ones(2, 2))
    state = {
        embedding_key: embedding,
        output_key: ShardedTensor.from_rank_offsets(output_key, torch.ones(2, 2)),
    }

    model.tie_embeddings_and_output_weights_state_dict(state, output_key, embedding_key)

    assert state[embedding_key] is embedding
    assert (output_key in state) is pre_process
