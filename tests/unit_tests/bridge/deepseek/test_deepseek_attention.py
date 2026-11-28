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

# Adapted from https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/c209ed397fb36272c2216053058cda4e37802f2d/tests/unit_tests/models/deepseek/test_deepseek_attention.py
# Source license: Apache-2.0.

"""Unit tests for the DeepSeek MLA attention spec helpers."""

import inspect
from dataclasses import dataclass
from functools import partial
from types import SimpleNamespace

import pytest
from transformers import DeepseekV3Config

from megatron.bridge.models.deepseek import attention as attention_module
from megatron.bridge.models.deepseek.attention import (
    MLASelfAttentionWithoutQueryNorm,
    get_deepseek_decoder_block_spec,
    replace_mla_self_attention,
)
from megatron.bridge.models.deepseek.deepseek_v3_bridge import DeepSeekV3Bridge
from megatron.bridge.models.gpt.model_config import BridgeGPTMLAModelConfig
from megatron.bridge.models.transformer_config import MLATransformerConfig
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.transformer.multi_latent_attention import MLASelfAttention
from megatron.core.transformer.spec_utils import ModuleSpec


def _mla_submodules():
    spec = get_gpt_layer_with_transformer_engine_spec(
        num_experts=8, moe_grouped_gemm=False, qk_layernorm=True, multi_latent_attention=True
    )
    return spec.submodules.self_attention.submodules


@dataclass
class _MLAConfig:
    q_lora_rank: int | None
    qk_layernorm: bool = True
    qk_l2_norm: bool = False
    multi_latent_attention: bool = True
    experimental_attention_variant: str | None = None
    normalization: str = "RMSNorm"
    transformer_impl: str = "transformer_engine"
    layernorm_epsilon: float = 1e-5


def _config(q_lora_rank):
    return _MLAConfig(q_lora_rank=q_lora_rank)


def _module_name(module_or_spec):
    """Return the implementation name for class- and ModuleSpec-valued resolver results."""
    module = module_or_spec.module if isinstance(module_or_spec, ModuleSpec) else module_or_spec
    return module.__name__


def _resolve(q_lora_rank):
    # `_resolve_qk_norm_config` only reads `self.config`, so bypass __init__ (which would
    # need process groups) but keep a real instance so its zero-arg `super()` resolves.
    attention = MLASelfAttentionWithoutQueryNorm.__new__(MLASelfAttentionWithoutQueryNorm)
    object.__setattr__(attention, "config", _config(q_lora_rank))
    return attention._resolve_qk_norm_config(_mla_submodules())


class TestMLASelfAttentionWithoutQueryNorm:
    """DeepSeek MLA must not gain a query norm the HF architecture does not define."""

    def test_no_query_lora_drops_the_fused_query_norm(self):
        """With q_lora_rank=None, HF has no query norm, so linear_q_proj must stay unfused."""
        resolved = _resolve(q_lora_rank=None)
        assert "LayerNorm" not in _module_name(resolved["linear_q_proj"])

    def test_no_query_lora_keeps_the_kv_norm(self):
        """kv_a_layernorm exists in every DeepSeek checkpoint and must still be built."""
        resolved = _resolve(q_lora_rank=None)
        assert "LayerNorm" in _module_name(resolved["linear_kv_up_proj"])

    def test_query_lora_is_untouched(self):
        """With a query LoRA the norm belongs on linear_q_up_proj and maps to q_a_layernorm."""
        resolved = _resolve(q_lora_rank=1536)
        assert "LayerNorm" in _module_name(resolved["linear_q_up_proj"])
        assert "LayerNorm" in _module_name(resolved["linear_kv_up_proj"])

    @pytest.mark.parametrize("q_lora_rank", [None, 1536])
    def test_standalone_norms_stay_disabled(self, q_lora_rank):
        """Norms stay fused into the projections; no standalone q/kv norm modules appear."""
        resolved = _resolve(q_lora_rank)
        assert _module_name(resolved["q_layernorm"]) == "IdentityOp"
        assert _module_name(resolved["kv_layernorm"]) == "IdentityOp"


class TestDeepSeekBridgeUsesTheSpecHelper:
    """`kakaocorp/kanana-2-30b-a3b-thinking` uses DeepSeekV3Bridge and ships `q_lora_rank: null`."""

    def test_model_config_installs_the_spec_helper(self):
        """The bridge must build its decoder block through get_deepseek_decoder_block_spec."""
        kwargs = DeepSeekV3Bridge().hf_config_to_model_config_kwargs(
            DeepseekV3Config(
                first_k_dense_replace=1,
                num_hidden_layers=4,
                moe_intermediate_size=128,
                n_shared_experts=1,
                q_lora_rank=None,
                torch_dtype="bfloat16",
            )
        )

        assert kwargs["qk_layernorm"] is True, "the KV norm still has to be requested"
        assert kwargs["transformer_layer_spec"].func is get_deepseek_decoder_block_spec


def _local_mla_submodules(qk_layernorm=True):
    """MLA submodules as the non-Transformer-Engine backend builds them.

    This backend differs from the TE one in exactly the way that matters here: it puts a
    real module in ``q_layernorm`` instead of folding the norm into the projection.
    """
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec

    spec = get_gpt_layer_local_spec(
        num_experts=8,
        moe_grouped_gemm=False,
        qk_layernorm=qk_layernorm,
        multi_latent_attention=True,
    )
    return spec.submodules.self_attention.submodules


class TestSpecBuilderContract:
    """The helper stands in for `get_gpt_decoder_block_spec`, so it must accept its arguments."""

    def test_signature_matches_the_mcore_builder(self):
        """Every parameter MCore's builder takes has to be taken here too.

        The GPT model builder decides whether to pass `vp_stage` by inspecting this
        signature. A missing parameter is not a type error at the call site; it silently
        routes interleaved pipeline parallelism into MCore's layer-offset helper without a
        virtual stage, which asserts.
        """
        ours = inspect.signature(get_deepseek_decoder_block_spec).parameters
        theirs = inspect.signature(get_gpt_decoder_block_spec).parameters
        assert list(ours) == list(theirs)
        for name, their_param in theirs.items():
            assert ours[name].default == their_param.default, name

    def test_virtual_stage_arguments_reach_mcore(self, monkeypatch):
        """vp_stage and pp_rank must be forwarded, not dropped."""
        seen = {}

        def _capture(config, **kwargs):
            seen.update(kwargs)
            return SimpleNamespace(layer_specs=[])

        monkeypatch.setattr(attention_module, "get_gpt_decoder_block_spec", _capture)
        get_deepseek_decoder_block_spec(
            _config(q_lora_rank=None), use_transformer_engine=True, vp_stage=1, pp_rank=2
        )

        assert seen["vp_stage"] == 1
        assert seen["pp_rank"] == 2

    def test_the_partial_the_bridge_builds_still_declares_vp_stage(self):
        """The builder inspects the partial, so binding use_transformer_engine must not hide it."""
        bound = partial(get_deepseek_decoder_block_spec, use_transformer_engine=True)
        assert "vp_stage" in inspect.signature(bound).parameters

    def test_builder_model_config_builds_from_its_transformer_config(self, monkeypatch):
        """The GPT builder passes its model config; MCore builds the layers from the nested transformer config."""
        seen = {}

        def _capture(config, **kwargs):
            seen["config"] = config
            return SimpleNamespace(layer_specs=[])

        monkeypatch.setattr(attention_module, "get_gpt_decoder_block_spec", _capture)
        transformer = MLATransformerConfig(
            num_layers=2, hidden_size=128, num_attention_heads=4, q_lora_rank=None, kv_lora_rank=32
        )
        model_config = BridgeGPTMLAModelConfig(transformer=transformer, vocab_size=256)

        get_deepseek_decoder_block_spec(model_config, use_transformer_engine=True, vp_stage=None)

        assert seen["config"] is transformer


class TestLocalBackend:
    """The local backend cannot express this architecture; say so, and say it clearly."""

    def test_local_backend_is_rejected_by_name(self):
        """Without a query LoRA the local backend has no viable MLA spec at all.

        MCore builds `linear_q_proj` from the backend's fused norm+linear implementation
        whenever `qk_layernorm` is on, and the local backend has none. Keeping the query
        norm instead trips `_raise_unused_q_norm`, so both branches are closed, and
        DeepSeek needs `qk_layernorm` on for `kv_a_layernorm`.
        """
        attention = MLASelfAttentionWithoutQueryNorm.__new__(MLASelfAttentionWithoutQueryNorm)
        config = _config(q_lora_rank=None)
        config.transformer_impl = "local"
        object.__setattr__(attention, "config", config)

        with pytest.raises(ValueError, match="transformer_engine"):
            attention._resolve_qk_norm_config(_local_mla_submodules())

    def test_local_spec_with_a_query_lora_is_untouched(self):
        """The query-LoRA path must resolve exactly as it did before."""
        attention = MLASelfAttentionWithoutQueryNorm.__new__(MLASelfAttentionWithoutQueryNorm)
        config = _config(q_lora_rank=1536)
        config.transformer_impl = "local"
        object.__setattr__(attention, "config", config)

        resolved = attention._resolve_qk_norm_config(_local_mla_submodules())

        assert resolved is not None


class TestReplaceMLASelfAttention:
    """The swap works on a block spec and on a single layer spec."""

    def test_replaces_attention_on_a_bare_layer_spec(self):
        layer_spec = get_gpt_layer_with_transformer_engine_spec(
            num_experts=8, moe_grouped_gemm=False, qk_layernorm=True, multi_latent_attention=True
        )
        assert layer_spec.submodules.self_attention.module is MLASelfAttention

        replace_mla_self_attention(_config(q_lora_rank=None), layer_spec)

        assert layer_spec.submodules.self_attention.module is MLASelfAttentionWithoutQueryNorm

    def test_leaves_the_query_lora_path_alone(self):
        """With a query LoRA the stock attention is correct and must stay."""
        layer_spec = get_gpt_layer_with_transformer_engine_spec(
            num_experts=8, moe_grouped_gemm=False, qk_layernorm=True, multi_latent_attention=True
        )

        replace_mla_self_attention(_config(q_lora_rank=1536), layer_spec)

        assert layer_spec.submodules.self_attention.module is MLASelfAttention


class TestStandaloneMTPStage:
    """The GPT builder derives the MTP layer spec of a stage without decoder layers from MCore."""

    @staticmethod
    def _model_config(q_lora_rank):
        transformer = MLATransformerConfig(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=4,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=32,
            mtp_num_layers=1,
        )
        return BridgeGPTMLAModelConfig(transformer=transformer, vocab_size=256)

    @staticmethod
    def _stage_without_decoder_layers(monkeypatch, *, mtp_layers_on_stage):
        monkeypatch.setattr(
            attention_module,
            "get_gpt_decoder_block_spec",
            lambda config, **kwargs: SimpleNamespace(layer_specs=[]),
        )
        monkeypatch.setattr(
            attention_module,
            "get_mtp_num_layers_to_build",
            lambda *args, **kwargs: mtp_layers_on_stage,
        )

    def test_builder_rejects_a_standalone_mtp_stage_without_query_lora(self, monkeypatch):
        """MCore's own MTP layer spec would bring the query norm back."""
        self._stage_without_decoder_layers(monkeypatch, mtp_layers_on_stage=1)

        with pytest.raises(ValueError, match="stage without decoder layers"):
            get_deepseek_decoder_block_spec(
                self._model_config(q_lora_rank=None), use_transformer_engine=True
            )

    def test_builder_accepts_a_standalone_mtp_stage_with_query_lora(self, monkeypatch):
        """With a query LoRA, MCore's own MTP layer spec is already correct."""
        self._stage_without_decoder_layers(monkeypatch, mtp_layers_on_stage=1)

        spec = get_deepseek_decoder_block_spec(
            self._model_config(q_lora_rank=1536), use_transformer_engine=True
        )

        assert spec.layer_specs == []

    def test_builder_accepts_a_stage_without_mtp_layers(self, monkeypatch):
        """A stage with neither decoder nor MTP layers needs no query-norm-free spec."""
        self._stage_without_decoder_layers(monkeypatch, mtp_layers_on_stage=0)

        spec = get_deepseek_decoder_block_spec(
            self._model_config(q_lora_rank=None), use_transformer_engine=True
        )

        assert spec.layer_specs == []
