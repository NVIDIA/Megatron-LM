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

# Adapted from https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/c209ed397fb36272c2216053058cda4e37802f2d/src/megatron/bridge/models/deepseek/attention.py
# Source license: Apache-2.0.

"""MLA attention spec helpers for the DeepSeek family."""

from dataclasses import replace
from typing import Optional

from megatron.core.models.backends import get_backend
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.multi_latent_attention import MLASelfAttention
from megatron.core.transformer.multi_token_prediction import get_mtp_num_layers_to_build
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.models.gpt import GPTModelConfig


class MLASelfAttentionWithoutQueryNorm(MLASelfAttention):
    """MLA self-attention that does not add a query norm when there is no query LoRA.

    MCore derives Q and KV normalization from a single ``qk_layernorm`` flag. DeepSeek
    needs it enabled for ``kv_a_layernorm``, which every checkpoint ships. When
    ``q_lora_rank`` is None, that same flag also makes MCore fuse a query normalization
    into ``linear_q_proj`` (``QKNormConfigResolver._resolve_mla_qk_layernorm``), but the
    HF architecture defines no query-side norm in that case: ``DeepseekV3Attention``
    builds a bare ``q_proj``.

    The result is a trainable parameter with no HF counterpart, which cannot be loaded
    and is silently dropped on export. This subclass keeps the KV norm and drops the
    query norm so the converted model matches the source architecture.

    Transformer Engine is required for the no-query-LoRA case. MCore builds
    ``linear_q_proj`` from the backend's fused norm+linear implementation, which only
    Transformer Engine provides, so the local backend is rejected with an explicit
    message rather than an internal one.
    """

    def _resolve_qk_norm_config(self, submodules):
        """Replace the fused query projection with a plain one when there is no query LoRA.

        The standalone-``q_layernorm`` case is neutralised *before* delegating: an MLA spec
        may set ``q_layernorm`` to a real norm whenever ``qk_layernorm`` is on, and the
        parent resolver rejects that outright when there is no query LoRA to consume it
        (``_raise_unused_q_norm``). Dropping the query norm is exactly what this class
        exists to do, so the rejection would fire on a configuration this class already
        knows how to satisfy.
        """
        if self.config.q_lora_rank is not None:
            return super()._resolve_qk_norm_config(submodules)

        backend = get_backend(self.config.transformer_impl)
        if backend.column_parallel_layer_norm_linear() is None:
            raise ValueError(
                "DeepSeek without a query LoRA (`q_lora_rank=None`) requires "
                f"`transformer_impl='transformer_engine'`; `{self.config.transformer_impl}` "
                "provides no fused norm+linear projection. MCore's MLA resolver builds "
                "`linear_q_proj` from that fused implementation whenever `qk_layernorm` is "
                "on, and DeepSeek needs `qk_layernorm` on for `kv_a_layernorm`, so this "
                "backend cannot express the architecture."
            )

        if submodules.q_layernorm not in (None, IdentityOp):
            submodules = replace(submodules, q_layernorm=IdentityOp)

        layer_classes = super()._resolve_qk_norm_config(submodules)
        layer_classes["linear_q_proj"] = backend.column_parallel_linear()
        return layer_classes


def get_deepseek_decoder_block_spec(
    config: TransformerConfig,
    use_transformer_engine: bool,
    normalization: Optional[str] = None,
    qk_l2_norm: Optional[bool] = False,
    vp_stage: Optional[int] = None,
    pp_rank: Optional[int] = None,
) -> ModuleSpec:
    """Build the decoder block spec, omitting the query norm when ``q_lora_rank`` is None.

    The signature mirrors ``get_gpt_decoder_block_spec`` exactly, including ``vp_stage``
    and ``pp_rank``. The GPT model builder inspects the callable's parameters and only
    forwards ``vp_stage`` when it is declared, so dropping it here would leave interleaved
    pipeline parallelism calling MCore's layer-offset helper without a virtual stage,
    which asserts.

    Args:
        config: A transformer config, or a builder model config whose nested transformer
            config describes the layers.
        use_transformer_engine: Whether to build Transformer Engine submodules.
        normalization: Optional normalization override, forwarded unchanged.
        qk_l2_norm: Optional QK L2 norm flag, forwarded unchanged.
        vp_stage: Virtual pipeline stage, forwarded unchanged.
        pp_rank: Pipeline rank, forwarded unchanged.

    Returns:
        The decoder block spec, with MLA self-attention replaced by
        :class:`MLASelfAttentionWithoutQueryNorm` when there is no query LoRA.

    Raises:
        ValueError: If a builder model config without a query LoRA places MTP layers on a
            pipeline stage that has no decoder layers.
    """
    transformer_config = config.transformer if isinstance(config, GPTModelConfig) else config
    spec = get_gpt_decoder_block_spec(
        transformer_config,
        use_transformer_engine=use_transformer_engine,
        normalization=normalization,
        qk_l2_norm=qk_l2_norm,
        vp_stage=vp_stage,
        pp_rank=pp_rank,
    )
    if (
        transformer_config is not config
        and not spec.layer_specs
        and getattr(transformer_config, "q_lora_rank", None) is None
        and transformer_config.mtp_num_layers
        and get_mtp_num_layers_to_build(transformer_config, vp_stage=vp_stage, pp_rank=pp_rank)
    ):
        # The GPT model builder derives the MTP layer spec of a stage without decoder layers
        # from MCore, which would add the query norm this module drops.
        raise ValueError(
            "DeepSeek without a query LoRA (`q_lora_rank=None`) cannot place MTP layers on a "
            "pipeline stage without decoder layers: the GPT model builder derives that stage's "
            "MTP layer spec from Megatron Core, which would add a query norm that the checkpoint "
            "does not define. Use a pipeline layout that keeps the MTP layers on a stage with "
            "decoder layers."
        )
    return replace_mla_self_attention(transformer_config, spec)


def replace_mla_self_attention(config: TransformerConfig, spec: ModuleSpec) -> ModuleSpec:
    """Swap MLA self-attention for the query-norm-free variant, in place, on every layer.

    Accepts either a block spec (``.layer_specs``) or a single layer spec.
    """
    if getattr(config, "q_lora_rank", None) is not None:
        return spec

    layer_specs = getattr(spec, "layer_specs", None)
    for layer_spec in layer_specs if layer_specs is not None else [spec]:
        self_attention = getattr(layer_spec.submodules, "self_attention", None)
        if self_attention is not None and self_attention.module is MLASelfAttention:
            self_attention.module = MLASelfAttentionWithoutQueryNorm
    return spec
