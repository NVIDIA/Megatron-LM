# Adapted from https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/0f06205924d4e19b0723aa8c33e34a1959ca17fe/src/megatron/bridge/models/qwen_vl/modeling_qwen35/hybrid_model.py
# Source license: Apache-2.0.
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Qwen3.5 vision-language composition with Core's HybridModel decoder."""

from contextvars import ContextVar
from copy import deepcopy
from typing import Any

import torch

from megatron.bridge.models.qwen.modeling_qwen35.hybrid import qwen35_hybrid_stack_spec
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.attention import Qwen3VLSelfAttention
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.model import Qwen3VLModel
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.rope import Qwen3VLMultimodalRotaryEmbedding
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.transformer_layer import (
    Qwen3VLRotaryTransformerLayer,
)
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig

_POSITION_IDS: ContextVar[torch.Tensor | None] = ContextVar(
    "qwen35_hybrid_position_ids", default=None
)


class _HybridMultimodalRotaryEmbedding(Qwen3VLMultimodalRotaryEmbedding):
    def forward(
        self,
        max_seq_len: int,
        *,
        packed_seq: bool = False,
        cp_group: torch.distributed.ProcessGroup | None = None,
    ) -> torch.Tensor:
        position_ids = _POSITION_IDS.get()
        if position_ids is None:
            raise RuntimeError("Qwen3.5 Hybrid mRoPE requires the current forward's position_ids.")
        return super().forward(position_ids, self.mrope_section, cp_group=cp_group)


class Qwen35VLHybridAttentionLayer(Qwen3VLRotaryTransformerLayer):
    """Keep language graph wrappers separate from vision and feed-forward layers."""


def qwen35_vl_hybrid_stack_spec() -> ModuleSpec:
    """Use Qwen's absolute multimodal rotary attention in the Hybrid stack."""
    spec = qwen35_hybrid_stack_spec()
    spec.submodules.attention_layer.module = Qwen35VLHybridAttentionLayer
    spec.submodules.attention_layer.submodules.self_attention.module = Qwen3VLSelfAttention
    return spec


class Qwen35VLHybridLanguageModel(HybridModel):
    """Add multimodal rotary inputs to the already configured Hybrid decoder."""

    def __init__(
        self,
        config: TransformerConfig,
        transformer_layer_spec: ModuleSpec,
        vocab_size: int,
        max_sequence_length: int,
        *,
        rotary_percent: float,
        rotary_base: int,
        hybrid_layer_pattern: str,
        position_embedding_type: str = "mrope",
        mtp_block_spec: ModuleSpec | None = None,
        **kwargs: Any,
    ) -> None:
        if transformer_layer_spec.submodules.attention_layer.module is Qwen35VLHybridAttentionLayer:
            transformer_layer_spec = deepcopy(transformer_layer_spec)
            transformer_layer_spec.submodules.attention_layer.params["rotary_dim"] = int(
                config.kv_channels * rotary_percent
            )

        super().__init__(
            config=config,
            hybrid_stack_spec=transformer_layer_spec,
            hybrid_layer_pattern=hybrid_layer_pattern,
            vocab_size=vocab_size,
            max_sequence_length=max_sequence_length,
            position_embedding_type="rope",
            rotary_percent=rotary_percent,
            rotary_base=rotary_base,
            **kwargs,
        )
        self.rotary_pos_emb = _HybridMultimodalRotaryEmbedding(
            kv_channels=config.kv_channels,
            rotary_percent=rotary_percent,
            rotary_interleaved=config.rotary_interleaved,
            rotary_base=rotary_base,
            seq_len_interpolation_factor=kwargs.get("seq_len_interpolation_factor"),
            cp_group=self.pg_collection.cp,
        )
        self.rotary_pos_emb.mrope_section = config.mrope_section

        if self.mtp_process:
            self.mtp.register_forward_pre_hook(self._mtp_token_embedding_inputs, with_kwargs=True)

    def tie_embeddings_and_output_weights_state_dict(
        self,
        sharded_state_dict: ShardedStateDict,
        output_layer_weight_key: str,
        first_stage_word_emb_key: str,
        metadata: dict | None = None,
    ) -> None:
        """Store tied MTP weights only in the embedding entry on the last stage."""
        if self.mtp_process and not self.pre_process:
            sharded_state_dict.pop(output_layer_weight_key, None)

        super().tie_embeddings_and_output_weights_state_dict(
            sharded_state_dict,
            output_layer_weight_key,
            first_stage_word_emb_key,
            metadata if metadata is not None else {},
        )

    @staticmethod
    def _mtp_token_embedding_inputs(
        _module: torch.nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        # The GPT VLM conditions MTP on shifted token embeddings. HybridModel
        # otherwise reuses the composed vision-language embeddings, changing
        # image positions and boundary values used by the MoE auxiliary loss.
        kwargs["decoder_input"] = None
        return args, kwargs

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        *,
        visual_pos_masks: torch.Tensor | None = None,
        deepstack_visual_embeds: list[torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Pass explicit multimodal positions to Core's rotary preparation and MTP path."""
        if deepstack_visual_embeds:
            raise ValueError("Qwen3.5 does not use DeepStack visual embeddings.")
        token = _POSITION_IDS.set(position_ids)
        try:
            return super().forward(input_ids, position_ids, attention_mask, **kwargs)
        finally:
            _POSITION_IDS.reset(token)


class Qwen35VLHybridModel(Qwen3VLModel):
    """Reuse Qwen vision encoding and modality insertion with a Hybrid language model."""

    language_model_class = Qwen35VLHybridLanguageModel

    def _expose_language_model_for_cuda_graph_helper(self) -> None:
        super()._expose_language_model_for_cuda_graph_helper()
        self.position_embedding_type = "mrope"
