# Adapted from https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/0f06205924d4e19b0723aa8c33e34a1959ca17fe/src/megatron/bridge/models/qwen_vl/modelling_qwen3_vl/text_model.py
# Source license: Apache-2.0.
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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


"""
Copied from https://github.com/Thaurun/mbridge/blob/4462d1e284626d2ed9d3e3e
3e5a40f2ee42a2c74/mbridge/models/qwen3_vl/gpt_model.py
"""

from dataclasses import replace
from typing import Any, Callable, Literal, Optional

import torch
from torch import Tensor

from megatron.bridge.models.logit_dtype import logit_dtype_kwarg
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.rope import Qwen3VLMultimodalRotaryEmbedding
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.transformer_block import (
    Qwen3VLTransformerBlock,
)
from megatron.bridge.models.transformer_config import TransformerConfig
from megatron.bridge.utils.packed_seq_utils import get_packed_seq_q_cu_seqlens
from megatron.core import tensor_parallel
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.quantization.utils import get_quant_config_or_none
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.utils import deprecate_inference_params


def _get_mtp_packed_seq_params(packed_seq_params: PackedSeqParams | None) -> PackedSeqParams | None:
    """Use physical padded offsets for MTP token rolling without changing attention metadata."""
    if packed_seq_params is None or packed_seq_params.cu_seqlens_q_padded is None:
        return packed_seq_params

    _, cu_seqlens_q = get_packed_seq_q_cu_seqlens(packed_seq_params)
    cu_seqlens_kv = packed_seq_params.cu_seqlens_kv_padded
    if cu_seqlens_kv is None:
        cu_seqlens_kv = cu_seqlens_q
    return replace(packed_seq_params, cu_seqlens_q=cu_seqlens_q, cu_seqlens_kv=cu_seqlens_kv)


class Qwen3VLGPTModel(GPTModel):
    """Qwen3-VL GPT model with vision-language capabilities."""

    def __init__(
        self,
        config: TransformerConfig,
        transformer_layer_spec: ModuleSpec,
        vocab_size: int,
        max_sequence_length: int,
        pre_process: bool = True,
        post_process: bool = True,
        fp16_lm_cross_entropy: bool = False,
        logit_dtype: torch.dtype | None = None,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = False,
        position_embedding_type: Literal[
            "learned_absolute", "rope", "mrope", "none"
        ] = "learned_absolute",
        rotary_percent: float = 1.0,
        rotary_base: int = 10000,
        rope_scaling: bool = False,
        rope_scaling_factor: float = 8.0,
        scatter_embedding_sequence_parallel: bool = True,
        seq_len_interpolation_factor: Optional[float] = None,
        mtp_block_spec: Optional[ModuleSpec] = None,
        vp_stage: Optional[int] = None,
        pg_collection: ProcessGroupCollection = None,
    ) -> None:
        super().__init__(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=vocab_size,
            max_sequence_length=max_sequence_length,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=fp16_lm_cross_entropy,
            **logit_dtype_kwarg(GPTModel, logit_dtype),
            parallel_output=parallel_output,
            share_embeddings_and_output_weights=share_embeddings_and_output_weights,
            position_embedding_type=position_embedding_type,
            rotary_percent=rotary_percent,
            rotary_base=rotary_base,
            rope_scaling=rope_scaling,
            rope_scaling_factor=rope_scaling_factor,
            scatter_embedding_sequence_parallel=scatter_embedding_sequence_parallel,
            seq_len_interpolation_factor=seq_len_interpolation_factor,
            mtp_block_spec=mtp_block_spec,
            vp_stage=vp_stage,
            pg_collection=pg_collection,
        )

        # rebuild rope
        self.rotary_pos_emb = Qwen3VLMultimodalRotaryEmbedding(
            kv_channels=self.config.kv_channels,
            rotary_percent=rotary_percent,
            rotary_interleaved=self.config.rotary_interleaved,
            seq_len_interpolation_factor=seq_len_interpolation_factor,
            rotary_base=rotary_base,
            cp_group=self.pg_collection.cp,
        )
        self.mrope_section = self.config.mrope_section
        assert (
            self.mrope_section is not None
        ), "mrope require mrope_section setting, but we got None from TransformerConfig"

        # rebuild the transformer block
        self.decoder = Qwen3VLTransformerBlock(
            config=self.config,
            spec=transformer_layer_spec,
            pre_process=self.pre_process,
            post_process=self.post_process,
            vp_stage=vp_stage,
            pg_collection=pg_collection,
        )
        for name, module in self.decoder.named_modules(prefix="decoder"):
            if hasattr(module, "finish_init"):
                quant_config = get_quant_config_or_none(name, self.config.quant_recipe)
                module.finish_init(quant_config)

    def tie_embeddings_and_output_weights_state_dict(
        self,
        sharded_state_dict: ShardedStateDict,
        output_layer_weight_key: str,
        first_stage_word_emb_key: str,
        metadata: dict | None = None,
    ) -> None:
        """Tie embedding/output checkpoint entries for Qwen3-VL MTP pipeline stages."""
        if getattr(self, "mtp_process", False) and not self.pre_process:
            sharded_state_dict.pop(output_layer_weight_key, None)
            return

        super().tie_embeddings_and_output_weights_state_dict(
            sharded_state_dict,
            output_layer_weight_key,
            first_stage_word_emb_key,
            metadata if metadata is not None else {},
        )

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        decoder_input: Tensor = None,
        labels: Tensor = None,
        inference_context: BaseInferenceContext = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
        runtime_gather_output: Optional[bool] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
        loss_mask: Optional[Tensor] = None,
        mtp_input_mask: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
        # args for deepstack
        visual_pos_masks: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[list[torch.Tensor]] = None,
        output_processor: Callable[..., Tensor] | None = None,
        output_processor_context: Any | None = None,
    ) -> Tensor:
        """Forward function of the GPT Model This function passes the input tensors
        through the embedding layer, and then the decoeder and finally into the post
        processing layer (optional).

         forward pass is overridden to add support for deepstack visual embeddings.

        It either returns the Loss values if labels are given  or the final hidden units

        Args:
            runtime_gather_output (bool): Gather output at runtime. Default None means
                `parallel_output` arg in the constructor will be used.
            padding_mask (Tensor, optional): Boolean padding mask forwarded to
                MoE layers. True positions are excluded from auxiliary-loss,
                z-loss, and expert-bias statistics by the current MCore router.
            output_processor (Callable, optional): Custom postprocess hook forwarded to
                the GPT model postprocessing path.
            output_processor_context (Any, optional): User-defined context forwarded to
                `output_processor`.
        """

        inference_context = deprecate_inference_params(inference_context, inference_params)

        # `_preprocess` can optionally return an extra fused cos/sin buffer (for
        # flash decode). Match the upstream GPTModel handling to avoid unpack
        # errors when six values are returned.
        preproc_output = self._preprocess(
            input_ids=input_ids,
            position_ids=position_ids,
            decoder_input=decoder_input,
            inference_context=inference_context,
            packed_seq_params=packed_seq_params,
            padding_mask=padding_mask,
        )

        decoder_input, rotary_pos_emb, rotary_pos_cos, rotary_pos_sin, sequence_len_offset = (
            preproc_output[:5]
        )
        if len(preproc_output) > 5:
            padding_mask = preproc_output[5]

        # Run decoder.
        hidden_states = self.decoder(
            hidden_states=decoder_input,
            attention_mask=attention_mask,
            inference_context=inference_context,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            # Qwen3 VL blocks do not currently consume fused cos/sin; pass along
            # the standard components only.
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
            padding_mask=padding_mask,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            **(extra_block_kwargs or {}),
        )

        # MTP calls self.embedding directly (bypassing the manual SP scatter that
        # model.py does for the combined VL embeddings). Temporarily wrap the embedding
        # to apply the SP scatter so its output shape matches hidden_states.
        # We write to self.__dict__ directly to bypass nn.Module.__setattr__'s type
        # check, which rejects non-Module values for registered child modules.
        _shadow_embedding = False
        if self.mtp_process and self.config.sequence_parallel:
            _original_embedding = self.embedding

            def _sp_scatter_embedding(input_ids, position_ids):
                out = _original_embedding(input_ids=input_ids, position_ids=position_ids)
                return tensor_parallel.scatter_to_sequence_parallel_region(
                    out, group=self.pg_collection.tp
                )

            _sp_scatter_embedding.word_embeddings = _original_embedding.word_embeddings
            self.__dict__["embedding"] = _sp_scatter_embedding
            _shadow_embedding = True

        postprocess_packed_seq_params = (
            _get_mtp_packed_seq_params(packed_seq_params) if self.mtp_process else packed_seq_params
        )
        result = self._postprocess(
            hidden_states=hidden_states,
            input_ids=input_ids,
            position_ids=position_ids,
            labels=labels,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            mtp_in_postprocess=self.mtp_process,
            loss_mask=loss_mask,
            mtp_input_mask=mtp_input_mask,
            decoder_input=decoder_input,
            attention_mask=attention_mask,
            padding_mask=padding_mask,
            inference_params=inference_params,
            packed_seq_params=postprocess_packed_seq_params,
            sequence_len_offset=sequence_len_offset,
            runtime_gather_output=runtime_gather_output,
            extra_block_kwargs=extra_block_kwargs,
            inference_context=inference_context,
            output_processor=output_processor,
            output_processor_context=output_processor_context,
        )

        if _shadow_embedding:
            del self.__dict__["embedding"]

        return result
