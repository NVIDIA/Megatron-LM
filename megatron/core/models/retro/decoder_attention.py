# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Retro's cross attention modules for the decoder block."""

from functools import partial
from typing import Callable, Optional

import numpy as np
import torch
from torch import Tensor

from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.models.retro.base_attention import BaseRetroCrossAttention
from megatron.core.models.retro.config import RetroConfig
from megatron.core.models.retro.utils import get_all_true_mask
from megatron.core.transformer import ModuleSpec
from megatron.core.transformer.attention import CrossAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.utils import deprecate_inference_params


class RetroDecoderCrossAttention(BaseRetroCrossAttention):
    """Retro decoder's chunked cross attention operator.

    See this paper for more details: https://arxiv.org/abs/2112.04426.
    Neighboring chunks retrieved from the chunk database are used here for
    chunked-cross attention.

    ** Note about 'encoder_block_spec' **

    Retro is an encoder-decoder model that uses its encoder for encoding
    neighboring chunks that are retrieved from a chunk database. These
    encoded neighbors are then used in the decoder stack for performing
    chunked-cross attention (see paper link above).

    In contrast to the T5 model, the encoder and decoder are computationally
    intertwined, since the input to the encoder is the output of the self-
    attention of the first decoder layer. As such, the encoder block itself
    is instantiated within the first Retro decoder layer, in order to receive
    the self-attention's output. (Note, that only the first decoder layer
    instantiates an encoder block, and the remaining decoder layers use the
    encoder output from the first decoder layer.)

    Args:
        config (RetroConfig): Retro config.
        submodules (CrossAttentionSubmodules): Cross attention submodules.
        layer_number (int): Layer number within transformer block.
        attn_mask_type (AttnMaskType): Mask type ('causal' or 'padding').
        encoder_block_spec (ModuleSpec): The first Retro decoder layer is
            provided with a transformer block spec to construct the neighbor encoder.
    """

    def __init__(
        self,
        config: RetroConfig,
        submodules: CrossAttentionSubmodules,
        layer_number: int = 1,
        attn_mask_type: AttnMaskType = AttnMaskType.padding,
        encoder_block_spec: ModuleSpec = None,
    ):
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
        )

        if encoder_block_spec:
            self.encoder = TransformerBlock(
                config=config, spec=encoder_block_spec, pre_process=True, post_process=False
            )
            # self._encoder_key = 'encoder' # ... necessary?
        else:
            self.encoder = None

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        key_value_states: Tensor = None,
        inference_context: BaseInferenceContext = None,
        # rotary_pos_emb: Tensor = None, # ... unsupported for retro.
        *,
        inference_params: Optional[BaseInferenceContext] = None,
    ) -> dict:
        """Cross attention for Retro decoder.

        Notation:
            ns : Sequence length.
            bs : Batch size.
            d  : Hidden size.
            l  : Number of chunks per sample (i.e., seq_length/chunk_length).
            m  : Number of tokens per chunk.
            k  : Number of neighbors.
            r  : Number of retrieved tokens (neighbors + continuation).

        Args:
            hidden_states (Tensor): Transformer layer hidden states.
            attention_mask (Tensor): Attention mask.
            key_value_states (Tensor): Neighbor embeddings if first decoder layer,
                else encoder output.
            inference_context (BaseInferenceContext): Inference context.

        Returns:
            A dict consisting of the attention output and context, along with
                other scalars necessary for performing the downstream bias-dropout-add.
        """

        # hidden_states: [ ns, bs, d ]
        # key_value_states: [ r, k*bs*l, d ]

        inference_context = deprecate_inference_params(inference_context, inference_params)

        ns, bs, d = hidden_states.shape
        l = int(np.ceil(ns / self.retro_chunk_length))

        # Retrieve neighbors.
        if self.encoder:

            # Sequence length remainder.
            first_ns = ns % self.retro_chunk_length

            # Case 1: Sequence length not divisible by chunk length.
            if first_ns > 0:

                # Split sequence into first partial chunk & remaining chunks.
                first_chunk, rest_chunk = hidden_states[:first_ns], hidden_states[first_ns:]

                # Pad partial chunk with zeros.
                first_chunk = torch.nn.functional.pad(
                    first_chunk, (0, 0, 0, 0, 0, self.retro_chunk_length - first_ns), 'constant', 0
                )

                # Concatenate padded chunk with remaining chunks.
                chunked_output = torch.cat((first_chunk, rest_chunk), dim=0)  # [ l*m, bs, d ]

            # Case 2: Sequence length is divisible by chunk length.
            else:
                chunked_output = hidden_states  # [ l*m, bs, d ]

            # Chunk & permute hidden states.
            # - hidden_states:  [ l*m, bs, d ]
            # - chunked_output: [ m, bs*l, d ]
            chunked_output = (
                chunked_output.reshape(l, self.retro_chunk_length, bs, d)
                .permute(1, 2, 0, 3)
                .reshape(self.retro_chunk_length, bs * l, d)
                .contiguous()
            )

            # flash attn: [ b, h, sq, sk ]
            # fused attn: [ b, 1, 1, sq ]
            chunked_output_mask = get_all_true_mask(
                size=(1, 1, chunked_output.shape[0], key_value_states.shape[0]),
                device=chunked_output.device,
            )

            # Encode neighbors. (Note: 'key_value_states' re-assigned here.)
            key_value_states = self.encoder(
                hidden_states=key_value_states,
                attention_mask=attention_mask,
                context=chunked_output,
                context_mask=chunked_output_mask,
                inference_context=inference_context,
            )  # [ r, k*bs*l, d ]
            key_value_states = key_value_states.reshape(
                self.retro_retrieved_length * self.retro_num_neighbors, bs * l, d
            )  # [ r*k, bs*l, d ]

        # Attend starting at last token of first chunk.
        pad = (ns - 1) % self.retro_chunk_length
        attending_chunks = hidden_states[pad:]

        # Pad attending tokens to sequence length.
        padded_chunks = torch.nn.functional.pad(
            attending_chunks, (0, 0, 0, 0, 0, self.retro_chunk_length - 1), 'constant', 0
        )

        # Permute attending chunks.
        # - padded_chunks:         [ l*m, bs, d ]
        # - padded_chunked_output: [ m, bs*l, d ] (matches 'chunked_output' above)
        padded_chunked_output = padded_chunks.reshape(l, self.retro_chunk_length, bs, d).permute(
            1, 2, 0, 3
        )
        padded_chunked_output = padded_chunked_output.reshape(
            self.retro_chunk_length, bs * l, d
        ).contiguous()

        # flash attn: [ b, h, sq, sk ]
        # fused attn: [ b, 1, 1, sq ]
        padded_chunked_output_mask = get_all_true_mask(
            size=(1, 1, padded_chunked_output.shape[0], key_value_states.shape[0]),
            device=padded_chunked_output.device,
        )

        # Attend to encoded neighbors.
        attention_output, attention_bias = self.attn(
            hidden_states=padded_chunked_output,
            attention_mask=padded_chunked_output_mask,
            key_value_states=key_value_states,
        )

        # Return dimensions for bias-dropout step.
        return {
            "ns": ns,
            "bs": bs,
            "d": d,
            "l": l,
            "pad": pad,
            "attention_output": attention_output,  # [ m, bs*l, d ]
            "attention_bias": attention_bias,  # [ d ]
            "context": key_value_states,  # [ r*k, bs*l, d ]
        }


class RetroDecoderBiasDropoutAdd(MegatronModule):
    """Retro decoder's bias-dropout-add operator.

    This operator takes care of reshaping and permuting the output from the
    chunk dimension to the sequence dimension.

    Args:
        config (RetroConfig): Retro config.
    """

    def __init__(self, config: RetroConfig):
        super().__init__(config=config)
        self.retro_chunk_length = config.retro_chunk_length

    @classmethod
    def _forward(
        cls,
        x_with_bias: dict,
        residual: Tensor,
        prob: float,
        retro_chunk_length: int,
        bias_dropout_add: Callable,
    ) -> Tensor:
        """Per-chunk bias-dropout-add.

        Args:
            x_with_bias (dict): Attention output and bias, along with other Retro
                relevant parameters.
            residual (Tensor): Transformer layer residual.
            prob (float): Dropout probability.
            retro_chunk_length (int): Retro chunk length (e.g., 64).
            bias_dropout_add (Callable): Bias-dropout-add function.

        Returns:
            Output of bias-dropout-add.
        """

        # Extract input dict.
        ns = x_with_bias["ns"]
        bs = x_with_bias["bs"]
        d = x_with_bias["d"]
        l = x_with_bias["l"]
        pad = x_with_bias["pad"]
        attention_output = x_with_bias["attention_output"]  # [ m, bs*l, d ]
        attention_bias = x_with_bias["attention_bias"]  # [ d ]

        # Re-enable torch grad to enable fused optimization.
        with torch.enable_grad():

            # Bias-dropout-add.
            x = bias_dropout_add(
                (
                    attention_output,
                    None if attention_bias is None else attention_bias.expand_as(attention_output),
                ),
                torch.zeros_like(attention_output),
                prob,
            )

            # Permute chunks back to sequence dimension.
            # 1. [ m, bs*l, d ]
            # 2. [ m, bs, l, d ]
            # 3. [ l, m, bs, d ]
            # 4. [ m*l, bs, d ] == [ ns, bs, d ]
            x = (
                x.reshape(retro_chunk_length, bs, l, d)
                .permute(2, 0, 1, 3)
                .reshape(retro_chunk_length * l, bs, d)
            )

            # Prepend zeros for non-attending tokens.
            x = torch.nn.functional.pad(x, (0, 0, 0, 0, pad, 0), 'constant', 0)[
                :ns
            ]  # [ ns, bs, d ]

            # Add residual. [ ns, bs, d ]
            x = x + residual

        # Output. [ ns, bs, d ]
        return x

    def forward(self, training: bool, fused: bool) -> partial:
        """Retro decoder bias-dropout-add.

        Args:
            training (bool): If training, then apply dropout.
            fused (bool): Fuse bias-dropout-add.

        Returns:
            The partial function for performing bias-dropout-add.
        """
        return partial(
            self._forward,
            retro_chunk_length=self.retro_chunk_length,
            bias_dropout_add=get_bias_dropout_add(training, fused),
        )
