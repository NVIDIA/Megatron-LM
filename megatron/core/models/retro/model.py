# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

"""Retro Model."""

from torch import Tensor

from megatron.core import InferenceParams
from megatron.core.models.gpt import GPTModel


class RetroModel(GPTModel):

    """Retro Model.

    A Retro model mostly re-uses the GPTModel interface, with the only difference
    being the embedding of the 'context' this is used by Retro for processing
    neighbor tokens. This embedded context is then forwarded to the Transformer
    Block.
    """

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        context_input_ids: Tensor = None,
        context_position_ids: Tensor = None,
        context_mask: Tensor = None,
        decoder_input: Tensor = None,
        labels: Tensor = None,
        inference_params: InferenceParams = None,
    ):

        # Context embedding (e.g., for Retro neighbor tokens).
        if context_input_ids is not None:
            context = self.embedding(context_input_ids, context_position_ids)
        else:
            context = None

        # Call GPTModel.forward, and pass in embedded context.
        return super().forward(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            decoder_input=decoder_input,
            labels=labels,
            inference_params=inference_params,
            extra_block_kwargs={
                "context" : context,
                "context_mask" : context_mask,
            },
        )
