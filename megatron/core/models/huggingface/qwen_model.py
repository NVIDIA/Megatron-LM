# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

import torch
from transformers.models.qwen2 import Qwen2ForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

from megatron.core.models.huggingface import HuggingFaceModule


class QwenHuggingFaceModel(HuggingFaceModule):
    """
    Wrapper for Qwen LM HuggingFace models.
    """

    # Currently applies to FSDP2 only, not the custom FSDP implementation.
    _fsdp_modules = [Qwen2DecoderLayer]

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen2ForCausalLM.from_pretrained(config.language_model_type.split("hf://")[1])

    def forward(self, *args, **kwargs):
        """Qwen forward."""
        labels = kwargs["labels"]
        combined_embeddings = kwargs["decoder_input"].permute(1, 0, 2)

        x = self.model(
            position_ids=None,  # uses arange
            attention_mask=kwargs['attention_mask'],  # Typically None -> causal.
            inputs_embeds=combined_embeddings,
        )
        logits = x["logits"]

        if labels is not None:
            loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
            x = loss_fn(logits.permute(0, 2, 1), labels)

        return x

    def embedding(self, input_ids, position_ids=None):
        """Function to run process tokens with input embeddings"""
        return self.model.get_input_embeddings()(input_ids).transpose(1, 0).contiguous()
