# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

from transformers.models.qwen2 import Qwen2ForCausalLM

from megatron.core.models.huggingface import HuggingFaceModule


class QwenHuggingFaceModel(HuggingFaceModule):
    """
    Wrapper for Qwen LM HuggingFace models
    """

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen2ForCausalLM.from_pretrained(config.huggingface_model_name_or_path)

    def forward(self, *args, **kwargs):
        """Forward function"""
        combined_embeddings = kwargs['decoder_input'].permute(1, 0, 2)
        x = self.model(
            position_ids=None,  # TODO: I guess we're just assuming no custom pos ids
            attention_mask=kwargs['attention_mask'],
            inputs_embeds=combined_embeddings,
            labels=kwargs['labels'],
        )

        if kwargs['labels'] is not None:
            x = x["loss"]
        else:
            x = x["logits"]

        return x

    def embedding(self, input_ids, position_ids=None):
        """Function to run process tokens with input embeddings"""
        return self.model.get_input_embeddings()(input_ids).transpose(1, 0).contiguous()
