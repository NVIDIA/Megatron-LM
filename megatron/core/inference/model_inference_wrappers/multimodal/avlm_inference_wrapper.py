# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import warnings
from typing import Any, Dict, Optional

import torch

from megatron.core import parallel_state
from megatron.core.inference.communication_utils import (
    is_pipeline_first_stage,
    is_pipeline_last_stage,
)
from megatron.core.inference.contexts import StaticInferenceContext
from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import (
    GPTInferenceWrapper,
)
from megatron.core.packed_seq_params import PackedSeqParams

# pylint: disable=line-too-long
class AVLMInferenceWrapper(GPTInferenceWrapper):
    """Inference wrapper for AVLMs"""

    def prep_model_for_inference(self, prompts_tokens: Optional[torch.Tensor] = None):
        """A utility function for preparing model for inference

        The function gets called once before the auto regressive inference loop.
        It puts the model in eval mode.

        Args:
            prompts_tokens (torch.Tensor): Deprecated, will be removed in `megatron-core` 0.13
        """
        if prompts_tokens is not None:
            warnings.warn(
                "Passing `prompts_tokens` is deprecated and this argument will be ignored."
                "This parameter will be removed in `megatron-core` 0.13."
            )

        super().prep_model_for_inference()

        # For TP only model both is_pp_first_stage and _is_pp_last_stage returns True
        # set ignore_virtual=True since vpp is not used in inference
        self.model_is_pipeline_parallel = not (
            is_pipeline_first_stage(self.pp_group) and is_pipeline_last_stage(self.pp_group)
        )

        self._recv_only_vision_embeds = False
        pp_rank = self.pp_group.rank()
        # Checks if the previous stage only has a vision encoder, and that the current stage
        # has part of the LM decoder. In this case, the current stage should only receive
        # vision embeddings.
        if pp_rank > 0:
            self._recv_only_vision_embeds = (
                parallel_state.is_inside_encoder(pp_rank - 1)
                and (not parallel_state.is_inside_decoder(pp_rank - 1))
                and parallel_state.is_inside_decoder()
            )

        # Checks if the current stage only has a vision encoder
        self._encoder_only = (
            parallel_state.is_inside_encoder() and not parallel_state.is_inside_decoder()
        )

    def maybe_cast_to_tensor(self, data: Any) -> torch.Tensor:
        if data is None:
            return None
        if isinstance(data, torch.Tensor):
            return data.cuda()
        
        if isinstance(data, (int, float)):
            data = [data]

        return torch.tensor(data).cuda()

    def prep_inference_input(
        self,
        prompts_tokens: torch.Tensor,
        num_img_embeddings: int,
        images: torch.Tensor,
        num_tiles: torch.Tensor,
        imgs_sizes: torch.Tensor,
        decoder_seq_length: int,
        vision_packed_seq_params: Optional[PackedSeqParams] = None,
        sound_clips: Optional[torch.Tensor] = None,
        sound_length: Optional[torch.Tensor] = None,
        num_sound_embeddings: int = 0,
    ):
        """Prepares the inference input data.

        Args:
            prompts_tokens (torch.Tensor): A tensor of shape [batch_size, max_seq_len]
            num_img_embeddings (int): The number of image embeddings
            images (torch.Tensor): The image embeddings
            num_tiles (torch.Tensor): The number of tiles for each input image
            imgs_sizes (torch.Tensor): The image sizes
            decoder_seq_length (int): The decoder sequence length
            vision_packed_seq_params (Optional[PackedSeqParams]): Vision packed sequence parameters
            sound_clips (Optional[torch.Tensor]): The sound clips
            sound_length (Optional[torch.Tensor]): The sound length
            num_sound_embeddings (int): The number of sound length
        """
        inference_input = super().prep_inference_input(prompts_tokens)

        inference_input["images"] = self.maybe_cast_to_tensor(images)
        inference_input["num_tiles"] = self.maybe_cast_to_tensor(num_tiles)
        inference_input["num_img_embeddings"] = num_img_embeddings
        inference_input["imgs_sizes"] = self.maybe_cast_to_tensor(imgs_sizes)
        inference_input["vision_packed_seq_params"] = vision_packed_seq_params
        inference_input["decoder_seq_length"] = decoder_seq_length
        inference_input["sound_clips"] = self.maybe_cast_to_tensor(sound_clips)
        inference_input["sound_length"] = self.maybe_cast_to_tensor(sound_length)
        inference_input["num_sound_embeddings"] = num_sound_embeddings

        return inference_input

    def get_batch_for_context_window(
        self,
        inference_input: Dict[str, Any],
        context_start_position: int,
        context_end_position: int,
    ) -> Dict[str, Any]:
        """Returns the inference data given context window

        This function gets called iteratively in a loop . Given the start and end context positions , it extracts the appropriate data.

        Args:
            inference_input (Dict[str, Any]): The inference input for the batch.
            context_start_position (int): Start of the context window. During the first inference step it is mostly 0
            context_end_position (int): End of the context window. During the last inference step it will mostly be the max generated sequence length.

        Returns:
            Dict[str, Any]: A dict of inputs that will be used by your model in the forward step
        """
        tokens = inference_input["tokens"]
        position_ids = inference_input["position_ids"]
        images = inference_input["images"]
        num_tiles = inference_input["num_tiles"]
        num_img_embeddings = inference_input["num_img_embeddings"]
        imgs_sizes = inference_input["imgs_sizes"]
        vision_packed_seq_params = inference_input["vision_packed_seq_params"]
        decoder_seq_length = inference_input["decoder_seq_length"]
        sound_clips = inference_input["sound_clips"]
        sound_length = inference_input["sound_length"]
        num_sound_embeddings = inference_input["num_sound_embeddings"]
        tokens2use = tokens[:, context_start_position:context_end_position]
        
        # Position IDs can be None - the model will handle it internally
        if position_ids is not None:
            positions2use = position_ids[:, context_start_position:context_end_position]
        else:
            positions2use = None

        return {
            "tokens": tokens2use,
            "position_ids": positions2use,
            "images": images,
            "num_tiles": num_tiles,
            "num_img_embeddings": num_img_embeddings,
            "imgs_sizes": imgs_sizes,
            "vision_packed_seq_params": vision_packed_seq_params,
            "decoder_seq_length": decoder_seq_length,
            "sound_clips": sound_clips,
            "sound_length": sound_length,
            "num_sound_embeddings": num_sound_embeddings,
        }

    def _forward(self, inference_input: Dict[str, Any]):
        """Runs a forward pass of the model.

        Args:
            inference_input(Dict[str, Any]): The input data.

        Returns:
            The model output logits.
        """
        images = inference_input["images"]
        tokens = inference_input["tokens"]
        position_ids = inference_input["position_ids"]
        num_image_tiles = inference_input["num_tiles"]
        imgs_sizes = inference_input["imgs_sizes"]
        vision_packed_seq_params = inference_input["vision_packed_seq_params"]
        sound_clips = inference_input["sound_clips"]
        sound_length = inference_input["sound_length"]

        # Normalize None to empty tensors so the model always receives tensors
        if images is None:
            images = torch.tensor([], dtype=torch.bfloat16, device=tokens.device)
        if num_image_tiles is None:
            num_image_tiles = torch.tensor([], dtype=torch.int, device=tokens.device)

        output = self.model(
            images,
            tokens,
            position_ids=position_ids,
            attention_mask=None,
            inference_context=self.inference_context,
            num_image_tiles=num_image_tiles,
            imgs_sizes=imgs_sizes,
            vision_packed_seq_params=vision_packed_seq_params,
            runtime_gather_output=True,
            sound_clips=sound_clips,
            sound_length=sound_length,
        )

        if isinstance(output, tuple):
            logits, _ = output
        else:
            logits = output
        return logits

    def run_one_forward_step(self, inference_input: Dict[str, Any]) -> torch.Tensor:
        """The forward pass of the model for inference

        Args:
            inference_input (Dict[str, Any]): A dict containing the inputs for the VLM model

        Returns:
            torch.Tensor: The output logits of shape [batch_size, seq_len, padded_vocab_size].
            The logits are returned only in the last pipeline stage for PP models.
        """
        tokens = inference_input["tokens"]
        num_image_tokens = (tokens == self.model.module.image_token_index).sum().item()
        num_sound_tokens = (tokens == self.model.module.sound_token_index).sum().item()
        num_img_embeddings = inference_input["num_img_embeddings"]
        if "image_tokens_count" in self.inference_context.key_value_memory_dict:
            num_img_embeddings = self.inference_context.key_value_memory_dict["image_tokens_count"]
        num_sound_embeddings = inference_input["num_sound_embeddings"]
        if "sound_tokens_count" in self.inference_context.key_value_memory_dict:
            num_sound_embeddings = self.inference_context.key_value_memory_dict["sound_tokens_count"]
        decoder_seq_length = inference_input["decoder_seq_length"]
        num_tokens = tokens.size(1)
        recv_buffer_seq_length = None
        if num_image_tokens > 0:
            # When there are image tokens and this stage only receives vision embeddings, adjust the recv buffer seq length to match the image embeddings sequence length.
            # If there are image tokens and this stage receives full embeddings, make sure we compensate for expansion of image tokens.
            # Note that this will set a recv_buffer_seq_length for the encoder stage, this length is irrelevant since that recv buffer is never allocated.
            if self._recv_only_vision_embeds:
                recv_buffer_seq_length = num_img_embeddings
            else:
                recv_buffer_seq_length = min(num_img_embeddings + num_tokens - num_image_tokens, decoder_seq_length)
        elif self._recv_only_vision_embeds:
            # If this stage only receives vision embeddings and there are no image tokens we won't run the encoder and therefore shouldn't try to recv.
            recv_buffer_seq_length = 0

        # If the pipeline stage only has a vision/sound encoder, then it only needs to
        # run when there are image/sound tokens
        if not (self._encoder_only and num_image_tokens == 0):
            output = super().run_one_forward_step(
                inference_input, recv_buffer_seq_len=recv_buffer_seq_length
            )
        else:
            output = None
        logits = output

        # On the first inference iteration, we compute image/sound tokens.
        # On every PP stage(although inference params should only matter for decoder),
        # update the sequence length offset by the number of image/sound tokens.
        if num_tokens > 1 and (num_image_tokens > 0 or num_sound_tokens > 0):
            if "image_tokens_count" not in self.inference_context.key_value_memory_dict:
                self.inference_context.key_value_memory_dict["image_tokens_count"] = num_img_embeddings
            if "sound_tokens_count" not in self.inference_context.key_value_memory_dict:
                self.inference_context.key_value_memory_dict["sound_tokens_count"] = num_sound_embeddings

            num_extra_image_tokens = max(0, self.inference_context.key_value_memory_dict["image_tokens_count"] - num_image_tokens)
            num_extra_sound_tokens = max(0, self.inference_context.key_value_memory_dict["sound_tokens_count"] - num_sound_tokens)
            if num_tokens + num_extra_image_tokens + num_extra_sound_tokens > decoder_seq_length:
                self.inference_context.sequence_len_offset += decoder_seq_length - num_tokens
            else:
                self.inference_context.sequence_len_offset += num_extra_image_tokens + num_extra_sound_tokens

        return logits
