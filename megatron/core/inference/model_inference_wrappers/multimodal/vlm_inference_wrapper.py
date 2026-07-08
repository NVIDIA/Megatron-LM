# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import warnings
from typing import Any, Dict, Optional

import torch

from megatron.core.inference.communication_utils import (
    is_pipeline_first_stage,
    is_pipeline_last_stage,
)
from megatron.core.inference.contexts import StaticInferenceContext
from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import (
    GPTInferenceWrapper,
)


# pylint: disable=line-too-long
class VLMInferenceWrapper(GPTInferenceWrapper):
    """Inference wrapper for VLMs"""

    _recv_only_vision_embeds: bool = False
    _encoder_only: bool = False

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
            self._recv_only_vision_embeds = False  # TODO: Implement new logic for vision embeddings

        # Checks if the current stage only has a vision encoder
        self._encoder_only = False  # TODO: Implement new logic for encoder-only stages

    def prep_inference_input(
        self,
        prompts_tokens: torch.Tensor,
        num_img_embeddings_per_tile: int,
        images: torch.Tensor,
        num_tiles: torch.Tensor,
        decoder_seq_length: int,
    ):
        """Prepares the inference input data.

        Args:
            prompts_tokens (torch.Tensor): A tensor of shape [batch_size, max_seq_len]
            num_img_embeddings_per_tile (int): The number of image embeddings per tile
            images (torch.Tensor): The image embeddings
            num_tiles (torch.Tensor): The number of tiles for each input image
            decoder_seq_length (int): The decoder sequence length
        """
        inference_input = super().prep_inference_input(prompts_tokens)

        total_num_tiles = torch.sum(num_tiles).item()
        num_img_embeddings = num_img_embeddings_per_tile * total_num_tiles

        batch_size, max_sequence_length = prompts_tokens.shape
        self.inference_context = StaticInferenceContext(
            batch_size, max_sequence_length + num_img_embeddings
        )

        inference_input["images"] = images
        inference_input["num_tiles"] = num_tiles
        inference_input["num_img_embeddings"] = num_img_embeddings
        inference_input["decoder_seq_length"] = decoder_seq_length

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
        decoder_seq_length = inference_input["decoder_seq_length"]

        tokens2use = tokens[:, context_start_position:context_end_position]
        positions2use = position_ids[:, context_start_position:context_end_position]

        return {
            "tokens": tokens2use,
            "position_ids": positions2use,
            "images": images,
            "num_tiles": num_tiles,
            "num_img_embeddings": num_img_embeddings,
            "decoder_seq_length": decoder_seq_length,
        }

    # ---- Dynamic inference methods ----

    def expand_image_tokens(self, tokens, num_tiles=None, imgs_sizes=None):
        """Expand image tokens to multiple pad tokens.

        Supports two modes:
        - Static resolution (num_tiles provided): each <image> is replaced with
          tiles_for_that_image * img_embeddings_per_tile padding values.
        - Dynamic resolution (imgs_sizes provided): each <image> is replaced with
          (H/patch_dim * W/patch_dim) / 4 (if pixel_shuffle) padding values per image.

        Args:
            tokens (List[List[int]]): List of token sequences, one per sample.
            num_tiles (torch.Tensor): Number of tiles per image (static resolution).
            imgs_sizes (torch.Tensor): Per-image sizes [N, 2] with [H, W] (dynamic resolution).

        Returns:
            expanded_tokens (List[List[int]]): Tokens with image tokens expanded to -1 pad values.
            mask (List[List[int or None]]): Mask indicating image embedding indices for each
                position, None for non-image positions.
        """
        module = (
            self.model.module.module if hasattr(self.model.module, "module") else self.model.module
        )
        image_token_index = module.image_token_index

        pad_value = -1
        batch_size = len(tokens)
        img_embeddings_per_tile = 0  # set below in the static-resolution branch

        # Compute per-image embedding counts
        if imgs_sizes is not None and getattr(module, '_dynamic_resolution', False):
            # Dynamic resolution: compute per-image embedding count from imgs_sizes
            patch_dim = module._patch_dim
            do_pixel_shuffle = module._pixel_shuffle
            drop_class_token = module._drop_vision_class_token
            class_token_len = module._class_token_len

            per_image_embeddings = []
            for i in range(imgs_sizes.shape[0]):
                h, w = imgs_sizes[i][0].item(), imgs_sizes[i][1].item()
                num_embeddings = (h // patch_dim) * (w // patch_dim)
                if do_pixel_shuffle:
                    num_embeddings //= 4
                per_image_embeddings.append(num_embeddings)
        else:
            # Static resolution: fixed embeddings per tile
            img_embeddings_per_tile = module.img_seq_len
            per_image_embeddings = None  # computed per-image below

        # Count images per sample
        num_images_per_sample = []
        for sample_tokens in tokens:
            num_images_per_sample.append(
                sum(1 for token in sample_tokens if token == image_token_index)
            )

        expanded_tokens_list = []
        mask_list = []

        if per_image_embeddings is not None:
            # Dynamic resolution path
            image_global_idx = 0
            for batch_idx in range(batch_size):
                sample_tokens = tokens[batch_idx]
                expanded_sample = []
                mask_sample = []
                image_embedding_offset = sum(per_image_embeddings[:image_global_idx])

                for token in sample_tokens:
                    if token == image_token_index and image_global_idx < len(per_image_embeddings):
                        tokens_for_image = per_image_embeddings[image_global_idx]
                        expanded_sample.extend([pad_value] * tokens_for_image)

                        start_idx = image_embedding_offset
                        end_idx = start_idx + tokens_for_image
                        mask_sample.extend(list(range(start_idx, end_idx)))

                        image_embedding_offset += tokens_for_image
                        image_global_idx += 1
                    else:
                        expanded_sample.append(token)
                        mask_sample.append(None)

                expanded_tokens_list.append(expanded_sample)
                mask_list.append(mask_sample)
        else:
            # Static resolution path (original logic)
            num_tiles_per_sample = num_tiles.split(num_images_per_sample, dim=0)

            for batch_idx in range(batch_size):
                sample_tokens = tokens[batch_idx]
                sample_num_tiles = (
                    num_tiles_per_sample[batch_idx]
                    if len(num_tiles_per_sample[batch_idx]) > 0
                    else torch.tensor([])
                )

                expanded_sample = []
                mask_sample = []

                image_idx = 0
                image_embedding_offset = (
                    sum(num_tiles_per_sample[i].sum().item() for i in range(batch_idx))
                    * img_embeddings_per_tile
                )

                for token in sample_tokens:
                    if token == image_token_index:
                        if image_idx < len(sample_num_tiles):
                            tiles_for_image = sample_num_tiles[image_idx].item()
                            tokens_for_image = tiles_for_image * img_embeddings_per_tile

                            expanded_sample.extend([pad_value] * tokens_for_image)

                            start_idx = image_embedding_offset
                            end_idx = start_idx + tokens_for_image
                            mask_sample.extend(list(range(start_idx, end_idx)))

                            image_embedding_offset += tokens_for_image
                            image_idx += 1
                        else:
                            expanded_sample.append(token)
                            mask_sample.append(None)
                    else:
                        expanded_sample.append(token)
                        mask_sample.append(None)

                expanded_tokens_list.append(expanded_sample)
                mask_list.append(mask_sample)

        return expanded_tokens_list, mask_list

    def _forward_vision_encoder(
        self, images, num_image_tiles=None, imgs_sizes=None
    ) -> torch.Tensor:
        """Run the vision encoder only, returning image embeddings.

        Temporarily disables the decoder so that the LLaVA forward only runs
        the vision encoder + projection.

        Args:
            images (torch.Tensor): Input images [num_tiles, C, H, W] or [1, total_patches, patch_features].
            num_image_tiles (torch.Tensor): Number of tiles per image (static resolution).
            imgs_sizes (torch.Tensor): Per-image sizes [N, 2] with [H, W] (dynamic resolution).

        Returns:
            torch.Tensor: Image embeddings [img_seq_len, num_tiles, hidden].
        """
        from megatron.core.packed_seq_params import PackedSeqParams

        module = (
            self.model.module.module if hasattr(self.model.module, "module") else self.model.module
        )

        # Build vision_packed_seq_params for dynamic resolution
        vision_packed_seq_params = None
        if imgs_sizes is not None and getattr(module, '_dynamic_resolution', False):
            patch_dim = module._patch_dim
            seq_lens = torch.prod(imgs_sizes // patch_dim, dim=-1)
            cu_seqlens = torch.cat(
                [
                    torch.zeros(1, dtype=torch.int32, device=imgs_sizes.device),
                    torch.cumsum(seq_lens, dim=0).to(torch.int32),
                ]
            )
            max_seqlen = int(seq_lens.max().item())
            vision_packed_seq_params = PackedSeqParams(
                qkv_format="thd",
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_kv=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_kv=max_seqlen,
            )

        old_add_decoder = module.add_decoder
        module.add_decoder = False
        output = self.model(
            images,
            [],
            position_ids=None,
            attention_mask=None,
            inference_context=self.inference_context,
            num_image_tiles=num_image_tiles,
            runtime_gather_output=True,
            imgs_sizes=imgs_sizes,
            vision_packed_seq_params=vision_packed_seq_params,
        )
        module.add_decoder = old_add_decoder

        if isinstance(output, tuple):
            image_embeddings, _ = output
        else:
            image_embeddings = output
        return image_embeddings

    def _forward_dynamic(self, inference_input: Dict[str, Any]) -> torch.Tensor:
        """Forward for dynamic inference with pre-computed image embeddings.

        On PP first stage: embeds text tokens (replacing -1 padding with 0 for
        embedding lookup), gets language embeddings, scatters pre-computed image
        embeddings at mask positions, calls forward_lm_only. On non-first PP stages:
        passes None embeddings.

        Args:
            inference_input (Dict[str, Any]): Must contain 'tokens', 'position_ids',
                'image_token_mask', 'image_embeddings', and optionally 'attention_mask'.

        Returns:
            torch.Tensor: Language model output logits.
        """
        tokens = inference_input["tokens"]
        position_ids = inference_input["position_ids"]
        image_token_mask = inference_input.get("image_token_mask", None)
        attention_mask = inference_input.get("attention_mask", None)
        image_embeddings = inference_input.get("image_embeddings", None)

        module = (
            self.model.module.module if hasattr(self.model.module, "module") else self.model.module
        )

        if is_pipeline_first_stage(self.pp_group) or self._recv_only_vision_embeds:
            # Replace -1 padding with 0 for embedding lookup
            input_ids_text = tokens.clone()
            input_ids_text[input_ids_text == -1] = 0

            # Get language embeddings: [seq_len, b, h_language]
            language_embeddings = module.language_model.embedding(
                input_ids=input_ids_text, position_ids=position_ids
            )

            # Transpose to [b, seq_len, h_language]
            language_embeddings = language_embeddings.transpose(1, 0).contiguous()

            embed_dim = language_embeddings.shape[-1]
            final_embedding = language_embeddings.clone()

            if image_token_mask is not None and image_embeddings is not None:
                image_positions = image_token_mask >= 0

                if image_positions.any():
                    image_indices = image_token_mask[image_positions]

                    # Reshape image embeddings to [total_image_tokens, embed_dim]
                    image_embeddings_flat = image_embeddings.permute(1, 0, 2).reshape(-1, embed_dim)

                    image_embeddings_flat = image_embeddings_flat.to(dtype=final_embedding.dtype)

                    final_embedding[image_positions] = image_embeddings_flat[image_indices]

            # Transpose back to [seq_len, batch, embed_dim]
            final_embedding = final_embedding.transpose(0, 1).contiguous()
        else:
            final_embedding = None

        output = module.forward_lm_only(
            combined_embeddings=final_embedding,
            attention_mask=attention_mask,
            labels=None,
            inference_context=self.inference_context,
            runtime_gather_output=True,
        )

        return output

    # ---- Static inference path ----

    def _forward(self, inference_input: Dict[str, Any]):
        """Runs a forward pass of the model.

        Dispatches to one of three paths:
        1. Dynamic VLM path: 'image_token_mask' key is present.
        2. Static VLM path: 'images' key is present (LLaVA forward).
        3. Pure text (GPT) path: neither key present — delegates to the base
           GPTInferenceWrapper._forward so that text-only models work unmodified.

        Args:
            inference_input(Dict[str, Any]): The input data.

        Returns:
            The model output logits.
        """
        # Dynamic path: image_token_mask is present
        if "image_token_mask" in inference_input:
            return self._forward_dynamic(inference_input)

        # Pure text path: no VLM keys.
        # Cannot delegate to super()._forward() because the abstract wrapper passes
        # (tokens, position_ids, attention_mask) positionally, but LLaVAModel.forward
        # expects (images, input_ids, position_ids, attention_mask).
        if "images" not in inference_input:
            tokens = inference_input["tokens"]
            position_ids = inference_input["position_ids"]
            attention_mask = inference_input["attention_mask"]
            # Pass an empty images tensor (not None) to match what the training
            # data pipeline provides for text-only samples.
            empty_images = torch.tensor([], device=tokens.device).reshape(0, 0, 0)
            output = self.model(
                empty_images,
                tokens,
                position_ids,
                attention_mask=attention_mask,
                inference_context=self.inference_context,
                runtime_gather_output=True,
            )
            if isinstance(output, tuple):
                logits, _ = output
            else:
                logits = output
            return logits

        # VLM path: standard LLaVA forward
        images = inference_input["images"]
        tokens = inference_input["tokens"]
        position_ids = inference_input["position_ids"]
        num_image_tiles = inference_input["num_tiles"]

        output = self.model(
            images,
            tokens,
            position_ids=position_ids,
            attention_mask=None,
            inference_context=self.inference_context,
            num_image_tiles=num_image_tiles,
            runtime_gather_output=True,
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

        # Dynamic path: image_token_mask present, no decoder_seq_length
        if "image_token_mask" in inference_input:
            num_tokens = tokens.size(1)
            recv_buffer_seq_len = num_tokens

            if self._recv_only_vision_embeds:
                pass  # TODO: recv image_embeddings when encoder is on separate stage

            if self._encoder_only:
                pass  # TODO: send image_embeddings down pipeline
            else:
                output = super().run_one_forward_step(
                    inference_input, recv_buffer_seq_len=recv_buffer_seq_len
                )
            logits = output
            return logits

        # Pure text path: no VLM keys, use base GPT forward
        if "images" not in inference_input:
            return super().run_one_forward_step(inference_input)

        # Static VLM path
        num_image_tokens = (tokens == self.model.module.image_token_index).sum().item()
        num_img_embeddings = inference_input["num_img_embeddings"]
        decoder_seq_length = inference_input["decoder_seq_length"]
        num_tokens = tokens.size(1)
        recv_buffer_seq_len = None
        if num_image_tokens > 0:
            # When there are image tokens and this stage only receives vision embeddings,
            # adjust the recv buffer seq length to match the image embeddings sequence length.
            # If there are image tokens and this stage receives full embeddings, make sure we
            # compensate for expansion of image tokens.
            # Note that this will set a recv_buffer_seq_len for the encoder stage,
            # this length is irrelevant since that recv buffer is never allocated.
            if self._recv_only_vision_embeds:
                recv_buffer_seq_len = num_img_embeddings
            else:
                recv_buffer_seq_len = min(
                    num_img_embeddings + num_tokens - num_image_tokens, decoder_seq_length
                )
        elif self._recv_only_vision_embeds:
            # If this stage only receives vision embeddings and there are no image tokens
            # we won't run the encoder and therefore shouldn't try to recv.
            recv_buffer_seq_len = 0

        # If the pipeline stage only has a vision encoder, then it only needs to
        # run when there are image tokens
        if not (self._encoder_only and num_image_tokens == 0):
            output = super().run_one_forward_step(
                inference_input, recv_buffer_seq_len=recv_buffer_seq_len
            )
        else:
            output = None
        logits = output

        # On the first inference iteration, we compute image tokens.
        # On every PP stage(although inference params should only matter for decoder),
        # update the sequence length offset by the number of image tokens.
        if num_tokens > 1 and num_image_tokens > 0:
            if "image_tokens_count" not in self.inference_context.key_value_memory_dict:
                self.inference_context.key_value_memory_dict["image_tokens_count"] = (
                    num_img_embeddings
                )

            if num_img_embeddings + num_tokens - num_image_tokens > decoder_seq_length:
                self.inference_context.sequence_len_offset += decoder_seq_length - num_tokens
            else:
                self.inference_context.sequence_len_offset += (
                    self.inference_context.key_value_memory_dict["image_tokens_count"]
                    - num_image_tokens
                )

        return logits
