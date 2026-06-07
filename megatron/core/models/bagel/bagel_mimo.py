# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import logging
import math
import warnings
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist

from megatron.core.models.mimo.config import MimoModelConfig
from megatron.core.models.mimo.model.base import MimoModel
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.models.bagel.mcore_bagel_llm import BagelMCoreModel
from megatron.core.models.bagel.hf_bagel_llm import BagelLLMHuggingFaceModel
from megatron.core.models.bagel.mot_packed_seq_params import MoTPackedSeqParams

logger = logging.getLogger(__name__)


def gather_pad_to_length(src: torch.Tensor, idx: torch.Tensor, length: int) -> torch.Tensor:
    """Gather rows ``src[idx]`` and zero-pad to ``length`` along dim 0."""
    t = src[idx]
    n = length - len(t)
    return torch.cat([t, torch.zeros(n, *t.shape[1:], dtype=t.dtype, device=t.device)]) if n > 0 else t


class _AllGatherImageEmbeddings(torch.autograd.Function):
    """All-gather variable-length ViT embeddings across CP ranks.

    Forward: concatenates each rank's local embeddings [V_r, H] into the full
    tensor [V, H] = cat([V_0, V_1, ..., V_{cp-1}], dim=0).

    Backward: all ranks already hold the full grad_output [V, H] (it flows
    back through the autograd graph to every rank identically), so rank r
    recovers its local gradient by a purely local slice — no communication.
    """

    @staticmethod
    def forward(ctx, embeddings: torch.Tensor, vit_tokens_encoded_per_cp: List[int], group) -> torch.Tensor:

        all_sizes = vit_tokens_encoded_per_cp
        print(f"all_sizes: {all_sizes}, group rank: {dist.get_rank(group)}, rank: {dist.get_rank()}")

        # Allocate output and all_gather into pre-split views.
        total = sum(all_sizes)
        output = torch.empty(
            total, embeddings.shape[1],
            dtype=embeddings.dtype, device=embeddings.device,
        )
        output_list = list(torch.split(output, all_sizes, dim=0))
        dist.all_gather(output_list, embeddings.contiguous(), group=group)

        ctx.group = group
        ctx.all_sizes = all_sizes
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        all_sizes = ctx.all_sizes
        cp_rank = dist.get_rank(ctx.group)

        # No communication required: every rank already holds the full
        # grad_output.  Rank r just extracts its own slice.
        offset = sum(all_sizes[:cp_rank])
        return grad_output[offset : offset + all_sizes[cp_rank]].contiguous(), None, None


def all_gather_image_embeddings(embeddings: torch.Tensor, vit_tokens_encoded_per_cp: List[int], group) -> torch.Tensor:
    """All-gather ViT embeddings from all CP ranks into [V, H].

    Each CP rank runs SigLIP on its assigned images and holds
    local_vit_emb [V_local, H].  This function gathers all shards into the
    full [V, H] tensor (ranked in CP-rank order, which matches the original
    image order set by shard_data_for_cp).

    Args:
        embeddings: Local ViT embeddings [V_local, H] on this CP rank.
        group: The context-parallel process group.

    Returns:
        Full ViT embeddings [V, H] available on every rank.
    """
    if dist.get_world_size(group) == 1:
        return embeddings
    return _AllGatherImageEmbeddings.apply(embeddings, vit_tokens_encoded_per_cp, group)


def align_bagel_embeddings(
    language_model,
    input_ids: torch.Tensor,
    vision_embeddings: Optional[torch.Tensor],
    visual_latents: Optional[torch.Tensor],
    sequence_length: int,
    packed_seq_params: Optional[MoTPackedSeqParams],
) -> dict:
    """Assemble compact decoder_input and pre-shard all CP-sensitive tensors.

    This is the single place where the full [S, H] intermediate lives.
    After scattering, the sequence is immediately compacted so
    BagelMCoreModel never sees the [S, H] allocation.

    Called by BagelMimoModel.align_embeddings_by_token_positions() and directly
    by unit tests that exercise BagelMCoreModel in isolation.

    Steps:
        1. Get text embeddings via language_model.embedding().
        2. Scatter text / vision / gen embeddings into a temporary [S, H] buffer.
        3. Compact [S, H] → [Lund+Lgen, 1, H] — shards embeddings to this CP rank.
        4. Pre-shard labels / loss_mask → [actual_lund].
        5. Build compact packed_position_ids [Lund+Lgen] — shards position IDs.

    Args:
        language_model: Any model with a .embedding(input_ids, position_ids) method.
        input_ids: Text token IDs [1, T].
        vision_embeddings: Projected ViT embeddings [V, H] or None.
        visual_latents: VAE token embeddings [G, H] or None.
        sequence_length: Total length S of the full packed sequence.
        packed_text_indexes: Global positions of text tokens [T].
        packed_vit_token_indexes: Global positions of ViT tokens [V] or None.
        packed_vae_token_indexes: Global positions of VAE tokens [G] or None.
        packed_seq_params: CP-aware MoTPackedSeqParams with local shard info.
        packed_position_ids: Full-sequence position IDs [S] for RoPE, or None.
        labels_full: Full-sequence label tensor [S] (0 at non-CE positions), or None.
        loss_mask_full: Full-sequence loss mask [S] (0.0 at non-CE positions), or None.

    Returns:
        dict:
            decoder_input        [Lund+Lgen, 1, H]  — compact, this-rank embeddings
            labels               [actual_lund] or None
            loss_mask            [actual_lund] or None
            packed_position_ids  [Lund+Lgen] or None
    """
    device = input_ids.device

    # ── Step 1: text embeddings ──────────────────────────────────────────────
    raw = language_model.embedding(input_ids=input_ids, position_ids=None)
    if raw.dim() == 3:
        if raw.shape[1] == 1:
            text_emb = raw.squeeze(1)
        elif raw.shape[0] == 1:
            text_emb = raw.squeeze(0)
        else:
            text_emb = raw[0]
    else:
        text_emb = raw   # [T, H]

    H    = text_emb.shape[-1]
    dtype = text_emb.dtype

    # ── Step 2: scatter into full [S, H] ────────────────────────────────────
    packed_seq = torch.zeros(sequence_length, H, dtype=dtype, device=device)
    packed_seq[packed_seq_params.packed_text_indexes.to(device)] = text_emb
    if vision_embeddings is not None and packed_seq_params.packed_vit_token_indexes is not None:
        packed_seq[packed_seq_params.packed_vit_token_indexes.to(device)] = vision_embeddings

    if visual_latents is not None and packed_seq_params.packed_vae_token_indexes is not None:
        packed_seq[packed_seq_params.packed_vae_token_indexes.to(device)] = visual_latents

    # ── Steps 3-5: MoT compaction + CP sharding ─────────────────────────────
    und_idx = packed_seq_params.local_und_token_indexes   # [actual_lund]
    gen_idx = packed_seq_params.local_gen_token_indexes   # [actual_lgen]
    Lund    = packed_seq_params.padded_und_seqlen
    Lgen    = packed_seq_params.padded_gen_seqlen

    # Step 3: compact [S, H] → [Lund+Lgen, 1, H]
    decoder_input = torch.cat(
        [gather_pad_to_length(packed_seq, und_idx, Lund),
         gather_pad_to_length(packed_seq, gen_idx, Lgen)], dim=0
    ).unsqueeze(1)  # [Lund+Lgen, 1, H]


    return dict(
        decoder_input=decoder_input,
    )


class BagelMimoModel(MimoModel):

    def __init__(
        self,
        mimo_config: MimoModelConfig,
        *,
        pg_collection: Optional[ProcessGroupCollection] = None,
        pre_process: bool = True,
        post_process: bool = True,
        vp_stage: Optional[int] = None,
    ) -> None:
        super().__init__(mimo_config)
        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        # LanguageModule only requires ``hasattr(pg_collection, 'embd')`` and
        # tolerates ``embd is None`` at middle PP stages
        # (language_module.py:67 - ``if self.embd_group is None: return False``).
        # An earlier Phase-A version of this code asserted ``embd is not None``,
        # which crashed at PP>=3 because ``mpu.get_embedding_group()`` only
        # returns a non-None group on first/last PP stages.
        self.pg_collection = pg_collection
        self.cp_group = pg_collection.cp
        self.cp_size = dist.get_world_size(self.cp_group)
        self.cp_rank = dist.get_rank(self.cp_group)

        # PP-stage flags. At PP=1 both are True and behaviour is unchanged.
        # set_input_tensor is inherited from MimoModel (delegates to
        # self.language_model.set_input_tensor).
        self.pre_process = pre_process
        self.post_process = post_process
        self.vp_stage = vp_stage

        # Megatron's finalize_model_grads.py:_get_shared_word_embedding_weight
        # walks each model_module and reads .share_embeddings_and_output_weights
        # to decide whether to all-reduce embedding gradients across PP. Forward
        # the inner GPTModel's value so the embedding-sync logic gets the right
        # answer regardless of which submodule is queried.
        if hasattr(self.language_model, 'share_embeddings_and_output_weights'):
            self.share_embeddings_and_output_weights = (
                self.language_model.share_embeddings_and_output_weights
            )
        else:
            self.share_embeddings_and_output_weights = False

    def _get_language_model_embeddings(self, input_ids: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """Get embeddings from the language model.

        Handles both HuggingFace (has embedding method) and Megatron Core (has get_word_embeddings method).

        Args:
            input_ids: [batch_size, seq_len]
            position_ids: [batch_size, seq_len]

        Returns:
            embeddings: [seq_len, batch_size, hidden_dim]
        """
        # Check for HuggingFace style (callable embedding method)
        if hasattr(self.language_model, 'embedding') and callable(self.language_model.embedding):
            return self.language_model.embedding(input_ids=input_ids, position_ids=position_ids)

        # Check for Megatron Core BagelMCoreModel style
        if hasattr(self.language_model, 'get_word_embeddings'):
            return self.language_model.get_word_embeddings(input_ids=input_ids, position_ids=position_ids)

        # Fallback: Megatron Core GPTModel style - use embedding module directly
        if hasattr(self.language_model, 'embedding') and hasattr(self.language_model.embedding, 'word_embeddings'):
            embeddings = self.language_model.embedding.word_embeddings(input_ids)
            return embeddings.transpose(0, 1).contiguous()

        raise RuntimeError("Language model does not have a recognized embedding interface")

    def get_text_embeddings(
        self, input_ids: torch.Tensor, position_ids: torch.Tensor, special_token_ids: Dict[str, int]
    ) -> torch.Tensor:
        """Get embeddings for text tokens in the input.
        Args:
            input_ids: Input token IDs of shape [batch_size, seq_len] containing text tokens
                and potentially special tokens for other modalities.
            position_ids: Position IDs corresponding to input tokens, used for positional encoding.
                Shape [batch_size, seq_len].
            special_token_ids: Dictionary mapping modality names to their special token IDs.
                Used to identify non-text tokens in the input_ids.

        Returns:
            torch.Tensor: Embeddings for text tokens, shape [num_text_tokens, hidden_dim].
        """
        text_mask = torch.ones_like(input_ids, dtype=torch.bool)  # [b, s]
        for special_token_id in special_token_ids.values():
            text_mask &= input_ids != special_token_id

        batch_idx, seq_idx = text_mask.nonzero(as_tuple=True)
        input_ids_text = input_ids[batch_idx, seq_idx].unsqueeze(0)

        position_ids_text = (
            position_ids[batch_idx, seq_idx].unsqueeze(0) if position_ids is not None else None
        )

        text_embeddings = self._get_language_model_embeddings(
            input_ids=input_ids_text, position_ids=position_ids_text
        ).squeeze(1)  # Shape: [num_text_tokens, hidden_dim]
        return text_embeddings

    def get_all_text_embeddings(self, input_ids: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """Get embeddings for ALL tokens in input_ids (for Bagel-style where input_ids contains only text).

        Args:
            input_ids: Input token IDs of shape [batch_size, seq_len]
            position_ids: Position IDs corresponding to input tokens. Shape [batch_size, seq_len].

        Returns:
            torch.Tensor: Embeddings for all tokens, shape [num_tokens, hidden_dim].
        """
        text_embeddings = self._get_language_model_embeddings(
            input_ids=input_ids, position_ids=position_ids
        ).squeeze(1)  # Shape: [num_tokens, hidden_dim]
        return text_embeddings

    def align_embeddings_by_token_positions(
        self,
        input_ids: torch.Tensor,
        vision_embeddings: Optional[torch.Tensor],
        visual_latents: Optional[torch.Tensor],
        sequence_length: int,
        packed_seq_params: Optional[MoTPackedSeqParams],
    ) -> dict:
        """Assemble compact decoder_input and pre-shard all CP-sensitive tensors.

        Calls the module-level align_bagel_embeddings() function with
        self.language_model as the embedding provider.

        Returns:
            dict with keys: decoder_input, labels, loss_mask, packed_position_ids
            (see align_bagel_embeddings for full documentation)
        """
        return align_bagel_embeddings(
            language_model=self.language_model,
            input_ids=input_ids,
            vision_embeddings=vision_embeddings,
            visual_latents=visual_latents,
            sequence_length=sequence_length,
            packed_seq_params=packed_seq_params,
        )



    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        modality_inputs: Optional[Dict[str, Dict[str, Any]]] = None,
        # Parameters for Bagel-style training
        sample_lens: Optional[List[int]] = None,
        packed_position_ids: Optional[torch.Tensor] = None,
        ce_loss_indexes: Optional[torch.Tensor] = None,
        packed_label_ids: Optional[torch.Tensor] = None,
        # Additional Bagel-specific parameters for sparse sequence construction
        sequence_length: Optional[int] = None,
        # mse_loss_indexes: Optional[torch.Tensor] = None,
        vis_gen_target: Optional[torch.Tensor] = None,
        gen_loss_mask: Optional[torch.Tensor] = None,
        # Parameters for attention mask creation (BlockMask)
        split_lens: Optional[List[int]] = None,
        attn_modes: Optional[List[str]] = None,
        packed_seq_params: Optional[MoTPackedSeqParams] = None,
    ):
        # if packed_text_indexes is not None:
        #     print("MimoModel packed_text_indexes:", packed_text_indexes.shape, packed_text_indexes.sum(), packed_text_indexes.flatten()[:10], torch.any(torch.isnan(packed_text_indexes)))
        # if packed_vit_token_indexes is not None:
        #     print("MimoModel packed_vit_token_indexes:", packed_vit_token_indexes.shape)
        """Forward pass through the multimodal model.

        Args:
            input_ids: Input token IDs [batch_size, seq_length]
            position_ids: Position IDs [batch_size, seq_length]
            attention_mask: Attention mask [batch_size, seq_length]
            loss_mask: Loss mask [batch_size, seq_length]
            labels: Labels for training
            modality_inputs: Dictionary mapping modality names to encoder inputs. For example:
                {
                    "images": {
                        "clip_encoder": {"pixel_values": clip_images},
                        "vit_encoder": {"images": vit_images}
                    },
                    "audio": {
                        "whisper_encoder": {"input_features": whisper_features}
                    }
                }
            sample_lens: List of sample lengths for packed sequences (Bagel-style)
            packed_position_ids: Packed position IDs for packed sequences
            ce_loss_indexes: Boolean tensor indicating where to compute CE loss
            packed_label_ids: Packed label IDs for loss computation
            sequence_length: Total length of the full packed sequence (Bagel-style)
            split_lens: List of split lengths for attention mask creation
            attn_modes: List of attention modes for each split

        Returns:
            tuple: Tuple containing model outputs and loss mask
        """

        # 1. Process each modality to get embeddings (first PP stage only — vision
        # encoder + diffusion input projections produce the inputs that feed into
        # align_embeddings_by_token_positions; middle/last stages receive the
        # post-aligned hidden state from the previous stage via set_input_tensor).
        modality_embeddings = {}
        vision_embeddings = None
        visual_latents = None

        if self.pre_process:
            for modality_name, submodule in self.modality_submodules.items():
                # Process the modality through its submodule
                if (
                    modality_inputs
                    and modality_name in modality_inputs
                    and modality_inputs[modality_name] is not None
                ):
                    logger.debug(f"Processing {modality_name} modality")
                    embeddings = submodule.forward(encoder_inputs=modality_inputs[modality_name])
                    assert embeddings is not None, f"embeddings is None for modality_name: {modality_name}"
                    if embeddings is not None:
                        # All embeddings are now in the format [num_tokens, hidden_dim]
                        modality_embeddings[modality_name] = embeddings
                        if modality_name == "images":
                            # In Bagel, there wouldn't be much vit images to process during each batch, so here we can not shard it according to the image batch size.
                            # because FA can not accept a empty input, and FSDP can not accept skip vit fwd path.
                            # Will not shard the vit seq len shard and packed vit token, and keep same vit input on each rank.
                            # if self.cp_size > 1:
                            #     # all gather embeddings to all ranks, since different CP ranks process different images
                            #     embeddings = all_gather_image_embeddings(embeddings, vit_tokens_encoded_per_cp=packed_seq_params.vit_tokens_encoded_per_cp, group=self.cp_group)
                            vision_embeddings = embeddings
                        if modality_name == "diffusion":
                            visual_latents = embeddings
                        logger.debug(
                            f"Generated embeddings for {modality_name} with shape {embeddings.shape}"
                        )

        rank = dist.get_rank()

        if isinstance(self.language_model, BagelMCoreModel):
            logger.debug(f"Using Bagel-style with input_ids shape {input_ids.shape}")

            # Build full-sequence label tensors from sparse ce_loss_indexes / packed_label_ids.
            # align_embeddings_by_token_positions will pre-shard them to [actual_lund].
            # labels_full    = None
            # loss_mask_full = None
            # if ce_loss_indexes is not None and packed_label_ids is not None:
            #     device = input_ids.device
            #     S = sequence_length
            #     labels_full    = torch.zeros(S, dtype=torch.long,    device=device)
            #     loss_mask_full = torch.zeros(S, dtype=torch.float16, device=device)
            #     ce_loss_indexes = ce_loss_indexes.to(device)
            #     labels_full[ce_loss_indexes]    = packed_label_ids.to(device)
            #     loss_mask_full[ce_loss_indexes] = 1.0

            # Assemble compact decoder_input and pre-shard CP-sensitive tensors.
            # align_embeddings_by_token_positions / align_bagel_embeddings is the
            # single place where the full [S, H] intermediate lives.
            # BagelMCoreModel is CP-unaware and only receives pre-assembled inputs.
            #
            # Only the first PP stage runs alignment; middle/last stages receive
            # the already-compacted hidden state from the previous stage via
            # set_input_tensor → BagelMCoreModel/GPTModel internal plumbing.
            decoder_input = None
            if self.pre_process:
                aligned = self.align_embeddings_by_token_positions(
                    input_ids=input_ids,
                    vision_embeddings=vision_embeddings,
                    visual_latents=visual_latents,
                    sequence_length=sequence_length,
                    packed_seq_params=packed_seq_params,
                )
                decoder_input = aligned['decoder_input']

            lm_output = self.language_model(
                decoder_input=decoder_input,
                attention_mask=attention_mask,
                labels=labels,
                loss_mask=loss_mask,
                packed_position_ids=packed_position_ids,
                packed_seq_params=packed_seq_params,
                sample_lens=sample_lens,
                split_lens=split_lens,
                attn_modes=attn_modes,
            )

            # Step 11: Compute MSE loss at specific indexes.
            # last_hidden_state is compact [Lund+Lgen, H]; gen tokens are at [Lund:Lund+actual_lgen].
            # Filter mse_loss_indexes to those on this rank via local_gen_token_indexes.
            # Always call llm2vae so it participates in backward (FSDP correctness).
            # gen_loss_mask/vis_gen_target are always tensors from the dataloader;
            # zero gen_loss_mask means MSE contributes 0 but llm2vae is in the autograd graph.
            #
            # Only the last PP stage owns the diffusion output projection (llm2vae)
            # and computes the MSE loss; middle stages skip both.
            if self.post_process:
                Lund = packed_seq_params.padded_und_seqlen
                hidden_state = lm_output['last_hidden_state'] #[U+G,H]
                vis_gen_target = vis_gen_target.to(hidden_state.device)
                gen_hidden_state = hidden_state[Lund:] #[G, H]
                noise_pred = self.modality_submodules['diffusion'].llm2vae(gen_hidden_state)
                mse = (noise_pred - vis_gen_target) ** 2 * gen_loss_mask.unsqueeze(1)
                lm_output['mse'] = mse
                lm_output['mse_loss_mask'] = gen_loss_mask
        elif isinstance(self.language_model, BagelLLMHuggingFaceModel):
            # Standard MIMO-style: combine embeddings based on special token positions
            # Get text embeddings (filtering out special tokens)
            text_embeddings = self.get_text_embeddings(input_ids, position_ids, self.special_token_ids)
            logger.debug(f"Generated text embeddings with shape {text_embeddings.shape}")

            modality_embeddings["text"] = text_embeddings

            # 2. Merge embeddings from different modalities (HuggingFace special-token path)
            logger.debug(f"Merging embeddings from {len(modality_embeddings)} modalities")
            combined_embeddings = super().align_embeddings_by_token_positions(
                modality_embeddings=modality_embeddings,  # [num_tokens, hidden_dim] for each modality
                input_ids=input_ids,  # Pass in batch-first format [b, s]
                special_token_ids=self.special_token_ids,
            )  # [s, b, h]
            logger.debug(f"Combined embeddings shape: {combined_embeddings.shape}")

            # 3. Forward pass through language model
            # Check if language model supports Bagel-style interface
             # Bagel-style interface (HuggingFace models like BagelLLMHuggingFaceModel)
            lm_output = self.language_model(
                 decoder_input=combined_embeddings,
                 sample_lens=sample_lens,
                 attention_mask=attention_mask,
                 packed_position_ids=packed_position_ids,
                 ce_loss_indexes=ce_loss_indexes,
                 packed_label_ids=packed_label_ids,
                 split_lens=split_lens,
                 attn_modes=attn_modes,
              )
        else:
            raise ValueError(f"Unsupported language model type: {type(self.language_model)}")
        
        logger.debug(f"Language model output: {type(lm_output)}")
        # Non-last PP stage: BagelMCoreModel returned the raw hidden-state tensor
        # so Megatron's pipeline schedule can ship it to the next stage. Return
        # it as-is (a single Tensor, not a 4-tuple).
        if not self.post_process:
            return lm_output

        # Last (or PP=1) stage: BagelMCoreModel returned a dict with ce / mse / etc.
        # Return a flat tuple so FSDP's create_custom_backward_hook can attach
        # _root_pre_backward. The hook extracts top-level tensors via:
        #   output_list = [t for t in output if isinstance(t, torch.Tensor)]
        # A dict is not a Tensor and is skipped, so returning (dict, loss_mask)
        # leaves only loss_mask (requires_grad=False) visible — the hook never
        # fires. Flattening puts ce/mse directly at the top level so the hook
        # registers on them. forward_step reconstructs the dict for loss_func.
        return (
            lm_output.get('ce'),
            lm_output.get('mse'),
            lm_output.get('mse_loss_mask'),
            loss_mask,
        )
