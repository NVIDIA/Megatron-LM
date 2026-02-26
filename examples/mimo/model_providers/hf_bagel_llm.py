# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

import torch
import torch.nn.functional as F

from megatron.core.models.huggingface import HuggingFaceModule

try:
    from bagel.modeling.bagel.qwen2_navit import Qwen2ForCausalLM, Qwen2MoTDecoderLayer

    HAVE_TRANSFORMERS = True
except ImportError:
    from unittest.mock import MagicMock

    Qwen2ForCausalLM = MagicMock()
    Qwen2MoTDecoderLayer = MagicMock()

    HAVE_TRANSFORMERS = False


class BagelLLMHuggingFaceModel(HuggingFaceModule):
    """
    Wrapper for Bagel LLM HuggingFace models.
    """

    # Currently applies to FSDP2 only, not the Megatron FSDP implementation.
    _fsdp_modules = [Qwen2MoTDecoderLayer]

    def __init__(self, config, llm_config, llm_path=None):
        if not HAVE_TRANSFORMERS:
            raise ImportError(
                "transformers is required for QwenHuggingFaceModel, "
                "please install it with `pip install transformers`"
            )

        super().__init__(config)
        if llm_path is None:
            self.model = Qwen2ForCausalLM(llm_config)
        else:
            self.model = Qwen2ForCausalLM.from_pretrained(llm_path, config=llm_config)

        self.num_heads = llm_config.num_attention_heads
        # Check if model uses MoE (Mixture of Transformers)
        self.use_moe = "Mo" in llm_config.layer_module if hasattr(llm_config, 'layer_module') else False

    def forward(self, *args, **kwargs):
        """Qwen forward.

        For Bagel-style training:
        - input_ids: text token IDs [batch, num_text_tokens]
        - position_ids: position IDs for text tokens [batch, num_text_tokens]
        - sequence_length: total length of the packed sequence (text + vision)
        - packed_text_indexes: indices where text tokens go in the full sequence
        - packed_vit_token_indexes: indices where vision tokens go
        - packed_vae_token_indexes: indices where VAE tokens go (for generation)
        - vision_embeddings: vision embeddings from vision encoder [num_vision_tokens, hidden]
        - split_lens: list of split lengths for attention mask creation
        - attn_modes: list of attention modes for each split
        """
        ce_loss_indexes = kwargs.get("ce_loss_indexes")
        packed_label_ids = kwargs.get("packed_label_ids")

        # Check if we're using Bagel-style full sequence construction
        sequence_length = kwargs.get("sequence_length")
        packed_text_indexes = kwargs.get("packed_text_indexes")

        if sequence_length is not None and packed_text_indexes is not None:
            # Bagel-style: construct full packed_sequence from input_ids + vision embeddings
            input_ids = kwargs.get("input_ids")
            position_ids = kwargs.get("position_ids")

            # Get text embeddings from input_ids
            # get_input_embeddings() returns embed_tokens layer
            print("language_model.model.embed_tokens forward")
            print("packed_text_ids", input_ids.shape, input_ids.sum())
            text_embeddings = self.model.get_input_embeddings()(input_ids)  # [batch, seq, hidden]
            text_embeddings = text_embeddings.squeeze(0)  # [num_text_tokens, hidden]
            print("after language model.model.embed_tokens forward, packed_text_embedding", text_embeddings.shape, text_embeddings.sum())

            hidden_size = text_embeddings.shape[-1]
            device = text_embeddings.device
            dtype = text_embeddings.dtype

            # Create full packed_sequence
            packed_sequence = torch.zeros(
                (sequence_length, hidden_size),
                dtype=dtype,
                device=device
            )

            # Place text embeddings at their positions
            packed_text_indexes = packed_text_indexes.to(device)
            packed_sequence[packed_text_indexes] = text_embeddings

            # Place vision embeddings if present
            vision_embeddings = kwargs.get("vision_embeddings")
            packed_vit_token_indexes = kwargs.get("packed_vit_token_indexes")

            if packed_vit_token_indexes is not None:
                print("packed_vit_token_indexes", packed_vit_token_indexes.shape, packed_vit_token_indexes.sum())
            else:
                print("packed_vit_token_indexes is None")
            if vision_embeddings is not None and packed_vit_token_indexes is not None:
                packed_vit_token_indexes = packed_vit_token_indexes.to(device)
                packed_sequence[packed_vit_token_indexes] = vision_embeddings

            # Create attention mask (BlockMask) here instead of in data conversion
            # This avoids the serialization issue with BlockMask
            attention_mask = kwargs.get("attention_mask", None)
            sample_lens = kwargs["sample_lens"]
            nested_attention_masks = kwargs.get("nested_attention_masks", None)

            # Priority: nested_attention_masks > create BlockMask from split_lens/attn_modes
            if nested_attention_masks is not None:
                # Use nested_attention_masks directly if provided (from use_flex=False mode)
                attention_mask = nested_attention_masks
            else:
                split_lens = kwargs.get("split_lens")
                attn_modes = kwargs.get("attn_modes")

                if split_lens is not None and attn_modes is not None:
                    from bagel.data.data_utils import create_sparse_mask
                    from torch.nn.attention.flex_attention import create_block_mask
                    sparse_mask = create_sparse_mask(sample_lens, split_lens, attn_modes, device)
                    seqlen = sum(sample_lens)
                    attention_mask = create_block_mask(
                        sparse_mask,
                        B=1, H=self.num_heads,
                        Q_LEN=seqlen,
                        KV_LEN=seqlen,
                        device=device,
                        BLOCK_SIZE=128,
                        _compile=True
                    )
                else:
                    # Fallback: create a simple causal mask if split_lens/attn_modes not provided
                    attention_mask = None

            # Prepare MoE extra inputs if needed
            extra_inputs = {}
            packed_vae_token_indexes = kwargs.get("packed_vae_token_indexes", None)
            if self.use_moe:
                # packed_und_token_indexes: understanding tokens (text + vit)
                packed_und_token_indexes = packed_text_indexes
                if packed_vit_token_indexes is not None:
                    packed_und_token_indexes = torch.cat([packed_text_indexes, packed_vit_token_indexes], dim=0)

                # packed_gen_token_indexes: generation tokens (vae)
                if packed_vae_token_indexes is not None:
                    packed_gen_token_indexes = packed_vae_token_indexes.to(device)
                else:
                    packed_gen_token_indexes = None

                extra_inputs.update(
                    packed_und_token_indexes=packed_und_token_indexes,
                    packed_gen_token_indexes=packed_gen_token_indexes,
                )
                print("BagelLLMHuggingFaceModel packed_und_token_indexes:", packed_und_token_indexes.shape)
                if packed_gen_token_indexes is not None:
                    print("BagelLLMHuggingFaceModel packed_gen_token_indexes:", packed_gen_token_indexes.shape)
                else:
                    print("BagelLLMHuggingFaceModel packed_gen_token_indexes is None")
        # else:
        #     # Legacy path: decoder_input is already the combined embeddings
        #     decoder_input = kwargs.get("decoder_input")
        #     if decoder_input is not None:
        #         combined_embeddings = decoder_input.permute(1, 0, 2)
        #         packed_sequence = combined_embeddings.squeeze(0)  # [seq_len, hidden]
        #     else:
        #         # Fallback: compute embeddings from input_ids
        #         input_ids = kwargs.get("input_ids")
        #         text_embeddings = self.model.get_input_embeddings()(input_ids)
        #         packed_sequence = text_embeddings.squeeze(0)
        #     attention_mask = kwargs.get("attention_mask")
        #     extra_inputs = {}
        # print("language_model forward")
        # print("packed_sequence", packed_sequence.shape, packed_sequence.to(torch.float32).sum())
        # print("sample_lens", sample_lens)
        # print("attention_mask", attention_mask.shape)
        # print("packed_position_ids", kwargs["packed_position_ids"].shape, kwargs["packed_position_ids"].sum())
        last_hidden_state = self.model.forward_train(
            packed_sequence=packed_sequence,
            sample_lens=sample_lens,
            attention_mask=attention_mask,
            packed_position_ids=kwargs["packed_position_ids"],
            **extra_inputs,
        )
        print("after language_model forward, last_hidden_state", last_hidden_state.shape, last_hidden_state.to(torch.float32).sum(), last_hidden_state.flatten()[:10])
        print("================================================")

        if ce_loss_indexes is not None:
            print("language_model.lm_head and ce loss")
            print("ce_loss_indexes", ce_loss_indexes.shape, ce_loss_indexes.sum())
            if packed_label_ids is not None:
                print("packed_label_ids", packed_label_ids.shape, packed_label_ids.sum())
            packed_ce_preds = self.model.lm_head(last_hidden_state[ce_loss_indexes])
            ce = F.cross_entropy(packed_ce_preds, packed_label_ids, reduction="none")
            print("after language_model.lm_head forward, ce", ce.shape, ce.sum(), ce)
            print("================================================")
        else:
            ce = None
        print("****************************************************")
        return dict(last_hidden_state=last_hidden_state, ce=ce)

    def embedding(self, input_ids, position_ids=None):
        """Function to run process tokens with input embeddings.

        Args:
            input_ids: [batch_size, seq_len]
            position_ids: [batch_size, seq_len] (optional, not used in Bagel)

        Returns:
            embeddings: [seq_len, batch_size, hidden_dim]
        """
        # get_input_embeddings() returns the embed_tokens layer
        # embed_tokens(input_ids) returns [batch_size, seq_len, hidden_dim]
        embeddings = self.model.get_input_embeddings()(input_ids)
        # Transpose to [seq_len, batch_size, hidden_dim] for Megatron convention
        return embeddings.transpose(1, 0).contiguous()
