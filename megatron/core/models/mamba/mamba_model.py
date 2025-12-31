# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from typing import Literal, Optional

from torch import Tensor
import torch
from typing import Dict
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core import parallel_state, tensor_parallel
from megatron.core.config_logger import has_config_logger_enabled, log_config_to_disk
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.quantization.utils import get_quant_config_or_none
from megatron.core.tensor_parallel import gather_from_sequence_parallel_region
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.enums import ModelType
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.utils import WrappedTensor, deprecate_inference_params
from megatron.core.transformer.multi_token_prediction import (
    MTPLossAutoScaler,
    MTPLossLoggingHelper,
    MultiTokenPredictionBlock,
    roll_tensor,
    tie_output_layer_state_dict,
    tie_word_embeddings_state_dict,
)
from megatron.core.packed_seq_params import PackedSeqParams


class MambaModel(LanguageModule):
    """Mamba language model.

    Args:
        config (TransformerConfig): Model config
        mamba_stack_spec (ModuleSpec): Specifies the modules to use for the various layer types
        vocab_size (int): Vocabulary size
        max_sequence_length (int): maximum size of sequence.
            This is used for positional embedding
        pre_process (bool, optional): Include embedding layer
            (used with pipeline parallelism). Defaults to True.
        hybrid_attention_ratio (float, optional): The target ratio of attention
            layers to total layers
        hybrid_mlp_ratio (float, optional): The target ratio of mlp layers to total layers
        hybrid_override_pattern (str, optional): The hybrid layer pattern to override with
        post_process (bool, optional): Include an output layer (used with pipeline parallelism).
            Defaults to True.
        fp16_lm_cross_entropy (bool, optional): Defaults to False.
        parallel_output (bool, optional): Do not gather the outputs, keep them split across tensor
            parallel ranks. Defaults to True.
        share_embeddings_and_output_weights (bool, optional): When True, input embeddings and
            output logit weights are shared. Defaults to False.
        position_embedding_type (Literal[learned_absolute,rope,none], optional):  Position
            embedding type. Defaults to 'none'.
        rotary_percent (float, optional): Percent of rotary dimension to use for rotary position
            embeddings. Ignored unless position_embedding_type is 'rope'. Defaults to 1.0.
        rotary_base (int, optional): Base period for rotary position embeddings. Ignored unless
            position_embedding_type is 'rope'. Defaults to 10000.
        seq_len_interpolation_factor (Optional[float], optional): scale of linearly
            interpolating RoPE for longer sequences. The value must be a float larger than 1.0.
             Defaults to None.
        pg_collection (ProcessGroupCollection, optional): Model communication process groups.
    """

    def __init__(
        self,
        config: TransformerConfig,
        mamba_stack_spec: ModuleSpec,
        vocab_size: int,
        max_sequence_length: int,
        pre_process: bool = True,
        hybrid_attention_ratio: float = 0.0,
        hybrid_mlp_ratio: float = 0.0,
        hybrid_override_pattern: str = None,
        mtp_hybrid_override_pattern: str = None,
        post_process: bool = True,
        fp16_lm_cross_entropy: bool = False,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = False,
        # Mamba with no attention has no need for position embeddings, so none is default
        position_embedding_type: Literal['learned_absolute', 'rope', 'none'] = 'none',
        rotary_percent: float = 1.0,
        rotary_base: int = 10000,
        scatter_embedding_sequence_parallel: bool = True,
        seq_len_interpolation_factor: Optional[float] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        vp_stage: Optional[int] = None,
        mtp_block_spec: Optional[ModuleSpec] = None,
    ) -> None:
        super().__init__(config=config, pg_collection=pg_collection)

        if has_config_logger_enabled(config):
            log_config_to_disk(config, locals(), prefix=type(self).__name__)

        self.mamba_stack_spec: ModuleSpec = mamba_stack_spec
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.pre_process = pre_process
        self.hybrid_attention_ratio = hybrid_attention_ratio
        self.hybrid_mlp_ratio = hybrid_mlp_ratio
        self.hybrid_override_pattern = hybrid_override_pattern
        self.mtp_hybrid_override_pattern = mtp_hybrid_override_pattern
        self.post_process = post_process
        self.fp16_lm_cross_entropy = fp16_lm_cross_entropy
        self.parallel_output = parallel_output
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights
        self.position_embedding_type = position_embedding_type
        self.mtp_block_spec = mtp_block_spec
        self.mtp_process = mtp_block_spec is not None

        # Cache for RoPE tensors which do not change between iterations.
        self.rotary_pos_emb_cache = {}


        # megatron core pipelining currently depends on model type
        # TODO: remove this dependency ?
        self.model_type = ModelType.encoder_or_decoder

        if self.pre_process or self.mtp_process:
            self.embedding = LanguageModelEmbedding(
                config=self.config,
                vocab_size=self.vocab_size,
                max_sequence_length=self.max_sequence_length,
                position_embedding_type=position_embedding_type,
                scatter_to_sequence_parallel=scatter_embedding_sequence_parallel,
                tp_group=self.pg_collection.tp,
            )

        if self.position_embedding_type == 'rope':
            self.rotary_pos_emb = RotaryEmbedding(
                kv_channels=self.config.kv_channels,
                rotary_percent=rotary_percent,
                seq_len_interpolation_factor=seq_len_interpolation_factor,
                rotary_base=rotary_base,
                use_cpu_initialization=self.config.use_cpu_initialization,
                cp_group=self.pg_collection.cp,
            )

        self.decoder = build_module(
            mamba_stack_spec,
            self.config,
            pre_process=self.pre_process,
            hybrid_attention_ratio=self.hybrid_attention_ratio,
            hybrid_mlp_ratio=self.hybrid_mlp_ratio,
            hybrid_override_pattern=self.hybrid_override_pattern,
            post_process=self.post_process,
            dtype=config.params_dtype,
            pg_collection=self.pg_collection,
        )

        if self.mtp_process:
            self.mtp = MultiTokenPredictionBlock(
                config=self.config, spec=self.mtp_block_spec, vp_stage=vp_stage, mtp_hybrid_override_pattern=self.mtp_hybrid_override_pattern
            )

        # Output
        if post_process or self.mtp_process:
            self.output_layer = tensor_parallel.ColumnParallelLinear(
                config.hidden_size,
                self.vocab_size,
                config=config,
                init_method=config.init_method,
                bias=False,
                skip_bias_add=False,
                gather_output=not self.parallel_output,
                skip_weight_param_allocation=self.pre_process
                and self.share_embeddings_and_output_weights,
                tp_group=self.pg_collection.tp,
            )

        if self.pre_process or self.post_process:
            self.setup_embeddings_and_output_layer()

        for name, module in self.named_modules():
            if hasattr(module, 'finish_init'):
                quant_config = get_quant_config_or_none(name, self.config.quant_recipe)
                module.finish_init(quant_config)

    def set_input_tensor(self, input_tensor: Tensor) -> None:
        """Sets input tensor to the model.

        See megatron.model.transformer.set_input_tensor()

        Args:
            input_tensor (Tensor): Sets the input tensor for the model.
        """
        # This is usually handled in schedules.py but some inference code still
        # gives us non-lists or None
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]

        assert len(input_tensor) == 1, 'input_tensor should only be length 1 for gpt/bert'
        self.decoder.set_input_tensor(input_tensor[0])

    def _preprocess(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        decoder_input: Tensor = None,
        inference_context: BaseInferenceContext = None,
        packed_seq_params: PackedSeqParams = None,
    ):
        """Preprocesses inputs for the transformer decoder.

        Applies embeddings to input tokens, or uses `decoder_input` from a previous
        pipeline stage. Also sets up rotary positional embeddings.
        """

        # If decoder_input is provided (not None), then input_ids and position_ids are ignored.
        # Otherwise, apply embedding layer on input_ids and position_ids to get decoder_input.

        # Decoder embedding.
        if decoder_input is not None:
            pass
        elif self.pre_process:
            decoder_input = self.embedding(input_ids=input_ids, position_ids=position_ids)
        else:
            # intermediate stage of pipeline
            # decoder will get hidden_states from encoder.input_tensor
            decoder_input = None

        # Rotary positional embeddings (embedding is None for PP intermediate devices)
        rotary_pos_emb = None
        rotary_pos_cos = None
        rotary_pos_sin = None
        if self.position_embedding_type == 'rope' and not self.config.multi_latent_attention:
            if not self.training and self.config.flash_decode and inference_context:
                assert (
                    inference_context.is_static_batching()
                ), "GPTModel currently only supports static inference batching."
                # Flash decoding uses precomputed cos and sin for RoPE
                rotary_pos_cos, rotary_pos_sin = self.rotary_pos_emb_cache.setdefault(
                    inference_context.max_sequence_length,
                    self.rotary_pos_emb.get_cos_sin(inference_context.max_sequence_length),
                )
            else:
                rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
                    inference_context, self.decoder, decoder_input, self.config, packed_seq_params
                )
                rotary_pos_emb = self.rotary_pos_emb(
                    rotary_seq_len,
                    packed_seq=packed_seq_params is not None
                    and packed_seq_params.qkv_format == 'thd',
                )
        elif self.position_embedding_type == 'mrope' and not self.config.multi_latent_attention:
            if self.training or not self.config.flash_decode:
                rotary_pos_emb = self.rotary_pos_emb(position_ids, self.mrope_section)
            else:
                # Flash decoding uses precomputed cos and sin for RoPE
                raise NotImplementedError(
                    "Flash decoding uses precomputed cos and sin for RoPE, not implmented in "
                    "MultimodalRotaryEmbedding yet."
                )

        if (
            (self.config.enable_cuda_graph or self.config.flash_decode)
            and rotary_pos_cos is not None
            and inference_context
            and inference_context.is_static_batching()
            and not self.training
        ):
            current_batch_size = input_ids.shape[0]
            sequence_len_offset = torch.tensor(
                [inference_context.sequence_len_offset] * current_batch_size,
                dtype=torch.int32,
                device=rotary_pos_cos.device,  # Co-locate this with the rotary tensors
            )
        else:
            sequence_len_offset = None

        # Wrap decoder_input to allow the decoder (TransformerBlock) to delete the
        # reference held by this caller function, enabling early garbage collection for
        # inference. Skip wrapping if decoder_input is logged after decoder completion.
        if (
            inference_context is not None
            and not self.training
            and not has_config_logger_enabled(self.config)
        ):
            decoder_input = WrappedTensor(decoder_input)

        return decoder_input, rotary_pos_emb, rotary_pos_cos, rotary_pos_sin, sequence_len_offset

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        decoder_input: Tensor = None,
        labels: Tensor = None,
        inference_context: BaseInferenceContext = None,
        runtime_gather_output: Optional[bool] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
        loss_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward function of the Mamba model. This function passes the input tensors
        through the embedding layer, and then the decoder and finally into the post
        processing layer (optional).

        It either returns the Loss values if labels are given or the final hidden units
        """
        # If decoder_input is provided (not None), then input_ids and position_ids are ignored.
        # Otherwise, apply embedding layer on input_ids and position_ids to get decoder_input.

        inference_context = deprecate_inference_params(inference_context, inference_params)

        in_inference_mode = inference_context is not None and not self.training

        if in_inference_mode:
            assert runtime_gather_output, "Inference must always gather TP logits"

        # Decoder embedding.
        if decoder_input is not None:
            pass
        elif self.pre_process:
            decoder_input = self.embedding(input_ids=input_ids, position_ids=position_ids)
        else:
            # intermediate stage of pipeline
            # decoder will get hidden_states from encoder.input_tensor
            decoder_input = None

        rotary_pos_emb = None
        if self.position_embedding_type == 'rope':
            rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
                inference_context, self.decoder, decoder_input, self.config
            )
            rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len)

        # Wrap decoder_input to allow the decoder (MambaBlock) to delete the
        # reference held by this caller function, enabling early garbage collection
        # for inference.
        if in_inference_mode:
            decoder_input = WrappedTensor(decoder_input)

        # The following assert will currently fail when running inference.
        # Commented out for now.
        # TODO (duncan/rwaleffe): (1) confirm that the externally-generated
        #   attention mask is not needed and is ignored by the model in
        #   inference mode, (2) reduce the size of the externally-generated
        #   attention mask to prevent CPU OOM (as we did for training), (3)
        #   force the attention mask passed to the model in inference mode to
        #   be None, so this assert will succeed.
        # assert attention_mask is None, "The attention mask is ignored and should be set to None"

        # Run decoder.
        hidden_states = self.decoder(
            hidden_states=decoder_input,
            attention_mask=attention_mask,
            inference_context=inference_context,
            rotary_pos_emb=rotary_pos_emb,
        )

        # Get output weight for shared embeddings
        output_weight = None
        if self.share_embeddings_and_output_weights:
            output_weight = self.shared_embedding_or_output_weight()            

        if self.mtp_process:
            extra_block_kwargs = None
            packed_seq_params = None
            decoder_input, rotary_pos_emb, rotary_pos_cos, rotary_pos_sin, sequence_len_offset = self._preprocess(
                input_ids=input_ids,
                position_ids=position_ids,
                decoder_input=decoder_input,
                inference_context=inference_context,
                packed_seq_params=packed_seq_params,
            )


            hidden_states = self.mtp(
                input_ids=input_ids,
                position_ids=position_ids,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                inference_params=inference_params,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                packed_seq_params=packed_seq_params,
                sequence_len_offset=sequence_len_offset,
                embedding=self.embedding,
                **(extra_block_kwargs or {}),
            )

        if not self.post_process:
            return hidden_states

        if self.mtp_process:
            hidden_states = self.mtp.process_loss(
                hidden_states=hidden_states,
                labels=labels,
                loss_mask=loss_mask,
                output_layer=self.output_layer,
                output_weight=output_weight,
                runtime_gather_output=runtime_gather_output,
                is_training=self.training,
                compute_language_model_loss=self.compute_language_model_loss,
            )

        
        sequence_parallel_override = False
        if in_inference_mode and inference_context.materialize_only_last_token_logits:
            if inference_context.is_static_batching():
                hidden_states = hidden_states[-1:, :, :]
            else:
                if self.output_layer.sequence_parallel:
                    # Perform the sequence parallel gather here instead of after the output layer
                    # because we need to slice the last token logits from the full view of the
                    # packed logits across all requests.
                    hidden_states = gather_from_sequence_parallel_region(
                        hidden_states, group=self.pg_collection.tp
                    )
                    self.output_layer.sequence_parallel = False
                    sequence_parallel_override = True

                # Reshape [B, 1, H] to [1, B, H] → extract each sample's true last‐token hidden
                # state ([B, H]) → unsqueeze back to [B, 1, H]
                # (so that the output layer, which expects S×B×H, receives only the final token)
                hidden_states = inference_context.last_token_logits(
                    hidden_states.squeeze(1).unsqueeze(0)
                ).unsqueeze(1)

        logits, _ = self.output_layer(
            hidden_states, weight=output_weight, runtime_gather_output=runtime_gather_output
        )

        # Restore sequence parallel execution to the output layer if necessary.
        if sequence_parallel_override:
            assert (
                in_inference_mode
                and inference_context.is_dynamic_batching()
                and inference_context.materialize_only_last_token_logits
            )
            self.output_layer.sequence_parallel = True

        if labels is None:
            # [s b h] => [b s h]
            return logits.transpose(0, 1).contiguous()

        loss = self.compute_language_model_loss(labels, logits)

        return loss

    def shared_embedding_or_output_weight(self) -> Optional[Tensor]:
        """Gets the embedding weight or output logit weights when share input embedding and
        output weights set to True or when use Multi-Token Prediction (MTP) feature.

        Returns:
            Optional[Tensor]: During pre processing or MTP process it returns the input embeddings weight.
            Otherwise, during post processing it returns the final output layers weight.
        """
        if self.pre_process or self.mtp_process:
            # Multi-Token Prediction (MTP) need both embedding layer and output layer.
            # So there will be both embedding layer and output layer in the mtp process stage.
            # In this case, if share_embeddings_and_output_weights is True, the shared weights
            # will be stored in embedding layer, and output layer will not have any weight.
            assert hasattr(
                self, 'embedding'
            ), f"embedding is needed in this pipeline stage, but it is not initialized."
            return self.embedding.word_embeddings.weight
        elif self.post_process:
            return self.output_layer.weight
        return None

    def sharded_state_dict(
        self, prefix: str = '', sharded_offsets: tuple = (), metadata: Optional[Dict] = None
    ) -> ShardedStateDict:
        """Sharded state dict implementation for MambaModel backward-compatibility.

        Removing extra state.
        Tie word embeddings and output layer in mtp process stage.

        Args:
            prefix (str): Module name prefix.
            sharded_offsets (tuple): PP related offsets, expected to be empty at this module level.
            metadata (Optional[Dict]): metadata controlling sharded state dict creation.

        Returns:
            ShardedStateDict: sharded state dict for the MambaModel
        """
        sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        
        # Multi-Token Prediction (MTP) need both embedding layer and output layer in
        # mtp process stage.
        # If MTP is not placed in the pre processing stage, we need to maintain a copy of
        # embedding layer in the mtp process stage and tie it to the embedding in the pre
        # processing stage.
        # Also, if MTP is not placed in the post processing stage, we need to maintain a copy
        # of output layer in the mtp process stage and tie it to the output layer in the post
        # processing stage.
        if self.mtp_process and not self.pre_process:
            emb_weight_key = f'{prefix}embedding.word_embeddings.weight'
            emb_weight = self.embedding.word_embeddings.weight
            tie_word_embeddings_state_dict(sharded_state_dict, emb_weight, emb_weight_key)
        if self.mtp_process and not self.post_process:
            # We only need to tie the output layer weight if share_embeddings_and_output_weights
            # is False. Because if share_embeddings_and_output_weights is True, the shared weight
            # will be stored in embedding layer, and output layer will not have any weight.
            if not self.share_embeddings_and_output_weights:
                output_layer_weight_key = f'{prefix}output_layer.weight'
                output_layer_weight = self.output_layer.weight
                tie_output_layer_state_dict(
                    sharded_state_dict, output_layer_weight, output_layer_weight_key
                )

        return sharded_state_dict
