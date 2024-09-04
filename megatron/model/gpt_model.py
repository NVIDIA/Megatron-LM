# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company.
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

"""GPT-2 model."""

import torch
from collections import OrderedDict

from megatron import get_args
from megatron.core import mpu, tensor_parallel, sequence_parallel
from .module import MegatronModule, fp32_to_float16, float16_to_fp32

from .enums import AttnMaskType
from .language_model import parallel_lm_logits
from .language_model import get_language_model
from .utils import init_method_normal
from .utils import scaled_init_method_normal

from megatron.model import LayerNorm, RMSNorm
from .language_model import EmbeddingPipe
from .transformer import ParallelTransformerLayerPipe, LMHeadPipe, get_num_experts_per_layer
from deepspeed.pipe import PipelineModule, LayerSpec, TiedLayerSpec


try:         
    from deepspeed.checkpoint import (
        VOCABULARY_PARAMETER_PATTERNS,
        PIPELINE_REPLICATED_PARAMETER_PATTERNS,
        TP_REPLICATED_PARAMETER_PATTERNS,
        PARAMETER_WITH_ROW_PARALLELISM_PATTERNS,
        PARAMETER_WITH_2_SUB_PARAMS_CAT_DIM_0,
    )
    DS_UNIVERSAL_CHECKPOINT_INFO = True 
except ImportError:
    DS_UNIVERSAL_CHECKPOINT_INFO = False  


def post_language_model_processing(lm_output, labels, logit_weights,
                                   parallel_output,
                                   fp16_lm_cross_entropy):

    # Output. Format [s b h]
    output = parallel_lm_logits(
        lm_output,
        logit_weights,
        parallel_output)

    if labels is None:
        # [s b h] => [b s h]
        return output.transpose(0,1).contiguous()
    else:
        # [b s] => [s b]
        labels = labels.transpose(0,1).contiguous()
        cross_entropy = sequence_parallel.vocab_sequence_parallel_cross_entropy if mpu.get_sequence_parallel_world_size() > 1 \
            else tensor_parallel.vocab_parallel_cross_entropy
        if fp16_lm_cross_entropy:
            assert output.dtype == torch.half
            loss = cross_entropy(output, labels)
        else:
            loss = cross_entropy(output.float(), labels)

        # [s b] => [b, s]
        loss = loss.transpose(0,1).contiguous()
        return loss


class UniversalCheckpointInfo:
    def __init__(self, using_model_pipe: bool):
        self.using_model_pipe = using_model_pipe
        self.args = get_args()
        self.info = self._build_universal_checkpoint_info()

    def get(self):
        return self.info

    def _build_universal_checkpoint_info(self):
        info = dict()
        if DS_UNIVERSAL_CHECKPOINT_INFO:
            # Vocabulary parameters (embeddings) that require special handling due to padding.
            info[VOCABULARY_PARAMETER_PATTERNS] = self._get_vocab_param_patterns()

            if self.using_model_pipe:
                # Replicated (shared) parameters on the pipeline dimension
                info[PIPELINE_REPLICATED_PARAMETER_PATTERNS] = self._get_pp_replicated_param_patterns()

            if self.args.tensor_model_parallel_size > 1:
                # Parameter slices that should be averaged not concatenated.
                info[TP_REPLICATED_PARAMETER_PATTERNS] = self._get_tp_replicated_param_patterns()

                # Parameter that are sliced on the row dimension
                info[PARAMETER_WITH_ROW_PARALLELISM_PATTERNS] = self._get_row_parallel_param_patterns()

            # SWIGLU parameters are first sliced on dim=0 to tp slices
            # Then, each tp slice is chunked into 2 to create the linear layers L1, L2 used for silu(L1(x)) * L2(x))
            info[PARAMETER_WITH_2_SUB_PARAMS_CAT_DIM_0] = self._get_swiglu_col_parallel_param_patterns()
        return info

    def _get_vocab_param_patterns(self):
        if self.using_model_pipe:
            if self.args.untie_embeddings_and_output_weights:
                patterns = [
                    r"\d+.word_embeddings.weight",
                    r"\d+.lm_head.weight"
                ]
            else:
                patterns = [
                    r"tied_modules.embed.word_embeddings.weight"
                ]
        else:
            patterns = [
                "language_model.embedding.word_embeddings.weight"
            ]
            if self.args.untie_embeddings_and_output_weights:
                patterns.append("language_model.output_layer.weight")
        return patterns

    def _get_pp_replicated_param_patterns(self):
        if self.args.untie_embeddings_and_output_weights:
            return []
        patterns = self._get_vocab_param_patterns()
        if self.args.add_position_embedding:
            patterns.append(r"tied_modules.embed.position_embeddings.weight")
        return patterns

    def _layers_prefix(self):
        return "" if self.using_model_pipe else "language_model.encoder.layers."

    def _get_tp_replicated_param_patterns(self):
        layers_prefix = self._layers_prefix()
        patterns = [
            layers_prefix + r"\d+.input_layernorm.weight",
            layers_prefix + r"\d+.post_attention_layernorm.weight",
        ]
        # Add final normalization layer
        final_norm_w_pattern = r"\d+.weight" if self.using_model_pipe \
            else "language_model.encoder.final_layernorm.weight"
        patterns.append(final_norm_w_pattern)
        if self.args.normalization == 'layernorm':
            final_norm_b_pattern = r"\d+.bias" if self.using_model_pipe \
                else "language_model.encoder.final_layernorm.bias"
            patterns.append(final_norm_b_pattern)
        # add Positional Embedding
        if self.args.add_position_embedding:
            pos_emb_pattern = "tied_modules.embed.position_embeddings.weight" if self.using_model_pipe \
                else "language_model.embedding.position_embeddings.weight"
            patterns.append(pos_emb_pattern)
        # add Linear bias
        if self.args.add_bias_linear:
            patterns.extend([
                layers_prefix + r"\d+.self_attention.dense.bias",
                layers_prefix + r"\d+.mlp.dense_4h_to_h.bias",
            ])
        # add LN bias
        if self.args.normalization == 'layernorm':
            patterns.extend([
                layers_prefix + r"\d+.input_layernorm.bias",
                layers_prefix + r"\d+.post_attention_layernorm.bias",
            ])
        return patterns

    def _get_row_parallel_param_patterns(self):
        layers_prefix = self._layers_prefix()
        return [
            layers_prefix + r"\d+.mlp.dense_4h_to_h.weight",
            layers_prefix + r"\d+.self_attention.dense.weight",
        ]

    def _get_swiglu_col_parallel_param_patterns(self):
        if not self.args.swiglu:
            return []
        layers_prefix = self._layers_prefix()
        patterns = [
            layers_prefix + r"\d+.mlp.dense_h_to_4h.weight",
        ]
        if self.args.add_bias_linear:
            patterns.append(layers_prefix + r"\d+.mlp.dense_h_to_4h.bias")
        return patterns


class GPTModel(MegatronModule):
    """GPT-2 Language model."""

    def __init__(self,
                 config,
                 num_tokentypes=0,
                 parallel_output=True,
                 pre_process=True,
                 post_process=True,
                 return_moe_loss=True):
        args = get_args()
        super().__init__(config=config, share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights)

        self.parallel_output = parallel_output
        self.pre_process = pre_process
        self.post_process = post_process
        self.fp16_lm_cross_entropy = args.fp16_lm_cross_entropy
        self.return_moe_loss = return_moe_loss
        self.untie_embeddings_and_output_weights = args.untie_embeddings_and_output_weights

        self.language_model, self._language_model_key = get_language_model(
            config=config,
            num_tokentypes=num_tokentypes,
            add_pooler=False,
            encoder_attn_mask_type=AttnMaskType.causal,
            pre_process=self.pre_process,
            post_process=self.post_process,
            num_experts=args.num_experts)

        if not args.untie_embeddings_and_output_weights:
            self.initialize_word_embeddings()

    def set_input_tensor(self, input_tensor):
        """See megatron.model.transformer.set_input_tensor()"""
        self.language_model.set_input_tensor(input_tensor)

    def forward(self, input_ids, position_ids, attention_mask,
                retriever_input_ids=None,
                retriever_position_ids=None,
                retriever_attn_mask=None,
                labels=None, tokentype_ids=None, inference_params=None,
                curriculum_seqlen=None):
        args = get_args()
        if curriculum_seqlen is not None:
            args.curriculum_seqlen = curriculum_seqlen
            if curriculum_seqlen < input_ids.size()[1]:
                # seqlen-based curriculum learning
                # input_ids, position_ids, labels have size [batch size, seqlen]
                input_ids = input_ids[:, :curriculum_seqlen].contiguous()
                position_ids = position_ids[:, :curriculum_seqlen].contiguous()
                if labels is not None:
                    labels = labels[:, :curriculum_seqlen].contiguous()

                # attention_mask has size [1, 1, seqlen, seqlen]
                attention_mask = attention_mask[:, :, :curriculum_seqlen, :curriculum_seqlen].contiguous()
        else:
            if args.curriculum_learning_legacy:
                # If got a None input, need to reset curriculum_seqlen on user side
                args.curriculum_seqlen = args.seq_length

        lm_output, moe_losses = self.language_model(
            input_ids,
            position_ids,
            attention_mask,
            retriever_input_ids=retriever_input_ids,
            retriever_position_ids=retriever_position_ids,
            retriever_attn_mask=retriever_attn_mask,
            inference_params=inference_params)

        if self.post_process:
            lm_output = post_language_model_processing(
                lm_output, labels,
                self.language_model.output_layer.weight if self.untie_embeddings_and_output_weights else self.shared_embedding_or_output_weight(),
                self.parallel_output,
                self.fp16_lm_cross_entropy)

        return lm_output, moe_losses if self.return_moe_loss else lm_output

    def state_dict_for_save_checkpoint(self, prefix='', keep_vars=False):

        state_dict_ = {}
        language_model_state_dict = self.language_model.state_dict_for_save_checkpoint(
                prefix=prefix, keep_vars=keep_vars)
        # MoE states need to be handled separately by DeepSpeed engine, thus
        # moving them to the top level dictionary
        if "moe_state_dict" in language_model_state_dict:
            for key in list(language_model_state_dict["moe_state_dict"].keys()):
                state_dict_[key] = language_model_state_dict["moe_state_dict"].pop(key)
            del language_model_state_dict["moe_state_dict"]
        state_dict_[self._language_model_key] = language_model_state_dict
        # Save word_embeddings.
        if self.post_process and not self.pre_process and not self.untie_embeddings_and_output_weights:
            state_dict_[self._word_embeddings_for_head_key] \
                = self.word_embeddings.state_dict(prefix=prefix,
                                                  keep_vars=keep_vars)
        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""

        # Load word_embeddings.
        if self.post_process and not self.pre_process and not self.untie_embeddings_and_output_weights:
            self.word_embeddings.load_state_dict(
                state_dict[self._word_embeddings_for_head_key], strict=strict)
        # Gather MoE states and move under language model
        moe_state_dict = {}
        for key in list(state_dict.keys()):
            if 'expert' in key and 'moe.gate.wg.weight' not in key:
                moe_state_dict[key] = state_dict.pop(key)
        if self._language_model_key in state_dict:
            state_dict = state_dict[self._language_model_key]
        if len(moe_state_dict) > 0:
            state_dict["moe_state_dict"] = moe_state_dict
        self.language_model.load_state_dict(state_dict, strict=strict)

    def universal_checkpoint_info(self):
        return UniversalCheckpointInfo(using_model_pipe=False).get()


def CrossEntropy(output, labels):
    labels, loss_mask = labels[0], labels[1]

    args = get_args()

    # [b s] => [s b]
    labels = labels.transpose(0, 1).contiguous()
    losses = tensor_parallel.vocab_parallel_cross_entropy(output.contiguous().float(), labels)
    # [s b] => [b, s]
    losses = losses.transpose(0, 1).contiguous()
    loss_mask = loss_mask.view(-1)
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
    return loss


class GPTModelPipe(PipelineModule,MegatronModule):
    """GPT-2 Language model."""

    def __init__(self,
                 config,
                 num_tokentypes=0,
                 parallel_output=True):
        args = get_args()
        self.parallel_output = parallel_output

        if config.init_method is None:
            config.init_method = init_method_normal(config.init_method_std)

        if config.output_layer_init_method is None:
            config.output_layer_init_method = scaled_init_method_normal(config.init_method_std,
                                                                        config.num_layers)

        self.specs = []

        def _to_float16(inputs):
            if args.fp16:
                return fp32_to_float16(inputs, lambda v: v.half())
            elif args.bf16:
                return fp32_to_float16(inputs, lambda v: v.bfloat16())
            else:
                return inputs

        self.specs.append(_to_float16)

        # Embedding layer
        if args.untie_embeddings_and_output_weights:
            self.specs.append(LayerSpec(EmbeddingPipe,
                                        args.hidden_size,
                                        args.padded_vocab_size,
                                        args.max_position_embeddings,
                                        args.hidden_dropout,
                                        config,
                                        num_tokentypes=num_tokentypes,
                                        embedding_weights_in_fp32=args.embedding_weights_in_fp32,))
        else:
            self.specs.append(TiedLayerSpec('embed',
                                            EmbeddingPipe,
                                            args.hidden_size,
                                            args.padded_vocab_size,
                                            args.max_position_embeddings,
                                            args.hidden_dropout,
                                            config,
                                            num_tokentypes=num_tokentypes,
                                            embedding_weights_in_fp32=args.embedding_weights_in_fp32,
                                            tied_weight_attr='word_embeddings_weight'))

        experts_per_layer = get_num_experts_per_layer(args.num_experts, args.num_layers, args.expert_interval)
        self.is_moe_model = any(n_experts > 1 for n_experts in experts_per_layer)

        # Currently PipelineEngine does not support more than 1 pipe and/or grad partitioned tensors that
        # require grads.
        # When using MoE, we have 2 tensors that are passed along pipeline stages and both require grads.
        # Therefore, verify that both pipe_partitioned / grad_partitioned are not enabled
        if self.is_moe_model and args.pipeline_model_parallel_size > 1 and args.tensor_model_parallel_size > 1:
            pipe_partitioned_enabled = args.deepspeed_config_dict.get('pipeline', {}).get('pipe_partitioned', False)
            grad_partitioned_enabled = args.deepspeed_config_dict.get('pipeline', {}).get('grad_partitioned', False)
            assert not pipe_partitioned_enabled and not grad_partitioned_enabled, \
                'Pipe and/or Grad partitioning are not supported for MoE model'

        for layer_idx in range(args.num_layers):
            self.specs.append(
                LayerSpec(ParallelTransformerLayerPipe,
                          config,
                          layer_number=layer_idx,
                          self_attn_mask_type=AttnMaskType.causal,
                          num_experts=experts_per_layer[layer_idx],
                          input_aggregated_moe_loss=(self.is_moe_model and layer_idx > 0),
                          return_aggregated_moe_loss=self.is_moe_model))

        # if model has experts, add a layer to get and cache the aggregated moe loss from the
        # last transformer layer
        if self.is_moe_model:
            self.specs.append(self._calculate_moe_loss)

        # Final layernorm after transformer layers
        if args.normalization == 'layernorm':
            self.specs.append(LayerSpec(LayerNorm,
                          args.hidden_size,
                          eps=args.layernorm_epsilon,
                          sequence_parallel=args.sequence_parallel))
        else:
            self.specs.append(LayerSpec(RMSNorm, args.hidden_size,
                                        args.layernorm_epsilon,
                                        sequence_parallel=args.sequence_parallel))

        def _logits_helper(embedding, lm_output):
            """A wrapper to massage inputs/outputs from pipeline. """
            return parallel_lm_logits(
                lm_output,
                embedding.word_embeddings_weight,
                self.parallel_output)
        if args.untie_embeddings_and_output_weights:
            self.specs.append(
                LayerSpec(LMHeadPipe, args.hidden_size, args.padded_vocab_size, config)
            )
        else:
            self.specs.append(
                TiedLayerSpec('embed',
                              EmbeddingPipe,
                              args.hidden_size,
                              args.padded_vocab_size,
                              args.max_position_embeddings,
                              args.hidden_dropout,
                              config,
                              num_tokentypes=num_tokentypes,
                              embedding_weights_in_fp32=args.embedding_weights_in_fp32,
                              forward_fn=_logits_helper,
                              tied_weight_attr='word_embeddings_weight')
            )

        # Convert to fp32 if needed
        if args.fp16 or args.bf16:
            self.specs.append(float16_to_fp32)

        # Cache losses
        self.moe_loss = None
        self.last_lm_loss = None    # detached, for display only
        self.last_moe_loss = None   # detached, for display only

        if args.checkpoint_activations:
            interval = args.checkpoint_num_layers
        elif args.recompute_granularity == "full" and args.recompute_method == 'uniform':
            # deepspeed's pipeline doesn't support the block recompute method
            interval = args.recompute_num_layers
        else:
            interval = 0

        from deepspeed.runtime.pipe.topology import PipeModelDataParallelTopology
        topo = PipeModelDataParallelTopology(num_pp=mpu.get_pipeline_model_parallel_world_size(),
                                             num_mp=mpu.get_tensor_model_parallel_world_size(),
                                             num_dp=mpu.get_data_parallel_world_size())

        super().__init__(layers=self.specs,
                         loss_fn=self.loss_func,
                         topology=topo,
                         activation_checkpoint_interval=interval,
                         partition_method='type:transformer')

    def _calculate_moe_loss(self, inputs):
        """ Calculate MoE auxiliary loss """
        assert isinstance(inputs, tuple) and len(inputs) == 2
        hidden, aggregated_moe_loss = inputs[0], inputs[1]
        args = get_args()
        self.moe_loss = aggregated_moe_loss * args.moe_loss_coeff
        return hidden

    def loss_func(self, output, labels):
        loss = CrossEntropy(output, labels)
        self.last_lm_loss = loss.clone().detach()
        if self.moe_loss is not None:
            loss += self.moe_loss
            self.last_moe_loss = self.moe_loss.clone().detach()
        return loss

    def universal_checkpoint_info(self):
        return UniversalCheckpointInfo(using_model_pipe=True).get()

    def get_additional_losses(self):
        if not self.is_moe_model:
            return None
        return OrderedDict({
            'lm loss': self.last_lm_loss,
            'moe loss': self.last_moe_loss
        })
