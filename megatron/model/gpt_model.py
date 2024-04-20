# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

"""GPT-2 model."""

import torch

from megatron import get_args
from megatron.core import tensor_parallel
from .module import MegatronModule

from .enums import AttnMaskType
from .language_model import parallel_lm_logits
from .language_model import get_language_model


def load_balancing_loss_func(gate_logits: torch.Tensor, num_experts: torch.Tensor = None, top_k=2) -> float:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.
    See Switch Transformer (https://arxiv.org/abs/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.
    Args:
        gate_logits (Union[`torch.Tensor`, Tuple[torch.Tensor]):
            Logits from the `gate`, should be a tuple of model.config.num_hidden_layers tensors of
            shape [batch_size X sequence_length, num_experts].
        num_experts (`int`, *optional*):
            Number of experts
    Returns:
        The auxiliary loss.
    """
    if gate_logits is None or not isinstance(gate_logits, list):
        return 0

    # if isinstance(gate_logits, list):
    compute_device = gate_logits[0].device
    concatenated_gate_logits = torch.cat([layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0)

    routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)
    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

    # num_expert = torch.tensor(8)を想定、レイヤーごとに異なる場合は未実装
    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

    # Compute the percentage of tokens routed to each experts
    tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

    # Compute the average probability of routing to these experts
    router_prob_per_expert = torch.mean(routing_weights, dim=0)

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts


def post_language_model_processing(
        lm_output, labels, logit_weights, parallel_output, fp16_lm_cross_entropy, all_router_logits=None
):

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
        if fp16_lm_cross_entropy:
            assert output.dtype == torch.half
            loss = tensor_parallel.vocab_parallel_cross_entropy(output, labels)
        else:
            loss = tensor_parallel.vocab_parallel_cross_entropy(output.float(), labels)

        if all_router_logits is not None and len(all_router_logits) > 0:
            compute_device = all_router_logits[0].device

            args = get_args()

            top_k = getattr(args, "num_experts_per_tok", 2)
            num_experts =  args.num_experts

            num_experts = torch.tensor(num_experts).to(compute_device)
            aux_loss = load_balancing_loss_func(all_router_logits, num_experts, top_k)

            loss += args.router_aux_loss_coef * aux_loss

        # [s b] => [b, s]
        loss = loss.transpose(0,1).contiguous()
        if all_router_logits is not None and len(all_router_logits) > 0:
            aux_loss = torch.broadcast_to(aux_loss, loss.shape)
            return torch.stack([loss, aux_loss], 0)
        return loss


class GPTModel(MegatronModule):
    """GPT-2 Language model."""

    def __init__(self,
                 config,
                 num_tokentypes=0,
                 parallel_output=True,
                 pre_process=True,
                 post_process=True):
        args = get_args()
        super().__init__(config=config, share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights)

        self.parallel_output = parallel_output
        self.pre_process = pre_process
        self.post_process = post_process
        self.fp16_lm_cross_entropy = args.fp16_lm_cross_entropy
        self.untie_embeddings_and_output_weights = args.untie_embeddings_and_output_weights
        self.output_router_logits = True if args.moe_type == "mixtral" else False
        self.router_aux_loss_coef = args.router_aux_loss_coef
        self.num_layers = args.num_layers
        self.num_experts = args.num_experts

        self.language_model, self._language_model_key = get_language_model(
            config=config,
            num_tokentypes=num_tokentypes,
            add_pooler=False,
            encoder_attn_mask_type=AttnMaskType.causal,
            pre_process=self.pre_process,
            post_process=self.post_process)
        
        if not args.untie_embeddings_and_output_weights:
            self.initialize_word_embeddings()

    def set_input_tensor(self, input_tensor):
        """See megatron.model.transformer.set_input_tensor()"""
        self.language_model.set_input_tensor(input_tensor)

    def forward(self, input_ids, position_ids, attention_mask,
                retriever_input_ids=None,
                retriever_position_ids=None,
                retriever_attn_mask=None,
                labels=None, tokentype_ids=None, inference_params=None):

        if self.output_router_logits:
            lm_output, all_router_logits = self.language_model(
                input_ids,
                position_ids,
                attention_mask,
                retriever_input_ids=retriever_input_ids,
                retriever_position_ids=retriever_position_ids,
                retriever_attn_mask=retriever_attn_mask,
                inference_params=inference_params
            )
        else:
            lm_output = self.language_model(
                input_ids,
                position_ids,
                attention_mask,
                retriever_input_ids=retriever_input_ids,
                retriever_position_ids=retriever_position_ids,
                retriever_attn_mask=retriever_attn_mask,
                inference_params=inference_params
            )

        if self.post_process:
            if self.output_router_logits:
                return post_language_model_processing(
                    lm_output, labels,
                    self.language_model.output_layer.weight if self.untie_embeddings_and_output_weights else self.shared_embedding_or_output_weight(),
                    self.parallel_output,
                    self.fp16_lm_cross_entropy,
                    all_router_logits
                )
            return post_language_model_processing(
                lm_output, labels,
                self.language_model.output_layer.weight if self.untie_embeddings_and_output_weights else self.shared_embedding_or_output_weight(),
                self.parallel_output,
                self.fp16_lm_cross_entropy)
        else:
            if self.output_router_logits:
                s, b, _ = lm_output.size()

                # all_router_logits: [(s, b, nb_experts), (s, b, nb_expetrs), ...] -> (s, b, nb_experts * nb_current_layers)
                all_router_logits = torch.cat(all_router_logits, dim=2)

                # total number of experts is self.num_experts * self.num_layers
                rest_num_layers = self.num_experts * self.num_layers - all_router_logits.size()[-1]
                rest_layers = torch.zeros([s, b, rest_num_layers], dtype=lm_output.dtype, device=torch.cuda.current_device())

                # (s, b, (hidden size + total number of experts))
                lm_output = torch.cat([lm_output, all_router_logits, rest_layers], dim=2)
            return lm_output

    def state_dict_for_save_checkpoint(self, prefix='', keep_vars=False):

        state_dict_ = {}
        state_dict_[self._language_model_key] \
            = self.language_model.state_dict_for_save_checkpoint(
                prefix=prefix, keep_vars=keep_vars)
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
        if self._language_model_key in state_dict:
            state_dict = state_dict[self._language_model_key]
        self.language_model.load_state_dict(state_dict, strict=strict)
