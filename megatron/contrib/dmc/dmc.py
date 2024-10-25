# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import math

import torch
import torch.nn.functional as F
from einops import rearrange
from flash_attn import flash_attn_with_kvcache
from torch.nn.attention import SDPBackend, sdpa_kernel

from megatron.core import tensor_parallel
from megatron.core.models.common.embeddings.rotary_pos_embedding import apply_rotary_pos_emb
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training import get_args

from megatron.contrib.dmc.inference import DMCPagedKVCache, InferencePoolingParamsTriton
from megatron.contrib.dmc.rotary import apply_rotary as apply_rotary_dmc
from megatron.contrib.dmc.triton import update_inference_params_triton_faster

MASK_CONST = -50000.0
layer_decisions = []


def add_dmc_layer(transformer_layer_spec):
    transformer_layer_spec.submodules.self_attention.module = DMCAttention


def get_prior_and_cr():
    if getattr(get_args(), 'curr_iteration', None) is None:
        return 1.0, 0.0

    if get_args().dmc_finetune:
        return get_args().dmc_cr, 1.0 - (1.0 / get_args().dmc_cr)

    curr_cr = 1.0 + (get_args().dmc_cr - 1.0) * min(
        1.0, (get_args().curr_iteration / get_args().train_iters)
    )
    curr_prior = 1.0 - (1.0 / curr_cr)
    return curr_cr, curr_prior


def get_decisions(probs):
    decisions = (probs > 0.5).to(probs.dtype)
    return decisions - probs.detach() + probs


def get_dmc_loss():
    global layer_decisions
    decisions = torch.stack(layer_decisions, dim=0)
    decisions = tensor_parallel.gather_from_tensor_model_parallel_region(decisions).float()
    layer_decisions = []

    tgt_cr, tgt_prior = get_prior_and_cr()
    dmc_loss = (tgt_prior - decisions.mean(dim=0)).clamp(min=0).mean()

    curr_cr = 1 / (1 - decisions.mean().clone().detach().requires_grad_(False))

    return dmc_loss, curr_cr, tgt_cr


class DMCAttention(SelfAttention):
    """Self-attention layer class.

    Self-attention layer takes input with size [s, b, h] and returns output of the same size.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: SelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type=AttnMaskType.padding,
    ):
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
        )

        self.pooler = DynamicPooler()

        assert (
            self.config.pipeline_model_parallel_size == 1
        ), "Pipeline model parallelism is not supported with DMC!"

        assert (
            self.config.context_parallel_size == 1
        ), "Context parallelism is not supported with DMC!"

        assert (
            self.config.window_size is None
        ), "Sliding Window Attention is not supported with DMC!"

        if get_args().dmc_is_stage_one:
            self.forward = self.forward_decay_decision_neuron

    def _run_dmc_for_inference(
        self, seq_start, inference_params, query, key, value, rotary_pos_emb
    ):
        """
        Saves the generated key and value tensors to the end of the buffers in inference_params.
        Returns the full size keys and values from the provided inference_params, as well as
        adjusted rotary_pos_emb.

        Returns a tuple: (key, value, rotary_pos_emb)

        """
        seq_end = seq_start + 1

        assert query.size(0) == key.size(0) == 1
        B = key.size(1)
        KV_H = self.num_attention_heads_per_partition
        Q_H = self.num_query_groups_per_partition
        D = self.hidden_size_per_attention_head

        args = get_args()

        is_prompt_phase = seq_start > inference_params.sequence_len_offset

        if self.layer_number == 1:

            # Allocate once for all layers to execute less kernels
            if 1 not in inference_params.key_value_memory_dict:
                inference_params.key_value_memory_dict[1] = DMCPagedKVCache(
                    inference_params.max_batch_size,
                    KV_H,
                    D,
                    args.num_layers,
                    args.dmc_paged_block_size,
                    inference_params.max_sequence_length,
                    args.dmc_paged_cache_size,
                )
            elif not is_prompt_phase:
                inference_params.key_value_memory_dict[1].allocate_blocks(inference_params.dmc)

        if is_prompt_phase:
            inference_params.key_value_memory_dict[1].allocate_blocks(
                inference_params.dmc, layer_idx=self.layer_number - 1
            )

        inference_params.dmc = getattr(inference_params, 'dmc', {})
        if self.layer_number not in inference_params.dmc:
            inference_params.dmc[self.layer_number] = InferencePoolingParamsTriton(
                B, KV_H, D, args.dmc_window_size
            )

        dmc_state = inference_params.dmc[self.layer_number]
        query, key, value = query.contiguous(), key.contiguous(), value.contiguous()

        update_inference_params_triton_faster(
            query,
            key,
            value,
            dmc_state.kv_win,
            dmc_state.w_win,
            dmc_state.lens,
            win_ptr=dmc_state.win_ptr,
            win_sz=dmc_state.win_sz,
            seq_start=seq_start,
            extra_val=args.dmc_init_val,
            eps=torch.finfo(torch.bfloat16).eps,
        )
        k_last, v_last = key, value

        if dmc_state.win_sz > 0:
            dmc_state.win_ptr = (dmc_state.win_ptr + 1) % dmc_state.win_sz
        else:
            dmc_state.win_ptr = 1

        # MCore
        if rotary_pos_emb is not None:
            pos_emb = rotary_pos_emb[0][seq_start:seq_end].to(key.dtype)
            emb_for_cos, emb_for_sin = torch.chunk(pos_emb.squeeze(2).squeeze(1), 2, dim=1)

        q = query.view(B * Q_H, 1, 1, D)
        k_last = k_last.view(B * KV_H, 1, 1, D)
        v_last = v_last.view(B * KV_H, 1, 1, D)

        if rotary_pos_emb is not None:
            # Modified rotary kernel, applies .cos() and .sin() to embeddings and
            # sets the last neuron to 0
            apply_rotary_dmc(q, emb_for_cos, emb_for_sin, inplace=True)
            apply_rotary_dmc(k_last, emb_for_cos, emb_for_sin, inplace=True)

        return q, k_last, v_last

    def forward(
        self,
        hidden_states,
        attention_mask,
        key_value_states=None,
        inference_params=None,
        rotary_pos_emb=None,
        packed_seq_params=None,
    ):
        # hidden_states: [sq, b, h]

        # For self attention we just duplicate the rotary_pos_emb if it isn't already
        if rotary_pos_emb is not None and not isinstance(rotary_pos_emb, tuple):
            rotary_pos_emb = (rotary_pos_emb,) * 2

        # Get the query, key and value tensors based on the type of attention -
        # self or cross attn.
        query, key, value = self.get_query_key_value_tensors(hidden_states, key_value_states)

        if inference_params is not None:
            assert packed_seq_params is None

            out = []

            for i in range(query.size(0)):
                q, k_last, v_last = self._run_dmc_for_inference(
                    inference_params.sequence_len_offset + i,
                    inference_params,
                    query[i: i + 1],
                    key[i: i + 1],
                    value[i: i + 1],
                    rotary_pos_emb,
                )
                dmc_state = inference_params.dmc[self.layer_number]
                paged_cache = inference_params.key_value_memory_dict[1]

                with tensor_parallel.get_cuda_rng_tracker().fork():
                    out.append(
                        flash_attn_with_kvcache(
                            q,
                            paged_cache.k_cache,
                            paged_cache.v_cache,
                            k=k_last,
                            v=v_last,
                            cache_seqlens=dmc_state.lens[0].view(-1),
                            causal=True,
                            softmax_scale=1.0 / self.core_attention.fused_attention.norm_factor,
                            block_table=paged_cache.get_block_table(self.layer_number - 1),
                        )
                    )

            core_attn_out = torch.cat(out, dim=1) if len(out) > 1 else out[0]
            core_attn_out = rearrange(
                core_attn_out, '(b h) s q d -> s b (h q d)', b=query.size(1), h=query.size(2)
            ).contiguous()
            output, bias = self.linear_proj(core_attn_out)
            return output, bias

        # Adjust key, value, and rotary_pos_emb for inference
        key, value, rotary_pos_emb, attn_mask_type = self._adjust_key_value_for_inference(
            inference_params, key, value, rotary_pos_emb
        )

        if packed_seq_params is not None:
            query = query.squeeze(1)
            key = key.squeeze(1)
            value = value.squeeze(1)

        query, key, value, attention_bias = self.pooler.downsample(
            query, key, value, optimized_attn_bias=(self.checkpoint_core_attention and self.training)
        )

        # relative positional embedding (rotary embedding)
        if rotary_pos_emb is not None:
            q_pos_emb, k_pos_emb = rotary_pos_emb

            if packed_seq_params is not None:
                cu_seqlens_q = packed_seq_params.cu_seqlens_q
                cu_seqlens_kv = packed_seq_params.cu_seqlens_kv
            else:
                cu_seqlens_q = cu_seqlens_kv = None

            foo, bar = query[..., -1].clone(), key[..., -1].clone()
            query[..., -1] = 0
            key[..., -1] = 0

            query = apply_rotary_pos_emb(
                query,
                q_pos_emb,
                config=self.config,
                cu_seqlens=cu_seqlens_q,
            ).contiguous()
            key = apply_rotary_pos_emb(
                key,
                k_pos_emb,
                config=self.config,
                cu_seqlens=cu_seqlens_kv,
            ).contiguous()

            query[..., -1], key[..., -1] = foo, bar

        q, k, v = [rearrange(x, 's b h d -> b h s d').contiguous() for x in (query, key, value)]
        q = q / math.sqrt(q.size(-1))

        if self.checkpoint_core_attention and self.training:

            def custom_forward(*inputs):
                q, k, v, attention_bias = inputs

                attention_bias = torch.diag_embed(
                    -attention_bias.permute(1, 2, 0).repeat_interleave(
                        q.size(1) // k.size(1), dim=1
                    )
                )
                with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
                    core_attn_out = F.scaled_dot_product_attention(
                        q, k, v,
                        attn_mask=attention_bias,
                        is_causal=True,
                        scale=1.0,
                    )
                return core_attn_out

            core_attn_out = tensor_parallel.checkpoint(
                custom_forward,
                False,
                q, k, v,
                attention_bias,
            )
        else:
            with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
                core_attn_out = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=attention_bias,
                    is_causal=True,
                    scale=1.0,
                )

        core_attn_out = rearrange(core_attn_out, 'b h s d -> s b (h d)').contiguous()

        if packed_seq_params is not None:
            # reshape to same output shape as unpacked case
            # (t, np, hn) -> (t, b=1, h=np*hn)
            # t is the pack size = sum (sq_i)
            # note that batch is a dummy dimension in the packed case
            core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)

        # Output [sq, b, h]
        output, bias = self.linear_proj(core_attn_out)

        return output, bias

    def forward_decay_decision_neuron(
        self,
        hidden_states,
        attention_mask,
        key_value_states=None,
        inference_params=None,
        rotary_pos_emb=None,
        packed_seq_params=None,
    ):
        """Attention forward without DMC, slowly zeros out the \alpha neuron"""

        # hidden_states: [sq, b, h]

        if rotary_pos_emb is not None and not isinstance(rotary_pos_emb, tuple):
            rotary_pos_emb = (rotary_pos_emb,) * 2

        query, key, value = self.get_query_key_value_tensors(hidden_states, key_value_states)

        key, value, rotary_pos_emb, attn_mask_type = self._adjust_key_value_for_inference(
            inference_params, key, value, rotary_pos_emb
        )

        if packed_seq_params is not None:
            query = query.squeeze(1)
            key = key.squeeze(1)
            value = value.squeeze(1)

        if rotary_pos_emb is not None:
            q_pos_emb, k_pos_emb = rotary_pos_emb

            if packed_seq_params is not None:
                cu_seqlens_q = packed_seq_params.cu_seqlens_q
                cu_seqlens_kv = packed_seq_params.cu_seqlens_kv
            else:
                cu_seqlens_q = cu_seqlens_kv = None

        mult = max(0, 1 - (get_args().curr_iteration / (0.8 * get_args().train_iters)))

        # [seq_len, batch_size, heads, hidden_dim]
        query = query.contiguous()
        key = key.contiguous()
        query[..., -1] *= mult
        key[..., -1] *= mult

        if rotary_pos_emb is not None:
            query = apply_rotary_pos_emb(
                query,
                q_pos_emb,
                config=self.config,
                cu_seqlens=cu_seqlens_q,
            ).contiguous()
            key = apply_rotary_pos_emb(
                key,
                k_pos_emb,
                config=self.config,
                cu_seqlens=cu_seqlens_kv,
            ).contiguous()

        # After using rotary position embeddings (rotation of pairs of q),
        # the last neuron's values can be != 0, so we set it to 0 once again
        query[..., -1] *= mult
        key[..., -1] *= mult

        if self.checkpoint_core_attention and self.training:
            core_attn_out = self._checkpointed_attention_forward(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type,
                packed_seq_params=packed_seq_params,
            )
        else:
            core_attn_out = self.core_attention(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type,
                packed_seq_params=packed_seq_params,
            )

        if packed_seq_params is not None:
            core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)

        output, bias = self.linear_proj(core_attn_out)

        return output, bias


class DynamicPooler(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.init_val = get_args().dmc_init_val
        self.window_size = get_args().dmc_window_size
        self.dmc_temp = get_args().dmc_temp

    def downsample(self, queries, keys, values, optimized_attn_bias):
        """
        Args:
            keys: L x B x H x D
            values: L x B x H x D
        Output:
            keys: L x B x H x D
            values: L x B x H x D
        """

        L, B, KV_H, D = keys.size()
        Q_H = queries.size(2)
        keys, values = keys.reshape(L, B * KV_H, D), values.reshape(L, B * KV_H, D)

        # Extract pooling logits for each head
        router_logits_nonoise = keys[:, :, -1].clone() - self.init_val  # L x B * H

        # Get pooling probabilities with noise
        if self.training:
            router_logits = self.add_noise(router_logits_nonoise)
            router_probs = torch.sigmoid(router_logits)
        else:
            router_logits = router_logits_nonoise
            router_probs = get_decisions(torch.sigmoid(router_logits))

        # Pool keys and values
        weight_logits = queries[:, :, :, -1].clone()
        weight_logits = (
            weight_logits.reshape(L, B, KV_H, Q_H // KV_H).mean(dim=-1).reshape(L, B * KV_H)
        )
        weights = torch.sigmoid(weight_logits + self.init_val) + torch.finfo(queries.dtype).eps

        pooled_kv = poolsum(
            x=torch.cat([keys, values], dim=-1),
            b=router_probs[1:],
            w=weights,
            ws=self.window_size,
            batch_first=False,
        )

        keys, values = pooled_kv[:, :, :D].reshape(L, B, KV_H, D), pooled_kv[:, :, D:].reshape(
            L, B, KV_H, D
        )

        # Masking for attention
        router_logits, router_probs = [x.reshape(L, B, KV_H) for x in [router_logits, router_probs]]

        attn_score_mask = self.get_attn_score_mask(router_logits, router_probs)
        attn_score_mask = F.pad(attn_score_mask, (0, 0, 0, 0, -1, 1), value=0)

        queries = queries.contiguous()
        keys = keys.contiguous()

        # The former is more efficient, but something weird may happen during eval
        # if we add and substract a big number (-1e5) from the diagonal
        if self.training:
            # Here we embed the mask into the queries and keys and pass an extra mask to attn-func
            # which un-masks the diagonal
            queries[:, :, :, -1] = math.sqrt(D)
            keys[:, :, :, -1] = attn_score_mask
        else:
            # Here we create a full mask with zeros on the diagonal and pass it to attn-func
            queries[:, :, :, -1] = 0
            keys[:, :, :, -1] = 0

        # For loss calculation
        global layer_decisions
        layer_decisions.append(get_decisions(router_probs).mean(dim=(0, 2)))

        if Q_H // KV_H != 1:
            # expand the key_layer and value_layer [sk, b, ng, hn] -> [sk, b, np, hn]
            keys = keys.repeat_interleave(Q_H // KV_H, dim=2)
            values = values.repeat_interleave(Q_H // KV_H, dim=2)

        if optimized_attn_bias:
            if Q_H // KV_H != 1:
                attn_score_mask = attn_score_mask.repeat_interleave(Q_H // KV_H, dim=2)
            return queries, keys, values, attn_score_mask

        if self.training:
            expanded_attn_score_mask = torch.diag_embed(
                -attn_score_mask.permute(1, 2, 0).repeat_interleave(Q_H // KV_H, dim=1)
            )
        else:
            expanded_attn_score_mask = attn_score_mask.permute(1, 2, 0)[:, :, None, :].repeat(
                1, 1, L, 1
            )

            expanded_attn_score_mask = torch.diagonal_scatter(
                input=expanded_attn_score_mask,
                src=torch.zeros_like(expanded_attn_score_mask[..., 0]),
                dim1=2,
                dim2=3,
            )

            # NOTE: For inference, attention_bias has shape [b, nh, sk, sk]
            if Q_H // KV_H != 1:
                expanded_attn_score_mask = expanded_attn_score_mask.repeat_interleave(
                    Q_H // KV_H,
                    dim=1
                )

        return queries, keys, values, expanded_attn_score_mask

    def get_attn_score_mask(self, router_logits, router_probs):
        if self.training:
            # log(1 - sigmoid(router_logits))
            attention_scores_mask = torch.nn.functional.logsigmoid(-router_logits)
        else:
            attention_scores_mask = router_probs * MASK_CONST

        return attention_scores_mask

    def add_noise(self, logits):
        temp = self.dmc_temp

        # When we sample in bf16 the range is (-8, 8), for fp32/64 it's (-16, 16)
        # Also, we init the dist here, because otherwise we don't know the specific device
        dist = torch.distributions.gumbel.Gumbel(
            loc=torch.tensor(0.0, dtype=torch.float32, device=logits.device),
            scale=torch.tensor(1.0, dtype=torch.float32, device=logits.device),
            validate_args=None,
        )
        sample1, sample2 = [dist.sample(sample_shape=logits.size()).bfloat16() for _ in range(2)]
        logits = (logits + sample1 - sample2) / temp

        return logits


@torch.compile
def _poolsum(x, b, w=None, ws: int = 12, batch_first: bool = True):
    """Apply continuous pooling with fractional boundaries within a window.

    Tokens up to (i-1) are added to the i-th token with weight b[i] and
    re-weighted with c:

        c[0] = 1
        x_acc[0] = x[0]

        x_acc[i] = x_acc[i-1] * b[i] + x[i]
        c[i] = c[i-1] * b[i] + 1

        x_pooled[i] = x_acc[i] / c[i]

    If weights w[] are provided, the formula are:

        c[0] = w[0]
        x_acc[0] = x[0]

        x_acc[i] = x_acc[i-1] * b[i] + x[i] * w[i]
        c[i] = c[i-1] * b[i] + w[i]

    Args:
        x: torch.Tensor, input tensor (B, T, D)
        b: torch.Tensor, boundaries (B, T-1) with values from [0, 1]
        w: (optional) torch.Tensor, weights for element inside a group (B, T)
        with values from (0, 1]
        ws: int, window size
        batch_first: bool, Expect B dim before T in input `x` and `b`
    """
    x = x.contiguous()
    b = b.contiguous()

    if batch_first:
        B, T, D = x.size()

        # Do not multiply x by w here because it's costly. Do it when x_acc is calculated.
        # if w is not None:
        #     x = x * w.unsqueeze(2)

        x_pad = F.pad(x, (0, 0, ws - 1, 0))
        x_pad = torch.as_strided(x_pad, (B, ws, T, D), (x_pad.size(1) * D, D, D, 1))

        b_pad = F.pad(b, (ws - 1, 0))
        b_pad = torch.as_strided(b_pad, (B, ws - 1, b.size(1) + 1), (b_pad.size(1), 1, 1))
        b_pad = F.pad(b_pad, (0, 0, 0, 1), value=1)
        coeffs = torch.flip(torch.cumprod(torch.flip(b_pad, dims=(1,)), dim=1), dims=(1,))

        if w is None:
            x_acc = (coeffs.unsqueeze(-1) * x_pad).sum(1)
            c = coeffs.sum(1).unsqueeze(-1)
        else:
            w_pad = F.pad(w, (ws - 1, 0), value=1)
            w_pad = torch.as_strided(w_pad, (B, ws, w.size(1)), (w_pad.size(1), 1, 1))
            x_acc = (w_pad.unsqueeze(-1) * coeffs.unsqueeze(-1) * x_pad).sum(1)
            c = (coeffs * w_pad).sum(1).unsqueeze(-1)

    else:
        T, B, D = x.size()

        x_pad = F.pad(x, (0, 0, 0, 0, ws - 1, 0))
        x_pad = torch.as_strided(x_pad, (ws, T, B, D), (B * D, B * D, D, 1))

        b_pad = F.pad(b, (0, 0, ws - 1, 0))
        b_pad = torch.as_strided(b_pad, (ws - 1, b.size(0) + 1, B), (B, B, 1))
        b_pad = F.pad(b_pad, (0, 0, 0, 0, 0, 1), value=1)
        coeffs = torch.flip(torch.cumprod(torch.flip(b_pad, dims=(0,)), dim=0), dims=(0,))

        if w is None:
            x_acc = (coeffs.unsqueeze(-1) * x_pad).sum(0)
            c = coeffs.sum(0).unsqueeze(-1)
        else:
            w_pad = F.pad(w, (0, 0, ws - 1, 0), value=1)
            w_pad = torch.as_strided(w_pad, (ws, w.size(0), B), (B, B, 1))
            x_acc = (w_pad.unsqueeze(-1) * coeffs.unsqueeze(-1) * x_pad).sum(0)
            c = (coeffs * w_pad).sum(0).unsqueeze(-1)

    return x_acc / c


# WAR: torch.compile doesn't fully support BF16 in poolsum
def poolsum(x, b, w=None, ws=12, batch_first=True):
    return _poolsum(
        x.float(),
        b.float(),
        w.float() if w is not None else w,
        ws=ws,
        batch_first=batch_first
    ).to(x.dtype)
