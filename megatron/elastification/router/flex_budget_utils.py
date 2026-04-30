# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from typing import Dict, List, Optional, Tuple, Union

import torch


def get_num_parameters(
    hybrid_pattern: str = None,
    mamba_num_heads: int = 0,
    mamba_d_head: int = 0,
    mamba_d_state: int = 0,
    num_attention_heads: int = 0,
    num_query_groups: int = 0,
    ffn_hidden_size: int = 0,
    hidden_size: int = 0,
    kv_channels: int = 0,
    vocab_size: int = 0,
    tied_vocab: bool = False,
    num_experts: int = 0,
    shared_expert_intermediate_size: int = 0,
    moe_router_topk: int = 0,
) -> int:

    norm_multiplier = 1

    embedding = vocab_size * hidden_size
    final_layernorm = hidden_size * 1
    output_layer = 0 if tied_vocab else (vocab_size * hidden_size)
    if isinstance(ffn_hidden_size, int):
        flex_hetero_ffn = False
    else:
        flex_hetero_ffn = ffn_hidden_size.shape[0] != 1

    if isinstance(mamba_num_heads, int):
        flex_hetero_mamba = False
    else:
        flex_hetero_mamba = mamba_num_heads.shape[0] != 1

    if isinstance(num_attention_heads, int):
        flex_hetero_head = False
    else:
        flex_hetero_head = num_attention_heads.shape[0] != 1

    if isinstance(num_experts, int):
        flex_hetero_moe_expert = False
    else:
        flex_hetero_moe_expert = num_experts.shape[0] != 1

    # MOE

    if flex_hetero_ffn or flex_hetero_moe_expert:
        if flex_hetero_ffn and not flex_hetero_moe_expert:
            num_experts = [num_experts] * ffn_hidden_size.shape[0]
        if flex_hetero_moe_expert and not flex_hetero_ffn:
            ffn_hidden_size = [ffn_hidden_size] * num_experts.shape[0]

        moe_all = []
        moe_active = []
        for i in range(len(num_experts)):
            pre_moe_ln = norm_multiplier * hidden_size
            linear_fc1 = ffn_hidden_size[i] * (
                hidden_size * num_experts[i] + shared_expert_intermediate_size
            )
            linear_fc2 = ffn_hidden_size[i] * (
                hidden_size * num_experts[i] + shared_expert_intermediate_size
            )
            linear_fc1_active = ffn_hidden_size[i] * (
                hidden_size * moe_router_topk + shared_expert_intermediate_size
            )
            linear_fc2_active = ffn_hidden_size[i] * (
                hidden_size * moe_router_topk + shared_expert_intermediate_size
            )
            moe_all.append(pre_moe_ln + linear_fc1 + linear_fc2)
            moe_active.append(pre_moe_ln + linear_fc1_active + linear_fc2_active)
    else:
        pre_mlp_ln = norm_multiplier * hidden_size
        linear_fc1 = ffn_hidden_size * (hidden_size * num_experts + shared_expert_intermediate_size)
        linear_fc2 = ffn_hidden_size * (hidden_size * num_experts + shared_expert_intermediate_size)
        linear_fc1_active = ffn_hidden_size * (
            hidden_size * moe_router_topk + shared_expert_intermediate_size
        )
        linear_fc2_active = ffn_hidden_size * (
            hidden_size * moe_router_topk + shared_expert_intermediate_size
        )
        moe_all = pre_mlp_ln + linear_fc1 + linear_fc2
        moe_active = pre_mlp_ln + linear_fc1_active + linear_fc2_active

    # ATT
    if flex_hetero_head:
        att = []
        for i in range(num_attention_heads.shape[0]):
            input_ln = norm_multiplier * hidden_size
            linear_proj = num_attention_heads[i] * kv_channels * hidden_size
            linear_qkv = (num_attention_heads[i] + 2 * num_query_groups) * kv_channels * hidden_size
            att.append(input_ln + linear_proj + linear_qkv)
    else:
        input_ln = norm_multiplier * hidden_size
        linear_proj = num_attention_heads * kv_channels * hidden_size
        linear_qkv = (num_attention_heads + 2 * num_query_groups) * kv_channels * hidden_size
        att = input_ln + linear_proj + linear_qkv

    # Mamba
    def mamba_params(mamba_nheads):
        d_inner = mamba_nheads * mamba_d_head
        ngroups = 8

        def get_conv_params(kernel_size, stride):
            cdim = d_inner + 2 * ngroups * mamba_d_state
            cbias = cdim
            cweight = cdim * stride * kernel_size
            return cbias + cweight

        mamba_dt_bias = mamba_nheads
        mamba_A_log = mamba_nheads
        # self.d_inner_local if self.D_has_hdim else self.nheads_local,
        mamba_D = mamba_nheads
        mamba_input_ln = norm_multiplier * hidden_size
        mamba_in_proj = hidden_size * (d_inner * 2 + 2 * ngroups * mamba_d_state + mamba_nheads)
        mamba_conv = get_conv_params(4, 1)
        mamba_norm = d_inner
        mamba_out_proj = d_inner * hidden_size
        return (
            mamba_dt_bias
            + mamba_A_log
            + mamba_D
            + mamba_input_ln
            + mamba_in_proj
            + mamba_conv
            + mamba_norm
            + mamba_out_proj
        )

    all_params = 0
    active_params = 0
    for i, c in enumerate(hybrid_pattern):

        if c == 'M':
            if flex_hetero_mamba:
                mamba_idx = hybrid_pattern[: i + 1].count('M') - 1
                all_params += mamba_params(mamba_num_heads[mamba_idx])
                active_params += mamba_params(mamba_num_heads[mamba_idx])
            else:
                all_params += mamba_params(mamba_num_heads)
                active_params += mamba_params(mamba_num_heads)
        elif c == '*':
            if flex_hetero_head:
                head_idx = hybrid_pattern[: i + 1].count('*') - 1
                all_params += att[head_idx]
                active_params += att[head_idx]
            else:
                all_params += att
                active_params += att
        elif c == 'E':
            if flex_hetero_ffn or flex_hetero_moe_expert:
                # Count how many 'E' characters appear before and including layer i
                moe_idx = hybrid_pattern[: i + 1].count('E') - 1
                all_params += moe_all[moe_idx]
                active_params += moe_active[moe_idx]
            else:
                all_params += moe_all
                active_params += moe_active
        elif c == '|':
            pass
        else:
            raise RuntimeError(f'Unknown layer type: {c}')

    return (
        embedding + all_params + final_layernorm + output_layer,
        embedding + active_params + final_layernorm + output_layer,
    )


def get_kv_cache_size(
    hybrid_pattern: str = None,
    num_attention_heads=None,
    num_query_groups=None,
    kv_channels=None,
    mem_infer_seq_len: int = 0,
    mem_batch_size: int = 0,
) -> Union[int, torch.Tensor]:

    if isinstance(num_attention_heads, int):
        flex_hetero_head = False
    else:
        flex_hetero_head = num_attention_heads.shape[0] != 1

    if flex_hetero_head:
        kv_cache_size = 0
        head_idx = 0

        for c in hybrid_pattern:
            if c == '*':
                current_heads = (
                    num_attention_heads[head_idx] if flex_hetero_head else num_attention_heads
                )

                kv_cache_size_per_layer = (
                    2.0
                    * mem_batch_size
                    * mem_infer_seq_len
                    * num_query_groups
                    * current_heads
                    * kv_channels
                    / current_heads.detach().item()
                )
                kv_cache_size += kv_cache_size_per_layer
                head_idx += 1

    else:
        num_attention_layers = hybrid_pattern.count('*')
        divider = (
            num_attention_heads.detach().item()
            if isinstance(num_attention_heads, torch.Tensor)
            else num_attention_heads
        )
        kv_cache_size = (
            2.0
            * mem_batch_size
            * mem_infer_seq_len
            * num_query_groups
            * num_attention_heads
            * kv_channels
            * num_attention_layers
            / divider
        )

    return kv_cache_size


def get_mamba_ssm_cache_size(
    hybrid_pattern: str = None,
    mamba_num_heads: int = 0,
    mamba_d_head: int = 0,
    mamba_d_state: int = 0,
    mem_batch_size: int = 0,
) -> int:

    if isinstance(mamba_num_heads, int):
        flex_hetero_mamba = False
    else:
        flex_hetero_mamba = mamba_num_heads.shape[0] != 1

    if flex_hetero_mamba:
        ssm_cache_size = 0
        mamba_idx = 0
        for c in hybrid_pattern:
            if c == 'M':
                current_mamba_num_heads = mamba_num_heads[mamba_idx]
                ssm_cache_size += (
                    mem_batch_size * current_mamba_num_heads * mamba_d_head * mamba_d_state
                )
                mamba_idx += 1

    else:
        num_mamba_layers = hybrid_pattern.count('M')
        ssm_cache_size = (
            mem_batch_size * mamba_num_heads * mamba_d_head * mamba_d_state * num_mamba_layers
        )

    return ssm_cache_size


def get_max_buffer_size(
    hybrid_pattern: str = None,
    moe_num_experts: int = 0,
    shared_expert_intermediate_size: int = 0,
    ffn_hidden_size: int = 0,
    moe_router_topk: int = 0,
    mem_batch_size: int = 0,
    prefill_chunk_size: int = 0,
) -> int:

    if isinstance(moe_num_experts, int) or moe_num_experts.shape[0] == 1:
        moe_num_experts = (
            torch.tensor([moe_num_experts] * hybrid_pattern.count('E'))
            .to(torch.cuda.current_device())
            .float()
        )

    if isinstance(ffn_hidden_size, int) or ffn_hidden_size.shape[0] == 1:
        ffn_hidden_size = (
            torch.tensor([ffn_hidden_size] * hybrid_pattern.count('E'))
            .to(torch.cuda.current_device())
            .float()
        )

    max_buffer_list = []
    moe_idx = 0
    for char in hybrid_pattern:
        if char == 'E':
            current_moe_num_experts = moe_num_experts[moe_idx]
            current_ffn_hidden_size = ffn_hidden_size[moe_idx]
            max_buffer_list.append(
                shared_expert_intermediate_size + current_ffn_hidden_size * moe_router_topk
            )
            moe_idx += 1

    max_buffer = torch.stack(max_buffer_list)
    max_buffer_softmax = torch.nn.functional.softmax(max_buffer, dim=0)
    max_buffer = (max_buffer_softmax * max_buffer).sum().unsqueeze(0)
    max_buffer *= mem_batch_size * prefill_chunk_size

    return max_buffer


def get_memory_footprint(
    hybrid_pattern: str = None,
    mamba_num_heads: int = 0,
    mamba_d_head: int = 80,
    mamba_d_state: int = 128,
    num_attention_heads: int = 0,
    num_query_groups: int = 8,
    ffn_hidden_size: int = 0,
    hidden_size: int = 0,
    kv_channels: int = 128,
    vocab_size: int = 131072,
    tied_vocab: bool = False,
    mem_infer_seq_len: int = 131072,
    mem_batch_size: int = 1,
    prefill_chunk_size: int = 16384,
    moe_num_experts: int = 0,
    shared_expert_intermediate_size: int = 0,
    moe_router_topk: int = 0,
    memory_config=None,
):
    """
    Returns total inference memory footprint in GB.

    Parameters
    ----------
    memory_config : MemoryConfig, optional
        Bytes-per-element values and param budget target.  When None, defaults
        to BF16 for all components (bpe=2).  Pass a MemoryConfig built via
        ``load_memory_config(args)`` to select a quantisation profile.
    """
    from megatron.elastification.memory_config import MemoryConfig

    if memory_config is None:
        memory_config = MemoryConfig()  # BF16 defaults

    # Select all-param or active-param count based on param_budget_target
    param_idx = 1 if memory_config.param_budget_target == "active" else 0

    mem_params = (
        memory_config.bpe_params
        * get_num_parameters(
            hybrid_pattern=hybrid_pattern,
            mamba_num_heads=mamba_num_heads,
            mamba_d_head=mamba_d_head,
            mamba_d_state=mamba_d_state,
            num_attention_heads=num_attention_heads,
            num_query_groups=num_query_groups,
            ffn_hidden_size=ffn_hidden_size,
            hidden_size=hidden_size,
            kv_channels=kv_channels,
            vocab_size=vocab_size,
            tied_vocab=tied_vocab,
            num_experts=moe_num_experts,
            shared_expert_intermediate_size=shared_expert_intermediate_size,
            moe_router_topk=moe_router_topk,
        )[param_idx]
    )

    mem_kv_cache = memory_config.bpe_kv_cache * get_kv_cache_size(
        hybrid_pattern=hybrid_pattern,
        num_attention_heads=num_attention_heads,
        num_query_groups=num_query_groups,
        kv_channels=kv_channels,
        mem_infer_seq_len=mem_infer_seq_len,
        mem_batch_size=mem_batch_size,
    )

    mem_max_buffer = memory_config.bpe_max_buffer * get_max_buffer_size(
        hybrid_pattern=hybrid_pattern,
        moe_num_experts=moe_num_experts,
        shared_expert_intermediate_size=shared_expert_intermediate_size,
        ffn_hidden_size=ffn_hidden_size,
        moe_router_topk=moe_router_topk,
        mem_batch_size=mem_batch_size,
        prefill_chunk_size=prefill_chunk_size,
    )

    mem_mamba_ssm_cache = memory_config.bpe_ssm_cache * get_mamba_ssm_cache_size(
        hybrid_pattern=hybrid_pattern,
        mamba_num_heads=mamba_num_heads,
        mamba_d_head=mamba_d_head,
        mamba_d_state=mamba_d_state,
        mem_batch_size=mem_batch_size,
    )
    return (mem_params + mem_kv_cache + mem_max_buffer + mem_mamba_ssm_cache) / 1024 / 1024 / 1024
