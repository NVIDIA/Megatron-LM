# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from typing import Sequence

import torch

from megatron.core.models.hybrid import ArchitectureEntry, MTPSplit, PipelineSplit
from megatron.core.ssm.mamba_layer_config import MambaLayerConfig
from megatron.core.transformer.attention_layer_config import AttentionLayerConfig
from megatron.core.transformer.moe.moe_layer_config import MoELayerConfig
from megatron.core.transformer.transformer_config import TransformerConfig

_FLEXTRON_STRUCTURE_FIELDS = {
    MambaLayerConfig: (
        "hidden_size",
        "mamba_num_heads",
        "mamba_head_dim",
        "mamba_state_dim",
        "mamba_num_groups",
    ),
    AttentionLayerConfig: ("hidden_size", "num_attention_heads", "num_query_groups", "kv_channels"),
    MoELayerConfig: (
        "hidden_size",
        "ffn_hidden_size",
        "moe_ffn_hidden_size",
        "num_moe_experts",
        "moe_shared_expert_intermediate_size",
        "moe_router_topk",
    ),
}
_FLEXTRON_LAYER_CONFIG_TYPES = tuple(_FLEXTRON_STRUCTURE_FIELDS)


def validate_flextron_layer_config_list(
    layer_config_list: Sequence[ArchitectureEntry], root_config: TransformerConfig | None = None
) -> tuple[TransformerConfig, ...]:
    """Validate and snapshot the subset of layer configs supported by Flextron.

    Flextron currently assumes one local decoder stack whose per-layer structural
    dimensions match the root model config. Pipeline markers, MTP markers, additional
    layer config types, and heterogeneous structural dimensions are therefore rejected.

    Args:
        layer_config_list: Flat HybridModel architecture to validate.
        root_config: Optional root model config used to reject heterogeneous sizing.

    Returns:
        A tuple containing the validated layer config objects in source order.

    Raises:
        NotImplementedError: If the architecture uses a feature Flextron does not support.
    """
    validated: list[TransformerConfig] = []
    for layer_idx, layer_config in enumerate(layer_config_list):
        if layer_config is PipelineSplit:
            raise NotImplementedError("Flextron does not support PipelineSplit markers.")
        if layer_config is MTPSplit:
            raise NotImplementedError("Flextron does not support MTPSplit markers or MTP layers.")

        layer_config_type = type(layer_config)
        if layer_config_type not in _FLEXTRON_LAYER_CONFIG_TYPES:
            raise NotImplementedError(
                "Flextron supports only MambaLayerConfig, AttentionLayerConfig, and "
                f"MoELayerConfig entries; got {layer_config_type.__name__} at index {layer_idx}."
            )

        if root_config is not None:
            for field_name in _FLEXTRON_STRUCTURE_FIELDS[layer_config_type]:
                layer_value = getattr(layer_config, field_name)
                root_value = getattr(root_config, field_name)
                if layer_value != root_value:
                    raise NotImplementedError(
                        "Flextron does not support heterogeneous structural layer configs: "
                        f"entry {layer_idx} has {field_name}={layer_value!r}, while the root "
                        f"config has {field_name}={root_value!r}."
                    )

        validated.append(layer_config)

    if root_config is not None and any(
        type(layer_config) is MoELayerConfig for layer_config in validated
    ):
        if root_config.moe_ffn_hidden_size != root_config.ffn_hidden_size:
            raise NotImplementedError(
                "Flextron requires moe_ffn_hidden_size to equal ffn_hidden_size because "
                "its MLP elasticity choices and masks use the root ffn_hidden_size; got "
                f"moe_ffn_hidden_size={root_config.moe_ffn_hidden_size!r} and "
                f"ffn_hidden_size={root_config.ffn_hidden_size!r}."
            )

    return tuple(validated)


def count_flextron_layer_configs(
    layer_config_list: Sequence[ArchitectureEntry], layer_config_type: type[TransformerConfig]
) -> int:
    """Count exact occurrences of a supported Flextron layer config type."""
    if layer_config_type not in _FLEXTRON_LAYER_CONFIG_TYPES:
        raise ValueError(f"Unsupported Flextron layer config type: {layer_config_type.__name__}.")
    validated = validate_flextron_layer_config_list(layer_config_list)
    return sum(type(layer_config) is layer_config_type for layer_config in validated)


def get_flextron_layer_ordinal(
    layer_config_list: Sequence[ArchitectureEntry],
    layer_idx: int,
    layer_config_type: type[TransformerConfig],
) -> int:
    """Return a layer's zero-based ordinal among configs of the requested exact type."""
    if layer_config_type not in _FLEXTRON_LAYER_CONFIG_TYPES:
        raise ValueError(f"Unsupported Flextron layer config type: {layer_config_type.__name__}.")
    validated = validate_flextron_layer_config_list(layer_config_list)
    if not 0 <= layer_idx < len(validated):
        raise IndexError(
            f"Flextron layer index {layer_idx} is outside the {len(validated)}-layer architecture."
        )
    if type(validated[layer_idx]) is not layer_config_type:
        raise ValueError(
            f"Flextron layer {layer_idx} is {type(validated[layer_idx]).__name__}, not "
            f"{layer_config_type.__name__}."
        )
    return sum(type(layer_config) is layer_config_type for layer_config in validated[:layer_idx])


def get_num_parameters(
    layer_config_list: Sequence[ArchitectureEntry],
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
) -> tuple[int | torch.Tensor, int | torch.Tensor]:

    layer_config_list = validate_flextron_layer_config_list(layer_config_list)

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

    # Per-layer attention head counts arise only from layer skipping; head
    # elasticity itself is no longer supported.
    if isinstance(num_attention_heads, int):
        per_layer_attn_heads = False
    else:
        per_layer_attn_heads = num_attention_heads.shape[0] != 1

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
    if per_layer_attn_heads:
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
    def mamba_params(mamba_nheads, mamba_num_groups):
        d_inner = mamba_nheads * mamba_d_head

        def get_conv_params(kernel_size, stride):
            cdim = d_inner + 2 * mamba_num_groups * mamba_d_state
            cbias = cdim
            cweight = cdim * stride * kernel_size
            return cbias + cweight

        mamba_dt_bias = mamba_nheads
        mamba_A_log = mamba_nheads
        # self.d_inner_local if self.D_has_hdim else self.nheads_local,
        mamba_D = mamba_nheads
        mamba_input_ln = norm_multiplier * hidden_size
        mamba_in_proj = hidden_size * (
            d_inner * 2 + 2 * mamba_num_groups * mamba_d_state + mamba_nheads
        )
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
    mamba_idx = 0
    attention_idx = 0
    moe_idx = 0
    for layer_config in layer_config_list:
        if type(layer_config) is MambaLayerConfig:
            if flex_hetero_mamba:
                all_params += mamba_params(
                    mamba_num_heads[mamba_idx], layer_config.mamba_num_groups
                )
                active_params += mamba_params(
                    mamba_num_heads[mamba_idx], layer_config.mamba_num_groups
                )
            else:
                all_params += mamba_params(mamba_num_heads, layer_config.mamba_num_groups)
                active_params += mamba_params(mamba_num_heads, layer_config.mamba_num_groups)
            mamba_idx += 1
        elif type(layer_config) is AttentionLayerConfig:
            if per_layer_attn_heads:
                all_params += att[attention_idx]
                active_params += att[attention_idx]
            else:
                all_params += att
                active_params += att
            attention_idx += 1
        elif type(layer_config) is MoELayerConfig:
            if flex_hetero_ffn or flex_hetero_moe_expert:
                all_params += moe_all[moe_idx]
                active_params += moe_active[moe_idx]
            else:
                all_params += moe_all
                active_params += moe_active
            moe_idx += 1

    return (
        embedding + all_params + final_layernorm + output_layer,
        embedding + active_params + final_layernorm + output_layer,
    )


def get_kv_cache_size(
    layer_config_list: Sequence[ArchitectureEntry],
    num_attention_heads=None,
    num_query_groups=None,
    kv_channels=None,
    mem_infer_seq_len: int = 0,
    mem_batch_size: int = 0,
) -> int | torch.Tensor:

    layer_config_list = validate_flextron_layer_config_list(layer_config_list)

    # Per-layer attention head counts arise only from layer skipping; head
    # elasticity itself is no longer supported.
    if isinstance(num_attention_heads, int):
        per_layer_attn_heads = False
    else:
        per_layer_attn_heads = num_attention_heads.shape[0] != 1

    if per_layer_attn_heads:
        kv_cache_size = 0
        head_idx = 0

        for layer_config in layer_config_list:
            if type(layer_config) is AttentionLayerConfig:
                current_heads = num_attention_heads[head_idx]

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
        num_attention_layers = sum(
            type(layer_config) is AttentionLayerConfig for layer_config in layer_config_list
        )
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
    layer_config_list: Sequence[ArchitectureEntry],
    mamba_num_heads: int = 0,
    mamba_d_head: int = 0,
    mamba_d_state: int = 0,
    mem_batch_size: int = 0,
) -> int | torch.Tensor:

    layer_config_list = validate_flextron_layer_config_list(layer_config_list)

    if isinstance(mamba_num_heads, int):
        flex_hetero_mamba = False
    else:
        flex_hetero_mamba = mamba_num_heads.shape[0] != 1

    if flex_hetero_mamba:
        ssm_cache_size = 0
        mamba_idx = 0
        for layer_config in layer_config_list:
            if type(layer_config) is MambaLayerConfig:
                current_mamba_num_heads = mamba_num_heads[mamba_idx]
                ssm_cache_size += (
                    mem_batch_size * current_mamba_num_heads * mamba_d_head * mamba_d_state
                )
                mamba_idx += 1

    else:
        num_mamba_layers = sum(
            type(layer_config) is MambaLayerConfig for layer_config in layer_config_list
        )
        ssm_cache_size = (
            mem_batch_size * mamba_num_heads * mamba_d_head * mamba_d_state * num_mamba_layers
        )

    return ssm_cache_size


def get_max_buffer_size(
    layer_config_list: Sequence[ArchitectureEntry],
    moe_num_experts: int = 0,
    shared_expert_intermediate_size: int = 0,
    ffn_hidden_size: int = 0,
    moe_router_topk: int = 0,
    mem_batch_size: int = 0,
    prefill_chunk_size: int = 0,
) -> torch.Tensor:

    layer_config_list = validate_flextron_layer_config_list(layer_config_list)
    num_moe_layers = sum(type(layer_config) is MoELayerConfig for layer_config in layer_config_list)

    if isinstance(moe_num_experts, int) or moe_num_experts.shape[0] == 1:
        moe_num_experts = (
            torch.tensor([moe_num_experts] * num_moe_layers).to(torch.cuda.current_device()).float()
        )

    if isinstance(ffn_hidden_size, int) or ffn_hidden_size.shape[0] == 1:
        ffn_hidden_size = (
            torch.tensor([ffn_hidden_size] * num_moe_layers).to(torch.cuda.current_device()).float()
        )

    max_buffer_list = []
    moe_idx = 0
    for layer_config in layer_config_list:
        if type(layer_config) is MoELayerConfig:
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
    layer_config_list: Sequence[ArchitectureEntry],
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
) -> torch.Tensor:
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
            layer_config_list=layer_config_list,
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
        layer_config_list=layer_config_list,
        num_attention_heads=num_attention_heads,
        num_query_groups=num_query_groups,
        kv_channels=kv_channels,
        mem_infer_seq_len=mem_infer_seq_len,
        mem_batch_size=mem_batch_size,
    )

    mem_max_buffer = memory_config.bpe_max_buffer * get_max_buffer_size(
        layer_config_list=layer_config_list,
        moe_num_experts=moe_num_experts,
        shared_expert_intermediate_size=shared_expert_intermediate_size,
        ffn_hidden_size=ffn_hidden_size,
        moe_router_topk=moe_router_topk,
        mem_batch_size=mem_batch_size,
        prefill_chunk_size=prefill_chunk_size,
    )

    mem_mamba_ssm_cache = memory_config.bpe_ssm_cache * get_mamba_ssm_cache_size(
        layer_config_list=layer_config_list,
        mamba_num_heads=mamba_num_heads,
        mamba_d_head=mamba_d_head,
        mamba_d_state=mamba_d_state,
        mem_batch_size=mem_batch_size,
    )
    return (mem_params + mem_kv_cache + mem_max_buffer + mem_mamba_ssm_cache) / 1024 / 1024 / 1024
