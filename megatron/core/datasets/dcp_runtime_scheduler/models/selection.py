# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Match the full Qwen3.5 prediction domain to normal training configuration."""

from .base import uses_p2p

# The one configuration of the measured source. A setting of the first table must be set to
# its value; one of the second may be absent, which reads as its first value.
_REQUIRED = {
    'num_layers': 40,
    'mtp_num_layers': 1,
    'hidden_size': 2048,
    'num_attention_heads': 16,
    'num_query_groups': 2,
    'kv_channels': 256,
    'linear_conv_kernel_dim': 4,
    'linear_key_head_dim': 128,
    'linear_value_head_dim': 128,
    'linear_num_key_heads': 16,
    'linear_num_value_heads': 32,
    'qk_layernorm': True,
    'attention_output_gate': True,
    'bf16': True,
    'linear_cp_mode': 'chunkwise',
    'cp_partition_mode': 'contiguous',
    'gdn_kernel_backend': 'fla',
}
# linear_cp_layout and the tail policy are not read: TransformerConfig fixes both for GDN with
# dynamic CP and a fixed per-rank buffer ('zigzag', converted module-locally; append_dummy_seq).
_ACCEPTED = {
    'is_hybrid_model': (False, None),
    'tensor_model_parallel_size': (1,),
    'pipeline_model_parallel_size': (1,),
    'virtual_pipeline_model_parallel_size': (None, 1),
    'mtp_use_repeated_layer': (False, None),
    'attention_dropout': (0,),
    'fp8': (None, False, ''),
    'fp4': (None, False, ''),
    'cuda_graph_impl': ('none', None, ''),
    'recompute_granularity': (None,),
    'attention_backend': ('auto', 'fused'),
    'pad_packed_seq_alignment': ('max', 16384),
}


def is_full_qwen35_config(config):
    """Select the GDN/GQA model through the training architecture, not a JSON profile."""
    variant = getattr(config, 'experimental_attention_variant', None) or getattr(
        config, 'linear_attention_type', None
    )
    return variant in ('gdn', 'gated_delta_net')


def validate_full_qwen35_config(config):
    """Require the 40-decoder-plus-MTP model and the execution setup its parameters describe."""
    if not is_full_qwen35_config(config):
        raise ValueError('Full Qwen3.5 costs require the GPT GDN/GQA layer configuration')
    for name, value in _REQUIRED.items():
        actual = getattr(config, name, None)
        if type(actual) is not type(value) or actual != value:
            raise ValueError(f'Full Qwen3.5 costs require {name}={value!r}, got {actual!r}')
    for name, accepted in _ACCEPTED.items():
        actual = getattr(config, name, accepted[0])
        actual = getattr(actual, 'name', actual)  # attention_backend is an enum in training.
        if actual not in accepted:
            wanted = ' or '.join(repr(value) for value in accepted)
            raise ValueError(f'Full Qwen3.5 costs require {name}={wanted}, got {actual!r}')
    frequency = getattr(config, 'linear_attention_freq', None)
    if not (
        type(frequency) is int
        and frequency == 4
        or isinstance(frequency, (list, tuple))
        and tuple(frequency) == (1, 1, 1, 0) * 10
        and all(type(value) is int for value in frequency)
    ):
        raise ValueError(
            'Full Qwen3.5 costs require linear_attention_freq=4 (30 GDN and 10 FA decoder '
            f'layers in 3:1 order), got {frequency!r}'
        )
    if not uses_p2p(config):
        raise ValueError(
            f"Full Qwen3.5 costs require cp_comm_type='p2p', got "
            f"{getattr(config, 'cp_comm_type', None)!r}"
        )
