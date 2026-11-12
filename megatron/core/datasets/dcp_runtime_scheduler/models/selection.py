# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Match the full Qwen3.5 prediction domain to normal training configuration."""


def is_full_qwen35_config(config):
    """Select the GDN/GQA model through the training architecture, not a JSON profile."""
    variant = getattr(config, 'experimental_attention_variant', None) or getattr(
        config, 'linear_attention_type', None
    )
    return variant in ('gdn', 'gated_delta_net')


def validate_full_qwen35_config(config):
    """Require the 40-decoder-plus-MTP model and the execution setup its parameters describe."""
    if not is_full_qwen35_config(config) or getattr(config, 'is_hybrid_model', False):
        raise ValueError('Full Qwen3.5 costs require the GPT GDN/GQA layer configuration')
    expected = {
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
    }
    for name, value in expected.items():
        actual = getattr(config, name, None)
        if type(actual) is not type(value) or actual != value:
            raise ValueError(f'Full Qwen3.5 costs require {name}={value!r}, got {actual!r}')
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
    for name in ('tensor_model_parallel_size', 'pipeline_model_parallel_size'):
        actual = getattr(config, name, 1)
        if actual != 1:
            raise ValueError(f'Full Qwen3.5 costs require TP1/PP1: {name}=1, got {actual!r}')
    stages = getattr(config, 'virtual_pipeline_model_parallel_size', None)
    if stages not in (None, 1):
        raise ValueError(
            'Full Qwen3.5 costs require VPP1: virtual_pipeline_model_parallel_size unset, '
            f'got {stages!r}'
        )
    if getattr(config, 'mtp_use_repeated_layer', False):
        raise ValueError(
            'Full Qwen3.5 costs require a distinct FA MTP layer: mtp_use_repeated_layer=False'
        )
    dropout = getattr(config, 'attention_dropout', None)
    if type(dropout) not in (int, float) or dropout != 0:
        raise ValueError(
            f'Full Qwen3.5 module parameters require attention_dropout=0.0, got {dropout!r}'
        )
    precision = {name: getattr(config, name, None) for name in ('bf16', 'fp8', 'fp4')}
    if not precision['bf16'] or precision['fp8'] or precision['fp4']:
        raise ValueError(
            'Full Qwen3.5 module parameters describe BF16 execution: bf16=True without fp8 or '
            f'fp4, got {precision}'
        )
    graphs = getattr(config, 'cuda_graph_impl', None)
    recompute = getattr(config, 'recompute_granularity', None)
    if graphs not in (None, 'none', '') or recompute is not None:
        raise ValueError(
            'Full Qwen3.5 parameters describe eager execution without recomputation: '
            f"cuda_graph_impl='none' and recompute_granularity unset, got {graphs!r} and "
            f'{recompute!r}'
        )
    mode = getattr(config, 'linear_cp_mode', None)
    if mode != 'chunkwise':
        raise ValueError(
            'Full Qwen3.5 module parameters describe chunkwise GDN context parallelism: '
            f"set linear_cp_mode='chunkwise' (got {mode!r})"
        )
    # linear_cp_layout is not read: TransformerConfig resolves an unset value to 'zigzag' and
    # rejects an explicit 'contiguous' with dynamic CP. GDN and full attention convert
    # module-locally from cp_partition_mode, which is the layout the parameters depend on.
    partition = getattr(config, 'cp_partition_mode', None)
    if partition != 'contiguous':
        raise ValueError(
            'Full Qwen3.5 module parameters describe packs whose tokens are owned contiguously '
            f"by the CP ranks: set cp_partition_mode='contiguous' (got {partition!r})"
        )
    kernel = getattr(config, 'gdn_kernel_backend', None)
    if kernel != 'fla':
        raise ValueError(
            'Full Qwen3.5 module parameters require the FLA GDN backend: '
            f"gdn_kernel_backend='fla', got {kernel!r}"
        )
    backend = getattr(config, 'attention_backend', None)
    backend = getattr(backend, 'name', backend)
    if backend not in ('auto', 'fused'):
        raise ValueError(
            'Full Qwen3.5 module parameters describe Transformer Engine attention selected '
            f"with attention_backend 'auto' or 'fused' (got {backend!r})"
        )
    communication = getattr(config, 'cp_comm_type', None)
    if communication != 'p2p' and not (
        isinstance(communication, list)
        and communication
        and all(value == 'p2p' for value in communication)
    ):
        raise ValueError(
            'Full Qwen3.5 module parameters require P2P attention communication: '
            f"cp_comm_type='p2p', got {communication!r}"
        )
    # The tail policy is not checked: for GDN with pad_packed_seq_alignment, TransformerConfig
    # accepts append_dummy_seq only, so the training configuration already fixes it.
    padding = getattr(config, 'pad_packed_seq_alignment', None)
    if padding not in ('max', 16384):
        raise ValueError(
            'Full Qwen3.5 module parameters describe a fixed capacity of 16384 tokens per rank: '
            f"set pad_packed_seq_alignment to 'max' or 16384 (got {padding!r})"
        )
