# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import MegatronModule
from megatron.core.utils import get_attr_wrapped_model


@dataclass
class MambaInferenceStateConfig:
    """
    Config for initializing Mamba model inference state tensors.

    Note that we maintain separate metadata for decode, regular prefill, and
    chunked prefill requests because the Mamba kernels do not yet support mixing
    these. Once the kernels have been updated we can simplify this code.
    """

    layer_type_list: List[str]
    """
    A list of strings that indicates the layer type (Mamba / Attention / MLP) for each layer.
    See `megatron/core/ssm/mamba_hybrid_layer_allocation.py` for the list of symbols.
    """

    mamba_conv_states_shape: Tuple[int]
    """Mamba conv states shape per request."""

    mamba_ssm_states_shape: Tuple[int]
    """Mamba ssm states shape per request."""

    @classmethod
    def from_model(cls, model: MegatronModule) -> Optional["MambaInferenceStateConfig"]:
        """Returns Mamba inference state config from the model if it is a hybrid model."""
        from megatron.core.ssm.mamba_hybrid_layer_allocation import Symbols

        decoder = get_attr_wrapped_model(model, "decoder")
        layer_type_list = getattr(decoder, "layer_type_list", None)
        if layer_type_list is not None and Symbols.MAMBA in layer_type_list:
            (mamba_conv_states_shape, mamba_ssm_states_shape) = (
                decoder.mamba_state_shapes_per_request()
            )
            return cls(
                layer_type_list=layer_type_list,
                mamba_conv_states_shape=mamba_conv_states_shape,
                mamba_ssm_states_shape=mamba_ssm_states_shape,
            )
        return None


@dataclass
class InferenceConfig:
    """
    Config for inference.

    NOTE: Must remain mutually exclusive with the `TransformerConfig`.
    """

    # =================================
    # KV cache config
    # =================================
    block_size_tokens: int = 256
    """Size of KV cache block size."""

    buffer_size_gb: int = 20
    """
    Buffer size reserved on the GPU for the KV cache.
    If `unified_memory_level` >= 1, then CPU memory is additionally utilized, resulting in a total
    buffer size of `buffer_size_gb + paused_buffer_size_gb`.
    """

    paused_buffer_size_gb: Optional[int] = None
    """
    Portion of buffer reserved for paused requests. Active requests are paused when there are not
    enough active blocks available to continue generating a request. The total buffer size
    (active + paused) depends on `unified_memory_level` (uvm):
        - uvm 0: buffer_size_gb (paused buffer is inclusive)
        - uvm 1: buffer_size_gb + paused_buffer_size_gb
    """

    mamba_memory_ratio: Optional[float] = None
    """
    Percentage of memory buffer to allocate for Mamba states. If not specified, allocates Mamba
    state tensors for each KV cache block. Only used for hybrid models.
    """

    max_requests: Optional[int] = None
    """
    Max number of active requests to use for decode-only forward passes.
    This is primarily limited by the combination of `buffer_size_gb` and `max_sequence_length`.
    """

    max_tokens: Optional[int] = None
    """
    Max number of tokens to use for forward passes. This is primarily limited by prefill activation
    memory usage. (Defaults to 16384).
    """

    unified_memory_level: int = 0
    """
    Sets unified memory usage within the dynamic inference context.
    The levels are:
        0) no unified memory (default)
        1) allocate `memory_buffer` in unified memory.
    Eventually, additional levels will be included to control other tensors within the context.
    """

    offload_kv_cache: bool = False
    """If True, offload KV cache during RL training."""

    # =================================
    # CUDA graph config
    # =================================
    num_cuda_graphs: Optional[int] = None
    """
    Maximum number of cuda graphs to capture, where the cuda graph batch sizes range from 1 to
    `max_requests`. Due to rounding, the actual number of cuda graphs may not equal this argument.
    """

    cuda_graph_mixed_prefill_count: Optional[int] = 16
    """ 
    The number of mixed prefill graphs to capture if mixed prefill/decode graphs are enabled.
    """

    use_cuda_graphs_for_non_decode_steps: bool = True
    """
    Whether to use CUDA graphs for non-decode steps.
    """

    persist_cuda_graphs: bool = False
    """
    Whether to persist CUDA graphs when the engine is suspended.
    If False and `unified_memory_level` is 0, CUDA graphs are deleted on `suspend()`
    and re-captured on `resume()` to save memory.
    """

    # =================================
    # Model config
    # =================================
    max_sequence_length: int = 2560
    """Max possible sequence length (prompt + output) that will occur."""

    mamba_inference_state_config: Optional[MambaInferenceStateConfig] = None
    """The Mamba inference state config if the model is a hybrid model."""

    pg_collection: Optional[ProcessGroupCollection] = None
    """A `ProcessGroupCollection` for distributed execution."""

    use_flashinfer_fused_rope: Optional[bool] = False
    """
    If True, use flashinfer's fused rope implementation.
    If None, defaults to using flash-infer if available.
    """

    materialize_only_last_token_logits: bool = True
    """
    Whether to only materialize logits for the last token. This should be set to False
    if returning log probs.
    """

    # =================================
    # Engine config
    # =================================
    enable_chunked_prefill: bool = False
    """Whether to enable chunked prefill."""

    # =================================
    # Logging config
    # =================================
    track_paused_request_events: bool = False
    """
    Whether to track paused request events. If True, `add_event_pause()` is called on
    requests when they are paused during bookkeeping.
    """

    track_generated_token_events: bool = False
    """
    Whether to track per-token events with timestamps for each generated token.
    When enabled, each generated token creates a GENERATED_TOKEN event with a
    timestamp, useful for per-token latency analysis.
    """

    metrics_writer: Optional["WandbModule"] = None
    """Wandb module for writing metrics."""

    logging_step_interval: int = 0
    """
    The step interval at which to log inference metrics to wandb.
    Defaults to 0, which means no logging.
    """

    request_metadata_types: Optional[List[Tuple[str, torch.dtype, bool]]] = None
    """
    A list of the per-request metadata types to track. Each entry is a tuple
    consisting of the string label, the target dtype, and whether to store the data on GPU.
    """
