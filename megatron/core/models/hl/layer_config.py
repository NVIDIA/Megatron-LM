# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass

import torch


@dataclass
class LayerConfig:
    """Base configuration for the `HLModel`."""

    ###################
    # Model parallelism
    ###################
    sequence_parallel: bool = False
    """Makes tensor parallelism more memory efficient for LLMs (20B+) by parallelizing layer norms
       and dropout sequentially.  See Reducing Activation Recomputation in Large Transformer Models
       (https://arxiv.org/abs/2205.05198) for more details.
    """

    context_parallel_size: int = 1
    """Splits network input along sequence dimension across GPU ranks."""

    hierarchical_context_parallel_sizes: list[int] | None = None
    """Degrees of the hierarchical context parallelism. Users should provide a list to specify
       the sizes for different levels. Taking the a2a+p2p cp comm type as example, it contains
       groups of two levels, so the first value of the list indicates the group size of the a2a
       communication type, and the second value indicates the group size of the p2p communication
       type.
    """

    max_seqlen_per_dp_cp_rank: int | None = None
    """
    Maximum sequence length per DPxCP rank. This is the maximum sequence length each rank
    can handle without overflowing the memory. Typically, a good starting point is to set this
    to maximum sequence length / context parallel size.
    This is used to calculate the number and length of sub-samples assigned to
    each rank when using hybrid_context_parallel.
    """

    hybrid_context_parallel: bool = False
    """
    If true, enables hybrid context parallel. This is used to balance the workload of
    each CP rank when we use packed samples with variable sequence lengths.
    Please set max_seqlen_per_dp_cp_rank when using hybrid_context_parallel.
    """

    expert_model_parallel_size: int = 1
    """Distributes Moe Experts across sub data parallel dimension."""

    expert_tensor_parallel_size: int | None = None
    """Intra-layer tensor model parallelsm for expert layer. Splits tensors across GPU ranks."""

    ###################
    # Initialization
    ###################
    perform_initialization: bool = True
    """If true, weights are initialized. This option can be useful when you know you are going to
       load values from a checkpoint.
    """

    use_cpu_initialization: bool = False
    """When set to False, we initialize the weights directly on the GPU. CPU initialization is the
       same regardless of tensor model parallelism, but GPU initialization is not. Transferring
       weights from CPU to GPU can take a significant amount of time for large models.
    """

    ###################
    # Training
    ###################
    fp16: bool = False
    """If true, train with fp16 mixed precision training."""

    bf16: bool = False
    """If true, train with bf16 mixed precision training."""

    params_dtype: torch.dtype = torch.float32
    """dtype used when intializing the weights."""

    deterministic_mode: bool = False
    """If true, code that has deterministic execution will be chosen. This usually
       means slower execution, but is good for debugging and testing. Defaults to False."""

    enable_autocast: bool = False
    """If true runs the forward step function inside torch.autocast context."""

    autocast_dtype: torch.dtype | None = None
    """dtype to pass to torch.amp.autocast when enabled. If None, is set to pipeline_dtype."""

    num_microbatches_with_partial_activation_checkpoints: int | None = None
    """If int, set the number of microbatches where not all of the layers will be checkpointed and
       recomputed. The rest of the microbatches within the window of maximum outstanding
       microbatches will recompute all layers (either full recompute or selective recompute). If
       None, the checkpoint and recompute will be left up to the forward_step function.

    """

    ###################
    # Optimizations
    ###################
    gradient_accumulation_fusion: bool = False
    """If true, fuses weight gradient accumulation to GEMMs. Requires the custom CUDA extension
       fused_weight_gradient_mlp_cuda module. To use gradient_accumulation_fusion you must install
       APEX with --cpp_ext and --cuda_ext. For example: "pip install --global-option=\"--cpp_ext\"
       --global-option=\"--cuda_ext\" ". Note that the extension requires CUDA>=11. Otherwise, you
       must turn off gradient accumulation fusion.
    """

    use_te_rng_tracker: bool = False
    """If true, uses RNG state tracker in TransformerEngine if exists.
    """

    # Unsure about these; their settings seem to be related, but the
    # MoE-related stuff is too specific too fit here. Likewise, the
    # `delay_wgrad_compute` setting only concerns Linear layers.

    # Required to be True by `delay_wgrad_compute` and `ep_overlap_early_attn_memory_release`.
    overlap_moe_expert_parallel_comm: bool = False
    """Overlap EP A2A communications with independent computations of different micro-batches
    in 1f1b phase of pipelining or non-pipelining schedule.
    """

    delay_wgrad_compute: bool = False
    """Delay the weight gradient computation to improve batch-level communication overlapping"""

    ep_overlap_early_attn_memory_release: bool = False
    """Enable early memory release of attention activations during EP overlap.
    EP overlap can increase peak memory usage when the overlapped forward module allocates
    more memory than what is freed by the backward module. This flag addresses this by
    reordering the attention backward pass to occur earlier in the schedule.
    Specifically:
    - Without this flag: attn_bwd executes after moe_combine_fwd
    - With this flag: attn_bwd executes before mlp_fwd
    The earlier execution releases attention activations sooner, reducing peak memory.
    Note: This may impact performance as moe_combine_fwd and moe_dispatch_bwd become
    exposed (not overlapped with other computation).
    """

    ###################
    # CPU Offloading
    ###################
    cpu_offloading: bool = False
    """When set to True, all the activations are offloaded to the CPU asynchronously."""

    cpu_offloading_activations: bool = True
    """If True, offloads the activations to CPU."""

    cpu_offloading_weights: bool = False
    """If True, offloads the weights to CPU."""

    cpu_offloading_double_buffering: bool = False
    """If True, enables double buffering across layers while reloading activations from CPU."""

    ###################
    # Timing
    ###################
    barrier_with_L1_time: bool = True
    """If true, use barrier with level 1 time measurements. It is up to the user to make sure
       calling barrier with their timers will not result in hangs. This can happen if for example
       the user adds a level 1 timer that is not called by all ranks.
    """
