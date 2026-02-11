# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass

import torch

from megatron.core.models.hl import HLLayerConfig


@dataclass
class HLModelConfig:

    ###################
    # Model parallelism
    ###################
    layer_pattern: list[HLLayerConfig]
    """Layer pattern specifying pipeline parallel partitioning."""

    tensor_model_parallel_size: int = 1
    """Intra-layer model parallelism. Splits tensors across GPU ranks."""

    pipeline_model_parallel_comm_backend: str | None = None
    """Configuring backend option of pipeline parallel communication (e.g., nccl, ucc)
       If None, the default backend will be used.
    """

    virtual_pipeline_model_parallel_size: int = 1
    """Interleaved pipeline parallelism is used to improve performance by reducing the pipeline
       bubble.  Considers a transformer block as a list of smaller transformer (virtual) blocks.
       The number of virtual blocks per pipeline model parallel rank is the virtual model parallel
       size.  See Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM:
       arxiv.org/pdf/2104.04473.pdf for more details.

       For example:
       - When 4 pipeline stages are specified in the layer pattern (i.e., 3 `PipelineSplit`s are
         present) and this is set to 1 (default), there will be 4 physical pipeline stages with no
         virtual blocks.
       - When 4 pipeline stages are specified in the layer pattern (i.e., 3 `PipelineSplit`s are
         present) and this is set to 2, there will be 4 physical pipeline stages with 2 virtual
         blocks in each physical stage, resulting in 8 total pipeline stages.
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

    ###################
    # Optimizations
    ###################
    tp_comm_bootstrap_backend: str = 'nccl'
    """
       Set the bootstrapping backend out of 'nccl', 'mpi', and 'gloo'
    """

    ###################
    # Pipeline Parallel
    ###################
    pipeline_dtype: torch.dtype = None
    """dtype used in p2p communication, usually params_dtype"""

    variable_seq_lengths: bool = False
    """Support for variable sequence lengths across microbatches. Setting this communicates the size
        of tensors during pipeline parallelism communication, because of this extra overhead it
        should only be set if the sequence length varies by microbatch within a global batch.
    """

    overlap_p2p_comm: bool = False
    """When True some of the peer to peer communication for pipeline parallelism will overlap with
       computation. Must be False if batch_p2p_comm is true.
    """

    batch_p2p_comm: bool = True
    """Use batch_isend_irecv instead of individual isend/irecv calls. Must be False if
       overlap_p2p_comm is True.
    """

    batch_p2p_sync: bool = True
    """When using batch_isend_irecv, do a cuda.device.synchronize afterward to work around a bug in
       older version of PyTorch.
    """

    use_ring_exchange_p2p: bool = False
    """Use custom ring_exchange kernel instead of torch.distributed.batch_isend_irecv(). Requires
       custom built torch with torch.distributed.ring_exchange.
    """

    deallocate_pipeline_outputs: bool = False
    """If True, output data is deallocated after the tensor is sent to the next pipeline stage.
       Helps with saving memory, does nothing when pipeline parallel is not used.
    """

    defer_embedding_wgrad_compute: bool = False
    """If true, defers the embedding WGRAD GEMMs while pipeline flush is
       taking place enabling us to hide pipeline flush latency. Defaults to False.
    """

    wgrad_deferral_limit: int = 0
    """This value tunes the number of micro-batches for which the embedding weight gradient compute
       needs to be deferred to pipeline flush, this argument is invalid if
       `defer_embedding_wgrad_compute` is False.
       Defaults to 0, which means all micro-batches are deferred.
    """

    overlap_p2p_comm_warmup_flush: bool = False
    """If true, overlap communication and computation in warm up and flush phase.
       Only valid when overlap_p2p_comm is True and batch_p2p_comm is False.
       Defaults to False.
    """

    microbatch_group_size_per_vp_stage: int | None = None
    """This value specifies the number of micro-batches that are executed
       at a time for a given virtual stage (both forward and backward).
       Default (in __post_init__() method below) to pipeline_parallel_size
       which specifies a depth-first schedule.
       Example: for PP=2 VP=2, when microbatch_group_size_per_vp_stage=2,
       num_microbatches = 4, we have
       rank 0 | 0 1 0 1 2 3 2 3
       rank 1 |   0 1 0 1 2 3 2 3
       When microbatch_group_size_per_vp_stage=3, num_microbatches = 5,
       we have
       rank 0 | 0 1 2 0 1 2 3 4 3 4
       rank 1 |   0 1 2 0 1 2 3 4 3 4
    """

    ###################
    # Timing
    ###################
    barrier_with_L1_time: bool = True
    """If true, use barrier with level 1 time measurements. It is up to the user to make sure
       calling barrier with their timers will not result in hangs. This can happen if for example
       the user adds a level 1 timer that is not called by all ranks.
    """

    ###################
    # Model settings
    ###################
    untie_embeddings_and_output_weights = False

    def build(self):
        ...
