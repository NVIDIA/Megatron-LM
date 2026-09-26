# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass

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

    pipeline_model_parallel_size: int = 1
    """Inter-layer model parallelism. Splits transformer layers across GPU ranks. The number of
       virtual pipeline model-parallel ranks is determined from the number of pipeline stages
       specified in the `layer_pattern` and the pipeline model-parallel size (this value):
       ```
       virtual_pipeline_model_parallel_size = num_pipeline_stages / pipeline_model_parallel_size
       ```

       For example:
       - When 8 pipeline stages are specified in the layer pattern (i.e., 7 `PipelineSplit`s are
         present) and this is 2, there will be 4 virtual pipeline stages with 2 physical splits.
       - When 8 pipeline stages are specified in the layer pattern (i.e., 7 `PipelineSplit`s are
         present) and this is 8, there will be no virtual pipeline stages with 8 physical splits.

       More information on virtual pipeline model-parallelism:
       Interleaved pipeline parallelism is used to improve performance by reducing the pipeline
       bubble.  Considers a transformer block as a list of smaller transformer (virtual) blocks.
       The number of virtual blocks per pipeline model parallel rank is the virtual model parallel
       size.  See Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM:
       arxiv.org/pdf/2104.04473.pdf for more details.
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
    pipeline_dtype: str | None = None
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

    def build(self): ...
