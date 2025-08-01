# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass
from typing import Callable, ContextManager, Optional

import torch


@dataclass
class ModelParallelConfig:
    """Base configuration for Megatron Core

    The initialization function has an argument for each parameter.
    """

    ###################
    # Model parallelism
    ###################
    tensor_model_parallel_size: int = 1
    """Intra-layer model parallelism. Splits tensors across GPU ranks."""

    pipeline_model_parallel_comm_backend: Optional[str] = None
    """Configuring backend option of pipeline parallel communication (e.g., nccl, ucc)
       If None, the default backend will be used.
    """

    pipeline_model_parallel_size: int = 1
    """Inter-layer model parallelism. Splits transformer layers across GPU ranks."""

    virtual_pipeline_model_parallel_size: Optional[int] = None
    """Interleaved pipeline parallelism is used to improve performance by reducing the pipeline
       bubble.  Considers a transformer block as a list of smaller transformer (virtual) blocks.
       The number of virtual blocks per pipeline model parallel rank is the virtual model parallel
       size.  See Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM:
       arxiv.org/pdf/2104.04473.pdf for more details.
    """

    sequence_parallel: bool = False
    """Makes tensor parallelism more memory efficient for LLMs (20B+) by parallelizing layer norms
       and dropout sequentially.  See Reducing Activation Recomputation in Large Transformer Models
       (https://arxiv.org/abs/2205.05198) for more details.
    """

    context_parallel_size: int = 1
    """Splits network input along sequence dimension across GPU ranks."""

    hierarchical_context_parallel_sizes: Optional[list[int]] = None
    """Degrees of the hierarchical context parallelism. Users should provide a list to specify 
       the sizes for different levels. Taking the a2a+p2p cp comm type as example, it contains
       groups of two levels, so the first value of the list indicates the group size of the a2a
       communication type, and the second value indicates the group size of the p2p communication
       type.
    """

    expert_model_parallel_size: int = 1
    """Distributes Moe Experts across sub data parallel dimension."""

    expert_tensor_parallel_size: Optional[int] = None
    """Intra-layer tensor model parallelsm for expert layer. Splits tensors across GPU ranks."""

    moe_extended_tp: bool = False
    """NOTE: Deprecated from MCore v0.10. This flag is ignored.
      Its functionality is replaced by expert_tensor_parallel_size.
    """

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

    timers: Optional[Callable] = None
    """Timers object to call for various timing functions. See megatron.core.timers.Timers"""

    finalize_model_grads_func: Optional[Callable] = None
    """Function that finalizes gradients on all workers. Could include ensuring that grads are
       all-reduced across data parallelism, pipeline parallelism, and sequence parallelism
       dimensions.
    """

    grad_scale_func: Optional[Callable] = None
    """If using loss scaling, this function should take the loss and return the scaled loss. If
       None, no function is called on the loss.
    """

    no_sync_func: Optional[Callable] = None
    """Function that creates a context that suppresses asynchronous data-parallel communication. If
       the model is an instance of core.distributed.DistributedDataParallel, the default is to use
       core.distributed.DistributedDataParallel.no_sync.
    """

    grad_sync_func: Optional[Callable] = None
    """Function that launches asynchronous gradient reductions (e.g. distributed optimizer gradient
       reduce-scatters). The function should take one argument: an iterable of parameters whose
       gradients are to be synchronized.
    """

    param_sync_func: Optional[Callable] = None
    """Function that launches asynchronous parameter synchronizations (e.g. distributed optimizer
       parameter all-gathers). The function should take one argument: an iterable of parameters to
       be synchronized.
    """

    deterministic_mode: bool = False
    """If true, code that has deterministic execution will be chosen. This usually
       means slower execution, but is good for debugging and testing. Defaults to False."""

    enable_autocast: bool = False
    """If true runs the forward step function inside torch.autocast context."""

    autocast_dtype: Optional[torch.dtype] = None
    """dtype to pass to torch.amp.autocast when enabled. If None, is set to pipeline_dtype."""

    num_microbatches_with_partial_activation_checkpoints: Optional[int] = None
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

    async_tensor_model_parallel_allreduce: bool = False
    """NOTE: Deprecated. This flag is ignored."""

    use_te_rng_tracker: bool = False
    """If true, uses RNG state tracker in TransformerEngine if exists.
    """

    tp_comm_overlap: bool = False
    """If true, allows overlapping of Linear layer execution with tensor parallel communication
       collectives like AllGather/ReduceScatter. Overlapping is done for the linear layers wherever
       possible during the forward and the backward pass.
    """

    tp_comm_bulk_wgrad: bool = True
    """If true, allows All-Gather overlap with Bprop activation gradient GEMM. Don't care if
       tp_comm_overlap is False.
    """

    tp_comm_bulk_dgrad: bool = True
    """If true, allows Reduce-Scatter overlap with Bprop weight gradient GEMM. Don't care if
       tp_comm_overlap is False.
    """

    tp_comm_overlap_ag: bool = True
    """If true, allows All-Gather overlap with GEMM by pipelining the GEMM and All-Gather.
       Don't care if tp_comm_overlap is False.
    """

    tp_comm_overlap_rs: bool = True
    """If true, allows Reduce-Scatter overlap with GEMM by pipelining the GEMM and Reduce-Scatter.
       Don't care if tp_comm_overlap is False.
    """

    tp_comm_overlap_rs_dgrad: bool = False
    """If true, allows Reduce-Scatter overlap with DGRAD GEMM by pipelining the
       GEMM and Reduce-Scatter splits. Don't care if tp_comm_overlap is False.
    """

    tp_comm_split_ag: bool = True
    """Deprecated from TransformerEngine v1.6.0.
       If true, allows All-Gather overlap with Fprop GEMM by pipelining the GEMM and All-Gather
       splits. Don't care if tp_comm_overlap is False.
    """

    tp_comm_atomic_ag: bool = False
    """Deprecated from TransformerEngine v1.6.0.
       If true, allows All-Gather overlap with Fprop GEMM by pipelining the GEMM and All-Gather
       both done atomically. Don't care if tp_comm_overlap is False.
    """

    tp_comm_split_rs: bool = True
    """Deprecated from TransformerEngine v1.6.0.
       If true, allows Reduce-Scatter overlap with Fprop GEMM by pipelining the GEMM and
       Reduce-Scatter splits. Don't care if tp_comm_overlap is False.
    """

    tp_comm_atomic_rs: bool = False
    """Deprecated from TransformerEngine v1.6.0.
       If true, allows Reduce-Scatter overlap with Fprop GEMM by pipelining the GEMM and
       Reduce-Scatter both done atomically. Don't care if tp_comm_overlap is False.
    """

    cross_entropy_loss_fusion: bool = False
    """If this is enabled, the fused cross entropy implementation would be used.
       Defaults to False.
    """

    cross_entropy_fusion_impl: str = 'native'
    """If 'native', MCore based CE loss fusion is used, if 'te', Parallel CE loss
       from Transformer Engine library is used. Defaults to 'native'.
    """

    tp_comm_overlap_disable_qkv: bool = False
    """
       If true, the AllGather -> Gemm overlap for QKV gets disabled
    """

    tp_comm_overlap_disable_fc1: bool = False
    """
       If true, the AllGather -> Gemm overlap for FC1 layer of MLP gets disabled
    """

    tp_comm_bootstrap_backend: str = 'nccl'
    """
       Set the bootstrapping backend out of 'nccl', 'mpi', and 'gloo'
    """

    overlap_moe_expert_parallel_comm: bool = False
    """Overlap EP A2A communications with independent computations of different micro-batches
    in 1f1b phase of pipelining or non-pipelining schedule.
    """

    delay_wgrad_compute: bool = False
    """Delay the weight gradient computation to improve batch-level communication overlapping"""

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

    microbatch_group_size_per_vp_stage: Optional[int] = None
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
    # CPU Offloading
    ###################
    cpu_offloading: bool = False
    """When set to True, all the activations are offloaded to the CPU asynchronously."""

    cpu_offloading_num_layers: int = 0
    """Tells the number of transformer layers for which activations has to be offloaded."""

    _cpu_offloading_context: Optional[ContextManager] = (
        None
        # Used for internal use only, not to be set by a user.
        # TODO: Need to move to the 'right' place when possible.
    )
    """For internal use only, do not set."""

    cpu_offloading_activations: bool = True
    """If True, offloads the activations to CPU."""

    cpu_offloading_weights: bool = True
    """If True, offloads the weights to CPU."""

    ###################
    # Timing
    ###################
    barrier_with_L1_time: bool = True
    """If true, use barrier with level 1 time measurements. It is up to the user to make sure
       calling barrier with their timers will not result in hangs. This can happen if for example
       the user adds a level 1 timer that is not called by all ranks.
    """

    def __post_init__(self):
        """Python dataclass method that is used to modify attributes after initialization.
        See https://docs.python.org/3/library/dataclasses.html#post-init-processing for more
        details.
        """
        if self.sequence_parallel:
            if self.tensor_model_parallel_size <= 1:
                raise ValueError("Can not use sequence paralllelism without tensor parallelism")

        if self.expert_tensor_parallel_size is None:
            self.expert_tensor_parallel_size = self.tensor_model_parallel_size

        if self.pipeline_model_parallel_size > 1:
            if self.pipeline_dtype is None:
                raise ValueError(
                    "When using pipeline parallelism, pipeline_dtype must be specified"
                )

        if self.autocast_dtype is None:
            self.autocast_dtype = self.params_dtype

        if self.defer_embedding_wgrad_compute and self.pipeline_model_parallel_size == 1:
            raise ValueError(
                "Cannot defer embedding wgrad compute when pipeline model parallel is not used"
            )

        if self.defer_embedding_wgrad_compute and not self.gradient_accumulation_fusion:
            raise ValueError(
                "Cannot defer embedding wgrad compute when gradient accumulation fusion is not used"
            )

        if self.defer_embedding_wgrad_compute and self.wgrad_deferral_limit < 0:
            raise ValueError(
                "Wgrad deferral limit should be greater than or equal to 0 when it is enabled!"
            )

        if self.expert_model_parallel_size > 1 and self.tensor_model_parallel_size > 1:
            if self.sequence_parallel is False:
                raise ValueError(
                    "When using expert parallelism and tensor parallelism, "
                    "sequence parallelism must be used"
                )

        if self.microbatch_group_size_per_vp_stage is None:
            self.microbatch_group_size_per_vp_stage = self.pipeline_model_parallel_size

        if self.overlap_p2p_comm_warmup_flush:
            if not self.overlap_p2p_comm or self.batch_p2p_comm:
                raise ValueError(
                    "Pipeline parallel communication overlapping in warmup and flush is only "
                    "compatible with overlap_p2p_comm but not batch_p2p_comm."
                )
