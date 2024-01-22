# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass
from typing import Callable, Optional

import torch


@dataclass
class ModelParallelConfig:
    """Base configuration for Megatron Core

    Model Parallelism
    -----------------

    tensor_model_parallel_size (int): Intra-layer model parallelism. Splits tensors across GPU ranks. Defaults to 1.

    context_parallel_size (int): Splits network input along sequence dimension across GPU ranks. Defaults to 1.

    pipeline_model_parallel_size (int): Inter-layer model parallelism. Splits transformer layers across GPU
        ranks. Defaults to 1.

    virtual_pipeline_model_parallel_size (int): Interleaved pipeline parallelism is used to improve performance by
        reducing the pipeline bubble.  Considers a transformer block as a list of smaller transformer (virtual) blocks.
        The number of virtual blocks per pipeline model parallel rank is the virtual model parallel size.  See Efficient
        Large-Scale Language Model Training on GPU Clusters Using Megatron-LM: https://arxiv.org/pdf/2104.04473.pdf for
        more details.  Defaults to None.

    sequence_parallel (bool): Makes tensor parallelism more memory efficient for LLMs (20B+) by
        parallelizing layer norms and dropout sequentially.  See Reducing Activation Recomputation in Large Transformer
        Models: https://arxiv.org/abs/2205.05198 for more details. Defaults to False.

    expert_model_parallel_size (int): Distributes Moe Experts across sub data parallel dimension. Defaults to False.

    Initialization
    --------------

    perform_initialization (bool, optional): If true, weights are initialized. This option can be useful when you
        know you are going to load values from a checkpoint. Defaults to True.

    use_cpu_initialization: (bool, optional): When set to False, we initialize the weights directly on the GPU.
        Transferring weights from CPU to GPU can take a significant amount of time for large models. Defaults to False.

    Training
    --------

    fp16 (bool): If true, train with fp16 mixed precision training. Defaults to False.

    bf16 (bool): If true, train with bf16 mixed precision training. Defaults to False.

    params_dtype (torch.dtype): dtype used when intializing the weights. Defaults to torch.float32

    timers (optional, default=None): TODO

    Optimizations
    -------------

    gradient_accumulation_fusion (bool): If true, fuses weight gradient accumulation to GEMMs. Requires the custom CUDA
        extension fused_weight_gradient_mlp_cuda module. To use gradient_accumulation_fusion you must install APEX with
        --cpp_ext and --cuda_ext. For example: "pip install --global-option=\"--cpp_ext\" --global-option=\"--cuda_ext\"
        ". Note that the extension requires CUDA>=11. Otherwise, you must turn off gradient accumulation fusion.
        Defaults to False.

    async_tensor_model_parallel_allreduce (bool, optional): If true, enables asynchronous execution of
        tensor-model-parallel all-reduce with weight gradient compuation of a column-linear layer.  Defaults to True.

    tp_comm_overlap (bool, optional): If true, allows overlapping of Linear layer execution with tensor parallel
        communication collectives like AllGather/ReduceScatter. Overlapping is done for the linear layers wherever
        possible during the forward and the backward pass.  Defaults to False.

    tp_comm_split_ag (bool, optional): If true, allows All-Gather overlap with Fprop GEMM by pipelining the GEMM 
        and All-Gather splits. Don't care if tp_comm_overlap is False. Defaults to True.

    tp_comm_atomic_ag (bool, optional): If true, allows All-Gather overlap with Fprop GEMM by pipelining the GEMM 
        and All-Gather both done atomically. Don't care if tp_comm_overlap is False. Defaults to True.

    tp_comm_split_rs (bool, optional): If true, allows Reduce-Scatter overlap with Fprop GEMM by pipelining the 
        GEMM and Reduce-Scatter splits. Don't care if tp_comm_overlap is False. Defaults to True.

    tp_comm_atomic_rs (bool, optional): If true, allows Reduce-Scatter overlap with Fprop GEMM by pipelining the
        GEMM and Reduce-Scatter both done atomically. Don't care if tp_comm_overlap is False. Defaults to True.

    tp_comm_bulk_dgrad (bool, optional): If true, allows All-Gather overlap with Bprop activation gradient GEMM. Don't 
        care if tp_comm_overlap is False. Defaults to True.

    tp_comm_bulk_wgrad (bool, optional): If true, allows Reduce-Scatter overlap with Bprop weight gradient GEMM. Don't 
        care if tp_comm_overlap is False. Defaults to True.

    Parallelism
    -----------

    finalize_model_grads_func (optional): Function that finalizes gradients on all workers. Could include ensuring that
        grads are all-reduced across data parallelism, pipeline parallelism, and sequence parallelism dimensions.

    Pipeline Parallelism
    --------------------

    pipeline_dtype (required): dtype used in p2p communication, usually params_dtype

    grad_scale_func (optional): If using loss scaling, this function should take the loss and return the
        scaled loss. If None, no function is called on the loss. Defaults to None.

    enable_autocast (bool): If true runs the forward step function inside torch.autocast context. Default is False.

    autocast_dtype (torch.dtype): dtype to pass to torch.amp.autocast when enabled. Default is pipeline_dtype.
    
    variable_seq_lengths (bool, optional): Support for variable sequence lengths across microbatches. Setting this
        communicates the size of tensors during pipeline parallelism communication, because of this extra overhead it
        should only be set if the sequence length varies by microbatch within a global batch. Defaults to False.

    num_microbatches_with_partial_activation_checkpoints (int, optional): If int, set the number of microbatches
        where not all of the layers will be checkpointed and recomputed. The rest of the microbatches within the window
        of maximum outstanding microbatches will recompute all layers (either full recompute or selective recompute). If
        None, the checkpoint and recompute will be left up to the forward_step function. Defaults to None.

    overlap_p2p_comm (bool, optional): When True some of the peer to peer communication for pipeline
        parallelism will overlap with computation. Must be False if batch_p2p_comm is true. Defaults to False.

    batch_p2p_comm (bool, optional): Use batch_isend_irecv instead of individual isend/irecv calls. Must be False
        if overlap_p2p_comm is True. Defaults to True.

    batch_p2p_sync (bool, optional): When using batch_isend_irecv, do a cuda.device.synchronize afterward to work
        around a bug in older version of PyTorch. Defaults to True.

    use_ring_exchange_p2p (bool, optional): Use custom ring_exchange kernel instead of
        torch.distributed.batch_isend_irecv(). Requires custom built torch with torch.distributed.ring_exchange.
        Defaults to False.

    deallocate_pipeline_outputs (optional): If True, output data is deallocated after the tensor is sent
        to the next pipeline stage.  Helps with saving memory, does nothing when pipeline parallel is not used.
        Defaults to False.

    no_sync_func (optional): Function that creates a context that suppresses asynchronous data-parallel
        communication. If the model is an instance of core.distributed.DistributedDataParallel, the default is to use
        core.distributed.DistributedDataParallel.no_sync.

    grad_sync_func (optional): Function that launches asynchronous gradient reductions (e.g. distributed optimizer
        gradient reduce-scatters). The function should take one argument: an iterable of parameters whose gradients are
        to be synchronized.

    param_sync_func (optional): Function that launches asynchronous parameter synchronizations (e.g. distributed
        optimizer parameter all-gathers). The function should take one argument: an iterable of parameters to be
        synchronized.

    pipeline_model_parallel_split_rank (int, optional): If int, rank where encoder and decoder should be split in
        cases where the model has both an encoder and decoder (e.g., T5). Ignored if None. Defaults to None.

    barrier_with_L1_time (bool, optional): If true, use barrier with level 1 time measurements. It is up to the user
        to make sure calling barrier with their timers will not result in hangs. This can happen if for example the user
        adds a level 1 timer that is not called by all ranks. Defaults to True.

    CPU Offloading
    --------------

    cpu_offloading (bool): When set to True, all the activations are offloaded to the CPU asynchronously. Defaults to True.
    cpu_offloading_num_layers (int): Tells the number of transformer layers for which activations has to be offloaded. Defaults to 0.
    cpu_offloading_activations (bool): If True, offloads the activations to CPU. Defaults to True.
    cpu_offloading_weights (bool): If True, offloads the weights to CPU. Defaults to True.

    """

    # Model parallelism
    tensor_model_parallel_size: int = 1
    context_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    virtual_pipeline_model_parallel_size: Optional[int] = None
    sequence_parallel: bool = False
    expert_model_parallel_size: int = 1

    # Initialization
    perform_initialization: bool = True
    use_cpu_initialization: bool = False

    # Training
    fp16: bool = False
    bf16: bool = False
    params_dtype: torch.dtype = torch.float32
    timers: Callable = None

    # Optimizations
    gradient_accumulation_fusion: bool = False
    async_tensor_model_parallel_allreduce: bool = False
    tp_comm_overlap: bool = False

    # Debug Options
    tp_comm_split_ag: bool = True
    tp_comm_atomic_ag: bool = True
    tp_comm_split_rs: bool = True
    tp_comm_atomic_rs: bool = True
    tp_comm_bulk_wgrad: bool = True
    tp_comm_bulk_dgrad: bool = True

    # Parallelism
    finalize_model_grads_func: Callable = None

    # Pipeline Parallel
    pipeline_dtype: torch.dtype = None
    grad_scale_func: Callable = None
    enable_autocast: bool = False
    autocast_dtype: torch.dtype = None
    variable_seq_lengths: bool = False
    num_microbatches_with_partial_activation_checkpoints: Optional[int] = None
    overlap_p2p_comm: bool = False
    batch_p2p_comm: bool = True
    batch_p2p_sync: bool = True
    use_ring_exchange_p2p: bool = False
    deallocate_pipeline_outputs: bool = False
    no_sync_func: Callable = None
    grad_sync_func: Callable = None
    param_sync_func: Callable = None
    pipeline_model_parallel_split_rank: Optional[int] = None

    #CPU Offloading
    cpu_offloading: bool = False
    cpu_offloading_num_layers: int = 0
    _cpu_offloading_context: ContextManager = None  # Used for internal use only, not to be set by the user. TODO: Need to move to the 'right' place when possible.
    cpu_offloading_activations: bool = True
    cpu_offloading_weights: bool = True

    # Timing
    barrier_with_L1_time: bool = True

    def __post_init__(self):
        """ Python dataclass method that is used to modify attributes after initialization.
            See https://docs.python.org/3/library/dataclasses.html#post-init-processing for more details.
        """
        if self.sequence_parallel:
            if self.tensor_model_parallel_size <= 1:
                raise ValueError("Can not use sequence paralllelism without tensor parallelism")
            if self.async_tensor_model_parallel_allreduce:
                # sequence_parallelism already does this async
                self.async_tensor_model_parallel_allreduce = False

        if self.pipeline_model_parallel_size > 1:
            if self.pipeline_dtype is None:
                raise ValueError(
                    "When using pipeline parallelism, pipeline_dtype must be specified"
                )

        if self.autocast_dtype is None:
            self.autocast_dtype = self.params_dtype

        if self.expert_model_parallel_size > 1 and self.tensor_model_parallel_size > 1:
            if self.sequence_parallel is False:
                raise ValueError(
                    "When using expert parallelism and tensor parallelism, sequence parallelism must be used"
                )
