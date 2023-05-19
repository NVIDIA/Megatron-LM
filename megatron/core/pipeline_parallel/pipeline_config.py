# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass
from typing import Callable

import torch

@dataclass
class PipelineConfig:
    """Pipeline configuration for Megatron Core

    sequence_parallel (bool): Makes tensor parallelism more memory efficient for LLMs (20B+) by
        parallelizing layer norms and dropout sequentially.  See Reducing Activation Recomputation in Large Transformer
        Models: https://arxiv.org/abs/2205.05198 for more details. Defaults to False.
    
    pipeline_dtype (required): dtype used in p2p communication, usually params_dtype

    grad_scaler (optional, default=None): If using loss scaling, this function should take the loss and return the
        scaled loss. If None, no function is called on the loss.

    enable_autocast (bool): If true runs the forward step function inside torch.autocast context. Default is False.

    autocast_dtype (torch.dtype): dtype to pass to torch.amp.autocast when emabled. Default is pipeline_dtype.

    tensor_shape (tuple, required when using pipeline parallelism): Shape of tensor. The tensor is expected to be 3D and
        its order of dimension is supposed to be ``(sequence, batch, hidden)``.  TODO: currently seq_length is
        automatically divided by tensor parallel size if sequence_parallel is True, is this the right behavior, or do we
        want the user to specify the correct tensor_shape?

    variable_seq_lengths (bool, default=False): Support for variable sequence lengths across microbatches. Setting this
        communicates the size of tensors during pipeline parallelism communication, because of this extra overhead it
        should only be set if the sequence length is not constant during training.

    num_microbatches_with_partial_activation_checkpoints (int, default=None): If int, set the number of microbatches
        where not all of the layers will be checkpointed and recomputed. The rest of the microbatches within the window
        of maximum outstanding microbatches will recompute all layers (either full recompute or selective recompute). If
        None, the checkpoint and recompute will be left up to the forward_step function.

    batch_p2p_comm (bool, default = False): Use batch_isend_irecv instead of individual isend/irecv calls.

    use_ring_exchange_p2p (bool, default = False): Use custom ring_exchange kernel instead of
        torch.distributed.batch_isend_irecv(). Requires custom built torch with torch.distributed.ring_exchange.

    deallocate_pipeline_outputs (optional, default=False): If True, output data is deallocated after the tensor is sent
        to the next pipeline stage.  Helps with saving memory, does nothing when pipeline parallel is not used.

    no_sync_func (optional): Function that creates a context that suppresses asynchronous data-parallel
        communication. If the model is an instance of torch.nn.DistributedDataParallel, the default is to use
        torch.nn.DistributedDataParallel.no_sync.

    grad_sync_func (optional): Function that launches asynchronous gradient reductions (e.g. distributed optimizer
        gradient reduce-scatters). The function should take one argument: an iterable of parameters whose gradients are
        to be synchronized.

    param_sync_func (optional): Function that launches asynchronous parameter synchronizations (e.g. distributed
        optimizer parameter all-gathers). The function should take one argument: an iterable of parameters to be
        synchronized.
    
    timers (optional, default=None): TODO

    Legacy args (TODO: remove these)
    ------------------
    decoder_seq_length (int, required for ModelType.encoder_and_decoder models):
        Sequence length of the decoder portion, used to determine tensor shapes.

    """

    sequence_parallel: bool = False
    grad_scaler: Callable = None
    enable_autocast: bool = False
    autocast_dtype: torch.dtype = None
    timers: Callable = None

    pipeline_dtype: torch.dtype = None
    tensor_shape: torch.Size = None
    variable_seq_lengths: bool = False
    num_microbatches_with_partial_activation_checkpoints: int = None
    batch_p2p_comm: bool = False
    use_ring_exchange_p2p: bool = False
    deallocate_pipeline_outputs: bool = False
    no_sync_func: Callable = None
    grad_sync_func: Callable = None
    param_sync_func: Callable = None

    # Legacy
    decoder_seq_length: int = None

    def __post__init__(self):
        if self.pipeline_dtype is None:
            raise ValueError("When using pipeline parallelism, pipeline_dtype must be specified")

        if self.tensor_shape is None:
            raise ValueError("tensor_shape must be provided")
        
        if self.autocast_dtype is None:
            self.autocast_dtype = self.pipeline_dtype

        if self.decoder_seq_length is None:
            self.decoder_seq_length = self.tensor_shape[0]
