# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

"""Utility functions used throughout Megatron core"""
import array
import hashlib
import logging
import math
import operator
import queue
import socket
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from functools import reduce
from types import TracebackType
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch

from megatron.core import parallel_state
from megatron.core.dist_checkpointing.mapping import ShardedTensor

logger = logging.getLogger(__name__)


def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, "{} is not divisible by {}".format(numerator, denominator)


def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


def get_attr_wrapped_model(model, attr, allow_none=True, return_model_obj=False):
    """Get an attribute from a wrapped model.
    If return_model_obj is true, return the object that has the 'attr' attribute;
    otherwise, return the attribute directly."""
    if isinstance(model, list):
        raise RuntimeError("_get_attr_wrapped_model given a list of models")

    if allow_none:

        def condition(model, attr):
            return not hasattr(model, attr)

    else:

        def condition(model, attr):
            return getattr(model, attr, None) is None

    while condition(model, attr):
        if not hasattr(model, "module"):
            raise RuntimeError(f"_get_attr_wrapped_model couldn't find attribute {attr}")

        model = model.module

    if return_model_obj:
        return model
    return getattr(model, attr)


def get_model_type(model):
    """Returns model_type attribute"""
    return get_attr_wrapped_model(model, 'model_type')


def get_model_xattn(model):
    """Returns whether the model has the xattn_needed attribute"""
    try:
        return get_attr_wrapped_model(model, 'xattn_needed')
    except RuntimeError:
        return False


def get_model_config(model):
    """Returns the config attribute, allowed to return None"""
    return get_attr_wrapped_model(model, 'config', allow_none=False)


class GlobalMemoryBuffer:
    """Global buffer to avoid dynamic memory allocations.
    Caller should ensure that buffers of the same name
    are not used concurrently."""

    def __init__(self):
        self.buffer = {}

    def get_tensor(self, tensor_shape, dtype, name):
        """
        Returns (potentially) a sub-tensor from the self.buffer for the given shape.
        """
        required_len = reduce(operator.mul, tensor_shape, 1)
        if (
            self.buffer.get((name, dtype), None) is None
            or self.buffer[(name, dtype)].numel() < required_len
        ):
            self.buffer[(name, dtype)] = torch.empty(
                required_len, dtype=dtype, device=torch.cuda.current_device(), requires_grad=False
            )

        return self.buffer[(name, dtype)][0:required_len].view(*tensor_shape)


def _kernel_make_viewless_tensor(inp, requires_grad):
    """Make a viewless tensor.

    View tensors have the undesirable side-affect of retaining a reference
    to the originally-viewed tensor, even after manually setting the '.data'
    field. This method creates a new tensor that links to the old tensor's
    data, without linking the viewed tensor, referenced via the '._base'
    field.
    """
    out = torch.empty((1,), dtype=inp.dtype, device=inp.device, requires_grad=requires_grad)
    out.data = inp.data
    return out


class MakeViewlessTensor(torch.autograd.Function):
    """
    Autograd function to make a viewless tensor.

    This function should be used in cases where the computation graph needs
    to be propagated, but we only want a viewless tensor (e.g.,
    ParallelTransformer's hidden_states). Call this function by passing
    'keep_graph = True' to 'make_viewless_tensor()'.
    """

    @staticmethod
    def forward(ctx, inp, requires_grad):
        """Runs the fwd pass of _kernel_make_viewless_tensor"""
        return _kernel_make_viewless_tensor(inp, requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        """No-op"""
        return grad_output, None


def make_viewless_tensor(inp, requires_grad, keep_graph):
    """
    Entry-point for creating viewless tensors.

    This method should be used, rather than calling 'MakeViewlessTensor'
    or '_kernel_make_viewless_tensor' directly. This method acts as a
    switch for determining if an autograd function or a regular method
    should be used to create the tensor.
    """

    # return tensor as-is, if not a 'view'
    if inp._base is None:
        return inp

    # create viewless tensor
    if keep_graph:
        return MakeViewlessTensor.apply(inp, requires_grad)
    else:
        return _kernel_make_viewless_tensor(inp, requires_grad)


def assert_viewless_tensor(tensor, extra_msg=None):
    """Assert that a tensor is not a view (i.e., its '._base' field is
    not set)."""
    if isinstance(tensor, list):
        [assert_viewless_tensor(t) for t in tensor]
        return tensor
    if not isinstance(tensor, torch.Tensor):
        return tensor
    assert tensor._base is None, (
        "Ensure tensor._base is None before setting tensor.data or storing "
        "tensor to memory buffer. Otherwise, a memory leak will occur (and "
        "likely accumulate over iterations). %s"
    ) % extra_msg
    return tensor


def safely_set_viewless_tensor_data(tensor, new_data_tensor):
    """Safely set tensor's '.data' field.

    Check first that the tensor is viewless (i.e., '._base' not set). If not,
    raise an exception.
    """
    assert_viewless_tensor(
        tensor,
        extra_msg="FYI, tensor._base has shape %s, and new_data_tensor has shape %s."
        % ("--" if tensor._base is None else tensor._base.shape, new_data_tensor.shape),
    )
    tensor.data = new_data_tensor


def init_method_normal(sigma):
    """Init method based on N(0, sigma)."""

    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=sigma)

    return init_


def scaled_init_method_normal(sigma, num_layers):
    """Init method based on N(0, sigma/sqrt(2*num_layers)."""
    std = sigma / math.sqrt(2.0 * num_layers)

    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=std)

    return init_


def log_single_rank(logger: logging.Logger, *args: Any, rank: int = 0, **kwargs: Any):
    """If torch distributed is initialized, log only on rank

    Args:
        logger (logging.Logger): The logger to write the logs

        args (Tuple[Any]): All logging.Logger.log positional arguments

        rank (int, optional): The rank to write on. Defaults to 0.

        kwargs (Dict[str, Any]): All logging.Logger.log keyword arguments
    """
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == rank:
            logger.log(*args, **kwargs)
    else:
        logger.log(*args, **kwargs)


def log_on_each_pipeline_stage(logger: logging.Logger, *args: Any, **kwargs: Any):
    """Log on first rank in each pipeline stage

    Args:
        logger (logging.Logger): The logger to write the logs

        args (Tuple[Any]): All logging.Logger.log positional arguments

        kwargs (Dict[str, Any]): All logging.Logger.log keyword arguments
    """
    assert torch.distributed.is_initialized()

    if (
        parallel_state.get_data_parallel_rank(with_context_parallel=True) == 0
        and parallel_state.get_tensor_model_parallel_rank() == 0
    ):
        logger.log(*args, **kwargs)


def check_param_hashes_across_dp_replicas(
    model: List[torch.nn.Module], cross_check: bool = False
) -> bool:
    """Computes hashes of all parameters in model, all-gathers hashes across DP replicas,
    and then checks for equality between the locally-computed hashes and those of other ranks.

    NOTE: This function computes SHA-1 hashes on the CPU and thus needs to move all param
    tensors from GPU to CPU first; as a result, this function is not intended to be called
    very frequently in the main training loop.

    Args:
        model (List[torch.nn.Module]): List of model chunks whose parameter hashes need to
            be checked.
        cross_check (bool): If true, will check whether hashes match across all DP replicas.

    Returns:
        True if all param hashes match with corresponding hash on DP replica 0 or
        across all replicas if cross_check is enabled, False otherwise.
    """

    # Compute per-parameter hashes on this rank.
    params = []
    local_param_hashes = []
    for model_chunk_id, model_chunk in enumerate(model):
        for param_name, param in model_chunk.named_parameters():
            param_hash = torch.frombuffer(
                array.array(
                    'B', hashlib.sha1(param.data.to("cpu").float().numpy(force=True)).digest()
                ),
                dtype=torch.uint8,
            )
            params.append((model_chunk_id, param_name, param))
            local_param_hashes.append(param_hash)
    local_param_hashes = torch.stack(local_param_hashes)

    # Collect per-parameter hashes across all ranks in DP group.
    all_param_hashes = [
        torch.zeros_like(local_param_hashes)
        for _ in range(parallel_state.get_data_parallel_world_size())
    ]
    torch.distributed.all_gather(
        all_param_hashes, local_param_hashes, group=parallel_state.get_data_parallel_group_gloo()
    )

    # Make sure local per-parameter hash matches DP rank 0.
    param_hashes_match = torch.equal(local_param_hashes, all_param_hashes[0])
    if not param_hashes_match:
        for i, (model_chunk_id, param_name, param) in enumerate(params):
            if not torch.equal(local_param_hashes[i], all_param_hashes[0][i]):
                rank = torch.distributed.get_rank()
                logger.info(
                    f"[Rank {rank}] Hash not matching for {param_name} in model chunk"
                    f"{model_chunk_id}"
                )
    if cross_check:
        # Make sure all ranks have the same hash.
        return all(map(lambda x: torch.equal(local_param_hashes, x), all_param_hashes))
    else:
        return param_hashes_match


def make_tp_sharded_tensor_for_checkpoint(
    tensor, key, tp_axis=0, replica_id=None, prepend_offsets=(), **kwargs
):
    """Helper for instantiating a ShardedTensor where the `tp_axis` dimension
    is sharded across TP group.

    Optionally, can provide offsets which prepend new dimensions to the tensor.
    """

    prepend_axis_num = len(prepend_offsets)

    if replica_id is None:
        replica_id = (0, 0, parallel_state.get_data_parallel_rank(with_context_parallel=True))

    return ShardedTensor.from_rank_offsets(
        key,
        tensor,
        *prepend_offsets,
        (
            tp_axis + prepend_axis_num,
            parallel_state.get_tensor_model_parallel_rank(),
            parallel_state.get_tensor_model_parallel_world_size(),
        ),
        replica_id=replica_id,
        prepend_axis_num=prepend_axis_num,
        **kwargs,
    )


def make_sharded_tensor_for_checkpoint(tensor, key, prepend_offsets=(), replica_id=None, **kwargs):
    """Helper for instantiating a non-sharded ShardedTensor (replicated across TP and DP group).

    Optionally, can provide offsets which prepend new dimensions to the tensor.
    """

    prepend_axis_num = len(prepend_offsets)

    if replica_id is None:
        replica_id = (
            0,
            parallel_state.get_tensor_model_parallel_rank(),
            parallel_state.get_data_parallel_rank(with_context_parallel=True),
        )

    return ShardedTensor.from_rank_offsets(
        key,
        tensor,
        *prepend_offsets,
        replica_id=replica_id,
        prepend_axis_num=prepend_axis_num,
        **kwargs,
    )


def prepare_input_tensors_for_wgrad_compute(grad_output, all_gathered_input):
    """Ensure grad_output is stored in a contiguous buffer."""
    # Doing gather + slicing during the NeMo forward pass can make this tensor
    # not be contiguous. PyTorch only checks if the tensor is contiguous, and only
    # clones it if it's not contiguous:
    # https://github.com/pytorch/pytorch/blob/c47cf9bc7f9e02f649ab4ed53fe4d35732c92ab6/torch/_refs/__init__.py#L2761
    grad_output = grad_output.contiguous()
    # Convert the tensor shapes to 2D for execution compatibility
    if grad_output.dim() == 3:
        grad_output = grad_output.view(
            grad_output.shape[0] * grad_output.shape[1], grad_output.shape[2]
        )
        all_gathered_input = all_gathered_input.view(
            all_gathered_input.shape[0] * all_gathered_input.shape[1], all_gathered_input.shape[2]
        )

    return grad_output, all_gathered_input


def drain_embedding_wgrad_compute(config, embedding_activation_buffer, grad_output_buffer, weight):
    """Helper for performing embedding wgrad GEMM's during the pipeline drain phase, pipelines the
    AllGather and GEMM's.

    Should only be used when pipeline model parallelism and gradient accumulation
    fusion are enabled.
    """

    assert len(embedding_activation_buffer) == len(
        grad_output_buffer
    ), "Length of activation and gradient buffers need to be equal!"

    import fused_weight_gradient_mlp_cuda

    from megatron.core.parallel_state import (
        get_global_memory_buffer,
        get_tensor_model_parallel_group,
        get_tensor_model_parallel_world_size,
    )

    input = embedding_activation_buffer.pop(0)
    world_size = get_tensor_model_parallel_world_size()
    dim_size = list(input.size())
    dim_size[0] = dim_size[0] * world_size

    all_gathered_input = [None, None]
    if config.sequence_parallel:
        all_gather_buffer = get_global_memory_buffer().get_tensor(dim_size, input.dtype, "mpu_0")
        handle = torch.distributed._all_gather_base(
            all_gather_buffer, input, group=get_tensor_model_parallel_group(), async_op=False
        )

        all_gathered_input[0] = all_gather_buffer
        all_gather_buffer = None
    else:
        all_gathered_input[0] = input

    input = None

    def wgrad_compute(all_gathered_input, grad_output, weight):

        grad_output, all_gathered_input = prepare_input_tensors_for_wgrad_compute(
            grad_output, all_gathered_input
        )

        if config.gradient_accumulation_fusion:
            if weight.main_grad.dtype == torch.float32:
                fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(
                    all_gathered_input, grad_output, weight.main_grad
                )
            elif weight.main_grad.dtype in (torch.float16, torch.bfloat16):
                fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp16(
                    all_gathered_input, grad_output, weight.main_grad
                )
            else:
                raise RuntimeError("Unsupported gradient type for gradient accumulation fusion")

    # We have all_gathered_input list acting as a double buffer here,
    # since we are pipelining the AllGather and GEMM,one buffer all gathers
    # the input while the other buffer reads from it for the GEMM. We use i
    # and (i+1) for indexing to enable this double buffering.
    for i in range(len(embedding_activation_buffer)):
        input = embedding_activation_buffer.pop(0)
        if config.sequence_parallel:
            name = "mpu_" + str((i + 1) % 2)
            all_gather_buffer = get_global_memory_buffer().get_tensor(dim_size, input.dtype, name)
            handle = torch.distributed._all_gather_base(
                all_gather_buffer, input, group=get_tensor_model_parallel_group(), async_op=True
            )

            all_gathered_input[(i + 1) % 2] = all_gather_buffer
            all_gather_buffer = None
        else:
            all_gathered_input[(i + 1) % 2] = input

        grad_output = grad_output_buffer.pop(0)
        wgrad_compute(all_gathered_input[i % 2], grad_output, weight)
        drain_idx = (i + 1) % 2
        input, all_gathered_input[i % 2], grad_output = None, None, None

        if config.sequence_parallel:
            handle.wait()

    grad_output = grad_output_buffer.pop(0)
    wgrad_compute(all_gathered_input[drain_idx], grad_output, weight)
    input, all_gathered_input[drain_idx], grad_output = None, None, None


def local_multi_tensor_applier(op, noop_flag_buffer, tensor_lists, *args):
    """Multi tensor op applier"""
    return op(2048 * 32, noop_flag_buffer, tensor_lists, *args)


# computes l2 norm for a list of contiguous tensors
# works as a drop-in replacement for amp_C.multi_tensor_l2norm
def local_multi_tensor_l2_norm(chunk_size, noop_flag, tensor_lists, per_tensor, *args):
    """
    Computes l2 norm for a list of contiguous tensors
    works as a drop-in replacement for amp_C.multi_tensor_l2norm
    """
    l2 = [[(torch.norm(tensor)) for tensor in tensor_list] for tensor_list in tensor_lists]
    l2_reduced = torch.norm(torch.tensor(l2))
    l2_cuda = torch.tensor([float(l2_reduced)], dtype=torch.float, device='cuda')
    return l2_cuda, None


# works as a drop-in replacement for amp_C.multi_tensor_scale
def local_multi_tensor_scale(chunk_size, noop_flag, tensor_lists, scale):
    """Works as a drop-in replacement for amp_C.multi_tensor_scale."""
    inputs, targets = tensor_lists[0], tensor_lists[1]
    if inputs == targets:
        for i in range(len(targets)):
            # for parity with apex implementation
            targets[i] *= scale
    else:
        for i in range(len(targets)):
            targets[i] = inputs[i] * scale


class _ValueWithRank:
    """This is an internal class, not for use outside this module

    Attributes:
        _rank (int): rank for the value
        _value (float) : the value it stores, eg elapsed time
        _unit (str) : unit for the value
    """

    def __init__(self, value: float, rank: int, unit: str = "") -> None:
        """Initializer

        Args:
            _value (float): the initial value with which it is inited
            _rank (int): the rank number
            _unit (str) : the unit of the value, eg ms or flops
        """
        self._rank = rank
        self._value = value
        self._unit = unit

    def __lt__(self, other) -> bool:
        """Check if value of self is smaller than other's value

        Args:
            other (_ValueWithRank): The other object to compare with

        Returns:
            bool: True if lhs._value of operand is less than rhs._value, else False
        """
        return self._value < other._value

    def __gt__(self, other) -> bool:
        """Check if value of self is larger than other's value

        Args:
            other (_ValueWithRank): The other object to compare with

        Returns:
            bool: True if lhs._value of operand is greater than rhs._value, else False
        """
        return self._value > other._value

    def __call__(self) -> Tuple[float, int, str]:
        """Returns the value, the rank, and unit as a Tuple

        Returns:
            Tuple[float, int, str]: value, rank, unit
        """
        return self._value, self._rank, self._unit

    def __str__(self) -> str:
        """String representation of the object

        Returns:
            str: strigified object
        """

        return f"{self._value:.2f}{self._unit}/{self._rank}"


@dataclass
class _StragglerData:
    """This is an internal dataclass, not for use outside this module

    Attributes:
        min_elapsed (_ValueWithRank) min iteration time across all ranks
        max_elapsed (_ValueWithRank) max iteration time across all ranks
        min_btime (_ValueWithRank) min cpu time across all ranks
        max_btime (_ValueWithRank) max cpu time across all ranks
        min_temp (_ValueWithRank): min gpu temp across all ranks
        max_temp (_ValueWithRank): max gpu temp across all ranks
        min_power (_ValueWithRank) min gpu power across all ranks
        max_power (_ValueWithRank) max gpu power across all ranks
        min_util (_ValueWithRank): min gpu util across all ranks
        max_util (_ValueWithRank): max gpu util across all ranks
        min_clock (_ValueWithRank): min gpu clock across all ranks
        max_clock (_ValueWithRank) max gpu clock across all ranks
        aflops (List[_ValueWithRank]): sorted array of (_ValueWithRank)
    """

    # gemm time
    min_elapsed = _ValueWithRank(sys.float_info.max, 0, "ms")
    max_elapsed = _ValueWithRank(sys.float_info.min, 0, "ms")
    # get_batch time
    min_btime = _ValueWithRank(sys.float_info.max, 0, "us")
    max_btime = _ValueWithRank(sys.float_info.min, 0, "us")
    # temp
    min_temp = _ValueWithRank(sys.float_info.max, 0, "C")
    max_temp = _ValueWithRank(sys.float_info.min, 0, "C")
    # power
    min_power = _ValueWithRank(sys.float_info.max, 0, "W")
    max_power = _ValueWithRank(sys.float_info.min, 0, "W")
    # util
    min_util = _ValueWithRank(sys.float_info.max, 0, "%")
    max_util = _ValueWithRank(sys.float_info.min, 0, "%")
    # clock
    min_clock = _ValueWithRank(sys.float_info.max, 0, "MHz")
    max_clock = _ValueWithRank(sys.float_info.min, 0, "MHz")
    aflops: Union[List[_ValueWithRank], None] = None


class StragglerDetector:
    """Singleton Class implementing per rank Straggler Detector

    It use cuda events to time operation of choice using the
    start and stop methods which can be directly invoked using
    the class instance or can be used like a python context.
    After collection, a report() method is available to display
    the collected metrics. It is only supported if CUDA is
    available. megatron/core/README_STRAGGLER.md for more info

    Note:
        The instance and class attributes mentioned below are all
        private to the class and has no use outside the class

    Attributes:
        _off (bool): current state of the toggle
        start (FunctionType): start method
        stop (FunctionType): stop method
        world (int): world size
        rank (int): rank for this instance
        mmcnt (int): number of ranks to report
        port (int): control port
        amp (float): amplification factor for TFLOPs, default 3.0
        toggle (bool): whether to start/stop detector collection
        bdata (bool): when true, just collect get_batch
        dev (int): cuda device
        evt_q (LifoQueue): cuda event queue
        start_gemm_ev (list[torch.cuda.Event]): cuda start event
        stop_gemm_ev (list[torch.cuda.Event]): cuda stop event
        start_data_ev (list[torch.cuda.Event]): cuda start event
        stop_data_ev (list[torch.cuda.Event]): cuda stop event
        start_gemm_tm (list[int]): start time (wallclock)
        stop_gemm_tm (list[int]): stop time (wallclock)
        start_data_tm (list[int]): start time for get_batch
        stop_data_tm (list[int]): stop time for get_batch
        sock (socket): the controller socket
        ctrlr (Thread): the controller thread
    """

    _configured = False
    """Indicates if the singleton instance is configured or not
    """

    def __new__(cls: Type["StragglerDetector"]) -> "StragglerDetector":
        """Constructor
        Creates an instance of the class if not created

        Args:
            cls (Type[&#39;StragglerDetector&#39;]): The class type

        Returns:
            StragglerDetector: the class instance
        """

        if not hasattr(cls, "_instance"):
            cls._instance = super(StragglerDetector, cls).__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initializer

        The inital state of the StragglerDetector instance is disabled.
        The enabled state is indicated using self._off member variable
        and the proerty enabled.
        """
        self._off: bool = True
        self.start = self.null_method
        self.stop = self.null_method
        self.world: int = 0
        self.rank: int = 0
        self.mmcnt: int = 1
        self.port: int = 0
        self.amp: float = 3.0
        self.toggle: bool = False
        self.bdata: bool = False
        self.dev: Union[torch.device, int, None] = None
        self.evt_q: Union[queue.LifoQueue, None] = None
        self.start_gemm_ev: List[torch.cuda.Event] = []
        self.stop_gemm_ev: List[torch.cuda.Event] = []
        self.start_data_ev: List[torch.cuda.Event] = []
        self.stop_data_ev: List[torch.cuda.Event] = []
        self.start_gemm_tm: List[int] = []
        self.stop_gemm_tm: List[int] = []
        self.start_data_tm: List[int] = []
        self.stop_data_tm: List[int] = []
        self.sock: Union[socket.socket, None] = None
        self.ctrlr: Union[threading.Thread, None] = None

    def configure(
        self,
        world: int,
        rank: int,
        mmcnt: int = 1,
        amp: float = 3.0,
        port: int = 65535,
        prefill: int = 1024,
        enabled: bool = False,
    ) -> None:
        """This method is called to configure the Singleton instance

        It should be called once per instantiation per process.

        Note:
            The constructor keeps the state of instance disabled
            i.e no collection will happen even when start/stop methods are
            called. Only when enabled is True (self._off is True), the
            start/stop method pointers get assigned the real collection
            methods, otherwise they are initialized with null_method

        Args:
            world (int): World Size
            rank (int): The rank of this trainer
            mmcnt (int, optional): Number of ranks to print for showing Min/Max Etpt.
                                   Defaults to 1.
            amp (float, optional): Set to 3.0 if we only use timers in fwd pass.
                                   Defaults to 3.0.
            port (int, optional): Control port, useful only for rank-0. Defaults to 65535.
            prefill (int, optional): Howmany Events to pre-populate. Defaults to 1024.
            enabled (bool, optional): Whether or not collection is enabled on startup.
                                      Defaults to False.
        """
        if StragglerDetector._configured:
            # don't throw
            return
        StragglerDetector._configured = True
        self.bdata = False
        self.start = self.null_method
        self.stop = self.null_method
        self._off = True
        # No CUDA, No Support
        if torch.cuda.is_available():
            self._off = not enabled
            self.world = world
            self.rank = rank
            self.mmcnt = mmcnt if mmcnt > 1 else 1
            self.amp = amp
            self.port = port
            self.toggle = False
            self.bdata = False
            self.evt_q = queue.LifoQueue()
            self.start_gemm_ev = []
            self.stop_gemm_ev = []
            self.start_data_ev = []
            self.stop_data_ev = []
            self.start_gemm_tm = []
            self.stop_gemm_tm = []
            self.start_data_tm = []
            self.stop_data_tm = []
            backend = torch.distributed.get_backend()
            if backend == "nccl":
                self.dev = torch.cuda.current_device()
            else:
                self.dev = torch.device("cpu")
            # cache some events
            for _ in range(prefill):
                self.evt_q.put(torch.cuda.Event(enable_timing=True))
            if self.rank == 0:
                # Start the controller
                self._controller()
            if not self._off:
                self.start = self.start_method
                self.stop = self.stop_method

    def reset(self) -> None:
        """This method is called to reset the metrics state of the instance

        It is generally called from within elapsed() after extracting per rank metrics.
        """
        if self._off:
            return
        # Pool them
        if self.evt_q is not None:
            _ = [self.evt_q.put(ev) for ev in self.start_gemm_ev]
            _ = [self.evt_q.put(ev) for ev in self.stop_gemm_ev]
            _ = [self.evt_q.put(ev) for ev in self.start_data_ev]
            _ = [self.evt_q.put(ev) for ev in self.stop_data_ev]
        self.start_gemm_ev = []
        self.stop_gemm_ev = []
        self.start_data_ev = []
        self.stop_data_ev = []
        # Use regular timers
        self.start_gemm_tm = []
        self.stop_gemm_tm = []
        self.start_data_tm = []
        self.stop_data_tm = []
        self.bdata = False

    def start_method(self) -> None:
        """This method adds the start timers.

        Both cuda event and perf_counter are added. If bdata is set to
        true from __call__, this method skips inserting cuda
        timer. This way it can be used to measure time spent on
        CPU - generally useful for timing get_batch()
        """
        # Not reentrant
        if self.evt_q is not None and self.evt_q.qsize() > 1:
            sev = self.evt_q.get()  # no try-catch
            eev = self.evt_q.get()  # no try-catch
        else:
            sev = torch.cuda.Event(enable_timing=True)
            eev = torch.cuda.Event(enable_timing=True)
        # First check if this start is for data
        if self.bdata:
            self.start_data_ev.append(sev)
            self.stop_data_ev.append(eev)
            self.start_data_tm.append(0)
            self.stop_data_tm.append(0)
            idx = len(self.stop_data_tm) - 1
            self.start_data_tm[idx] = time.perf_counter_ns()
            self.start_data_ev[idx].record()
            self.bdata = False
            return
        self.start_gemm_ev.append(sev)
        self.stop_gemm_ev.append(eev)
        self.start_gemm_tm.append(0)
        self.stop_gemm_tm.append(0)
        idx = len(self.stop_gemm_tm) - 1
        self.start_gemm_tm[idx] = time.perf_counter_ns()
        self.start_gemm_ev[idx].record()

    def stop_method(self) -> None:
        """This method adds the stop timers.

        Both cuda event and perf_counter are added. If bdata is set to
        true from __call__, this method skips inserting cuda
        timer. Also see start_method()
        """
        # Not reentrant
        # First check if this stop is for data
        idx = len(self.stop_data_tm) - 1
        if idx >= 0 and self.stop_data_tm[idx] == 0:
            self.stop_data_tm[idx] = time.perf_counter_ns()
            self.stop_data_ev[idx].record()
            return
        idx = len(self.stop_gemm_tm) - 1
        if idx >= 0 and self.stop_gemm_tm[idx] == 0:
            self.stop_gemm_tm[idx] = time.perf_counter_ns()
            self.stop_gemm_ev[idx].record()

    def elapsed(self) -> Tuple[float, float, int, int, int, int]:
        """This method is called from report(), or can be called directly

         It is called to collect all the elapsed time since last reset().
         It finally calls reset()

        Returns:
            Tuple[float, float, int, int, int, int]: see below for returns
                delta       : time spent in kernel
                batch_delta : time spent in get_batch
                temp        : observed gpu temp
                power       : observed gpu power
                util        : observed gpu utilization
                clock       : observed gpu clock
        """
        if self._off:
            # match with return below
            return 0, 0, 0, 0, 0, 0
        ls_ev = len(self.start_gemm_ev)
        le_ev = len(self.stop_gemm_ev)
        ls_bs = len(self.start_data_ev)
        ls_be = len(self.stop_data_ev)
        delta = 0.0
        batch_delta = 0.0
        temp = 0
        power = 0
        clock = 0
        if ls_ev != le_ev:
            logger.warning(f"Event Start/Stop out of sync {ls_ev}/{le_ev}")
        elif ls_bs != ls_be:
            logger.warning(f"get_batch Start/Stop out of sync {ls_bs}/{ls_be}")
        else:
            temp = torch.cuda.temperature()
            power = torch.cuda.power_draw()
            util = torch.cuda.utilization()
            clock = torch.cuda.clock_rate()
            torch.cuda.synchronize()
            # Process Events
            for i in range(ls_ev):
                e_ev = self.start_gemm_ev[i].elapsed_time(self.stop_gemm_ev[i])
                e_tm = (self.stop_gemm_tm[i] - self.start_gemm_tm[i]) / 1e6  # ns to ms
                # Pick the larger of Event and perf_counter time?
                delta += max(e_ev, e_tm)
            # Process get_batch
            for i in range(ls_bs):
                b_ev = self.start_data_ev[i].elapsed_time(self.stop_data_ev[i])
                b_tm = (self.stop_data_tm[i] - self.start_data_tm[i]) / 1e6  # ns to ms
                # data fetching has prefetch, hence take the max, instead of avg
                batch_delta = max(batch_delta, max(b_ev, b_tm))
        self.reset()  # Prepare for next round
        # time in ms, batch_delta in ms, check return above
        return delta, batch_delta, temp, power, util, clock

    def report(self, total_flops: float = 0.0, log_interval: int = 0) -> bool:
        """Function to log the min/max metircs and the associated rank over a time period

        It finds the slowest and fastest rank among all ranks. It should be
        called by all ranks, but only rank-0 prints the analysis
        At the end it checks, if the straggler detector should
        remain active or if it should be deactivated.

        Args:
            total_flops (float, optional): The theoretical flops over the period. Defaults to 0.0.
            log_interval (int, optional): The training interval over which reporting is called(ms)
                                          Defaults to 0.

        Returns:
            bool: True if reported, else False
        """
        ret = False
        if not self._off and total_flops > 0.0 and log_interval > 0:
            elapsed, btime, temp, power, util, clock = self.elapsed()  # get raw time
            # btime (get_batch time is max in the iteration)
            ptime = elapsed / (log_interval * 1.0)  # avg per iteration elapsed time, ms
            api_flops = total_flops / (log_interval * 1.0)  # avg per iteration flops, ms
            apir_flops = api_flops / (
                ptime * 10**9 * self.world
            )  # this is avg per iteration this rank's thruput, TFLOP/s (note 10**9),
            et_flops = apir_flops / self.amp  # Estimated TFLOPs, not tracing backward

            o_dt = self._min_max(
                ptime, btime, float(temp), float(power), float(util), float(clock), et_flops
            )
            if self.rank == 0 and o_dt is not None and o_dt.aflops is not None:
                now = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]"
                min_flops, min_frank, _ = o_dt.aflops[0]()
                max_flops, max_frank, _ = o_dt.aflops[-1]()
                logger.info(
                    f"{now} | "
                    f"MnRtt/Rnk: {o_dt.min_elapsed} | "
                    f"MxRtt/Rnk: {o_dt.max_elapsed} | "
                    f"MnPwr/Rnk: {o_dt.min_power} | "
                    f"MxPwr/Rnk: {o_dt.max_power} | "
                    f"MnTmp/Rnk: {o_dt.min_temp} | "
                    f"MxTmp/Rnk: {o_dt.max_temp} | "
                    f"MnUtl/Rnk: {o_dt.min_util} | "
                    f"MxUtl/Rnk: {o_dt.max_util} | "
                    f"MnClk/Rnk: {o_dt.min_clock} | "
                    f"MxClk/Rnk: {o_dt.max_clock} | "
                    f"MnDRtt/Rnk: {o_dt.min_btime} | "
                    f"MxDRtt/Rnk: {o_dt.max_btime} | "
                    f"MnEtpt/Rnk: {min_flops:.2f}TF/{min_frank} | "
                    f"MxEtpt/Rnk: {max_flops:.2f}TF/{max_frank}"
                )
                if self.mmcnt > 1 and self.mmcnt < self.world:
                    line = f"^^^^ Bottom {self.mmcnt} Ranks with lowest  Etpt(TF):"
                    for i in range(self.mmcnt):
                        line += f" {o_dt.aflops[i]},"
                    logger.info(line)
                    line = f"^^^^ Top    {self.mmcnt} Ranks with highest Etpt(TF):"
                    shift = self.world - self.mmcnt
                    for i in range(self.mmcnt):
                        line += f" {o_dt.aflops[i+shift]},"
                    logger.info(line)
                ret = True

        # Check/Communicate if tracking is turned off or on
        self._check_toggle()
        return ret

    def _check_toggle(self) -> None:
        """Helper method to check if a request to toggle the collection state was made

        It checks iof collection state toggle req was made via the server listening on
        rank-0 since last call to report(). Called by report(). Calling this method
        indirectly from report() is the only way to activate the change that is made
        via rank-0
        """
        # If no change just commnunicate the current
        off = self._off
        if self.rank == 0 and self.toggle:
            off = not self._off
            self.toggle = False
        st = torch.tensor(off, dtype=torch.bool, device=self.dev)
        torch.distributed.broadcast(st, 0)  # Blocking
        # save old switch
        off = self._off
        self._off = bool(st.item())
        if off != self._off:
            if not self._off:
                self.start = self.start_method
                self.stop = self.stop_method
                state = "ON"
            else:
                self.start = self.null_method
                self.stop = self.null_method
                state = "OFF"
            if self.rank == 0:
                logger.info(f"Toggling StragglerDetector State {state}")

    def _handler(self) -> None:
        """Thread function for the controller.

        It is a tcp-server that listens on a port. Uses HTTP protocol.
        If connected to it using curl, it indicates a toggle of the
        collection state. The actual toggling happens at the end of
        calling report() when _check_toggle() is called.
        """
        resp = r"HTTP/1.0 200 OK\r\nConnection: Close\r\nContent-length: "

        if self.rank == 0:
            state = "OFF" if self._off else "ON"
            logger.info(
                f"Controller ready to recv " f"commands on port {self.port}. Current state {state}"
            )
            while True and self.sock is not None:
                try:
                    conn, _ = self.sock.accept()
                    _ = conn.recv(1024)
                    self.toggle = True
                    state = "ON" if self._off else "OFF"
                    msg = f"Will turn StragglerDetector {state} at next logging interval"
                    msg_len = len(msg)
                    final_resp = f"{resp}{msg_len}\r\n\r\n{msg}"
                    conn.send(final_resp.encode())
                    conn.close()
                    logger.info(msg)
                except Exception as err:
                    logger.error(f"Error in stragler handler.. {str(err)}")
                    return

    def _controller(self):
        """Installs a controller listener that is used to toggle collection state.

        Called from configure(). Ignored for all ranks other than rank-0
        """
        try:
            if self.rank == 0:
                neth = "0.0.0.0"
                netp = self.port
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.sock.bind((neth, netp))
                self.sock.listen(128)
                self.ctrlr = threading.Thread(
                    target=self._handler, args=(), name="straggler", daemon=True
                )
                self.ctrlr.start()
        except Exception as err:
            logger.warning(f"StragglerDetector cannot be controlled.. {str(err)}")

    def _min_max(
        self,
        ptime: float,
        btime: float,
        temp: float,
        power: float,
        util: float,
        clock: float,
        flops: float,
    ) -> Union[_StragglerData, None]:
        """Helper function to find the min/max values

        Args:
            ptime (float): avg per iteration gpu time
            btime (float): avg per iteration cpu time
            temp (float): gpu temp at the time of reporting
            power (float): gpu power at the time of reporting
            util (float): gpu util at the time of reporting
            clock (float): gpu clock at the time of reporting
            flops (float): estimated flops for the rank

        Returns:
            Union[_StragglerData, None]: It contains the min/max of few metrics and the
                                         corresponding rank it also has sorted list of
                                         all (flops, rank) sorted by flops (aflops)
                                         or returns None if collecton is disabled
        """
        if self._off:
            return None
        # initialize output data object
        o_dt = _StragglerData()

        prof_data: Dict[str, Union[int, float]] = {}
        data_list: List[Dict[str, Union[int, float]]] = []
        prof_data["rank"] = self.rank
        prof_data["time"] = ptime
        prof_data["btime"] = btime
        prof_data["temp"] = temp
        prof_data["power"] = power
        prof_data["util"] = util
        prof_data["clock"] = clock
        prof_data["flops"] = flops

        if self.rank == 0:
            data_list = [prof_data] * self.world

        # this is blocking by default
        torch.distributed.gather_object(prof_data, object_gather_list=data_list, dst=0)

        if self.rank == 0:
            min_ctime = min(data_list, key=lambda k: k["time"])  # elapsed
            max_ctime = max(data_list, key=lambda k: k["time"])  # elapsed

            min_cbatch = min(data_list, key=lambda k: k["btime"])  # batch time
            max_cbatch = max(data_list, key=lambda k: k["btime"])  # batch time

            min_ctemp = min(data_list, key=lambda k: k["temp"])  # temp
            max_ctemp = max(data_list, key=lambda k: k["temp"])  # temp

            min_cpower = min(data_list, key=lambda k: k["power"])  # power
            max_cpower = max(data_list, key=lambda k: k["power"])  # power

            min_cutil = min(data_list, key=lambda k: k["util"])  # gpu util
            max_cutil = max(data_list, key=lambda k: k["util"])  # gpu util

            min_cclock = min(data_list, key=lambda k: k["clock"])  # gpu clock
            max_cclock = max(data_list, key=lambda k: k["clock"])  # gpu clock

            min_val = min_ctime["time"]
            min_rank = min_ctime["rank"]
            max_val = max_ctime["time"]
            max_rank = max_ctime["rank"]
            o_dt.min_elapsed = _ValueWithRank(min_val, int(min_rank), "ms")
            o_dt.max_elapsed = _ValueWithRank(max_val, int(max_rank), "ms")

            min_val = min_cbatch["btime"]
            min_rank = min_cbatch["rank"]
            max_val = max_cbatch["btime"]
            max_rank = max_cbatch["rank"]
            o_dt.min_btime = _ValueWithRank(min_val, int(min_rank), "ms")
            o_dt.max_btime = _ValueWithRank(max_val, int(max_rank), "ms")

            min_val = min_ctemp["temp"]
            min_rank = min_ctemp["rank"]
            max_val = max_ctemp["temp"]
            max_rank = max_ctemp["rank"]
            o_dt.min_temp = _ValueWithRank(min_val, int(min_rank), "C")
            o_dt.max_temp = _ValueWithRank(max_val, int(max_rank), "C")

            min_val = min_cpower["power"]
            min_rank = min_cpower["rank"]
            max_val = max_cpower["power"]
            max_rank = max_cpower["rank"]
            o_dt.min_power = _ValueWithRank(min_val, int(min_rank), "W")
            o_dt.max_power = _ValueWithRank(max_val, int(max_rank), "W")

            min_val = min_cutil["util"]
            min_rank = min_cutil["rank"]
            max_val = max_cutil["util"]
            max_rank = max_cutil["rank"]
            o_dt.min_util = _ValueWithRank(min_val, int(min_rank), "%")
            o_dt.max_util = _ValueWithRank(max_val, int(max_rank), "%")

            min_val = min_cclock["clock"]
            min_rank = min_cclock["rank"]
            max_val = max_cclock["clock"]
            max_rank = max_cclock["rank"]
            o_dt.min_clock = _ValueWithRank(min_val, int(min_rank), "MHz")
            o_dt.max_clock = _ValueWithRank(max_val, int(max_rank), "MHz")

            o_dt.aflops = [
                _ValueWithRank(d.get("flops", 0.0), int(d.get("rank", -1)))
                for _, d in enumerate(data_list)
            ]
            o_dt.aflops.sort(key=lambda val_with_rank: val_with_rank()[0])
        # wait for everyone here
        torch.distributed.barrier()

        return o_dt

    @property
    def enabled(self) -> bool:
        """Can be called to check the enabled state of the instance

        Note:
            After the request to toggle the state, the
            actual state change happens at end of call
            to report()
        """
        return not self._off

    @property
    def configured(self) -> bool:
        """Can be called to check if the the instance is already configured

        Returns:
            bool: returns True if configure was called and was a success, else False
        """
        return StragglerDetector._configured

    @property
    def my_rank(self):
        """Can be called to get configured rank of this instance

        Returns:
            int: Configured rank for this instance
        """
        return self.rank

    @property
    def world_size(self) -> int:
        """Can be called to get configured world of this instance

        Returns:
            int: World size configured for this instance
        """
        return self.world

    def null_method(self) -> None:
        """Default method to initialize start/stop method ptrs"""
        pass

    def __enter__(self) -> "StragglerDetector":
        """Define context/instance entry

        Returns:
            StragglerDetector: the instance
        """
        self.start()
        return self

    def __call__(self, bdata: bool = False) -> "StragglerDetector":
        """Callable for the instance. Set context state,

        Useful when the context is used for cpu timers only when bdata=True

        Args:
            bdata (bool, optional): when true, only enables cpu timers. Defaults to False.

        Returns:
            StragglerDetector: the instance
        """
        self.bdata = bdata
        return self

    def __exit__(
        self,
        ex_type: Optional[Type[BaseException]],
        ex_val: Optional[BaseException],
        ex_tb: Optional[TracebackType],
    ) -> bool:
        """Define context/instance exit, calls the stop method

        Args:
            ex_type (Optional[Type[BaseException]]): Exception type
            ex_val (Optional[BaseException]): _description_
            ex_tb (Optional[TracebackType]): _description_

        Returns:
            bool: True if the exception was handled
        """
        # Should not suppress errors even if turned off
        if ex_type is not None:
            err = traceback.format_exception(ex_type, ex_val, ex_tb)
            logger.warning(f"{str(ex_val)}\n{err}")
        self.stop()
        return False


# Singleton, global visibility
__straggler__ = StragglerDetector()
"""StragglerDetector: private module variable, not be directly accessed
"""


# Check if Transformer Engine has Float8Tensor class
HAVE_TE_FLOAT8TENSOR = False
try:
    from transformer_engine.pytorch.float8_tensor import Float8Tensor

    HAVE_TE_FLOAT8TENSOR = True
except (ImportError, ModuleNotFoundError):
    # Float8Tensor not found
    pass


def is_float8tensor(tensor: torch.Tensor) -> bool:
    """Check if a tensor is a Transformer Engine Float8Tensor"""
    return HAVE_TE_FLOAT8TENSOR and isinstance(tensor, Float8Tensor)
