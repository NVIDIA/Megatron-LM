# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

# Parts of the code here are adapted from PyTorch
# repo: https://github.com/pytorch/pytorch

import contextlib
import logging
from functools import partial
from typing import Union

from megatron.core.device_utils import get_current_rng_state, get_xla_model, set_current_rng_state, set_device_manual_seed
import torch
from pkg_resources import packaging
from torch.utils.checkpoint import detach_variable

from megatron.core.parallel_state import (
    get_expert_model_parallel_rank,
    get_expert_tensor_parallel_rank,
    get_tensor_model_parallel_rank,
)
from megatron.core.utils import is_te_min_version, safely_set_viewless_tensor_data

from .utils import gather_split_1d_tensor, split_tensor_into_1d_equal_chunks

try:
    import transformer_engine  # pylint: disable=unused-import

    HAVE_TE = True
except ModuleNotFoundError:
    HAVE_TE = False


# Default name for the model parallel rng tracker.
_MODEL_PARALLEL_RNG_TRACKER_NAME = 'model-parallel-rng'
_EXPERT_PARALLEL_RNG_TRACKER_NAME = 'expert-parallel-rng'
_DATA_PARALLEL_RNG_TRACKER_NAME = 'data-parallel-rng'

def get_expert_parallel_rng_tracker_name():
    """Get the expert parallel rng tracker name"""
    global _EXPERT_PARALLEL_RNG_TRACKER_NAME
    return _EXPERT_PARALLEL_RNG_TRACKER_NAME


def get_data_parallel_rng_tracker_name():
    """Get the data parallel rng tracker name"""
    global _DATA_PARALLEL_RNG_TRACKER_NAME
    return _DATA_PARALLEL_RNG_TRACKER_NAME


class DeviceRNGStatesTracker:
    """Tracker for the device RNG states.

    Using the `add` method, a device rng state is initialized based on
    the input `seed` and is assigned to `name`. Later, by forking the
    rng state, we can perform operations and return to our starting
    device state.
    """

    def __init__(self, use_cudagraphable_rng=False, is_inference_rng_tracker=False):
        self.reset()
        self.use_cudagraphable_rng = use_cudagraphable_rng
        self.is_inference_rng_tracker = is_inference_rng_tracker

        if self.use_cudagraphable_rng:
            assert (
                hasattr(torch.cuda.CUDAGraph, "register_generator_state")
                and hasattr(torch.Generator, "graphsafe_set_state")
                and hasattr(torch.Generator, "graphsafe_get_state")
                and hasattr(torch.Generator, "clone_state")
            ), "Tried using cudagraphs with RNG, however not detected in pytorch!"

    def is_initialized(self):
        """Checks if the internal RNG state has been set wirth set_states()."""
        return self._is_initialized

    def reset(self):
        """Set to the initial state (no tracker)."""

        # Track if initialized.
        self._is_initialized = False

        # Map from a string name to the device rng state.
        self.states_ = {}

        # Seeds are just for book keeping and ensure no seed is set twice.
        self.seeds_ = set()

    def get_states(self):
        """Get rng states. Copy the dictionary so we have direct
        pointers to the states, not just a pointer to the dictionary."""
        states = {}
        for name in self.states_:
            states[name] = self.states_[name]
        return states

    def set_states(self, states):
        """Set the rng states. For efficiency purposes, we do not check
        the size of seed for compatibility."""
        self._is_initialized = True
        self.states_ = states

    def add(self, name, seed):
        """Track the rng state."""
        self._is_initialized = True
        # Check seed is not already used.
        if seed in self.seeds_:
            raise Exception('seed {} already exists'.format(seed))
        self.seeds_.add(seed)
        # Check that state is not already defined.
        if name in self.states_:
            raise Exception('device rng state {} already exists'.format(name))
        # Get the current rng state.
        orig_rng_state = get_current_rng_state()
            
        # Set the new state and store it.
        set_device_manual_seed(seed)
        self.states_[name] = get_current_rng_state()
        # Reset rng state to what it was.
        set_current_rng_state(orig_rng_state)

    @contextlib.contextmanager
    def fork(self, name=_MODEL_PARALLEL_RNG_TRACKER_NAME):
        """Fork the device rng state, perform operations, and exit with
        the original state."""
        # Check if we have added the state
        if name not in self.states_:
            raise Exception('device rng state {} is not added'.format(name))
        # Store current rng state.
        orig_rng_state = get_current_rng_state()
        # Set rng state to the desired one
        set_current_rng_state(self.states_[name])
        # Record cpu RNG state
        cpu_rng_state = torch.get_rng_state()
        # Do the stuff we wanted to do.
        try:
            yield
        finally:
            # Throw a warning if cpu RNG state changed
            if not torch.all(cpu_rng_state == torch.get_rng_state()).item():
                logging.getLogger(__name__).warning('CPU RNG state changed within GPU RNG context')
            # Update the current rng state for later use.
            self.states_[name] = get_current_rng_state()
            # And set the state to the original state we started with.
            set_current_rng_state(orig_rng_state)


# RNG tracker object.
_DEVICE_RNG_STATE_TRACKER = None
_DEVICE_RNG_STATE_TRACKER_INITIALIZED = False


def initialize_rng_tracker(
    use_te_rng_tracker: bool = False,
    inference_rng_tracker: bool = False,
    use_cudagraphable_rng: bool = False,
):
    """Create the RNG tracker. 'use_te_rng_tracker' determines whether to use
    Megatron or TransformerEngine's implementation.
    In particular, TransformerEngine's implementation is cudagraphable and supports FP8.
    """


    global _DEVICE_RNG_STATE_TRACKER
    global _DEVICE_RNG_STATE_TRACKER_INITIALIZED
    if _DEVICE_RNG_STATE_TRACKER_INITIALIZED:
        return

    if get_xla_model() and use_te_rng_tracker:
        import warnings
        warnings.warn("XLA model fall back: use_te_rng_tracker=False")
        use_te_rng_tracker = False

    # Get the base tracker class
    base_tracker = None
    if HAVE_TE and use_te_rng_tracker:
        if not is_te_min_version("1.5.0"):
            raise RuntimeError("use_te_rng_tracker requires TransformerEngine version >= 1.5")
        from megatron.core.extensions.transformer_engine import TECudaRNGStatesTracker

        base_tracker = TECudaRNGStatesTracker
        tracker_kwargs = {"is_inference_rng_tracker": inference_rng_tracker}
    else:
        base_tracker = partial(DeviceRNGStatesTracker, use_cudagraphable_rng=use_cudagraphable_rng)

    if inference_rng_tracker:

        class InferenceDeviceRNGStatesTracker(base_tracker):
            """RNG tracker for inference."""

            def add(self, name, seed):
                """Mirrors the interface from the training RNG tracker."""
                pass

            def set_states(self, states):
                """Mirrors the interface from the training RNG tracker."""
                pass

            def fork(self, name=_MODEL_PARALLEL_RNG_TRACKER_NAME):
                """Mirrors the interface from the training RNG tracker."""
                return contextlib.nullcontext()

        tracker_class = InferenceDeviceRNGStatesTracker
    else:
        tracker_class = base_tracker

    _DEVICE_RNG_STATE_TRACKER = tracker_class()
    _DEVICE_RNG_STATE_TRACKER_INITIALIZED = True


def get_device_rng_tracker():
    """Get device rng tracker."""
    initialize_rng_tracker()
    return _DEVICE_RNG_STATE_TRACKER


def get_all_rng_states():
    """Returns all generator states used by the current `DeviceRNGStatesTracker`."""

    assert (
        _DEVICE_RNG_STATE_TRACKER_INITIALIZED
    ), "Tried getting all rng states but RNG Tracker has not been initalized!"

    if isinstance(_DEVICE_RNG_STATE_TRACKER, DeviceRNGStatesTracker):
        return _DEVICE_RNG_STATE_TRACKER.states_
    # If TE is installed, check if we are using TE's RNG tracker
    elif HAVE_TE and is_te_min_version("1.5.0"):
        from megatron.core.extensions.transformer_engine import TECudaRNGStatesTracker

        if isinstance(_DEVICE_RNG_STATE_TRACKER, TECudaRNGStatesTracker):
            from transformer_engine.pytorch.distributed import get_all_rng_states

            return get_all_rng_states()
    # no valid tracker, return an empty dict
    else:
        return {}
    
def model_parallel_device_manual_seed(seed, te_rng_tracker=False, inference_rng_tracker=False):
    """Initialize model parallel device seed.

    This function should be called after the model parallel is
    initialized. Also, no set_manual_seed should be called
    after this function. Basically, this is replacement for that
    function.
    Three set of RNG states are tracked:
    default state: This is for data parallelism and is the same among a set of model parallel GPUs
    but different across different model parallel groups. This is used for example for dropout
    in the non-tensor-model-parallel regions.
    tensor-model-parallel state: This state is different among a set of model parallel GPUs,
    but the same across data parallel groups. This is used for example for dropout
    in model parallel regions.
    expert-parallel-seed: This state is only used for the expert layer of MoE models.
    It is different among expert-tensor and expert-model parallel GPUs, and the same
    across expert-data parallel groups.
    """
    # 2718 is just for fun and any POSITIVE value will work.
    offset = seed + 2718
    tensor_model_parallel_seed = offset + get_tensor_model_parallel_rank()
    # Data parallel gets the original seed.
    data_parallel_seed = seed

    initialize_rng_tracker(te_rng_tracker, inference_rng_tracker)
    _DEVICE_RNG_STATE_TRACKER.reset()
    
    # Set the default state.
    set_device_manual_seed(data_parallel_seed)
    _DEVICE_RNG_STATE_TRACKER.add(_DATA_PARALLEL_RNG_TRACKER_NAME, data_parallel_seed)

    # and model parallel state.
    _DEVICE_RNG_STATE_TRACKER.add(_MODEL_PARALLEL_RNG_TRACKER_NAME, tensor_model_parallel_seed)

    expert_parallel_seed = (
        seed + 1024 + 100 * get_expert_model_parallel_rank() + get_expert_tensor_parallel_rank()
    )
    _DEVICE_RNG_STATE_TRACKER.add(_EXPERT_PARALLEL_RNG_TRACKER_NAME, expert_parallel_seed)


class CheckpointFunction(torch.autograd.Function):
    """Checkpoint Function

    This function is adapted from torch.utils.checkpoint with two main changes:
    1) torch.cuda.set_rng_state is replaced with `set_rng_state`
    2) the states in the model parallel tracker are also properly tracked/set/reset.
    """

    # pylint: disable=missing-function-docstring
    @staticmethod
    def forward(ctx, run_function, distribute_saved_activations, *args):
        """Forward pass."""
        ctx.run_function = run_function
        ctx.distribute_saved_activations = distribute_saved_activations

        # Copy the rng states.
        ctx.fwd_cpu_rng_state = torch.get_rng_state()
        ctx.fwd_device_rng_state = get_current_rng_state()
        ctx.fwd_device_rng_state_tracker = get_device_rng_tracker().get_states()

        with torch.no_grad():
            outputs = run_function(*args)

        # Divide hidden states across model parallel group and only keep
        # the chunk corresponding to the current rank.
    
        if distribute_saved_activations:
            ctx.input_0_shape = args[0].data.shape
            split_tensor = split_tensor_into_1d_equal_chunks(args[0].data, new_buffer=True)
            xm = get_xla_model()
            if xm:
                target_device = args[0].get_device()
                if target_device != split_tensor.get_device():
                    split_tensor = split_tensor.cpu() if target_device == -1 else split_tensor.to(device=target_device)
            safely_set_viewless_tensor_data(args[0], split_tensor)

        # Store everything.
        ctx.save_for_backward(*args)

        return outputs

    # pylint: disable=missing-function-docstring
    @staticmethod
    def backward(ctx, *args):
        """Backward pass."""
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError(
                "Checkpointing is not compatible with .grad(), "
                "please use .backward() if possible"
            )
        inputs = ctx.saved_tensors
        if ctx.distribute_saved_activations:
            safely_set_viewless_tensor_data(
                inputs[0], gather_split_1d_tensor(inputs[0].data).view(ctx.input_0_shape)
            )

        # Store the current states.
        bwd_cpu_rng_state = torch.get_rng_state()
        bwd_device_rng_state = get_current_rng_state()
        bwd_device_rng_state_tracker = get_device_rng_tracker().get_states()

        # Set the states to what it used to be before the forward pass.
        torch.set_rng_state(ctx.fwd_cpu_rng_state)
        set_current_rng_state(ctx.fwd_device_rng_state)
        get_device_rng_tracker().set_states(ctx.fwd_device_rng_state_tracker)

        # Compute the forward pass.
        detached_inputs = detach_variable(inputs)
        with torch.enable_grad():
            outputs = ctx.run_function(*detached_inputs)

        # Set the states back to what it was at the start of this function.
        torch.set_rng_state(bwd_cpu_rng_state)
        set_current_rng_state(bwd_device_rng_state)
        get_device_rng_tracker().set_states(bwd_device_rng_state_tracker)

        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)

        # filter out non tensor outputs for backward pass
        outputs, args = zip(
            *filter(lambda x: torch.is_tensor(x[0]) and x[0].requires_grad, zip(outputs, args))
        )
        torch.autograd.backward(outputs, args)
        grads = tuple(inp.grad if isinstance(inp, torch.Tensor) else inp for inp in detached_inputs)
        return (None, None) + grads


def checkpoint(function, distribute_saved_activations, *args):
    """Checkpoint a model or part of the model.
    This has been directly copied from torch.utils.checkpoint."""
    return CheckpointFunction.apply(function, distribute_saved_activations, *args)
