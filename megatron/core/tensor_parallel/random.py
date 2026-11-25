# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

# Parts of the code here are adapted from PyTorch
# repo: https://github.com/pytorch/pytorch
from __future__ import annotations

import contextlib
import logging
from typing import Optional, Union

import torch
from torch import _C
from torch.cuda import _lazy_call, _lazy_init
from torch.cuda import device as device_ctx_manager

from megatron.core.parallel_state import (
    get_expert_gtp_weight_remat_rank,
    get_expert_gtp_weight_remat_world_size,
    get_expert_model_parallel_rank,
    get_expert_tensor_parallel_rank,
    get_gtp_weight_remat_rank,
    get_gtp_weight_remat_world_size,
    get_tensor_model_parallel_rank,
)
from megatron.core.utils import is_te_min_version, is_torch_min_version

try:
    import transformer_engine  # pylint: disable=unused-import

    HAVE_TE = True
except ModuleNotFoundError:
    HAVE_TE = False


# Default name for the model parallel rng tracker.
_MODEL_PARALLEL_RNG_TRACKER_NAME = 'model-parallel-rng'
_EXPERT_PARALLEL_RNG_TRACKER_NAME = 'expert-parallel-rng'
_DATA_PARALLEL_RNG_TRACKER_NAME = 'data-parallel-rng'
# GTP_remat weight-init trackers: shards init per-rank, so each peer must draw DIFFERENT values;
# registered only when the axis is active (see model_parallel_cuda_manual_seed).
_GTP_REMAT_RNG_TRACKER_NAME = 'gtp-remat-rng'
_EXPERT_GTP_REMAT_RNG_TRACKER_NAME = 'egtp-remat-rng'


def _get_cuda_rng_state(
    device: Union[int, str, torch.device] = "cuda", clone: bool = False, graph_safe: bool = False
) -> torch.Tensor:
    """Return the random number generator state of the specified GPU.

    Arguments:
        device (int): The gpu to retrieve the rng state
        clone (bool): Whether to also clone the retrieved RNG state
        graph_safe (bool): Get the rng state in a graph safe manner.

    This function is adapted from torch.cuda.random.get_rng_state()"""

    # if not using cuda graphs, just use the builtin pytorch function
    if not graph_safe:
        return torch.cuda.random.get_rng_state(device=device)

    _lazy_init()
    if isinstance(device, str):
        device = torch.device(device)
    elif isinstance(device, int):
        device = torch.device("cuda", device)
    idx = device.index
    if idx is None:
        idx = torch.cuda.current_device()

    default_generator = torch.cuda.default_generators[idx]
    if clone:
        return default_generator.clone_state()
    return default_generator.graphsafe_get_state()


def _set_cuda_rng_state(new_state: torch.Tensor, device: int = -1, graph_safe: bool = False):
    """Sets the random number generator state of the current GPU.

    Arguments:
        new_state (torch.ByteTensor): The desired state
        device (int): The gpu to retrieve the rng state
        graph_safe (bool): Set the rng state in a graph safe manner.

    This function is adapted from PyTorch repo (torch.cuda.set_rng_state)
    with a single change: the input state is not cloned. Cloning caused
    major performance issues for +4 GPU cases.
    """
    if hasattr(_C, '_cuda_setRNGState') and callable(_C._cuda_setRNGState):
        # older PyTorch
        def cb():
            with device_ctx_manager(device):
                _C._cuda_setRNGState(new_state)

    else:
        # newer PyTorch
        if device == -1:
            device = torch.device('cuda')
        elif isinstance(device, str):
            device = torch.device(device)
        elif isinstance(device, int):
            device = torch.device('cuda', device)

        def cb():
            idx = device.index
            if idx is None:
                idx = torch.cuda.current_device()
            default_generator = torch.cuda.default_generators[idx]

            # if graph capturing, set the rng state in a cudagraphable way
            if graph_safe:
                default_generator.graphsafe_set_state(new_state)
            else:
                default_generator.set_state(new_state)

    _lazy_call(cb)


def convert_cuda_rng_state(
    state: Union[torch.Tensor, torch.Generator], to_graphable: bool = False
) -> Union[torch.Tensor, torch.Generator]:
    """
    Convert the cuda rng state tensor to the graphable version,
    or from the graphable version to the non-graphable tensor version.
    """
    if to_graphable:
        if isinstance(state, torch.Tensor):
            # Convert to the graphable version.
            # Store current rng state.
            orig_cuda_rng_state = _get_cuda_rng_state(graph_safe=False)
            # Set rng state to the desired one
            _set_cuda_rng_state(state, graph_safe=False)
            # Get the graphable state
            graphable_state = _get_cuda_rng_state(clone=True, graph_safe=True)
            # And set the state to the original state we started with.
            _set_cuda_rng_state(orig_cuda_rng_state, graph_safe=False)
            return graphable_state
        elif isinstance(state, torch.Generator):
            # already graphable, just return it.
            return state
        else:
            raise ValueError(f"Invalid state type: {type(state)}")
    else:
        if isinstance(state, torch.Tensor):
            # already non-graphable, just return it.
            return state
        elif isinstance(state, torch.Generator):
            # Convert to the non-graphable tensor version.
            return state.get_state()
        else:
            raise ValueError(f"Invalid state type: {type(state)}")


def get_expert_parallel_rng_tracker_name():
    """Get the expert parallel rng tracker name"""
    global _EXPERT_PARALLEL_RNG_TRACKER_NAME
    return _EXPERT_PARALLEL_RNG_TRACKER_NAME


def get_data_parallel_rng_tracker_name():
    """Get the data parallel rng tracker name"""
    global _DATA_PARALLEL_RNG_TRACKER_NAME
    return _DATA_PARALLEL_RNG_TRACKER_NAME


def get_gtp_remat_rng_tracker_name(is_expert=False):
    """Get the (E)GTP_remat weight-init rng tracker name (per-(E)GTP-rank distinct draws)."""
    return _EXPERT_GTP_REMAT_RNG_TRACKER_NAME if is_expert else _GTP_REMAT_RNG_TRACKER_NAME


class CudaRNGStatesTracker:
    """Tracker for the cuda RNG states.

    Using the `add` method, a cuda rng state is initialized based on
    the input `seed` and is assigned to `name`. Later, by forking the
    rng state, we can perform operations and return to our starting
    cuda state.
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

        # Map from a string name to the cuda rng state.
        self.states_ = {}

        # Seeds are just for book keeping and ensure no seed is set twice.
        self.seeds_ = set()

        # Name of the rng state currently being used in the generator.
        # The default one is "default-rng" and won't be pushed to the self.states_ dictionary.
        self._current_state_name = "default-rng"

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
            raise Exception('cuda rng state {} already exists'.format(name))

        # If available, create the state in a graph safe manner
        if self.use_cudagraphable_rng:
            new_state = _get_cuda_rng_state(clone=True, graph_safe=True)
            new_state.manual_seed(seed)
            self.states_[name] = new_state
        else:
            # Get the current rng state.
            orig_rng_state = torch.cuda.get_rng_state()
            # Set the new state and store it.
            torch.cuda.manual_seed(seed)
            self.states_[name] = torch.cuda.get_rng_state()
            # Reset rng state to what it was.
            _set_cuda_rng_state(orig_rng_state)

    @contextlib.contextmanager
    def fork(self, name=_MODEL_PARALLEL_RNG_TRACKER_NAME):
        """Fork the cuda rng state, perform operations, and exit with
        the original state."""
        # Check if we have added the state
        if name not in self.states_:
            raise Exception('cuda rng state {} is not added'.format(name))
        # Store current rng state and name. Store in self.states_ if it's not the default state.
        orig_cuda_rng_state = _get_cuda_rng_state(graph_safe=self.use_cudagraphable_rng)
        orig_state_name = self._current_state_name
        if orig_state_name != "default-rng":
            self.states_[orig_state_name] = orig_cuda_rng_state
        # Set rng state and name to the desired one.
        _set_cuda_rng_state(self.states_[name], graph_safe=self.use_cudagraphable_rng)
        self._current_state_name = name
        # Record cpu RNG state
        cpu_rng_state = torch.get_rng_state()
        # Do the stuff we wanted to do.
        try:
            yield
        finally:
            # Throw a warning if cpu RNG state changed
            if not torch.all(cpu_rng_state == torch.get_rng_state()).item():
                logging.getLogger(__name__).warning('CPU RNG state changed within GPU RNG context')
            # Check if the current state name is the same as the desired state name.
            if self._current_state_name != name:
                raise Exception(
                    f'current state name {self._current_state_name} is not the same as the desired '
                    f'state name {name}.'
                )
            # Update the current rng state for later use.
            self.states_[name] = _get_cuda_rng_state(graph_safe=self.use_cudagraphable_rng)
            # And set the state and name to the original state we started with.
            if orig_state_name != "default-rng":
                orig_cuda_rng_state = self.states_[orig_state_name]
            _set_cuda_rng_state(orig_cuda_rng_state, graph_safe=self.use_cudagraphable_rng)
            self._current_state_name = orig_state_name


# RNG tracker object.
_CUDA_RNG_STATE_TRACKER = None
_CUDA_RNG_STATE_TRACKER_INITIALIZED = False


def initialize_rng_tracker(
    use_te_rng_tracker: bool = False,
    inference_rng_tracker: bool = False,
    use_cudagraphable_rng: bool = False,
    force_reset: bool = False,
):
    """Create the RNG tracker. 'use_te_rng_tracker' determines whether to use
    Megatron or TransformerEngine's implementation.
    In particular, TransformerEngine's implementation is cudagraphable and supports FP8.
    """
    global _CUDA_RNG_STATE_TRACKER
    global _CUDA_RNG_STATE_TRACKER_INITIALIZED
    if force_reset:
        _CUDA_RNG_STATE_TRACKER = None
        _CUDA_RNG_STATE_TRACKER_INITIALIZED = False

    if _CUDA_RNG_STATE_TRACKER_INITIALIZED:
        return

    # Get the base tracker class
    base_tracker = None
    if HAVE_TE and use_te_rng_tracker:
        if not is_te_min_version("1.5.0"):
            raise RuntimeError("use_te_rng_tracker requires TransformerEngine version >= 1.5")
        from megatron.core.extensions.transformer_engine import TECudaRNGStatesTracker

        base_tracker = TECudaRNGStatesTracker
        tracker_kwargs = {"is_inference_rng_tracker": inference_rng_tracker}
    else:
        base_tracker = CudaRNGStatesTracker
        tracker_kwargs = {
            "use_cudagraphable_rng": use_cudagraphable_rng,
            "is_inference_rng_tracker": inference_rng_tracker,
        }

    if inference_rng_tracker:

        class InferenceCudaRNGStatesTracker(base_tracker):  # type: ignore[valid-type, misc]
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

        tracker_class = InferenceCudaRNGStatesTracker
    else:
        tracker_class = base_tracker

    _CUDA_RNG_STATE_TRACKER = tracker_class(**tracker_kwargs)
    _CUDA_RNG_STATE_TRACKER_INITIALIZED = True


def get_cuda_rng_tracker(
    use_te_rng_tracker: bool = False,
    inference_rng_tracker: bool = False,
    use_cudagraphable_rng: bool = False,
):
    """Get cuda rng tracker."""
    initialize_rng_tracker(use_te_rng_tracker, inference_rng_tracker, use_cudagraphable_rng)
    return _CUDA_RNG_STATE_TRACKER


def get_all_rng_states():
    """Returns all generator states used by the current `CudaRNGStatesTracker`."""

    assert (
        _CUDA_RNG_STATE_TRACKER_INITIALIZED
    ), "Tried getting all rng states but RNG Tracker has not been initalized!"

    if isinstance(_CUDA_RNG_STATE_TRACKER, CudaRNGStatesTracker):
        return _CUDA_RNG_STATE_TRACKER.states_
    # If TE is installed, check if we are using TE's RNG tracker
    elif HAVE_TE and is_te_min_version("1.5.0"):
        from megatron.core.extensions.transformer_engine import TECudaRNGStatesTracker

        if isinstance(_CUDA_RNG_STATE_TRACKER, TECudaRNGStatesTracker):
            from transformer_engine.pytorch.distributed import get_all_rng_states

            return get_all_rng_states()
    # no valid tracker, return an empty dict
    else:
        return {}


_CUDAGRAPH_NEEDS_GENERATOR_REGISTRATION = None


def _probe_cudagraph_needs_generator_registration() -> bool:
    """Capture an RNG op on an unregistered generator and report whether capture rejects it."""
    # cuda_graphs imports this module, so import its capture helper here.
    from megatron.core.transformer.cuda_graphs import no_gc_during_capture

    generator = torch.Generator(device="cuda")
    graph = torch.cuda.CUDAGraph()
    try:
        with no_gc_during_capture(), torch.cuda.graph(graph, capture_error_mode="thread_local"):
            torch.rand(1, device="cuda", generator=generator)
    except RuntimeError:
        # Without lazy registration, capture fails with "RNG op during graph capture but
        # generator is not registered with the capturing graph". Treat any capture failure
        # as needing registration: on PyTorch with lazy registration, registering anyway
        # only costs a deprecation warning, while skipping it on older builds is fatal.
        return True
    finally:
        del graph
    return False


def cudagraph_needs_generator_registration() -> bool:
    """Whether generators must be registered with a `torch.cuda.CUDAGraph` before capture.

    PyTorch with pytorch/pytorch#176753 lazily registers every generator whose Philox
    state is consumed during capture, and `CUDAGraph.register_generator_state()` became a
    deprecated no-op that prints a warning on *every* call. Skip the explicit registration
    there: it does nothing, and with one call per layer, per graph and per generator it floods
    stderr (tens of thousands of lines per rank for dynamic inference with CUDA graphs).

    The version string cannot identify that change: 2.14.0a0 nightlies built before it landed
    still require explicit registration. Builds older than 2.14.0a0 always require it; for
    newer builds, a small capture probe decides, and its result is cached.
    """
    global _CUDAGRAPH_NEEDS_GENERATOR_REGISTRATION
    if _CUDAGRAPH_NEEDS_GENERATOR_REGISTRATION is not None:
        return _CUDAGRAPH_NEEDS_GENERATOR_REGISTRATION
    if not is_torch_min_version("2.14.0a0"):
        needs_registration = True
    elif torch.cuda.is_current_stream_capturing():
        # The probe cannot capture while another capture is active; register to be safe
        # and probe on a later call.
        return True
    else:
        needs_registration = _probe_cudagraph_needs_generator_registration()
    _CUDAGRAPH_NEEDS_GENERATOR_REGISTRATION = needs_registration
    return needs_registration


def prime_cuda_rng_states_for_graph_capture() -> None:
    """Create each CUDA generator's capture state at the start of a graph capture.

    Call as the first operation inside a `torch.cuda.graph` capture. PyTorch >= 2.14
    (pytorch/pytorch#176753) creates this state lazily on the generator's first Philox use in
    the capture and allocates it on the default stream. Mid-capture, that allocation makes the
    caching allocator record and query events for blocks freed earlier in the capture; if a
    block's recorded stream has since joined the capture (e.g. an NCCL stream after a pipeline
    recv), the query raises cudaErrorCapturedEvent and invalidates the capture
    (pytorch/pytorch#193982). Priming here, while nothing is pending, restores the pre-2.14
    behavior of `capture_begin`. Each primed generator advances by one tiny draw per replay.
    No-op outside a capture and on PyTorch < 2.14.
    """
    # TODO: Remove once the minimum supported PyTorch includes the upstream fix for lazy RNG
    # capture-state allocation (pytorch/pytorch#193993 or its successor).
    if cudagraph_needs_generator_registration() or not torch.cuda.is_current_stream_capturing():
        return
    scratch = torch.empty(1, device=torch.cuda.current_device())
    generators = [torch.cuda.default_generators[scratch.device.index]]
    if _CUDA_RNG_STATE_TRACKER_INITIALIZED:
        generators.extend((get_all_rng_states() or {}).values())
    for generator in generators:
        # Graph-safe trackers hold generators; tensor states cannot be drawn from under capture.
        if isinstance(generator, torch.Generator) and generator.device == scratch.device:
            scratch.uniform_(generator=generator)


def model_parallel_cuda_manual_seed(
    seed: int,
    te_rng_tracker: bool = False,
    inference_rng_tracker: bool = False,
    use_cudagraphable_rng: bool = False,
    tp_rank: Optional[int] = None,
    ep_rank: Optional[int] = None,
    etp_rank: Optional[int] = None,
    gtp_remat_rank: Optional[int] = None,
    egtp_remat_rank: Optional[int] = None,
    force_reset_rng: bool = False,
    gtp_remat_world_size: Optional[int] = None,
    egtp_remat_world_size: Optional[int] = None,
):
    """Initialize model parallel cuda seed.

    This function should be called after the model parallel is
    initialized. Also, no torch.cuda.manual_seed should be called
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
    if tp_rank is None:
        tp_rank = get_tensor_model_parallel_rank()
    if ep_rank is None:
        ep_rank = get_expert_model_parallel_rank()
    if etp_rank is None:
        etp_rank = get_expert_tensor_parallel_rank()
    if gtp_remat_rank is None:
        gtp_remat_rank = get_gtp_weight_remat_rank()
    if egtp_remat_rank is None:
        egtp_remat_rank = get_expert_gtp_weight_remat_rank()
    if gtp_remat_world_size is None:
        gtp_remat_world_size = get_gtp_weight_remat_world_size()
    if egtp_remat_world_size is None:
        egtp_remat_world_size = get_expert_gtp_weight_remat_world_size()
    # 2718 is just for fun and any POSITIVE value will work.
    offset = seed + 2718
    tensor_model_parallel_seed = offset + tp_rank
    # Data parallel gets the original seed.
    data_parallel_seed = seed

    initialize_rng_tracker(
        te_rng_tracker, inference_rng_tracker, use_cudagraphable_rng, force_reset=force_reset_rng
    )
    _CUDA_RNG_STATE_TRACKER.reset()
    # Set the default state.
    torch.cuda.manual_seed(data_parallel_seed)
    _CUDA_RNG_STATE_TRACKER.add(_DATA_PARALLEL_RNG_TRACKER_NAME, data_parallel_seed)

    # and model parallel state.
    _CUDA_RNG_STATE_TRACKER.add(_MODEL_PARALLEL_RNG_TRACKER_NAME, tensor_model_parallel_seed)

    expert_parallel_seed = seed + 1024 + 100 * ep_rank + etp_rank
    _CUDA_RNG_STATE_TRACKER.add(_EXPERT_PARALLEL_RNG_TRACKER_NAME, expert_parallel_seed)

    # GTP_remat weight-init states: shards are initialized per-rank (GTP-agnostic init), so peers
    # must draw DIFFERENT values (everything above is identical across peers by design). The 65536
    # stride keeps these disjoint from the tp/ep/etp seeds. Added only when the axis is active, so
    # non-GTP runs keep a byte-identical tracker set (and checkpoint rng payload).
    if gtp_remat_world_size > 1:
        gtp_remat_seed = tensor_model_parallel_seed + 65536 * (1 + gtp_remat_rank)
        _CUDA_RNG_STATE_TRACKER.add(_GTP_REMAT_RNG_TRACKER_NAME, gtp_remat_seed)
    if egtp_remat_world_size > 1:
        egtp_remat_seed = expert_parallel_seed + 32768 + 65536 * (1 + egtp_remat_rank)
        _CUDA_RNG_STATE_TRACKER.add(_EXPERT_GTP_REMAT_RNG_TRACKER_NAME, egtp_remat_seed)


def is_graph_safe_cuda_rng_tracker(cuda_rng_tracker):
    """Check if the cuda rng tracker is graph safe version."""
    if HAVE_TE and is_te_min_version("1.5.0"):
        from megatron.core.extensions.transformer_engine import TECudaRNGStatesTracker

        if isinstance(cuda_rng_tracker, TECudaRNGStatesTracker):
            return True
    if getattr(cuda_rng_tracker, "use_cudagraphable_rng", False):
        return True
    return False


def _get_all_rng_states():
    """Get all the rng states."""
    cpu_rng_state = torch.get_rng_state()
    cuda_rng_state = _get_cuda_rng_state(
        graph_safe=is_graph_safe_cuda_rng_tracker(get_cuda_rng_tracker())
    )
    cuda_rng_state_tracker = get_cuda_rng_tracker().get_states()
    return cpu_rng_state, cuda_rng_state, cuda_rng_state_tracker


def _set_all_rng_states(cpu_rng_state, cuda_rng_state, cuda_rng_state_tracker):
    """Set all the rng states."""
    torch.set_rng_state(cpu_rng_state)
    _set_cuda_rng_state(
        cuda_rng_state, graph_safe=is_graph_safe_cuda_rng_tracker(get_cuda_rng_tracker())
    )
    get_cuda_rng_tracker().set_states(cuda_rng_state_tracker)


@contextlib.contextmanager
def _fork_rng():
    """Fork the rng state."""
    # Store the current states.
    current_states = _get_all_rng_states()
    try:
        yield
    finally:
        # Set the states back to what it was at the start of this function.
        _set_all_rng_states(*current_states)


# Activation checkpointing moved to megatron.core.activation_policy.recompute.checkpointing.
from megatron.core.activation_policy.recompute.checkpointing import (  # pylint: disable=unused-import
    CheckpointFunction,
    CheckpointWithoutOutput,
    CheckpointWithoutOutputManager,
    _ArgLayout,
    _as_tuple,
    _get_checkpoint_without_output_storage,
    _get_share_storage,
    _InputStash,
    _mark_checkpoint_without_output_tensor,
    _RecomputeRegion,
    _RegionFunction,
    _RegionState,
    _register_rebuild_hook,
    _set_checkpointing,
    _TERecomputeContext,
    _unset_checkpointing,
    checkpoint,
    is_checkpoint_without_output_tensor,
    is_checkpointing,
)
