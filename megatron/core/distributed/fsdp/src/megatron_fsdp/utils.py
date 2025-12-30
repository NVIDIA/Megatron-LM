# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
import inspect
import logging
import operator
from contextlib import nullcontext
from functools import reduce
from importlib.metadata import version
from typing import Callable, Optional, Sequence, Union

try:
    import einops

    HAVE_EINOPS = True
except ImportError:
    HAVE_EINOPS = False

import torch
from packaging.version import Version as PkgVersion
from torch import _C
from torch.cuda import _lazy_call, _lazy_init
from torch.cuda import device as device_ctx_manager
from torch.distributed import DeviceMesh, ProcessGroup

logger = logging.getLogger(__name__)

try:
    import transformer_engine  # pylint: disable=W0611

    HAVE_TE = True
except (ImportError, ModuleNotFoundError):
    # Transformer Engine not found
    HAVE_TE = False


_MODEL_PARALLEL_RNG_TRACKER_NAME = "model-parallel-rng"


def get_te_version():
    """Get TE version from __version__; if not available use pip's. Use caching."""
    if not HAVE_TE:
        # No TE installed, so return None.
        return None

    def get_te_version_str():
        import transformer_engine as te

        if hasattr(te, "__version__"):
            return str(te.__version__)
        else:
            return version("transformer-engine")

    return PkgVersion(get_te_version_str())


def is_te_min_version(vers, check_equality=True):
    """Check if minimum version of `transformer-engine` is installed."""
    te_version = get_te_version()
    if not isinstance(te_version, PkgVersion):
        # No TE installed, so cannot satisfy any version requirement.
        return False

    if check_equality:
        return te_version >= PkgVersion(vers)
    return te_version > PkgVersion(vers)


def is_submodule(module, parent_module, strict=True):
    """
    Check if a module is a submodule of another module.
    """
    if strict:
        if module is parent_module:
            return False
    for m in parent_module.modules():
        if m is module:
            return True
    return False


def is_float8tensor(tensor: torch.Tensor) -> bool:
    """Check if a tensor is a Transformer Engine Float8Tensor.

    Note that in TE2.x, in order to support more recipes, the design of the fp8 tensor class has
    changed. Now Float8Tensor is only used for current scaling and delayed scaling. And mxfp8
    and blockwise scaling have their own fp8 tensor classes. These different fp8 tensor classes
    are both inherited from QuantizedTensor. So, for TE1.x, FP8_TENSOR_CLASS is Float8Tensor,
    and for TE2.x, FP8_TENSOR_CLASS is QuantizedTensor.
    """
    return HAVE_TE_FP8_TENSOR_CLASS and isinstance(tensor, FP8_TENSOR_CLASS)


def get_mesh_names(
    device_mesh: Optional[DeviceMesh] = None, only_submesh_dims: bool = False
) -> list[str]:
    """
    Get all the sub-mesh ("dp", "cp", etc.) and flattened-mesh ("dp_cp", etc.) names
    in the DeviceMesh. When only_submesh_dims=True, only checks for sub-mesh dimensions.
    """
    if device_mesh is None:
        # Device mesh does not exist.
        return []

    # Sub-mesh dimension names.
    submesh_dim_names = (
        list(device_mesh.mesh_dim_names) if device_mesh.mesh_dim_names is not None else []
    )

    # Flattened mesh dimension names.
    try:
        # Retrieve all flattened meshes associated with DeviceMesh.
        # The flattened DeviceMesh are all located in the _flatten_mapping
        # dictionary of the root DeviceMesh.
        flatten_mesh_names = [
            flat_dim
            for flat_dim, flat_mesh in device_mesh._get_root_mesh()._flatten_mapping.items()
        ]
    except AttributeError:
        # Fallback to the DeviceMesh global state to retrieve flattened
        # meshes associated with the DeviceMesh.
        from torch.distributed.device_mesh import _mesh_resources

        flatten_mesh_names = [
            child_mesh_dim_name
            for child_mesh, root_mesh in _mesh_resources.child_to_root_mapping.items()
            for child_mesh_dim_name in (child_mesh.mesh_dim_names or [])
            if root_mesh == device_mesh and child_mesh_dim_name not in submesh_dim_names
        ]

    # Order of the returned list of mesh dimension names must match the index
    # of the root mesh dimension names followed by flattened sub-meshes:
    # [<root mesh dimension names>, <flattened mesh dimension names>]
    if only_submesh_dims:
        return submesh_dim_names
    else:
        return submesh_dim_names + flatten_mesh_names


def contains_submesh(
    device_mesh: Optional[DeviceMesh], submesh_names: Optional[str | Sequence[str]]
) -> bool:
    """
    Check if a sub-mesh exists in the device mesh by name.
    """
    if device_mesh is None or submesh_names is None:
        # Device mesh does not exist.
        return False
    if isinstance(submesh_names, str):
        submesh_names = (submesh_names,)
    device_mesh_names = get_mesh_names(device_mesh)
    return all(submesh_name in device_mesh_names for submesh_name in submesh_names)


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
    if hasattr(_C, "_cuda_setRNGState") and callable(_C._cuda_setRNGState):
        # older PyTorch
        def cb():
            with device_ctx_manager(device):
                _C._cuda_setRNGState(new_state)

    else:
        # newer PyTorch
        if device == -1:
            device = torch.device("cuda")
        elif isinstance(device, str):
            device = torch.device(device)
        elif isinstance(device, int):
            device = torch.device("cuda", device)

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

    if "_CUDA_RNG_STATE_TRACKER_INITIALIZED" in globals() and _CUDA_RNG_STATE_TRACKER_INITIALIZED:
        return

    # Get the base tracker class
    base_tracker = None
    if HAVE_TE and use_te_rng_tracker:
        if not is_te_min_version("1.5.0"):
            raise RuntimeError("use_te_rng_tracker requires TransformerEngine version >= 1.5")

        class TECudaRNGStatesTracker(transformer_engine.pytorch.distributed.CudaRNGStatesTracker):
            """Wraps TransformerEngine's CudaRNGStatesTracker so that it is
            interchangeable with Megatron's RNG tracker"""

            def __init__(self, is_inference_rng_tracker=False):
                super().__init__()
                self.reset()
                self.is_inference_rng_tracker = is_inference_rng_tracker

            def is_initialized(self):
                """Checks if the internal RNG state has been set with set_states()."""
                return self._is_initialized

            def reset(self):
                """Reset the internal RNG state."""
                super().reset()
                self._is_initialized = False

            def set_states(self, states):
                """Set the internal RNG state."""
                super().set_states(states)
                self._is_initialized = True

            def add(self, name, seed):
                """Track the rng state."""
                super().add(name, seed)
                self._is_initialized = True

        base_tracker = TECudaRNGStatesTracker
        tracker_kwargs = {"is_inference_rng_tracker": inference_rng_tracker}
    else:

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
                    raise Exception("seed {} already exists".format(seed))
                self.seeds_.add(seed)
                # Check that state is not already defined.
                if name in self.states_:
                    raise Exception("cuda rng state {} already exists".format(name))

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
                    raise Exception("cuda rng state {} is not added".format(name))
                # Store current rng state.
                orig_cuda_rng_state = _get_cuda_rng_state(graph_safe=self.use_cudagraphable_rng)
                # Set rng state to the desired one
                _set_cuda_rng_state(self.states_[name], graph_safe=self.use_cudagraphable_rng)
                # Record cpu RNG state
                cpu_rng_state = torch.get_rng_state()
                # Do the stuff we wanted to do.
                try:
                    yield
                finally:
                    # Throw a warning if cpu RNG state changed
                    if not torch.all(cpu_rng_state == torch.get_rng_state()).item():
                        logging.getLogger(__name__).warning(
                            "CPU RNG state changed within GPU RNG context"
                        )
                    # Update the current rng state for later use.
                    self.states_[name] = _get_cuda_rng_state(graph_safe=self.use_cudagraphable_rng)
                    # And set the state to the original state we started with.
                    _set_cuda_rng_state(orig_cuda_rng_state, graph_safe=self.use_cudagraphable_rng)

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


class FSDPDistributedIndex:
    """
    Class containing references to the process groups utilized by Megatron-FSDP.

    This class tracks the device mesh and different process groups required
    for full-sharded data parallelism (FSDP), including support for hybrid
    and tensor/data parallel strategies.
    """

    def __init__(
        self,
        device_mesh: DeviceMesh,
        dp_shard_dim: Optional[str] = None,
        dp_outer_dim: Optional[str] = None,
        tp_dim: Optional[str] = None,
        hybrid_fsdp_group: Optional[torch.distributed.ProcessGroup] = None,
        hsdp_outer_dp_shard: bool = False,
        expt_device_mesh: Optional[DeviceMesh] = None,
    ):
        """
        Args:
            device_mesh (DeviceMesh): The DeviceMesh to use for the DistributedIndex.
            dp_shard_dim (Optional[str]): The dimension name of the data parallel
                (and context parallel) sharding sub-mesh.
            dp_outer_dim (Optional[str]): The dimension name of the "outer" data parallel
                sub-mesh for replication or sharding when using HSDP.
            tp_dim (Optional[str]): The dimension name of the tensor parallel sub-mesh.
            hybrid_fsdp_group (Optional[torch.distributed.ProcessGroup]): The
                process group for hybrid FSDP communication, which is the flattened
                combination of the dp_outer and dp_shard process groups.
            hsdp_outer_dp_shard (bool): Whether to have outer DP group sharding
                in hybrid FSDP. Specifying outer sharding will lift the bucket sharding
                coordinate system to flattened ranks of (dp_shard, dp_outer) instead of
                just sharding across dp_shard ranks and replicating across dp_outer ranks.
            expt_device_mesh (Optional[DeviceMesh]): The expert parallel device mesh
                to use for the DistributedIndex.
        """
        # Device mesh arguments.
        self.device_mesh = device_mesh
        self.dp_shard_dim = dp_shard_dim
        self.dp_outer_dim = dp_outer_dim
        self.tp_dim = tp_dim
        # Helper flag to denote if we are using hybrid FSDP.
        self.use_hybrid_fsdp = dp_outer_dim is not None
        # Helper flag to denote if we are outer-sharding in hybrid FSDP.
        self.hsdp_outer_dp_shard = hsdp_outer_dp_shard
        self.expt_device_mesh = expt_device_mesh

        # Handling the situation where M-Core MoE EP=1
        if self.expt_device_mesh is None:
            self.expt_device_mesh = device_mesh

        # Hybrid FSDP Process Groups
        # Retrieve the FSDP process group from the DeviceMesh.
        self.fsdp_group = (
            self.device_mesh[self.dp_shard_dim].get_group()
            if contains_submesh(self.device_mesh, self.dp_shard_dim)
            else None
        )
        # Retrieve the outer-FSDP process group from the DeviceMesh.
        self.outer_fsdp_group = (
            self.device_mesh[self.dp_outer_dim].get_group()
            if contains_submesh(self.device_mesh, self.dp_outer_dim)
            else None
        )
        # Save a reference to the overall HSDP process group, which is the flattened
        # combination of the outer-FSDP and FSDP process groups.
        self.hybrid_fsdp_group = hybrid_fsdp_group

        # Retrieve the expert parallel process groups from the DeviceMesh.
        self.expt_fsdp_group = (
            self.expt_device_mesh[self.dp_shard_dim].get_group()
            if self.expt_device_mesh is not None
            and contains_submesh(self.expt_device_mesh, self.dp_shard_dim)
            else None
        )

        """
        Megatron-FSDP is responsible for storing all required DeviceMesh
        as per best practices recommended by the DeviceMesh API.

        NOTE(@cspades): In PyTorch 2.11, retrieving flattened mesh dimensions
        will be impossible via the device_mesh[...] API. We will require all
        users to correctly _unflatten() their DeviceMesh such that all
        dimensions used by Megatron-FSDP are sub-meshes of the DeviceMesh.
        contains_submesh(...) -> get_mesh_names(only_submesh_dims=True).
        """
        self.mesh_library = {}

        def register_submesh(device_mesh, submesh, is_expert_parallel):
            """Register a submesh with identifier: (*submesh, is_expert_parallel)
            in the mesh library."""
            if contains_submesh(device_mesh, submesh):
                submesh_identifier = tuple(list(submesh) + [is_expert_parallel])
                self.mesh_library[submesh_identifier] = device_mesh[submesh]

        # Define common submesh patterns
        tp_submesh = (self.tp_dim,)
        hsdp_tp_submesh = (self.dp_outer_dim, self.dp_shard_dim, self.tp_dim)
        fsdp_tp_submesh = (self.dp_shard_dim, self.tp_dim)
        hsdp_submesh = (self.dp_outer_dim, self.dp_shard_dim)
        fsdp_submesh = (self.dp_shard_dim,)

        # Register non-EP submeshes
        register_submesh(self.device_mesh, tp_submesh, False)
        register_submesh(self.device_mesh, hsdp_tp_submesh, False)
        register_submesh(self.device_mesh, fsdp_tp_submesh, False)
        register_submesh(self.device_mesh, hsdp_submesh, False)
        register_submesh(self.device_mesh, fsdp_submesh, False)

        # Register EP submeshes
        if self.expt_device_mesh is not None:
            register_submesh(self.device_mesh, hsdp_submesh, True)
            register_submesh(self.device_mesh, hsdp_tp_submesh, True)
            register_submesh(self.expt_device_mesh, tp_submesh, True)
            register_submesh(self.expt_device_mesh, fsdp_tp_submesh, True)
            register_submesh(self.expt_device_mesh, fsdp_submesh, True)

        # Validate FSDP arguments.
        if self.fsdp_group is None:
            raise ValueError(
                "Megatron-FSDP (FSDPDistributedIndex) requires an FSDP process group "
                "(dp_shard_dim, fsdp_group) for core functionality."
            )

        # Validate HSDP arguments.
        if self.use_hybrid_fsdp:
            if self.outer_fsdp_group is None:
                raise ValueError(
                    "[FSDPDistributedIndex][use_hybrid_fsdp=True] Hybrid FSDP requires "
                    "an outer-DP process group (dp_outer_dim, outer_fsdp_group)."
                )
            if self.hybrid_fsdp_group is None:
                raise ValueError(
                    "[FSDPDistributedIndex][use_hybrid_fsdp=True] Hybrid FSDP requires "
                    "a hybrid FSDP process group (hybrid_fsdp_group). "
                    "This group can be manufactured by flattening the outer-DP "
                    "(dp_outer_dim, outer_fsdp_group) and FSDP (dp_shard_dim, fsdp_group) "
                    "process groups or sub-meshes."
                )

    def get_submesh(
        self, mesh_dim_names: str | Sequence[str], is_expert_parallel: bool = False
    ) -> DeviceMesh:
        """
        Retrieve an Megatron-FSDP-registered submesh by name(s).
        """
        if isinstance(mesh_dim_names, str):
            mesh_dim_names = (mesh_dim_names,)

        # Construct submesh identifier: (*mesh_dim_names, is_expert_parallel)
        submesh_identifier = tuple(list(mesh_dim_names) + [is_expert_parallel])

        # Retrieve the submesh from the mesh library
        device_submesh = self.mesh_library.get(submesh_identifier, None)

        if device_submesh is None:
            # Warn about not specifying tp_dim for layers or frameworks that depend on this.
            if self.tp_dim is None and not is_expert_parallel:
                logger.warning(
                    "[FSDPDistributedIndex] Note: For TransformerEngine, or "
                    "other machine learning frameworks like Megatron that assume "
                    "TP=1, you must specify tp_dim to use Megatron-FSDP. "
                    "Create a trivial TP dimension by setting the TP dimension size "
                    "to 1 in the DeviceMesh.\n"
                    f"DeviceMesh: {self.device_mesh}"
                )
            elif self.tp_dim is None and is_expert_parallel:
                logger.warning(
                    "[FSDPDistributedIndex] Note: For TransformerEngine, or "
                    "other machine learning frameworks like Megatron that assume "
                    "ETP=1, you must specify tp_dim to use Megatron-FSDP. "
                    "Create a trivial ETP dimension by setting the ETP dimension size "
                    "to 1 in the DeviceMesh.\n"
                    f"DeviceMesh: {self.expt_device_mesh}"
                )

            raise ValueError(
                f"[FSDPDistributedIndex][get_submesh] No submesh with "
                f"mesh_dim_names={mesh_dim_names}, is_expert_parallel={is_expert_parallel} "
                f"has been registered with Megatron-FSDP."
            )

        return device_submesh

    def get_dp_group(self, is_expert_parallel: bool = False) -> ProcessGroup:
        """Get the data parallel process group."""
        if is_expert_parallel:
            return self.expt_fsdp_group
        if self.use_hybrid_fsdp:
            return self.hybrid_fsdp_group
        return self.fsdp_group

    def get_fsdp_group(self, is_expert_parallel: bool = False) -> ProcessGroup:
        """Get the FSDP process group."""
        if is_expert_parallel:
            return self.expt_fsdp_group
        return self.fsdp_group

    def get_outer_fsdp_group(self) -> ProcessGroup:
        """Get the outer-FSDP process group."""
        if not self.use_hybrid_fsdp:
            return None
        return self.outer_fsdp_group

    def get_root_mesh(self, is_expert_parallel: bool = False) -> DeviceMesh:
        """Get the device mesh."""
        # NOTE(@cspades): This is FSDPDistributedIndex's root mesh, NOT the actual
        # root mesh that the DeviceMesh or expert DeviceMesh was un-flattened from.
        # To get the root mesh, use: DeviceMesh._get_root_mesh().
        if is_expert_parallel:
            return self.expt_device_mesh
        return self.device_mesh

    def get_logical_hybrid_fsdp_rank(self):
        """
        Returns the logical rank of the current process within the full-shard hybrid FSDP group.

        In full-shard hybrid FSDP, parameters are first sharded across the inner
        data-parallel group, then across the outer data-parallel group. This changes
        the effective rank mapping compared to standard data parallelism. Use this
        method to get the correct rank index for the hybrid group.

        Returns:
            int: The index of the current process in the hybrid FSDP group.

        Raises:
            AssertionError: If full-shard hybrid FSDP is not enabled.
        """
        assert HAVE_EINOPS, "get_logical_hybrid_fsdp_rank requires einops to be installed."
        assert (
            self.hsdp_outer_dp_shard
        ), "get_logical_hybrid_fsdp_rank is only valid when full-shard hybrid FSDP is enabled."

        if not hasattr(self, "_hybrid_fsdp_group_ranks"):
            dp_world_size = self.get_dp_group().size()

            # Reorder the flat ranks: (outer_dp, inner_dp) -> (inner_dp, outer_dp)
            mesh = einops.rearrange(
                torch.arange(dp_world_size),
                "(outer_dp inner_dp) -> (inner_dp outer_dp)",
                outer_dp=self.outer_fsdp_group.size(),
                inner_dp=self.fsdp_group.size(),
            )
            self._hybrid_fsdp_group_ranks = mesh.tolist()

        # Find the index for the current rank in the hybrid group
        return self._hybrid_fsdp_group_ranks.index(self.hybrid_fsdp_group.rank())


class GlobalMemoryBuffer:
    """Global buffer to avoid dynamic memory allocations.
    Caller should ensure that buffers of the same name
    are not used concurrently."""

    def __init__(self):
        self.buffer = {}

    def get_tensor(self, tensor_shape, dtype, name, mem_alloc_context: Optional[Callable] = None):
        """
        Returns (potentially) a sub-tensor from the self.buffer for the given shape.
        """
        required_len = reduce(operator.mul, tensor_shape, 1)
        if (
            self.buffer.get((name, dtype), None) is None
            or self.buffer[(name, dtype)].numel() < required_len
        ):
            mem_alloc_context = mem_alloc_context if mem_alloc_context else nullcontext
            with mem_alloc_context():
                self.buffer[(name, dtype)] = torch.empty(
                    required_len,
                    dtype=dtype,
                    device=torch.cuda.current_device(),
                    requires_grad=False,
                )

        return self.buffer[(name, dtype)][0:required_len].view(*tensor_shape)


def get_global_memory_buffer():
    """Return the global GlobalMemoryBuffer object"""
    global _GLOBAL_MEMORY_BUFFER
    if "_GLOBAL_MEMORY_BUFFER" not in globals() or _GLOBAL_MEMORY_BUFFER is None:
        _GLOBAL_MEMORY_BUFFER = GlobalMemoryBuffer()
    return _GLOBAL_MEMORY_BUFFER


def create_updated_function_signature(original_function, **extended_kwargs: dict):
    """
    Given a function, create a new version of the function with
    extended keyword-only arguments or parameters. Used to patch
    or extend methods in instances of a class.
    """
    # Get the original function signature.
    params = list(inspect.signature(original_function).parameters.values())

    # Add new keyword-only parameters
    for name, value in extended_kwargs.items():
        params.append(
            inspect.Parameter(
                name,
                kind=inspect.Parameter.KEYWORD_ONLY,
                default=value,
                annotation=(type(value) if value is not None else inspect.Parameter.empty),
            )
        )

    # Return the updated function signature.
    return inspect.Signature(params)


def is_mcore_tensor_model_parallel(param: torch.Tensor) -> bool:
    """
    Check if the given parameter is Megatron-Core tensor model parallel.
    """
    return getattr(param, "_mcore_tp", False) or getattr(param, "tensor_model_parallel", False)


def is_mcore_tensor_parallel_duplicated(param: torch.Tensor) -> bool:
    """
    Check if the given parameter is Megatron-Core tensor model parallel and duplicated.
    """
    return getattr(param, "_tp_duplicated", False)


def get_mcore_tensor_parallel_partition_dim(param: torch.Tensor) -> Optional[int]:
    """
    Get the partition dimension for a Megatron-Core tensor model parallel parameter.
    """
    if is_mcore_tensor_model_parallel(param):
        if hasattr(param, "_tp_partition_dim"):
            return param._tp_partition_dim
        else:
            return param.partition_dim
    return None
