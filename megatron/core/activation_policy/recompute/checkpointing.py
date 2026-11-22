# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

# Parts of the code here are adapted from PyTorch
# repo: https://github.com/pytorch/pytorch
"""Activation checkpointing: reentrant checkpointing and output-discarding recompute.

Moved from ``megatron.core.tensor_parallel.random``, which re-exports these names.
"""

from __future__ import annotations

import contextlib
import enum
import functools
import weakref
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Optional, TypeVar

import torch
from torch.utils.checkpoint import detach_variable
from torch.utils.cpp_extension import load_inline
from typing_extensions import TypeVarTuple, Unpack

from megatron.core.tensor_observation import suspend_tensor_observations
from megatron.core.utils import safely_set_viewless_tensor_data

if TYPE_CHECKING:
    from megatron.core.transformer.transformer_config import TransformerConfig

try:
    import transformer_engine  # pylint: disable=unused-import
    from transformer_engine.pytorch.distributed import activation_recompute_forward
    from transformer_engine.pytorch.fp8 import FP8GlobalStateManager, fp8_autocast

    HAVE_TE = True
except ModuleNotFoundError:
    HAVE_TE = False


def _tp_random():
    """``megatron.core.tensor_parallel.random`` (RNG state helpers), looked up at call time:
    it re-exports this module, so it cannot be imported here at module load."""
    from megatron.core.tensor_parallel import random

    return random


def _tp_utils():
    """``megatron.core.tensor_parallel.utils``, looked up at call time (see :func:`_tp_random`)."""
    from megatron.core.tensor_parallel import utils

    return utils


# ---------------------------------------------------------------------------
# C++ extension: zero-copy storage sharing for CheckpointWithoutOutput
# ---------------------------------------------------------------------------
# Makes dst's UntypedStorage point to src's data WITHOUT copying bytes.
# Holds a refcounted reference to src's StorageImpl so the memory stays alive.
# Operates below the Tensor / autograd layer → no version-counter bump,
# and ALL TensorImpls that reference dst's StorageImpl (including views
# created by reshape / split / etc. inside TE GroupedLinear) see the data.
# ---------------------------------------------------------------------------

_SHARE_STORAGE_SRC = r"""
#include <torch/extension.h>

void share_storage(at::Tensor dst, at::Tensor src) {
    auto* dst_impl = dst.storage().unsafeGetStorageImpl();

    // Copy src's c10::Storage (increments StorageImpl refcount).
    auto* src_storage_ref = new c10::Storage(src.storage());

    void*       data   = src_storage_ref->data_ptr().get();
    size_t      nbytes = src_storage_ref->nbytes();
    c10::Device device = src_storage_ref->device();

    // Build a DataPtr whose deleter releases our StorageImpl reference.
    c10::DataPtr shared(
        data,
        static_cast<void*>(src_storage_ref),
        [](void* ctx) { delete static_cast<c10::Storage*>(ctx); },
        device);

    dst_impl->set_data_ptr(std::move(shared));
    dst_impl->set_nbytes(nbytes);
}
"""

_share_storage_ext = None

# Set on the output StorageImpl wrapper used by CheckpointWithoutOutput.  The
# storage-level marker is visible through aliases/views of the output tensor.
_CHECKPOINT_WITHOUT_OUTPUT_STORAGE_ATTR = "_mcore_checkpoint_without_output"


def _get_checkpoint_without_output_storage(tensor: torch.Tensor) -> torch.UntypedStorage | None:
    """Return a tensor's storage, or None for storage-less tensor subclasses."""
    try:
        return tensor.untyped_storage()
    except (AttributeError, NotImplementedError, RuntimeError):
        return None


def _mark_checkpoint_without_output_tensor(tensor: torch.Tensor) -> None:
    """Mark a tensor's storage as owned by CheckpointWithoutOutput."""
    storage = _get_checkpoint_without_output_storage(tensor)
    if storage is not None:
        setattr(storage, _CHECKPOINT_WITHOUT_OUTPUT_STORAGE_ATTR, True)


def is_checkpoint_without_output_tensor(tensor: torch.Tensor) -> bool:
    """Return whether a tensor aliases a CheckpointWithoutOutput output storage."""
    storage = _get_checkpoint_without_output_storage(tensor)
    return storage is not None and bool(
        getattr(storage, _CHECKPOINT_WITHOUT_OUTPUT_STORAGE_ATTR, False)
    )


def _get_share_storage():
    """Lazily compile & cache the share_storage extension."""
    global _share_storage_ext
    if _share_storage_ext is None:
        _share_storage_ext = load_inline(
            name="share_storage_ext",
            cpp_sources=_SHARE_STORAGE_SRC,
            functions=["share_storage"],
            verbose=False,
        )
    return _share_storage_ext.share_storage


# Global flag that's toggled whenever inside a checkpointing context
IS_CHECKPOINTING = False


def _set_checkpointing():
    """Set state to checkpointing enabled."""
    global IS_CHECKPOINTING
    IS_CHECKPOINTING = True


def _unset_checkpointing():
    """Unset state to checkpointing enabled."""
    global IS_CHECKPOINTING
    IS_CHECKPOINTING = False


def is_checkpointing():
    """Check if currently in a checkpoint context."""
    return IS_CHECKPOINTING


_R = TypeVar('_R')
_Ts = TypeVarTuple('_Ts')


class CheckpointFunction(torch.autograd.Function):
    """Checkpoint Function

    This function is adapted from torch.utils.checkpoint with two main changes:
    1) torch.cuda.set_rng_state is replaced with `_set_cuda_rng_state`
    2) the states in the model parallel tracker are also properly tracked/set/reset.
    """

    # pylint: disable=missing-function-docstring
    @staticmethod
    def forward(
        ctx: Any,
        run_function: Callable[[Unpack[_Ts]], _R],
        distribute_saved_activations: bool,
        *args: Unpack[_Ts],
    ) -> _R:
        """Forward pass."""
        _set_checkpointing()

        ctx.run_function = run_function
        ctx.distribute_saved_activations = distribute_saved_activations
        # Activation recomputation must use the same autocast policy as the original forward.
        # Otherwise it replays under the ambient autocast state.
        ctx.autocast_enabled = torch.is_autocast_enabled("cuda")
        ctx.autocast_dtype = torch.get_autocast_dtype("cuda")
        ctx.autocast_cache_enabled = torch.is_autocast_cache_enabled()

        # Copy the rng states.
        ctx.rng_states = _tp_random()._get_all_rng_states()

        with torch.no_grad():
            outputs = run_function(*args)

        # Divide hidden states across model parallel group and only keep
        # the chunk corresponding to the current rank.
        if distribute_saved_activations:
            ctx.input_0_shape = args[0].data.shape
            safely_set_viewless_tensor_data(
                args[0],
                _tp_utils().split_tensor_into_1d_equal_chunks(args[0].data, new_buffer=True),
            )

        # Store everything.
        ctx.save_for_backward(*args)

        _unset_checkpointing()
        return outputs

    # pylint: disable=missing-function-docstring
    @staticmethod
    def backward(ctx, *args):
        """Backward pass."""
        from megatron.core.transformer.cuda_graphs import is_graph_capturing

        if not torch.autograd._is_checkpoint_valid() and not is_graph_capturing():
            raise RuntimeError(
                "Checkpointing is not compatible with .grad(), "
                "please use .backward() if possible"
            )
        _set_checkpointing()

        inputs = ctx.saved_tensors
        if ctx.distribute_saved_activations:
            safely_set_viewless_tensor_data(
                inputs[0],
                _tp_utils().gather_split_1d_tensor(inputs[0].data).view(ctx.input_0_shape),
            )

        with _tp_random()._fork_rng():
            # Set the states to what it used to be before the forward pass.
            _tp_random()._set_all_rng_states(*ctx.rng_states)

            # Compute the forward pass.
            detached_inputs = detach_variable(inputs)
            autocast_context = torch.autocast(
                device_type="cuda",
                dtype=ctx.autocast_dtype,
                enabled=ctx.autocast_enabled,
                cache_enabled=ctx.autocast_cache_enabled,
            )
            with torch.enable_grad(), autocast_context, suspend_tensor_observations():
                outputs = ctx.run_function(*detached_inputs)

        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)

        # filter out non tensor outputs for backward pass
        outputs, args = zip(
            *filter(lambda x: torch.is_tensor(x[0]) and x[0].requires_grad, zip(outputs, args))
        )
        torch.autograd.backward(outputs, args)
        grads = tuple(inp.grad if isinstance(inp, torch.Tensor) else inp for inp in detached_inputs)

        _unset_checkpointing()
        return (None, None) + grads


def checkpoint(
    function: Callable[[Unpack[_Ts]], _R], distribute_saved_activations: bool, *args: Unpack[_Ts]
) -> _R:
    """Checkpoint a model or part of the model.
    This has been directly copied from torch.utils.checkpoint."""
    from megatron.core.transformer.cuda_graphs import is_graph_capturing, is_graph_warmup

    # Skip checkpointing during CUDA graph warmup and capture, matching the behavior of
    # CheckpointWithoutOutput. The graph captures all ops directly; recomputation cannot
    # run inside a captured graph.
    if is_graph_warmup() or is_graph_capturing():
        return function(*args)
    return CheckpointFunction.apply(function, distribute_saved_activations, *args)


class _RegionState(enum.Enum):
    """Lifecycle of a recompute region. Transitions only move forward."""

    RECORDED = enum.auto()  # function ran without building a graph; inputs are stashed
    RELEASED = enum.auto()  # output storages handed back to the allocator
    REBUILT = enum.auto()  # function re-run with grad; output storages hold the new data
    DONE = enum.auto()  # gradients delivered; the region holds nothing


def _as_tuple(value: Any) -> tuple:
    """View a function result as a tuple of outputs."""
    return tuple(value) if isinstance(value, (tuple, list)) else (value,)


class _ArgLayout:
    """Positional layout of a call: tensors travel through autograd, other values stay here."""

    def __init__(self, args: tuple):
        self.tensor_positions = tuple(
            i for i, arg in enumerate(args) if isinstance(arg, torch.Tensor)
        )
        self._template = [None if isinstance(arg, torch.Tensor) else arg for arg in args]

    def tensors_of(self, args: tuple) -> tuple:
        """Return the tensor arguments in positional order."""
        return tuple(args[i] for i in self.tensor_positions)

    def rebuild(self, tensors: tuple) -> list:
        """Return the full argument list with `tensors` placed back at their positions."""
        args = list(self._template)
        for position, tensor in zip(self.tensor_positions, tensors):
            args[position] = tensor
        return args


class _TERecomputeContext:
    """Transformer Engine state that the recomputation has to reproduce.

    The first pass is marked ``recompute_phase=False`` so TE keeps its FP8 scaling state for the
    second pass, which runs under the forward-time ``fp8_autocast`` settings and is marked
    ``recompute_phase=True``.
    """

    def __init__(self, enabled: bool):
        self.enabled = enabled and HAVE_TE
        self.fp8_enabled = self.enabled and FP8GlobalStateManager.is_fp8_enabled()
        self.fp8_recipe = FP8GlobalStateManager.get_fp8_recipe() if self.fp8_enabled else None

    def first_pass(self):
        """Context for the forward pass, which does not record a graph."""
        if not self.enabled:
            return contextlib.nullcontext()
        return activation_recompute_forward(activation_recompute=True, recompute_phase=False)

    @contextlib.contextmanager
    def second_pass(self):
        """Context for the recomputation in the backward pass."""
        if not self.enabled:
            yield
            return
        with fp8_autocast(enabled=self.fp8_enabled, fp8_recipe=self.fp8_recipe):
            with activation_recompute_forward(activation_recompute=True, recompute_phase=True):
                yield


class _InputStash(torch.autograd.Function):
    """Holds the tensor inputs of a region in autograd's saved-tensor storage.

    Saving through ``save_for_backward`` keeps the inputs visible to active
    ``saved_tensors_hooks`` (e.g. fine-grained activation offloading) and to the CUDA graph
    saved-tensor observer, like inputs saved by any other op. The returned token never takes
    part in a loss, so ``backward`` never runs; the inputs are read back once, through
    ``token.grad_fn.saved_tensors``, when the region is rebuilt. As for any custom Function, the
    node (and so the stash) only exists if some input requires grad.
    """

    @staticmethod
    def forward(ctx, *tensors):
        """Save the inputs and return an empty token."""
        ctx.save_for_backward(*tensors)
        return torch.empty(0)

    @staticmethod
    def backward(ctx, *grads):
        """Never called: the token is not used by any loss."""
        raise RuntimeError("The input stash of CheckpointWithoutOutput must not reach a loss.")


class _RegionFunction(torch.autograd.Function):
    """Autograd node that stands in for the checkpointed function.

    It keeps its region in ``ctx.region`` so that backward can route the output gradients through
    the recomputed graph.
    """

    @staticmethod
    def forward(ctx, region, *tensors):
        """Run the function once without recording a graph."""
        ctx.region = region
        return region.run_without_graph(tensors)

    @staticmethod
    def backward(ctx, *grad_outputs):
        """Back-propagate through the rebuilt graph and return the input gradients."""
        region, ctx.region = ctx.region, None
        if region is None:
            raise RuntimeError(
                "CheckpointWithoutOutput does not support running backward through the same "
                "graph twice."
            )
        return (None,) + region.backpropagate(grad_outputs)


class _RecomputeRegion:
    """One call of a checkpointed function, plus what is needed to re-run it in backward.

    The region keeps ``detach()``-ed aliases of the outputs rather than the outputs themselves:
    an output reaches ``_RegionFunction`` through its grad_fn, and that node owns the region, so
    holding the output would close a reference cycle through the autograd graph.
    """

    def __init__(
        self, function: Callable, args: tuple, use_te_context: bool, retain_input_tensors: bool
    ):
        self.state: Optional[_RegionState] = None
        self._function = function
        self._layout = _ArgLayout(args)
        # Taken before the function draws any random number.
        self._rng_states = _tp_random()._get_all_rng_states()
        self._te_context = _TERecomputeContext(use_te_context)
        self._retained_storage_ptrs = (
            frozenset(t.untyped_storage().data_ptr() for t in self._layout.tensors_of(args))
            if retain_input_tensors
            else frozenset()
        )
        self._stash: Optional[torch.Tensor] = None
        self._output_aliases: tuple = ()
        # (rebuilt outputs, rebuilt input leaves), kept from the rebuild until backward.
        self._rebuilt: Optional[tuple] = None

    def record(self, args: tuple):
        """Run the function under no_grad and stash its tensor inputs for the rebuild."""
        tensors = self._layout.tensors_of(args)
        result = _RegionFunction.apply(self, *tensors)
        self._stash = _InputStash.apply(*tensors)
        self.state = _RegionState.RECORDED
        return result

    def run_without_graph(self, tensors: tuple):
        """Body of the forward pass, called from `_RegionFunction.forward`."""
        with torch.no_grad(), self._te_context.first_pass():
            return self._function(*self._layout.rebuild(tensors))

    def release(self, outputs: tuple) -> None:
        """Free the output storages; tensors and views over them stay valid handles."""
        if self.state is not _RegionState.RECORDED:
            return
        self._output_aliases = tuple(
            output.detach() if isinstance(output, torch.Tensor) else None for output in outputs
        )
        for alias in self._output_aliases:
            if alias is None:
                continue
            storage = alias.untyped_storage()
            # Outputs sharing storage with a retained input (e.g. TE.ops.Sequential operations
            # with MakeExtraOutput) keep their data: the rebuild reads that input.
            if storage.data_ptr() not in self._retained_storage_ptrs:
                storage.resize_(0)
        self.state = _RegionState.RELEASED

    def rebuild(self) -> None:
        """Re-run the function with grad enabled and give the released outputs the new data."""
        from megatron.core.transformer.cuda_graphs import is_graph_capturing, is_graph_warmup

        if self.state not in (_RegionState.RECORDED, _RegionState.RELEASED) or is_graph_warmup():
            return
        if not torch.autograd._is_checkpoint_valid() and not is_graph_capturing():
            raise RuntimeError(
                "Checkpointing is not compatible with .grad(), "
                "please use .backward() if possible"
            )

        leaves = self._unstash_inputs()
        with _tp_random()._fork_rng():
            _tp_random()._set_all_rng_states(*self._rng_states)
            with torch.enable_grad(), self._te_context.second_pass(), suspend_tensor_observations():
                outputs = _as_tuple(self._function(*self._layout.rebuild(leaves)))

        self._share_rebuilt_storages(outputs)
        self._rebuilt = (outputs, leaves)
        self._function = self._layout = self._rng_states = self._te_context = None
        self._output_aliases = ()
        self.state = _RegionState.REBUILT

    def _unstash_inputs(self) -> tuple:
        """Read the stashed inputs back (unpack hooks run here, once) as new graph leaves."""
        # checkpoint() only records a region when some input requires grad, so the stash node
        # exists and holds every tensor input.
        stash, self._stash = self._stash, None
        leaves = []
        for saved in stash.grad_fn.saved_tensors:
            leaf = saved.detach()
            leaf.requires_grad_(saved.requires_grad)
            leaves.append(leaf)
        return tuple(leaves)

    def _share_rebuilt_storages(self, outputs: tuple) -> None:
        """Point every released output storage at the rebuilt data."""
        if not self._output_aliases:
            return
        if len(outputs) != len(self._output_aliases):
            raise RuntimeError(
                f"CheckpointWithoutOutput: the recomputation returned {len(outputs)} outputs, "
                f"the forward pass returned {len(self._output_aliases)}."
            )
        # Zero-copy: make the output's StorageImpl point to the rebuilt data. This operates at
        # the UntypedStorage level (below TensorImpl), so:
        #   - ALL views / reshapes that reference the output's StorageImpl see the data
        #     (e.g. TE GroupedLinear's inp.reshape() + torch.split() saved for backward)
        #   - No tensor version-counter bump (no autograd complaint)
        share_storage = _get_share_storage()
        for alias, output in zip(self._output_aliases, outputs):
            if alias is None or not isinstance(output, torch.Tensor):
                continue
            # Outputs viewing one buffer share a storage that is re-pointed only once.
            if alias.untyped_storage().data_ptr() != output.untyped_storage().data_ptr():
                share_storage(alias, output)

    def backpropagate(self, grad_outputs: tuple) -> tuple:
        """Route the output gradients through the rebuilt graph; return the input gradients."""
        if self.state is _RegionState.RECORDED:
            # The outputs were never released, so no hook asked for a rebuild; do it now.
            self.rebuild()
        if self.state is not _RegionState.REBUILT:
            raise RuntimeError(
                "CheckpointWithoutOutput: backward reached the checkpointed function before its "
                f"outputs were recomputed (state {self.state.name}). The hook tensor must "
                "receive its gradient before the discarded outputs are used in backward."
            )
        (outputs, leaves), self._rebuilt = self._rebuilt, None
        self.state = _RegionState.DONE

        roots, grads = [], []
        for output, grad in zip(outputs, grad_outputs):
            if isinstance(output, torch.Tensor) and output.requires_grad and grad is not None:
                roots.append(output)
                grads.append(grad)
        if roots:
            # Use the accumulating entry point rather than torch.autograd.grad: parameters used
            # inside the function only get their gradients through their AccumulateGrad nodes.
            torch.autograd.backward(roots, grads)
        return tuple(leaf.grad if leaf.requires_grad else None for leaf in leaves)


def _register_rebuild_hook(hook_tensor: torch.Tensor, rebuild: Callable[[], None]) -> None:
    """Call `rebuild()` once the gradient of `hook_tensor` is computed, before it is used."""
    if not hook_tensor.requires_grad:
        return

    def _rebuild_on_grad(_grad):
        rebuild()

    hook_tensor.register_hook(_rebuild_on_grad)


class CheckpointWithoutOutputManager:
    """
    Coordinates activation recomputation across multiple CheckpointWithoutOutput instances
    within a TransformerBlock, enabling unified recomputation during backward pass.
    This is particularly useful for scenarios where multiple checkpoint operations have
    sequential dependencies (i.e., the output of one checkpoint is the input of the next).

    Usage:
        manager = CheckpointWithoutOutputManager()
        ckpt_function = CheckpointWithoutOutput(ckpt_manager=manager)
        ckpt_function.checkpoint(run_function, *args)
        # other checkpointed operations
        manager.discard_all_outputs_and_register_unified_recompute(final_output)
    """

    def __init__(self):
        self.checkpoints = []
        # Legacy block-end flag. HybridStack sets it for the Hybrid mHC wrapper, and
        # HyperConnectionTransformerLayer reads it only via the deprecated
        # mhc_recompute_manager keyword; TransformerBlock passes an immutable
        # ResidualStreamRecomputeContext instead. When True, the layer should keep
        # block-boundary output uncheckpointed.
        self.is_last_layer_in_recompute_block = False

    def add_checkpoint(self, ckpt):
        """Add a checkpoint to the manager."""
        if not isinstance(ckpt, CheckpointWithoutOutput):
            raise TypeError("Expected CheckpointWithoutOutput object")
        if ckpt.outputs is None:
            raise ValueError("CheckpointWithoutOutput must call checkpoint() before adding")
        self.checkpoints.append(ckpt)

    def discard_all_outputs_and_register_unified_recompute(self, hook_tensor):
        """Discard all checkpoint outputs to save memory and register unified recompute hook."""
        regions = [ckpt._release_outputs() for ckpt in self.checkpoints]
        regions = [region for region in regions if region is not None]

        def rebuild_in_forward_order():
            # A later region may read the outputs of an earlier one, which are rebuilt first.
            for region in regions:
                region.rebuild()

        _register_rebuild_hook(hook_tensor, rebuild_in_forward_order)


class CheckpointWithoutOutput:
    """
    Checkpoint a model or part of the model and release the output.

    For the normal 'checkpoint` function, the outputs of it may be saved by the following
    modules for their backward computation. However, the output of the checkpointed function is
    re-generated at recomputation, so the output store is not technically needed. This method can
    manually discard the output in the forward pass and restore it by recomputation in the
    backward pass to reduce the memory usage.

    Due to the reason above, to save memory with this method, the caller should make sure that the
    discarded output tensors are directly saved in the following modules for backward computation.
    """

    def __init__(self, fp8=False, ckpt_manager=None, retain_input_tensors=False):
        """
        Initialize CheckpointWithoutOutput.

        Args:
            fp8: Quantization recipe, or a bool. Note that the default `fp8=False`
                 still evaluates to `self.fp8 = True`; every caller that constructs
                 `CheckpointWithoutOutput()` with no arguments therefore takes the
                 TE `activation_recompute_forward` path. That is long-standing
                 behavior which several selective-recompute modules ("layernorm",
                 "moe_act", "gdn_norm_out") depend on for correct FP8 amax
                 bookkeeping, so do NOT "fix" this to `bool(fp8)` here — tightening
                 it changes FP8 numerics and needs its own PR with FP8
                 functional-test evidence.
            ckpt_manager: Optional CheckpointWithoutOutputManager instance. When provided,
                         checkpoint() will auto-register to the manager, and
                         discard_output_and_register_recompute() will only discard
                         output without registering individual hooks.
            retain_input_tensors: Whether outputs sharing storage with checkpoint inputs
                                  should be retained when discarding outputs.
        """
        self.fp8 = fp8 is not None
        self.ckpt_manager = ckpt_manager
        self.retain_input_tensors = retain_input_tensors
        # Outputs of the latest checkpoint() call, as a tuple, until they are released.
        self.outputs = None
        self._region: Optional[_RecomputeRegion] = None
        self._released_region: Optional[weakref.ref] = None

    def checkpoint(self, run_function: Callable[[Unpack[_Ts]], _R], *args: Unpack[_Ts]) -> _R:
        """
        Checkpoint function.

        If ckpt_manager was provided during initialization, this checkpoint
        will be automatically registered to the manager after execution.

        When grad is disabled, or no tensor argument requires grad, backward cannot need the
        outputs, so the function runs as a plain call and later discard calls do nothing.
        """

        # If in cuda graph warmup, disable checkpointing, as 'discard_output_and_register_recompute'
        # may be called in a separate graph warmup.
        from megatron.core.transformer.cuda_graphs import is_graph_warmup

        # A plain call also keeps parameters used inside the function on the normal autograd
        # path, and enters no Transformer Engine recompute context, whose first-pass state is
        # only consumed by a recomputation.
        records_graph = torch.is_grad_enabled() and any(
            isinstance(arg, torch.Tensor) and arg.requires_grad for arg in args
        )
        if is_graph_warmup() or not records_graph:
            self.outputs = None
            self._region = self._released_region = None
            return run_function(*args)

        region = _RecomputeRegion(run_function, args, self.fp8, self.retain_input_tensors)
        result = region.record(args)
        self._region, self._released_region = region, None
        self.outputs = _as_tuple(result)

        # Offload group commits can run before discard_output_and_register_recompute().
        # Mark the output storages now so activation offload does not copy or release
        # memory that this checkpoint owns and regenerates itself.
        for output in self.outputs:
            _mark_checkpoint_without_output_tensor(output)

        # Auto-register to manager if provided
        if self.ckpt_manager is not None:
            self.ckpt_manager.add_checkpoint(self)

        return result

    def _release_outputs(self) -> Optional[_RecomputeRegion]:
        """Release the outputs of the latest checkpoint() call and return its region.

        From then on the region is owned by its autograd node and by its recompute hooks; the
        facade only keeps a weak reference (for further hooks), so a region whose backward never
        runs does not keep its recomputed graph alive through the facade.
        """
        region = self._region
        if region is None:
            return self._released_region() if self._released_region is not None else None
        if self.outputs is not None:
            region.release(self.outputs)
        self.outputs = None
        self._region, self._released_region = None, weakref.ref(region)
        return region

    def discard_output_and_register_recompute(self, hook_tensor):
        """
        Release the output tensor storages and register the recompute function as a grad hook of
        the hook_tensor.

        Note: the caller should make sure that the output tensors are no longer used
        in the forward pass and the gradient of the hook_tensor is computed before the recomputed
        tensors are used. Calling this again with another hook_tensor adds another trigger; the
        recomputation runs once, at the first trigger.
        """
        # When ckpt_manager is set, this is a no-op.
        # Manager handles all discarding and hook registration uniformly.
        from megatron.core.transformer.cuda_graphs import is_graph_warmup

        if self.ckpt_manager is not None or is_graph_warmup():
            return

        region = self._release_outputs()
        if region is not None:
            _register_rebuild_hook(hook_tensor, region.rebuild)


def precision_aware_checkpoint(
    function: Callable[..., Any],
    config: 'TransformerConfig',
    tp_group: Optional[torch.distributed.ProcessGroup],
    /,
    *args: Any,
    distribute_saved_activations: bool = False,
    **kwargs: Any,
) -> Any:
    """Activation-checkpoint ``function`` with the implementation its precision requires.

    FP8 and FP4 recompute must go through Transformer Engine's checkpoint, because FP4 is
    also implemented with TE's ``fp8_autocast`` (see ``fp4_utils.get_fp4_context``). Other
    precisions use ``tensor_parallel.checkpoint``.

    Args:
        function: Callable to checkpoint.
        config: Transformer config whose ``fp8`` and ``fp4`` fields select the implementation.
        tp_group: Tensor-parallel group for Transformer Engine's checkpoint.
        *args: Positional arguments for ``function``.
        distribute_saved_activations: Passed through to the checkpoint implementation.
        **kwargs: Keyword arguments for ``function``.

    Returns:
        The output of ``function``.
    """
    from megatron.core import tensor_parallel

    if config.fp8 or config.fp4:
        # Imported lazily to avoid a circular import.
        from megatron.core.extensions.transformer_engine import te_checkpoint

        return te_checkpoint(
            function,
            distribute_saved_activations,
            tensor_parallel.random.get_cuda_rng_tracker,
            tp_group,
            *args,
            **kwargs,
        )
    if kwargs:
        function = functools.partial(function, **kwargs)
    return tensor_parallel.checkpoint(function, distribute_saved_activations, *args)
