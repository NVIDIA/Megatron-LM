# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Owner-compute orthogonalized optimizer for MFSDP v2.

Implements the Muon-style orthogonalizing optimizer step on top of MFSDP v2's all-`Flat`
placements: the Newton-Schulz kernels come from `emerging_optimizers` through a composed
`OrthogonalizedOptimizer`, and the communication runs through the
`owner_planning`/`p2p` modules.

Every >=2D parameter is contracted to a 2D matrix `(shape[0], rest)` for orthogonalization.
Like the upstream `OrthogonalizedOptimizer`, `<2D` parameters are not contracted and error
when `orthogonalize` is called; `FsdpMuon` rejects them at construction instead, expecting
them to be routed to a separate optimizer.
"""

import contextlib
import inspect
import warnings
from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any, cast, overload, override

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.optim.optimizer import ParamsT

try:
    from emerging_optimizers import utils as eo_utils
    from emerging_optimizers.orthogonalized_optimizers import OrthogonalizedOptimizer
    from emerging_optimizers.orthogonalized_optimizers.muon import Muon

    HAVE_EMERGING_OPTIMIZERS = True
except (ModuleNotFoundError, ImportError):
    eo_utils = cast(Any, None)
    OrthogonalizedOptimizer = cast(Any, object)
    Muon = cast(Any, object)
    HAVE_EMERGING_OPTIMIZERS = False

from .owner_planning import GroupOwnerLayout, ns_cost_fn
from .p2p import gather, scatter
from .parameter_group import FsdpParameterGroup, get_containing_parameter_group
from .placement import Flat


def _fsdp_group_of(param: torch.nn.Parameter) -> FsdpParameterGroup:
    """Return the owning `FsdpParameterGroup`, raising for unsharded parameters."""
    fsdp_group = get_containing_parameter_group(param)
    if fsdp_group is None:
        raise RuntimeError(
            "FsdpOrthogonalizedOptimizer parameters must be FSDP-sharded; "
            f"parameter {param!r} is not owned by an FsdpParameterGroup."
        )
    return fsdp_group


def _require_emerging_optimizers() -> None:
    """Raise if `emerging_optimizers` is not installed."""
    if not HAVE_EMERGING_OPTIMIZERS:
        raise ModuleNotFoundError(
            "Emerging-Optimizers is required for orthogonalized optimizer support. "
            "Please install the necessary dependencies with "
            "`pip install 'megatron_fsdp[emerging-optimizers]'`."
        )


class FsdpOrthogonalizedOptimizer(torch.optim.Optimizer):
    """Owner-compute orthogonalized optimizer for all-`Flat` MFSDP v2 parameters.

    Subclasses `torch.optim.Optimizer` directly so it is a drop-in for the training loop
    and checkpointer. It composes an `OrthogonalizedOptimizer` (held as `self._inner`)
    only for the Newton-Schulz kernel, weight decay, and the pre/post weight-update
    hooks; the inner's `defaults`/`param_groups`/`state` are this optimizer's (the same
    objects), and the `step` override replaces the inner's per-parameter all-gather path
    with the pipelined owner-compute algorithm.

    Each step, per `torch` param group and per contained `FsdpParameterGroup`: the
    active parameters' gather is posted, then their inputs are orthogonalized,
    scattered, and applied on the group's own CUDA stream — a group's Newton-Schulz
    work thereby overlaps the next group's communications. At the end, the caller's
    stream is ordered behind every group stream and the touched groups' model weights
    are synced from their main weights.

    Args:
        params: Parameters or param-group dicts to optimize.
        inner_optimizer: Pre-built `OrthogonalizedOptimizer` over the same parameters;
            its arguments (`lr`, `momentum`, `weight_decay`, `nesterov`, ...) are the
            optimization hyperparameters.
        dp_mesh: Device mesh of the FSDP data-parallel group. Every parameter group's
            mesh must be this mesh.
        num_ns_steps: Newton-Schulz iteration count, used by the owner load-balancing
            cost heuristic. If `None`, defaults to 1.
    """

    def __init__(
        self,
        params: ParamsT,
        inner_optimizer: OrthogonalizedOptimizer,
        dp_mesh: DeviceMesh,
        num_ns_steps: int | None = None,
    ) -> None:
        _require_emerging_optimizers()

        if num_ns_steps is None:
            num_ns_steps = 1
        if num_ns_steps < 1:
            raise ValueError(f"num_ns_steps must be at least 1, got {num_ns_steps}.")

        self.dp_mesh = dp_mesh
        self._num_ns_steps: int = num_ns_steps
        self._owner_layouts: dict[FsdpParameterGroup, GroupOwnerLayout] = {}
        self._streams: dict[FsdpParameterGroup, torch.cuda.Stream] = {}

        # Disable properties while initializing this instance. We'd either have a missing attribute
        # or would reset the inner optimizer's attributes.
        with self._without_property_methods():
            super().__init__(params, {})
        self._inner = inner_optimizer

        # Validate: FSDP-sharded, dimensionalities, one mesh, all-`Flat` placements.
        params_list = self._all_params()
        self._validate_param_ndims(params_list)
        seen: set[FsdpParameterGroup] = set()
        for param in params_list:
            fsdp_group = _fsdp_group_of(param)
            if fsdp_group in seen:
                continue
            seen.add(fsdp_group)
            if fsdp_group.mesh is not self.dp_mesh:
                raise ValueError(
                    "FsdpOrthogonalizedOptimizer requires every parameter group's mesh "
                    f"to be the optimizer's dp_mesh; {fsdp_group!r} has a different mesh."
                )
            for buf in (fsdp_group.main_weight, fsdp_group.model_weight, fsdp_group.main_grad):
                if buf is not None and not all(isinstance(p, Flat) for p in buf.placements):
                    raise ValueError(
                        "FsdpOrthogonalizedOptimizer requires all-Flat placements "
                        "(parameter=Flat, gradient=Flat, optimizer=Flat), but "
                        f"{fsdp_group!r} has non-Flat placements "
                        f"{[type(p).__name__ for p in buf.placements]}."
                    )

    def _validate_param_ndims(self, params: Sequence[torch.Tensor]) -> None:
        """Validate parameter dimensionalities; subclasses may restrict them.

        The base class performs no dimensionality filtering: every >=2D parameter is
        contracted to a 2D matrix `(shape[0], rest)` for orthogonalization, and `<2D`
        parameters — like in the upstream `OrthogonalizedOptimizer` — error when
        `orthogonalize` is called. Subclasses may override this to reject parameters
        earlier; `FsdpMuon` rejects `<2D` parameters at construction per Muon's
        convention.
        """
        pass

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        """Delegate `param_groups` to the inner optimizer."""
        return self._inner.param_groups

    @param_groups.setter
    def param_groups(self, value: list[dict[str, Any]]) -> None:
        """Set `param_groups` on the inner optimizer."""
        self._inner.param_groups = value

    @property
    def defaults(self) -> dict[str, Any]:
        """Delegate `defaults` to the inner optimizer."""
        return self._inner.defaults

    @defaults.setter
    def defaults(self, value: dict[str, Any]) -> None:
        """Set `defaults` on the inner optimizer."""
        self._inner.defaults = value

    @property
    def state(self) -> defaultdict[torch.Tensor, Any]:
        """Delegate `state` to the inner optimizer."""
        return self._inner.state

    @state.setter
    def state(self, value: defaultdict[torch.Tensor, Any]) -> None:
        """Set `state` on the inner optimizer."""
        self._inner.state = value

    def _all_params(self) -> list[torch.nn.Parameter]:
        """Flatten this optimizer's params from its (now-materialized) groups."""
        return [p for group in self.param_groups for p in group["params"]]

    def _init_group(self, group: dict, skip_non_grad_params: bool = True) -> None:
        """Performs lazy momentum-state initialization, delegated to the inner optimizer."""
        self._inner._init_group(group, skip_non_grad_params=skip_non_grad_params)

    @contextlib.contextmanager
    def _without_property_methods(self):
        """Temporarily remove the delegating property descriptors.

        The properties are defined on `FsdpOrthogonalizedOptimizer` and inherited by
        subclasses, so `delattr` must target the defining class (found via the MRO),
        not `type(self)` (which is the subclass and does not own the descriptors).
        """
        names = ["param_groups", "defaults", "state"]
        saved: dict[str, tuple[type, property]] = {}
        for name in names:
            for cls_ in type(self).__mro__:
                descriptor = cls_.__dict__.get(name)
                if descriptor is not None and isinstance(descriptor, property):
                    saved[name] = (cls_, descriptor)
                    try:
                        delattr(cls_, name)
                    except AttributeError:
                        pass
                    break
        try:
            yield
        finally:
            for name in names:
                self.__dict__.pop(name, None)
            for name, (cls_, descriptor) in saved.items():
                setattr(cls_, name, descriptor)

    def _this_rank(self) -> int:
        """This rank's index in the DP mesh."""
        return self.dp_mesh.get_local_rank()

    def _device(self) -> torch.device:
        """The DP mesh's device."""
        return torch.device(self.dp_mesh.device_type)

    def _sharded_dtensor(self, fsdp_group: FsdpParameterGroup, tensor_index: int) -> DTensor:
        """Return a group's sharded parameter as a `DTensor`, asserting its type.

        The sharded parameters are constructed as `nn.Parameter`s over `DBuffer` DTensors,
        which preserves the `DTensor` class at runtime; the static types only see
        `nn.Parameter`. Each use asserts the `DTensor` view, so a misconstructed group
        fails loudly instead of dispatching wrongly.
        """
        param = fsdp_group.fsdp_parameters[tensor_index].sharded
        assert isinstance(
            param, DTensor
        ), f"Sharded parameter {tensor_index} of {fsdp_group!r} is not a `DTensor`."
        return param

    def _stream_for(self, fsdp_group: FsdpParameterGroup) -> torch.cuda.Stream | None:
        """Return the pipelining stream for one group, pooled and reused across steps.

        Returns `None` on CPU, where everything runs on the current stream.
        """
        if not torch.cuda.is_available():
            return None
        if fsdp_group not in self._streams:
            self._streams[fsdp_group] = torch.cuda.Stream(device=self._device())
        return self._streams[fsdp_group]

    def _compute_orthogonalization_inputs(
        self, param: DTensor, grad: DTensor, group: dict[str, Any], lr: float
    ) -> torch.Tensor:
        """For the given parameter, apply weight decay and update momentum state, then
        produce and return the inputs for orthogonalization.
        """
        p_local = param.to_local()
        state = self.state[param]
        momentum = state["momentum_buffer"]
        mom_local = momentum.to_local()
        local_grad = grad.to_local()
        if local_grad.dtype != mom_local.dtype:
            local_grad = local_grad.to(dtype=mom_local.dtype)
        if local_grad.shape != mom_local.shape:
            local_grad = local_grad.reshape(mom_local.shape)

        self._inner._apply_weight_decay_inplace(p_local, local_grad, lr, group["weight_decay"])
        mom_local.lerp_(local_grad, 1 - group["momentum"])
        if self._inner.nesterov:
            pre_ns = local_grad.lerp(mom_local, group["momentum"])
        else:
            pre_ns = mom_local
        return pre_ns

    def _orthogonalize_active(
        self,
        active_layout: GroupOwnerLayout,
        destination: dict[int, torch.Tensor],
        group_kwargs: dict[str, Any],
    ) -> dict[int, torch.Tensor]:
        """Orthogonalize the group's gathered inputs into full updates.

        Each >=2D input is contracted to a 2D matrix `(shape[0], rest)` per Muon's matrix
        view of weights; `<2D` inputs are passed through uncontracted and error in
        `orthogonalize`, like in the upstream `OrthogonalizedOptimizer`. This method is
        the seam for future batching: parameters of the same shape could be stacked
        into one batched kernel instead of the per-parameter loop.
        """
        this_rank = self._this_rank()
        updates: dict[int, torch.Tensor] = {}
        for i, layout in active_layout.layouts.items():
            if active_layout.owners[i] != this_rank:
                continue
            shape = layout.full_shape
            matrix = destination[i].view(shape[0], -1) if len(shape) >= 2 else destination[i]
            with eo_utils.fp32_matmul_precision(self._inner.fp32_matmul_prec):
                # The kernel ignores the parameter argument, so `None` is fine.
                updates[i] = self._inner.orthogonalize(
                    None,  # ty: ignore[invalid-argument-type]
                    matrix.to(torch.float32),
                    **group_kwargs,
                )
        return updates

    def _apply_update(self, param: DTensor, update_shard: torch.Tensor, lr: float) -> None:
        """Update the given parameter with the result of orthogonalization."""
        p_local = param.to_local()
        if update_shard.shape != p_local.shape:
            update_shard = update_shard.view(p_local.shape)
        if update_shard.dtype != p_local.dtype:
            update_shard = update_shard.to(dtype=p_local.dtype)
        self._inner.pre_weight_update_fn_inplace(p_local, update_shard)
        p_local.add_(update_shard, alpha=-lr)
        self._inner.post_weight_update_fn_inplace(p_local)

    @overload
    def step(self, closure: None = None) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    @torch.no_grad()
    @override
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """Perform a single pipelined owner-compute optimization step."""
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        else:
            loss = None

        touched: set[FsdpParameterGroup] = set()
        temporaries: list[Any] = []
        for torch_group in self.param_groups:
            self._init_group(torch_group)
            lr = torch_group["lr"]
            params = {p for p in torch_group["params"] if p.grad is not None}
            if not params:
                continue
            fsdp_groups: dict[FsdpParameterGroup, None] = {}
            for param in params:
                fsdp_group = _fsdp_group_of(param)
                fsdp_groups[fsdp_group] = None
                touched.add(fsdp_group)
            for fsdp_group in fsdp_groups:
                owner_layout = self._owner_layouts.get(fsdp_group)
                if owner_layout is None:
                    # Built once per group — layouts and owners are step-independent.
                    # Everything participates; dimensionalities are a validation concern.
                    owner_layout = self._owner_layouts[fsdp_group] = GroupOwnerLayout.from_group(
                        fsdp_group,
                        cost_fn=ns_cost_fn(self._num_ns_steps),
                        eligible_fn=lambda _param: True,
                    )
                active_indices = [
                    i for i, fp in enumerate(fsdp_group.fsdp_parameters) if fp.sharded in params
                ]
                active = GroupOwnerLayout(
                    group=fsdp_group,
                    layouts={i: owner_layout.layouts[i] for i in active_indices},
                    owners={i: owner_layout.owners[i] for i in active_indices},
                )
                if active.layouts:
                    temporaries.extend(self._step_active(fsdp_group, active, torch_group, lr))

        # Order the caller's stream behind every group stream. All step temporaries stay
        # referenced until after these waits, so their blocks cannot be reused before
        # the pending group work completes (the caching allocator assumes the
        # allocation stream; the waits make any later reuse stream-ordered after it).
        for stream in self._streams.values():
            torch.cuda.current_stream().wait_stream(stream)
        temporaries.clear()
        for fsdp_group in touched:
            fsdp_group.sync_model_weight_from_main_weight()
        return loss

    def _step_active(
        self,
        fsdp_group: FsdpParameterGroup,
        active: GroupOwnerLayout,
        torch_group: dict[str, Any],
        lr: float,
    ) -> list[Any]:
        """Three-stage pipelined owner-compute step for one group's active parameters.

        Stage 1 computes the local pre-orthogonalization chunks and posts the group's
        gather. Stages 2 and 3 run on the group's stream: orthogonalize the gathered
        inputs, scatter the updates, and apply them — so this group's Newton-Schulz
        work overlaps the next group's communications (posting is host-asynchronous
        and each group has its own stream).

        Returns the step temporaries, which the caller must keep referenced until it
        has ordered its stream behind every group stream.
        """
        this_rank = self._this_rank()
        group_kwargs = {k: v for k, v in torch_group.items() if k != "params"}
        temporaries: list[Any] = []

        # Stage 1: local pre-orthogonalization chunks, then post the group's gather.
        source: dict[int, torch.Tensor] = {}
        for i in active.layouts:
            if active.layouts[i].rank_numel(this_rank) == 0:
                continue
            param = self._sharded_dtensor(fsdp_group, i)
            grad = param.grad
            assert isinstance(
                grad, DTensor
            ), f"Gradient of sharded parameter {i} is not a `DTensor`."
            source[i] = self._compute_orthogonalization_inputs(param, grad, torch_group, lr)
        owned = [i for i in active.layouts if active.owners[i] == this_rank]
        destination: dict[int, torch.Tensor] = {}
        if owned:
            reference = next(iter(source.values()))
            destination = {
                i: torch.empty(
                    active.layouts[i].full_numel(), dtype=reference.dtype, device=reference.device
                )
                for i in owned
            }
        temporaries.extend((source, destination))
        gather(source, destination, owner_layout=active, stream=self._stream_for(fsdp_group))

        # Stages 2 and 3, on the group's stream.
        stream = self._stream_for(fsdp_group)
        with torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext():
            updates = self._orthogonalize_active(active, destination, group_kwargs)
            # The update dtype matches `_orthogonalize_active`'s fp32 output.
            scratch = {
                i: torch.empty(
                    active.layouts[i].rank_numel(this_rank),
                    dtype=torch.float32,
                    device=self._device(),
                )
                for i in active.layouts
                if active.owners[i] != this_rank and active.layouts[i].rank_numel(this_rank) > 0
            }
            scatter(updates, scratch, owner_layout=active, stream=stream)
            for i in active.layouts:
                layout = active.layouts[i]
                numel = layout.rank_numel(this_rank)
                if numel == 0:
                    continue
                if active.owners[i] == this_rank:
                    offset = layout.rank_offset(this_rank)
                    update_shard = updates[i].reshape(-1)[offset : offset + numel]
                else:
                    update_shard = scratch[i]
                self._apply_update(self._sharded_dtensor(fsdp_group, i), update_shard, lr)
            temporaries.extend((updates, scratch))
        return temporaries


class FsdpMuon(FsdpOrthogonalizedOptimizer):
    """Muon optimizer for all-`Flat` MFSDP v2 parameters.

    Composes a `Muon` inner optimizer (an `OrthogonalizedOptimizer`) for the Newton-Schulz
    orthogonalization and update scaling. The inner `Muon` installs its own
    `scaled_orthogonalize_fn`, so the base `scaled_orthogonalize_fn` is ignored.

    Rejects `<2D` parameters at construction per Muon's convention — route them (biases,
    norms) to a separate optimizer.
    """

    def __init__(
        self,
        params: ParamsT,
        inner_optimizer: Muon,  # ty: ignore[invalid-type-form]
        dp_mesh: DeviceMesh,
    ) -> None:
        """Build the FSDP Muon optimizer; see `FsdpOrthogonalizedOptimizer` for the rest."""
        _require_emerging_optimizers()

        if hasattr(inner_optimizer, "num_ns_steps"):
            self._num_ns_steps = inner_optimizer.num_ns_steps
        else:
            # For older `emerging_optimizers` versions, we use introspection
            # techniques to get a sensible value for `num_ns_steps`.
            try:
                ortho_fn_vars = inspect.getclosurevars(inner_optimizer.scaled_orthogonalize_fn)
                self._num_ns_steps = ortho_fn_vars.nonlocals["num_ns_steps"]
            except KeyError:
                warnings.warn(
                    "Cannot access Muon closure non-locals; going with "
                    "`emerging_optimizers.orthogonalized_optimizer.Muon` "
                    "default `num_ns_steps` for compute cost estimation"
                )
                muon_sig = inspect.signature(inner_optimizer)
                self._num_ns_steps = muon_sig.parameters["num_ns_steps"].default
        super().__init__(params, inner_optimizer, dp_mesh=dp_mesh, num_ns_steps=self._num_ns_steps)

    @override
    def _validate_param_ndims(self, params: Sequence[torch.Tensor]) -> None:
        """Reject `<2D` parameters: Muon orthogonalizes matrices only."""
        offending_shapes = [tuple(p.shape) for p in params if p.ndim < 2]
        if offending_shapes:
            raise ValueError(
                "FsdpMuon only supports >=2D parameters (they are contracted to 2D "
                "matrices for orthogonalization). Route <2D parameters — biases, norms "
                f"— to a separate optimizer. Offending shapes: {offending_shapes}."
            )
