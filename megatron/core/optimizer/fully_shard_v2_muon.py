# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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


from collections import OrderedDict
from dataclasses import dataclass
from typing import List

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Shard

from megatron.core.distributed.fsdp.src.megatron_fsdp.uneven_dtensor import copy_chunk_metadata

from .emerging_optimizers import HAVE_EMERGING_OPTIMIZERS
from .optimizer import ChainedOptimizer, MegatronOptimizer

if HAVE_EMERGING_OPTIMIZERS:
    from .emerging_optimizers import get_muon_scale_factor, newton_schulz_tp


@torch.no_grad()
def _vanilla_newton_schulz(grad: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Self-contained quintic Newton-Schulz orthogonalization (fp32).

    Fallback used when ``emerging_optimizers`` is not installed; matches the
    canonical Muon ``zeropower`` iteration. Operates on the shorter dimension by
    transposing when rows > cols.
    """
    assert grad.ndim == 2, f"newton_schulz expects a 2D matrix, got {tuple(grad.shape)}"
    a, b, c = 3.4445, -4.7750, 2.0315
    x = grad.to(torch.float32)
    x = x / (x.norm() + eps)
    transposed = x.size(0) > x.size(1)
    if transposed:
        x = x.t()
    for _ in range(steps):
        gram = x @ x.t()
        x = a * x + (b * gram + c * (gram @ gram)) @ x
    if transposed:
        x = x.t()
    return x


def _vanilla_muon_scale(rows: int, cols: int) -> float:
    """Muon spectral scale factor ``max(1, rows / cols) ** 0.5`` (fallback)."""
    return max(1.0, rows / cols) ** 0.5


def _validate_muon_hyperparams(
    num_ns_steps: int, coefficient_type: str, scale_mode: str, fp32_matmul_prec: str
) -> None:
    """Validate algorithm choices before mutating live optimizer state."""
    if num_ns_steps < 1:
        raise ValueError(f"num_ns_steps must be at least 1, got {num_ns_steps}")
    if not HAVE_EMERGING_OPTIMIZERS and coefficient_type != "quintic":
        raise ValueError(
            "FullyShardV2Muon's built-in fallback supports only "
            f"coefficient_type='quintic', got {coefficient_type!r}."
        )
    if not HAVE_EMERGING_OPTIMIZERS and scale_mode != "spectral":
        raise ValueError(
            "FullyShardV2Muon's built-in fallback supports only "
            f"scale_mode='spectral', got {scale_mode!r}."
        )
    if fp32_matmul_prec not in ("medium", "high"):
        raise ValueError(
            "FullyShardV2Muon supports fp32_matmul_prec='medium' or 'high', "
            f"got {fp32_matmul_prec!r}."
        )


@dataclass
class MuonGradPackage:
    """Static Muon package over aligned main-weight / grad / momentum indices."""

    param_ids: List[int]
    has_comm: bool

    @torch.no_grad()
    def update_momentum(self, opt, coef):
        """Update local momentum shards and apply Nesterov correction when enabled."""
        for param_idx in self.param_ids:
            grad = opt._main_grads[param_idx]
            if grad is None:
                continue
            grad_shard = grad.to_local().reshape(-1)
            if grad_shard.numel() == 0:
                continue
            momentum = opt._momentum_buffers[param_idx]
            momentum.mul_(coef).add_(grad_shard)
            if opt._nesterov:
                torch.add(grad_shard, momentum, alpha=coef, out=grad_shard)

    @torch.no_grad()
    def issue_gather(self, opt, full_grads):
        """Gather this package's NS input shards to each param root."""
        if not self.has_comm:
            return []

        ops = []
        this_rank = dist.get_rank()
        for param_idx in self.param_ids:
            root = opt._roots[param_idx]
            ranges = opt._shard_ranges[param_idx]
            this_offset, this_size = 0, 0
            for rank, offset, size in ranges:
                if rank == this_rank:
                    this_offset, this_size = offset, size
                    break
            ns_input_shard = None
            if this_size > 0:
                if opt._nesterov:
                    ns_input_shard = opt._main_grads[param_idx].to_local().reshape(-1)
                else:
                    ns_input_shard = opt._momentum_buffers[param_idx]

            if root == this_rank:
                main_weight = opt._main_weights[param_idx]
                momentum = opt._momentum_buffers[param_idx]
                full_grad = torch.empty(
                    main_weight.shape.numel(), dtype=momentum.dtype, device=momentum.device
                )
                full_grads[param_idx] = full_grad
                if this_size > 0:
                    full_grad[this_offset : this_offset + this_size].copy_(ns_input_shard)
                for src, offset, size in ranges:
                    if size == 0 or src == this_rank:
                        continue
                    ops.append(dist.P2POp(dist.irecv, full_grad[offset : offset + size], src))
            elif this_size > 0:
                ops.append(dist.P2POp(dist.isend, ns_input_shard, root))
        return dist.batch_isend_irecv(ops) if ops else []

    @torch.no_grad()
    def finish_gather(self, requests):
        """Wait for all asynchronous gradient-gather requests."""
        for request in requests:
            request.wait()

    @torch.no_grad()
    def orthogonalize(self, opt, requests, full_grads):
        """Orthogonalize the gathered gradients assigned to this rank."""
        self.finish_gather(requests)
        orths = {}
        this_rank = dist.get_rank()
        for param_idx in self.param_ids:
            if opt._roots[param_idx] != this_rank:
                continue
            if param_idx in full_grads:
                full_grad = full_grads.pop(param_idx)
            elif opt._nesterov:
                full_grad = opt._main_grads[param_idx].to_local().reshape(-1)
            else:
                full_grad = opt._momentum_buffers[param_idx]
            full_grad = full_grad.view(opt._main_weights[param_idx].shape)

            grad = full_grad.to(torch.float32)
            if HAVE_EMERGING_OPTIMIZERS:
                orth = newton_schulz_tp(
                    grad,
                    steps=opt._num_ns_steps,
                    coefficient_type=opt._coefficient_type,
                    tp_group=None,
                    partition_dim=None,
                    tp_mode="duplicated",
                )
                scale = get_muon_scale_factor(grad.size(-2), grad.size(-1), mode=opt._scale_mode)
            else:
                orth = _vanilla_newton_schulz(grad, steps=opt._num_ns_steps)
                scale = _vanilla_muon_scale(grad.size(-2), grad.size(-1))
            orths[param_idx] = (orth * (scale * opt._extra_scale_factor)).to(full_grad.dtype)
        return orths

    @torch.no_grad()
    def issue_scatter(self, opt, orths):
        """Scatter this package's orthogonalized shards from each param root."""
        if not self.has_comm:
            return []

        ops = []
        this_rank = dist.get_rank()
        for param_idx in self.param_ids:
            root = opt._roots[param_idx]
            ranges = opt._shard_ranges[param_idx]
            if root == this_rank:
                orth = orths[param_idx].reshape(-1)
                for dst, offset, size in ranges:
                    if size == 0 or dst == this_rank:
                        continue
                    ops.append(dist.P2POp(dist.isend, orth[offset : offset + size], dst))
            else:
                this_size = 0
                for rank, _offset, size in ranges:
                    if rank == this_rank:
                        this_size = size
                        break
                if this_size > 0:
                    ops.append(
                        dist.P2POp(
                            dist.irecv, opt._main_grads[param_idx].to_local().reshape(-1), root
                        )
                    )
        return dist.batch_isend_irecv(ops) if ops else []

    @torch.no_grad()
    def finish_scatter(self, requests):
        """Wait for all asynchronous gradient-scatter requests."""
        for request in requests:
            request.wait()


class FullyShardV2Muon(torch.optim.Optimizer):
    """Single-root distributed Muon for Megatron-FSDP v2 (a torch optimizer).

    Initialization:
      1. Layout: compute each rank's flat shard range for every parameter.
      2. Root selection: assign each parameter's Newton-Schulz work to one rank.
         The greedy policy balances estimated NS cost while preferring ranks that
         already hold the shard.
      3. Package construction: group communicating params by DP group and sort by
         NS cost. No-comm params are split around communication packages to cover
         the first gather and final scatter.

    Step:
      1. Momentum: update local momentum shards; Nesterov writes the NS input into
         the local grad shard.
      2. Gather: send NS input shards to each parameter's root rank.
      3. Orthogonalize: root ranks run Newton-Schulz and keep full orth updates.
      4. Scatter: roots send orthogonalized shards back to shard owners.
      5. Apply: each rank updates its local optimizer-weight shard.

    Args:
        params: optimizer-facing params — all assumed to be FSDP-v2 ``dist_param``
            DTensors (asserted). A flat list or param groups, as any torch optimizer.
        grads: gradient slots aligned 1:1 with the flattened params. Entries are
            DTensors or ``None`` for empty/local-yet-unpublished shards; the outer
            wrapper binds live optimizer-facing DTensors immediately around each
            step.
        momentum_buffers: flat local momentum tensors owned by the outer Megatron
            optimizer adapter and updated in-place by this inner optimizer.
        lr / momentum / nesterov / weight_decay: Muon update hyperparameters.
        num_ns_steps / coefficient_type / scale_mode / extra_scale_factor: NS +
            scaling config used by ``orthogonalize``.
        fp32_matmul_prec: FP32 matrix-multiplication precision used during
            Newton-Schulz. Megatron-FSDP v2 supports ``"medium"`` and ``"high"``.
        tp_mode: Reserved for the future tensor-parallel NS path (currently TP
            size 1; accepted and ignored). Params are grouped into communication
            and no-communication packages after root assignment.
    """

    def __init__(
        self,
        params,
        grads,
        momentum_buffers,
        *,
        lr: float,
        momentum: float = 0.95,
        nesterov: bool = True,
        weight_decay: float = 0.01,
        num_ns_steps: int = 5,
        coefficient_type: str = "quintic",
        scale_mode: str = "spectral",
        extra_scale_factor: float = 1.0,
        fp32_matmul_prec: str = "medium",
        tp_mode: str = "duplicated",
    ) -> None:
        _validate_muon_hyperparams(num_ns_steps, coefficient_type, scale_mode, fp32_matmul_prec)
        params = list(params)
        super().__init__(
            params if params else [{"params": []}],
            dict(
                lr=lr,
                momentum=momentum,
                nesterov=nesterov,
                weight_decay=weight_decay,
                num_ns_steps=num_ns_steps,
                coefficient_type=coefficient_type,
                scale_mode=scale_mode,
                extra_scale_factor=extra_scale_factor,
                fp32_matmul_prec=fp32_matmul_prec,
            ),
        )

        param_list = [param for group in self.param_groups for param in group["params"]]
        grad_list = list(grads)
        assert all(
            isinstance(param, DTensor) for param in param_list
        ), "FullyShardV2Muon expects every param to be a DTensor (FSDP-v2 dist_param)."
        assert len(grad_list) == len(
            param_list
        ), f"FullyShardV2Muon got {len(grad_list)} grads for {len(param_list)} params."
        assert all(
            grad is None or isinstance(grad, DTensor) for grad in grad_list
        ), "FullyShardV2Muon expects every grad to be a DTensor or None."
        momentum_list = list(momentum_buffers)
        assert len(momentum_list) == len(param_list), (
            f"FullyShardV2Muon got {len(momentum_list)} momentum buffers for "
            f"{len(param_list)} params."
        )
        for param_idx, (param, momentum_buffer) in enumerate(zip(param_list, momentum_list)):
            assert momentum_buffer.numel() == param.to_local().numel(), (
                f"Muon momentum buffer for param {param_idx} has {momentum_buffer.numel()} "
                f"elements, expected {param.to_local().numel()}."
            )

        self._main_weights = param_list
        self._main_grads = grad_list
        self._momentum_buffers = momentum_list

        # Hyperparameters read by step() and MuonGradPackage methods.
        self._momentum_coef = momentum
        self._weight_decay = weight_decay
        self._nesterov = nesterov
        self._num_ns_steps = num_ns_steps
        self._coefficient_type = coefficient_type
        self._scale_mode = scale_mode
        self._extra_scale_factor = extra_scale_factor
        self._fp32_matmul_prec = fp32_matmul_prec

        self._shard_ranks = [
            self._get_shard_ranks(main_weight) for main_weight in self._main_weights
        ]
        self._shard_ranges = self._compute_shard_ranges()
        self._roots = self._assign_roots()

        self._ns_costs = [
            main_weight.shape.numel() * min(main_weight.shape) if len(main_weight.shape) == 2 else 0
            for main_weight in self._main_weights
        ]
        self._packages = self._build_packages()

    def _get_shard_ranks(self, main_weight):
        """Return global ranks that own shards of ``main_weight`` in offset order."""
        mesh = main_weight.device_mesh
        shard_dims = [
            mesh_dim
            for mesh_dim, placement in enumerate(main_weight.placements)
            if isinstance(placement, Shard)
        ]
        if not shard_dims:
            return (dist.get_rank(),)

        mesh_tensor = mesh.mesh.detach().cpu()
        current_rank = dist.get_rank()
        current_coord = (mesh_tensor == current_rank).nonzero(as_tuple=False)
        if current_coord.numel() == 0:
            raise RuntimeError(f"Rank {current_rank} is not present in Muon mesh {mesh}.")
        current_coord = current_coord[0].tolist()

        shard_order = [
            mesh_dim
            for mesh_dim in getattr(mesh, "_shard_order", shard_dims)
            if mesh_dim in shard_dims
        ]
        if sorted(shard_order) != sorted(shard_dims):
            shard_order = shard_dims

        ranks = []

        def visit(order_idx):
            if order_idx == len(shard_order):
                ranks.append(int(mesh_tensor[tuple(current_coord)].item()))
                return
            mesh_dim = shard_order[order_idx]
            original = current_coord[mesh_dim]
            for coord in range(mesh_tensor.shape[mesh_dim]):
                current_coord[mesh_dim] = coord
                visit(order_idx + 1)
            current_coord[mesh_dim] = original

        visit(0)
        return tuple(ranks)

    def _compute_shard_ranges(self):
        """Compute each communication peer's ``(flat_offset, size)`` per param."""
        local_ranges = []
        for main_weight in self._main_weights:
            row_numel = 1
            for dim in main_weight.shape[1:]:
                row_numel *= dim
            chunk_list = getattr(main_weight._local_tensor, "__create_chunk_list__", None)
            if chunk_list is None:
                offset = 0
                size = main_weight.to_local().shape[0] * row_numel
            else:
                chunks = chunk_list()
                assert len(chunks) == 1, f"Expected one Muon shard chunk, got {len(chunks)}."
                offset = chunks[0].offsets[0] * row_numel
                size = chunks[0].sizes[0] * row_numel
            local_ranges.append((offset, size))

        all_rank_ranges = [None] * dist.get_world_size()
        dist.all_gather_object(all_rank_ranges, local_ranges)

        ranges = []
        for param_idx, shard_ranks in enumerate(self._shard_ranks):
            rank_ranges = [(rank, *all_rank_ranges[rank][param_idx]) for rank in shard_ranks]
            ranges.append(rank_ranges)
        return ranges

    def _assign_roots(self):
        """Assign each param an NS root rank, load-balancing NS work per shard domain.

        Steps:
          1. group params by shard domain and estimate each param's NS cost;
          2. compute the average NS target load for the group;
          3. assign params heaviest-first, preferring no-comm roots until their
             ranks reach the target, then allowing remote roots for balance.

        Deterministic + identical on every rank (depends only on shard layout).

        Example (world_size=2):
            target load = 400
            param A: single holder {0}, cost 300  -> root 0          (load [300,   0])
            param B: single holder {0}, cost 300  -> root 0          (load [600,   0])
            param C: split {0,1},       cost 200  -> root 1          (load [600, 200])
        """
        params_by_group = OrderedDict()
        for param_idx in range(len(self._main_weights)):
            params_by_group.setdefault(self._shard_ranks[param_idx], []).append(param_idx)
        roots = [dist.get_rank()] * len(self._main_weights)
        for shard_ranks, param_indices in params_by_group.items():
            load = {rank: 0 for rank in shard_ranks}
            work = []
            for param_idx in param_indices:
                shape = self._main_weights[param_idx].shape
                ns_cost = shape.numel() * min(shape) if len(shape) == 2 else 0
                work.append((ns_cost, param_idx))
            target_load = sum(ns_cost for ns_cost, _param_idx in work) / len(shard_ranks)
            for ns_cost, param_idx in sorted(work, key=lambda e: (-e[0], e[1])):
                ranges = self._shard_ranges[param_idx]
                param_numel = self._main_weights[param_idx].shape.numel()
                full_holders = [rank for rank, _offset, size in ranges if size == param_numel]
                holders = [rank for rank, _offset, size in ranges if size > 0]
                size_by_rank = {rank: size for rank, _offset, size in ranges}

                # Prefer no-comm full holders until they reach the average target.
                # For split params, prefer shard holders under target before using
                # a non-holder; larger local shards win ties to limit traffic.
                candidates = [rank for rank in full_holders if load[rank] < target_load]
                if not candidates:
                    candidates = [rank for rank in holders if load[rank] < target_load]
                if not candidates:
                    candidates = shard_ranks

                best = min(candidates, key=lambda rank: (load[rank], -size_by_rank[rank], rank))
                roots[param_idx] = best
                load[best] += ns_cost
        return roots

    def _build_packages(self, package_size=4):
        """Build static Muon packages from the assigned roots."""
        ns_costs = self._ns_costs
        comm_params, comm_free_params = [], []
        for param_idx in range(len(self._main_weights)):
            root = self._roots[param_idx]
            has_comm = any(
                rank != root for rank, _offset, size in self._shard_ranges[param_idx] if size > 0
            )
            (comm_params if has_comm else comm_free_params).append(param_idx)

        comm_params_by_group = OrderedDict()
        for param_idx in comm_params:
            comm_params_by_group.setdefault(self._shard_ranks[param_idx], []).append(param_idx)

        comm_packages = []
        for params_in_group in comm_params_by_group.values():
            params_in_group.sort(key=lambda param_idx: (-ns_costs[param_idx], param_idx))
            for start in range(0, len(params_in_group), package_size):
                comm_packages.append(
                    MuonGradPackage(
                        param_ids=params_in_group[start : start + package_size], has_comm=True
                    )
                )

        no_comm_packages = [
            MuonGradPackage(
                param_ids=comm_free_params[start : start + package_size], has_comm=False
            )
            for start in range(0, len(comm_free_params), package_size)
        ]

        # Put no-comm NS on both sides of the communication packages so it can
        # cover the first gather and the final scatter.
        prefix_target = sum(ns_costs[param_idx] for param_idx in comm_free_params) / 2
        split = 0
        prefix_cost = 0
        while split < len(no_comm_packages) and (split == 0 or prefix_cost < prefix_target):
            prefix_cost += sum(
                ns_costs[param_idx] for param_idx in no_comm_packages[split].param_ids
            )
            split += 1

        return no_comm_packages[:split] + comm_packages + no_comm_packages[split:]

    @torch.no_grad()
    def step(self, closure=None):
        """Run one Muon update over all managed 2D matrix params.

        Per param: (1) momentum on this rank's grad shard; (2) gather the full grad to
        its root; (3) root orthogonalizes (NS); (4) scatter the orthogonalized shards
        back; (5) decoupled-WD update of this rank's master shard.

        Gather/scatter use one batched P2P call per communication package. No-comm
        packages are placed before and after the communicating packages so their NS
        can cover the first gather and final scatter.
        """
        assert closure is None, "FullyShardV2Muon does not support a closure."
        previous_matmul_precision = torch.get_float32_matmul_precision()
        try:
            torch.set_float32_matmul_precision(self._fp32_matmul_prec)
            group = self.param_groups[0]
            lr = group["lr"]  # set by the LR scheduler
            weight_decay = group.get("weight_decay", self._weight_decay)
            gather_reqs = [[] for _ in range(len(self._packages))]
            full_grads = {}

            # Phase 1: update local momentum. Communication packages launch their
            # gather immediately after their shards are ready, overlapping the rest of
            # momentum work and the no-comm NS prefix.
            coef = self._momentum_coef
            for package_idx, package in enumerate(self._packages):
                package.update_momentum(self, coef)
                gather_reqs[package_idx] = package.issue_gather(self, full_grads)

            # Phase 2: consume packages in NS order, run root NS, then launch package
            # scatter for the orthogonalized shards.
            scatter_reqs = []  # in-flight scatter requests per communication package
            owned_orths = {}  # param_idx -> full orth this rank rooted (read by Phase 3b)

            for package_idx, package in enumerate(self._packages):
                orths = package.orthogonalize(self, gather_reqs[package_idx], full_grads)
                owned_orths.update(orths)
                if package.has_comm:
                    scatter_reqs.append((package, package.issue_scatter(self, orths)))

            # Phase 3a: drain scatters before reading remote orth shards from grad.
            for package, requests in scatter_reqs:
                package.finish_scatter(requests)

            # Phase 3b: apply decoupled-WD update from each rank's orth shard.
            for param_idx, (main_weight, grad) in enumerate(
                zip(self._main_weights, self._main_grads)
            ):
                if grad is None:
                    continue
                weight_shard = main_weight.to_local().reshape(-1)
                if weight_shard.numel() == 0:
                    continue
                orth = owned_orths.get(param_idx)
                if orth is not None:  # root: read its own orth segment directly
                    own_offset, own_size = 0, 0
                    for rank, offset, size in self._shard_ranges[param_idx]:
                        if rank == dist.get_rank():
                            own_offset, own_size = offset, size
                            break
                    orth_shard = orth.reshape(-1)[own_offset : own_offset + own_size]
                else:  # non-root holder: scatter delivered the orth shard into the grad
                    orth_shard = grad.to_local().reshape(-1)
                if weight_decay != 0.0:
                    weight_shard.mul_(1.0 - lr * weight_decay)
                weight_shard.add_(orth_shard.to(weight_shard.dtype), alpha=-lr)
        finally:
            torch.set_float32_matmul_precision(previous_matmul_precision)


class FullyShardV2MuonOptimizer(MegatronOptimizer):
    """MegatronOptimizer adapter wrapping FullyShardV2Muon for ChainedOptimizer.

    The inner Muon optimizer updates its FSDP buffers end-to-end (gather -> NS ->
    scatter -> update), so its params are excluded from the chained grad clip /
    grad-norm / zero-count (``get_parameters`` / ``get_main_grads_for_grad_norm``
    return ``[]``) — orthogonalized updates are already self-normalized.
    """

    def __init__(self, config, model_chunks, **muon_hyperparams):
        self.model_chunks = list(model_chunks) if model_chunks else []
        from megatron.core.distributed.fsdp.src.megatron_fsdp.v2 import FSDPModule
        from megatron.core.transformer.fsdp_dtensor_checkpoint import get_global_unique_param_name

        # Build the inner optimizer from the FSDP-v2 2D-matrix dist_params. Grad
        # DTensors are published lazily by FSDP after backward, so the wrapper
        # binds them immediately around each step instead of retaining a stale
        # construction-time snapshot.
        params, names = [], []
        momentum_buffers, momentum_dtensors = [], []
        seen_names = set()
        for chunk in self.model_chunks:
            root = chunk if isinstance(chunk, FSDPModule) else chunk.module
            for _module_name, m in root.named_modules():
                if not isinstance(m, FSDPModule):
                    continue
                for param_names, param_group in m._named_param_groups:
                    if not param_group.requires_grad:
                        continue
                    for _param_name, dist_param in zip(param_names, param_group.dist_params):
                        # filter by the param's global attrs only (rank-consistent); keep
                        # empty local shards aligned so collectives match
                        if dist_param.dim() == 2 and not getattr(
                            dist_param, "is_embedding_or_output_parameter", False
                        ):
                            params.append(dist_param)
                            full_name = get_global_unique_param_name([chunk], dist_param)
                            if full_name in seen_names:
                                raise ValueError(
                                    f"Duplicate Muon parameter name for checkpointing: {full_name}"
                                )
                            if param_group.main_grad_buffer is None:
                                raise RuntimeError(
                                    f"Muon parameter {full_name} has no FSDP main-grad buffer."
                                )
                            dtype = param_group.main_grad_buffer.dtype
                            local_param = dist_param.to_local()
                            local_momentum = torch.zeros(
                                local_param.numel(), dtype=dtype, device=local_param.device
                            )
                            momentum_dtensor = DTensor.from_local(
                                local_tensor=local_momentum.view(local_param.shape),
                                device_mesh=dist_param.device_mesh,
                                placements=dist_param.placements,
                                run_check=False,
                                shape=dist_param.shape,
                                stride=dist_param.stride(),
                            )
                            copy_chunk_metadata(dist_param, momentum_dtensor)
                            names.append(full_name)
                            momentum_buffers.append(local_momentum)
                            momentum_dtensors.append(momentum_dtensor)
                            seen_names.add(full_name)
        super().__init__(
            FullyShardV2Muon(params, [None] * len(params), momentum_buffers, **muon_hyperparams),
            config,
        )
        self._param_names = names
        self._momentum_buffers = momentum_buffers
        self._momentum_dtensors = momentum_dtensors
        self.is_stub_optimizer = False

    # --- Excluded from the chained grad-norm / clip / zero-count machinery. ---
    def get_parameters(self) -> List[torch.nn.Parameter]:
        return []

    def get_main_grads_for_grad_norm(self) -> List[torch.Tensor]:
        """Return no gradients because Muon is excluded from chained grad norms."""
        return []

    def count_zeros(self) -> float:
        return 0.0

    # --- Step contract used by ChainedOptimizer. ---
    def prepare_grads(self) -> bool:
        # FSDP gradients are already reduced into the grad buffers; no loss-scale
        # inf/nan unscaling is applied here. Returns False = "no inf found".
        return False

    @torch.no_grad()
    def step_with_ready_grads(self) -> bool:
        try:
            main_grads = []
            local_errors = []
            for param_idx, dist_param in enumerate(self.optimizer._main_weights):
                grad = getattr(dist_param, "decoupled_grad", None)
                if grad is None:
                    grad = dist_param.grad
                if grad is None:
                    if dist_param.to_local().numel() != 0:
                        param_name = self._param_names[param_idx]
                        local_errors.append(
                            f"Muon parameter {param_name} has a non-empty local shard "
                            "but no optimizer-facing gradient."
                        )
                elif not isinstance(grad, DTensor):
                    param_name = self._param_names[param_idx]
                    local_errors.append(
                        f"Muon gradient for parameter {param_name} must be a DTensor, "
                        f"got {type(grad).__name__}."
                    )
                main_grads.append(grad)

            local_error = " ".join(local_errors) if local_errors else None
            if dist.is_initialized():
                backend = str(dist.get_backend()).lower()
                flag_device = (
                    torch.device("cuda", torch.cuda.current_device())
                    if "nccl" in backend
                    else torch.device("cpu")
                )
                error_flag = torch.tensor(
                    [int(local_error is not None)], dtype=torch.int32, device=flag_device
                )
                dist.all_reduce(error_flag, op=dist.ReduceOp.MAX)
                if error_flag.item():
                    gathered_errors = [None] * dist.get_world_size()
                    dist.all_gather_object(gathered_errors, local_error)
                    details = " ".join(
                        f"rank {rank}: {message}"
                        for rank, message in enumerate(gathered_errors)
                        if message is not None
                    )
                    raise RuntimeError(
                        "Muon gradient validation failed before distributed step. " + details
                    )
            elif local_error is not None:
                raise RuntimeError(
                    "Muon gradient validation failed before distributed step. " + local_error
                )

            self.optimizer._main_grads[:] = main_grads
            self.optimizer.step()
        finally:
            self.optimizer._main_grads[:] = [None] * len(self.optimizer._main_weights)
        # The inner optimizer only updated the fp32 master weights; cast them back into the
        # model (bf16) buffers via the v2 FSDPModule API (it recurses over child
        # FSDPModules, so one call per chunk root suffices).
        from megatron.core.distributed.fsdp.src.megatron_fsdp.v2 import FSDPModule

        for model_chunk in self.model_chunks:
            fsdp_module = model_chunk if isinstance(model_chunk, FSDPModule) else model_chunk.module
            fsdp_module._copy_main_weights_to_model_weights()
        return True

    def step(self):
        if self.prepare_grads():
            return False, None, None
        return self.step_with_ready_grads(), None, None

    # --- Minimal remainder of the MegatronOptimizer interface. ---
    def zero_grad(self, set_to_none: bool = True):
        from megatron.core.distributed.fsdp.src.megatron_fsdp.v2 import FSDPModule

        for model_chunk in self.model_chunks:
            fsdp_module = model_chunk if isinstance(model_chunk, FSDPModule) else model_chunk.module
            fsdp_module.zero_grad(set_to_none=set_to_none)

    def get_loss_scale(self) -> torch.Tensor:
        return torch.tensor([1.0], dtype=torch.float32, device=torch.cuda.current_device())

    def reload_model_params(self, state_dict=None):
        pass

    def _param_name(self, param: torch.Tensor) -> str:
        params = [p for group in self.optimizer.param_groups for p in group["params"]]
        param_to_name = dict(zip(params, self._param_names))
        assert param in param_to_name, f"Muon parameter {param} is not named."
        return param_to_name[param]

    def _param_groups_to_param2group_meta(self) -> dict[str, dict]:
        param_to_group_meta = {}
        for group in self.optimizer.param_groups:
            group_meta = group.copy()
            del group_meta["params"]
            for param in group["params"]:
                param_to_group_meta[self._param_name(param)] = group_meta
        return param_to_group_meta

    def _sync_hyperparams_from_state_dict(self, state_dict: dict):
        param_groups = state_dict.get("param_groups")
        if param_groups is None and "param_to_group_meta" in state_dict:
            param_to_group_meta = state_dict["param_to_group_meta"]
            param_groups = []
            for group in self.optimizer.param_groups:
                group_meta = None
                for param in group["params"]:
                    name = self._param_name(param)
                    if name in param_to_group_meta:
                        group_meta = param_to_group_meta[name]
                        break
                if group_meta is not None:
                    param_groups.append(group_meta)
        if not param_groups:
            return

        group = param_groups[0]
        group = {key: value for key, value in group.items() if key != "params"}
        num_ns_steps = group.get("num_ns_steps", self.optimizer._num_ns_steps)
        coefficient_type = group.get("coefficient_type", self.optimizer._coefficient_type)
        scale_mode = group.get("scale_mode", self.optimizer._scale_mode)
        fp32_matmul_prec = group.get("fp32_matmul_prec", self.optimizer._fp32_matmul_prec)
        _validate_muon_hyperparams(num_ns_steps, coefficient_type, scale_mode, fp32_matmul_prec)
        self.optimizer.param_groups[0].update(group)
        self.optimizer._momentum_coef = group.get("momentum", self.optimizer._momentum_coef)
        self.optimizer._weight_decay = group.get("weight_decay", self.optimizer._weight_decay)
        self.optimizer._nesterov = group.get("nesterov", self.optimizer._nesterov)
        self.optimizer._num_ns_steps = num_ns_steps
        self.optimizer._coefficient_type = coefficient_type
        self.optimizer._scale_mode = scale_mode
        self.optimizer._extra_scale_factor = group.get(
            "extra_scale_factor", self.optimizer._extra_scale_factor
        )
        self.optimizer._fp32_matmul_prec = fp32_matmul_prec

    def _load_momentum_state(self, state: dict):
        named = not all(isinstance(key, int) for key in state.keys())
        for idx, momentum_buffer in enumerate(self._momentum_buffers):
            key = self._param_names[idx] if named else idx
            if key not in state:
                momentum_buffer.zero_()
                continue
            loaded = state[key].get("momentum_buffer")
            if loaded is None:
                momentum_buffer.zero_()
                continue
            loaded_local = loaded.to_local() if isinstance(loaded, DTensor) else loaded
            assert loaded_local.numel() == momentum_buffer.numel(), (
                f"Loaded Muon momentum for param {idx} has {loaded_local.numel()} "
                f"elements, expected {momentum_buffer.numel()}."
            )
            momentum_buffer.copy_(
                loaded_local.reshape(-1).to(
                    device=momentum_buffer.device, dtype=momentum_buffer.dtype
                )
            )

    def state_dict(self):
        param_groups = []
        next_param_id = 0
        for group in self.optimizer.param_groups:
            group_state = {key: value for key, value in group.items() if key != "params"}
            param_count = len(group["params"])
            group_state["params"] = list(range(next_param_id, next_param_id + param_count))
            next_param_id += param_count
            param_groups.append(group_state)
        return {
            "state": {
                idx: {"momentum_buffer": momentum}
                for idx, momentum in enumerate(self._momentum_dtensors)
            },
            "param_groups": param_groups,
        }

    def load_state_dict(self, state_dict):
        self._sync_hyperparams_from_state_dict(state_dict)
        self._load_momentum_state(state_dict["state"])

    def sharded_state_dict(self, model_sharded_state_dict, is_loading: bool = False, **kwargs):
        named_state = {
            param_name: {"momentum_buffer": momentum}
            for param_name, momentum in zip(self._param_names, self._momentum_dtensors)
        }
        return {
            "state": named_state,
            "param_to_group_meta": self._param_groups_to_param2group_meta(),
        }


class FullyShardV2MuonChainedOptimizer(ChainedOptimizer):
    """Adam+Muon chain with an explicit checkpoint boundary."""

    def sharded_state_dict(self, model_sharded_state_dict, is_loading: bool = False, **kwargs):
        raise NotImplementedError(
            "Megatron-FSDP v2 Muon optimizer checkpointing is not yet supported "
            "when Adam and Muon are chained; use --no-save-optim when saving and "
            "--no-load-optim when loading."
        )
