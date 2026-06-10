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

from .emerging_optimizers import HAVE_EMERGING_OPTIMIZERS
from .optimizer import MegatronOptimizer

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


@dataclass
class MuonGradPackage:
    """Static Muon package over aligned main-weight / grad / momentum indices."""

    param_ids: List[int]
    has_comm: bool

    @torch.no_grad()
    def update_momentum(self, opt, coef):
        for param_idx in self.param_ids:
            grad = opt._main_grads[param_idx]
            momentum = opt._momentum_buffers[param_idx]
            if grad is None:
                continue
            grad_shard = grad.to_local().reshape(-1)
            if grad_shard.numel() == 0:
                continue
            momentum.mul_(coef).add_(grad_shard)
            if opt._nesterov:
                torch.add(grad_shard, momentum, alpha=coef, out=grad_shard)

    @torch.no_grad()
    def issue_gather(self, opt, full_grads):
        """Gather this package's NS input shards to each param root."""
        if not self.has_comm:
            return []

        group, this_rank, world_size = opt._comm[self.param_ids[0]]
        ops = []
        for param_idx in self.param_ids:
            root = opt._roots[param_idx]
            ranges = opt._shard_ranges[param_idx]
            this_offset, this_size = ranges[this_rank]
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
                    main_weight.shape.numel(),
                    dtype=momentum.dtype,
                    device=main_weight.device,
                )
                full_grads[param_idx] = full_grad
                if this_size > 0:
                    full_grad[this_offset : this_offset + this_size].copy_(ns_input_shard)
                for src in range(world_size):
                    offset, size = ranges[src]
                    if size == 0 or src == this_rank:
                        continue
                    ops.append(
                        dist.P2POp(
                            dist.irecv,
                            full_grad[offset : offset + size],
                            dist.get_global_rank(group, src),
                            group=group,
                        )
                    )
            elif this_size > 0:
                ops.append(
                    dist.P2POp(
                        dist.isend,
                        ns_input_shard,
                        dist.get_global_rank(group, root),
                        group=group,
                    )
                )
        return dist.batch_isend_irecv(ops) if ops else []

    @torch.no_grad()
    def finish_gather(self, requests):
        for request in requests:
            request.wait()

    @torch.no_grad()
    def orthogonalize(self, opt, requests, full_grads):
        self.finish_gather(requests)
        orths = {}
        for param_idx in self.param_ids:
            if opt._roots[param_idx] != opt._comm[param_idx][1]:
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
                scale = get_muon_scale_factor(
                    grad.size(-2), grad.size(-1), mode=opt._scale_mode
                )
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

        group, this_rank, world_size = opt._comm[self.param_ids[0]]
        ops = []
        for param_idx in self.param_ids:
            root = opt._roots[param_idx]
            ranges = opt._shard_ranges[param_idx]
            if root == this_rank:
                orth = orths[param_idx].reshape(-1)
                for dst in range(world_size):
                    offset, size = ranges[dst]
                    if size == 0 or dst == this_rank:
                        continue
                    ops.append(
                        dist.P2POp(
                            dist.isend,
                            orth[offset : offset + size],
                            dist.get_global_rank(group, dst),
                            group=group,
                        )
                    )
            elif ranges[this_rank][1] > 0:
                ops.append(
                    dist.P2POp(
                        dist.irecv,
                        opt._main_grads[param_idx].to_local().reshape(-1),
                        dist.get_global_rank(group, root),
                        group=group,
                    )
                )
        return dist.batch_isend_irecv(ops) if ops else []

    @torch.no_grad()
    def finish_scatter(self, requests):
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
        grads: the gradient DTensor for each param, aligned 1:1 with the flattened
            params (asserted to be DTensors). This is the sole gradient source —
            replaces the old ParameterGroup back-reference path.
        lr / momentum / nesterov / weight_decay: Muon update hyperparameters.
        num_ns_steps / coefficient_type / scale_mode / extra_scale_factor: NS +
            scaling config used by ``orthogonalize``.
        fp32_matmul_prec / tp_mode: reserved for the future tensor-parallel NS path
            (currently TP size 1; accepted and ignored). Params are grouped into
            communication and no-communication packages after root assignment.
    """

    def __init__(
        self,
        params,
        grads,
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
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay))

        param_list = [param for group in self.param_groups for param in group["params"]]
        grad_list = list(grads)
        if not param_list:
            raise ValueError("FullyShardV2Muon got no parameters to manage.")
        assert all(
            isinstance(param, DTensor) for param in param_list
        ), "FullyShardV2Muon expects every param to be a DTensor (FSDP-v2 dist_param)."
        assert all(
            grad is None or isinstance(grad, DTensor) for grad in grad_list
        ), "FullyShardV2Muon expects every grad to be a DTensor or None."

        self._main_weights = param_list
        self._main_grads = grad_list

        # Hyperparameters read by step() and MuonGradPackage methods.
        self._momentum_coef = momentum
        self._weight_decay = weight_decay
        self._nesterov = nesterov
        self._num_ns_steps = num_ns_steps
        self._coefficient_type = coefficient_type
        self._scale_mode = scale_mode
        self._extra_scale_factor = extra_scale_factor

        # Per param: (dp_group, this rank's rank within it, that group's world size).
        self._comm = []
        for main_weight in self._main_weights:
            group = main_weight.device_mesh.get_group()
            self._comm.append((group, dist.get_rank(group), dist.get_world_size(group)))

        self._shard_ranges = self._compute_shard_ranges()
        self._roots = self._assign_roots()

        self._momentum_buffers = []
        for param_idx, (main_weight, grad) in enumerate(zip(self._main_weights, self._main_grads)):
            _offset, local_size = self._shard_ranges[param_idx][self._comm[param_idx][1]]
            dtype = grad.dtype if grad is not None else main_weight.dtype
            momentum_buffer = torch.zeros(local_size, dtype=dtype, device=main_weight.device)
            self._momentum_buffers.append(momentum_buffer)

        self._ns_costs = [
            main_weight.shape.numel() * min(main_weight.shape)
            if len(main_weight.shape) == 2
            else 0
            for main_weight in self._main_weights
        ]
        self._packages = self._build_packages()

    def _compute_shard_ranges(self):
        """Compute, per param, each DP rank's (flat_offset, size) within the full param.

        Steps:
          1. group params by dp_group (one all_gather per group);
          2. all_gather this rank's per-param row counts -> all_ranks_rows[rank][j];
          3. per param: size = rows * row_numel; prefix-sum the sizes into (offset, size).

        Example (world_size=4, param shape (10, 8) so row_numel=8, rows [3, 3, 2, 2]):
            ranges = [(0, 24), (24, 24), (48, 16), (64, 16)]
            #           rank0     rank1     rank2      rank3      (size == 0 => no shard)
        """
        params_by_group = OrderedDict()
        for param_idx in range(len(self._main_weights)):
            params_by_group.setdefault(self._comm[param_idx][0], []).append(param_idx)
        ranges = [None] * len(self._main_weights)
        for group, param_indices in params_by_group.items():
            world_size = dist.get_world_size(group)
            this_rank_rows = [self._main_weights[p].to_local().shape[0] for p in param_indices]
            all_ranks_rows = [None] * world_size  # all_ranks_rows[rank][j] = rank's rows for param j
            dist.all_gather_object(all_ranks_rows, this_rank_rows, group=group)
            for j, param_idx in enumerate(param_indices):
                main_weight = self._main_weights[param_idx]
                row_numel = 1
                for dim in main_weight.shape[1:]:
                    row_numel *= dim
                rank_ranges, offset = [], 0
                for rank in range(world_size):
                    size = all_ranks_rows[rank][j] * row_numel
                    rank_ranges.append((offset, size))
                    offset += size
                ranges[param_idx] = rank_ranges
        return ranges

    def _assign_roots(self):
        """Assign each param an NS root rank, load-balancing NS work per dp_group.

        Steps:
          1. group params by dp_group and estimate each param's NS cost;
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
            params_by_group.setdefault(self._comm[param_idx][0], []).append(param_idx)
        roots = [0] * len(self._main_weights)
        for _group, param_indices in params_by_group.items():
            world_size = self._comm[param_indices[0]][2]
            load = [0] * world_size
            work = []
            for param_idx in param_indices:
                shape = self._main_weights[param_idx].shape
                ns_cost = shape.numel() * min(shape) if len(shape) == 2 else 0
                work.append((ns_cost, param_idx))
            target_load = sum(ns_cost for ns_cost, _param_idx in work) / world_size
            for ns_cost, param_idx in sorted(work, key=lambda e: (-e[0], e[1])):
                ranges = self._shard_ranges[param_idx]
                param_numel = self._main_weights[param_idx].shape.numel()
                full_holders = [
                    rank for rank, (_offset, size) in enumerate(ranges) if size == param_numel
                ]
                holders = [rank for rank, (_offset, size) in enumerate(ranges) if size > 0]

                # Prefer no-comm full holders until they reach the average target.
                # For split params, prefer shard holders under target before using
                # a non-holder; larger local shards win ties to limit traffic.
                candidates = [rank for rank in full_holders if load[rank] < target_load]
                if not candidates:
                    candidates = [rank for rank in holders if load[rank] < target_load]
                if not candidates:
                    candidates = range(world_size)

                best = min(candidates, key=lambda rank: (load[rank], -ranges[rank][1], rank))
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
                rank != root
                for rank, (_offset, size) in enumerate(self._shard_ranges[param_idx])
                if size > 0
            )
            (comm_params if has_comm else comm_free_params).append(param_idx)

        comm_params_by_group = OrderedDict()
        for param_idx in comm_params:
            comm_params_by_group.setdefault(self._comm[param_idx][0], []).append(param_idx)

        comm_packages = []
        for params_in_group in comm_params_by_group.values():
            params_in_group.sort(key=lambda param_idx: (-ns_costs[param_idx], param_idx))
            for start in range(0, len(params_in_group), package_size):
                comm_packages.append(
                    MuonGradPackage(
                        param_ids=params_in_group[start : start + package_size],
                        has_comm=True,
                    )
                )

        no_comm_packages = [
            MuonGradPackage(
                param_ids=comm_free_params[start : start + package_size],
                has_comm=False,
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
        lr = self.param_groups[0]["lr"]  # set by the LR scheduler
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
        owned_orths = {}   # param_idx -> full orth this rank rooted (read by Phase 3b)

        for package_idx, package in enumerate(self._packages):
            orths = package.orthogonalize(self, gather_reqs[package_idx], full_grads)
            owned_orths.update(orths)
            if package.has_comm:
                scatter_reqs.append((package, package.issue_scatter(self, orths)))

        # Phase 3a: drain scatters before reading remote orth shards from grad.
        for package, requests in scatter_reqs:
            package.finish_scatter(requests)

        # Phase 3b: apply decoupled-WD update from each rank's orth shard.
        for param_idx, (main_weight, grad) in enumerate(zip(self._main_weights, self._main_grads)):
            if grad is None:
                continue
            weight_shard = main_weight.to_local().reshape(-1)
            if weight_shard.numel() == 0:
                continue
            orth = owned_orths.get(param_idx)
            if orth is not None:  # this rank is the root: read its own orth segment directly
                this_rank = self._comm[param_idx][1]
                own_offset, own_size = self._shard_ranges[param_idx][this_rank]
                orth_shard = orth.reshape(-1)[own_offset : own_offset + own_size]
            else:  # non-root holder: scatter delivered the orth shard into the grad
                orth_shard = grad.to_local().reshape(-1)
            if self._weight_decay != 0.0:
                weight_shard.mul_(1.0 - lr * self._weight_decay)
            weight_shard.add_(orth_shard.to(weight_shard.dtype), alpha=-lr)


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

        # Build the inner optimizer from the FSDP-v2 2D-matrix dist_params + their grad
        # DTensors. The inner optimizer orders params by communication need.
        params, grads = [], []
        for chunk in self.model_chunks:
            root = chunk if isinstance(chunk, FSDPModule) else chunk.module
            for m in root.modules():
                if not isinstance(m, FSDPModule):
                    continue
                for param_group in m._fsdp_param_groups:
                    for dist_param, dist_grad in zip(param_group.dist_params, param_group.dist_grads):
                        # filter by the param's global attrs only (rank-consistent); keep
                        # grad aligned even when None (empty shard) so collectives match
                        if dist_param.dim() == 2 and not getattr(
                            dist_param, "is_embedding_or_output_parameter", False
                        ):
                            params.append(dist_param)
                            grads.append(dist_grad)
        super().__init__(FullyShardV2Muon(params, grads, **muon_hyperparams), config)
        self.is_stub_optimizer = False

    # --- Excluded from the chained grad-norm / clip / zero-count machinery. ---
    def get_parameters(self) -> List[torch.nn.Parameter]:
        return []

    def get_main_grads_for_grad_norm(self) -> List[torch.Tensor]:
        return []

    # --- Step contract used by ChainedOptimizer. ---
    def prepare_grads(self) -> bool:
        # FSDP gradients are already reduced into the grad buffers; no loss-scale
        # inf/nan unscaling is applied here. Returns False = "no inf found".
        return False

    @torch.no_grad()
    def step_with_ready_grads(self) -> bool:
        self.optimizer.step()
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
        # FSDP zeroes its grad buffers via the module's own grad-sync path.
        pass

    def get_loss_scale(self) -> torch.Tensor:
        return torch.tensor([1.0], dtype=torch.float32, device=torch.cuda.current_device())

    def reload_model_params(self, state_dict=None):
        pass

    def state_dict(self):
        # TODO: checkpoint the sharded Muon momentum buffers.
        return {}

    def load_state_dict(self, state_dict):
        # TODO: restore the sharded Muon momentum buffers.
        pass

    def sharded_state_dict(self, model_sharded_state_dict, is_loading: bool = False, **kwargs):
        # TODO: sharded checkpoint support for the Muon momentum.
        return {}
