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


class FullyShardV2Muon(torch.optim.Optimizer):
    """Single-root distributed Muon for Megatron-FSDP v2 (a torch optimizer).

    Args:
        params: optimizer-facing params — all assumed to be FSDP-v2 ``dist_param``
            DTensors (asserted). A flat list or param groups, as any torch optimizer.
        grads: the gradient DTensor for each param, aligned 1:1 with the flattened
            params (asserted to be DTensors). This is the sole gradient source —
            replaces the old ParameterGroup back-reference path.
        lr / momentum / nesterov / weight_decay: Muon update hyperparameters.
        num_ns_steps / coefficient_type / scale_mode / extra_scale_factor: NS +
            scaling config used by ``orthogonalize``.
        fp32_matmul_prec / tp_mode: reserved for the Phase-2 tensor-parallel NS path
            (currently TP size 1; accepted and ignored).
        packages: list of param-index lists grouping params for batched P2P (one
            batch_isend_irecv per package; packages pipeline so a package's gather
            overlaps the previous package's NS). Every package must share one dp_group.
            None -> one package per dp_group.
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
        packages=None,
    ) -> None:
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay))

        flat_params = [param for group in self.param_groups for param in group["params"]]
        assert all(
            isinstance(p, DTensor) for p in flat_params
        ), "FullyShardV2Muon expects every param to be a DTensor (FSDP-v2 dist_param)."
        # grads align 1:1 with params; each a Shard(0) DTensor, or None if our shard is empty.
        assert all(
            g is None or isinstance(g, DTensor) for g in grads
        ), "FullyShardV2Muon expects every grad to be a DTensor or None."
        assert len(grads) == len(flat_params), (
            f"grads ({len(grads)}) must align 1:1 with params ({len(flat_params)})."
        )
        self._main_grads = list(grads)

        # Muon hyperparameters. lr is NOT stored — it lives in param_groups[0]["lr"]
        # (where the LR scheduler writes), read in step().
        self._momentum_coef = momentum
        self._nesterov = nesterov
        self._weight_decay = weight_decay
        # NS + scaling config for orthogonalize().
        self._num_ns_steps = num_ns_steps
        self._coefficient_type = coefficient_type
        self._scale_mode = scale_mode
        self._extra_scale_factor = extra_scale_factor

        if not flat_params:
            raise ValueError("FullyShardV2Muon got no parameters to manage.")
        self._main_weights = list(flat_params)

        # Per param: (dp_group, this rank's rank within it, that group's world size).
        self._comm = []
        for main_weight in self._main_weights:
            group = main_weight.device_mesh.get_group()
            self._comm.append((group, dist.get_rank(group), dist.get_world_size(group)))

        # Per-rank (flat_offset, size) of each param + a load-balanced NS root
        # (layout-only, so identical on every rank).
        self._shard_ranges = self._compute_shard_ranges()
        self._roots = self._assign_roots()

        # Persistent full-grad buffer per SPLIT param rooted here (the root assembles its
        # shards for NS). Single-holder params need none — their grad shard already IS the
        # full grad.
        self._full_grads = {}
        for i, main_weight in enumerate(self._main_weights):
            _group, this_rank, _world_size = self._comm[i]
            if self._roots[i] != this_rank:
                continue
            holders = [r for r, (_offset, size) in enumerate(self._shard_ranges[i]) if size > 0]
            if len(holders) <= 1:
                continue
            grad = self._main_grads[i]
            dtype = grad.dtype if grad is not None else main_weight.dtype
            self._full_grads[i] = torch.empty(
                main_weight.shape.numel(), dtype=dtype, device=main_weight.device
            )

        # Each entry: (master weight, grad or None, momentum, post_momentum_shard).
        # post_momentum_shard is THIS rank's slice of the momentum-adjusted gradient —
        # both where Phase 1 puts it AND where gather reads it (isend source / a single
        # root's full grad), so there is no copy:
        #   nesterov: grad + coef*m (a fresh value) -> a split root's full-grad slice, else
        #     the grad shard written in place; momentum is a separate buffer.
        #   non-nesterov: == m -> it IS the momentum buffer (which for a split root already
        #     lives in the full-grad slice); Phase 1 just updates it, nothing else.
        self._managed = []
        for i in range(len(self._main_weights)):
            main_weight = self._main_weights[i]
            grad = self._main_grads[i]
            _group, this_rank, _world_size = self._comm[i]
            this_offset, this_size = self._shard_ranges[i][this_rank]
            dtype = grad.dtype if grad is not None else main_weight.dtype
            # this rank's slice inside the preallocated full grad (split root only, else None)
            grad_slice = (
                self._full_grads[i].reshape(-1)[this_offset : this_offset + this_size]
                if i in self._full_grads
                else None
            )
            if self._nesterov:
                momentum = torch.zeros(this_size, dtype=dtype, device=main_weight.device)
                # nesterov's post-momentum (grad + coef*m) is a fresh value, written into a
                # split root's full-grad slice, else the grad shard in place.
                post_momentum_shard = grad_slice if grad_slice is not None else (
                    grad.to_local().reshape(-1) if grad is not None else None
                )
            else:
                # non-nesterov post-momentum == m, so momentum IS the post-momentum; put it
                # where gather reads it (the full-grad slice for a split root, else a
                # standalone buffer) and let post_momentum_shard alias it.
                if grad_slice is not None:
                    momentum = grad_slice
                    momentum.zero_()
                else:
                    momentum = torch.zeros(this_size, dtype=dtype, device=main_weight.device)
                post_momentum_shard = momentum
            self._managed.append((main_weight, grad, momentum, post_momentum_shard))

        # `packages`: list of param-index lists (into self._main_weights). Each package
        # does ONE batched isend/irecv; packages pipeline (package i+1's gather overlaps
        # package i's NS). Every package must share one dp_group. The wrapper passes one
        # package per FSDPModule (== layer), so layers pipeline:
        #     packages = [[0, 1, 2], [3, 4, 5]]   # layer 0 | layer 1
        # Default (None): one package per dp_group, e.g. dense vs expert params:
        #     packages = [[0, 1, 3], [2, 4]]      # dp-group | edp-group; no intra-group pipeline
        if packages is None:
            by_group = OrderedDict()
            for param_idx in range(len(self._main_weights)):
                by_group.setdefault(self._comm[param_idx][0], []).append(param_idx)
            packages = list(by_group.values())
        self._packages = [list(p) for p in packages]

        # Precompute the static per-package isend/irecv plan; step() only fires the P2Ps.
        self._gather_plans, self._scatter_plans = self._build_comm_plans()

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
          1. group params by dp_group; track each rank's NS load (cost ~ numel*min(dim));
          2. pin single-holder params to their only holder (zero gather/scatter) and add
             their cost to that rank's load;
          3. assign split params heaviest-first (LPT) to the least-loaded holder (ties:
             the holder with more of this param's data, then lowest rank).

        Deterministic + identical on every rank (depends only on shard layout).

        Example (world_size=2):
            param A: single holder {0}, cost 100  -> root 0          (load [100,   0])
            param B: split {0,1},       cost 300  -> root 1 (idler)  (load [100, 300])
            param C: split {0,1},       cost 200  -> root 0          (load [300, 300])
        """
        params_by_group = OrderedDict()
        for param_idx in range(len(self._main_weights)):
            params_by_group.setdefault(self._comm[param_idx][0], []).append(param_idx)
        roots = [0] * len(self._main_weights)
        for _group, param_indices in params_by_group.items():
            world_size = self._comm[param_indices[0]][2]
            load = [0] * world_size
            split = []
            for param_idx in param_indices:
                holders = [
                    rank
                    for rank, (_offset, size) in enumerate(self._shard_ranges[param_idx])
                    if size > 0
                ]
                shape = self._main_weights[param_idx].shape
                ns_cost = shape.numel() * min(shape) if len(shape) == 2 else 0
                if len(holders) == 1:
                    roots[param_idx] = holders[0]
                    load[holders[0]] += ns_cost
                elif len(holders) > 1:
                    split.append((ns_cost, param_idx, holders))
            for ns_cost, param_idx, holders in sorted(split, key=lambda e: (-e[0], e[1])):
                ranges = self._shard_ranges[param_idx]
                best = min(holders, key=lambda rank: (load[rank], -ranges[rank][1], rank))
                roots[param_idx] = best
                load[best] += ns_cost
        return roots

    def _build_comm_plans(self):
        """Precompute the STATIC isend/irecv plan per package; step() just fires the P2Ps,
        which read/write the grad / full-grad / orth tensors directly (no pack, no alloc —
        split-root full grads are preallocated in __init__).

        Both ends walk params in ``package`` order so same-(sender, receiver) P2Ps pair
        up, and each plan is the symmetric {group, send, recv}.
          1. gather — route each rank's grad shard to that param's root:
             - root == this rank, SPLIT: irecv the OTHER holders' shards into the
               preallocated full grad (Phase 1 wrote our own segment).
             - root == this rank, SINGLE holder: no P2P — the grad shard already IS the
               full grad.
             - else we hold a shard: isend it to the root.
          2. scatter — the inverse: isend each OTHER holder's shard out of the orth, irecv
             our shard into the grad. Our own shard is read straight from the orth by
             Phase 3b, never touching the grad.

        Returns (gather_plans, scatter_plans), parallel to self._packages; each a dict:
          gather:  group,
                   send = [(param_idx, root)],                # isend grad.to_local() -> root
                   recv = [(param_idx, src, offset, size)],   # irecv src shard -> full_grad[offset:]
          scatter: group,
                   send = [(param_idx, dst, offset, size)],   # isend orth[offset:] -> dst
                   recv = [(param_idx, root)],                # irecv -> grad.to_local()
        (_finish_gather derives the rooted params on the fly from self._roots /
        self._full_grads.)
        """
        gather_plans, scatter_plans = [], []
        for package in self._packages:
            group, this_rank, world_size = self._comm[package[0]]

            # ---- gather: each rank's grad shard -> that param's root ----
            send, recv = [], []
            for param_idx in package:
                root = self._roots[param_idx]
                ranges = self._shard_ranges[param_idx]
                if root == this_rank:
                    if param_idx in self._full_grads:  # split: assemble from other holders
                        for src in range(world_size):
                            offset, size = ranges[src]
                            if size == 0 or src == this_rank:  # own segment came from Phase 1
                                continue
                            recv.append((param_idx, src, offset, size))
                    # single holder: grad shard already is the full grad -> no P2P
                elif ranges[this_rank][1] > 0:
                    send.append((param_idx, root))
            gather_plans.append({"group": group, "send": send, "recv": recv})

            # ---- scatter: each root's orth shard -> its OTHER holders (inverse of gather);
            #      the root's own shard is read straight from the orth by Phase 3b ----
            send, recv = [], []
            for param_idx in package:
                root = self._roots[param_idx]
                ranges = self._shard_ranges[param_idx]
                if root == this_rank:
                    for dst in range(world_size):
                        offset, size = ranges[dst]
                        if size == 0 or dst == this_rank:  # own shard: used directly, not sent
                            continue
                        send.append((param_idx, dst, offset, size))
                elif ranges[this_rank][1] > 0:
                    recv.append((param_idx, root))
            scatter_plans.append({"group": group, "send": send, "recv": recv})
        return gather_plans, scatter_plans

    @torch.no_grad()
    def _issue_gather(self, p):
        """Fire the package's gather P2Ps: isend our shards to their roots, irecv remote
        shards straight into the preallocated full grad. Returns the in-flight Works."""
        plan = self._gather_plans[p]
        group = plan["group"]
        ops = []
        for param_idx, root in plan["send"]:
            post_momentum_shard = self._managed[param_idx][3]
            ops.append(torch.distributed.P2POp(
                torch.distributed.isend, post_momentum_shard, root, group=group,
            ))
        for param_idx, src, offset, size in plan["recv"]:
            ops.append(torch.distributed.P2POp(
                torch.distributed.irecv,
                self._full_grads[param_idx].reshape(-1)[offset : offset + size], src, group=group,
            ))
        return torch.distributed.batch_isend_irecv(ops) if ops else []

    @torch.no_grad()
    def _finish_gather(self, p, reqs):
        """Wait the gather P2Ps and return {param_idx: full_grad (2D)} for params rooted
        here — split from the preallocated buffer, single from the grad view."""
        for req in reqs:
            req.wait()
        package = self._packages[p]
        this_rank = self._comm[package[0]][1]
        full_grads = {}
        for param_idx in package:
            if self._roots[param_idx] != this_rank:
                continue
            shape = self._main_weights[param_idx].shape
            if param_idx in self._full_grads:  # split: assembled in the preallocated buffer
                full_grads[param_idx] = self._full_grads[param_idx].view(shape)
            else:  # single holder: this rank's post-momentum shard already is the full grad
                post_momentum_shard = self._managed[param_idx][3]
                full_grads[param_idx] = post_momentum_shard.view(shape)
        return full_grads

    @torch.no_grad()
    def _issue_scatter(self, p, orths):
        """Fire the package's scatter P2Ps: isend each remote holder's shard out of the
        orth, irecv our shards into the grad. (The root's own shard is read straight from
        the orth by Phase 3b, never round-tripping the grad.) Returns the in-flight Works.
        The isend source ``orths`` is kept alive by the caller's ``owned_orths``."""
        plan = self._scatter_plans[p]
        group = plan["group"]
        ops = []
        for param_idx, dst, offset, size in plan["send"]:
            ops.append(torch.distributed.P2POp(
                torch.distributed.isend,
                orths[param_idx].reshape(-1)[offset : offset + size], dst, group=group,
            ))
        for param_idx, root in plan["recv"]:
            ops.append(torch.distributed.P2POp(
                torch.distributed.irecv,
                self._main_grads[param_idx].to_local().reshape(-1), root, group=group,
            ))
        return torch.distributed.batch_isend_irecv(ops) if ops else []

    @torch.no_grad()
    def _finish_scatter(self, p, reqs):
        """Wait the scatter P2Ps (the orth shards have landed in this rank's grad)."""
        for req in reqs:
            req.wait()

    @torch.no_grad()
    def orthogonalize(self, param, grad: torch.Tensor) -> torch.Tensor:
        """Newton-Schulz orthogonalize + Muon spectral scale of a 2D gradient.

        Uses emerging_optimizers' production NS when available, else the built-in
        vanilla NS fallback. (Phase 1 runs the non-TP path; ``param`` is accepted
        for parity with the eventual TP / QKV-split path.)
        """
        g = grad.to(torch.float32)
        if HAVE_EMERGING_OPTIMIZERS:
            orth = newton_schulz_tp(
                g,
                steps=self._num_ns_steps,
                coefficient_type=self._coefficient_type,
                tp_group=None,
                partition_dim=None,
                tp_mode="duplicated",
            )
            scale = get_muon_scale_factor(g.size(-2), g.size(-1), mode=self._scale_mode)
        else:
            orth = _vanilla_newton_schulz(g, steps=self._num_ns_steps)
            scale = _vanilla_muon_scale(g.size(-2), g.size(-1))
        return orth * (scale * self._extra_scale_factor)

    @torch.no_grad()
    def step(self, closure=None):
        """Run one Muon update over all managed 2D matrix params.

        Per param: (1) momentum on this rank's grad shard; (2) gather the full grad to
        its root; (3) root orthogonalizes (NS); (4) scatter the orthogonalized shards
        back; (5) decoupled-WD update of this rank's master shard.

        Gather/scatter are batched per package (one async batch_isend_irecv each, into/out
        of the grad and full-grad tensors directly) and pipelined across packages: package
        i+1's gather (NCCL stream) is fired before package i's NS (compute stream) so the
        two overlap.
        """
        assert closure is None, "FullyShardV2Muon does not support a closure."
        lr = self.param_groups[0]["lr"]  # set by the LR scheduler
        items = self._managed

        # Phase 1: momentum on each rank's grad shard, producing post_momentum_shard with
        # no copy. nesterov writes the look-ahead grad + coef*m into it (a split root's
        # full-grad slice / the grad shard). non-nesterov's value == m and
        # post_momentum_shard already aliases momentum, so updating momentum is all we do.
        coef = self._momentum_coef
        for _main_weight, grad, momentum, post_momentum_shard in items:
            if grad is None:
                continue
            grad_shard = grad.to_local().reshape(-1)
            if grad_shard.numel() == 0:
                continue
            momentum.mul_(coef).add_(grad_shard)
            if self._nesterov:
                torch.add(grad_shard, momentum, alpha=coef, out=post_momentum_shard)

        # Phase 2/3 (per-package batched P2P, pipelined): one batch_isend_irecv per
        # package gathers grads to roots; NS runs there; one more scatters orth shards
        # back. Each package's gather is fired before the previous package's NS so the
        # P2Ps (NCCL stream) overlap the NS GEMMs (compute stream).
        num_packages = len(self._packages)
        scatter_reqs = []  # (package index, in-flight scatter Works)
        owned_orths = {}   # param_idx -> full orth this rank rooted (read by Phase 3b)
        gather_reqs = self._issue_gather(0) if num_packages else None
        for i in range(num_packages):
            next_gather_reqs = self._issue_gather(i + 1) if i + 1 < num_packages else None
            full_grads = self._finish_gather(i, gather_reqs)
            orths = {
                param_idx: self.orthogonalize(self._main_weights[param_idx], full_grad).to(
                    full_grad.dtype
                )
                for param_idx, full_grad in full_grads.items()
            }
            owned_orths.update(orths)
            scatter_reqs.append((i, self._issue_scatter(i, orths)))
            gather_reqs = next_gather_reqs

        # Drain scatters: each non-root holder now has its orth shard irecv'd into its grad.
        for i, reqs in scatter_reqs:
            self._finish_scatter(i, reqs)

        # Phase 3b: decoupled-WD update of each rank's master shard (fp32) from its orth
        # shard. The root reads its own segment straight out of the orth it computed (no
        # grad round-trip); every other holder reads the orth shard scatter delivered into
        # its grad.
        for param_idx, (main_weight, grad, _momentum, _post_momentum_shard) in enumerate(items):
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
        # Pull every chunk's 2D matrix dist_param + its grad DTensor straight from the
        # FSDP v2 ParameterGroups, in module-declaration (== layer) order, and build the
        # inner torch optimizer purely from those DTensors (no factory plumbing needed).
        from megatron.core.distributed.fsdp.src.megatron_fsdp.v2 import FSDPModule

        # One package per FSDPModule (== fsdp unit / transformer layer), so each package's
        # gather can pipeline with the next layer's NS. A single FSDPModule's params share
        # one dp_group (the per-package batch_isend_irecv requirement).
        params, grads, packages = [], [], []
        for chunk in self.model_chunks:
            root = chunk if isinstance(chunk, FSDPModule) else chunk.module
            for m in root.modules():
                if not isinstance(m, FSDPModule):
                    continue
                package = []
                for param_group in m._fsdp_param_groups:
                    for dist_param, dist_grad in zip(param_group.dist_params, param_group.dist_grads):
                        # Filter by the PARAM's global attrs only (rank-consistent); the
                        # grad may be None where this rank's shard is empty — keep it
                        # aligned so every rank's managed set matches (collectives align).
                        if dist_param.dim() == 2 and not getattr(
                            dist_param, "is_embedding_or_output_parameter", False
                        ):
                            package.append(len(params))
                            params.append(dist_param)
                            grads.append(dist_grad)
                if package:
                    packages.append(package)
        super().__init__(
            FullyShardV2Muon(params, grads, packages=packages, **muon_hyperparams), config
        )
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
        # Cast the updated fp32 master weights back into the model (bf16) weight buffers
        # via the Megatron-FSDP v2 FSDPModule API. The inner optimizer only updates
        # master weights; this master->model cast (and all ZeRO-1/2/3 / quantization
        # handling) lives in FSDP. _copy_main_weights_to_model_weights recurses over
        # child FSDPModules, so one call on each chunk's root FSDPModule suffices.
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
