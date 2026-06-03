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

"""DP-distributed Muon optimizer for Megatron-FSDP v2 (single-root, async-pipelined).

Muon orthogonalizes the gradient of 2D matrix parameters via Newton-Schulz (NS).
NS needs the *full* gradient matrix, but under FSDP the gradient is sharded over
the data-parallel (DP) group. This optimizer reconstructs each parameter's full
gradient on a single, load-balanced "root" DP rank (chosen by the parameter
group's ``BufferIndex.item_root_ranks``), runs NS there *once*, and scatters the
orthogonalized gradient back — rather than all-gathering the full gradient to
every rank and running NS redundantly. Momentum and the weight update stay fully
local (elementwise on each rank's shard), so optimizer state remains sharded.

Two classes, following the "MegatronOptimizer wraps a torch optimizer" pattern:

- :class:`FullyShardV2Muon` is a ``torch.optim.Optimizer``. Its ``orthogonalize``
  uses ``emerging_optimizers``' production Newton-Schulz (``newton_schulz_tp`` +
  ``get_muon_scale_factor``) WHEN that optional package is installed, and falls
  back to a self-contained vanilla quintic NS otherwise — so it runs (and can be
  unit-tested) even without the package. Its ``step()`` does the single-root
  gather/scatter + sharded momentum/update.
- :class:`FullyShardV2MuonOptimizer` is the thin ``MegatronOptimizer`` adapter
  that lets the above slot into ``ChainedOptimizer`` alongside Adam.

Per managed 2D matrix parameter, each ``step()`` runs three phases:

1. Local momentum on this rank's gradient shard (sharded optimizer state).
2. Gather the (post-momentum) full gradient to the parameter's root rank
   (``ParameterGroup.unshard_grad_to_root_async``).
3a. Root only: ``self.orthogonalize(param, full_grad)`` (NS + scale), then scatter
    back (``ParameterGroup.scatter_grad_from_root_async``).
3b. Local elementwise weight update on this rank's master shard.

The gather/NS/scatter run as an async pipeline on the FSDP reduce-scatter stream:
each parameter's gather is prefetched and the previous parameter's scatter is left
in flight so they overlap the current parameter's Newton-Schulz, while the
collective ops are still issued in the same parameter order on every rank.

Scope: tensor-parallel size 1 (orthogonalize runs the non-TP NS path). Real
tensor-parallel NS + QKV split and distributed checkpointing of the sharded
momentum are later phases. Needs validation on a GPU node.
"""

from typing import List

import torch

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
        params: optimizer-facing params (FSDP-v2 ``dist_param`` DTensors) or param
            groups — the same form every torch/Megatron optimizer takes. Each
            dist_param carries ``_fsdp_param_group`` / ``_fsdp_orig_param``
            back-references (set in ``ParameterGroup._init_dist_params``) used to
            drive the per-param gather/scatter.
        lr / momentum / nesterov / weight_decay: Muon update hyperparameters.
        num_ns_steps / coefficient_type / scale_mode / extra_scale_factor: NS +
            scaling config used by ``orthogonalize``.
        fp32_matmul_prec / pg_collection / tp_mode: reserved for the Phase-2
            tensor-parallel NS path (currently TP size 1; accepted and ignored).
    """

    def __init__(
        self,
        params,
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
        pg_collection=None,
        tp_mode: str = "duplicated",
    ) -> None:
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay))

        # Muon update hyperparameters owned by this class. lr is NOT stored — it
        # lives in param_groups[0]["lr"] (where an LR scheduler writes), read in step.
        self._momentum_coef = momentum
        self._nesterov = nesterov
        self._weight_decay = weight_decay
        # NS + scaling config for orthogonalize().
        self._num_ns_steps = num_ns_steps
        self._coefficient_type = coefficient_type
        self._scale_mode = scale_mode
        self._extra_scale_factor = extra_scale_factor

        # Resolve the managed 2D matrix params to their FSDP ParameterGroup via
        # the dist_param back-references, and PRE-ALLOCATE the (sharded) momentum
        # buffer for each one here at init (keeps step() free of allocation /
        # lazy-init on the hot path). Iterating self.param_groups in a fixed order
        # keeps the collective send/recv in phases 2/3a aligned across ranks.
        # Each entry: (ParameterGroup, idx, original param, momentum shard).
        self._managed = []
        for group in self.param_groups:
            for dist_param in group["params"]:
                pg = getattr(dist_param, "_fsdp_param_group", None)
                orig = getattr(dist_param, "_fsdp_orig_param", None)
                if pg is None or orig is None:
                    continue  # not an FSDP-v2 param
                if pg.sharding_strategy == "no_shard":
                    raise NotImplementedError(
                        "FullyShardV2Muon does not support the 'no_shard' strategy."
                    )
                if pg.main_grad_buffer is None:
                    continue
                if (
                    orig.requires_grad
                    and orig.dim() == 2
                    and not getattr(orig, "is_embedding_or_output_parameter", False)
                ):
                    idx = pg.param_idx[orig]
                    grad_shard = pg.main_grad_buffer.get_item(idx, as_shard=True)
                    momentum_shard = torch.zeros_like(grad_shard)
                    self._managed.append((pg, idx, orig, momentum_shard))

        if not self._managed:
            raise ValueError("FullyShardV2Muon got no 2D matrix parameters to manage.")


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

        Per param: (1) local momentum on this rank's grad shard; (2) gather the full
        grad to the param's root rank; (3a) root orthogonalizes (NS + scale) and
        scatters the result back into every holder's grad shard; (3b) local
        decoupled-weight-decay update of this rank's master shard. Momentum/update
        stay sharded so optimizer state is sharded.

        The gather/scatter are async (non-blocking ``batch_isend_irecv``); the NCCL
        backend runs them on its own stream, so they overlap the Newton-Schulz
        compute on the main stream — no explicit stream/event management needed. It
        is a 2-stage pipeline: we prefetch the next param's gather so it overlaps the
        current param's NS, and leave each scatter in flight so it overlaps the next
        param's NS (NS dominates, so one gather in flight is enough to hide the comm);
        the scatters are drained at the end. ``work.wait()``
        orders compute-after-comm; the backend orders comm-after-compute (a send
        waits for the grad/NS that produced its buffer). P2P is issued in the same
        param order on every rank, so it is deadlock-free.

        NOTE: the async pipeline is UNVALIDATED on GPU — needs a multi-rank run to
        confirm overlap and numerical parity with a serial reference.
        """
        lr = self.param_groups[0]["lr"]  # set by the LR scheduler
        items = self._managed
        n = len(items)

        # Phase 1: in-place momentum on every rank's grad shard (view into gbuf), so
        # the gathers below see a fully post-momentum grad. Cheap elementwise.
        for pg, idx, _, momentum in items:
            grad_shard = pg.main_grad_buffer.get_item(idx, as_shard=True)
            if grad_shard.numel() > 0:
                momentum.mul_(self._momentum_coef).add_(grad_shard)
                if self._nesterov:
                    grad_shard.add_(momentum, alpha=self._momentum_coef)
                else:
                    grad_shard.copy_(momentum)

        # Phase 2/3a: 2-stage pipeline — prefetch the next param's gather so it
        # overlaps the current param's NS; fire each scatter and don't wait, drain
        # at the end.
        inflight = []        # all in-flight P2P Works + tensors kept alive until drained

        pg, _, param, _ = items[0]
        pending = pg.unshard_grad_to_root_async(param)  # kick off the first gather

        for i in range(n):
            pg, _, param, _ = items[i]
            full, gather_reqs = pending
            if i + 1 < n:                             # prefetch next gather before this NS
                npg, _, nparam, _ = items[i + 1]
                pending = npg.unshard_grad_to_root_async(nparam)

            if full is not None:                      # this rank is param i's root
                for req in gather_reqs:
                    req.wait()                        # only the root waits its gather (NS needs it)
                orth = self.orthogonalize(param, full).to(full.dtype)
                inflight.append((pg.scatter_grad_from_root_async(param, orth), orth, full))
            else:                                     # non-root holder / uninvolved
                # Fire-and-forget: its grad isend (gather) + the scatter recv straight
                # into its grad shard. Don't block here; drained below.
                inflight.append((gather_reqs, None, None))
                inflight.append((pg.scatter_grad_from_root_async(param, None), None, None))

        # Drain all in-flight P2P. The scatter recvs have written each rank's grad
        # shard in place, so Phase 3b can read it directly.
        for reqs, _orth, _full in inflight:
            for req in reqs:
                req.wait()

        # Phase 3b: decoupled-WD update of each rank's master shard from the (now
        # orthogonalized) grad shard, then refresh model weights from master.
        for pg, idx, _, _ in items:
            wbuf = pg.main_weight_buffer if pg.main_weight_buffer is not None else pg.model_weight_buffer
            weight_shard = wbuf.get_item(idx, as_shard=True)
            if weight_shard.numel() == 0:
                continue
            orth_shard = pg.main_grad_buffer.get_item(idx, as_shard=True)
            if self._weight_decay != 0.0:
                weight_shard.mul_(1.0 - lr * self._weight_decay)
            weight_shard.add_(orth_shard.to(weight_shard.dtype), alpha=-lr)

        refreshed = set()
        for pg, _, _, _ in items:
            if id(pg) in refreshed or pg.main_weight_buffer is None:
                continue
            pg.copy_main_weights_to_model_weights()
            refreshed.add(id(pg))


class FullyShardV2MuonOptimizer(MegatronOptimizer):
    """MegatronOptimizer adapter that lets FullyShardV2Muon slot into ChainedOptimizer.

    Presents the MegatronOptimizer interface around the inner FullyShardV2Muon
    (a torch-style optimizer that updates the FSDP buffers directly). The actual
    Muon update runs in ``step_with_ready_grads`` (-> ``self.optimizer.step()``).

    NOTE (UNVALIDATED): the inner Muon optimizer updates its 2D matrix params
    end-to-end on the FSDP buffers (gather -> NS -> scatter -> update), so those
    params are intentionally excluded from the ChainedOptimizer's global gradient
    clipping / zero-counting / grad-norm (``get_parameters`` and
    ``get_main_grads_for_grad_norm`` return ``[]``). Orthogonalized Muon updates
    are already self-normalized, so this matches common practice; revisit if a
    joint Muon+Adam grad-norm is desired. Checkpointing of the sharded momentum
    is a TODO. This whole adapter needs validation on a GPU node.
    """

    def __init__(self, muon: "FullyShardV2Muon", config, model_chunks=None):
        super().__init__(muon, config)
        self.model_chunks = list(model_chunks) if model_chunks else []
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
