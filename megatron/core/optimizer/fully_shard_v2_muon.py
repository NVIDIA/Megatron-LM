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
                param_group = getattr(dist_param, "_fsdp_param_group", None)
                orig = getattr(dist_param, "_fsdp_orig_param", None)
                if param_group is None or orig is None:
                    continue  # not an FSDP-v2 param
                if param_group.sharding_strategy == "no_shard":
                    raise NotImplementedError(
                        "FullyShardV2Muon does not support the 'no_shard' strategy."
                    )
                if param_group.main_grad_buffer is None:
                    continue
                if (
                    orig.requires_grad
                    and orig.dim() == 2
                    and not getattr(orig, "is_embedding_or_output_parameter", False)
                ):
                    idx = param_group.param_idx[orig]
                    grad_shard = param_group.main_grad_buffer.get_item(idx, as_shard=True)
                    momentum_shard = torch.zeros_like(grad_shard)
                    self._managed.append((param_group, idx, orig, momentum_shard))

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

        Per param, in order:
          1. momentum  - update this rank's grad shard locally (stays sharded);
          2. gather    - send the full grad to the param's root rank;
          3. NS        - root orthogonalizes the full grad (Newton-Schulz);
          4. scatter   - root sends each orthogonalized shard back to its holder;
          5. update    - each rank applies a local decoupled-WD step to its master shard.

        Gather/scatter are async: all gathers (step 2) fire up front so the later ones
        overlap the NS (step 3) of earlier params.
        """
        assert closure is None, "FullyShardV2Muon does not support a closure."
        lr = self.param_groups[0]["lr"]  # set by the LR scheduler
        items = self._managed

        # Phase 1: in-place momentum on each rank's grad shard, so gathers see post-momentum grad.
        for param_group, idx, _, momentum in items:
            grad_shard = param_group.main_grad_buffer.get_item(idx, as_shard=True)
            if grad_shard.numel() > 0:
                momentum.mul_(self._momentum_coef).add_(grad_shard)
                if self._nesterov:
                    grad_shard.add_(momentum, alpha=self._momentum_coef)
                else:
                    grad_shard.copy_(momentum)

        # Phase 2: fire ALL gathers up front so later ones overlap Phase-3 NS.
        gather_work = []   # (param_group, param, full_grad_or_None, reqs)
        for param_group, idx, param, _ in items:
            full_grad, reqs = param_group.unshard_grad_to_root_async(param)
            gather_work.append((param_group, param, full_grad, reqs))

        # Phase 3: root waits its own gather, orthogonalizes, fires scatter; non-root fires recv.
        scatter_work = []   # (reqs, orth_keepalive)
        for param_group, param, full_grad, reqs in gather_work:
            if full_grad is not None:                       # this rank is the param's root
                for req in reqs:
                    req.wait()
                orth = self.orthogonalize(param, full_grad).to(full_grad.dtype)
                scatter_work.append((param_group.scatter_grad_from_root_async(param, orth), orth))
            else:                                           # non-root holder / uninvolved
                scatter_work.append((param_group.scatter_grad_from_root_async(param, None), None))

        # Drain: non-root gather isends, then all scatters (recvs wrote each grad shard in place).
        for _param_group, _param, _full_grad, reqs in gather_work:
            for req in reqs:
                req.wait()
        for reqs, _orth in scatter_work:
            for req in reqs:
                req.wait()

        # Phase 3b: decoupled-WD master-shard update from the orthogonalized grad, then refresh weights.
        for param_group, idx, _, _ in items:
            wbuf = (
                param_group.main_weight_buffer
                if param_group.main_weight_buffer is not None
                else param_group.model_weight_buffer
            )
            weight_shard = wbuf.get_item(idx, as_shard=True)
            if weight_shard.numel() == 0:
                continue
            orth_shard = param_group.main_grad_buffer.get_item(idx, as_shard=True)
            if self._weight_decay != 0.0:
                weight_shard.mul_(1.0 - lr * self._weight_decay)
            weight_shard.add_(orth_shard.to(weight_shard.dtype), alpha=-lr)

        refreshed = set()
        for param_group, _, _, _ in items:
            if id(param_group) in refreshed or param_group.main_weight_buffer is None:
                continue
            param_group.copy_main_weights_to_model_weights()
            refreshed.add(id(param_group))


class FullyShardV2MuonOptimizer(MegatronOptimizer):
    """MegatronOptimizer adapter wrapping FullyShardV2Muon for ChainedOptimizer.

    The inner Muon optimizer updates its FSDP buffers end-to-end (gather -> NS ->
    scatter -> update), so its params are excluded from the chained grad clip /
    grad-norm / zero-count (``get_parameters`` / ``get_main_grads_for_grad_norm``
    return ``[]``) — orthogonalized updates are already self-normalized.
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
