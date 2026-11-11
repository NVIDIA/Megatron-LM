# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""One MoELayer per arm, driven with replayed logical routes.

Every arm is the repository's own ``MoELayer`` built from ``arms.config_for``: the same TE
grouped experts, router, precision and initial parameters, with only the arm's dispatcher,
load balancer or scheduler settings changed. The router still runs; its top-k is replaced by
the workload through ``RouterReplay``, and the routes the layer actually executed are checked
against the workload at every call.
"""

from __future__ import annotations

import gc
from typing import Any

import torch
from arms import Arm, Geometry, config_for


def canonical_routes(probs, routes, topk):
    """Normalize full-width bool maps and compact IDs without changing gate gradients."""
    if routes.dtype == torch.bool:
        if not torch.all(routes.sum(-1) == topk):
            raise AssertionError("router did not preserve exactly topk assignments")
        ids = routes.to(torch.int8).topk(topk, dim=-1).indices.sort(-1).values
        return ids, probs.gather(1, ids)
    ids, order = routes.long().sort(-1)
    return ids, probs.gather(1, order)


def numerical_error(actual, expected):
    """Relative L2 and peak-normalized absolute error, with an exact zero reference gate."""
    actual, expected = actual.double(), expected.double()
    if actual.shape != expected.shape or not torch.isfinite(actual).all():
        return {"relative_l2": float("inf"), "relative_peak": float("inf")}
    delta = actual - expected
    norm, peak = expected.norm().item(), expected.abs().max().item()
    return {
        "relative_l2": (
            delta.norm().item() / norm if norm else (0.0 if not delta.any() else float("inf"))
        ),
        "relative_peak": (
            delta.abs().max().item() / peak if peak else (0.0 if not delta.any() else float("inf"))
        ),
    }


class ArmAdapter:
    """A BF16, TP=ETP=1, no-recompute MoELayer for one arm, with explicit process groups."""

    def __init__(
        self,
        geometry: Geometry,
        arm: Arm,
        *,
        te_op_fuser: bool = False,
        scheduler_transport: str = "replica_nccl",
    ):
        geometry.validate()
        self.geometry, self.arm = geometry, arm
        self.name = arm.name
        self.config_kwargs = config_for(
            arm, geometry, te_op_fuser=te_op_fuser, scheduler_transport=scheduler_transport
        )
        self.layer = None
        self._route = None

    def capability(self) -> dict[str, Any]:
        """What this arm exercises in this run; anything not exercised is marked as such."""
        return {
            "backend": self.name,
            "family": self.arm.family,
            "source": self.arm.source,
            "precision": "bf16",
            "ep": self.geometry.ep,
            "tp": 1,
            "etp": 1,
            "fixed_tokens_per_rank": self.geometry.tokens,
            "grouped_experts": True,
            "te_op_fuser": self.config_kwargs["use_transformer_engine_op_fuser"],
            "recompute": "NOT_EXERCISED",
            "cuda_graph": "NOT_EXERCISED",
            "variable_topk": "UNSUPPORTED",
            "zero_source_tokens": "UNSUPPORTED",
        }

    def build(self, pg_collection, seed=1234):
        """Construct the layer. Every arm seeds identically, and the caller copies one shared
        set of logical parameters into it afterwards."""
        from megatron.core.models.gpt.gpt_layer_specs import (
            get_gpt_layer_with_transformer_engine_spec,
        )
        from megatron.core.tensor_parallel.random import (
            initialize_rng_tracker,
            model_parallel_cuda_manual_seed,
        )
        from megatron.core.transformer.moe.moe_layer import MoELayer
        from megatron.core.transformer.spec_utils import get_submodules
        from megatron.core.transformer.transformer_config import TransformerConfig

        if self.layer is not None:
            raise RuntimeError("close the previous layer before build")
        g = self.geometry
        if pg_collection.ep.size() != g.ep:
            raise ValueError("EP process group does not match geometry")
        initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
        model_parallel_cuda_manual_seed(seed)
        config = TransformerConfig(
            **self.config_kwargs,
            activation_func=torch.nn.functional.silu,
            params_dtype=torch.bfloat16,
        )
        spec = get_submodules(
            get_gpt_layer_with_transformer_engine_spec(
                num_experts=g.experts, moe_grouped_gemm=True
            ).submodules.mlp
        )
        self.layer = MoELayer(config, spec, pg_collection=pg_collection).cuda()
        for parameter in self.layer.parameters():
            parameter.main_grad = torch.zeros_like(parameter, dtype=torch.float32)
            parameter.grad_added_to_main_grad = False
        self.layer.router.register_forward_hook(self._capture_routes)
        return self.layer

    def _capture_routes(self, module, args, output):
        probs, routes = output
        probs.retain_grad()
        self._route = probs, routes

    def run_workload(self, hidden, indices, upstream, *, learning_rate=0.0, collect=True):
        """Time router, forward, backward and optional SGD; collect evidence afterwards."""
        from megatron.core.transformer.moe.router_replay import RouterReplayAction

        if self.layer is None:
            raise RuntimeError("build must precede run_workload")
        g = self.geometry
        if hidden.shape != (g.tokens, 1, g.hidden) or indices.shape != (g.tokens, g.topk):
            raise ValueError("fixed hidden/route geometry violated")
        x = hidden.detach().clone().requires_grad_()
        replay = self.layer.router.router_replay
        replay.clear_indices()
        replay.set_target_indices(indices)
        replay.set_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
        for parameter in self.layer.parameters():
            parameter.grad = None
            parameter.main_grad.zero_()
            parameter.grad_added_to_main_grad = False
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
        y, bias = self.layer(x)
        if bias is not None:
            raise AssertionError("unexpected expert bias")
        y.backward(upstream)
        if learning_rate:
            with torch.no_grad():
                for name, parameter in self.layer.named_parameters():
                    grad = parameter.main_grad if name.startswith("experts.") else parameter.grad
                    if grad is not None:
                        parameter.add_(grad, alpha=-learning_rate)
        end.record()
        end.synchronize()
        probs, routes = self._route
        self._route = None
        ids, gates = canonical_routes(probs, routes, g.topk)
        if not torch.equal(ids, indices.long().sort(-1).values):
            raise AssertionError("executed routes differ from the logical workload")
        result = {"elapsed_ms": begin.elapsed_time(end), "routes": ids.cpu(), "values": None}
        if not collect:
            if not torch.isfinite(y).all():
                raise AssertionError("nonfinite output")
            return result
        _, gate_grads = canonical_routes(probs.grad, routes, g.topk)
        values = {
            "output": y.detach().cpu(),
            "input_grad": x.grad.cpu(),
            "gates": gates.detach().cpu(),
            "gate_grad": gate_grads.cpu(),
        }
        for name, parameter in self.layer.named_parameters():
            grad = parameter.main_grad if name.startswith("experts.") else parameter.grad
            if grad is not None:
                values["grad:" + name] = grad.detach().cpu().clone()
            if learning_rate:
                values["parameter:" + name] = parameter.detach().cpu().clone()
        if not all(torch.isfinite(value).all() for value in values.values()):
            raise AssertionError("nonfinite output, gradient or updated parameter")
        result["values"] = values
        result["plan"] = self.runtime_plan()
        return result

    def runtime_plan(self):
        """The virtual-expert planner's last runtime loads, when the arm has one."""
        manager = getattr(self.layer.token_dispatcher, "_comm_manager", None)
        if self.arm.family != "virtual_experts" or manager is None:
            return None
        tokens = getattr(manager, "tokens_per_expert", None)
        return {
            "runtime_tokens_per_expert": tokens.cpu().tolist() if tokens is not None else None,
            "rank_capacity": getattr(manager, "rank_capacity", None),
        }

    def close(self):
        """Release the layer and the process-wide state its arm created."""
        from megatron.core.transformer.moe import fused_a2a
        from megatron.core.transformer.moe.moe_logging import destroy_moe_metrics_tracker
        from megatron.core.transformer.moe.router_replay import RouterReplay
        from megatron.core.transformer.moe.virtual_expert_load_balancer import (
            VirtualExpertLoadBalancer,
        )

        if self.layer is not None:
            torch.cuda.synchronize()
            if self.arm.family == "virtual_experts":
                VirtualExpertLoadBalancer.finalize()
            self.layer = None
            self._route = None
            destroy_moe_metrics_tracker()
            RouterReplay.clear_global_router_replay_instances()
            gc.collect()
            if self.config_kwargs.get("moe_token_dispatcher_type") == "flex":
                fused_a2a.reset_hybrid_ep_buffer()
