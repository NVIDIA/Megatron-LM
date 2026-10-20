"""Explicit HybridEP/virtual-expert adapters for the finite workload benchmark.

These adapters use the existing MoELayer interface, not a proposed scheduler API.
Both modes use identical TE grouped experts. Virtual experts require an intra-node
NVLink EP group; availability is checked separately from numerical qualification.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Geometry:
    ep: int = 2
    experts: int = 8
    topk: int = 2
    tokens: int = 512
    hidden: int = 128
    ffn: int = 256

    def validate(self):
        if not 2 <= self.ep <= 64 or not self.ep <= self.experts <= 8192:
            raise ValueError("requires 2 <= EP <= experts <= 8192, EP <= 64")
        if self.experts % self.ep or not 1 <= self.topk <= min(32, self.experts):
            raise ValueError("experts must divide EP evenly and 1 <= topk <= min(32, experts)")
        if (
            self.tokens <= 0
            or self.hidden <= 0
            or self.hidden % 128
            or self.ffn <= 0
            or self.ffn % 128
        ):
            raise ValueError("positive token count and hidden/FFN sizes divisible by 128 required")


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


class HybridEPAdapter:
    """A BF16, TP=ETP=1, no-recompute MoELayer with explicit process groups."""

    def __init__(self, geometry: Geometry, virtual: bool):
        geometry.validate()
        self.geometry, self.virtual = geometry, virtual
        self.name = "hybridep_virtual_experts" if virtual else "hybridep"
        self.layer = None
        self._route = None

    def capability(self):
        return {
            "backend": self.name,
            "precision": "bf16",
            "ep": self.geometry.ep,
            "tp": 1,
            "etp": 1,
            "fixed_tokens_per_rank": self.geometry.tokens,
            "grouped_experts": True,
            "recompute": "NOT_EXERCISED",
            "cuda_graph": "NOT_EXERCISED",
            "variable_topk": "UNSUPPORTED",
            "zero_source_tokens": "UNSUPPORTED",
            "numerical_qualification": "RUN_REQUIRED",
            "topology": "single-host NVLink EP group required",
        }

    def build(self, pg_collection, seed=1234):
        from megatron.core.models.gpt.gpt_layer_specs import (
            get_gpt_layer_with_transformer_engine_spec,
        )
        from megatron.core.tensor_parallel.random import (
            initialize_rng_tracker,
            model_parallel_cuda_manual_seed,
        )
        from megatron.core.transformer.moe import fused_a2a
        from megatron.core.transformer.moe.moe_layer import MoELayer
        from megatron.core.transformer.spec_utils import get_submodules
        from megatron.core.transformer.transformer_config import TransformerConfig

        if self.layer is not None:
            raise RuntimeError("close the previous layer before build")
        if not fused_a2a.HAVE_HYBRIDEP_DENSE_ROUTING:
            raise RuntimeError("HybridEP with compact topk support is required")
        g = self.geometry
        if pg_collection.ep.size() != g.ep:
            raise ValueError("EP process group does not match geometry")
        initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
        model_parallel_cuda_manual_seed(seed)
        config = TransformerConfig(
            num_layers=1,
            hidden_size=g.hidden,
            ffn_hidden_size=g.ffn,
            moe_ffn_hidden_size=g.ffn,
            num_attention_heads=4,
            num_moe_experts=g.experts,
            expert_model_parallel_size=g.ep,
            expert_tensor_parallel_size=1,
            moe_router_topk=g.topk,
            moe_token_dispatcher_type="flex",
            moe_flex_dispatcher_backend="hybridep",
            moe_virtual_expert_load_balance=self.virtual,
            moe_grouped_gemm=True,
            moe_use_grouped_tensor=True,
            use_transformer_engine_op_fuser=False,
            gradient_accumulation_fusion=True,
            moe_router_dtype="fp32",
            moe_router_load_balancing_type="none",
            moe_aux_loss_coeff=0,
            moe_enable_routing_replay=True,
            moe_router_fusion=False,
            add_bias_linear=False,
            activation_func=torch.nn.functional.silu,
            gated_linear_unit=True,
            bf16=True,
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

    def run_workload(self, hidden, indices, upstream, *, learning_rate=0.0):
        """Time real router, forward, backward and optional SGD; collect evidence afterward."""
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
        ids, gates = canonical_routes(probs, routes, g.topk)
        _, gate_grads = canonical_routes(probs.grad, routes, g.topk)
        if not torch.equal(ids, indices.long().sort(-1).values):
            raise AssertionError("executed routes differ from the logical workload")
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
        plan = None
        if self.virtual:
            manager = self.layer.token_dispatcher._comm_manager
            plan = {
                "runtime_tokens_per_expert": manager.tokens_per_expert.cpu().tolist(),
                "rank_capacity": manager.rank_capacity,
            }
        self._route = None
        return {
            "elapsed_ms": begin.elapsed_time(end),
            "routes": ids.cpu(),
            "values": values,
            "plan": plan,
        }

    def close(self):
        from megatron.core.transformer.moe import fused_a2a
        from megatron.core.transformer.moe.moe_utils import destroy_moe_metrics_tracker
        from megatron.core.transformer.moe.router_replay import RouterReplay
        from megatron.core.transformer.moe.virtual_expert_load_balancer import (
            VirtualExpertLoadBalancer,
        )

        if self.layer is not None:
            torch.cuda.synchronize()
            VirtualExpertLoadBalancer.finalize()
            self.layer = None
            self._route = None
            destroy_moe_metrics_tracker()
            RouterReplay.clear_global_router_replay_instances()
            gc.collect()
            fused_a2a.reset_hybrid_ep_buffer()
