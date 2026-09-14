"""CUDA unit tests for migrated MoE accuracy-compatible production paths."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import torch

ROOT = Path(__file__).resolve().parents[4]
_UAC = {"on": False}


def _use_accuracy_compatible():
    return _UAC["on"]


class MoECudaGraphPartialCaptureSignal(Exception):
    pass


class NVLSAllGatherVDispatcher:
    pass


def _load_fn(rel: str, name: str):
    src = (ROOT / rel).read_text()
    tree = ast.parse(src)
    class_name = (
        "MoEFlexTokenDispatcher"
        if rel.endswith("token_dispatcher.py")
        else "MoELayer" if rel.endswith("moe_layer.py") else None
    )
    body = tree.body
    if class_name:
        body = next(
            node.body
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        )
    target = next(node for node in body if isinstance(node, ast.FunctionDef) and node.name == name)
    target.decorator_list = []
    mod = ast.Module(body=[target], type_ignores=[])
    ast.fix_missing_locations(mod)
    ns = {
        "torch": torch,
        "Optional": Optional,
        "_use_accuracy_compatible": _use_accuracy_compatible,
        "MoECudaGraphPartialCaptureSignal": MoECudaGraphPartialCaptureSignal,
        "NVLSAllGatherVDispatcher": NVLSAllGatherVDispatcher,
        "Tuple": tuple,
    }
    exec(compile(mod, rel, "exec"), ns)
    return ns[name]


token_dispatch = _load_fn("megatron/core/transformer/moe/token_dispatcher.py", "token_dispatch")
token_combine = _load_fn("megatron/core/transformer/moe/token_dispatcher.py", "token_combine")
moe_forward = _load_fn("megatron/core/transformer/moe/moe_layer.py", "forward")
moe_postprocess = _load_fn("megatron/core/transformer/moe/moe_layer.py", "postprocess")
topk_routing_with_score_function = _load_fn(
    "megatron/core/transformer/moe/moe_utils.py", "topk_routing_with_score_function"
)


class _FakeComm:
    def __init__(self):
        self.calls = []
        self.dispatched_probs = None

    def dispatch(self, hidden, async_finish, allocate_on_comm_stream):
        self.calls.append(("dispatch", async_finish, allocate_on_comm_stream))
        self.dispatched_probs = hidden
        return hidden

    def combine(self, hidden, async_finish, allocate_on_comm_stream):
        self.calls.append(("combine", async_finish, allocate_on_comm_stream))
        return hidden

    def combine_postprocess(self, output):
        return output


class _Flex:
    @property
    def config(self):
        return SimpleNamespace(dsa_accuracy_compatible=_UAC["on"])

    def __init__(self):
        self.shared_experts = None
        self._comm_manager = _FakeComm()

    token_dispatch = token_dispatch
    token_combine = token_combine


class _MoE:
    @property
    def config(self):
        return SimpleNamespace(
            dsa_accuracy_compatible=_UAC["on"],
            sequence_parallel=True,
            moe_shared_expert_overlap=False,
            moe_latent_size=0,
            fp8=False,
            fp4=False,
        )

    def __init__(self):
        self.training = False
        self.attn_tp_group = SimpleNamespace(size=lambda: 1)

        self.token_dispatcher = SimpleNamespace(combine_postprocess=lambda x: x)
        self.shared_expert_overlap = False
        self.fwd_execution_map = {"route", "expert_compute", "postprocess"}
        self.moe_layer_recompute = False
        self._accuracy_shared_input = None
        self.order = []

    def shared_experts_compute(self, x):
        self.order.append("shared")
        return x * 2

    def route(self, x, padding_mask):
        self.order.append("route")
        return x, x

    def preprocess(self, x, probs, routing_map):
        self.order.append("preprocess")
        return x, probs

    def dispatch(self, x, probs):
        self.order.append("dispatch")
        return x, probs

    def routed_experts_compute(self, x, probs):
        self.order.append("routed")
        return x * 3, None

    def combine(self, x):
        self.order.append("combine")
        return x

    postprocess = moe_postprocess
    forward = moe_forward


class TestAccuracyMigration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        torch.cuda.set_device(0)

    def setUp(self):
        torch.cuda.set_device(0)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")

    def test_token_dispatch_combine_flags(self):
        flex = _Flex()
        x = torch.ones(2, 4, device="cuda")
        _UAC["on"] = True
        flex.token_dispatch(x, async_finish=True, allocate_on_comm_stream=True)
        flex.token_combine(x, async_finish=True, allocate_on_comm_stream=True)
        self.assertEqual(flex._comm_manager.calls[0][1:], (False, False))
        self.assertEqual(flex._comm_manager.calls[1][1:], (False, False))
        flex._comm_manager.calls.clear()
        _UAC["on"] = False
        flex.token_dispatch(x, async_finish=True, allocate_on_comm_stream=False)
        flex.token_combine(x, async_finish=False, allocate_on_comm_stream=True)
        self.assertEqual(flex._comm_manager.calls[0][1:], (True, False))
        self.assertEqual(flex._comm_manager.calls[1][1:], (False, True))

    def test_forward_shared_order_and_value(self):
        layer = _MoE()
        x = torch.ones(2, 3, 4, device="cuda", requires_grad=True)
        _UAC["on"] = True
        out, _ = layer.forward(x)
        self.assertEqual(
            layer.order, ["route", "preprocess", "dispatch", "routed", "combine", "shared"]
        )
        self.assertTrue(torch.equal(out, 5 * x.detach()))
        out.sum().backward()
        self.assertTrue(torch.equal(x.grad, torch.full_like(x, 5)))
        self.assertIsNone(getattr(layer, "_accuracy_shared_input", None))
        layer.order.clear()
        x2 = torch.ones(2, 3, 4, device="cuda", requires_grad=True)
        out2, _ = layer.forward(x2)
        self.assertTrue(torch.equal(out2, 5 * x2.detach()))
        self.assertIsNone(getattr(layer, "_accuracy_shared_input", None))
        layer = _MoE()
        _UAC["on"] = False
        x3 = torch.ones(2, 3, 4, device="cuda", requires_grad=True)
        out3, _ = layer.forward(x3)
        self.assertEqual(
            layer.order, ["shared", "route", "preprocess", "dispatch", "routed", "combine"]
        )
        self.assertTrue(torch.equal(out3, 5 * x3.detach()))

    def test_postprocess_mixed_dtype(self):
        layer = _MoE()
        routed = torch.ones(2, 4, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        shared = torch.ones(2, 4, device="cuda", dtype=torch.float32, requires_grad=True)
        _UAC["on"] = True
        out = layer.postprocess(routed, shared)
        self.assertEqual(out.dtype, torch.bfloat16)
        out.float().sum().backward()
        self.assertIsNotNone(routed.grad)
        self.assertIsNotNone(shared.grad)
        routed2 = torch.ones(2, 4, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        shared2 = torch.ones(2, 4, device="cuda", dtype=torch.float32, requires_grad=True)
        _UAC["on"] = False
        out2 = layer.postprocess(routed2, shared2)
        self.assertEqual(out2.dtype, torch.float32)

    def test_topk_routing_output_dtype_and_gradients(self):
        _UAC["on"] = True
        logits = torch.tensor(
            [[1.0, 2.0, 0.5, 3.0], [0.2, 4.0, 1.5, 0.8]], device="cuda", dtype=torch.bfloat16
        )
        logits = logits.clone().requires_grad_(True)
        for score in ("sigmoid", "sqrtsoftplus"):
            probs, _idx = topk_routing_with_score_function(
                logits,
                topk=2,
                score_function=score,
                dense_output=True,
                fused=False,
                router_replay=None,
            )
            self.assertEqual(probs.dtype, logits.dtype)
            self.assertTrue(
                torch.allclose(
                    probs.float().sum(dim=-1), torch.ones(probs.size(0), device="cuda"), atol=0.01
                )
            )
            probs.sum().backward()
            self.assertTrue(torch.isfinite(logits.grad).all())
            logits.grad = None


if __name__ == "__main__":
    unittest.main()
