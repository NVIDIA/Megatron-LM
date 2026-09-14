"""CUDA unit tests for native C2 TP1 accuracy-compatible migration."""

from __future__ import annotations

import ast
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional
from unittest.mock import patch

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
_UAC = {"on": False}
_TP = {"size": 1}
_CUSTOM_BWD = {"calls": []}


def _use_accuracy_compatible():
    return _UAC["on"]


def _custom_backward(output, grad):
    _CUSTOM_BWD["calls"].append((output, grad))
    output.backward(grad)


class _SentinelApply(torch.autograd.Function):
    last = None

    @staticmethod
    def forward(ctx, *args):
        _SentinelApply.last = args
        inp = args[0]
        return inp.new_zeros(inp.shape[:-1] + (args[1].shape[0],))

    @staticmethod
    def backward(ctx, grad_output):
        return (grad_output,) + (None,) * 8


class _FakeGroup:
    def __init__(self, size):
        self._size = size

    def size(self):
        return self._size

    def rank(self):
        return 0


def _load_named(rel: str, name: str, extra_ns=None, class_name=None):
    src = (ROOT / rel).read_text()
    tree = ast.parse(src)
    body = tree.body
    if class_name:
        body = next(
            node.body
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        )
    target = next(
        node
        for node in body
        if isinstance(node, ast.ClassDef)
        and node.name == name
        or isinstance(node, ast.FunctionDef)
        and node.name == name
    )
    if isinstance(target, ast.FunctionDef):
        target.decorator_list = []
    mod = ast.Module(body=[target], type_ignores=[])
    ast.fix_missing_locations(mod)
    ns = {
        "torch": torch,
        "F": F,
        "Optional": Optional,
        "List": List,
        "os": __import__("os"),
        "warnings": __import__("warnings"),
        "_use_accuracy_compatible": _use_accuracy_compatible,
        "get_tensor_model_parallel_group_if_none": lambda g: g,
        "LinearWithGradAccumulationAndAsyncCommunication": _SentinelApply,
        "parallel_state": SimpleNamespace(get_tensor_model_parallel_world_size=lambda: _TP["size"]),
        "custom_backward": _custom_backward,
        "Variable": torch.autograd.Variable,
    }
    if extra_ns:
        ns.update(extra_ns)
    exec(compile(mod, rel, "exec"), ns)
    return ns[name]


_EmbedFp32MainGrad = _load_named("megatron/core/tensor_parallel/layers.py", "_EmbedFp32MainGrad")
linear_with_grad_accumulation_and_async_allreduce = _load_named(
    "megatron/core/tensor_parallel/layers.py", "linear_with_grad_accumulation_and_async_allreduce"
)
linear_with_grad_accumulation_and_async_allreduce.warned = True
deallocate_output_tensor = _load_named(
    "megatron/core/pipeline_parallel/schedules.py", "deallocate_output_tensor"
)
backward_step = _load_named("megatron/core/pipeline_parallel/schedules.py", "backward_step")


def _ref_embed_fp32_wgrad(weight_bf16, ids, grad_out):
    table = weight_bf16.detach().clone().requires_grad_(True)
    looked = F.embedding(ids, table)
    (gw,) = torch.autograd.grad(looked, table, grad_outputs=grad_out)
    return gw.float()


def _cuda_bf16(values, shape, device):
    return torch.tensor(values, device=device, dtype=torch.bfloat16).reshape(shape)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestEmbedFp32MainGradCuda(unittest.TestCase):
    def test_embedding_configuration_controls_gradient_destination(self):
        forward = _load_named(
            "megatron/core/tensor_parallel/layers.py",
            "forward",
            {"_EmbedFp32MainGrad": _EmbedFp32MainGrad},
            class_name="VocabParallelEmbedding",
        )
        ids = torch.tensor([1, 1, 3], device="cuda")
        instances = []
        for enabled in (False, True):
            weight = torch.ones(4, 8, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            weight.main_grad = torch.zeros_like(weight, dtype=torch.float32)
            instances.append(
                SimpleNamespace(
                    config=SimpleNamespace(dsa_accuracy_compatible=enabled),
                    deterministic_mode=True,
                    tp_group=_FakeGroup(1),
                    weight=weight,
                    reduce_scatter_embeddings=False,
                )
            )
        for instance in instances:
            enabled = instance.config.dsa_accuracy_compatible
            with patch.dict(
                os.environ,
                {
                    "MODEL_REPRO_TWO_FP32_ACCUM": str(int(not enabled)),
                    "USE_ACCURACY_COMPATIBLE": str(int(not enabled)),
                },
            ):
                output = forward(instance, ids)
            output.sum().backward()
            expected = torch.zeros(4, 8, device="cuda", dtype=torch.float32)
            expected[1] = 2
            expected[3] = 1
            if enabled:
                self.assertIsNone(instance.weight.grad)
                torch.testing.assert_close(instance.weight.main_grad, expected, atol=0, rtol=0)
            else:
                torch.testing.assert_close(instance.weight.grad.float(), expected, atol=0, rtol=0)
                self.assertEqual(torch.count_nonzero(instance.weight.main_grad).item(), 0)

    def test_repeated_indices_matches_independent_full_table_autograd(self):
        device = torch.device("cuda")
        vocab, dim = 8, 4
        weight = _cuda_bf16(
            [
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                24,
                25,
                26,
                27,
                28,
                29,
                30,
                31,
                32,
            ],
            (vocab, dim),
            device,
        )
        ids = torch.tensor([[1, 3, 1, 5], [3, 3, 0, 1]], device=device)
        grad_out = _cuda_bf16(list(range(1, 33)), (2, 4, dim), device)
        w = weight.clone().requires_grad_(True)
        w.main_grad = torch.zeros(vocab, dim, device=device, dtype=torch.float32)
        w.grad_added_to_main_grad = False
        out = _EmbedFp32MainGrad.apply(w, ids)
        self.assertEqual(out.dtype, torch.bfloat16)
        torch.testing.assert_close(out, w[ids], atol=0, rtol=0)
        out.backward(grad_out)
        ref = _ref_embed_fp32_wgrad(weight, ids, grad_out)
        torch.testing.assert_close(w.main_grad, ref, atol=0, rtol=0)
        self.assertIsNone(w.grad)
        self.assertTrue(w.grad_added_to_main_grad)
        unused = [i for i in range(vocab) if i not in set(ids.reshape(-1).tolist())]
        self.assertTrue((w.main_grad[unused] == 0).all())

    def test_main_grad_accumulates_two_backwards_unused_rows_stay_zero(self):
        device = torch.device("cuda")
        vocab, dim = 6, 3
        weight = _cuda_bf16(list(range(1, 19)), (vocab, dim), device)
        ids_a = torch.tensor([2, 2, 4], device=device)
        ids_b = torch.tensor([4, 1, 2], device=device)
        go_a = _cuda_bf16([1, 2, 3, 4, 5, 6, 7, 8, 9], (3, dim), device)
        go_b = _cuda_bf16([2, 1, 0, 1, 2, 3, 4, 5, 6], (3, dim), device)
        w = weight.clone().requires_grad_(True)
        w.main_grad = torch.zeros(vocab, dim, device=device, dtype=torch.float32)
        w.grad_added_to_main_grad = False
        _EmbedFp32MainGrad.apply(w, ids_a).backward(go_a)
        first = w.main_grad.clone()
        self.assertTrue(w.grad_added_to_main_grad)
        _EmbedFp32MainGrad.apply(w, ids_b).backward(go_b)
        ref = _ref_embed_fp32_wgrad(weight, ids_a, go_a) + _ref_embed_fp32_wgrad(
            weight, ids_b, go_b
        )
        torch.testing.assert_close(w.main_grad, ref, atol=0, rtol=0)
        self.assertFalse(torch.equal(first, w.main_grad))
        self.assertTrue((w.main_grad[0] == 0).all())
        self.assertTrue((w.main_grad[3] == 0).all())
        self.assertTrue((w.main_grad[5] == 0).all())
        self.assertIsNone(w.grad)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestLinearTp1Native(unittest.TestCase):
    def setUp(self):
        _UAC["on"] = True
        _SentinelApply.last = None

    def tearDown(self):
        _UAC["on"] = False

    def test_tp1_forward_dgrad_wgrad_bias_matches_f_linear(self):
        device = torch.device("cuda")
        x = _cuda_bf16(
            [1, 2, 0, -1, 1, 0, 2, 1, -2, 0, 1, 1, 2, 0, 1], (5, 3), device
        ).requires_grad_(True)
        w = _cuda_bf16([1, 0, -1, 0, 1, 1, 1, -1, 0, 0, 1, -1], (4, 3), device).requires_grad_(True)
        b = _cuda_bf16([1, -1, 0, 2], (4,), device).requires_grad_(True)
        out = linear_with_grad_accumulation_and_async_allreduce(
            x, w, b, False, False, False, None, 0, _FakeGroup(1), dsa_accuracy_compatible=True
        )
        xref = x.detach().clone().requires_grad_(True)
        wref = w.detach().clone().requires_grad_(True)
        bref = b.detach().clone().requires_grad_(True)
        ref = F.linear(xref, wref, bref)
        torch.testing.assert_close(out, ref, atol=0, rtol=0)
        self.assertIsNone(_SentinelApply.last)
        go = _cuda_bf16(
            [1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1], (5, 4), device
        )
        out.backward(go)
        F.linear(xref, wref, bref).backward(go)
        torch.testing.assert_close(x.grad, xref.grad, atol=0, rtol=0)
        torch.testing.assert_close(w.grad, wref.grad, atol=0, rtol=0)
        torch.testing.assert_close(b.grad, bref.grad, atol=0, rtol=0)

    def test_global_accuracy_without_dsa_keeps_native_custom_function(self):
        _UAC["on"] = True
        device = torch.device("cuda")
        x = torch.ones(2, 3, device=device, dtype=torch.bfloat16, requires_grad=True)
        w = torch.ones(4, 3, device=device, dtype=torch.bfloat16)
        linear_with_grad_accumulation_and_async_allreduce(
            x, w, None, False, False, False, None, 0, _FakeGroup(1)
        )
        self.assertIsNotNone(_SentinelApply.last)
        self.assertTrue(torch.equal(_SentinelApply.last[0], x))

    def test_tp2_or_allreduce_skips_tp1_matmul_path(self):
        device = torch.device("cuda")
        x = torch.ones(2, 3, device=device, dtype=torch.bfloat16, requires_grad=True)
        w = torch.ones(4, 3, device=device, dtype=torch.bfloat16)
        linear_with_grad_accumulation_and_async_allreduce(
            x, w, None, False, False, False, None, 0, _FakeGroup(2), dsa_accuracy_compatible=True
        )
        self.assertIsNotNone(_SentinelApply.last)
        _SentinelApply.last = None
        linear_with_grad_accumulation_and_async_allreduce(
            x, w, None, False, True, False, None, 0, _FakeGroup(1), dsa_accuracy_compatible=True
        )
        self.assertIsNotNone(_SentinelApply.last)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestPipelineHelpersTp1(unittest.TestCase):
    def tearDown(self):
        _UAC["on"] = False
        _TP["size"] = 1
        _CUSTOM_BWD["calls"] = []

    def test_deallocate_uac_tp1_preserves_storage(self):
        _UAC["on"] = True
        _TP["size"] = 1
        t = torch.arange(4.0, device="cuda", requires_grad=True)
        data_before = t.data.clone()
        deallocate_output_tensor(
            t, True, SimpleNamespace(dsa_accuracy_compatible=True, tensor_model_parallel_size=1)
        )
        torch.testing.assert_close(t.data, data_before, atol=0, rtol=0)
        self.assertEqual(tuple(t.shape), (4,))

    def test_deallocate_off_or_tp2_still_frees(self):
        _UAC["on"] = False
        _TP["size"] = 1
        t = torch.arange(4.0, device="cuda")
        deallocate_output_tensor(t, True)
        self.assertEqual(tuple(t.shape), (1,))
        _UAC["on"] = True
        _TP["size"] = 2
        t2 = torch.arange(4.0, device="cuda")
        deallocate_output_tensor(t2, True)
        self.assertEqual(tuple(t2.shape), (1,))

    def test_backward_step_uac_tp1_uses_autograd_not_custom(self):
        _UAC["on"] = True
        _CUSTOM_BWD["calls"] = []
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device="cuda", requires_grad=True)
        y = x * 3
        go = torch.ones_like(y)
        cfg = SimpleNamespace(
            timers=None,
            grad_scale_func=None,
            deallocate_pipeline_outputs=True,
            tensor_model_parallel_size=1,
            dsa_accuracy_compatible=True,
        )
        gin = backward_step(x, y, go, cfg)
        torch.testing.assert_close(gin, go * 3, atol=0, rtol=0)
        self.assertEqual(_CUSTOM_BWD["calls"], [])

    def test_backward_step_off_or_tp2_uses_custom_backward(self):
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="cuda", requires_grad=True)
        y = x * 2
        go = torch.ones_like(y)
        cfg = SimpleNamespace(
            timers=None,
            grad_scale_func=None,
            deallocate_pipeline_outputs=True,
            tensor_model_parallel_size=1,
            dsa_accuracy_compatible=False,
        )
        _UAC["on"] = False
        _CUSTOM_BWD["calls"] = []
        gin = backward_step(x, y, go, cfg)
        self.assertEqual(len(_CUSTOM_BWD["calls"]), 1)
        torch.testing.assert_close(gin, go * 2, atol=0, rtol=0)

        x2 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="cuda", requires_grad=True)
        y2 = x2 * 2
        go2 = torch.ones_like(y2)
        cfg.dsa_accuracy_compatible = True
        cfg.tensor_model_parallel_size = 2
        _UAC["on"] = True
        _CUSTOM_BWD["calls"] = []
        gin2 = backward_step(x2, y2, go2, cfg)
        self.assertEqual(len(_CUSTOM_BWD["calls"]), 1)
        torch.testing.assert_close(gin2, go2 * 2, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
