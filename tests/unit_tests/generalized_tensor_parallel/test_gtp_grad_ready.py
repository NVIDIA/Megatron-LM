# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""GTP readiness must not consume autograd placeholders left by warmup or repeated use."""

import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from megatron.core.distributed import DistributedDataParallel
from megatron.core.tensor_parallel.gtp_api import HAVE_GTP

if not HAVE_GTP:
    pytest.skip(
        "GTP requires TransformerEngine with distributed-weight support", allow_module_level=True
    )

from megatron.core.tensor_parallel.generalized_tensor_parallelism import GTPShardedParam


def _make_hook(param):
    """Use the real DDP callback with a mock bucket to observe readiness notifications."""
    bucket = Mock()
    ddp = SimpleNamespace(
        param_to_bucket_group={param: bucket},
        ddp_config=SimpleNamespace(overlap_grad_reduce=True),
        force_all_reduce=False,
    )
    return DistributedDataParallel._make_backward_post_hook(ddp, param), bucket


@pytest.mark.parametrize(
    "gtp,already_added,zero_out,expected",
    [
        (True, True, True, 3),
        (True, True, False, 3),
        (False, True, True, 5),
        (False, True, False, 3),
        (False, False, False, 5),
    ],
)
def test_grad_ready_accumulation(gtp, already_added, zero_out, expected) -> None:
    """GTP publishes its real gradient; ordinary DDP still consumes autograd gradients."""
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    param_type = GTPShardedParam if gtp else torch.nn.Parameter
    param = param_type(torch.ones(4, device="cuda"))
    param.main_grad = torch.full_like(param, 3)
    param.grad = torch.full_like(param, 2)
    param.is_gtp_weight_remat = gtp
    param.grad_added_to_main_grad = already_added
    param.zero_out_wgrad = zero_out
    hook, bucket = _make_hook(param)
    if gtp:
        param.register_grad_accum_hook(None, hook)
        GTPShardedParam._handle_megatron_grad_accum(param)
    else:
        hook()
    torch.testing.assert_close(param.main_grad, torch.full_like(param, expected))
    assert param.grad is None
    bucket.register_grad_ready.assert_called_once_with(param, False)


class _FusedGradient(torch.autograd.Function):
    """Model GTP's real-gradient write and dummy return for a repeatedly used leaf."""

    @staticmethod
    def forward(ctx, weight, scale):
        ctx.weight = weight
        ctx.save_for_backward(scale)
        return weight * scale

    @staticmethod
    def backward(ctx, grad_output):
        (scale,) = ctx.saved_tensors
        ctx.weight.main_grad.add_(grad_output * scale)
        return GTPShardedParam._handle_megatron_grad_accum(ctx.weight), None


def test_repeated_gtp_dummy_autograd_capture() -> None:
    """Repeated backward consumes preserve real gradients after clearing stale dummies."""
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        param = GTPShardedParam(torch.ones(1024, device="cuda"))
        param.main_grad = torch.zeros_like(param)
        param.grad_added_to_main_grad = False
        param.zero_out_wgrad = True
        scale = torch.ones_like(param)
        hook, bucket = _make_hook(param)
        grad_acc = param.expand_as(param).grad_fn.next_functions[0][0]
        param.register_grad_accum_hook(grad_acc, hook)

        def step():
            param.main_grad.zero_()
            for _ in range(3):
                _FusedGradient.apply(param, scale).sum().backward()

        for _ in range(3):
            step()
        torch.cuda.synchronize()
        param.grad = torch.full_like(param, 123)
        bucket.reset_mock()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            step()
        torch.cuda.synchronize()
        torch.testing.assert_close(param.main_grad, scale * 3)
        assert bucket.register_grad_ready.call_count == 3
        torch.cuda.empty_cache()
        for index in range(20):
            scale.fill_(index + 1)
            graph.replay()
            torch.testing.assert_close(param.main_grad, scale * 3)
    torch.cuda.current_stream().wait_stream(stream)


def test_gtp_capture_does_not_retain_warmup_dummy() -> None:
    """A captured GTP readiness hook must survive release of its ordinary warmup grad."""
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    param = GTPShardedParam(torch.ones(1024, device="cuda"))
    param.main_grad = torch.zeros_like(param)
    param.grad = torch.full_like(param, 2)
    param.is_gtp_weight_remat = True
    param.grad_added_to_main_grad = True
    param.zero_out_wgrad = True
    real_grad = torch.full_like(param, 3)
    hook, bucket = _make_hook(param)
    param.register_grad_accum_hook(None, hook)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        param.main_grad.add_(real_grad)
        GTPShardedParam._handle_megatron_grad_accum(param)
    assert param.grad is None
    torch.cuda.empty_cache()
    for _ in range(3):
        param.main_grad.zero_()
        graph.replay()
        torch.testing.assert_close(param.main_grad, real_grad)
    bucket.register_grad_ready.assert_called_once_with(param, False)
