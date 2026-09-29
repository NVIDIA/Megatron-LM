# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Typed boundary protocol and autograd checks independent of model semantics."""

import gc
import os
import weakref
from dataclasses import dataclass
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from megatron.core.pipeline_parallel.pipeline_payload import (
    PipelineGradientMessage,
    PipelinePayload,
    PipelinePayloadPlan,
    PipelinePayloadSpec,
    PipelineTensorSpec,
    backward_pipeline_payload,
)
from megatron.core.pipeline_parallel.schedules import backward_step, deallocate_output_tensor
from megatron.core.pipeline_parallel.typed_p2p_communication import (
    TypedP2PCommunicator,
    _decode_header,
    _pack_header,
    _unpack_header,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.experimental_attention_variant.dsa import DSAIndexerLossAutoScaler
from megatron.core.transformer.state_boundary import TensorField
from megatron.core.transformer.transformer_config import TransformerConfig


def _spec(name, shape, dtype, requires_grad, layout="strided", present=True):
    return PipelineTensorSpec(name, TensorField(name, shape, dtype, layout, requires_grad, present))


@dataclass(frozen=True)
class _Payload(PipelinePayload):
    tensors: tuple[torch.Tensor, ...]

    @property
    def metadata(self):
        return self.tensors[0].shape[0], self.tensors[1].shape[0]

    @property
    def tensor_specs(self):
        return tuple(
            _spec(str(i), tuple(t.shape), t.dtype, t.requires_grad)
            for i, t in enumerate(self.tensors)
        )


@pytest.mark.parametrize("release", [False, True])
def test_joint_backward_sums_local_and_relay_contributions(release):
    hidden = torch.tensor([1.0, 2.0], requires_grad=True)
    shared = torch.tensor([3.0, 4.0], requires_grad=True)
    indexer = torch.tensor([5.0, 6.0], requires_grad=True)
    indices = torch.tensor([0, 1], dtype=torch.int32)
    incoming = _Payload((hidden, shared, indexer, indices))
    weight = torch.tensor(2.0, requires_grad=True)
    outgoing = _Payload((hidden * weight + shared.square(), shared, indexer, indices))
    deallocate_output_tensor(outgoing, release)
    assert bool(outgoing.tensors) is not release
    assert shared.shape == indexer.shape == hidden.shape == (2,)
    gradients = backward_step(
        incoming,
        outgoing,
        (torch.tensor([7.0, 8.0]), torch.tensor([9.0, 10.0]), None, None),
        SimpleNamespace(
            timers=None, grad_scale_func=lambda loss: loss * 100, deallocate_pipeline_outputs=True
        ),
    )
    torch.testing.assert_close(gradients[0], torch.tensor([14.0, 16.0]))
    torch.testing.assert_close(gradients[1], torch.tensor([51.0, 74.0]))
    assert gradients[2] is None
    assert gradients[3] is None
    torch.testing.assert_close(weight.grad, torch.tensor(23.0))


def test_payload_release_preserves_saved_views_hooks_and_empty_gradients():
    """Only unused output ownership is dropped; local backward still owns its saved values."""

    class Copy(torch.autograd.Function):
        @staticmethod
        def forward(ctx, tensor):
            return tensor.clone()

        @staticmethod
        def backward(ctx, grad):
            return grad

    x = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    hidden = Copy.apply(x * 2)
    saved = x.sigmoid()
    empty = x[:0]
    hooks = []
    hidden.register_hook(lambda grad: hooks.append(grad.clone()))
    hidden_ref = weakref.ref(hidden)
    payload = _Payload((hidden, saved.view(2, 2), empty))
    saved_value = saved.detach().clone()
    del hidden, saved, empty
    payload.release_output()
    payload.release_output()  # The schedule may retire the same output more than once.
    gc.collect()
    assert hidden_ref() is not None  # Original roots remain alive until backward completes.
    assert not payload.tensors
    backward_step(
        None,
        payload,
        (torch.ones(4), torch.ones(2, 2), None),
        SimpleNamespace(timers=None, grad_scale_func=None, deallocate_pipeline_outputs=True),
    )
    torch.testing.assert_close(x.grad, 2 + saved_value * (1 - saved_value))
    assert len(hooks) == 1
    torch.testing.assert_close(hooks[0], torch.ones(4))
    assert payload._backward_state.outputs == ()
    assert hidden_ref() is None


def test_terminal_payload_backward_scales_loss_once():
    hidden = torch.tensor([2.0, 3.0], requires_grad=True)
    state = torch.tensor([4.0, 5.0], requires_grad=True)
    incoming = _Payload((hidden, state))
    grads = backward_step(
        incoming,
        (hidden * state).sum(),
        None,
        SimpleNamespace(
            timers=None, grad_scale_func=lambda loss: loss * 3, deallocate_pipeline_outputs=True
        ),
    )
    torch.testing.assert_close(grads[0], torch.tensor([12.0, 15.0]))
    torch.testing.assert_close(grads[1], torch.tensor([6.0, 9.0]))


def test_header_preserves_mixed_dtypes_empty_shapes_and_host_metadata():
    specs = (
        _spec("hidden", (9, 1, 128), torch.bfloat16, True),
        _spec("mix", (9, 1, 4), torch.float32, True),
        _spec("empty", (0, 1, 16), torch.bfloat16, True),
        _spec("ids", (9, 3), torch.int32, False),
        _spec("prefixes", (5,), torch.int64, False),
        _spec("scalar", (), torch.float64, False),
    )
    values = _pack_header(specs, (12345, 4), 7)
    decoded, metadata = _unpack_header(values, 7)
    assert _pack_header(decoded, metadata, 7) == values
    assert metadata == (12345, 4)
    with pytest.raises(ValueError, match="microbatch mismatch"):
        _unpack_header(values, 8)
    with pytest.raises(ValueError, match="chunk mismatch"):
        _unpack_header(values, 7, chunk_id=1)


def test_generic_schema_has_no_csa2_field_dimension_or_metadata_limit():
    specs = tuple(
        _spec(
            f"feature/value:L{i}",
            (2, 1, 3, 1),
            torch.float32,
            i % 3 == 0,
            layout="bshd",
            present=i % 2 == 0,
        )
        for i in range(12)
    )
    descriptor = PipelinePayloadSpec(specs, (0, 1, 2, 3, 4, 5), "decoder/cut:20")
    values = _pack_header(specs, descriptor.metadata, 7, 2, boundary_id=descriptor.boundary_id)
    assert _decode_header(values, 7, 2) == descriptor
    assert descriptor.schema.present_spec_indices == (0, 2, 4, 6, 8, 10)
    assert descriptor.schema.grad_tensor_indices == (0, 3)
    message = PipelineGradientMessage(
        (torch.zeros(1), None, torch.ones(1)),
        torch.tensor([7, 2, descriptor.fingerprint, 1, 0]),
        (7, 2, descriptor.fingerprint),
        (0, 2),
    )
    grads = message.resolve()
    assert grads[0] is not None and grads[1:] == (None, None)
    message.control[0] += 1
    with pytest.raises(ValueError, match="microbatch"):
        message.resolve()


def test_inactive_boundary_backward_with_mcore_ddp(monkeypatch):
    """Exercise real DDP buffers/reduction with changing activity on two DP ranks.

    CPU runs replace only CUDA allocation/stream access; autograd, MCore hooks,
    buffer lifecycle and Gloo collectives execute normally. Overlapping DDP's
    fixed parameter-readiness counts are outside this dynamic-activity contract.
    """
    if int(os.environ.get("WORLD_SIZE", "1")) != 2:
        pytest.skip("Requires two distributed ranks")
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig

    if not dist.is_initialized():
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
    rank = dist.get_rank()
    singletons = [dist.new_group([i]) for i in range(2)]
    groups = ProcessGroupCollection()
    groups.dp = groups.dp_cp = groups.expt_dp = dist.group.WORLD
    groups.tp = groups.pp = groups.ep = singletons[rank]
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        device = torch.device("cuda", torch.cuda.current_device())
    else:
        device = torch.device("cpu")
        monkeypatch.setattr(torch.cuda, "current_device", lambda: device)
        monkeypatch.setattr(torch.cuda, "current_stream", lambda: None)
    monkeypatch.setattr(DSAIndexerLossAutoScaler, "main_loss_backward_scale", None)

    class Stage(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(2.0, device=device))
            self.auxiliary = torch.nn.Parameter(torch.tensor(3.0, device=device))

        def forward(self, x):
            return _Payload(
                (
                    DSAIndexerLossAutoScaler.apply(x.square(), self.auxiliary.square()),
                    x * self.weight,
                )
            )

    config = TransformerConfig(num_layers=1, hidden_size=4, num_attention_heads=1)
    module = Stage()
    ddp = DistributedDataParallel(
        config,
        DistributedDataParallelConfig(
            overlap_grad_reduce=False, use_distributed_optimizer=False, check_for_nan_in_grad=False
        ),
        module,
        pg_collection=groups,
    )
    try:
        for activity in ("kv", "inactive", "zero_hidden", "different_ranks", "inactive"):
            ddp.zero_grad_buffer()
            x = torch.full((3,), rank + 1.0, device=device, requires_grad=True)
            output = ddp(x)
            active = activity != "inactive" and (activity != "different_ranks" or rank == 0)
            backward_pipeline_payload(
                None,
                output,
                (
                    torch.zeros_like(x) if activity == "zero_hidden" else None,
                    torch.ones_like(x) if active else None,
                ),
            )
            ddp.finish_grad_sync()
            expected_weight = 1.5 if activity == "different_ranks" else (4.5 if active else 0.0)
            torch.testing.assert_close(
                module.weight.main_grad, torch.tensor(expected_weight, device=device)
            )
            torch.testing.assert_close(
                module.auxiliary.main_grad,
                torch.tensor(6.0 if activity == "zero_hidden" else 0.0, device=device),
            )
            assert (x.grad is not None) == active
    finally:
        dist.destroy_process_group(singletons[rank])


def _make_payload(tensors, descriptor):
    payload = _Payload(tensors)
    assert payload.descriptor == descriptor
    return payload


@pytest.fixture
def delayed_transport(monkeypatch):
    """Control request completion without letting the fake Work retain wire buffers."""
    group = SimpleNamespace(size=lambda: 2, rank=lambda: 0)
    monkeypatch.setattr(dist, "get_global_rank", lambda group, rank: rank)
    config = TransformerConfig(
        num_layers=4,
        hidden_size=8,
        num_attention_heads=1,
        pipeline_model_parallel_size=2,
        pipeline_dtype=torch.float32,
        virtual_pipeline_model_parallel_size=2,
        overlap_p2p_comm=True,
        batch_p2p_comm=False,
    )
    communicator = TypedP2PCommunicator(
        group, config, [_make_payload, _make_payload], device=torch.device("cpu")
    )
    requests = []

    class DelayedWork:
        def __init__(self, name, tensor):
            self.name = name
            self.buffer = weakref.ref(tensor)
            self.wait_count = 0
            self.received_value = None

        def wait(self):
            assert self.buffer() is not None, "Communication buffer was released before wait"
            self.wait_count += 1
            if self.received_value is not None:
                with torch.no_grad():
                    self.buffer().copy_(self.received_value)

    def post(**kwargs):
        result = {}
        for key, tensor in kwargs.items():
            if key.startswith("tensor_") and tensor is not None:
                name = key.removeprefix("tensor_")
                request = DelayedWork(name, tensor)
                requests.append(request)
                result[name] = request
        return result

    monkeypatch.setattr("megatron.core.pipeline_parallel.typed_p2p_communication._p2p_ops", post)
    return communicator, requests, post


@pytest.mark.parametrize("mismatch", ["shape", "dtype", "grad", "metadata"])
@pytest.mark.parametrize("forward_only", [False, True])
def test_prepared_sender_rejects_stale_plan_before_posting(
    delayed_transport, mismatch, forward_only
):
    communicator, requests, _ = delayed_transport
    communicator.forward_only = forward_only
    source = _Payload((torch.ones(3, requires_grad=True), torch.empty(0)))
    spec = source.tensor_specs[0]
    replacement = _spec(
        spec.name,
        (5,) if mismatch == "shape" else spec.shape,
        torch.bfloat16 if mismatch == "dtype" else spec.dtype,
        False if mismatch == "grad" else spec.requires_grad,
    )
    descriptor = PipelinePayloadSpec(
        (replacement, source.tensor_specs[1]), (5, 0) if mismatch == "metadata" else source.metadata
    )
    communicator.payload_plans = [PipelinePayloadPlan((None,), (descriptor,))] * 2
    with pytest.raises(ValueError, match="disagrees"):
        communicator.send_forward_recv_forward(source, False, None, overlap_p2p_comm=True)
    assert not requests


class _Timer:
    def __init__(self):
        self.started = self.stopped = 0

    def start(self, **kwargs):
        assert self.started == self.stopped
        self.started += 1

    def stop(self):
        assert self.started == self.stopped + 1
        self.stopped += 1


class _Timers(dict):
    def __call__(self, name, **kwargs):
        if name not in self:
            self[name] = _Timer()
        return self[name]


@pytest.fixture
def transport_groups():
    """Use NCCL on GPUs; Gloo also exercises the real protocol on CPU-only hosts."""
    size = int(os.environ.get("WORLD_SIZE", "1"))
    if size < 2:
        pytest.skip("Launch with torch.distributed.run and at least two processes")
    rank = int(os.environ["RANK"])
    device = torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        device = torch.device("cuda", torch.cuda.current_device())
    if not dist.is_initialized():
        dist.init_process_group(
            "nccl" if device.type == "cuda" else "gloo", timeout=timedelta(seconds=60)
        )
    singleton_groups = [dist.new_group([i]) for i in range(size)]
    groups = ProcessGroupCollection()
    groups.pp = dist.group.WORLD
    groups.tp = groups.cp = singleton_groups[rank]
    dist.barrier()
    yield groups, device
    dist.barrier()
    dist.destroy_process_group(singleton_groups[rank])
