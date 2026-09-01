# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
import sys
from pathlib import Path

import torch

VERL_EXAMPLE_ROOT = Path(__file__).resolve().parents[3] / "examples" / "verl"
if str(VERL_EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(VERL_EXAMPLE_ROOT))

from verl_mlite.weight_sync import (
    PPBroadcastContext,
    PPBroadcastWeightStream,
    PPBucketPlanCache,
    _LocalBucketPacker,
    _PPBucketProducer,
    install_pp_bucketed_sender,
)


def test_mlite_runtime_local_stage_export_preserves_non_pp_groups() -> None:
    from megatron.lite.primitive.parallel.state import ParallelState
    from megatron.lite.runtime.backends.mlite.runtime import MegatronLiteRuntime
    from megatron.lite.runtime.contracts.handle import ModelHandle

    captured = {}

    class Protocol:
        @staticmethod
        def export_hf_weights(_chunks, _model_cfg, ps, **kwargs):
            captured["ps"] = ps
            captured["kwargs"] = kwargs
            yield "model.layers.2.weight", torch.ones(1)

    tp_group = object()
    ep_group = object()
    ps = ParallelState(
        tp_group=tp_group,
        ep_group=ep_group,
        tp_size=2,
        ep_size=4,
        pp_group=object(),
        pp_cpu_group=object(),
        pp_size=4,
        pp_rank=2,
        pp_global_ranks=[0, 1, 2, 3],
    )
    handle = ModelHandle(
        model=torch.nn.Linear(1, 1),
        parallel_state=ps,
        _extras={"protocol": Protocol(), "model_cfg": object()},
    )

    runtime = object.__new__(MegatronLiteRuntime)
    result = list(
        runtime.export_weights(
            handle, local_pipeline_stage=True, export_dtype="bfloat16"
        )
    )

    assert result[0][0] == "model.layers.2.weight"
    local_ps = captured["ps"]
    assert local_ps.pp_size == 1
    assert local_ps.pp_group is None
    assert local_ps.pp_global_ranks is None
    assert local_ps.tp_group is tp_group
    assert local_ps.ep_group is ep_group
    assert local_ps.tp_size == 2
    assert local_ps.ep_size == 4
    assert captured["kwargs"] == {"export_dtype": "bfloat16"}


def test_local_bucket_packer_preserves_layer_clusters_and_alignment() -> None:
    weights = iter(
        [
            ("model.layers.0.fp8", torch.arange(3, dtype=torch.uint8)),
            ("model.layers.0.scale", torch.arange(2, dtype=torch.float32)),
            ("model.layers.1.weight", torch.arange(4, dtype=torch.bfloat16)),
        ]
    )
    packer = _LocalBucketPacker(weights, bucket_size=128)
    staging = torch.empty(128, dtype=torch.uint8)

    first = packer.next_bucket(staging)
    second = packer.next_bucket(staging)

    assert list(first[0]) == ["model.layers.0.fp8", "model.layers.0.scale"]
    assert first[0]["model.layers.0.fp8"]["offset"] == 0
    assert first[0]["model.layers.0.scale"]["offset"] == 8
    assert list(second[0]) == ["model.layers.1.weight"]
    assert packer.next_bucket(staging) is None


def test_pp_bucket_plan_is_learned_once_and_reused(monkeypatch) -> None:
    object_broadcasts = []
    tensor_broadcasts = []

    def broadcast_object_list(header, **kwargs):
        object_broadcasts.append((header[0], kwargs))

    def broadcast(tensor, **kwargs):
        tensor_broadcasts.append((tensor.numel(), kwargs))

    monkeypatch.setattr(
        torch.distributed, "broadcast_object_list", broadcast_object_list
    )
    monkeypatch.setattr(torch.distributed, "broadcast", broadcast)

    context = PPBroadcastContext(
        rank=0, size=1, global_ranks=(0,), group=object(), cpu_group=object()
    )
    cache = PPBucketPlanCache()

    def make_stream():
        return PPBroadcastWeightStream(
            [
                ("model.layers.0.weight", torch.ones(4)),
                ("model.layers.1.weight", torch.ones(2)),
            ],
            context,
            cache,
        )

    staging = torch.empty(128, dtype=torch.uint8)
    first = _PPBucketProducer(make_stream(), 128)
    assert first.next_bucket(staging)[0] == "bucket"
    assert first.next_bucket(staging)[0] == "bucket"
    assert first.next_bucket(staging)[0] == "eof"
    assert cache.ready is True
    assert len(cache.entries) == 2
    first_header_count = len(object_broadcasts)

    second = _PPBucketProducer(make_stream(), 128)
    assert second.next_bucket(staging)[-1] is False
    assert second.next_bucket(staging)[-1] is True
    assert len(object_broadcasts) == first_header_count
    assert len(tensor_broadcasts) == 4


def _run_distributed_smoke() -> None:
    """Executable with torchrun; validates real two-rank Gloo collectives."""
    torch.distributed.init_process_group("gloo")
    try:
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        assert world_size == 2
        context = PPBroadcastContext(
            rank=rank,
            size=world_size,
            global_ranks=tuple(range(world_size)),
            group=torch.distributed.group.WORLD,
            cpu_group=torch.distributed.group.WORLD,
        )
        cache = PPBucketPlanCache()

        def make_stream():
            return PPBroadcastWeightStream(
                [
                    (
                        f"model.layers.{rank}.weight",
                        torch.full((4,), rank + 1, dtype=torch.float32),
                    )
                ],
                context,
                cache,
            )

        for sync_index in range(2):
            producer = _PPBucketProducer(make_stream(), bucket_size=128)
            staging = torch.empty(128, dtype=torch.uint8)
            received = []
            while True:
                kind, metadata, _used_bytes, _ready, is_last = producer.next_bucket(
                    staging
                )
                if kind == "eof":
                    break
                name, meta = next(iter(metadata.items()))
                size = meta["shape"].numel() * meta["dtype"].itemsize
                value = staging[meta["offset"] : meta["offset"] + size].view(
                    meta["dtype"]
                )
                received.append((name, value.clone()))
                if is_last:
                    break
            assert [name for name, _ in received] == [
                "model.layers.0.weight",
                "model.layers.1.weight",
            ]
            torch.testing.assert_close(received[0][1], torch.ones(4))
            torch.testing.assert_close(received[1][1], torch.full((4,), 2.0))
            if sync_index == 0:
                assert cache.ready and len(cache.entries) == 2
    finally:
        torch.distributed.destroy_process_group()


def _run_sender_smoke() -> None:
    class Socket:
        def __init__(self):
            self.messages = []

        def send_pyobj(self, message):
            self.messages.append(message)

        def recv(self):
            return b""

    class Sender:
        async def async_send_weights(self, _weights):
            raise AssertionError("PP marker stream fell through to the stock sender")

        def __init__(self):
            self.bucket_size = 128
            self.use_shm = True
            self.socket = Socket()
            self.buffer = None

        def _init_socket(self):
            pass

        def _init_buffer(self):
            self.buffer = torch.empty(self.bucket_size, dtype=torch.uint8)

        def _cleanup(self):
            pass

    install_pp_bucketed_sender(Sender)
    context = PPBroadcastContext(
        rank=0, size=1, global_ranks=(0,), group=object(), cpu_group=object()
    )
    stream = PPBroadcastWeightStream(
        [("model.layers.0.weight", torch.ones(4))],
        context,
        PPBucketPlanCache(),
    )
    sender = Sender()
    original_object_broadcast = torch.distributed.broadcast_object_list
    original_broadcast = torch.distributed.broadcast
    torch.distributed.broadcast_object_list = lambda *_args, **_kwargs: None
    torch.distributed.broadcast = lambda *_args, **_kwargs: None
    try:
        asyncio.run(sender.async_send_weights(stream))
    finally:
        torch.distributed.broadcast_object_list = original_object_broadcast
        torch.distributed.broadcast = original_broadcast
    assert [message["is_last"] for message in sender.socket.messages] == [False, True]
    assert list(sender.socket.messages[0]["bucket_meta"]) == ["model.layers.0.weight"]


if __name__ == "__main__":
    test_mlite_runtime_local_stage_export_preserves_non_pp_groups()
    _run_sender_smoke()
    _run_distributed_smoke()
