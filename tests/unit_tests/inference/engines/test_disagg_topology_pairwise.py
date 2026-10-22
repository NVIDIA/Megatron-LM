# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Real-model heterogeneous TP/PP handoff with checkpoint-resharded identical weights."""

import asyncio
import time
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from pathlib import Path
from unittest import mock

import pytest
import torch
import torch.distributed as dist
import zmq

from megatron.core import dist_checkpointing
from megatron.core.inference.disaggregation.utils import transfer_peer_records
from tests.unit_tests.inference.engines.disagg_test_utils import (
    ForwardWitness,
    _enqueue_decode_handoff,
    assert_released,
    decode_peer_meta,
    disagg_config,
    prompt,
    push_handoff,
    real_engine,
    run_to_completion,
    sampling,
)
from tests.unit_tests.test_utilities import Utils


def gather(value, control):
    values = [None] * dist.get_world_size()
    dist.all_gather_object(values, value, group=control)
    return values


def model_config(tp, pp, transformer_impl="transformer_engine", model_provider="gpt"):
    return disagg_config(
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=pp,
        transformer_impl=transformer_impl,
        sequence_parallel=transformer_impl == "inference_optimized",
        model_provider=model_provider,
    )


def load_canonical_model(model, checkpoint_path):
    loaded = dist_checkpointing.load(model.sharded_state_dict(), checkpoint_path)
    model.load_state_dict(loaded)
    current = model.state_dict()
    for name, value in loaded.items():
        if torch.is_tensor(value):
            assert torch.equal(current[name].cpu(), value.cpu()), name


@contextmanager
def implementation_witness(model):
    """Observe real target layers and inference-optimized TP collectives."""
    events = {"layers": [], "gathers": [], "scatters": []}
    layer_types = {
        "MambaMixer",
        "ColumnParallelLinear",
        "RowParallelLinear",
        "InferenceLayerNormColumnParallelLinear",
        "InferenceColumnParallelLinear",
        "InferenceRowParallelLinear",
    }
    with ExitStack() as stack:
        for module in model.modules():
            class_name = type(module).__name__
            if class_name in layer_types:
                original_forward = module.forward

                def forward(*args, _forward=original_forward, _name=class_name, **kwargs):
                    result = _forward(*args, **kwargs)
                    events["layers"].append((_name, tuple(args[0].shape)))
                    return result

                stack.enter_context(mock.patch.object(module, "forward", side_effect=forward))
            for method, key in (("_all_gather", "gathers"), ("_matmul_reduce_scatter", "scatters")):
                if not hasattr(module, method):
                    continue
                original_collective = getattr(module, method)

                def collective(*args, _call=original_collective, _key=key, **kwargs):
                    output = _call(*args, **kwargs)
                    events[_key].append((int(args[0].shape[0]), int(output.shape[0])))
                    return output

                stack.enter_context(mock.patch.object(module, method, side_effect=collective))
        yield events


def canonical_kv(source_shards):
    """Reconstruct global coordinates independently of the production reshard planner."""
    first_meta, first_blocks = source_shards[0]
    shape = list(first_blocks.shape)
    shape[1] = first_meta["num_layers_global"]
    shape[4] = first_meta["num_kv_heads_global"]
    full = torch.empty(shape, dtype=first_blocks.dtype)
    covered = torch.zeros(shape[1], shape[4], dtype=torch.bool)
    for metadata, values in source_shards:
        lo, hi = metadata["layer_start"], metadata["layer_end"]
        heads = metadata["heads_per_partition"]
        head_lo = metadata["tp_rank"] * heads
        assert not covered[lo:hi, head_lo : head_lo + heads].any()
        covered[lo:hi, head_lo : head_lo + heads] = True
        full[:, lo:hi, :, :, head_lo : head_lo + heads] = values
    assert covered.all(), "Source model shards did not cover the global KV layout"
    return full


def canonical_ssm(source_shards):
    """Reconstruct TP-sharded Mamba bands without using the reshard planner."""
    ordered = sorted(source_shards, key=lambda shard: shard[0]["ssm_layout"]["tp_rank"])
    layout = ordered[0][0]["ssm_layout"]
    dims = layout["dims"]
    tp_size = layout["tp_size"]
    inner = dims["nheads"] * dims["headdim"] // tp_size
    group = dims["ngroups"] * dims["d_state"] // tp_size
    conv = [shard[1]["conv"] for shard in ordered]
    bands = ((0, inner), (inner, inner + group), (inner + group, inner + 2 * group))
    return {
        "conv": torch.cat(
            [torch.cat([value[:, lo:hi] for value in conv], dim=1) for lo, hi in bands], dim=1
        ),
        "recurrent": torch.cat([shard[1]["recurrent"] for shard in ordered], dim=1),
    }


def assert_logical_import(engine, pending, full, prompt_length):
    metadata = engine._kv_transfer_agent.export_meta()
    layers = slice(metadata["layer_start"], metadata["layer_end"])
    head_count = metadata["heads_per_partition"]
    head_start = metadata["tp_rank"] * head_count
    heads = slice(head_start, head_start + head_count)
    for index, block in enumerate(pending.local_blocks):
        count = min(16, prompt_length - 16 * index)
        actual = engine.context.memory_buffer[:, :, block, :count].cpu()
        assert torch.equal(actual, full[:, layers, index, :count, heads])


def admit_model_group(engine, failed=False):
    engine._report_completed_kv_imports()
    tracker = engine._handoff_completion_tracker
    if tracker.world_size == 1:
        assert engine._handoff_completion_notifications == {101: (False, True)}
        assert engine._admit_pending_kv_imports() == 1
        return
    # All participants have locally waited for the real transfer before entering
    # this point. Synchronize reports, not a polling CUDA collective.
    dist.barrier(group=engine.pg_collection.mp)
    notifications = []
    if tracker.is_coordinator:
        deadline = time.monotonic() + 10
        while (not notifications or tracker._reports) and time.monotonic() < deadline:
            notifications.extend(tracker.drain_completed())
        assert notifications == [(101, failed, True)]
        assert not tracker._reports
    src_rank = dist.get_process_group_ranks(engine.pg_collection.mp)[0]
    payload = [notifications]
    dist.broadcast_object_list(payload, src=src_rank, group=engine.pg_collection.mp)
    for request_id, failed, source_safe in payload[0]:
        engine._record_handoff_completion_notification(request_id, failed, source_safe)
    assert engine._admit_pending_kv_imports() == int(not failed)
    assert not engine._pending_kv_imports
    assert not engine.waiting_request_ids


@pytest.mark.parametrize(
    "case",
    [
        ("nccl", 2, 1, 1, 1),
        ("nccl", 1, 1, 2, 1),
        ("nccl", 1, 2, 1, 1),
        ("nccl", 2, 2, 1, 2),
        ("nixl", 2, 1, 1, 1),
        ("nixl", 1, 2, 1, 1),
        ("nccl", 2, 1, 2, 1, "local"),
        ("nccl", 2, 1, 2, 1, "inference_optimized"),
        ("nixl", 2, 1, 2, 1, "transformer_engine", "gpt", True),
    ],
    ids=[
        "tp2-to-tp1",
        "tp1-to-tp2",
        "pp2-to-pp1",
        "tp2pp2-to-tp1pp2",
        "nixl-tp2-to-tp1",
        "nixl-pp2-to-pp1",
        "local-tp2",
        "inference-optimized-tp2",
        "native-prepare-failure-tp2",
    ],
)
@torch.inference_mode()
def test_real_model_heterogeneous_handoff(tmp_path, case):
    """Prefill and decode consume one checkpoint, not independent seeded models.

    Building topology replicas is collective across the test world; only a
    disjoint source/destination model-group pair performs the actual transfer.
    Old source buffers remain alive after global model-group bootstrap switches
    to decode topology. Their engines retain explicit original group handles.
    """
    backend, source_tp, source_pp, decode_tp, decode_pp, *implementation = case
    transformer_impl = implementation[0] if implementation else "transformer_engine"
    model_provider = implementation[1] if len(implementation) > 1 else "gpt"
    failure = len(implementation) > 2
    Utils.initialize_model_parallel(source_tp, source_pp)
    control = dist.new_group(backend="gloo")
    rank = dist.get_rank()
    assert dist.get_world_size() == 8, "Heterogeneous model-group selection requires eight ranks"
    checkpoint_path = gather(str(tmp_path / "canonical-weights") if rank == 0 else None, control)[0]
    if rank == 0:
        Path(checkpoint_path).mkdir()
    dist.barrier(group=control)
    tokens = prompt(33)
    try:
        with real_engine(
            model_config(source_tp, source_pp, transformer_impl, model_provider)
        ) as source_engine:
            model = source_engine.controller.inference_wrapped_model.model
            # The checkpoint is the canonical source of truth and its access
            # integrity validation checks complete, non-overlapping weight shards.
            dist_checkpointing.save(model.sharded_state_dict(), checkpoint_path)
            load_canonical_model(model, checkpoint_path)
            source_group = dist.get_process_group_ranks(source_engine.pg_collection.mp)
            source_groups = gather(source_group, control)
            chosen_source = source_groups[0]
            reference = run_to_completion(
                source_engine, source_engine.add_request(101, tokens, sampling())
            )
            expected = list(reference.generated_tokens)
            source_engine.context.config.kv_transport_backend = backend
            source_engine.setup_kv_transfer("prefill")
            with ForwardWitness(source_engine) as source_witness:
                request = run_to_completion(
                    source_engine,
                    source_engine.add_request(101, tokens, sampling(1, do_kv_handoff=True)),
                )
            assert request.generated_tokens == expected[:1]
            assert source_witness.steps and all(step[2] for step in source_witness.steps)
            block_ids = request.disaggregated_params["block_ids"]
            local_shard = (
                source_engine._kv_transfer_agent.export_meta(),
                source_engine.context.memory_buffer[:, :, block_ids].cpu().clone(),
            )
            shards = gather(local_shard if rank in chosen_source else None, control)
            full = canonical_kv([shards[index] for index in chosen_source])
            full_ssm = None
            if model_provider == "hybrid":
                slot = source_engine._pinned_handoff_ssm_slots[101]
                source_ssm = (
                    source_engine._ssm_transfer_agents["conv"].export_meta(),
                    {
                        "conv": source_engine.context.mamba_conv_states[:, slot].cpu().clone(),
                        "recurrent": source_engine.context.mamba_ssm_states[:, slot].cpu().clone(),
                    },
                )
                ssm_shards = gather(source_ssm if rank in chosen_source else None, control)
                full_ssm = canonical_ssm([ssm_shards[index] for index in chosen_source])
            handoff = gather(
                request.disaggregated_params if rank == chosen_source[0] else None, control
            )[chosen_source[0]]
            # This resets MPU globals only; the source engine and its registered
            # storage/explicit process groups remain alive through the transfer.
            Utils.initialize_model_parallel(decode_tp, decode_pp)
            with real_engine(
                model_config(decode_tp, decode_pp, transformer_impl, model_provider),
                role="decode",
                backend=backend,
            ) as decode_engine:
                decode_model = decode_engine.controller.inference_wrapped_model.model
                load_canonical_model(decode_model, checkpoint_path)
                groups = gather(
                    dist.get_process_group_ranks(decode_engine.pg_collection.mp), control
                )
                chosen_decode = next(
                    group for group in groups if not set(group) & set(chosen_source)
                )
                assert not set(chosen_source) & set(chosen_decode)
                decode_engine.zmq_context = zmq.Context()
                decode_engine.zmq_sockets = []
                decode_engine._setup_handoff_completion_tracking(hostname="127.0.0.1")
                pending = future = None
                mixer_seen = True
                native_failure_seen = True
                if rank in chosen_decode:
                    bad_peer = failure and rank == chosen_decode[0]
                    if bad_peer:
                        handoff = deepcopy(handoff)
                        for meta, _ in transfer_peer_records(handoff["kv_meta"], block_ids):
                            meta["base_addr"] = 0
                    with ExitStack() as stack:
                        if failure:
                            agent = decode_engine._kv_transfer_agent._agent
                            prepare = stack.enter_context(
                                mock.patch.object(
                                    agent, "initialize_xfer", wraps=agent.initialize_xfer
                                )
                            )
                            post = stack.enter_context(
                                mock.patch.object(agent, "transfer", wraps=agent.transfer)
                            )
                        future = _enqueue_decode_handoff(decode_engine, handoff, tokens, sampling())
                    pending = decode_engine._pending_kv_imports[0]
                    if failure:
                        native_failure_seen = prepare.call_count > 0 and (
                            (pending.local_error is not None and post.call_count == 0)
                            if bad_peer
                            else (pending.local_error is None and post.call_count > 0)
                        )
                peers = gather(
                    decode_peer_meta(decode_engine, pending) if pending is not None else None,
                    control,
                )
                if backend == "nccl" and rank in chosen_source:
                    push_handoff(source_engine, 101, [peers[index] for index in chosen_decode])
                    for _, handles in source_engine._pending_kv_pushes:
                        for handle in handles:
                            handle.wait()
                            assert handle.poll()
                    assert source_engine._poll_pending_kv_pushes() == 1
                if failure and rank in chosen_decode:
                    for handle in decode_engine._pending_transfer_handles(pending):
                        handle.wait()
                        assert handle.poll()
                    if not bad_peer:
                        assert_logical_import(decode_engine, pending, full, len(tokens))
                    admit_model_group(decode_engine, failed=True)
                    assert future.done()
                    assert type(future.exception()).__name__ == (
                        "nixlNotFoundError" if bad_peer else "RuntimeError"
                    )
                    assert not decode_engine.context.total_request_count
                    assert not decode_engine._quarantined_kv_imports
                    assert_released(decode_engine)
                if rank in chosen_decode and not failure:
                    for handle in decode_engine._pending_transfer_handles(pending):
                        handle.wait()
                        assert handle.poll()
                    assert_logical_import(decode_engine, pending, full, len(tokens))
                    if full_ssm is not None:
                        slot = pending.ssm.live_slot
                        assert torch.equal(
                            decode_engine.context.mamba_conv_states[:, slot].cpu(), full_ssm["conv"]
                        )
                        assert torch.equal(
                            decode_engine.context.mamba_ssm_states[:, slot].cpu(),
                            full_ssm["recurrent"],
                        )
                    assert not decode_engine.context.total_request_count
                    admit_model_group(decode_engine)
                    with (
                        ForwardWitness(decode_engine) as witness,
                        implementation_witness(decode_model) as native,
                    ):
                        actual = run_to_completion(decode_engine, future)
                    assert actual.generated_tokens == expected
                    assert witness.steps and all(not step[2] for step in witness.steps)
                    assert witness.steps[0][0] == len(tokens)
                    if model_provider == "hybrid":
                        mixer_seen = any(name == "MambaMixer" for name, _ in native["layers"])
                    if transformer_impl == "local":
                        assert any(name == "ColumnParallelLinear" for name, _ in native["layers"])
                    elif transformer_impl == "inference_optimized":
                        assert any(name.startswith("Inference") for name, _ in native["layers"])
                        assert native["gathers"] and native["scatters"]
                        assert all(
                            after == before * decode_tp for before, after in native["gathers"]
                        )
                        assert all(
                            before == after * decode_tp for before, after in native["scatters"]
                        )
                    assert_released(decode_engine)
                    decode_engine._loop.run_until_complete(asyncio.sleep(0))
                dist.barrier(group=control)
                # All prefill replicas ran the reference workload, even replicas
                # not chosen as transfer peers; release their own pins as well.
                source_engine.release_handoff_blocks(101)
                assert_released(source_engine)
                for socket in decode_engine.zmq_sockets:
                    socket.close(linger=0)
                decode_engine.zmq_context.term()
                assert all(gather(mixer_seen, control))
                assert all(gather(native_failure_seen, control))
    finally:
        dist.destroy_process_group(control)
        Utils.destroy_model_parallel()
