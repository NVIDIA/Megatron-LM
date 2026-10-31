# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Public preparation/runtime contracts with real scheduling and feature transport.

Run with four or eight CUDA ranks. A small differentiable objective checks graph
lifetime and feature-gradient sums without model-specific processors or trainers.
"""

import os
import weakref

import pytest
import torch
import torch.distributed as dist

from megatron.core.datasets.data_schedule import DefaultDynamicCPScheduler, DpBalancedScheduler
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.models.mimo.packing import prepare_mimo_step
from megatron.core.models.mimo.partition.utils import PartitionAdapter, PartitionConfig
from megatron.core.models.mimo.planning import MediaTask, assign_media_to_producers
from megatron.core.models.mimo.runtime import LazyMimoStepRuntime, MimoStepRuntime

_MEDIA_TOKEN = 511


@pytest.fixture(scope="module", params=(False, True), ids=("colocated", "disaggregated"))
def runtime_groups(request):
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    width = 8 if request.param else 4
    if dist.get_world_size() % width:
        pytest.skip(f"This MIMO runtime case requires a multiple of {width} ranks")
    owned, created = {}, []
    rank = dist.get_rank()
    for base in range(0, dist.get_world_size(), width):
        all_ranks = tuple(range(base, base + width))
        decoder_ranks = all_ranks[-4:]
        producer_ranks = all_ranks[:4]
        if request.param:
            group = dist.new_group(ranks=list(all_ranks), backend="nccl")
            if rank in all_ranks:
                transport = group
                created.append(group)
        if rank in all_ranks:
            producers, decoders = producer_ranks, decoder_ranks
        for size in (4, 2, 1):
            for start in range(decoder_ranks[0], decoder_ranks[-1] + 1, size):
                ranks = tuple(range(start, start + size))
                group = dist.new_group(ranks=list(ranks), backend="nccl")
                if rank in ranks:
                    owned[size] = group
                    created.append(group)
                    if not request.param and size == 4:
                        transport = group
    yield transport, producers, decoders, owned
    for group in reversed(created):
        dist.destroy_process_group(group)


def _source(generation, *, pure_text, axes):
    tasks = (
        []
        if pure_text
        else [
            MediaTask(("asset", generation, i), 2 + i % 3, cost)
            for i, cost in enumerate((100, 90, 1, 1, 1, 1, 1, 1))
        ]
    )
    media_by_sample = ((0, 2, 3, 4, 7), (1, 2, 5), (3, 4, 6), (6, 7), (7,), (2,))
    samples = {}
    for ordinal, physical in enumerate((96, 64, 48, 32, 32, 16, 16, 16, 16)):
        logical = physical - 3
        media = []
        if tasks and ordinal < len(media_by_sample):
            media = [tasks[index] for index in media_by_sample[ordinal]]
        tokens = torch.full((physical,), ordinal + 1, dtype=torch.long)
        tokens[1 : 1 + sum(task.length for task in media)] = _MEDIA_TOKEN
        tokens[logical:] = 0
        mask = torch.zeros(physical)
        mask[2:logical] = 1  # Prompt positions remain real tokens despite zero loss.
        positions = torch.arange(physical) + ordinal * 100
        if axes > 1:
            positions = positions[None] + 1000 * torch.arange(axes)[:, None]
        positions[..., logical:] = 777
        samples[f"sample-{generation}-{ordinal * 17 + 3}"] = dict(
            tokens=tokens,
            labels=tokens.clone(),
            loss_mask=mask,
            position_ids=positions,
            original_seq_len=logical,
            padded_seq_len=physical,
            media_ids=[task.media_id for task in media],
        )
    if generation:
        samples = dict(reversed(tuple(samples.items())))
        tasks.reverse()
    return samples, tasks


def _rows(task):
    return torch.arange(task.length, device="cuda", dtype=torch.float64) + 1 + task.media_id[-1] * 5


def _objective(samples, tasks, encoder, decoder):
    """Independent reference in original sample order, without plans or CP indices."""
    rows = {task.media_id: _rows(task) for task in tasks}
    total = decoder * 0
    for sample in samples.values():
        values = sample["tokens"].to(device="cuda", dtype=torch.float64).clone()
        if sample["media_ids"]:
            values[values == _MEDIA_TOKEN] = (
                torch.cat([rows[mid] for mid in sample["media_ids"]]) * encoder
            )
        total = total + ((values * decoder).square() * sample["loss_mask"].cuda()).sum()
    return total


@pytest.mark.parametrize("dynamic", (False, True))
@pytest.mark.parametrize("lazy", (False, True))
def test_public_preparation_and_runtime_lifecycle(runtime_groups, dynamic, lazy):
    domain, producers, ranks, cp_groups = runtime_groups
    rank = dist.get_rank()
    is_decoder = rank in ranks
    prefetch = lazy and not is_decoder
    limit = 32 if dynamic else 64
    scheduler = (DefaultDynamicCPScheduler if dynamic else DpBalancedScheduler)(
        limit, cp_size=2, dp_size=2, microbatch_group_size_per_vp_stage=None
    )
    config = ModelParallelConfig(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=2,
        max_seqlen_per_dp_cp_rank=limit,
        pad_packed_seq_alignment="max",
    )
    partition = (
        PartitionAdapter(
            PartitionConfig(False, True, False, limit * 4, kv_format="thd", cp_group=cp_groups[2])
        )
        if is_decoder
        else None
    )
    runtime = (LazyMimoStepRuntime if lazy else MimoStepRuntime)(
        domain, producer_ranks=producers, decoder_ranks=ranks
    )
    old_transfer = None
    observed_cp = torch.zeros(5, device="cuda", dtype=torch.int)
    for step, (pure_text, forward_only, axes) in enumerate(
        ((False, False, 3), (True, False, 1), (False, True, 2), (False, False, 3))
    ):
        samples, tasks = _source(step // 2, pure_text=pure_text, axes=axes)
        snapshots = {
            sid: {key: value.clone() for key, value in sample.items() if torch.is_tensor(value)}
            for sid, sample in samples.items()
        }
        # Keep one producer empty while six media share another producer. Their
        # overlapping lifetimes require independent graphs and compacted offsets.
        assigned, slices = assign_media_to_producers(tasks, producers[:-1])
        assigned[producers[-1]] = []
        if tasks:
            # A second descriptor for the same rows must not encode them twice.
            slices[("alias", step)] = slices[tasks[0].media_id]
        if is_decoder:
            plan = prepare_mimo_step(
                samples,
                slices,
                scheduler=scheduler,
                domain_ranks=ranks,
                cp_groups=cp_groups,
                config=config,
                device=torch.device("cuda"),
            )
            assert plan.num_source_samples == len(samples)
            assert plan.num_supervised_tokens == sum(
                int(s["loss_mask"].sum()) for s in samples.values()
            )
            assert plan.source_tokens == sum(s["padded_seq_len"] for s in samples.values())
            assert plan.source_tokens_squared == sum(
                s["padded_seq_len"] ** 2 for s in samples.values()
            )
        # Only route descriptors cross the pools; producers never build decoder batches.
        shared = [tuple(item.feature_plans for item in plan.rounds) if is_decoder else None]
        dist.broadcast_object_list(shared, src=ranks[0], group=domain)
        rounds = shared[0]
        assert len(rounds) > 1
        encoder = torch.tensor(2.0, device="cuda", dtype=torch.float64, requires_grad=True)
        decoder = torch.tensor(0.25, device="cuda", dtype=torch.float64, requires_grad=True)
        backward_calls = []
        encoder.register_hook(lambda gradient: backward_calls.append(1))
        empty = torch.empty((0, 1), device="cuda", dtype=torch.float64)
        by_id = {task.media_id: task for task in assigned.get(rank, [])}
        usage = [{feature for pack in plans for feature in pack.features} for plans in rounds]
        lifetimes = {}
        for media_id in by_id:
            consumers = [i for i, features in enumerate(usage) if slices[media_id] in features]
            if consumers:
                lifetimes[media_id] = (consumers[0], consumers[-1])
        if dynamic and not pure_text and rank == producers[2]:
            assert len(by_id) >= 3 and len(set(lifetimes.values())) >= 3
            assert any(list(lifetimes.values()).count(span) > 1 for span in lifetimes.values())
            assert any(
                first <= other_last and other_first <= last
                for first, last in lifetimes.values()
                for other_first, other_last in lifetimes.values()
                if (first, last) != (other_first, other_last)
            )
        encode_calls, graph_refs = [], []
        previous_offsets = {}
        rebased_after_release = False
        round_index = -1
        encode_round = -1

        def encode(media_ids):
            media_ids = tuple(media_ids)
            assert media_ids
            assert all(lifetimes[mid][0] == encode_round for mid in media_ids)
            assert len({lifetimes[mid] for mid in media_ids}) == 1
            assert list(media_ids) == sorted(media_ids, key=lambda mid: slices[mid].offset)
            encode_calls.append((round_index, media_ids))
            features = torch.cat([_rows(by_id[mid]) for mid in media_ids])[:, None] * encoder
            graph_refs.append((lifetimes[media_ids[0]][1], weakref.ref(features)))
            return features

        if lazy:

            def begin():
                runtime.begin_lazy_step(rounds, slices, encode, empty, forward_only=forward_only)

        else:
            with torch.set_grad_enabled(not forward_only):
                features = (
                    torch.cat([_rows(task) for task in assigned.get(rank, [])])[:, None] * encoder
                    if assigned.get(rank)
                    else empty
                )

            def begin():
                runtime.begin_step(features, len(rounds), forward_only=forward_only)

        # Preparation context must not override the explicit execution mode.
        with torch.no_grad():
            begin()
        assert runtime.active and not encode_calls
        if lazy:
            with pytest.raises(RuntimeError, match="pending round"):
                runtime.prefetch_next_round()
        with pytest.raises(RuntimeError, match="active MIMO"):
            begin()
        finish = runtime.finish_forward if forward_only else runtime.finish_backward
        with pytest.raises(RuntimeError, match="complete every"):
            finish()
        for round_index, plans in enumerate(rounds):
            item = plan.rounds[round_index] if is_decoder else None
            group = item.cp_group if is_decoder else None
            encode_round = round_index
            transfer = runtime.route(plans, group)
            if prefetch:
                if runtime._ready_backward:
                    with pytest.raises(RuntimeError, match="Drain ready"):
                        runtime.prefetch_next_round()
                    with pytest.raises(RuntimeError, match="Drain ready"):
                        runtime.complete_round(transfer, defer_encoder_backward=True)
                runtime.backward_ready()
                encode_round = round_index + 1
                runtime.prefetch_next_round()
                calls = len(encode_calls)
                runtime.prefetch_next_round()
                assert len(encode_calls) == calls
                assert runtime.pending is transfer and runtime.completed_rounds == round_index
                assert all(
                    chunk.first_round <= round_index + 1 and chunk.last_round >= round_index
                    for chunk in runtime.live_chunks
                )
            elif lazy:
                with pytest.raises(ValueError, match="producer-only"):
                    runtime.prefetch_next_round()
            if lazy:
                for original, routed in zip(plans, transfer.plans):
                    for source, destination in zip(original.features, routed.features):
                        if source.producer_rank == rank:
                            if (
                                source in previous_offsets
                                and destination.offset != previous_offsets[source]
                                and any(last < round_index for _, last in lifetimes.values())
                            ):
                                rebased_after_release = True
                            previous_offsets[source] = destination.offset
            if is_decoder:
                assert transfer.features.requires_grad == (not forward_only)
            else:
                assert transfer.features is transfer.local_plan is transfer.cp_group is None
            if old_transfer is not None:
                with pytest.raises(RuntimeError, match="stale or completed"):
                    runtime.complete_round(old_transfer)
            with pytest.raises(RuntimeError, match="pending"):
                runtime.route(plans, group)
            with pytest.raises(RuntimeError, match="complete every"):
                finish()
            if is_decoder:
                _decode_round(
                    item, transfer, samples, tasks, partition, decoder, forward_only, axes, limit
                )
                observed_cp[group.size()] = 1
            completed_chunks = (
                sum(last < round_index for _, last in set(lifetimes.values()))
                if lazy and not forward_only
                else 0
            )
            assert len(backward_calls) == completed_chunks
            if not completed_chunks:
                assert encoder.grad is None
            if prefetch:
                runtime.complete_round(transfer, defer_encoder_backward=True)
            else:
                runtime.complete_round(transfer)
            if lazy:
                encoded_ids = [mid for _, ids in encode_calls for mid in ids]
                expected_ids = {
                    mid for mid, (first, _) in lifetimes.items() if first <= round_index + prefetch
                }
                assert len(encoded_ids) == len(set(encoded_ids))
                assert set(encoded_ids) == expected_ids
                last_released = round_index - (prefetch and not forward_only)
                expected_backward = (
                    sum(last <= last_released for _, last in set(lifetimes.values()))
                    if not forward_only
                    else 0
                )
                assert len(backward_calls) == expected_backward
                assert all((ref() is None) == (last <= last_released) for last, ref in graph_refs)
                assert runtime.active  # Also true between empty / pure-text rounds.
            with pytest.raises(RuntimeError, match="stale or completed"):
                runtime.complete_round(transfer)
            old_transfer = transfer
        if lazy and dynamic and step == 0 and rank == producers[2]:
            assert rebased_after_release
        with pytest.raises(RuntimeError, match="exhausted" if lazy else "already completed"):
            runtime.route(plans, group)
        wrong_finish = runtime.finish_backward if forward_only else runtime.finish_forward
        with pytest.raises(RuntimeError, match="requested execution mode"):
            wrong_finish()
        finish()
        assert not runtime.active
        assert runtime.local_features is runtime.feature_gradient is runtime.pending is None
        if lazy:
            assert not runtime.live_chunks and not runtime._ready_backward
            with pytest.raises(RuntimeError, match="active source step"):
                runtime.backward_ready()
        expected_backward = len(set(lifetimes.values())) if lazy else bool(assigned.get(rank))
        assert backward_calls == [1] * (expected_backward if not forward_only else 0)
        assert all(ref() is None for _, ref in graph_refs)
        with pytest.raises(RuntimeError, match="requested execution mode"):
            finish()
        if forward_only:
            assert encoder.grad is decoder.grad is None
        else:
            encoder_ref = encoder.detach().requires_grad_()
            decoder_ref = decoder.detach().requires_grad_()
            _objective(samples, tasks, encoder_ref, decoder_ref).backward()
            for parameter, reference in ((encoder, encoder_ref), (decoder, decoder_ref)):
                gradient = (
                    parameter.grad if parameter.grad is not None else torch.zeros_like(parameter)
                )
                dist.all_reduce(gradient, group=domain)
                torch.testing.assert_close(
                    gradient,
                    reference.grad if reference.grad is not None else torch.zeros_like(reference),
                    rtol=0,
                    atol=0,
                )
        for sid, snapshot in snapshots.items():
            for key, value in snapshot.items():
                torch.testing.assert_close(samples[sid][key], value, rtol=0, atol=0)
    dist.all_reduce(observed_cp, op=dist.ReduceOp.MAX, group=domain)
    assert observed_cp.nonzero().flatten().tolist() == ([1, 2, 4] if dynamic else [2])


def _decode_round(item, transfer, samples, tasks, partition, decoder, forward_only, axes, limit):
    """Check an independently assembled pack before the existing CP partition and objective."""
    inputs = item.model_inputs
    assert inputs["input_ids"].numel() == limit * item.cp_group.size()
    assert inputs["position_ids"].shape[:-2] == (() if axes == 1 else (axes,))
    for slot, sid in enumerate(item.sample_ids):
        sample = samples[sid]
        start = int(inputs["cu_seqlens_padded"][slot])
        end = start + sample["original_seq_len"]
        assert int(inputs["cu_seqlens"].diff()[slot]) == sample["original_seq_len"]
        torch.testing.assert_close(
            inputs["position_ids"][..., 0, start:end].cpu(),
            sample["position_ids"][..., : sample["original_seq_len"]],
        )
        assert not inputs["padding_mask"][0, start:end].any()
        assert not inputs["loss_mask"][0, start : start + 2].any()
    padding = inputs["padding_mask"][0]
    assert not inputs["position_ids"][..., 0, padding].any()
    assert not inputs["loss_mask"][0, padding].any()
    rows = {task.media_id: _rows(task) for task in tasks}
    pieces = [rows[mid] for sid in item.sample_ids for mid in samples[sid]["media_ids"]]
    expected = torch.cat(pieces)[:, None] * 2 if pieces else transfer.features.new_empty((0, 1))
    torch.testing.assert_close(transfer.features, expected, rtol=0, atol=0)
    with torch.set_grad_enabled(not forward_only):
        values = inputs["input_ids"].to(torch.float64).clone()
        values[values == _MEDIA_TOKEN] = transfer.features[:, 0]
        batch = partition.partition(
            embeddings=values.transpose(0, 1).unsqueeze(-1),
            input_ids=inputs["input_ids"],
            position_ids=inputs["position_ids"],
            labels=inputs["labels"],
            loss_mask=inputs["loss_mask"],
            mtp_input_mask=None,
            cu_seqlens=inputs["cu_seqlens"],
            cu_seqlens_padded=inputs["cu_seqlens_padded"],
            max_seqlen=inputs["max_seqlen"],
            padding_mask=inputs["padding_mask"],
            cp_group=item.cp_group,
        ).get_batch()
        output = (batch["decoder_input"].squeeze(-1).transpose(0, 1) * decoder).square()
        loss = (output * batch["loss_mask"]).sum()
        if not forward_only:
            loss.backward()
