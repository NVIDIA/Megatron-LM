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


@pytest.fixture(scope="module")
def runtime_groups():
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    if dist.get_world_size() % 4:
        pytest.skip("MIMO runtime tests require a multiple of four ranks")
    owned = {}
    for base in range(0, dist.get_world_size(), 4):
        for size in (4, 2, 1):
            for start in range(base, base + 4, size):
                ranks = tuple(range(start, start + size))
                group = dist.new_group(ranks=list(ranks), backend="nccl")
                if dist.get_rank() in ranks:
                    owned[size] = group
    yield owned
    for group in reversed(tuple(owned.values())):
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
    domain = runtime_groups[4]
    ranks = tuple(dist.get_process_group_ranks(domain))
    rank = dist.get_rank()
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
    partition = PartitionAdapter(
        PartitionConfig(False, True, False, limit * 4, kv_format="thd", cp_group=runtime_groups[2])
    )
    runtime = (LazyMimoStepRuntime if lazy else MimoStepRuntime)(domain)
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
        assigned, slices = assign_media_to_producers(tasks, ranks[:-1])
        assigned[ranks[-1]] = []
        if tasks:
            # A second descriptor for the same rows must not encode them twice.
            slices[("alias", step)] = slices[tasks[0].media_id]
        plan = prepare_mimo_step(
            samples,
            slices,
            scheduler=scheduler,
            domain_ranks=ranks,
            cp_groups=runtime_groups,
            config=config,
            device=torch.device("cuda"),
        )
        assert plan.num_source_samples == len(samples)
        assert plan.num_supervised_tokens == sum(
            int(s["loss_mask"].sum()) for s in samples.values()
        )
        assert plan.source_tokens == sum(s["padded_seq_len"] for s in samples.values())
        assert plan.source_tokens_squared == sum(s["padded_seq_len"] ** 2 for s in samples.values())
        assert len(plan.rounds) > 1
        encoder = torch.tensor(2.0, device="cuda", dtype=torch.float64, requires_grad=True)
        decoder = torch.tensor(0.25, device="cuda", dtype=torch.float64, requires_grad=True)
        backward_calls = []
        encoder.register_hook(lambda gradient: backward_calls.append(1))
        empty = torch.empty((0, 1), device="cuda", dtype=torch.float64)
        by_id = {task.media_id: task for task in assigned[rank]}
        usage = [
            {feature for pack in item.feature_plans for feature in pack.features}
            for item in plan.rounds
        ]
        lifetimes = {}
        for media_id in by_id:
            consumers = [i for i, features in enumerate(usage) if slices[media_id] in features]
            if consumers:
                lifetimes[media_id] = (consumers[0], consumers[-1])
        if dynamic and not pure_text and rank == ranks[2]:
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

        def encode(media_ids):
            media_ids = tuple(media_ids)
            assert media_ids
            assert all(lifetimes[mid][0] == round_index for mid in media_ids)
            assert len({lifetimes[mid] for mid in media_ids}) == 1
            assert list(media_ids) == sorted(media_ids, key=lambda mid: slices[mid].offset)
            encode_calls.append((round_index, media_ids))
            features = torch.cat([_rows(by_id[mid]) for mid in media_ids])[:, None] * encoder
            graph_refs.append((lifetimes[media_ids[0]][1], weakref.ref(features)))
            return features

        if lazy:

            def begin():
                runtime.begin_lazy_step(
                    [item.feature_plans for item in plan.rounds],
                    slices,
                    encode,
                    empty,
                    forward_only=forward_only,
                )

        else:
            with torch.set_grad_enabled(not forward_only):
                features = (
                    torch.cat([_rows(task) for task in assigned[rank]])[:, None] * encoder
                    if assigned[rank]
                    else empty
                )

            def begin():
                runtime.begin_step(features, len(plan.rounds), forward_only=forward_only)

        # Preparation context must not override the explicit execution mode.
        with torch.no_grad():
            begin()
        assert runtime.active and not encode_calls
        with pytest.raises(RuntimeError, match="active MIMO"):
            begin()
        finish = runtime.finish_forward if forward_only else runtime.finish_backward
        with pytest.raises(RuntimeError, match="complete every"):
            finish()
        for round_index, item in enumerate(plan.rounds):
            inputs = item.model_inputs
            observed_cp[item.cp_group.size()] = 1
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
            transfer = runtime.route(item.feature_plans, item.cp_group)
            if lazy:
                for original, routed in zip(item.feature_plans, transfer.plans):
                    for source, destination in zip(original.features, routed.features):
                        if source.producer_rank == rank:
                            if (
                                source in previous_offsets
                                and destination.offset != previous_offsets[source]
                                and any(last < round_index for _, last in lifetimes.values())
                            ):
                                rebased_after_release = True
                            previous_offsets[source] = destination.offset
            assert transfer.features.requires_grad == (not forward_only)
            if old_transfer is not None:
                with pytest.raises(RuntimeError, match="stale or completed"):
                    runtime.complete_round(old_transfer)
            with pytest.raises(RuntimeError, match="pending"):
                runtime.route(item.feature_plans, item.cp_group)
            with pytest.raises(RuntimeError, match="complete every"):
                finish()
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
            completed_chunks = (
                sum(last < round_index for _, last in set(lifetimes.values()))
                if lazy and not forward_only
                else 0
            )
            assert len(backward_calls) == completed_chunks
            if not completed_chunks:
                assert encoder.grad is None
            runtime.complete_round(transfer)
            if lazy:
                encoded_ids = [mid for _, ids in encode_calls for mid in ids]
                expected_ids = {
                    mid for mid, (first, _) in lifetimes.items() if first <= round_index
                }
                assert len(encoded_ids) == len(set(encoded_ids))
                assert set(encoded_ids) == expected_ids
                expected_backward = (
                    sum(last <= round_index for _, last in set(lifetimes.values()))
                    if not forward_only
                    else 0
                )
                assert len(backward_calls) == expected_backward
                assert all((ref() is None) == (last <= round_index) for last, ref in graph_refs)
                assert runtime.active  # Also true between empty / pure-text rounds.
            with pytest.raises(RuntimeError, match="stale or completed"):
                runtime.complete_round(transfer)
            old_transfer = transfer
        if lazy and dynamic and step == 0 and rank == ranks[2]:
            assert rebased_after_release
        with pytest.raises(RuntimeError, match="exhausted" if lazy else "already completed"):
            runtime.route(item.feature_plans, item.cp_group)
        wrong_finish = runtime.finish_backward if forward_only else runtime.finish_forward
        with pytest.raises(RuntimeError, match="requested execution mode"):
            wrong_finish()
        finish()
        assert not runtime.active
        assert runtime.local_features is runtime.feature_gradient is runtime.pending is None
        expected_backward = len(set(lifetimes.values())) if lazy else bool(assigned[rank])
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
            encoder_gradient = (
                encoder.grad if encoder.grad is not None else torch.zeros_like(encoder)
            )
            dist.all_reduce(encoder_gradient, group=domain)
            dist.all_reduce(decoder.grad, group=domain)
            torch.testing.assert_close(
                encoder_gradient,
                encoder_ref.grad if encoder_ref.grad is not None else torch.zeros_like(encoder),
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(decoder.grad, decoder_ref.grad, rtol=0, atol=0)
        for sid, snapshot in snapshots.items():
            for key, value in snapshot.items():
                torch.testing.assert_close(samples[sid][key], value, rtol=0, atol=0)
    dist.all_reduce(observed_cp, op=dist.ReduceOp.MAX, group=domain)
    assert observed_cp.nonzero().flatten().tolist() == ([1, 2, 4] if dynamic else [2])
