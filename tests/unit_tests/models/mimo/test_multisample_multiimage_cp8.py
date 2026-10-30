# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""One exact multi-sample, multi-image routing case chosen by the real DCP scheduler.

Run with sixteen colocated ranks or twenty disjoint ranks (four encoder and
sixteen decoder) using torch.distributed.run -m pytest. This checks the MIMO
embedding boundary, not transformer or optimizer numerical equivalence.
"""

import os

import pytest
import torch
import torch.distributed as dist

from megatron.core.datasets.data_schedule import DefaultDynamicCPScheduler
from megatron.core.datasets.data_schedule_utils import build_packed_microbatches
from megatron.core.models.mimo.comm.pack_bridge import PackFeatureBridge
from megatron.core.models.mimo.model.base import MimoModel
from megatron.core.models.mimo.partition.utils import PartitionAdapter, PartitionConfig
from megatron.core.models.mimo.planning import (
    MediaTask,
    assign_media_to_producers,
    build_round_plans,
)

_WIDTH, _IMAGE_TOKEN, _LABEL_STRIDE = 3, 100000, 4096
# (physical length, real length, (image ID, first position, number of rows) ...)
_SAMPLES = (
    (768, 755, ((4, 48, 24), (0, 72, 48), (3, 120, 40))),
    (256, 243, ((1, 20, 12), (2, 32, 28))),
    (512, 507, ()),
    (512, 499, ()),
)
_IMAGES = {
    image: (sid, start, length)
    for sid, (_, _, images) in enumerate(_SAMPLES)
    for image, start, length in images
}
_PRODUCERS = (8, 11, 14, 4, 7, 0, 1, 2, 3, 5, 6, 9, 10, 12, 13, 15)


@pytest.fixture(scope="module", params=(False, True), ids=("colocated", "disaggregated"))
def runtime_cp_group(request):
    producers = tuple(range(4)) if request.param else _PRODUCERS
    decoders = tuple(range(4, 20)) if request.param else tuple(range(16))
    required = len(set(producers) | set(decoders))
    if int(os.environ.get("WORLD_SIZE", "1")) != required:
        pytest.skip(f"This joint DCP routing placement requires {required} CUDA ranks")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank, owned, runtime = dist.get_rank(), [], None
    decoder_group = dist.new_group(list(decoders), backend="nccl")
    if rank in decoders:
        owned.append(decoder_group)
    for start, size in ((0, 8), (8, 4), (12, 4)):
        ranks = decoders[start : start + size]
        group = dist.new_group(list(ranks), backend="nccl")
        if rank in ranks:
            owned.append(group)
            runtime = group
    yield producers, decoders, decoder_group, runtime
    for group in reversed(owned):
        dist.destroy_process_group(group)


def _positions(sample_ids):
    """Original identities; never read candidate feature plans or TE indices."""
    result = []
    for sid in sample_ids:
        physical, _, images = _SAMPLES[sid]
        visual = {start + row: (image, row) for image, start, n in images for row in range(n)}
        result.extend((sid, pos, visual.get(pos)) for pos in range(physical))
    return result


def _raw(image, row):
    sid = _IMAGES[image][0]
    # Unique (sample, image, row) vectors, small enough for exact FP32 VJPs.
    return [sid + 1, image + 1, row + 1]


def _cotangent(sid, pos):
    return [
        1 + 1024 * sid + pos + channel if pos < _SAMPLES[sid][1] else 0 for channel in range(_WIDTH)
    ]


def _exact(actual, expected):
    expected = torch.as_tensor(expected, dtype=actual.dtype, device=actual.device).reshape(
        actual.shape
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def _decode_pack(samples, assignments, decoder_index, transfer, runtime_cp_group, decoder_group):
    """Exercise the same full-pack and CP-row oracle for either physical placement."""
    device = torch.device("cuda")
    tensor_samples = {
        sid: {key: value for key, value in sample.items() if isinstance(value, torch.Tensor)}
        for sid, sample in samples.items()
    }
    batch = build_packed_microbatches(
        tensor_samples, assignments, decoder_index, device, is_dynamic_cp=True
    )[0]
    sample_ids = assignments[0][decoder_index]
    for key, field in (('cu_seqlens', 1), ('cu_seqlens_padded', 0)):
        _exact(
            batch[key],
            [
                sum(_SAMPLES[sid][field] for sid in sample_ids[:i])
                for i in range(len(sample_ids) + 1)
            ],
        )
    assert int(batch['local_cp_size']) == runtime_cp_group.size()
    identities = _positions(sample_ids)
    expected = []
    for sid, pos, image_row in identities:
        token = 1000 + sid * _LABEL_STRIDE + pos if pos < _SAMPLES[sid][1] else 0
        expected.append(
            [value * 2 for value in _raw(*image_row)]
            if image_row
            else [-token - channel for channel in range(_WIDTH)]
        )
    visual_indices = [i for i, (_, _, image) in enumerate(identities) if image is not None]
    _exact(transfer.features, [expected[i] for i in visual_indices])
    tokens = batch['tokens'][None]
    text_indices = (tokens.flatten() != _IMAGE_TOKEN).nonzero().flatten()
    text = (
        (-tokens.flatten()[text_indices, None] - torch.arange(_WIDTH, device=device))
        .float()
        .requires_grad_()
    )
    merged = MimoModel.align_embeddings_by_token_positions(
        None, {'text': text, 'images': transfer.features}, tokens, {'images': _IMAGE_TOKEN}
    )
    _exact(merged, expected)
    adapter = PartitionAdapter(
        PartitionConfig(
            seq_parallel=False,
            use_cp=True,
            tp_comm_overlap=False,
            max_seq_len=1024,
            kv_format='thd',
            cp_group=decoder_group,
        )
    )
    cp_batch = adapter.partition(
        embeddings=merged,
        input_ids=tokens,
        position_ids=batch['position_ids'][None],
        labels=batch['labels'][None],
        loss_mask=batch['loss_mask'][None],
        mtp_input_mask=None,
        padding_mask=~batch['loss_mask'][None].bool(),
        cu_seqlens=batch['cu_seqlens'],
        cu_seqlens_padded=batch['cu_seqlens_padded'],
        max_seqlen=batch['max_seqlen'],
        cp_group=runtime_cp_group,
    ).get_batch()
    local, labels, mask = (cp_batch[key] for key in ('decoder_input', 'labels', 'loss_mask'))
    # Independent per-document zigzag, not TE's partition indices.
    owned, offset = [], 0
    for sid in sample_ids:
        length = _SAMPLES[sid][0]
        cp_size, cp_rank = runtime_cp_group.size(), runtime_cp_group.rank()
        chunk = length // (2 * cp_size)
        for part in (cp_rank, 2 * cp_size - 1 - cp_rank):
            owned.extend(range(offset + part * chunk, offset + (part + 1) * chunk))
        offset += length
    _exact(local, [expected[i] for i in owned])
    _exact(
        labels,
        [
            sid * _LABEL_STRIDE + pos if pos < _SAMPLES[sid][1] else 0
            for sid, pos, _ in (identities[i] for i in owned)
        ],
    )
    _exact(mask, [int(identities[i][1] < _SAMPLES[identities[i][0]][1]) for i in owned])
    _exact(
        cp_batch['padding_mask'], [identities[i][1] >= _SAMPLES[identities[i][0]][1] for i in owned]
    )
    cotangent = torch.tensor(
        [_cotangent(*identities[i][:2]) for i in owned], dtype=torch.float32, device=device
    )[:, None]
    (local * cotangent).sum().backward()
    owned_set = set(owned)
    _exact(
        transfer.features.grad,
        [
            _cotangent(*identities[i][:2]) if i in owned_set else [0] * _WIDTH
            for i in visual_indices
        ],
    )
    _exact(
        text.grad,
        [
            _cotangent(*identities[i][:2]) if i in owned_set else [0] * _WIDTH
            for i in text_indices.tolist()
        ],
    )
    return labels


def test_dynamic_cp_two_samples_five_images_share_cp8(runtime_cp_group):
    producers, decoders, decoder_group, group = runtime_cp_group
    rank = dist.get_rank()
    device = torch.device("cuda")
    sample_device = device if rank in decoders else torch.device("cpu")
    scheduler = DefaultDynamicCPScheduler(
        max_seqlen_per_dp_cp_rank=128, cp_size=8, dp_size=2, microbatch_group_size_per_vp_stage=None
    )
    assignments = scheduler.get_groups_and_subsamples(
        [(sid, spec[0]) for sid, spec in enumerate(_SAMPLES)]
    )
    assert assignments == [
        [[0, 1]] * 8 + [[2]] * 4 + [[3]] * 4
    ], 'The fixture must actually schedule A/B together at CP8'
    samples = {}
    for sid, (physical, real, images) in enumerate(_SAMPLES):
        identities = _positions((sid,))
        tokens = [
            (
                _IMAGE_TOKEN
                if image is not None
                else 1000 + sid * _LABEL_STRIDE + pos if pos < real else 0
            )
            for _, pos, image in identities
        ]
        samples[sid] = {
            'tokens': torch.tensor(tokens, device=sample_device),
            'labels': torch.arange(physical, device=sample_device) + sid * _LABEL_STRIDE,
            'loss_mask': (torch.arange(physical, device=sample_device) < real).float(),
            'position_ids': torch.arange(physical, device=sample_device),
            'original_seq_len': torch.tensor(real, dtype=torch.int32, device=sample_device),
            'padded_seq_len': torch.tensor(physical, dtype=torch.int32, device=sample_device),
            'media_ids': [image for image, _, _ in images],
        }
    # Equal costs make the expected producer assignment independent and explicit.
    media = [MediaTask(image, info[2]) for image, info in sorted(_IMAGES.items())]
    tasks, slices = assign_media_to_producers(media, producers)
    image_producers = (0, 1, 2, 3, 0) if len(producers) == 4 else _PRODUCERS[:5]
    for image, piece in slices.items():
        offset = sum(
            _IMAGES[other][2]
            for other in range(image)
            if image_producers[other] == image_producers[image]
        )
        assert (piece.producer_rank, piece.offset, piece.length) == (
            image_producers[image],
            offset,
            _IMAGES[image][2],
        )
    local_images = [image for image in sorted(_IMAGES) if image_producers[image] == rank]
    assert [item.media_id for item in tasks.get(rank, ())] == local_images
    raw_rows = [_raw(image, row) for image in local_images for row in range(_IMAGES[image][2])]
    raw = torch.tensor(raw_rows, dtype=torch.float32, device=device).reshape(-1, _WIDTH)
    gain = torch.tensor(2.0, device=device, requires_grad=True)
    source = raw * gain if rank in producers else raw
    encoder_backward_calls = []
    if source.requires_grad:
        source.retain_grad()
        source.register_hook(lambda gradient: encoder_backward_calls.append(1))
    bridge = PackFeatureBridge(dist.group.WORLD, producer_ranks=producers, decoder_ranks=decoders)
    plans = build_round_plans(assignments[0], samples, slices, decoders)
    assert [plan.cp_ranks for plan in plans] == [decoders[:8], decoders[8:12], decoders[12:]]
    transfer = bridge.forward(source, plans, group)
    if rank in decoders:
        labels = _decode_pack(
            samples, assignments, decoders.index(rank), transfer, group, decoder_group
        )
    else:
        assert transfer.features is transfer.local_plan is transfer.cp_group is None
        labels = None
    assert source.grad is None and gain.grad is None and encoder_backward_calls == []
    returned = bridge.backward(transfer)
    source_reference = [
        _cotangent(_IMAGES[image][0], _IMAGES[image][1] + row)
        for image in local_images
        for row in range(_IMAGES[image][2])
    ]
    _exact(returned, source_reference)
    if source.requires_grad:
        source.backward(returned)
        assert encoder_backward_calls == [1]
        _exact(source.grad, source_reference)
    else:
        assert not local_images and encoder_backward_calls == []
    expected_gain_gradient = sum(
        a * b for values, grads in zip(raw_rows, source_reference) for a, b in zip(values, grads)
    )
    assert expected_gain_gradient < 2**24  # Every positive FP32 partial sum is exact.
    if rank in producers:
        _exact(gain.grad, expected_gain_gradient)
    else:
        assert gain.grad is None
    # Count identities recovered from ACTUAL sharded labels, without gathering features.
    canonical_visual = [
        (sid, pos, image) for sid, pos, image in _positions((0, 1)) if image is not None
    ]
    row_ids = {(sid, pos): index for index, (sid, pos, _) in enumerate(canonical_visual)}
    counts = torch.zeros(
        (len(canonical_visual), dist.get_world_size()), dtype=torch.int32, device=device
    )
    for label in labels.flatten().tolist() if labels is not None else ():
        sid, pos = divmod(label, _LABEL_STRIDE)
        if (sid, pos) in row_ids:
            row_id = row_ids[sid, pos]
            counts[row_id, rank] += 1
    dist.all_reduce(counts)
    _exact(counts.sum(dim=1), [1] * 152)
    expected_counts = [0] * dist.get_world_size()
    for owner, count in ((1, 60), (2, 64), (3, 28)):
        expected_counts[decoders[owner]] = count
    _exact(counts.sum(dim=0), expected_counts)
    image_owners = {
        image: counts[[i for i, row in enumerate(canonical_visual) if row[2][0] == image]]
        .sum(0)
        .nonzero()
        .flatten()
        .tolist()
        for image in _IMAGES
    }
    assert image_owners == {
        image: [decoders[owner] for owner in owners]
        for image, owners in {4: [1], 0: [1, 2], 3: [2, 3], 1: [1], 2: [2, 3]}.items()
    }
    assert any(producer not in decoders[:8] for producer in image_producers)
