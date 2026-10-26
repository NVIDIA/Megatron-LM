# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Integration of reference values with the saver, payload decoder, and loss."""

import io
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from megatron.training.distillation import cached_logits_loss, logits_saver
from megatron.training.distillation.logits_saver import LogitsSaverHooks
from megatron.training.distillation.reference_logits import reference_logprobs
from megatron.training.distillation.utils import (
    LOGPROBS_FORMAT_VERSION,
    decode_logprobs_payload,
    v2_pack_indices,
)

zstandard = pytest.importorskip("zstandard")


def _bare_saver(k=3, p=None):
    saver = LogitsSaverHooks.__new__(LogitsSaverHooks)
    saver.tp_rank = 0
    saver.tp_size = 1
    saver.tp_group = None
    saver.k = k
    saver.p = p
    saver.min_k = 1
    saver._save_dtype = torch.float16
    saver._topp_kept_counts = []
    return saver


def _payload(values, indices, version):
    low, high = v2_pack_indices(indices)
    buffer = io.BytesIO()
    torch.save(
        {
            "reference_values" if version == 3 else "values": values,
            "indices_low": low,
            "bit_17": high,
            "format_version": version,
        },
        buffer,
    )
    return zstandard.ZstdCompressor().compress(buffer.getvalue())


@pytest.mark.parametrize("p", [None, 0.6, 0.9])
def test_saver_payload_and_student_loss(p):
    saver = _bare_saver(p=p)
    teacher = torch.tensor([[[0.0, -1.0, -2.0, -30.0]], [[-2.0, 0.0, -1.0, -3.0]]])
    values, indices = saver._process_single_microbatch(teacher)
    assert values.dtype == torch.float16
    assert values.shape == (2, 1, 3)
    assert LOGPROBS_FORMAT_VERSION == 3
    loaded, unpacked, _ = decode_logprobs_payload(
        _payload(values, indices, 3), expected_format_version=3
    )
    torch.testing.assert_close(loaded, values, atol=0, rtol=0)
    decoded = reference_logprobs(loaded).exp()
    keep = torch.cat(
        (torch.ones_like(values[..., :1], dtype=torch.bool), values[..., 1:].isfinite()), -1
    )
    expected_selected = teacher.softmax(-1).gather(-1, indices.clamp(min=0)).masked_fill(~keep, 0)
    # This subtraction is only a modest-confidence FP64/FP32 reference for
    # this integration test; tiny tails have a separate numerical test.
    expected_tail = 1 - expected_selected.sum(-1, keepdim=True)
    torch.testing.assert_close(
        decoded, torch.cat((expected_selected, expected_tail), -1), atol=2e-4, rtol=0.01
    )
    student = (teacher + torch.tensor([0.1, -0.1, 0.2, -0.2])).requires_grad_()
    before = student.detach().clone()
    loss = cached_logits_loss.topk_kl_div(
        student, loaded, unpacked, 1, 0, None, add_ghost_token=True, cache_format_version=3
    )
    assert loss.shape == (1, 2)
    loss.sum().backward()
    assert torch.isfinite(student.grad).all()
    torch.testing.assert_close(before, student.detach(), atol=0, rtol=0)


def test_payload_rejects_mixed_and_unknown_formats():
    values = torch.zeros(1, 1, 1, dtype=torch.float16)
    indices = torch.zeros(1, 1, 1, dtype=torch.long)
    with pytest.raises(ValueError, match="does not match"):
        decode_logprobs_payload(_payload(values, indices, 2), expected_format_version=3)
    with pytest.raises(ValueError, match="Unsupported"):
        decode_logprobs_payload(_payload(values, indices, 4))


def test_saver_rejects_appending_to_an_old_cache(monkeypatch):
    for name in (
        "get_tensor_model_parallel_rank",
        "get_tensor_model_parallel_world_size",
        "get_tensor_model_parallel_group",
        "get_context_parallel_rank",
        "get_context_parallel_world_size",
        "get_context_parallel_group",
        "get_data_parallel_rank",
        "get_data_parallel_world_size",
        "get_tensor_model_parallel_src_rank",
    ):
        monkeypatch.setattr(logits_saver.parallel_state, name, lambda: 0, raising=False)
    monkeypatch.setattr(logits_saver.dist, "get_global_rank", lambda *args: 0)
    monkeypatch.setattr(logits_saver, "get_args", lambda: SimpleNamespace(mtp_num_layers=0))
    monkeypatch.setattr(logits_saver, "compute_dataset_hash", lambda: ("hash", {}))
    monkeypatch.setattr(
        logits_saver, "peek_first_logprobs_metadata", lambda path: {"saver": {"format_version": 2}}
    )
    with pytest.raises(ValueError, match="fresh logits-save-dir"):
        LogitsSaverHooks("unused", 2)


def test_replicated_loss_is_not_reduced_again(monkeypatch):
    wrapper = cached_logits_loss.LossFuncCallable("unused", kd_loss_alpha=1.0)

    class ReplicatedLoss:
        loss_is_tp_replicated = True

        def __call__(self, logits):
            return logits.sum(-1).transpose(0, 1)

    wrapper.kd_func = ReplicatedLoss()
    logits = torch.ones(2, 1, 4, requires_grad=True)
    monkeypatch.setattr(
        cached_logits_loss,
        "get_student_logits_capture",
        lambda: SimpleNamespace(pop=lambda: logits),
    )

    def unexpected_reduce(*args, **kwargs):
        raise AssertionError("The complete reference-logits loss must not be summed across TP.")

    monkeypatch.setattr(cached_logits_loss.dist, "all_reduce", unexpected_reduce)
    loss, _, _ = wrapper(torch.ones(1, 2), torch.zeros(1, 2), SimpleNamespace(training=True))
    loss.backward()
    assert loss.item() == 8
    assert logits.grad.eq(1).all()


def _writer_tp_worker(rank, rendezvous):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        # All four selected entries can live on rank 0, leaving an empty
        # local tail there; selecting all eight tests a globally empty tail.
        teacher = torch.tensor([[[0.0, -1, -2, -3, -30, -31, -32, -33]]])
        for k in (1, 4, 8):
            for p in (None, 0.8):
                saver = _bare_saver(k, p)
                saver.tp_rank = rank
                saver.tp_size = 2
                saver.tp_group = dist.group.WORLD
                saver._tp_dst_rank_global = 0
                result = saver._process_single_microbatch(teacher.chunk(2, -1)[rank])
                if rank == 0:
                    expected = _bare_saver(k, p)._process_single_microbatch(teacher)
                    torch.testing.assert_close(result[0], expected[0], atol=0, rtol=0)
                    torch.testing.assert_close(result[1], expected[1], atol=0, rtol=0)
                else:
                    assert result is None
    finally:
        dist.destroy_process_group()


def test_tensor_parallel_saver_matches_unsharded_saver(tmp_path):
    if dist.is_initialized():
        pytest.skip("Standalone Gloo test; model-parallel CI already owns the process group.")
    mp.spawn(
        _writer_tp_worker, args=(f"file://{tmp_path / 'writer_rendezvous'}",), nprocs=2, join=True
    )
