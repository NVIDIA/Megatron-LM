# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import gc
import logging
import os
from contextlib import nullcontext
from typing import Dict, List, Optional, Tuple
from unittest.mock import Mock

import pytest
import torch

from megatron.core._rank_utils import safe_get_rank
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.pipeline_parallel.fine_grained_activation_offload import ChunkOffloadHandler
from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
    FineGrainedActivationOffloadingInterface as off_interface,
)
from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
    OffloadTensorGroup,
    OffloadTensorPool,
    PipelineOffloadManager,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.transformer_config import MLATransformerConfig, TransformerConfig
from megatron.core.utils import is_te_min_version
from tests.unit_tests.test_utilities import Utils

# Tolerance for memory expectation check (GPU allocator jitter etc).
EPSILON = 0.30
EPSILON_A2A = 0.30
DELTA = 20  # MiB


def _reset_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _make_chunk_handler_for_offload_checker(min_offloaded_tensor_size: int = 1):
    handler = ChunkOffloadHandler.__new__(ChunkOffloadHandler)
    handler.min_offloaded_tensor_size = min_offloaded_tensor_size
    return handler


def test_offload_summary_uses_explicit_process_group(monkeypatch):
    from megatron.core.pipeline_parallel import fine_grained_activation_offload as off_module

    process_group = object()
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(
        torch.distributed,
        "get_rank",
        lambda *, group=None: 0 if group is process_group else pytest.fail("wrong process group"),
    )
    monkeypatch.setattr(
        torch.distributed,
        "get_world_size",
        lambda *, group=None: 1 if group is process_group else pytest.fail("wrong process group"),
    )

    def all_gather_object(output, value, *, group=None):
        assert group is process_group
        output[0] = value

    def barrier(*, group=None):
        assert group is process_group

    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather_object)
    monkeypatch.setattr(torch.distributed, "barrier", barrier)
    warning_messages = []
    monkeypatch.setattr(off_module.logger, "warning", warning_messages.append)

    off_module.print_offload_summary_table(
        {"decoder": 1024}, {"decoder": (1, 1024)}, process_group=process_group
    )

    assert len(warning_messages) == 1
    assert "same tensor storage" in warning_messages[0]


def test_chunk_offload_handler_skips_non_offloadable_tensor_types():
    handler = _make_chunk_handler_for_offload_checker()

    cpu_tensor = torch.empty(1024)
    assert not handler.tensor_need_offloading_checker(cpu_tensor)
    assert handler.tensor_push(cpu_tensor) is cpu_tensor
    assert handler.tensor_pop(cpu_tensor) is cpu_tensor

    parameter = torch.nn.Parameter(torch.empty(1024))
    assert not handler.tensor_need_offloading_checker(parameter)
    assert handler.tensor_push(parameter) is parameter
    assert handler.tensor_pop(parameter) is parameter

    try:
        from torch._subclasses.fake_tensor import FakeTensorMode
    except ImportError:
        pytest.skip("FakeTensorMode is not available in this PyTorch version.")

    with FakeTensorMode():
        fake_tensor = torch.empty(1024, device="cuda")
    assert not handler.tensor_need_offloading_checker(fake_tensor)
    assert handler.tensor_push(fake_tensor) is fake_tensor
    assert handler.tensor_pop(fake_tensor) is fake_tensor


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for offload check.")
def test_chunk_offload_handler_respects_tensor_opt_out_flags():
    handler = _make_chunk_handler_for_offload_checker()

    tensor = torch.empty(1024, device="cuda")
    assert handler.tensor_need_offloading_checker(tensor)

    tensor._TE_do_not_offload = True
    assert not handler.tensor_need_offloading_checker(tensor)


def _make_chunk_handler_for_offload_reload():
    handler = ChunkOffloadHandler.__new__(ChunkOffloadHandler)
    handler.cpu_tensor_pool = OffloadTensorPool(device="cpu", pin_memory=True)
    return handler


def test_chunk_offload_handler_reload_uses_default_stream_allocation(monkeypatch):
    from megatron.core.pipeline_parallel import fine_grained_activation_offload as offload

    handler = _make_chunk_handler_for_offload_reload()
    cpu_backup = Mock()
    cpu_backup.is_pinned.return_value = True
    cpu_backup.size.return_value = (16,)
    cpu_backup.dtype = torch.float32
    cpu_backup.layout = torch.strided
    gpu_tensor = Mock()
    consumer_stream = Mock()
    allocation_context = Mock(return_value=nullcontext())
    empty = Mock(return_value=gpu_tensor)

    monkeypatch.setattr(offload, "default_stream_allocation", allocation_context)
    monkeypatch.setattr(torch.cuda, "current_stream", Mock(return_value=consumer_stream))
    monkeypatch.setattr(torch, "empty", empty)

    state = (torch.device("cuda"), cpu_backup, False, None)
    assert handler.reload(state) is gpu_tensor

    allocation_context.assert_called_once_with()
    gpu_tensor.record_stream.assert_called_once_with(consumer_stream)
    gpu_tensor.copy_.assert_called_once_with(cpu_backup, non_blocking=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for offload check.")
def test_chunk_offload_handler_offloads_base_storage_for_covering_views():
    handler = _make_chunk_handler_for_offload_reload()

    base = torch.randn(64, 96, device="cuda")
    view = base[:, :80]  # non-contiguous last-dim slice covering >50% of the storage
    assert not view.is_contiguous()

    state = handler.offload(view)
    _, cpu_backup, _, view_meta = state
    # The full flat storage is offloaded, not a gathered copy of the view.
    assert cpu_backup.shape == (base.numel(),)
    assert view_meta == (view.size(), view.stride(), view.storage_offset())

    reloaded = handler.reload(state)
    torch.cuda.synchronize()
    assert reloaded.shape == view.shape
    assert reloaded.stride() == view.stride()
    assert torch.equal(reloaded, view)
    # The pool backup was returned after the reload.
    assert handler.cpu_tensor_pool._stats["current_in_use"] == 0

    # Autograd hands the saved-tensor hooks a detach()-ed alias (e.g. the
    # checkpoint's saved input), which has no ._base; the storage-based
    # resolution must still take the base path.
    detached = base[:, :80].detach()
    state_d = handler.offload(detached)
    assert state_d[3] is not None
    reloaded_d = handler.reload(state_d)
    torch.cuda.synchronize()
    assert torch.equal(reloaded_d, detached)
    assert handler.cpu_tensor_pool._stats["current_in_use"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for offload check.")
def test_chunk_offload_handler_gathers_low_coverage_views():
    handler = _make_chunk_handler_for_offload_reload()

    base = torch.randn(64, 96, device="cuda")
    view = base[:, :8]  # covers well under BASE_OFFLOAD_MIN_COVERAGE of base

    state = handler.offload(view)
    _, cpu_backup, _, view_meta = state
    # Low-coverage views fall back to the contiguous gather.
    assert view_meta is None
    assert cpu_backup.shape == view.shape

    reloaded = handler.reload(state)
    torch.cuda.synchronize()
    assert reloaded.is_contiguous()
    assert torch.equal(reloaded, view)


def _make_warmup_chunk(groups: List["OffloadTensorGroup"]) -> ChunkOffloadHandler:
    """Build a warmup chunk handler that already owns ``groups``."""
    handler = ChunkOffloadHandler(
        min_offloaded_tensor_size=1,
        cpu_tensor_pool=OffloadTensorPool(device="cpu", pin_memory=True),
    )
    handler.offload_groups = list(groups)
    handler._max_group_size = len(groups)
    return handler


def _run_post_warmup_callback(chunk: ChunkOffloadHandler) -> None:
    """Drive post_warmup_callback over a single hand-built chunk."""
    manager = PipelineOffloadManager.__new__(PipelineOffloadManager)
    manager._is_warmup = True
    manager._cached_chunks_forward = [chunk]
    manager._cached_chunks_backward = [chunk]
    manager._offload_margin = 0
    manager._pp_rank = 0
    manager._delta_offload_bytes_across_pp_ranks = 0
    manager._activation_offload_fraction = 1.0
    manager.post_warmup_callback()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for offload check.")
@pytest.mark.skipif(
    Utils.world_size < 2, reason="Rank-0 duplicate aggregation requires at least two ranks."
)
def test_post_warmup_callback_warns_on_duplicated_storage_offload(caplog, monkeypatch):
    from megatron.core.pipeline_parallel import fine_grained_activation_offload as off_module

    # post_warmup_callback gathers the per-rank duplicate summaries, so this test needs a
    # live process group. The skipif above reads the launcher's world size rather than
    # torch.distributed.get_world_size(), which raises when no group exists yet.
    Utils.initialize_model_parallel()

    try:
        rank = safe_get_rank()
        duplicate_rank = 1
        base = torch.randn(1024, 512, device="cuda")
        storage_bytes = base.numel() * base.element_size()
        # Two strided views that each cover >50% of the shared storage, so both take
        # the whole-storage offload path and copy the same bytes to CPU twice. Only
        # rank 1 creates this duplication, exercising the rank-0 aggregation path.
        views = (base[:, :400], base[:, 112:]) if rank == duplicate_rank else (base[:, :400],)
        assert all(not view.is_contiguous() for view in views)

        # post_warmup_callback disables the last group of each name (offload margin),
        # so build two same-named groups and assert on the surviving one.
        groups = []
        for group_idx in range(2):
            group = OffloadTensorGroup("core_attn")
            for tensor_idx, view in enumerate(views):
                group.push_tensor((group_idx, tensor_idx), view)
            groups.append(group)

        chunk = _make_warmup_chunk(groups)
        for group in groups:
            chunk.bulk_offload_group(group)
        torch.cuda.synchronize()

        # The second view of each group on rank 1 is redundant; other ranks have no
        # local duplication to report.
        for group in groups:
            assert group.duplicate_storage_tensor_count == int(rank == duplicate_rank)
            assert group.duplicate_storage_bytes == (storage_bytes if rank == duplicate_rank else 0)

        real_all_gather_object = torch.distributed.all_gather_object
        all_gather_object_calls = 0

        def counting_all_gather_object(*args, **kwargs):
            nonlocal all_gather_object_calls
            all_gather_object_calls += 1
            return real_all_gather_object(*args, **kwargs)

        monkeypatch.setattr(torch.distributed, "all_gather_object", counting_all_gather_object)
        with caplog.at_level(logging.WARNING, logger=off_module.__name__):
            _run_post_warmup_callback(chunk)

        assert all_gather_object_calls == 1

        # The margin rule keeps only the first group offloaded. Rank 0 gathers and
        # reports the duplication even though it occurred only on rank 1.
        if rank == 0:
            assert "copying the same tensor storage to CPU more than once" in caplog.text
            assert (
                f"rank {duplicate_rank}, core_attn: 1 redundant copies, "
                f"{storage_bytes / (1024 * 1024):.2f} MB" in caplog.text
            )
            assert "Offloading proceeds with the duplicated copies." in caplog.text
        else:
            assert "copying the same tensor storage to CPU more than once" not in caplog.text
        # The duplicated offloading is diagnosed, not suppressed.
        assert groups[0].offload
        assert groups[0].total_tensor_count == len(views)
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for offload check.")
def test_bulk_offload_group_flags_disjoint_views_taking_the_whole_storage_path():
    base = torch.randn(1024, 512, device="cuda")
    storage_bytes = base.numel() * base.element_size()
    # Disjoint halves, but each covers exactly BASE_OFFLOAD_MIN_COVERAGE of the
    # storage, so both are offloaded as the full buffer: the views not sharing a
    # single element does not make the second copy any less redundant.
    group = OffloadTensorGroup("core_attn")
    group.push_tensor((1, 0), base[:, :256])
    group.push_tensor((1, 1), base[:, 256:])

    chunk = _make_warmup_chunk([group])
    chunk.bulk_offload_group(group)
    torch.cuda.synchronize()

    assert all(state[3] is not None for state in group._tensors.values())
    assert group.total_offload_bytes == 2 * storage_bytes
    assert group.duplicate_storage_tensor_count == 1
    assert group.duplicate_storage_bytes == storage_bytes


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for offload check.")
@pytest.mark.parametrize("full_storage_first", [False, True])
def test_bulk_offload_group_counts_full_storage_and_logical_view_duplicates(full_storage_first):
    base = torch.randn(1024, 512, device="cuda")
    storage_bytes = base.numel() * base.element_size()
    logical_view = base[:128]
    logical_view_bytes = logical_view.numel() * logical_view.element_size()
    full_storage_view = base[:, :400]
    tensors = (
        (full_storage_view, logical_view)
        if full_storage_first
        else (logical_view, full_storage_view)
    )

    group = OffloadTensorGroup("core_attn")
    for tensor_idx, tensor in enumerate(tensors):
        group.push_tensor((1, tensor_idx), tensor)

    chunk = _make_warmup_chunk([group])
    chunk.bulk_offload_group(group)
    torch.cuda.synchronize()

    assert group.total_offload_bytes == storage_bytes + logical_view_bytes
    assert group.duplicate_storage_tensor_count == 1
    assert group.duplicate_storage_bytes == logical_view_bytes


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for offload check.")
def test_bulk_offload_group_does_not_flag_views_copying_their_own_bytes(caplog, monkeypatch):
    from megatron.core.pipeline_parallel import fine_grained_activation_offload as off_module

    base = torch.randn(1024, 512, device="cuda")
    storage_bytes = base.numel() * base.element_size()
    # Disjoint contiguous slices copy their own bytes, together exactly the
    # storage once.
    group = OffloadTensorGroup("core_attn")
    group.push_tensor((1, 0), base[:512])
    group.push_tensor((1, 1), base[512:])
    # Low-coverage strided views are gathered, so they too copy only their own
    # elements even though the gathered spans interleave in the storage.
    other_group = OffloadTensorGroup("core_attn")
    other_group.push_tensor((2, 0), base[:, :8])
    other_group.push_tensor((2, 1), base[:, 8:16])

    chunk = _make_warmup_chunk([group, other_group])
    chunk.bulk_offload_group(group)
    chunk.bulk_offload_group(other_group)
    torch.cuda.synchronize()

    assert group.total_offload_bytes == storage_bytes
    for offload_group in (group, other_group):
        assert all(state[3] is None for state in offload_group._tensors.values())
        assert offload_group.duplicate_storage_tensor_count == 0
        assert offload_group.duplicate_storage_bytes == 0

    monkeypatch.setattr(off_module, "print_offload_summary_table", lambda *args, **kwargs: None)
    with caplog.at_level(logging.WARNING, logger=off_module.__name__):
        _run_post_warmup_callback(chunk)
    assert "more than once" not in caplog.text


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for offload check.")
def test_bulk_offload_group_does_not_analyze_logical_view_overlap():
    base = torch.randn(1024, 512, device="cuda")
    half_bytes = base[:512].numel() * base.element_size()
    # General overlap analysis for logical views is intentionally out of scope;
    # duplicate-byte accounting is exact only when the storage is copied in full.
    group = OffloadTensorGroup("core_attn")
    group.push_tensor((1, 0), base[:512])
    group.push_tensor((1, 1), base[:512])

    chunk = _make_warmup_chunk([group])
    chunk.bulk_offload_group(group)
    torch.cuda.synchronize()

    assert group.total_offload_bytes == 2 * half_bytes
    assert group.duplicate_storage_tensor_count == 0
    assert group.duplicate_storage_bytes == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for offload check.")
def test_duplicate_storage_accounting_is_scoped_to_one_group():
    base = torch.randn(1024, 512, device="cuda")
    groups = []
    for group_idx, view in enumerate((base[:, :400], base[:, 112:])):
        group = OffloadTensorGroup("core_attn")
        group.push_tensor((group_idx, 0), view)
        groups.append(group)

    chunk = _make_warmup_chunk(groups)
    for group in groups:
        chunk.bulk_offload_group(group)
    torch.cuda.synchronize()

    for group in groups:
        assert group.duplicate_storage_tensor_count == 0
        assert group.duplicate_storage_bytes == 0


def _build_gpt_model(
    *,
    seed: int,
    num_layers: int,
    hidden_size: int,
    num_attention_heads: int,
    vocab_size: int,
    seq_length: int,
    num_experts: Optional[int],
    fine_grained_activation_offloading: bool,
    offload_modules: Optional[List[str]],
    min_offloaded_tensor_size: int,
    is_mla: bool,
) -> GPTModel:
    """Build a GPTModel that uses TE-based transformer layer spec."""
    model_parallel_cuda_manual_seed(seed)
    torch.manual_seed(seed)
    ConfigClass = MLATransformerConfig if is_mla else TransformerConfig
    transformer_config = ConfigClass(
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        use_cpu_initialization=True,
        attention_backend=AttnBackend.unfused,
        bf16=True,
        # Recompute
        recompute_modules=["layernorm", "moe_act"] if num_experts is not None else ["layernorm"],
        recompute_granularity="selective",
        # MoE
        num_moe_experts=num_experts,
        moe_grouped_gemm=(num_experts is not None),
        # Fine-grained activation offloading
        fine_grained_activation_offloading=fine_grained_activation_offloading,
        offload_modules=offload_modules,
        min_offloaded_tensor_size=min_offloaded_tensor_size,
    )
    gpt_model = GPTModel(
        config=transformer_config,
        transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(
            num_experts=num_experts,
            moe_grouped_gemm=num_experts is not None,
            multi_latent_attention=is_mla,
        ),
        vocab_size=vocab_size,
        max_sequence_length=seq_length,
    ).bfloat16()
    return gpt_model


def _make_gpt_inputs(
    *, seq_length: int, micro_batch_size: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    data = list(range(seq_length))
    input_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).to(device)
    position_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).to(device)
    attention_mask = torch.ones((micro_batch_size, 1, seq_length, seq_length), dtype=bool).to(
        device
    )
    return input_ids, position_ids, attention_mask


def _capture_params(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    params: Dict[str, torch.Tensor] = {}
    for name, p in model.named_parameters():
        params[name] = p.detach().cpu().clone()
    return params


def _restore_params(model: torch.nn.Module, params: Dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.copy_(params[name].to(device=p.device, dtype=p.dtype))


def _run_one_iter_and_capture(
    model: GPTModel,
    *,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    enable_offload_reset: bool,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], int]:
    """
    Run a single forward+backward iteration.

    Returns:
      - logits (CPU float32)
      - selected grads (CPU float32)
      - peak_memory_allocated (bytes) during the iteration
    """

    if enable_offload_reset:
        off_interface.reset()

    # Keep warmup-created grad buffers resident so the peak-memory check still
    # compares the steady-state allocator footprint, but remove accumulated
    # warmup values before capturing correctness grads.
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()

    torch.cuda.reset_peak_memory_stats()
    logits = model(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask)
    loss = logits.float().sum()
    loss.backward()
    torch.cuda.synchronize()
    peak_bytes = int(torch.cuda.max_memory_allocated())

    # capture all gradients for correctness
    grads: Dict[str, torch.Tensor] = {}
    for name, p in model.named_parameters():
        grads[name] = p.grad.detach().float().cpu() if p.grad is not None else None

    return logits.detach().float().cpu(), grads, peak_bytes


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for offloading tests.")
@pytest.mark.parametrize(
    "is_moe, is_mla, offload_modules",
    [
        # Dense GPT modules
        (False, True, ["attn_norm"]),
        (True, False, ["qkv_linear"]),
        (True, False, ["core_attn"]),
        # # attn_proj depends on core_attn (validated in TransformerConfig.__post_init__)
        (True, True, ["core_attn", "attn_proj"]),
        (True, False, ["mlp_norm"]),
        (True, False, ["expert_fc1"]),
        (True, False, ["moe_act"]),
    ],
)
def test_gpt_fine_grained_activation_offloading_correctness_and_memory(
    is_moe: bool, is_mla: bool, offload_modules: List[str]
):
    """
    Initialize a GPTModel and verify:
    - forward output correctness under each offload_modules setting
    - backward gradient correctness (subset)
    - peak GPU memory is reduced roughly as expected (based on recorded offload bytes)
    """
    # setup distributed/model-parallel (same pattern as other UTs)
    os.environ.pop("NVTE_FUSED_ATTN", None)
    os.environ.pop("NVTE_FLASH_ATTN", None)
    os.environ.pop("NVTE_UNFUSED_ATTN", None)
    # os.environ["NVTE_FLASH_ATTN"] = "1"
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)

    seed = 123
    # Choose shapes large enough to make memory deltas stable but still fast.
    num_experts = 4 if is_moe else None
    num_layers = 8
    hidden_size = 2048 if num_experts is None else 1024
    num_attention_heads = 16 if hidden_size >= 2048 else 8
    vocab_size = 1024
    seq_length = 1024
    micro_batch_size = 2
    device = torch.device("cuda")

    input_ids, position_ids, attention_mask = _make_gpt_inputs(
        seq_length=seq_length, micro_batch_size=micro_batch_size, device=device
    )

    from megatron.core.pipeline_parallel import fine_grained_activation_offload as off

    off_interface.reset_instance()

    try:
        # 1) Baseline run (no offloading)
        _reset_cuda_memory()
        base_model = _build_gpt_model(
            seed=seed,
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            vocab_size=vocab_size,
            seq_length=seq_length,
            num_experts=num_experts,
            fine_grained_activation_offloading=False,
            offload_modules=None,
            min_offloaded_tensor_size=1024 * 1024,
            is_mla=is_mla,
        ).cuda()
        base_model.train()
        base_params = _capture_params(base_model)

        # Warmup baseline once for allocator stability
        _run_one_iter_and_capture(
            base_model,
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            enable_offload_reset=False,
        )
        _reset_cuda_memory()
        base_logits, base_grads, base_peak = _run_one_iter_and_capture(
            base_model,
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            enable_offload_reset=False,
        )
        # Free baseline model GPU memory before offload path
        del base_model
        _reset_cuda_memory()

        # 2) Offload run (warmup to record bytes + steady-state measurement)
        off_model = _build_gpt_model(
            seed=seed,
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            vocab_size=vocab_size,
            seq_length=seq_length,
            num_experts=num_experts,
            fine_grained_activation_offloading=True,
            offload_modules=offload_modules,
            min_offloaded_tensor_size=1024,  # force offloading for UT determinism
            is_mla=is_mla,
        ).cuda()
        _restore_params(off_model, base_params)
        off_model.train()

        # Warmup 1 iter to populate cached chunks, then reset to finish warmup bookkeeping.
        _run_one_iter_and_capture(
            off_model,
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            enable_offload_reset=True,
        )
        # Reset once more to trigger post_warmup_callback and apply steady-state offload decisions.
        off_interface.reset()

        from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
            PipelineOffloadManager,
        )

        mgr = PipelineOffloadManager.get_instance()
        expected_offload_bytes = int(
            sum(mgr.offload_summary_bytes.get(k, 0) for k in offload_modules)
        )
        expected_offload_mib = expected_offload_bytes / (1024**2)

        _reset_cuda_memory()
        off_logits, off_grads, off_peak = _run_one_iter_and_capture(
            off_model,
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            enable_offload_reset=True,
        )
        del off_model
        _reset_cuda_memory()

        # 3) Correctness checks (forward + selected grads)
        assert torch.allclose(off_logits, base_logits, rtol=1e-3, atol=1e-3)
        assert set(off_grads.keys()) == set(base_grads.keys())
        for name, gb in base_grads.items():
            go = off_grads[name]
            if gb is None or go is None:
                assert gb is None and go is None, f"Grad None mismatch for {name}"
                continue
            assert torch.allclose(go, gb, rtol=1e-3, atol=1e-3), f"Grad mismatch for {name}"

        # 4) Memory checks (peak allocated over forward+backward)
        saved_mib = (base_peak - off_peak) / (1024**2)
        assert saved_mib > 0.0, (
            f"Expected GPU peak memory reduction for offload_modules={offload_modules}, "
            f"but got saved={saved_mib:.2f}MiB (base={base_peak/(1024**2):.2f}MiB, "
            f"off={off_peak/(1024**2):.2f}MiB)"
        )

        # If expectation is large enough, enforce approximate match.
        # For tiny expectations, allocator noise may dominate; we only require a positive reduction.
        if expected_offload_mib >= 2.0:
            rel_err = abs(saved_mib - expected_offload_mib) / max(expected_offload_mib, 1e-6)
            abs_err = abs(saved_mib - expected_offload_mib)
            assert rel_err <= EPSILON and abs_err <= DELTA, (
                f"Memory saving mismatch for offload_modules={offload_modules}: "
                f"saved={saved_mib:.2f}MiB expected~={expected_offload_mib:.2f}MiB "
                f"(rel_err={rel_err:.2f}, abs_err={abs_err:.2f})"
            )
            print(
                f"Rank {torch.distributed.get_rank()}: Saved {saved_mib:.2f}MiB, expected {expected_offload_mib:.2f}MiB"
            )
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.flaky_in_dev
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for offloading tests.")
@pytest.mark.skipif(
    not is_te_min_version("1.9.0.dev0"),
    reason="EP A2A overlap requires TE 1.9.0.dev0+ in this repo's tests.",
)
@pytest.mark.parametrize(
    "dispatcher_backend, is_mla, offload_modules",
    [
        ("alltoall", True, ["attn_norm"]),
        ("alltoall", True, ["core_attn"]),
        ("alltoall", True, ["attn_norm", "core_attn", "attn_proj"]),
        ("alltoall", True, ["mlp_norm"]),
        ("alltoall", False, ["expert_fc1"]),
        ("alltoall", False, ["moe_act"]),
        (
            "alltoall",
            True,
            ["attn_norm", "core_attn", "attn_proj", "mlp_norm", "expert_fc1", "moe_act"],
        ),
        (
            "alltoall",
            False,
            ["attn_norm", "core_attn", "attn_proj", "mlp_norm", "expert_fc1", "moe_act"],
        ),
    ],
)
def test_fine_grained_activation_offload_with_ep_a2a_overlap_compatibility(
    dispatcher_backend: str, is_mla: bool, offload_modules: List[str]
):
    """
    Compatibility test for:
      - fine-grained activation offloading
      - EP all-to-all overlap (overlap_moe_expert_parallel_comm)
      - memory saving roughly matches expected offload bytes (when expectation is large enough)

    The EP A2A overlap initialization pattern is aligned with
    `tests/unit_tests/a2a_overlap/test_schedule_chunk_1f1b.py`.
    """
    from megatron.core.models.common.model_chunk_schedule_plan import (
        TransformerModelChunkSchedulePlan,
    )
    from megatron.core.pipeline_parallel.utils import set_streams
    from tests.unit_tests.a2a_overlap.utils import deterministic_mode

    # EP overlap requires distributed initialization with EP groups.
    ep_size = 4
    if Utils.world_size % ep_size != 0:
        pytest.skip(
            f"Skipping: WORLD_SIZE={Utils.world_size} must be divisible by ep_size={ep_size}."
        )

    seed = 123
    num_experts = 8  # must be divisible by ep_size
    if num_experts % ep_size != 0:
        pytest.skip(
            f"Skipping: num_moe_experts={num_experts} must be divisible by ep_size={ep_size}."
        )

    # Small shapes to keep this compatibility test fast.
    num_layers = 8
    hidden_size = 1024
    num_attention_heads = 16
    vocab_size = 1024
    seq_length = 1024
    micro_batch_size = 2
    device = torch.device("cuda")

    from megatron.core.pipeline_parallel import fine_grained_activation_offload as off

    def _make_schedule_inputs() -> Dict[str, torch.Tensor]:
        data = list(range(seq_length))
        input_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).to(device)
        position_ids = (
            torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).to(device)
        )
        attention_mask = torch.ones((micro_batch_size, 1, seq_length, seq_length), dtype=bool).to(
            device
        )
        labels = input_ids.clone()
        return {
            "input_ids": input_ids,
            "labels": labels,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
        }

    def _capture_params(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
        params: Dict[str, torch.Tensor] = {}
        for name, p in model.named_parameters():
            params[name] = p.detach().clone()
        return params

    def _restore_params(model: torch.nn.Module, params: Dict[str, torch.Tensor]) -> None:
        for name, p in model.named_parameters():
            p.data.copy_(params[name])

    def _build_overlap_moe_gpt(
        *, enable_offload: bool, is_mla: bool, dispatcher_backend: str
    ) -> GPTModel:
        model_parallel_cuda_manual_seed(seed)
        torch.manual_seed(seed)
        ConfigClass = MLATransformerConfig if is_mla else TransformerConfig
        transformer_config = ConfigClass(
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            use_cpu_initialization=True,
            attention_backend=AttnBackend.unfused,
            # Recompute
            recompute_modules=["layernorm", "moe_act"],
            recompute_granularity="selective",
            bf16=True,
            # MoE + EP overlap
            num_moe_experts=num_experts,
            moe_grouped_gemm=True,
            expert_model_parallel_size=ep_size,
            moe_token_dispatcher_type="alltoall" if dispatcher_backend == "alltoall" else "flex",
            moe_flex_dispatcher_backend=dispatcher_backend,
            moe_router_dtype="fp32" if dispatcher_backend == "hybridep" else "fp64",
            overlap_moe_expert_parallel_comm=True,
            delay_wgrad_compute=True,
            # Fine-grained activation offloading
            fine_grained_activation_offloading=enable_offload,
            offload_modules=offload_modules if enable_offload else None,
            min_offloaded_tensor_size=1024,  # force offloading to exercise the code path
        )
        return (
            GPTModel(
                config=transformer_config,
                transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(
                    num_experts=num_experts, moe_grouped_gemm=True, multi_latent_attention=is_mla
                ),
                vocab_size=vocab_size,
                max_sequence_length=seq_length,
            )
            .bfloat16()
            .cuda()
        )

    def _run_schedule_1f1b_two_microbatches(
        model: GPTModel, *, enable_offload_reset: bool
    ) -> Tuple[List[torch.Tensor], Dict[str, torch.Tensor], int]:
        """
        Run a minimal 1F1B schedule (2 microbatches) using ModelChunkSchedulePlan.run().
        This is the execution path that exercises EP A2A overlap scheduling.
        """
        if enable_offload_reset:
            off_interface.reset()

        # Keep warmup-created grad buffers resident for stable peak-memory comparisons,
        # but clear warmup values before capturing correctness grads.
        for p in model.parameters():
            if p.grad is not None:
                p.grad.zero_()

        data0 = _make_schedule_inputs()
        data1 = _make_schedule_inputs()
        plan0 = model.build_schedule_plan(**data0)

        torch.cuda.reset_peak_memory_stats()
        out0 = TransformerModelChunkSchedulePlan.run(plan0, None)
        plan1 = model.build_schedule_plan(**data1)
        out1 = TransformerModelChunkSchedulePlan.run(plan1, plan0, b_grad=torch.ones_like(out0))
        TransformerModelChunkSchedulePlan.run(None, plan1, b_grad=torch.ones_like(out1))
        torch.cuda.synchronize()
        peak_bytes = int(torch.cuda.max_memory_allocated())

        # capture outputs and grads
        outputs = [out0.detach().float().cpu(), out1.detach().float().cpu()]
        grads: Dict[str, torch.Tensor] = {}
        for name, p in model.named_parameters():
            grads[name] = p.grad.detach().float().cpu() if p.grad is not None else None
        return outputs, grads, peak_bytes

    # setup distributed/model-parallel
    os.environ.pop("NVTE_FUSED_ATTN", None)
    os.environ.pop("NVTE_FLASH_ATTN", None)
    os.environ.pop("NVTE_UNFUSED_ATTN", None)

    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=ep_size,
    )
    set_streams()

    off_interface.reset_instance()

    try:
        with deterministic_mode():
            # Baseline: EP overlap on, offload off.
            _reset_cuda_memory()
            base_model = _build_overlap_moe_gpt(
                enable_offload=False, is_mla=is_mla, dispatcher_backend=dispatcher_backend
            )
            base_model.train()
            base_params = _capture_params(base_model)
            # Warmup once for allocator stability / graph caching
            _run_schedule_1f1b_two_microbatches(base_model, enable_offload_reset=False)
            _reset_cuda_memory()
            base_outs, base_grads, base_peak = _run_schedule_1f1b_two_microbatches(
                base_model, enable_offload_reset=False
            )
            del base_model
            _reset_cuda_memory()

            # Offload: EP overlap on, fine-grained offload on.
            off_model = _build_overlap_moe_gpt(
                enable_offload=True, is_mla=is_mla, dispatcher_backend=dispatcher_backend
            )
            _restore_params(off_model, base_params)
            off_model.train()
            # Warmup once to populate cached chunks, then reset to apply steady-state offload decisions.
            off_interface.reset()
            _run_schedule_1f1b_two_microbatches(off_model, enable_offload_reset=False)
            off_interface.reset()
            from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
                PipelineOffloadManager,
            )

            mgr = PipelineOffloadManager.get_instance()
            expected_offload_bytes = int(
                sum(mgr.offload_summary_bytes.get(k, 0) for k in offload_modules)
            )
            expected_offload_mib = expected_offload_bytes / (1024**2)

            _reset_cuda_memory()
            off_outs, off_grads, off_peak = _run_schedule_1f1b_two_microbatches(
                off_model, enable_offload_reset=True
            )
            del off_model
            _reset_cuda_memory()

            # Correctness (forward outputs + all grads)
            assert len(off_outs) == len(base_outs) == 2
            for i in range(2):
                assert torch.allclose(off_outs[i], base_outs[i], rtol=1e-3, atol=1e-3)
            assert set(off_grads.keys()) == set(base_grads.keys())
            for name, gb in base_grads.items():
                go = off_grads[name]
                if gb is None or go is None:
                    assert gb is None and go is None, f"Grad None mismatch for {name}"
                    continue
                assert torch.allclose(
                    go, gb, rtol=1e-3, atol=1e-3
                ), f"Rank {torch.distributed.get_rank()}: Grad mismatch for {name}"

            # Memory checks (peak allocated during the scheduled 1F1B run)
            saved_mib = (base_peak - off_peak) / (1024**2)
            assert saved_mib > 0.0, (
                f"Expected GPU peak memory reduction for offload_modules={offload_modules}, "
                f"but got saved={saved_mib:.2f}MiB (base={base_peak/(1024**2):.2f}MiB, "
                f"off={off_peak/(1024**2):.2f}MiB)"
            )
            # If expectation is large enough, enforce approximate match.
            if expected_offload_mib >= 2.0:
                rel_err = abs(saved_mib - expected_offload_mib) / max(expected_offload_mib, 1e-6)
                abs_err = abs(saved_mib - expected_offload_mib)
                print(
                    f"Rank {torch.distributed.get_rank()}: Saved {saved_mib:.2f}MiB, expected {expected_offload_mib:.2f}MiB"
                )
                if abs_err > DELTA:
                    assert rel_err <= EPSILON_A2A, (
                        f"Memory saving mismatch for offload_modules={offload_modules}: "
                        f"saved={saved_mib:.2f}MiB expected~={expected_offload_mib:.2f}MiB "
                        f"(rel_err={rel_err:.2f}, abs_err={abs_err:.2f})"
                    )
    finally:
        Utils.destroy_model_parallel()


# =============================================================================
# CUDA Graph + Fine-grained Activation Offloading Tests
# =============================================================================


def _build_gpt_model_with_cuda_graph(
    *,
    seed: int,
    num_layers: int,
    hidden_size: int,
    num_attention_heads: int,
    vocab_size: int,
    seq_length: int,
    num_experts: Optional[int],
    fine_grained_activation_offloading: bool,
    offload_modules: Optional[List[str]],
    min_offloaded_tensor_size: int,
    is_mla: bool,
    cuda_graph_impl: str,
    cuda_graph_scope: Optional[List[str]],
    cuda_graph_warmup_steps: int,
    delay_offload_until_cuda_graph: bool = False,
    activation_offload_fraction: float = 1.0,
) -> GPTModel:
    """Build a GPTModel with CUDA Graph support and fine-grained activation offloading."""
    model_parallel_cuda_manual_seed(seed)
    torch.manual_seed(seed)
    ConfigClass = MLATransformerConfig if is_mla else TransformerConfig
    transformer_config = ConfigClass(
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        use_cpu_initialization=True,
        attention_backend=AttnBackend.unfused,
        bf16=True,
        # Recompute
        recompute_modules=["layernorm", "moe_act"] if num_experts is not None else ["layernorm"],
        recompute_granularity="selective",
        # MoE
        num_moe_experts=num_experts,
        moe_grouped_gemm=(num_experts is not None),
        # Fine-grained activation offloading
        fine_grained_activation_offloading=fine_grained_activation_offloading,
        offload_modules=offload_modules,
        min_offloaded_tensor_size=min_offloaded_tensor_size,
        delay_offload_until_cuda_graph=delay_offload_until_cuda_graph,
        activation_offload_fraction=activation_offload_fraction,
        # CUDA Graph settings
        cuda_graph_impl=cuda_graph_impl,
        cuda_graph_scope=cuda_graph_scope,
        cuda_graph_warmup_steps=cuda_graph_warmup_steps,
        use_te_rng_tracker=True,
    )
    gpt_model = GPTModel(
        config=transformer_config,
        transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(
            num_experts=num_experts,
            moe_grouped_gemm=num_experts is not None,
            multi_latent_attention=is_mla,
        ),
        vocab_size=vocab_size,
        max_sequence_length=seq_length,
    ).bfloat16()
    return gpt_model


def _run_iters_with_cuda_graph(
    model: GPTModel,
    *,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    num_warmup_iters: int,
    num_measure_iters: int,
    enable_offload_reset: bool,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], int]:
    """
    Run multiple forward+backward iterations with CUDA graph capture.

    Returns:
      - logits from last iteration (CPU float32)
      - selected grads from last iteration (CPU float32)
      - peak_memory_allocated (bytes) during measurement iterations
    """
    from megatron.core.transformer.cuda_graphs import _CudagraphGlobalRecord, delete_cuda_graphs

    if enable_offload_reset:
        off_interface.reset()

    # Warmup iterations (before CUDA graph capture)
    for _ in range(num_warmup_iters):
        if enable_offload_reset:
            off_interface.reset()
        logits = model(
            input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask
        )
        loss = logits.float().sum()
        loss.backward()
        # Zero grads for next iteration
        for p in model.parameters():
            if p.grad is not None:
                p.grad.zero_()

    # Trigger post-warmup offload decisions
    if enable_offload_reset:
        off_interface.reset()

    # Create CUDA graphs after warmup
    _CudagraphGlobalRecord.create_cudagraphs()

    # Measurement iterations (with CUDA graph replay)
    torch.cuda.reset_peak_memory_stats()
    for i in range(num_measure_iters):
        if enable_offload_reset:
            off_interface.reset()
        logits = model(
            input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask
        )
        loss = logits.float().sum()
        loss.backward()
        if i < num_measure_iters - 1:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.zero_()

    torch.cuda.synchronize()
    peak_bytes = int(torch.cuda.max_memory_allocated())

    # Capture grads from last iteration
    grads: Dict[str, torch.Tensor] = {}
    for name, p in model.named_parameters():
        grads[name] = p.grad.detach().float().cpu() if p.grad is not None else None

    # Cleanup CUDA graphs
    delete_cuda_graphs()

    return logits.detach().float().cpu(), grads, peak_bytes


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for offloading tests.")
@pytest.mark.skipif(
    not is_te_min_version("2.14.0"), reason="CUDA Graph with TE RNG tracker requires TE >= 2.13.0"
)
@pytest.mark.parametrize(
    "is_mla, offload_modules, cuda_graph_scope, activation_offload_fraction, delay_offload",
    [
        # MoE model with attention CUDA graph + attn offloading
        (False, ["core_attn", "attn_proj"], ["attn", "moe_router"], 1.0, True),
        (False, ["expert_fc1", "moe_act"], ["attn", "moe_router", "moe_preprocess"], 1.0, True),
        (False, ["core_attn", "attn_proj", "expert_fc1"], ["attn", "moe_router"], 1.0, True),
        (
            False,
            ["core_attn", "attn_proj", "expert_fc1", "moe_act"],
            ["attn", "moe_router"],
            1.0,
            True,
        ),
        (
            False,
            ["core_attn", "expert_fc1", "moe_act"],
            ["attn", "moe_router", "moe_preprocess"],
            1.0,
            True,
        ),
        (
            True,
            ["core_attn", "attn_proj", "expert_fc1", "moe_act"],
            ["attn", "moe_router", "moe_preprocess"],
            1.0,
            True,
        ),
        # Test activation_offload_fraction parameter
        (False, ["core_attn", "attn_proj", "expert_fc1"], ["attn", "moe_router"], 0.0, True),
        (False, ["core_attn", "attn_proj", "expert_fc1"], ["attn", "moe_router"], 0.5, True),
        # Test delay_offload_until_cuda_graph parameter
        (False, ["core_attn", "attn_proj", "expert_fc1"], ["attn", "moe_router"], 1.0, False),
    ],
)
def test_fine_grained_activation_offloading_with_cuda_graph(
    is_mla: bool,
    offload_modules: List[str],
    cuda_graph_scope: List[str],
    activation_offload_fraction: float,
    delay_offload: bool,
):
    """
    Test fine-grained activation offloading combined with CUDA graph capture.

    Verifies:
    - Forward output correctness with CUDA graph + offloading
    - Backward gradient correctness
    - Memory savings from offloading are preserved with CUDA graphs
    - Different activation_offload_fraction values work correctly
    - Both delay_offload_until_cuda_graph=True/False produce correct results
    """
    from megatron.core.tensor_parallel.random import initialize_rng_tracker

    os.environ.pop("NVTE_FUSED_ATTN", None)
    os.environ.pop("NVTE_FLASH_ATTN", None)
    os.environ.pop("NVTE_UNFUSED_ATTN", None)

    initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)

    seed = 123
    num_experts = 4  # Always MoE model
    num_layers = 4  # Smaller for faster test with CUDA graphs
    hidden_size = 1024
    num_attention_heads = 8
    vocab_size = 512
    seq_length = 512
    micro_batch_size = 2
    device = torch.device("cuda")
    cuda_graph_warmup_steps = 3

    input_ids, position_ids, attention_mask = _make_gpt_inputs(
        seq_length=seq_length, micro_batch_size=micro_batch_size, device=device
    )

    off_interface.reset_instance()

    try:
        # 1) Baseline: CUDA graph enabled, offloading disabled
        _reset_cuda_memory()
        base_model = _build_gpt_model_with_cuda_graph(
            seed=seed,
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            vocab_size=vocab_size,
            seq_length=seq_length,
            num_experts=num_experts,
            fine_grained_activation_offloading=False,
            offload_modules=None,
            min_offloaded_tensor_size=1024 * 1024,
            is_mla=is_mla,
            cuda_graph_impl="transformer_engine",
            cuda_graph_scope=cuda_graph_scope,
            cuda_graph_warmup_steps=cuda_graph_warmup_steps,
        ).cuda()
        base_model.train()

        base_logits, base_grads, base_peak = _run_iters_with_cuda_graph(
            base_model,
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            num_warmup_iters=cuda_graph_warmup_steps,
            num_measure_iters=2,
            enable_offload_reset=False,
        )
        del base_model
        _reset_cuda_memory()

        # 2) Test: CUDA graph enabled + offloading enabled
        off_interface.reset_instance()

        off_model = _build_gpt_model_with_cuda_graph(
            seed=seed,
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            vocab_size=vocab_size,
            seq_length=seq_length,
            num_experts=num_experts,
            fine_grained_activation_offloading=True,
            offload_modules=offload_modules,
            min_offloaded_tensor_size=1024,  # Force offloading for determinism
            is_mla=is_mla,
            cuda_graph_impl="transformer_engine",
            cuda_graph_scope=cuda_graph_scope,
            cuda_graph_warmup_steps=cuda_graph_warmup_steps,
            delay_offload_until_cuda_graph=delay_offload,
            activation_offload_fraction=activation_offload_fraction,
        ).cuda()
        off_model.train()

        off_logits, off_grads, off_peak = _run_iters_with_cuda_graph(
            off_model,
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            num_warmup_iters=cuda_graph_warmup_steps,
            num_measure_iters=2,
            enable_offload_reset=True,
        )
        del off_model
        _reset_cuda_memory()

        # 3) Correctness checks
        assert torch.allclose(
            off_logits, base_logits, rtol=1e-2, atol=1e-2
        ), f"Logits mismatch: max_diff={torch.max(torch.abs(off_logits - base_logits))}"
        assert set(off_grads.keys()) == set(base_grads.keys())
        for name, gb in base_grads.items():
            go = off_grads[name]
            if gb is None or go is None:
                assert gb is None and go is None, f"Grad None mismatch for {name}"
                continue
            assert torch.allclose(
                go, gb, rtol=1e-2, atol=1e-2
            ), f"Grad mismatch for {name}: max_diff={torch.max(torch.abs(go - gb))}"

        # 4) Memory checks - offloading should still reduce memory with CUDA graphs
        saved_mib = (base_peak - off_peak) / (1024**2)
        print(
            f"CUDA Graph + Offload test (fraction={activation_offload_fraction}, delay={delay_offload}): "
            f"base_peak={base_peak/(1024**2):.2f}MiB, "
            f"off_peak={off_peak/(1024**2):.2f}MiB, "
            f"saved={saved_mib:.2f}MiB"
        )

        # Basic sanity checks
        assert not torch.isnan(off_logits).any(), "NaN detected in logits"
        assert not torch.isinf(off_logits).any(), "Inf detected in logits"

        # Check gradients are valid
        for name, g in off_grads.items():
            if g is not None:
                assert not torch.isnan(g).any(), f"NaN detected in grad for {name}"
                assert not torch.isinf(g).any(), f"Inf detected in grad for {name}"

        # Note: With CUDA graphs, memory behavior may differ from eager mode.
        # We check that offloading doesn't significantly increase memory.
        # In some cases, graph capture overhead may offset offload savings.
        assert saved_mib >= -DELTA, (
            f"Offloading with CUDA graph significantly increased memory: "
            f"saved={saved_mib:.2f}MiB (negative means increase)"
        )

    finally:
        Utils.destroy_model_parallel()
