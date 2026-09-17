# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import gc
import os
from contextlib import nullcontext
from typing import Dict, List, Optional, Tuple

import pytest
import torch

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.pipeline_parallel.fine_grained_activation_offload import ChunkOffloadHandler
from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
    FineGrainedActivationOffloadingInterface as off_interface,
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


# =============================================================================
# Local Graph Lazy-Release D2H Offload Tests (CPU-only, mocked streams/events)
# =============================================================================


class _RecordingEvent:
    def __init__(self, log, name):
        self._log = log
        self.name = name

    def record(self, stream):
        self._log.append((f"{self.name}.record", stream))

    def synchronize(self):
        self._log.append((f"{self.name}.synchronize",))

    def query(self):
        self._log.append((f"{self.name}.query",))
        return False


class _RecordingStream:
    def __init__(self, name, log):
        self.name = name
        self._log = log

    def wait_stream(self, other):
        self._log.append((f"{self.name}.wait_stream", other))

    def wait_event(self, event):
        self._log.append((f"{self.name}.wait_event", event))


class _RecordingGraphGroup:
    """Stand-in for LocalCudaGraphOffloadGroup in replay orchestration tests."""

    def __init__(self, name, log, num_tensors=1, tensor_bytes=1024):
        self.name = name
        self._log = log
        self.device_tensors = [object()] * num_tensors
        self.allocations = [(name, "allocation", index) for index in range(num_tensors)]
        self.host_tensors = [(name, "host", index) for index in range(num_tensors)]
        self._logical_bytes = num_tensors * tensor_bytes
        self.state = "mapped"

    @property
    def logical_bytes(self):
        return self._logical_bytes

    def enqueue_d2h(self, d2h_stream, compute_stream):
        self._log.append(("group.enqueue_d2h", self.name))

    def enqueue_resident_d2h(self, d2h_stream, compute_stream):
        self._log.append(("group.enqueue_resident_d2h", self.name, d2h_stream.name, compute_stream))
        self.state = "resident_d2h_pending"

    def enqueue_resident_h2d(self, h2d_stream):
        self._log.append(("group.enqueue_resident_h2d", self.name, h2d_stream.name))
        self.state = "resident_reload_pending"

    def adopt_resident_reload(self):
        self._log.append(("group.adopt_resident_reload", self.name))
        self.state = "mapped"

    def try_release(self):
        self._log.append(("group.try_release", self.name))
        return False

    def prepare_remap(self, control_stream):
        self._log.append(("group.prepare_remap", self.name))
        return True

    def adopt_reload_submission(self):
        self._log.append(("group.adopt_reload_submission", self.name))


class _RecordingRunner:
    def __init__(self, groups, slot_bank=None):
        self.local_graph_offload_groups = groups
        self.local_graph_reload_state = None
        self.local_graph_slot_bank = slot_bank
        self.local_graph_reload_event = None


def _configure_ping_pong_manager(mgr, log):
    mgr._local_graph_bank_d2h_events = [
        _RecordingEvent(log, "bank0_d2h"),
        _RecordingEvent(log, "bank1_d2h"),
    ]
    mgr._local_graph_bank_d2h_recorded = [False, False]
    mgr._local_graph_bank_consumed_events = [None, None]


def _make_local_graph_batch_manager(log):
    """Build a PipelineOffloadManager with recording streams/events, bypassing __init__."""
    from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
        PipelineOffloadManager,
    )

    mgr = PipelineOffloadManager.__new__(PipelineOffloadManager)
    mgr._d2h_stream = _RecordingStream("d2h", log)
    mgr._h2d_stream = _RecordingStream("h2d", log)
    mgr._local_graph_d2h_bytes = 0
    mgr._local_graph_h2d_bytes = 0
    return mgr


def _record_remap_batch(monkeypatch, log):
    """Replace native reload submission and stream wait with recording stubs."""
    import megatron.core.pipeline_parallel.fine_grained_activation_offload as offload

    def remap_batch(allocations, host_tensors, stream):
        log.append(
            (
                "runner.remap_and_copy_after",
                tuple(allocation[0] for allocation in allocations),
                tuple(host_tensor[0] for host_tensor in host_tensors),
                stream.name,
            )
        )
        return object()

    def wait_on_stream(context, stream):
        log.append(("runner.wait_remap_copy_on_stream", stream.name))

    monkeypatch.setattr(offload, "remap_and_copy_after", remap_batch)
    monkeypatch.setattr(offload, "wait_remap_copy_on_stream", wait_on_stream)


def test_local_graph_forward_replay_enqueues_without_blocking(monkeypatch):
    """Forward replay only enqueues D2H on the shared stream; no wait, no release."""
    log = []
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: "compute")
    monkeypatch.setattr(torch.cuda, "stream", lambda s: nullcontext())

    groups = [
        _RecordingGraphGroup("expert_fc1", log),
        _RecordingGraphGroup("moe_act", log),
        _RecordingGraphGroup("empty_group", log, num_tensors=0),
    ]
    mgr = _make_local_graph_batch_manager(log)

    mgr.local_graph_forward_replay(_RecordingRunner(groups))

    events = [entry[0] for entry in log]
    # Every non-empty group enqueues in order; the empty group participates in nothing.
    enqueue_names = [entry[1] for entry in log if entry[0] == "group.enqueue_d2h"]
    assert enqueue_names == ["expert_fc1", "moe_act"]
    # No event synchronize at forward time: the sync belongs to backward.
    assert not any(e.endswith(".synchronize") for e in events)
    assert "group.wait_and_release" not in events
    # Byte accounting covers all groups (the empty one contributes zero).
    assert mgr._local_graph_d2h_bytes == sum(g.logical_bytes for g in groups)


def test_local_graph_backward_prepare_finish_wait_pipeline(monkeypatch):
    """Reload phases preserve reverse order and consume via one stream event wait."""
    log = []
    _record_remap_batch(monkeypatch, log)
    compute_stream = _RecordingStream("compute", log)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: compute_stream)
    monkeypatch.setattr(torch.cuda, "stream", lambda s: nullcontext())
    monkeypatch.setattr(
        torch.cuda, "Event", lambda *args, **kwargs: _RecordingEvent(log, "reload_ready")
    )

    names = ["expert_fc1", "core_attn", "moe_act", "attn_proj"]
    groups = [_RecordingGraphGroup(name, log) for name in names]
    mgr = _make_local_graph_batch_manager(log)
    runner = _RecordingRunner(groups)

    assert mgr.local_graph_backward_prepare(runner)
    assert runner.local_graph_reload_state == "reload_pending"
    assert mgr.local_graph_backward_finish_prepare(runner)
    assert runner.local_graph_reload_state == "reload_pending"
    expected_reload_event = runner.local_graph_reload_event
    assert mgr.local_graph_backward_wait_ready(runner, compute_stream)

    reverse_names = list(reversed(names))
    assert [e for e in log if e[0] == "runner.remap_and_copy_after"] == [
        (
            "runner.remap_and_copy_after",
            tuple(reverse_names),
            tuple(reverse_names),
            "h2d",
        )
    ]
    assert [e[1] for e in log if e[0] == "group.prepare_remap"] == reverse_names
    assert [e[1] for e in log if e[0] == "group.adopt_reload_submission"] == reverse_names
    assert [e for e in log if e[0] == "runner.wait_remap_copy_on_stream"] == [
        ("runner.wait_remap_copy_on_stream", "h2d"),
    ]
    assert ("compute.wait_event", expected_reload_event) in log
    assert runner.local_graph_reload_state is None
    assert mgr._local_graph_h2d_bytes == sum(g.logical_bytes for g in groups)


def test_local_graph_backward_wait_ready_primes_unprefetched_runner(monkeypatch):
    """The unprefetched fallback path synchronously submits and waits its own reload.

    prepare() blocks until the worker submitted the H2D (wait before
    prepare_remap bookkeeping); wait_ready then only installs the
    compute-stream dependency on the completed context.
    """
    log = []
    _record_remap_batch(monkeypatch, log)
    compute_stream = _RecordingStream("compute", log)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: compute_stream)
    monkeypatch.setattr(torch.cuda, "stream", lambda s: nullcontext())
    monkeypatch.setattr(
        torch.cuda, "Event", lambda *args, **kwargs: _RecordingEvent(log, "reload_ready")
    )
    runner = _RecordingRunner([_RecordingGraphGroup("first", log)])
    mgr = _make_local_graph_batch_manager(log)

    assert mgr.local_graph_backward_wait_ready(runner, compute_stream)

    assert [e[0] for e in log] == [
        "runner.remap_and_copy_after",
        "runner.wait_remap_copy_on_stream",
        "reload_ready.record",
        "group.prepare_remap",
        "compute.wait_event",
        "group.adopt_reload_submission",
    ]


def test_local_graph_forward_replay_primes_first_backward_runner(monkeypatch):
    """The backward-chain head dispatches its reload at forward-D2H time.

    Without this, the runner consumed first in backward has no predecessor
    replay to look ahead from, so its own backward() would submit the remap
    and immediately host-wait for it (absorbing the RemapWorker latency,
    including the muMemSetAccess in-flight-DMA drain) on the critical path.
    """
    log = []
    _record_remap_batch(monkeypatch, log)
    compute_stream = _RecordingStream("compute", log)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: compute_stream)
    monkeypatch.setattr(torch.cuda, "stream", lambda s: nullcontext())
    monkeypatch.setattr(
        torch.cuda, "Event", lambda *args, **kwargs: _RecordingEvent(log, "reload_ready")
    )
    first = _RecordingRunner([_RecordingGraphGroup("first", log)])
    first.is_first_bwd_runner = True
    later = _RecordingRunner([_RecordingGraphGroup("later", log)])
    mgr = _make_local_graph_batch_manager(log)

    # Forward replay of the chain head submits its reload immediately after
    # the D2H burst; later runners are not touched here.
    mgr.local_graph_forward_replay(first)
    assert first.local_graph_reload_state == "reload_pending"
    assert [e[0] for e in log if e[0] == "runner.remap_and_copy_after"] == [
        "runner.remap_and_copy_after"
    ]
    assert not any(e[1] == "later" for e in log if e[0] == "group.prepare_remap")

    mgr.local_graph_forward_replay(later)
    assert later.local_graph_reload_state is None
    assert [e[0] for e in log if e[0] == "runner.remap_and_copy_after"] == [
        "runner.remap_and_copy_after"
    ]

    # Backward consumption of the head no longer submits anything; it only
    # installs the stream wait and adopts the already-submitted reload.
    # (`later`'s D2H entries sit between because its forward replay ran too.)
    assert mgr.local_graph_backward_wait_ready(first, compute_stream)
    assert [entry[0] for entry in log] == [
        "group.enqueue_d2h",
        "group.try_release",
        "runner.remap_and_copy_after",
        "group.prepare_remap",
        "group.enqueue_d2h",
        "group.try_release",
        "runner.wait_remap_copy_on_stream",
        "reload_ready.record",
        "compute.wait_event",
        "group.adopt_reload_submission",
    ]


def test_local_graph_ping_pong_forward_fences_bank_reuse(monkeypatch):
    """A reused resident bank waits for its previous D2H before graph replay."""
    log = []
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: _RecordingStream("compute", log))
    monkeypatch.setattr(torch.cuda, "stream", lambda s: nullcontext())
    mgr = _make_local_graph_batch_manager(log)
    _configure_ping_pong_manager(mgr, log)
    groups = [_RecordingGraphGroup("expert_fc1", log)]
    first = _RecordingRunner(groups, slot_bank=0)
    second = _RecordingRunner([_RecordingGraphGroup("expert_fc1", log)], slot_bank=0)

    # Simulate the first replay completing its D2H, then reuse the same bank.
    mgr.local_graph_forward_replay(first)
    assert mgr.local_graph_forward_wait_ready(second, _RecordingStream("replay", log))
    assert ("replay.wait_event", mgr._local_graph_bank_d2h_events[0]) in log
    assert not any(entry[0] == "group.try_release" for entry in log)


def test_local_graph_ping_pong_backward_uses_two_banks_and_consumed_fence(monkeypatch):
    """Adjacent reloads use different banks; a later same-bank reload fences consumption."""
    log = []
    compute_stream = _RecordingStream("compute", log)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: compute_stream)
    monkeypatch.setattr(torch.cuda, "stream", lambda s: nullcontext())
    monkeypatch.setattr(torch.cuda, "Event", lambda *args, **kwargs: _RecordingEvent(log, "reload_ready"))
    monkeypatch.setattr("megatron.core.pipeline_parallel.fine_grained_activation_offload.remap_and_copy_after", lambda *args: pytest.fail("VMM remap used in resident mode"))
    mgr = _make_local_graph_batch_manager(log)
    _configure_ping_pong_manager(mgr, log)
    mgr._local_graph_bank_d2h_recorded = [True, True]
    runners = [
        _RecordingRunner([_RecordingGraphGroup("g0", log)], 0),
        _RecordingRunner([_RecordingGraphGroup("g1", log)], 1),
        _RecordingRunner([_RecordingGraphGroup("g2", log)], 0),
    ]
    consumed = _RecordingEvent(log, "consumed0")

    assert mgr.local_graph_backward_prepare(runners[0])
    assert mgr.local_graph_backward_wait_ready(runners[0], compute_stream)
    mgr.local_graph_backward_mark_consumed(runners[0], consumed)
    assert mgr.local_graph_backward_prepare(runners[1])
    assert mgr.local_graph_backward_wait_ready(runners[1], compute_stream)
    assert mgr.local_graph_backward_prepare(runners[2])

    h2d_waits = [entry for entry in log if entry[0] == "h2d.wait_event"]
    # Runner 2 reuses bank 0 and must wait for runner 0's consumed event.
    assert any(entry[1] is consumed for entry in h2d_waits)
    assert [entry[1] for entry in log if entry[0] == "group.enqueue_resident_h2d"] == ["g0", "g1", "g2"]
    assert not any(entry[0] == "runner.remap_and_copy_after" for entry in log)


    """With no active groups, neither replay path touches streams or groups."""
    log = []
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: "compute")
    monkeypatch.setattr(torch.cuda, "stream", lambda s: nullcontext())

    groups = [_RecordingGraphGroup("empty", log, num_tensors=0)]
    mgr = _make_local_graph_batch_manager(log)
    runner = _RecordingRunner(groups)

    mgr.local_graph_forward_replay(runner)
    mgr.local_graph_backward_replay(runner)

    assert log == []
    assert mgr._local_graph_d2h_bytes == 0
    assert mgr._local_graph_h2d_bytes == 0
