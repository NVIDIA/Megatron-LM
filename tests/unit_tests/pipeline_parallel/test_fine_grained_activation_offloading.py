# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import gc
import os
from contextlib import nullcontext
from typing import Dict, List, Optional, Tuple

import pytest
import torch

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.transformer_config import MLATransformerConfig, TransformerConfig
from megatron.core.utils import is_te_min_version
from tests.unit_tests.test_utilities import Utils

# Tolerance for memory expectation check (GPU allocator jitter etc).
EPSILON = 0.30


def _reset_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


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
        # Make sure model weights / activations are BF16 so TE fused attention isn't disabled.
        bf16=True,
        # params_dtype=torch.bfloat16,
        # enable_autocast=True,
        # autocast_dtype=torch.bfloat16,
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
            moe_use_legacy_grouped_gemm=False,
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
    from megatron.core.pipeline_parallel import fine_grained_activation_offload as off

    if enable_offload_reset:
        off.fine_grained_offloading_reset()

    # for p in model.parameters():
    #     if p.grad is not None:
    #         p.grad = None

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
@pytest.mark.skipif(
    not is_te_min_version("1.13.0"),
    reason="Fine-grained activation offloading requires TE-based GPT layer spec (TE 1.13+ in this repo's tests).",
)
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
    Utils.initialize_model_parallel(1, 1)

    seed = 123
    # Choose shapes large enough to make memory deltas stable but still fast.
    num_experts = 4 if is_moe else None
    num_layers = 8
    hidden_size = 2048 if num_experts is None else 1024
    num_attention_heads = 16 if hidden_size >= 2048 else 8
    vocab_size = 512
    seq_length = 512
    micro_batch_size = 2
    device = torch.device("cuda")

    input_ids, position_ids, attention_mask = _make_gpt_inputs(
        seq_length=seq_length, micro_batch_size=micro_batch_size, device=device
    )

    from megatron.core.pipeline_parallel import fine_grained_activation_offload as off

    off.fine_grained_offloading_reset_instance()

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
        off.fine_grained_offloading_reset()

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
            assert rel_err <= EPSILON, (
                f"Memory saving mismatch for offload_modules={offload_modules}: "
                f"saved={saved_mib:.2f}MiB expected~={expected_offload_mib:.2f}MiB "
                f"(rel_err={rel_err:.2f})"
            )
            print(
                f"Rank {torch.distributed.get_rank()}: Saved {saved_mib:.2f}MiB, expected {expected_offload_mib:.2f}MiB"
            )
    finally:
        Utils.destroy_model_parallel()
