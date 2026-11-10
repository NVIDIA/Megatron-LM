# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Single-device FakePG probes: allocation shapes/lifetimes, not numerical parity."""

import gc
import json
import os
import weakref
from pathlib import Path

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.ops.moe import fake_deepep
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.moe import fused_a2a
from megatron.core.transformer.moe.token_dispatcher import MoEFlexTokenDispatcher
from megatron.core.transformer.transformer_config import TransformerConfig


@pytest.mark.parametrize(
    "options",
    [
        {"bf16": False},
        {"fp8": "hybrid"},
        {"moe_flex_dispatcher_backend": "hybridep"},
        {"moe_flex_dispatcher_backend": "deepepv2"},
        {"moe_expert_capacity_factor": 1.0},
        {"moe_pad_expert_input_to_capacity": True},
        {"cuda_graph_impl": "local"},
        {"moe_router_force_load_balancing": False},
    ],
)
def test_reject_unsupported_memory_models(options):
    args = {
        "bf16": True,
        "moe_token_dispatcher_type": "flex",
        "moe_router_force_load_balancing": True,
        **options,
    }
    with pytest.raises(ValueError):
        fake_deepep.validate_fake_flex_config(args)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU memory probe")
@pytest.mark.parametrize("etp,rank", [(1, 0), (1, 7), (2, 0), (2, 15)])
@pytest.mark.parametrize("permute_fusion", [False, True])
def test_native_fake_flex_memory(etp, rank, permute_fusion, monkeypatch):
    if not fused_a2a.HAVE_DEEP_EP:
        pytest.skip("DeepEP v1 size hints required")
    from torch.testing._internal.distributed.fake_pg import FakeStore

    # Each pytest worker owns its FakePG. No other GPU or logical rank runs.
    if torch.distributed.is_initialized():
        pytest.skip("requires an isolated process without a pre-existing process group")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    world = 8 * etp
    torch.distributed.init_process_group("fake", store=FakeStore(), rank=rank, world_size=world)
    fake_deepep._fake_buffer = None
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=etp,
        pipeline_model_parallel_size=2,
        expert_model_parallel_size=4,
        expert_tensor_parallel_size=etp,
        create_gloo_process_groups=False,
    )
    try:
        groups = ProcessGroupCollection.use_mpu_process_groups()
        config = TransformerConfig(
            num_layers=4,
            hidden_size=128,
            num_attention_heads=8,
            tensor_model_parallel_size=etp,
            pipeline_model_parallel_size=2,
            expert_model_parallel_size=4,
            expert_tensor_parallel_size=etp,
            sequence_parallel=etp > 1,
            add_bias_linear=False,
            bf16=True,
            params_dtype=torch.bfloat16,
            pipeline_dtype=torch.bfloat16,
            num_moe_experts=16,
            moe_router_topk=4,
            moe_router_dtype="fp32",
            moe_token_dispatcher_type="flex",
            moe_flex_dispatcher_backend="deepep",
            moe_router_force_load_balancing=True,
            moe_permute_fusion=permute_fusion,
        )
        from megatron.core.utils import get_batch_on_this_tp_rank

        batch = {
            "tokens": torch.zeros((1, 17), dtype=torch.int64, device="cuda"),
            "labels": torch.zeros((1, 17), dtype=torch.int64, device="cuda"),
            "loss_mask": torch.ones((1, 17), dtype=torch.float32, device="cuda"),
            "position_ids": torch.zeros((1, 17), dtype=torch.int64, device="cuda"),
            "attention_mask": None,
        }
        batch = get_batch_on_this_tp_rank(
            batch,
            has_cu_seqlens=False,
            dynamic_cp=False,
            create_attention_mask_in_dataloader=False,
            broadcast_src_rank=torch.distributed.get_global_rank(groups.tp, 0),
            broadcast_group=groups.tp,
            cp_size=1,
            tp_rank=groups.tp.rank(),
            micro_batch_size=1,
            seq_length=17,
            mtp_on_this_rank=False,
            pipeline_model_parallel_size=2,
            is_pipeline_first_stage=groups.pp.rank() == 0,
            is_pipeline_last_stage=groups.pp.rank() == 1,
        )
        if batch["labels"] is not None:
            assert torch.count_nonzero(batch["labels"]).item() == 0
            assert torch.all(batch["loss_mask"] == 1)
        if batch["tokens"] is not None:
            assert torch.count_nonzero(batch["tokens"]).item() == 0

        ep_rank = groups.ep.rank()
        dispatcher = MoEFlexTokenDispatcher(
            4, list(range(ep_rank * 4, ep_rank * 4 + 4)), config, groups
        )

        # A real Buffer constructor must never be reached, including in backward.
        def forbidden(*args, **kwargs):
            raise AssertionError("real DeepEP Buffer initialized in fake mode")

        monkeypatch.setattr(fused_a2a.Buffer, "__init__", forbidden)
        torch.cuda.memory._record_memory_history(max_entries=200000)
        torch.cuda.reset_peak_memory_stats()
        losses, inputs, references = [], [], []
        buffers = []
        for _ in range(3):
            x = torch.zeros((17, 1, 128), dtype=torch.bfloat16, device="cuda", requires_grad=True)
            probabilities = torch.ones(
                (17, 16), dtype=torch.float32, device="cuda", requires_grad=True
            )
            routing = torch.zeros((17, 16), dtype=torch.bool, device="cuda")
            routing[:, :4] = True
            local, probs = dispatcher.dispatch_preprocess(x, routing, probabilities)
            received, probs = dispatcher.token_dispatch(local, probs)
            assert received.shape == (17 * 4 * etp, 128)
            local, counts, probs = dispatcher.dispatch_postprocess(received, probs)
            assert counts.sum().item() == 17 * 4 * etp
            assert int(counts.max() - counts.min()) <= 1
            weight = torch.ones((128, 128), dtype=torch.bfloat16, device="cuda", requires_grad=True)
            expert_output = (local @ weight) * probs.to(torch.bfloat16).unsqueeze(1)
            local = dispatcher.combine_preprocess(expert_output)
            references.extend(weakref.ref(t) for t in dispatcher._comm_manager.handle["metadata"])
            result = dispatcher.combine_postprocess(dispatcher.token_combine(local))
            assert result.shape == x.shape
            losses.append(result.float().sum())
            inputs.append((x, probabilities, weight))
            buffers.append(fake_deepep._fake_buffer)
        assert buffers[0] is buffers[1] is buffers[2]
        assert all(ref() is not None for ref in references)
        sum(losses).backward()
        for x, probabilities, weight in inputs:
            assert x.grad is not None and x.grad.shape == x.shape
            assert (
                probabilities.grad is not None and probabilities.grad.shape == probabilities.shape
            )
            assert weight.grad is not None
        del losses, result, local, expert_output, received, probs
        gc.collect()
        assert all(ref() is None for ref in references)
        torch.cuda.synchronize()
        output = os.environ.get("FAKE_FLEX_TEST_ARTIFACTS")
        if output:
            directory = Path(output)
            directory.mkdir(parents=True, exist_ok=True)
            name = f"etp{etp}-rank{rank}-fused{int(permute_fusion)}"
            torch.cuda.memory._dump_snapshot(str(directory / (name + ".pickle")))
            (directory / (name + ".json")).write_text(
                json.dumps(
                    {
                        "world_size": world,
                        "rank": rank,
                        "etp": etp,
                        "permute_fusion": permute_fusion,
                        "allocated_peak_bytes": torch.cuda.max_memory_allocated(),
                        "reserved_peak_bytes": torch.cuda.max_memory_reserved(),
                        "buffer": buffers[0].allocation_summary,
                        "model": fake_deepep.fake_flex_memory_model(vars(config)),
                        "passed": True,
                    },
                    indent=2,
                )
            )
    finally:
        torch.cuda.memory._record_memory_history(enabled=None)
        parallel_state.destroy_model_parallel()
        fake_deepep._fake_buffer = None
        torch.distributed.destroy_process_group()
