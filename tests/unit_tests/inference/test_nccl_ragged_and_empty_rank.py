# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Regression coverage for the NCCL inference dispatcher under ragged and empty-rank inputs.

The existing ``test_dispatch_combine`` in ``test_moe_dispatching_and_routing.py`` already
proves, for equal token counts and for "rank r owns (r+1)*8 tokens", that dispatch gathers
every rank's slice back into the global tensor exactly, and that combine returns
``ep_size *`` this rank's slice. This file adds what that case does not cover:

1. **A rank with zero local tokens.** One rank contributes nothing while another carries the
   load. The empty rank must still take part in the collectives, must receive the full gathered
   tensor rather than zero rows, and the loaded rank's result must be unaffected.
2. **The compact row-order contract, asserted directly.** The gathered rows must be
   ``[rank 0 rows..., rank 1 rows..., ...]``. A wrong permutation can still satisfy a
   shape-and-equality check against a matching permutation, so the ordering is verified from a
   per-source-rank tag instead of being left implicit.
3. **Path transitions on one live dispatcher** (equal -> ragged -> zero-local -> equal), so no
   per-path state leaks across steps.
4. **The capability boundary.** ``inference_optimized`` MoE with ``expert_tensor_parallel_size >
   1`` is rejected on this base. The rejection and its message are pinned so a silent change is
   caught. No positive ETP path is implemented or implied here.

The two dispatcher paths are not interchangeable. `dynamic_context` sets
`_use_allgather_v = not using_cuda_graph_this_step()`, so

  * `_use_allgather_v=False` is the CUDA-graph (decode) path, where equal token counts across
    ranks are guaranteed **by construction**, and
  * `_use_allgather_v=True` is the eager/ragged (prefill) path, where counts may differ.

Pairing unequal counts with `_use_allgather_v=False` is therefore not a reachable state, and
feeding it one does not produce a wrong answer but a hang: the equal-count gather assumes a
shared row count. These tests keep each count profile with its legal path.

Scope: dispatcher level. No dynamic engine is constructed, and this makes no claim about ETP>1
inference, which this base still rejects.

Run it with a real EP group:

    torchrun --standalone --nproc_per_node=2 -m pytest -q \
        tests/unit_tests/inference/test_nccl_ragged_and_empty_rank.py
"""

import gc
import os

import pytest
import torch

from megatron.core.transformer.moe.token_dispatcher_inference import NCCLAllGatherDispatcher
from tests.unit_tests.inference.test_moe_dispatching_and_routing import (
    NANOV3_BASE,
    _make_base_config,
)
from tests.unit_tests.test_utilities import Utils

SEED = 20260918


def _empty_rank_counts(ep_size: int) -> list[int]:
    """Per-rank token counts with rank 0 empty and the remaining ranks loaded."""
    plans = {2: [0, 64], 4: [0, 64, 0, 32], 8: [0, 64, 0, 32, 0, 16, 0, 8]}
    return plans.get(ep_size, [0] + [32] * (ep_size - 1))


def _equal_counts(ep_size: int) -> list[int]:
    return [16] * ep_size


def _ragged_counts(ep_size: int) -> list[int]:
    """Uneven but non-empty counts, to pair with the empty-rank case."""
    plans = {2: [40, 8], 4: [40, 8, 24, 4], 8: [40, 8, 24, 4, 16, 2, 9, 1]}
    return plans.get(ep_size, [40] + [8] * (ep_size - 1))


@pytest.fixture(scope="module")
def ep_domain():
    """A real model-parallel domain with EP = world size.

    Self-contained on purpose: this module must be runnable on its own, and it tears down what
    it initialises so it cannot race another module's setup in a shared torchrun session.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        pytest.skip("requires torchrun with RANK/WORLD_SIZE")

    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    Utils.initialize_model_parallel(1, 1, expert_model_parallel_size=world)
    try:
        yield world
    finally:
        Utils.destroy_model_parallel()


class TestNCCLRaggedAndEmptyRank:
    """Dispatcher-level regressions for ragged and empty-rank inputs."""

    def teardown_method(self, method):
        gc.collect()
        torch.cuda.empty_cache()

    def _make_dispatcher(self, **config_overrides):
        from megatron.core.transformer.moe.moe_utils import get_default_pg_collection

        NCCLAllGatherDispatcher.allocate_buffers()
        config_overrides.setdefault("expert_model_parallel_size", Utils.world_size)
        config = _make_base_config(**config_overrides)
        num_local_experts = config.num_moe_experts // Utils.world_size
        ep_rank = torch.distributed.get_rank() if Utils.world_size > 1 else 0
        local_expert_indices = [ep_rank * num_local_experts + i for i in range(num_local_experts)]
        return NCCLAllGatherDispatcher(
            num_local_experts=num_local_experts,
            local_expert_indices=local_expert_indices,
            config=config,
            pg_collection=get_default_pg_collection(),
            runs_metadata_sync=True,
        )

    def _build(self, counts: list[int], rank: int):
        """Per-rank inputs plus the global reference, in one deterministic pass.

        Values are integers on a 2**-8 grid, which bf16 represents exactly, so the comparisons
        below can be exact equalities rather than tolerances. Rebuilding from the same seed
        makes two calls with the same ``counts`` produce identical values.
        """
        hidden_size = NANOV3_BASE["hidden_size"]
        topk = NANOV3_BASE["moe_router_topk"]
        num_experts = NANOV3_BASE["num_moe_experts"]
        total = sum(counts)
        offset = sum(counts[:rank])
        local = counts[rank]

        g = torch.Generator(device="cpu").manual_seed(SEED)
        hidden = (torch.randint(-128, 129, (total, hidden_size), generator=g).float() / 256.0)
        hidden = hidden.to(torch.bfloat16)
        probs = torch.rand(total, topk, generator=g, dtype=torch.float32)
        rmap = torch.randint(0, num_experts, (total, topk), generator=g)

        # a per-source-rank tag in column 0 makes the compact ORDER observable
        for src, n in enumerate(counts):
            start = sum(counts[:src])
            hidden[start : start + n, 0] = float(src + 1)

        return {
            "global_hidden": hidden.cuda(),
            "global_probs": probs.cuda(),
            "global_rmap": rmap.cuda(),
            "local_hidden": hidden[offset : offset + local].contiguous().cuda(),
            "local_probs": probs[offset : offset + local].contiguous().cuda(),
            "local_rmap": rmap[offset : offset + local].contiguous().cuda(),
            "offset": offset,
            "local": local,
            "total": total,
        }

    def _dispatch(self, dispatcher, data):
        dispatcher.routing_map = data["local_rmap"]
        return dispatcher.token_dispatch(data["local_hidden"], data["local_probs"])

    def test_zero_local_tokens(self, ep_domain):
        """A rank with no tokens must still participate and must not be dropped.

        Only the ragged path can legally see a zero-token rank: the equal-count path is the
        CUDA-graph decode path where equal counts hold by construction.
        """
        dispatcher = self._make_dispatcher()
        rank = torch.distributed.get_rank()
        counts = _empty_rank_counts(dispatcher.ep_size)
        NCCLAllGatherDispatcher._use_allgather_v = True

        data = self._build(counts, rank)
        d_hidden, d_probs = self._dispatch(dispatcher, data)

        assert d_hidden.shape == (data["total"], data["global_hidden"].shape[1]), (
            f"rank {rank}: gathered hidden {tuple(d_hidden.shape)} for total={data['total']}"
        )
        assert d_probs.shape == (data["total"], data["global_probs"].shape[1])
        torch.testing.assert_close(d_hidden, data["global_hidden"], atol=0, rtol=0)
        torch.testing.assert_close(d_probs, data["global_probs"], atol=0, rtol=0)
        torch.testing.assert_close(dispatcher.routing_map, data["global_rmap"], atol=0, rtol=0)

        combined = dispatcher.token_combine(d_hidden)
        assert combined.shape == (data["local"], data["global_hidden"].shape[1])
        if data["local"] == 0:
            assert combined.numel() == 0, f"empty rank {rank} produced non-empty output"
        else:
            expected = (
                data["global_hidden"][data["offset"] : data["offset"] + data["local"]].float()
                * dispatcher.ep_size
            ).bfloat16()
            torch.testing.assert_close(combined, expected, atol=0, rtol=0)

    def test_compact_row_order_is_rank_major(self, ep_domain):
        """The gathered rows must be rank-major, verified from the per-source-rank tag."""
        dispatcher = self._make_dispatcher()
        rank = torch.distributed.get_rank()
        counts = _empty_rank_counts(dispatcher.ep_size)
        NCCLAllGatherDispatcher._use_allgather_v = True

        data = self._build(counts, rank)
        d_hidden, _ = self._dispatch(dispatcher, data)

        col0 = d_hidden[:, 0].float()
        for src, n in enumerate(counts):
            if n == 0:
                continue
            start = sum(counts[:src])
            block = col0[start : start + n]
            assert bool((block == float(src + 1)).all()), (
                f"rows [{start}, {start + n}) should come from source rank {src} "
                f"(tag {src + 1}) but carry tags {sorted(set(block.tolist()))}"
            )

    def test_path_transitions_on_one_dispatcher(self, ep_domain):
        """equal -> ragged -> zero-local -> equal on a live dispatcher, with no state leak."""
        dispatcher = self._make_dispatcher()
        ep = dispatcher.ep_size
        rank = torch.distributed.get_rank()

        # each profile is paired with the path that legally carries it (see module docstring)
        sequence = [
            ("equal_graph", _equal_counts(ep), False),
            ("ragged_eager", _ragged_counts(ep), True),
            ("zero_local_eager", _empty_rank_counts(ep), True),
            ("equal_graph_again", _equal_counts(ep), False),
        ]
        for name, counts, use_v in sequence:
            NCCLAllGatherDispatcher._use_allgather_v = use_v
            data = self._build(counts, rank)
            d_hidden, d_probs = self._dispatch(dispatcher, data)
            assert d_hidden.shape[0] == data["total"], (
                f"{name}: gathered {d_hidden.shape[0]} rows for total {data['total']}"
            )
            torch.testing.assert_close(d_hidden, data["global_hidden"], atol=0, rtol=0)
            torch.testing.assert_close(d_probs, data["global_probs"], atol=0, rtol=0)


class TestInferenceETPCapability:
    """The ETP>1 rejection on this base is a documented boundary, not an accident.

    Pinned so that a silent loosening is caught, and so that a future ETP implementation has to
    update this expectation deliberately. It does not skip when the guard disappears, because
    that would defeat the test.
    """

    def test_inference_optimized_rejects_etp_gt_1(self):
        from megatron.core.transformer.transformer_config import TransformerConfig

        with pytest.raises(ValueError, match="does not support expert tensor parallelism"):
            TransformerConfig(
                num_layers=1, hidden_size=128, num_attention_heads=8, ffn_hidden_size=256,
                num_moe_experts=8, moe_router_topk=2, moe_grouped_gemm=False,
                moe_router_dtype="fp32", add_bias_linear=False, use_cpu_initialization=True,
                normalization="RMSNorm", transformer_impl="inference_optimized",
                expert_model_parallel_size=2, expert_tensor_parallel_size=2,
            )

    def test_inference_optimized_accepts_etp_eq_1(self):
        """The legal baseline stays legal. RMSNorm is required by this path."""
        from megatron.core.transformer.transformer_config import TransformerConfig

        cfg = TransformerConfig(
            num_layers=1, hidden_size=128, num_attention_heads=8, ffn_hidden_size=256,
            num_moe_experts=8, moe_router_topk=2, moe_grouped_gemm=False,
            moe_router_dtype="fp32", add_bias_linear=False, use_cpu_initialization=True,
            normalization="RMSNorm", transformer_impl="inference_optimized",
            expert_model_parallel_size=2, expert_tensor_parallel_size=1,
        )
        assert cfg.expert_tensor_parallel_size == 1
