# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""``use_cpu_initialization`` must build the same global expert stack at every EP degree.

CPU initialization exists so a run's initial weights are a function of the seed alone, not of the
parallel layout. Expert weights are the case that is easy to get wrong: the expert-parallel RNG
separation lives in the **CUDA** tracker (``expert_parallel_seed`` folds in ``ep_rank``), which the
plain CPU generator never sees. Without a per-expert CPU stream every EP rank initializes its local
experts from the same draws, so a model with ``num_experts`` experts silently ends up holding only
``num_experts // ep_size`` distinct ones -- correct shapes, correct dtypes, no crash, wrong model.

Both expert implementations are covered because they reach the CPU path differently:
``SequentialMLP`` goes through ``_initialize_affine_weight_cpu`` via one Linear per expert, while
``TEGroupedLinear`` holds ``weight0..weightN-1`` in a single module.
"""

import pytest
import torch
import torch.distributed as dist

from megatron.core import parallel_state as ps
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import get_pg_rank
from tests.unit_tests.test_utilities import Utils

WORLD = 4
NUM_EXPERTS = 8
HIDDEN = 128
FFN = 256
SEED = 1234
DTYPE = torch.bfloat16


def _config(ep, grouped):
    return TransformerConfig(
        num_layers=1,
        hidden_size=HIDDEN,
        num_attention_heads=8,
        ffn_hidden_size=FFN,
        num_moe_experts=NUM_EXPERTS,
        moe_router_topk=2,
        moe_grouped_gemm=grouped,
        add_bias_linear=False,
        params_dtype=DTYPE,
        bf16=True,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        bias_dropout_fusion=False,
        expert_model_parallel_size=ep,
        use_cpu_initialization=True,
        init_method_std=0.02,
    )


def _build(ep, grouped):
    """Re-initialize model parallelism at this EP degree and build a CPU-initialized block."""
    ps.destroy_model_parallel()
    ps.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, expert_model_parallel_size=ep
    )
    # CPU init draws from the CPU generator, which every rank seeds identically; the CUDA tracker
    # is seeded too so a weight that falls back to GPU init is not silently skipped.
    torch.manual_seed(SEED)
    model_parallel_cuda_manual_seed(SEED, force_reset_rng=True)
    pg_collection = ProcessGroupCollection.use_mpu_process_groups(
        required_pgs=['tp', 'cp', 'pp', 'ep', 'expt_tp', 'gtp_remat', 'expt_gtp_remat']
    )
    spec = get_gpt_layer_with_transformer_engine_spec(
        num_experts=NUM_EXPERTS, moe_grouped_gemm=grouped
    )
    block = TransformerBlock(_config(ep, grouped), spec, pg_collection=pg_collection).cuda()
    return block, pg_collection


def _global_digest(block, pg_collection, ep):
    """name -> checksum for the whole model, keyed so layouts are comparable.

    Local expert indices are rewritten to global ones (``weight3`` / ``local_experts.3`` ->
    ``#e{ep_rank * num_local + 3}``) and every rank's entries are merged, so the result describes
    the global model rather than one rank's slice of it.
    """
    import re

    ep_rank = get_pg_rank(pg_collection.ep)
    num_local = NUM_EXPERTS // ep
    local = {}
    for name, param in block.named_parameters():
        grouped_match = re.search(r"(weight|bias)(\d+)$", name)
        sequential_match = re.search(r"local_experts\.(\d+)\.", name)
        if grouped_match:
            index = ep_rank * num_local + int(grouped_match.group(2))
            key = name[: grouped_match.start()] + f"{grouped_match.group(1)}#e{index}"
        elif sequential_match:
            index = ep_rank * num_local + int(sequential_match.group(1))
            key = (
                name[: sequential_match.start()]
                + f"local_experts.#e{index}."
                + name[sequential_match.end() :]
            )
        else:
            key = name
        values = param.data.detach().float().cpu()
        local[key] = (tuple(values.shape), round(values.double().sum().item(), 5))

    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local)
    merged = {}
    for rank_digest in gathered:
        merged.update(rank_digest)
    return merged


def _expert_weight_keys(digest):
    return [k for k in digest if "#e" in k and "fc1" in k]


def _require_world():
    """Skip unless this is a WORLD-rank torchrun job, bringing the process group up if needed.

    The guard has to initialize first: under torchrun nothing has called init_process_group when
    collection runs, so querying the world size would raise instead of skipping.
    """
    if torch.cuda.device_count() < WORLD:
        pytest.skip(f"requires {WORLD} GPUs")
    if not dist.is_initialized():
        Utils.initialize_distributed()
    if dist.get_world_size() != WORLD:
        pytest.skip(f"requires world_size={WORLD}")


class TestExpertCpuInitialization:
    """CPU-initialized experts must not depend on the expert-parallel layout."""

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("grouped", [True, False])
    def test_experts_are_distinct(self, grouped):
        """Every global expert must get its own values, not one draw shared across EP ranks."""
        _require_world()
        block, pg_collection = _build(WORLD, grouped)
        digest = _global_digest(block, pg_collection, WORLD)
        expert_keys = _expert_weight_keys(digest)
        assert len(expert_keys) == NUM_EXPERTS, f"expected {NUM_EXPERTS} experts, got {expert_keys}"
        distinct = {digest[k] for k in expert_keys}
        assert len(distinct) == NUM_EXPERTS, (
            f"only {len(distinct)} distinct experts out of {NUM_EXPERTS}: every EP rank drew the "
            "same values (the CPU generator carries no expert-parallel term)"
        )

    @pytest.mark.parametrize("grouped", [True, False])
    @pytest.mark.parametrize("ep", [2, 4])
    def test_global_model_matches_ep1(self, ep, grouped):
        """The whole model, experts included, must be bit-identical to the EP=1 build."""
        _require_world()
        reference_block, reference_pgs = _build(1, grouped)
        reference = _global_digest(reference_block, reference_pgs, 1)
        del reference_block

        block, pg_collection = _build(ep, grouped)
        actual = _global_digest(block, pg_collection, ep)
        del block

        assert set(actual) == set(reference), (
            f"EP={ep} and EP=1 disagree on which parameters exist: "
            f"{sorted(set(actual) ^ set(reference))[:5]}"
        )
        differing = [k for k in sorted(reference) if reference[k] != actual[k]]
        assert (
            not differing
        ), f"EP={ep} init differs from EP=1 for {len(differing)} params, e.g. " + "; ".join(
            f"{k}: {reference[k]} vs {actual[k]}" for k in differing[:3]
        )
