# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""FSDP + EP overlap + Hybrid model integration test.

End-to-end check that a HybridModel with a bracketed layer pattern (each
bracket builds an inner ``HybridStack`` that becomes the FSDP unit) trains
identically through two paths:

- reference: standard FSDP forward/backward, no EP overlap
- test: ``combined_1f1b_schedule_for_no_pipelining`` with
  ``overlap_moe_expert_parallel_comm=True``

Both paths use ``fsdp_unit_modules=[HybridStack]`` so meta-device
materialization traversal is identical and the seed lands on the same
parameter each draw -- a precondition for bit-exact comparison. The outer
HybridStack root (``is_layer_group_stack=False``) is filtered out of the
FSDP unit set by ``MegatronFSDP._is_fsdp_unit_module`` so each bracket
group's inner HybridStack is its own unit; this test also asserts that
property directly.
"""

import gc

import pytest
import torch

from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed.fsdp.mcore_fsdp_adapter import FullyShardedDataParallel
from megatron.core.distributed.fsdp.src.megatron_fsdp.fully_shard import fully_shard_optimizer
from megatron.core.models.hybrid.hybrid_block import HybridStack
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.pipeline_parallel.utils import set_streams
from megatron.core.ssm.mamba_mixer import HAVE_MAMBA_SSM
from megatron.core.transformer import TransformerConfig

try:
    import causal_conv1d  # noqa: F401

    HAVE_CAUSAL_CONV1D = True
except ImportError:
    HAVE_CAUSAL_CONV1D = False
from megatron.core.utils import is_te_min_version, is_torch_min_version
from tests.unit_tests.a2a_overlap.utils import (
    assert_models_equal,
    build_input_data,
    deterministic_mode,
    fsdp_train_step,
    get_valid_flex_dispatcher_backend,
    get_valid_token_dispatcher_types,
    overlap_train_step,
    reset_model,
)
from tests.unit_tests.test_utilities import Utils

SEQ_LEN = 32
VOCAB_SIZE = 128
NUM_STEPS = 3
LR = 0.01


def _hybrid_config(hybrid_layer_pattern, num_moe_experts=8, extra_kwargs=None):
    """Build a TransformerConfig usable by HybridModel + EP overlap."""
    extra_kwargs = dict(extra_kwargs or {})
    # HybridModel derives effective num_layers from the pattern; we still pass
    # the flattened count so TransformerConfig.__post_init__ checks pass.
    flat = hybrid_layer_pattern.replace("[", "").replace("]", "")
    return TransformerConfig(
        # ``deterministic_mode`` (utils.deterministic_mode) sets
        # ``NVTE_FUSED_ATTN=0`` for reproducibility; the default attention
        # backend ``auto`` asserts that env is unset, so pin it to ``unfused``
        # like the GPT-side a2a_overlap tests do.
        attention_backend="unfused",
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=4,
        deterministic_mode=True,
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        num_layers=len(flat),
        hidden_size=512,
        num_attention_heads=8,
        num_query_groups=8,
        ffn_hidden_size=512,
        kv_channels=64,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        add_bias_linear=False,
        num_moe_experts=num_moe_experts,
        moe_grouped_gemm=True,
        moe_router_dtype="fp32",
        **extra_kwargs,
    )


def _hybrid_model(config, hybrid_layer_pattern):
    return HybridModel(
        config=config,
        hybrid_stack_spec=hybrid_stack_spec,
        vocab_size=VOCAB_SIZE,
        max_sequence_length=SEQ_LEN,
        hybrid_layer_pattern=hybrid_layer_pattern,
        pre_process=True,
        post_process=True,
    ).cuda()


def _make_ddp_config():
    return DistributedDataParallelConfig(
        use_megatron_fsdp=True,
        data_parallel_sharding_strategy="optim_grads_params",
        overlap_grad_reduce=True,
        overlap_param_gather=True,
        megatron_fsdp_main_params_dtype=None,
    )


def _count_fsdp_units(fsdp_wrapper):
    """Count HybridStack instances actually registered as FSDP units."""
    inner = fsdp_wrapper.module  # MegatronFSDP wrapper
    return sum(
        1
        for m in inner.module.modules()
        if isinstance(m, HybridStack) and inner._is_fsdp_unit_module(m)
    )


class TestFSDPHybridOverlap:
    """FSDP + EP overlap + hybrid model: per-step loss and final weights
    must match the no-overlap reference (both using HybridStack as the
    FSDP unit)."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=4,
        )
        set_streams()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(not is_te_min_version("2.3.0"), reason="Requires TE >= 2.3.0")
    @pytest.mark.skipif(
        not is_torch_min_version("2.6.0"), reason="EP overlap hangs on torch < 2.6.0"
    )
    @pytest.mark.parametrize("dispatcher_type", get_valid_token_dispatcher_types())
    @pytest.mark.parametrize("shared_expert_intermediate_size", [None, 512])
    @pytest.mark.parametrize("hybrid_layer_pattern", ["[*E][*E]", "[M*E][M*E]"])
    def test_fsdp_hybrid_overlap_training_step(
        self, dispatcher_type, shared_expert_intermediate_size, hybrid_layer_pattern
    ):
        if "M" in hybrid_layer_pattern and not (HAVE_MAMBA_SSM and HAVE_CAUSAL_CONV1D):
            pytest.skip(
                "Mamba pattern requires both mamba-ssm and causal-conv1d "
                "(`pip install mamba-ssm causal-conv1d`)."
            )
        extra_kwargs = {"moe_token_dispatcher_type": dispatcher_type}
        if dispatcher_type == "flex":
            backend = get_valid_flex_dispatcher_backend()
            if backend is None:
                pytest.skip("No flex dispatcher backend available")
            extra_kwargs["moe_flex_dispatcher_backend"] = backend
        if shared_expert_intermediate_size is not None:
            extra_kwargs["moe_shared_expert_intermediate_size"] = shared_expert_intermediate_size

        with deterministic_mode():
            data = build_input_data(seq_len=SEQ_LEN, vocab_size=VOCAB_SIZE)

            # Reference: no EP overlap, but same FSDP unit class so init
            # consumes the seeded RNG in the same order as the test path.
            ref_config = _hybrid_config(hybrid_layer_pattern, extra_kwargs=extra_kwargs)
            ref_model = _hybrid_model(ref_config, hybrid_layer_pattern)
            init_params = reset_model(ref_model)

            ref_fsdp = FullyShardedDataParallel(
                config=ref_config,
                ddp_config=_make_ddp_config(),
                module=ref_model,
                fsdp_unit_modules=[HybridStack],
            )
            ref_opt = torch.optim.SGD(ref_fsdp.parameters(), lr=LR)
            ref_opt = fully_shard_optimizer(optimizer=ref_opt)

            # Test: EP overlap on.
            test_kwargs = {**extra_kwargs, "overlap_moe_expert_parallel_comm": True}
            test_config = _hybrid_config(hybrid_layer_pattern, extra_kwargs=test_kwargs)
            test_model = _hybrid_model(test_config, hybrid_layer_pattern)
            reset_model(test_model, init_params)

            test_fsdp = FullyShardedDataParallel(
                config=test_config,
                ddp_config=_make_ddp_config(),
                module=test_model,
                fsdp_unit_modules=[HybridStack],
            )
            test_opt = torch.optim.SGD(test_fsdp.parameters(), lr=LR)
            test_opt = fully_shard_optimizer(optimizer=test_opt)

            # Lock in the FSDP-unit selection: each bracket group's inner
            # HybridStack is its own unit, and the outer root is excluded.
            # Pattern `[*E][*E]` has 2 bracket groups.
            expected_units = hybrid_layer_pattern.count("[")
            assert _count_fsdp_units(ref_fsdp) == expected_units, (
                f"reference: expected {expected_units} FSDP units, "
                f"got {_count_fsdp_units(ref_fsdp)}"
            )
            assert _count_fsdp_units(test_fsdp) == expected_units, (
                f"test: expected {expected_units} FSDP units, "
                f"got {_count_fsdp_units(test_fsdp)}"
            )

            rank = torch.distributed.get_rank()
            for step in range(NUM_STEPS):
                if hasattr(ref_fsdp, "set_is_first_microbatch"):
                    ref_fsdp.set_is_first_microbatch()
                ref_loss = fsdp_train_step(ref_fsdp, ref_opt, data)
                test_loss = overlap_train_step(test_fsdp, test_opt, test_config, data)

                assert torch.equal(ref_loss, test_loss), (
                    f"[rank {rank}] Loss mismatch at step {step}: "
                    f"ref={ref_loss.item()}, test={test_loss.item()}"
                )

            assert_models_equal(ref_fsdp, test_fsdp)

            del ref_fsdp, test_fsdp, ref_opt, test_opt
            gc.collect()
            torch.cuda.empty_cache()
