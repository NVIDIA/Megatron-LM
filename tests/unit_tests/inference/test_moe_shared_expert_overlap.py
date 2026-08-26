# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for side-stream shared-expert overlap on the NVLS inference path.

The overlap launches the shared-expert forward on `SharedExpertMLP.stream` while
the main stream runs AGV -> expert GEMMs -> RSV. Two properties matter and are
covered here:

1. The side stream reads a main-stream-allocated tensor, so the launch must call
   `record_stream` to stop the caching allocator from recycling the block while
   those kernels are still reading it.

2. The side stream owns its own allocator pool. Prefill token counts vary with
   the request mix, so an unbucketed launch keeps missing that pool; a miss that
   falls through to `release_cached_blocks()` blocks the host in
   `cudaDeviceSynchronize` while the main stream may already have an NVLS
   `symm_mem_sync` spin-wait barrier in flight. Bucketing bounds the shape set.

Padding introduced by (2) must not change numerics, which is the main assertion
of the end-to-end test below.
"""

import pytest
import torch

from megatron.core.activations import squared_relu
from megatron.core.inference.batch_dimensions_utils import TOKEN_ROUNDER
from megatron.core.inference.utils import InferenceMode
from megatron.core.transformer.enums import AttnBackend, InferenceCudaGraphScope
from megatron.core.transformer.moe.token_dispatcher_inference import (
    bucket_shared_expert_token_count,
    launch_shared_experts_on_side_stream,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.initialize import _set_random_seed
from tests.unit_tests.test_utilities import Utils

_NVLS_ENGINE_MAX_TOKENS = 512

# Token counts a prefill mix would realistically produce: not multiples of
# TOKEN_ROUNDER, spread across several octaves.
_PREFILL_TOKEN_COUNTS = [1, 17, 63, 64, 65, 100, 128, 129, 200, 256, 257, 300, 384, 500]


NANOV3_BASE = dict(
    num_layers=4,
    hidden_size=128,
    ffn_hidden_size=128,
    num_attention_heads=4,
    num_query_groups=2,
    num_moe_experts=8,
    moe_ffn_hidden_size=128,
    moe_router_topk=6,
    moe_router_score_function="sigmoid",
    moe_router_enable_expert_bias=True,
    moe_router_topk_scaling_factor=2.5,
    moe_shared_expert_intermediate_size=256,
    moe_router_dtype='fp32',
    moe_shared_expert_overlap=True,
    moe_grouped_gemm=True,
    moe_token_dispatcher_type="alltoall",
    moe_aux_loss_coeff=0.01,
    activation_func=squared_relu,
    normalization="RMSNorm",
    add_bias_linear=False,
    bf16=True,
    params_dtype=torch.bfloat16,
    transformer_impl="inference_optimized",
    expert_tensor_parallel_size=1,
    use_cpu_initialization=True,
    attention_backend=AttnBackend.local,
    cuda_graph_impl="local",
    inference_cuda_graph_scope=InferenceCudaGraphScope.block,
    moe_pad_experts_for_cuda_graph_inference=False,
    mamba_state_dim=128,
    mamba_head_dim=64,
    mamba_num_groups=8,
    mamba_num_heads=64,
)


def _make_base_config(**overrides):
    return TransformerConfig(**{**NANOV3_BASE, **overrides})


@pytest.mark.internal
class TestSharedExpertTokenBucketing:
    """Pure-function properties of the bucket ladder. No distributed state needed."""

    def test_bucket_never_shrinks_and_is_monotone(self):
        previous = 0
        for tokens in range(1, 4 * _NVLS_ENGINE_MAX_TOKENS):
            bucketed = bucket_shared_expert_token_count(tokens)
            assert bucketed >= tokens, f"bucket {bucketed} < requested {tokens}"
            assert bucketed >= previous, "bucket ladder must be monotone non-decreasing"
            previous = bucketed

    def test_bucket_bounds_distinct_shape_count(self):
        """The whole point: far fewer distinct shapes than TOKEN_ROUNDER rounding."""
        span = range(1, 8 * _NVLS_ENGINE_MAX_TOKENS + 1)
        bucketed_shapes = {bucket_shared_expert_token_count(t) for t in span}
        token_rounder_shapes = {
            ((t + TOKEN_ROUNDER - 1) // TOKEN_ROUNDER) * TOKEN_ROUNDER for t in span
        }
        assert len(bucketed_shapes) <= 16
        assert len(bucketed_shapes) < len(token_rounder_shapes) // 4

    def test_bucket_padding_stays_bounded_above_one_octave(self):
        """Above TOKEN_ROUNDER the quarter-octave ladder wastes at most ~50%.

        Waste peaks just past a power of two (65 -> 128) and is ~25% elsewhere.
        Small absolute counts are decode-shaped and skip bucketing entirely.
        """
        for tokens in range(TOKEN_ROUNDER + 1, 8 * _NVLS_ENGINE_MAX_TOKENS):
            bucketed = bucket_shared_expert_token_count(tokens)
            assert (bucketed - tokens) / bucketed <= 0.5, f"excessive padding at {tokens}"

    def test_bucket_is_identity_on_aligned_powers_of_two(self):
        for tokens in [64, 128, 256, 512, 1024, 2048]:
            assert bucket_shared_expert_token_count(tokens) == tokens


@pytest.mark.internal
class TestSharedExpertSideStreamLaunch:
    """Behaviour of the side-stream launch helper against a real SharedExpertMLP."""

    @classmethod
    def setup_class(cls):
        Utils.initialize_model_parallel(1, 1, expert_model_parallel_size=Utils.world_size)
        _set_random_seed(seed_=1234, data_parallel_random_init=False)

    @classmethod
    def teardown_class(cls):
        from megatron.core.inference.symmetric_memory import SymmetricMemoryManager

        SymmetricMemoryManager.destroy()
        Utils.destroy_model_parallel()

    def _make_shared_experts(self):
        from megatron.core.models.gpt.moe_module_specs import get_inference_optimized_moe_spec

        config = _make_base_config(
            expert_model_parallel_size=Utils.world_size,
            inference_moe_token_dispatcher_type="nvls",
        )
        layer = get_inference_optimized_moe_spec()(config=config).cuda().eval()
        assert layer.shared_experts is not None, "config must build shared experts"
        return config, layer.shared_experts

    @pytest.mark.parametrize("local_tokens", _PREFILL_TOKEN_COUNTS)
    def test_padding_does_not_change_numerics(self, local_tokens, monkeypatch):
        """Bucketed padding rows must not perturb the unpadded prefix."""
        from megatron.core.transformer.moe.shared_experts import SharedExpertMLP

        config, shared_experts = self._make_shared_experts()
        hidden_states = torch.randn(
            local_tokens, 1, config.hidden_size, device="cuda", dtype=torch.bfloat16
        )

        with torch.no_grad(), InferenceMode.active():
            bucketed = launch_shared_experts_on_side_stream(shared_experts, hidden_states)
            torch.cuda.current_stream().wait_stream(SharedExpertMLP.stream)

            # Same launch with bucketing disabled -> exact reference.
            monkeypatch.setattr(
                "megatron.core.transformer.moe.token_dispatcher_inference."
                "bucket_shared_expert_token_count",
                lambda tokens: tokens,
            )
            unbucketed = launch_shared_experts_on_side_stream(shared_experts, hidden_states)
            torch.cuda.current_stream().wait_stream(SharedExpertMLP.stream)

        assert bucketed.shape == unbucketed.shape == hidden_states.shape
        torch.testing.assert_close(bucketed, unbucketed, atol=0, rtol=0)

    def test_launch_records_stream_on_input(self, monkeypatch):
        """The main-stream input block must be pinned against allocator recycling."""
        from megatron.core.transformer.moe.shared_experts import SharedExpertMLP

        config, shared_experts = self._make_shared_experts()
        hidden_states = torch.randn(
            128, 1, config.hidden_size, device="cuda", dtype=torch.bfloat16
        )

        recorded = []
        original_record_stream = torch.Tensor.record_stream

        def _spy_record_stream(self, stream):
            recorded.append(stream)
            return original_record_stream(self, stream)

        monkeypatch.setattr(torch.Tensor, "record_stream", _spy_record_stream, raising=False)

        with torch.no_grad(), InferenceMode.active():
            launch_shared_experts_on_side_stream(shared_experts, hidden_states)
            torch.cuda.current_stream().wait_stream(SharedExpertMLP.stream)

        assert SharedExpertMLP.stream in recorded, (
            "shared-expert side-stream launch must record_stream the main-stream input; "
            "without it the caching allocator can recycle the block mid-read"
        )

    def test_prefill_sweep_keeps_side_stream_shape_set_small(self):
        """Drive a realistic prefill spread and count distinct side-stream shapes.

        This is the regression guard for the allocator-churn hang: an unbucketed
        launch produces one shape per distinct token count, each a fresh
        side-stream pool allocation.
        """
        from megatron.core.transformer.moe.shared_experts import SharedExpertMLP

        config, shared_experts = self._make_shared_experts()
        observed_shapes = set()

        with torch.no_grad(), InferenceMode.active():
            for local_tokens in _PREFILL_TOKEN_COUNTS:
                hidden_states = torch.randn(
                    local_tokens, 1, config.hidden_size, device="cuda", dtype=torch.bfloat16
                )
                launch_shared_experts_on_side_stream(shared_experts, hidden_states)
                torch.cuda.current_stream().wait_stream(SharedExpertMLP.stream)
                observed_shapes.add(bucket_shared_expert_token_count(local_tokens))

        assert len(observed_shapes) <= len(_PREFILL_TOKEN_COUNTS) // 2, (
            f"bucketing collapsed {len(_PREFILL_TOKEN_COUNTS)} token counts into "
            f"{len(observed_shapes)} shapes; expected substantially fewer"
        )

    def test_capture_skips_bucketing(self):
        """Under graph capture the pool is private and replay allocates nothing.

        Padding there would bake permanent extra compute into the graph, so the
        launch must feed the unpadded token count through.
        """
        from megatron.core.transformer.moe.shared_experts import SharedExpertMLP

        config, shared_experts = self._make_shared_experts()
        # 100 is not on the bucket ladder; capture must still use exactly 100 rows.
        hidden_states = torch.randn(
            100, 1, config.hidden_size, device="cuda", dtype=torch.bfloat16
        )

        with torch.no_grad(), InferenceMode.active():
            warmup = torch.cuda.Stream()
            warmup.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(warmup):
                for _ in range(3):
                    launch_shared_experts_on_side_stream(shared_experts, hidden_states)
                    torch.cuda.current_stream().wait_stream(SharedExpertMLP.stream)
            torch.cuda.current_stream().wait_stream(warmup)

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = launch_shared_experts_on_side_stream(shared_experts, hidden_states)
                torch.cuda.current_stream().wait_stream(SharedExpertMLP.stream)
            graph.replay()
            torch.cuda.synchronize()

        assert captured.shape == hidden_states.shape
