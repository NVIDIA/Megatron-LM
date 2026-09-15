# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Focused differential coverage for RECOMPUTE suspension continuity."""

from collections import Counter

import pytest
import torch

from megatron.core.utils import is_fa_min_version
from tests.unit_tests.inference.engines.request_lifecycle_test_utils import (
    RequestLifecyclePairwiseBase,
    _install_incrementing_logits,
)
from tests.unit_tests.inference.engines.test_dynamic_engine_async_sched import _AsyncPairScenario

_CHUNKED_RECOMPUTE = _AsyncPairScenario(
    name="chunked-partial-recompute-api-coordinator",
    pairs=("prefill:chunked", "kv:recompute", "api:coordinator"),
    config={
        "enable_chunked_prefill": True,
        "context_max_tokens": 16,
        "kv_cache_management_mode": "recompute",
        "static_kv_memory_pointers": False,
        "materialize_only_last_token_logits": False,
        "return_log_probs": True,
    },
    sampling=(
        {
            "return_log_probs": True,
            "skip_prompt_log_probs": False,
            "top_n_logprobs": 3,
            "return_prompt_tokens": True,
        },
    ),
    signals=("chunked", "logprobs", "top-n"),
)


@pytest.mark.internal
@pytest.mark.skipif(
    not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
)
class TestRequestLifecycleRecomputeContinuity(RequestLifecyclePairwiseBase):

    @classmethod
    def _build_test_env(cls, test_config):
        env = super()._build_test_env(test_config)
        _install_incrementing_logits(env, Counter(), test_config.vocab_size)
        return env

    @torch.inference_mode()
    def test_chunked_partial_recompute_api_coordinator(self):
        result = self._assert_pair(_CHUNKED_RECOMPUTE, coordinator=True)
        assert result.witness["finished_chunk_token_count"] > 0
        assert result.checkpoint_counts == {3: 1, 1: 1, 0: 0}
