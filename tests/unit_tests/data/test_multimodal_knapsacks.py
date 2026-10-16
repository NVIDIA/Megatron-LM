# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass

import pytest

from examples.multimodal.data_loading.knapsacks import streaming_prompt_dedup_first_fit_knapsack


@dataclass
class Sample:
    size: int
    prompt_hash: str


def test_streaming_knapsack_avoids_duplicate_prompts():
    samples = [Sample(6, "a"), Sample(4, "a"), Sample(6, "b"), Sample(4, "c")]

    packs = streaming_prompt_dedup_first_fit_knapsack(
        [sample.size for sample in samples], samples, max_capacity=10, tolerance=0
    )

    assert sorted(sum(sample.size for sample in pack) for pack in packs) == [10, 10]
    assert all(len({sample.prompt_hash for sample in pack}) == len(pack) for pack in packs)


def test_streaming_knapsack_rejects_oversized_samples():
    sample = Sample(11, "a")

    with pytest.raises(ValueError, match="exceeds max_capacity"):
        streaming_prompt_dedup_first_fit_knapsack([sample.size], [sample], max_capacity=10)
