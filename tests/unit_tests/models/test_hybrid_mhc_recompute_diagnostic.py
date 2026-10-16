# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Seeded reproduction of the unresolved FP32 block-recompute failure.

Eight independent TP=PP=1 replicas explore predefined seeds, not TP parity.
A pass cannot resolve the original failure because its CPU RNG was unrecorded.
"""

import hashlib
import json
import sys
import traceback

import pytest
import torch

from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from tests.unit_tests.models.test_hybrid_mhc import (
    TestHybridMHCContraction as _OriginalHybridMHCContraction,
)
from tests.unit_tests.test_utilities import Utils

_SEEDS = (123, 2718, 31415, 5353, 7054, 7521, 7522, 7523)


def _fingerprint(state):
    return hashlib.sha256(state.cpu().contiguous().numpy().tobytes()).hexdigest()


@pytest.mark.internal
class TestHybridMHCRecomputeDiagnostic:
    def setup_method(self):
        Utils.initialize_model_parallel(1, 1)
        rank = torch.distributed.get_rank()
        seed = _SEEDS[rank % len(_SEEDS)]
        torch.manual_seed(seed)
        model_parallel_cuda_manual_seed(seed)
        print(
            "MHC_DIAGNOSTIC "
            + json.dumps(
                dict(
                    rank=rank,
                    seed=seed,
                    cpu_seed=torch.initial_seed(),
                    cuda_seed=torch.cuda.initial_seed(),
                    cpu_rng_sha256=_fingerprint(torch.get_rng_state()),
                    cuda_rng_sha256=_fingerprint(torch.cuda.get_rng_state()),
                ),
                sort_keys=True,
            ),
            flush=True,
        )

    def teardown_method(self):
        Utils.destroy_model_parallel()

    def test_original_fp32_block_learned_recompute(self):
        try:
            _OriginalHybridMHCContraction().test_full_recompute_matches_forward_and_all_gradients(
                learned_output_contract=True,
                recompute_method="block",
                fp32_mixing=False,
                autocast_forward=False,
                distribute_saved_activations=False,
            )
        except BaseException:
            traceback.print_exc(file=sys.stdout)
            sys.stdout.flush()
            sys.stderr.flush()
            raise
