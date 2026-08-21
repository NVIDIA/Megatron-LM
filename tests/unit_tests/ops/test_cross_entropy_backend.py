# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The vocab-parallel cross entropy backends, against each other and against Torch.

Which of them a run gets is a provider's decision and is tested where the providers live.
What is tested here is that they are interchangeable, which is what makes that decision safe.
"""

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.ops.loss import (
    LossMegatron,
    LossMegatronFused,
    fused_vocab_parallel_cross_entropy,
    vocab_parallel_cross_entropy,
)
from tests.unit_tests.test_utilities import Utils

BACKENDS = [("megatron", LossMegatron), ("megatron_fused", LossMegatronFused)]


class TestTargets:
    def test_each_backend_returns_its_own_kernel(self):
        assert LossMegatron().vocab_parallel_cross_entropy() is vocab_parallel_cross_entropy
        assert (
            LossMegatronFused().vocab_parallel_cross_entropy() is fused_vocab_parallel_cross_entropy
        )

    def test_the_fused_backend_declares_it_is_not_bit_exact(self):
        """--deterministic-mode has to be able to tell these apart."""
        assert LossMegatron().DETERMINISM == "deterministic"
        assert LossMegatronFused().DETERMINISM == "nondeterministic"


class TestDistributedContract:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize(("name", "backend"), BACKENDS)
    def test_target_accepts_a_default_tensor_parallel_group(self, name, backend):
        """tp_group=None means the default group, whatever the underlying kernel needs."""
        del name
        torch.manual_seed(0)
        logits = torch.randn(4, 2, 8).cuda()
        labels = torch.randint(0, 8, (4, 2)).cuda()

        target = backend().vocab_parallel_cross_entropy()
        assert target(logits, labels, None).shape == labels.shape


class TestCrossEntropyParity:
    """Every backend has to agree on the same inputs, or swapping one changes the model."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_every_target_matches_torch_cross_entropy(self, dtype):
        """Each target gets its own copy: the family contract says they consume logits."""
        torch.manual_seed(1234)
        logits = torch.randn(6, 2, 16).cuda().to(dtype)
        labels = torch.randint(0, 16, (6, 2)).cuda()
        tp_group = parallel_state.get_tensor_model_parallel_group()

        expected = torch.nn.functional.cross_entropy(
            logits.float().reshape(-1, 16), labels.reshape(-1), reduction="none"
        ).reshape(6, 2)

        for name, backend in BACKENDS:
            target = backend().vocab_parallel_cross_entropy()
            loss = target(logits.clone(), labels, tp_group)
            torch.testing.assert_close(
                loss.float(), expected, rtol=1e-2, atol=1e-2, msg=lambda m: f"{name}: {m}"
            )
