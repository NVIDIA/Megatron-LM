# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Shared-prefix HybridModel forwards reject training-mode MTP and otherwise skip it."""

import pytest
import torch

from tests.unit_tests.models.hybrid.shared_prefix_test_utils import (
    TokenProblem,
    build_hybrid_model,
    clear_attention_env,
    run_shared,
)
from tests.unit_tests.ssm.shared_prefix_test_utils import SharedPrefixProblem
from tests.unit_tests.test_utilities import Utils

MTP_PATTERN = "M*EM*E/*E/*E"


class _BackboneReached(Exception):
    pass


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
class TestSharedPrefixMTPRejected:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def _run(self, monkeypatch, *, training, **forward_kwargs):
        """Run a shared forward on an MTP model; stop it once the backbone starts."""
        from megatron.core.models.hybrid import shared_prefix

        def backbone(*args, **kwargs):
            raise _BackboneReached

        monkeypatch.setattr(shared_prefix, "forward_hybrid_stack_shared_prefix", backbone)
        clear_attention_env(monkeypatch)
        torch.manual_seed(0)
        model = build_hybrid_model(MTP_PATTERN, torch.bfloat16)
        model.train(training)
        tokens = TokenProblem(
            SharedPrefixProblem(((5, (6, 3)),), padding_multiple=8, topology_multiple=8),
            vocab_size=2048,
            seed=1,
        )
        run_shared(model, tokens, tokens.problem.layout(forest=False), **forward_kwargs)

    def test_training_mtp_is_rejected_before_the_backbone(self, monkeypatch):
        with pytest.raises(NotImplementedError, match="does not support MTP yet"):
            self._run(monkeypatch, training=True)

    @pytest.mark.parametrize(
        "training, forward_kwargs",
        [(True, {"compute_mtp_loss": False}), (False, {})],
        ids=["compute_mtp_loss_false", "eval"],
    )
    def test_mtp_is_skipped_otherwise(self, monkeypatch, training, forward_kwargs):
        with pytest.raises(_BackboneReached):
            self._run(monkeypatch, training=training, **forward_kwargs)
