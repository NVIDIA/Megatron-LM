# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Cross entropy is chosen while the model is built, not on every forward."""

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.ops import BackendOptions, build_spec_provider
from megatron.core.ops.loss import fused_vocab_parallel_cross_entropy, vocab_parallel_cross_entropy
from tests.unit_tests.test_utilities import Utils

_needs_te = pytest.mark.skipif(not HAVE_TE, reason="Transformer Engine is required")


def _provider(**kwargs):
    return build_spec_provider(BackendOptions(transformer_impl="local", **kwargs))


class TestCrossEntropySelection:
    def test_default_is_the_reference_implementation(self):
        assert _provider().vocab_parallel_cross_entropy() is vocab_parallel_cross_entropy

    def test_fusion_off_ignores_the_implementation_setting(self):
        provider = _provider(cross_entropy_loss_fusion=False, cross_entropy_fusion_impl="te")
        assert provider.vocab_parallel_cross_entropy() is vocab_parallel_cross_entropy

    def test_native_fusion_selects_the_megatron_kernel(self):
        provider = _provider(cross_entropy_loss_fusion=True, cross_entropy_fusion_impl="native")
        assert provider.vocab_parallel_cross_entropy() is fused_vocab_parallel_cross_entropy

    def test_unknown_fusion_implementation_is_rejected(self):
        """This used to fall through the branch and raise UnboundLocalError mid-forward."""
        with pytest.raises(ValueError, match="Unknown cross_entropy_fusion_impl='jax'"):
            _provider(cross_entropy_loss_fusion=True, cross_entropy_fusion_impl="jax")

    def test_explicit_override_selects_the_same_kernel(self):
        provider = _provider(operation_backends={"vocab_parallel_cross_entropy": "megatron_fused"})
        assert provider.vocab_parallel_cross_entropy() is fused_vocab_parallel_cross_entropy

    def test_te_fusion_is_refused_the_same_way_the_flag_check_refuses_it(self):
        """--op-backend must not route around the stability block in arguments.py."""
        with pytest.raises(ValueError, match="disabled due to stability issues"):
            _provider(cross_entropy_loss_fusion=True, cross_entropy_fusion_impl="te")
        with pytest.raises(ValueError, match="Unknown backend 'te_fused'"):
            _provider(operation_backends={"vocab_parallel_cross_entropy": "te_fused"})


class TestCrossEntropyContract:
    """Every target in the family has to be callable the same way."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("backend_name", ["megatron", "megatron_fused"])
    def test_target_accepts_a_default_tensor_parallel_group(self, backend_name):
        """tp_group=None means the default group, whatever the underlying kernel needs."""
        torch.manual_seed(0)
        logits = torch.randn(4, 2, 8).cuda()
        labels = torch.randint(0, 8, (4, 2)).cuda()

        target = _provider(
            operation_backends={"vocab_parallel_cross_entropy": backend_name}
        ).vocab_parallel_cross_entropy()

        assert target(logits, labels, None).shape == labels.shape


class TestCrossEntropyParity:
    """Every selectable target has to agree on the same inputs."""

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

        names = ["megatron", "megatron_fused"]
        for name in names:
            target = _provider(
                operation_backends={"vocab_parallel_cross_entropy": name}
            ).vocab_parallel_cross_entropy()
            loss = target(logits.clone(), labels, tp_group)
            torch.testing.assert_close(
                loss.float(), expected, rtol=1e-2, atol=1e-2, msg=lambda m: f"{name}: {m}"
            )
