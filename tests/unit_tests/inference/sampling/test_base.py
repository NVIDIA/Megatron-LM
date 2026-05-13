# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

from megatron.core.inference.sampling.base import Sampling


class _FakeContext:
    pass


class TestSampling:

    def test_cannot_instantiate_abstract_directly(self):
        """Sampling is abstract; instantiation must fail."""
        with pytest.raises(TypeError):
            Sampling()

    def test_subclass_must_implement_sample_kernel(self):
        """A subclass that doesn't override sample_kernel cannot be instantiated."""

        class IncompleteSampler(Sampling):
            pass

        with pytest.raises(TypeError):
            IncompleteSampler()

    def test_concrete_subclass_can_instantiate(self):
        """A subclass that implements sample_kernel can be instantiated."""

        class IdentitySampler(Sampling):
            def sample_kernel(
                self,
                logits,
                n,
                context,
                *,
                gather_indices=None,
                token_to_request_index=None,
                eager=False,
                cache_key=None,
            ):
                return torch.arange(n)

        sampler = IdentitySampler()
        out = sampler.sample_kernel(torch.zeros(4, 8), 4, _FakeContext())
        assert out.tolist() == [0, 1, 2, 3]

    def test_sample_speculative_builds_token_to_request_index(self):
        """sample_speculative passes a per-token request mapping to sample_kernel."""

        captured = {}

        class CapturingSampler(Sampling):
            def sample_kernel(
                self,
                logits,
                n,
                context,
                *,
                gather_indices=None,
                token_to_request_index=None,
                eager=False,
                cache_key=None,
            ):
                captured["n"] = n
                captured["mapping"] = token_to_request_index.cpu().tolist()
                captured["eager"] = eager
                return torch.zeros(n, dtype=torch.int64)

        sampler = CapturingSampler()
        # 2 decode requests, each contributing 1 + 2 = 3 tokens; 1 prefill request contributing 1.
        # Total tokens = 6 + 1 = 7.
        logits = torch.zeros(7, 8)
        out = sampler.sample_speculative(
            logits, num_decode=2, num_prefill=1, num_speculative_tokens=2, context=_FakeContext()
        )
        assert out.shape == (7,)
        # token_to_request_index = [0,0,0,1,1,1, 2]  (decode tokens grouped by request, then prefill)
        assert captured["mapping"] == [0, 0, 0, 1, 1, 1, 2]
        # sample_kernel must be called with eager=True (the docstring guarantees this).
        assert captured["eager"] is True
        assert captured["n"] == 7

    def test_sample_speculative_zero_decode(self):
        """sample_speculative handles num_decode=0 (only prefill)."""

        captured = {}

        class CapturingSampler(Sampling):
            def sample_kernel(
                self,
                logits,
                n,
                context,
                *,
                gather_indices=None,
                token_to_request_index=None,
                eager=False,
                cache_key=None,
            ):
                captured["n"] = n
                captured["mapping"] = token_to_request_index.cpu().tolist()
                return torch.zeros(n, dtype=torch.int64)

        sampler = CapturingSampler()
        logits = torch.zeros(3, 8)
        sampler.sample_speculative(
            logits, num_decode=0, num_prefill=3, num_speculative_tokens=4, context=_FakeContext()
        )
        # No decode → token_to_request_index = [0, 1, 2] (just the prefill requests).
        assert captured["mapping"] == [0, 1, 2]
        assert captured["n"] == 3

    def test_sample_speculative_forwards_gather_indices(self):
        """sample_speculative passes gather_indices through to sample_kernel."""

        captured = {}

        class CapturingSampler(Sampling):
            def sample_kernel(
                self,
                logits,
                n,
                context,
                *,
                gather_indices=None,
                token_to_request_index=None,
                eager=False,
                cache_key=None,
            ):
                captured["gather_indices"] = gather_indices
                return torch.zeros(n, dtype=torch.int64)

        sampler = CapturingSampler()
        gather = torch.arange(4)
        sampler.sample_speculative(
            torch.zeros(4, 8),
            num_decode=1,
            num_prefill=1,
            num_speculative_tokens=2,
            context=_FakeContext(),
            gather_indices=gather,
        )
        assert torch.equal(captured["gather_indices"], gather)
