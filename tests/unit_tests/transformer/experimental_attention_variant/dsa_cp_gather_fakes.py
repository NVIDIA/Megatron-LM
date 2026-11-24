# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Route DSAttention's CP all-gathers, synchronous and deferred, through one test fake.

DSAttention gathers the indexer keys with ``gather_from_sequence_parallel_region`` and launches
the latent K/V gathers with ``async_gather_from_sequence_parallel_region`` (finished later
through the returned handle). Tests that simulate CP ranks fake both with one function.
"""

_DSA_MODULE = "megatron.core.transformer.experimental_attention_variant.dsa"


class CompletedGather:
    """An ``async_gather_from_sequence_parallel_region`` handle whose gather already ran."""

    def __init__(self, tensor):
        self.tensor = tensor

    def wait(self):
        """Return the gathered tensor."""
        return self.tensor


def fake_async_gather(fake_gather):
    """Wrap ``fake_gather(tensor, group)`` as an asynchronous gather that completes at once."""

    def launch(tensor, tensor_parallel_output_grad=True, group=None):
        return CompletedGather(fake_gather(tensor, group=group))

    return launch


def patch_dsa_cp_gather(monkeypatch, fake_gather):
    """Make DSAttention's synchronous and deferred CP gathers call ``fake_gather``."""
    monkeypatch.setattr(f"{_DSA_MODULE}.gather_from_sequence_parallel_region", fake_gather)
    monkeypatch.setattr(
        f"{_DSA_MODULE}.async_gather_from_sequence_parallel_region", fake_async_gather(fake_gather)
    )
