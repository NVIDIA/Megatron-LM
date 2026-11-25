# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU Core checkpoint RNG snapshot fixture; no production RNG changes."""
import pytest
import torch


@pytest.fixture
def cpu_core_checkpoint(monkeypatch):
    """CPU-only CUDA RNG snapshot adapter; restore it before leaving the fixture."""
    from megatron.core.tensor_parallel import random as core_random

    def implementation():
        return (
            core_random.CheckpointFunction,
            core_random.CheckpointFunction.forward,
            core_random.CheckpointFunction.backward,
            torch.autograd.backward,
        )

    original = implementation()
    get_rng, set_rng = core_random._get_all_rng_states, core_random._set_all_rng_states
    with monkeypatch.context() as fixture:
        fixture.setattr(core_random, "_get_all_rng_states", lambda: (torch.get_rng_state(),))
        fixture.setattr(
            core_random, "_set_all_rng_states", lambda state: torch.set_rng_state(state)
        )
        assert original == implementation()
        yield core_random.checkpoint
    assert core_random._get_all_rng_states is get_rng and core_random._set_all_rng_states is set_rng
    assert original == implementation()
