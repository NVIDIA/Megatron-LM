# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import torch

from megatron.core.transformer.moe.paged_stash import PagedStashManager, PagedStashRunner


def _config(moe_paged_stash):
    return SimpleNamespace(moe_paged_stash=moe_paged_stash, moe_expert_rank_capacity_factor=1.5)


class _FakeTokenDispatcher:

    def __init__(self, config):
        self.config = config
        self._comm_manager = SimpleNamespace(moe_expert_rank_capacity_factor=1.5)
        self.reset_count = 0

    def check_over_budget(self):
        return None

    def reset_over_budget(self):
        self.reset_count += 1


class _FakeMoELayer(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_dispatcher = _FakeTokenDispatcher(config)


class _FakeTransformerLayer(torch.nn.Module):

    def __init__(self, mlp):
        super().__init__()
        self.mlp = mlp


class _FakeHybridStack(torch.nn.Module):

    def __init__(self, layer):
        super().__init__()
        self.layers = torch.nn.ModuleList([layer])


class _FakeMTPPredictionLayer(torch.nn.Module):

    def __init__(self, mtp_model_layer):
        super().__init__()
        self.mtp_model_layer = mtp_model_layer


class _FakeModelChunk(torch.nn.Module):

    def __init__(self, config, decoder_moe, mtp_moe):
        super().__init__()
        self.config = config
        self.decoder = _FakeHybridStack(_FakeTransformerLayer(decoder_moe))
        self.mtp = _FakeHybridStack(
            _FakeMTPPredictionLayer(_FakeHybridStack(_FakeTransformerLayer(mtp_moe)))
        )
        # Register the decoder MoE through a second path to verify identity deduplication.
        self.duplicate_decoder_moe = decoder_moe
        self.zero_grad_count = 0

    def zero_grad_buffer(self):
        self.zero_grad_count += 1


def test_retry_disables_and_restores_per_layer_configs(monkeypatch):
    """Retry must disable direct and nested-MTP MoE configs, then restore each value."""
    training_config = _config(True)
    model_config = _config(True)
    decoder_moe_config = _config(True)
    mtp_moe_config = _config(False)
    decoder_moe = _FakeMoELayer(decoder_moe_config)
    mtp_moe = _FakeMoELayer(mtp_moe_config)
    model = _FakeModelChunk(model_config, decoder_moe, mtp_moe)

    values_seen_by_forward = []

    def forward_backward_func(**_):
        values_seen_by_forward.append(
            (
                training_config.moe_paged_stash,
                model_config.moe_paged_stash,
                decoder_moe_config.moe_paged_stash,
                mtp_moe_config.moe_paged_stash,
            )
        )
        return len(values_seen_by_forward)

    release_stash_buffer_calls = []
    fake_stash_manager = SimpleNamespace(
        overflow=None,
        host_spill=None,
        release_stash_buffers=lambda: release_stash_buffer_calls.append(None),
    )
    monkeypatch.setattr(PagedStashManager, 'STASH_MGR', fake_stash_manager)
    runner = PagedStashRunner(
        config=training_config,
        copy_main_params=False,
        model=[model],
        optimizer=None,
        forward_backward_func=forward_backward_func,
    )
    overflow_results = iter([(1, 0, 0), (0, 0, 0)])
    runner.check_moe_overflow = lambda: next(overflow_results)

    result = runner(
        model=[model], data_iterator=None, num_microbatches=1, seq_length=1, forward_only=False
    )

    assert runner.moe_layers == [decoder_moe, mtp_moe]
    assert [id(config) for config in runner._configs_to_sync_moe_paged_stash] == [
        id(training_config),
        id(model_config),
        id(decoder_moe_config),
        id(mtp_moe_config),
    ]
    assert values_seen_by_forward == [(True, True, True, False), (False, False, False, False)]
    assert result == 2
    assert model.zero_grad_count == 1
    assert len(release_stash_buffer_calls) == 1
    assert decoder_moe.token_dispatcher.reset_count == 1
    assert mtp_moe.token_dispatcher.reset_count == 1
    assert decoder_moe.token_dispatcher._comm_manager.moe_expert_rank_capacity_factor == 1.5
    assert mtp_moe.token_dispatcher._comm_manager.moe_expert_rank_capacity_factor == 1.5
    assert (
        training_config.moe_paged_stash,
        model_config.moe_paged_stash,
        decoder_moe_config.moe_paged_stash,
        mtp_moe_config.moe_paged_stash,
    ) == (True, True, True, False)
