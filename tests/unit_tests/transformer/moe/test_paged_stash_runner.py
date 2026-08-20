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
        self.invalidate_count = 0
        self.reset_count = 0

    def check_over_budget(self):
        return None

    def reset_over_budget(self):
        self.reset_count += 1

    def invalidate_ep_bootstrap(self):
        self.invalidate_count += 1


class _FakeMoELayer(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_dispatcher = _FakeTokenDispatcher(config)


class _FakeTransformerLayer(torch.nn.Module):

    def __init__(self, mlp):
        super().__init__()
        self.mlp = mlp


class _FakeStack(torch.nn.Module):

    def __init__(self, layer):
        super().__init__()
        self.layers = torch.nn.ModuleList([layer])


class _FakeMTPPredictionLayer(torch.nn.Module):

    def __init__(self, mtp_model_layer):
        super().__init__()
        self.mtp_model_layer = mtp_model_layer


class _FakeModelChunk(torch.nn.Module):

    def __init__(self, config, decoder_moe, mtp_moe, nested_mtp):
        super().__init__()
        self.config = config
        self.decoder = _FakeStack(_FakeTransformerLayer(decoder_moe))
        mtp_model_layer = _FakeTransformerLayer(mtp_moe)
        if nested_mtp:
            mtp_model_layer = _FakeStack(mtp_model_layer)
        self.mtp = _FakeStack(_FakeMTPPredictionLayer(mtp_model_layer))
        self.mtp_process = True
        # Register the decoder MoE through a second path to verify identity deduplication.
        self.duplicate_decoder_moe = decoder_moe
        self.zero_grad_count = 0

    def zero_grad_buffer(self):
        self.zero_grad_count += 1


def _run_retry(
    monkeypatch, training_config, model_config, decoder_config, mtp_config, nested_mtp=False
):
    monkeypatch.setattr(
        "megatron.core.transformer.multi_token_prediction.MultiTokenPredictionLayer",
        _FakeMTPPredictionLayer,
    )
    decoder_moe = _FakeMoELayer(decoder_config)
    mtp_moe = _FakeMoELayer(mtp_config)
    model = _FakeModelChunk(model_config, decoder_moe, mtp_moe, nested_mtp=nested_mtp)

    values_seen_by_forward = []

    def forward_backward_func(**_):
        values_seen_by_forward.append(
            (
                training_config.moe_paged_stash,
                model_config.moe_paged_stash,
                decoder_config.moe_paged_stash,
                mtp_config.moe_paged_stash,
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

    return SimpleNamespace(
        runner=runner,
        result=result,
        model=model,
        decoder_moe=decoder_moe,
        mtp_moe=mtp_moe,
        values_seen_by_forward=values_seen_by_forward,
        release_stash_buffer_calls=release_stash_buffer_calls,
    )


def test_retry_preserves_shared_root_config_behavior(monkeypatch):
    """Models whose MoE modules share the root config retain their existing behavior."""
    training_config = _config(True)
    model_config = _config(True)

    run = _run_retry(
        monkeypatch,
        training_config=training_config,
        model_config=model_config,
        decoder_config=model_config,
        mtp_config=model_config,
    )

    assert run.runner.moe_layers == [run.decoder_moe, run.mtp_moe]
    assert [id(config) for config in run.runner._configs_to_sync_moe_paged_stash] == [
        id(training_config),
        id(model_config),
    ]
    assert run.values_seen_by_forward == [(True, True, True, True), (False, False, False, False)]
    assert run.result == 2
    assert run.model.zero_grad_count == 1
    assert len(run.release_stash_buffer_calls) == 1
    assert run.decoder_moe.token_dispatcher.reset_count == 1
    assert run.mtp_moe.token_dispatcher.reset_count == 1
    assert run.decoder_moe.token_dispatcher.invalidate_count == 2
    assert run.mtp_moe.token_dispatcher.invalidate_count == 2
    assert run.decoder_moe.token_dispatcher._comm_manager.moe_expert_rank_capacity_factor == 1.5
    assert run.mtp_moe.token_dispatcher._comm_manager.moe_expert_rank_capacity_factor == 1.5
    assert training_config.moe_paged_stash is True
    assert model_config.moe_paged_stash is True


def test_retry_disables_and_restores_per_module_configs(monkeypatch):
    """Retry must disable direct and nested-MTP MoE configs, then restore each value."""
    training_config = _config(True)
    model_config = _config(True)
    decoder_moe_config = _config(True)
    mtp_moe_config = _config(False)

    run = _run_retry(
        monkeypatch,
        training_config=training_config,
        model_config=model_config,
        decoder_config=decoder_moe_config,
        mtp_config=mtp_moe_config,
        nested_mtp=True,
    )

    assert run.runner.moe_layers == [run.decoder_moe]
    assert [id(config) for config in run.runner._configs_to_sync_moe_paged_stash] == [
        id(training_config),
        id(model_config),
        id(decoder_moe_config),
        id(mtp_moe_config),
    ]
    assert run.values_seen_by_forward == [(True, True, True, False), (False, False, False, False)]
    assert run.result == 2
    assert run.model.zero_grad_count == 1
    assert len(run.release_stash_buffer_calls) == 1
    assert run.decoder_moe.token_dispatcher.reset_count == 1
    assert run.mtp_moe.token_dispatcher.reset_count == 0
    assert run.decoder_moe.token_dispatcher.invalidate_count == 2
    assert run.mtp_moe.token_dispatcher.invalidate_count == 0
    assert run.decoder_moe.token_dispatcher._comm_manager.moe_expert_rank_capacity_factor == 1.5
    assert run.mtp_moe.token_dispatcher._comm_manager.moe_expert_rank_capacity_factor == 1.5
    assert (
        training_config.moe_paged_stash,
        model_config.moe_paged_stash,
        decoder_moe_config.moe_paged_stash,
        mtp_moe_config.moe_paged_stash,
    ) == (True, True, True, False)
