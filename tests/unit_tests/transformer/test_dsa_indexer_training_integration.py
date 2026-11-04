# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Regression tests for the DSA indexer's training-loop integration.

These cover the parts of DSA that live in megatron/training rather than in the attention layer.
Each one pins a bug that failed silently: no exception, no crash, just a run that trains the
wrong parameters or reports the wrong number. The loudly-failing paths around them are left to
the convergence run rather than mocked here.
"""

from types import SimpleNamespace

import torch

from megatron.core.models.hybrid.hybrid_layer_allocation import parse_hybrid_pattern
from megatron.training import training


def test_dsa_reset_on_load_initializes_only_a_checkpoint_without_an_indexer():
    """The flag converts a dense checkpoint, so it keys off what the checkpoint holds.

    Keying off args.iteration instead coupled this to --finetune in both directions: it
    refused to convert under a plain --load, and it happily re-initialised an already-trained
    indexer when --finetune zeroed the iteration. The second one silently destroys work.
    """
    # A dense checkpoint has no indexer to preserve, so the conversion runs -- and now does
    # so under a plain --load, without requiring --finetune.
    dense = SimpleNamespace(dsa_reset_indexer_on_load=True, checkpoint_has_dsa_indexer=False)
    assert training._should_reset_dsa_indexer_after_load(dense)

    # A checkpoint that already carries an indexer is never re-initialised, whether this is a
    # requeue or a --finetune run that left the flag set in its launch script.
    trained = SimpleNamespace(dsa_reset_indexer_on_load=True, checkpoint_has_dsa_indexer=True)
    assert not training._should_reset_dsa_indexer_after_load(trained)

    # Without the flag the initialization never runs.
    off = SimpleNamespace(dsa_reset_indexer_on_load=False, checkpoint_has_dsa_indexer=False)
    assert not training._should_reset_dsa_indexer_after_load(off)

    # An absent attribute means no checkpoint args were recorded, which is the no-indexer case.
    missing = SimpleNamespace(dsa_reset_indexer_on_load=True)
    assert training._should_reset_dsa_indexer_after_load(missing)


def test_dsa_train_indexer_only_freezes_moe_router_expert_bias(monkeypatch):
    """The router's expert_bias is a buffer, so requires_grad does not reach it.

    finalize_model_grads rewrites it every step from observed token counts for load balancing,
    and only frozen_expert_bias stops that. Without this the backbone's routing keeps moving
    while the run is meant to train the indexer alone.
    """

    class _Router(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.frozen_expert_bias = False
            self.register_buffer("expert_bias", torch.zeros(4))

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.block = torch.nn.Module()
            self.block.indexer = torch.nn.Linear(3, 2)
            self.block.mlp = torch.nn.Module()
            self.block.mlp.router = _Router()

    model = _Model()
    monkeypatch.setattr(training, "_global_dsa_indexer_reset_count", lambda count: count)

    training._freeze_non_dsa_indexer_parameters([model])

    assert model.block.mlp.router.frozen_expert_bias is True


def test_dsa_indexer_optimizer_refresh_copies_only_owned_distributed_shard(monkeypatch):
    """Under the distributed optimizer a rank owns a slice, not the whole parameter.

    Copying the full tensor into a shard-sized master buffer corrupts the ranks that own the
    rest of it, which surfaces much later as divergence rather than as an error here.
    """
    model = torch.nn.Module()
    model.indexer = torch.nn.Linear(4, 2, bias=False)
    with torch.no_grad():
        model.indexer.weight.copy_(torch.arange(8, dtype=torch.float32).reshape(2, 4))
    indexer_shard = torch.full((3,), -1.0, dtype=torch.float32)

    class _Optimizer:
        is_stub_optimizer = False
        config = SimpleNamespace(use_precision_aware_optimizer_no_fp8_or_ds_fp8=False)

        @staticmethod
        def _get_model_param_range_map(param):
            assert param is model.indexer.weight
            return {"param": SimpleNamespace(start=2, end=5)}

    monkeypatch.setattr(
        training,
        "get_model_to_optimizer_param_map",
        lambda optimizer: {model.indexer.weight: indexer_shard},
    )

    assert training._reload_dsa_indexer_optimizer_params([model], _Optimizer()) == 1
    torch.testing.assert_close(indexer_shard, torch.tensor([2.0, 3.0, 4.0]))


def _indexer_count_args(pattern, num_layers, mtp_num_layers=0):
    return SimpleNamespace(
        num_layers=num_layers,
        mtp_num_layers=mtp_num_layers,
        mtp_use_repeated_layer=False,
        hybrid_layer_pattern=pattern,
        csa_compress_ratios=None,
        experimental_attention_variant='dsa',
    )


def test_indexer_layer_count_counts_attention_layers_not_every_layer(monkeypatch):
    """The tracker spans every decoder layer; only the attention layers own an indexer.

    Averaging over the tracker length instead of the indexer count under-reports the loss by the
    ratio between them -- 13x for the 8B pattern, which is four attention layers in fifty-two.
    """
    monkeypatch.setattr(training, "is_hybrid_model", lambda args: True)
    pattern = "M-" * 24 + "*-" * 2  # 48 mamba/mlp layers, then 2 attention layers
    args = _indexer_count_args(pattern, num_layers=len(pattern))

    tracker_layers, indexer_layers = training._get_indexer_logging_layer_counts(args)

    assert tracker_layers == len(pattern)
    assert indexer_layers == 2

    # MTP repeats its pattern once per prediction depth, and each copy owns an indexer.
    mtp_args = _indexer_count_args("M*M*/M*", num_layers=4, mtp_num_layers=2)
    tracker_layers, indexer_layers = training._get_indexer_logging_layer_counts(mtp_args)
    depths = parse_hybrid_pattern(mtp_args.hybrid_layer_pattern).mtp_num_depths
    assert (tracker_layers, indexer_layers) == (4 + 2, 2 + depths)

    # A non-DSA run keeps the old behaviour: no count, so the tracker length is used.
    plain_args = _indexer_count_args("M*M*", num_layers=4)
    plain_args.experimental_attention_variant = None
    assert training._get_indexer_logging_layer_counts(plain_args) == (4, None)
