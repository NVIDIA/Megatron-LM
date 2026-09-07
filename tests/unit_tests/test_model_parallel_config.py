# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import sys

import pytest

from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.arguments import parse_args, validate_args


def test_te_cross_entropy_loss_fusion_warns_in_model_parallel_config():
    with pytest.warns(UserWarning, match="known stability issues"):
        config = ModelParallelConfig(cross_entropy_loss_fusion=True, cross_entropy_fusion_impl='te')

    assert config.cross_entropy_loss_fusion
    assert config.cross_entropy_fusion_impl == 'te'


def test_native_cross_entropy_loss_fusion_is_allowed():
    config = ModelParallelConfig(cross_entropy_loss_fusion=True, cross_entropy_fusion_impl='native')

    assert config.cross_entropy_loss_fusion
    assert config.cross_entropy_fusion_impl == 'native'


def test_invalid_thd_tail_padding_policy_is_rejected_during_config_initialization():
    with pytest.raises(ValueError, match="thd_tail_padding_policy must be"):
        ModelParallelConfig(thd_tail_padding_policy="bogus")


def test_contiguous_context_parallel_rejects_bshd_inputs():
    with pytest.raises(
        ValueError,
        match="cp_partition_mode='contiguous'.*requires THD.*BSHD inputs are not supported",
    ):
        TransformerConfig(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=4,
            context_parallel_size=2,
            cp_partition_mode="contiguous",
        )


def test_te_cross_entropy_loss_fusion_is_disabled_by_training_args(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['test_model_parallel_config.py'])
    args = parse_args()
    args.num_layers = 2
    args.hidden_size = 128
    args.num_attention_heads = 4
    args.max_position_embeddings = 1024
    args.seq_length = 1024
    args.micro_batch_size = 1
    # Let validate_args derive a global batch size that is valid for the
    # active data-parallel size in distributed unit-test jobs.
    args.train_iters = 1
    args.lr = 1e-4
    args.tokenizer_type = 'NullTokenizer'
    args.vocab_size = 1024
    args.cross_entropy_loss_fusion = True
    args.cross_entropy_fusion_impl = 'te'

    with pytest.raises(AssertionError, match="Transformer Engine cross entropy loss fusion"):
        validate_args(args)


@pytest.mark.parametrize(
    ('overrides', 'match'),
    [
        ({'perform_rl_step': True}, 'do not support --perform-rl-step'),
        ({'inprocess_restart': True}, 'do not support --inprocess-restart'),
        ({'save_wgrads_interval': 7}, 'require --save-wgrads-interval'),
        ({'logits_load_dir': '/tmp/teacher'}, 'do not support --logits-load-dir'),
        ({'logits_save_dir': '/tmp/student'}, 'do not support --logits-save-dir'),
        ({'save_activations_interval': 5}, 'do not support --save-activations-interval'),
        ({'save_dgrads_interval': 5}, 'do not support --save-dgrads-interval'),
        ({'check_for_spiky_loss': True}, 'do not support --check-for-spiky-loss'),
    ],
)
@pytest.mark.parametrize("prepared_mode", ["base", "balanced"])
def test_full_iteration_dynamic_packs_rejects_unsupported_run_modes(
    monkeypatch, overrides, match, prepared_mode
):
    monkeypatch.setattr(sys, 'argv', ['test_model_parallel_config.py'])
    args = parse_args()
    args.cuda_graph_impl = 'full_iteration'
    args.world_size = 2
    args.context_parallel_size = 2
    args.sequence_packing_scheduler = "dp_balanced"
    args.experimental_attention_variant = "dsv4_hybrid"
    args.cp_partition_mode = "contiguous"
    args.dsa_cp_balance_indexer = prepared_mode == "balanced"
    for name, value in overrides.items():
        setattr(args, name, value)

    with pytest.raises(ValueError, match=match):
        validate_args(args)


def test_prepared_base_lifecycle_guard_observes_auto_selected_scheduler(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['test_model_parallel_config.py'])
    args = parse_args()
    args.num_layers = 2
    args.hidden_size = 128
    args.num_attention_heads = 4
    args.max_position_embeddings = 1024
    args.seq_length = 1024
    args.micro_batch_size = 1
    args.train_iters = 1
    args.lr = 1e-4
    args.tokenizer_type = 'NullTokenizer'
    args.vocab_size = 1024
    args.cuda_graph_impl = 'full_iteration'
    args.world_size = 2
    args.context_parallel_size = 2
    args.use_varlen_dataset = True
    args.sequence_packing_scheduler = None
    args.experimental_attention_variant = 'dsv4_hybrid'
    args.cp_partition_mode = 'contiguous'
    args.dsa_cp_balance_indexer = False
    # Satisfy the independent full-iteration prerequisite so validation
    # reaches VarlenDataset's scheduler auto-selection below.
    args.check_for_nan_in_loss_and_grad = False
    args.check_for_spiky_loss = True

    with pytest.raises(ValueError, match='do not support --check-for-spiky-loss'):
        validate_args(args)
    assert args.sequence_packing_scheduler == 'dp_balanced'
