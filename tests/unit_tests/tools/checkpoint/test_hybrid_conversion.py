# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for the legacy hybrid TP/PP checkpoint conversion tool."""

import argparse
import os
import sys
from collections import OrderedDict

import torch

# Add the tools/checkpoint directory to the path so we can import the module
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'tools', 'checkpoint')
)

import hybrid_conversion

D_MODEL, D_STATE, N_GROUPS, HEAD_DIM, FFN, VOCAB = 16, 4, 2, 8, 24, 32
D_INNER = 2 * D_MODEL
N_HEADS = D_INNER // HEAD_DIM


def _hybrid_state_dict():
    """State dict of a tiny 'M*-' hybrid model with TE fused input norms."""
    torch.manual_seed(0)
    conv_dim = D_INNER + 2 * N_GROUPS * D_STATE
    mamba = 'decoder.layers.0.mixer.'
    attn = 'decoder.layers.1.self_attention.'
    mlp = 'decoder.layers.2.mlp.'
    shapes = OrderedDict(
        [
            ('embedding.word_embeddings.weight', (VOCAB, D_MODEL)),
            (mamba + 'in_proj.layer_norm_weight', (D_MODEL,)),
            (mamba + 'in_proj.weight', (2 * D_INNER + 2 * N_GROUPS * D_STATE + N_HEADS, D_MODEL)),
            (mamba + 'conv1d.weight', (conv_dim, 1, 4)),
            (mamba + 'conv1d.bias', (conv_dim,)),
            (mamba + 'A_log', (N_HEADS,)),
            (mamba + 'D', (N_HEADS,)),
            (mamba + 'dt_bias', (N_HEADS,)),
            (mamba + 'norm.weight', (D_INNER,)),
            (mamba + 'out_proj.weight', (D_MODEL, D_INNER)),
            (attn + 'linear_qkv.layer_norm_weight', (D_MODEL,)),
            (attn + 'linear_qkv.weight', (3 * D_MODEL, D_MODEL)),
            (attn + 'linear_proj.weight', (D_MODEL, D_MODEL)),
            (mlp + 'linear_fc1.layer_norm_weight', (D_MODEL,)),
            (mlp + 'linear_fc1.weight', (FFN, D_MODEL)),
            (mlp + 'linear_fc2.weight', (D_MODEL, FFN)),
            ('decoder.final_norm.weight', (D_MODEL,)),
            ('output_layer.weight', (VOCAB, D_MODEL)),
        ]
    )
    return OrderedDict((key, torch.randn(shape)) for key, shape in shapes.items())


def _save_legacy_checkpoint(root, model):
    ckpt_args = argparse.Namespace(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, num_layers=3
    )
    ckpt_dir = os.path.join(root, 'iter_0000001', 'mp_rank_00')
    os.makedirs(ckpt_dir)
    torch.save(
        {
            'model': model,
            'args': ckpt_args,
            'checkpoint_version': 3.0,
            'iteration': 1,
            'rng_state': [],
        },
        os.path.join(ckpt_dir, 'model_optim_rng.pt'),
    )
    with open(os.path.join(root, 'latest_checkpointed_iteration.txt'), 'w') as f:
        f.write('1')


def _load_model(root, rank_dir):
    path = os.path.join(root, 'iter_0000001', rank_dir, 'model_optim_rng.pt')
    return torch.load(path, weights_only=True)['model']


def _convert(load_dir, save_dir, target_tp_size):
    hybrid_conversion.main(
        argparse.Namespace(
            load_dir=load_dir,
            save_dir=save_dir,
            target_tp_size=target_tp_size,
            target_pp_size=1,
            reset_iterations=False,
            d_model=D_MODEL,
            mamba_version=2,
            mamba_d_state=D_STATE,
            mamba2_n_groups=N_GROUPS,
            mamba2_head_dim=HEAD_DIM,
        )
    )


def test_tp_round_trip_preserves_hybrid_checkpoint(tmp_path):
    """TP1 -> TP2 -> TP1 must reproduce every tensor of a current hybrid checkpoint."""
    reference = _hybrid_state_dict()
    tp1, tp2, tp1_again = (str(tmp_path / name) for name in ('tp1', 'tp2', 'tp1_again'))
    _save_legacy_checkpoint(tp1, reference)

    _convert(tp1, tp2, target_tp_size=2)
    _convert(tp2, tp1_again, target_tp_size=1)

    norm_key = 'decoder.layers.0.mixer.in_proj.layer_norm_weight'
    for rank_dir in ('mp_rank_00', 'mp_rank_01'):
        # The fused input norm is replicated, not sharded, across TP ranks.
        assert torch.equal(_load_model(tp2, rank_dir)[norm_key], reference[norm_key])

    converted = _load_model(tp1_again, 'mp_rank_00')
    assert list(converted.keys()) == list(reference.keys())
    for key, tensor in reference.items():
        assert torch.equal(converted[key], tensor), key
