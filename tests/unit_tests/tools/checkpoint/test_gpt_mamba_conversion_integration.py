# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""
Integration tests for gpt_mamba_conversion.py.

Creates minimal synthetic GPT checkpoints on disk, runs the full conversion
pipeline (load -> combine TP -> stitch PP -> convert -> split TP/PP -> save),
and verifies:
    - Shapes, dtypes, and key names in the output checkpoint.
    - Round-trip GPT -> Mamba -> GPT preserves attention and MLP weights exactly.

Designed to run on a single-GPU node via SLURM (no distributed launch needed).
"""

import argparse
import copy
import os
import shutil
import sys
import tempfile
from collections import OrderedDict
from types import SimpleNamespace

import torch

# Ensure the conversion tool is importable
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'tools', 'checkpoint'),
)

from gpt_mamba_conversion import (
    get_checkpoint_iteration,
    initialize_ssm_layer_params,
    main as conversion_main,
    parse_hybrid_layer_pattern,
)


# ---------------------------------------------------------------------------
# Helpers: create a minimal on-disk GPT checkpoint
# ---------------------------------------------------------------------------

def make_checkpoint_args(
    num_layers=4,
    hidden_size=128,
    num_attention_heads=4,
    seq_length=256,
    max_position_embeddings=256,
    tp_size=1,
    pp_size=1,
    iteration=100,
):
    """Build a minimal checkpoint 'args' namespace mirroring Megatron's."""
    return SimpleNamespace(
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        ffn_hidden_size=hidden_size * 4,
        seq_length=seq_length,
        max_position_embeddings=max_position_embeddings,
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=pp_size,
        iteration=iteration,
        consumed_train_samples=0,
        consumed_valid_samples=0,
        train_iters=1000,
        train_samples=0,
        tokenizer_type='GPT2BPETokenizer',
        position_embedding_type='rope',
        params_dtype=torch.float32,
        fp16=False,
        bf16=False,
    )


def make_gpt_state_dict(num_layers, hidden_size, vocab_size=1024, dtype=torch.float32):
    """Create a minimal GPT model state dict."""
    sd = OrderedDict()

    sd['embedding.word_embeddings.weight'] = torch.randn(vocab_size, hidden_size, dtype=dtype)

    for i in range(num_layers):
        p = f'decoder.layers.{i}.'
        # attention
        sd[p + 'input_layernorm.weight'] = torch.randn(hidden_size, dtype=dtype)
        sd[p + 'self_attention.linear_qkv.weight'] = torch.randn(
            3 * hidden_size, hidden_size, dtype=dtype
        )
        sd[p + 'self_attention.linear_proj.weight'] = torch.randn(
            hidden_size, hidden_size, dtype=dtype
        )
        # MLP
        sd[p + 'pre_mlp_layernorm.weight'] = torch.randn(hidden_size, dtype=dtype)
        sd[p + 'mlp.linear_fc1.weight'] = torch.randn(
            4 * hidden_size, hidden_size, dtype=dtype
        )
        sd[p + 'mlp.linear_fc2.weight'] = torch.randn(
            hidden_size, 4 * hidden_size, dtype=dtype
        )

    sd['decoder.final_layernorm.weight'] = torch.randn(hidden_size, dtype=dtype)
    sd['output_layer.weight'] = torch.randn(vocab_size, hidden_size, dtype=dtype)

    return sd


def write_checkpoint_to_disk(root_dir, state_dict, ckpt_args, iteration=100):
    """Write a single-rank (TP=1, PP=1) checkpoint to disk in Megatron format.

    Directory layout:
        root_dir/
            latest_checkpointed_iteration.txt   ->  "100"
            iter_0000100/
                mp_rank_00/
                    model_optim_rng.pt
    """
    iter_dir = os.path.join(root_dir, f'iter_{iteration:07d}', 'mp_rank_00')
    os.makedirs(iter_dir, exist_ok=True)

    checkpoint = {
        'model': state_dict,
        'args': copy.deepcopy(ckpt_args),
        'checkpoint_version': 3.0,
        'iteration': iteration,
        'rng_state': [
            {
                'random_rng_state': [0] * 625,
                'np_rng_state': ('MT19937', [0] * 625, 0, 0, 0.0),
                'torch_rng_state': torch.ByteTensor(8),
                'cuda_rng_state': torch.ByteTensor(8),
                'rng_tracker_states': {},
            }
        ],
    }

    torch.save(checkpoint, os.path.join(iter_dir, 'model_optim_rng.pt'))

    with open(os.path.join(root_dir, 'latest_checkpointed_iteration.txt'), 'w') as f:
        f.write(str(iteration))

    return root_dir


def load_converted_state_dict(ckpt_dir):
    """Load the state dict from a converted checkpoint (TP=1, PP=1)."""
    iteration = get_checkpoint_iteration(ckpt_dir)
    model_file = os.path.join(
        ckpt_dir, f'iter_{iteration:07d}', 'mp_rank_00', 'model_optim_rng.pt'
    )
    checkpoint = torch.load(model_file, map_location='cpu', weights_only=False)
    return checkpoint['model'], checkpoint['args']


# ---------------------------------------------------------------------------
# Test 1: GPT -> Mamba shapes, dtypes, and key names
# ---------------------------------------------------------------------------

def test_gpt_to_mamba_shapes_and_keys():
    """Create a 4-layer GPT ckpt, convert to Mamba with M*-M*-M*-M*-, verify output."""
    print("\n=== Test 1: GPT -> Mamba shapes, dtypes, and key names ===")

    num_layers = 4
    hidden_size = 128
    d_state = 16
    n_groups = 2
    head_dim = 32
    d_inner = hidden_size * 2
    n_heads = d_inner // head_dim
    pattern = "M*-M*-M*-M*-"  # 12 layers: 4 SSM, 4 attn, 4 MLP

    tmpdir = tempfile.mkdtemp(prefix='gpt_mamba_test_')
    try:
        src_dir = os.path.join(tmpdir, 'gpt_src')
        dst_dir = os.path.join(tmpdir, 'mamba_dst')

        ckpt_args = make_checkpoint_args(num_layers=num_layers, hidden_size=hidden_size)
        gpt_sd = make_gpt_state_dict(num_layers, hidden_size)
        write_checkpoint_to_disk(src_dir, gpt_sd, ckpt_args)

        # Run conversion
        args = argparse.Namespace(
            direction='gpt-to-mamba',
            load_dir=src_dir,
            save_dir=dst_dir,
            hybrid_layer_pattern=pattern,
            target_tp_size=1,
            target_pp_size=1,
            d_model=hidden_size,
            mamba_version=2,
            mamba_d_state=d_state,
            mamba2_n_groups=n_groups,
            mamba2_head_dim=head_dim,
            d_conv=4,
            init_method_std=0.02,
            reset_iterations=False,
        )
        conversion_main(args)

        # Load and verify
        mamba_sd, mamba_args = load_converted_state_dict(dst_dir)

        layer_types = parse_hybrid_layer_pattern(pattern)
        total_layers = len(layer_types)

        # 1) Check total layer count in args
        assert mamba_args.num_layers == total_layers, (
            f"Expected num_layers={total_layers}, got {mamba_args.num_layers}"
        )

        # 2) Check key names
        assert 'decoder.final_norm.weight' in mamba_sd, "Missing decoder.final_norm.weight"
        assert 'decoder.final_layernorm.weight' not in mamba_sd, "Old final_layernorm key present"
        assert 'embedding.word_embeddings.weight' in mamba_sd
        assert 'output_layer.weight' in mamba_sd

        # 3) Check SSM layer params exist with correct shapes
        ssm_indices = [i for i, t in enumerate(layer_types) if t == 'M']
        conv_dim = d_inner + 2 * n_groups * d_state
        in_proj_out = 2 * d_inner + 2 * n_groups * d_state + n_heads

        for idx in ssm_indices:
            prefix = f'decoder.layers.{idx}.mixer.'
            assert prefix + 'A_log' in mamba_sd, f"Missing {prefix}A_log"
            assert mamba_sd[prefix + 'A_log'].shape == (n_heads,)
            assert mamba_sd[prefix + 'A_log'].dtype == torch.float32

            assert prefix + 'D' in mamba_sd
            assert mamba_sd[prefix + 'D'].shape == (n_heads,)

            assert prefix + 'dt_bias' in mamba_sd
            assert mamba_sd[prefix + 'dt_bias'].shape == (n_heads,)

            assert prefix + 'conv1d.weight' in mamba_sd
            assert mamba_sd[prefix + 'conv1d.weight'].shape == (conv_dim, 1, 4)

            assert prefix + 'conv1d.bias' in mamba_sd
            assert mamba_sd[prefix + 'conv1d.bias'].shape == (conv_dim,)

            assert prefix + 'in_proj.weight' in mamba_sd
            assert mamba_sd[prefix + 'in_proj.weight'].shape == (in_proj_out, hidden_size)

            assert prefix + 'norm.weight' in mamba_sd
            assert mamba_sd[prefix + 'norm.weight'].shape == (d_inner,)

            assert prefix + 'out_proj.weight' in mamba_sd
            assert mamba_sd[prefix + 'out_proj.weight'].shape == (hidden_size, d_inner)

        # 4) Check attention layer params exist at correct indices
        attn_indices = [i for i, t in enumerate(layer_types) if t == '*']
        for idx in attn_indices:
            prefix = f'decoder.layers.{idx}.'
            assert prefix + 'self_attention.linear_qkv.weight' in mamba_sd
            assert mamba_sd[prefix + 'self_attention.linear_qkv.weight'].shape == (
                3 * hidden_size, hidden_size
            )

        # 5) Check MLP layer params exist at correct indices
        mlp_indices = [i for i, t in enumerate(layer_types) if t == '-']
        for idx in mlp_indices:
            prefix = f'decoder.layers.{idx}.'
            assert prefix + 'mlp.linear_fc1.weight' in mamba_sd
            assert mamba_sd[prefix + 'mlp.linear_fc1.weight'].shape == (
                4 * hidden_size, hidden_size
            )

        print("PASSED: All shapes, dtypes, and key names verified.\n")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 2: Round-trip GPT -> Mamba -> GPT weight preservation
# ---------------------------------------------------------------------------

def test_roundtrip_weight_preservation():
    """Convert GPT -> Mamba -> GPT and verify attention/MLP weights match exactly."""
    print("\n=== Test 2: Round-trip GPT -> Mamba -> GPT weight preservation ===")

    num_layers = 2
    hidden_size = 64
    pattern = "M*-M*-"

    tmpdir = tempfile.mkdtemp(prefix='gpt_mamba_rt_test_')
    try:
        src_gpt_dir = os.path.join(tmpdir, 'gpt_src')
        mamba_dir = os.path.join(tmpdir, 'mamba_mid')
        dst_gpt_dir = os.path.join(tmpdir, 'gpt_dst')

        # Create and save source GPT checkpoint
        ckpt_args = make_checkpoint_args(num_layers=num_layers, hidden_size=hidden_size)
        gpt_sd = make_gpt_state_dict(num_layers, hidden_size)
        write_checkpoint_to_disk(src_gpt_dir, gpt_sd, ckpt_args)

        common_args = dict(
            hybrid_layer_pattern=pattern,
            target_tp_size=1,
            target_pp_size=1,
            d_model=hidden_size,
            mamba_version=2,
            mamba_d_state=16,
            mamba2_n_groups=2,
            mamba2_head_dim=32,
            d_conv=4,
            init_method_std=0.02,
            reset_iterations=False,
        )

        # Step 1: GPT -> Mamba
        conversion_main(argparse.Namespace(
            direction='gpt-to-mamba',
            load_dir=src_gpt_dir,
            save_dir=mamba_dir,
            **common_args,
        ))

        # Step 2: Mamba -> GPT
        conversion_main(argparse.Namespace(
            direction='mamba-to-gpt',
            load_dir=mamba_dir,
            save_dir=dst_gpt_dir,
            **common_args,
        ))

        # Load and compare
        recovered_sd, recovered_args = load_converted_state_dict(dst_gpt_dir)

        assert recovered_args.num_layers == num_layers, (
            f"Expected num_layers={num_layers}, got {recovered_args.num_layers}"
        )

        # Compare every key in the original
        mismatches = []
        for key, original_tensor in gpt_sd.items():
            # final_layernorm is renamed in the round trip
            if key not in recovered_sd:
                mismatches.append(f"MISSING: {key}")
                continue
            if not torch.equal(original_tensor, recovered_sd[key]):
                max_diff = (original_tensor - recovered_sd[key]).abs().max().item()
                mismatches.append(f"MISMATCH: {key} (max_diff={max_diff})")

        if mismatches:
            for m in mismatches:
                print(f"  FAIL: {m}")
            raise AssertionError(
                f"Round-trip failed with {len(mismatches)} mismatches:\n"
                + "\n".join(mismatches)
            )

        print("PASSED: All attention and MLP weights preserved exactly.\n")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 3: Verify that Mamba -> GPT discards SSM params cleanly
# ---------------------------------------------------------------------------

def test_mamba_to_gpt_discards_ssm():
    """Convert Mamba -> GPT and verify no SSM keys leak through."""
    print("\n=== Test 3: Mamba -> GPT discards SSM params ===")

    hidden_size = 64
    pattern = "M*-M*-"
    d_inner = hidden_size * 2
    d_state = 16
    n_groups = 2
    head_dim = 32
    n_heads = d_inner // head_dim

    tmpdir = tempfile.mkdtemp(prefix='gpt_mamba_discard_test_')
    try:
        # Build a Mamba-style state dict
        mamba_sd = OrderedDict()
        mamba_sd['embedding.word_embeddings.weight'] = torch.randn(512, hidden_size)
        mamba_sd['output_layer.weight'] = torch.randn(512, hidden_size)
        mamba_sd['decoder.final_norm.weight'] = torch.randn(hidden_size)

        layer_types = parse_hybrid_layer_pattern(pattern)
        for i, lt in enumerate(layer_types):
            p = f'decoder.layers.{i}.'
            if lt == 'M':
                ssm = initialize_ssm_layer_params(
                    i, hidden_size, d_inner, d_state, n_groups, n_heads, head_dim
                )
                mamba_sd.update(ssm)
            elif lt == '*':
                mamba_sd[p + 'input_layernorm.weight'] = torch.randn(hidden_size)
                mamba_sd[p + 'self_attention.linear_qkv.weight'] = torch.randn(
                    3 * hidden_size, hidden_size
                )
                mamba_sd[p + 'self_attention.linear_proj.weight'] = torch.randn(
                    hidden_size, hidden_size
                )
            elif lt == '-':
                mamba_sd[p + 'pre_mlp_layernorm.weight'] = torch.randn(hidden_size)
                mamba_sd[p + 'mlp.linear_fc1.weight'] = torch.randn(
                    4 * hidden_size, hidden_size
                )
                mamba_sd[p + 'mlp.linear_fc2.weight'] = torch.randn(
                    hidden_size, 4 * hidden_size
                )

        # Write to disk
        src_dir = os.path.join(tmpdir, 'mamba_src')
        dst_dir = os.path.join(tmpdir, 'gpt_dst')
        ckpt_args = make_checkpoint_args(
            num_layers=len(layer_types), hidden_size=hidden_size
        )
        write_checkpoint_to_disk(src_dir, mamba_sd, ckpt_args)

        # Convert
        conversion_main(argparse.Namespace(
            direction='mamba-to-gpt',
            load_dir=src_dir,
            save_dir=dst_dir,
            hybrid_layer_pattern=pattern,
            target_tp_size=1,
            target_pp_size=1,
            d_model=hidden_size,
            mamba_version=2,
            mamba_d_state=d_state,
            mamba2_n_groups=n_groups,
            mamba2_head_dim=head_dim,
            d_conv=4,
            init_method_std=0.02,
            reset_iterations=False,
        ))

        gpt_sd, gpt_args = load_converted_state_dict(dst_dir)

        # Verify no SSM keys
        ssm_keys = [k for k in gpt_sd if 'mixer.' in k]
        assert len(ssm_keys) == 0, f"SSM keys leaked: {ssm_keys}"

        # Verify correct GPT layer count
        assert gpt_args.num_layers == 2

        # Verify final_layernorm renamed back
        assert 'decoder.final_layernorm.weight' in gpt_sd
        assert 'decoder.final_norm.weight' not in gpt_sd

        print("PASSED: No SSM keys in GPT output, norms renamed correctly.\n")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=" * 60)
    print("GPT <-> Mamba Conversion Integration Tests")
    print("=" * 60)

    test_gpt_to_mamba_shapes_and_keys()
    test_roundtrip_weight_preservation()
    test_mamba_to_gpt_discards_ssm()

    print("=" * 60)
    print("ALL INTEGRATION TESTS PASSED")
    print("=" * 60)
