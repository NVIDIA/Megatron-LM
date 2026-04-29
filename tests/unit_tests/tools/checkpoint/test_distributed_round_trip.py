# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""
Multi-rank distributed round-trip test for gpt_mamba_conversion.

Each rank participates in a multi-rank DCP save of a synthetic GPT (or MoE
GPT) state dict; rank 0 then runs the converter and verifies the GPT->Mamba->
GPT round-trip exactly.

This test is meant to be launched under SLURM/srun (or torchrun) with
WORLD_SIZE = TP * PP * FSDP * EP. The (tp, pp, fsdp, ep) values are passed
as flags purely as labels — the converter sees only the DCP-stored
``global_shape`` per tensor and is agnostic to *which* dimension(s) the
source was sharded along. The test value is in:

    1. Coordinating a real multi-rank ``dcp.save`` (cross-node networking,
       collective barriers, shared-filesystem write).
    2. Verifying the converter loads a multi-rank-written checkpoint and
       round-trips it through both directions.

Usage (under srun):
    export RANK=$SLURM_PROCID
    export LOCAL_RANK=$SLURM_LOCALID
    export WORLD_SIZE=$SLURM_NTASKS
    export MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -1)
    export MASTER_PORT=29500
    python test_distributed_round_trip.py \\
        --tp 2 --pp 2 --fsdp 2 --ep 2 --label TP2-PP2-FSDP2-EP2 \\
        --output-root /lustre/.../scratch/dist_test
"""

import argparse
import copy
import os
import shutil
import sys
import time
from collections import OrderedDict
from types import SimpleNamespace

import torch
import torch.distributed as dist

# Make the conversion tool and helpers importable.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.join(_THIS_DIR, '..', '..', '..', '..')
sys.path.insert(0, os.path.join(_REPO_ROOT, 'tools', 'checkpoint'))
sys.path.insert(0, _THIS_DIR)


def _log(msg, rank, label=""):
    prefix = f"[rank={rank}{(' ' + label) if label else ''}]"
    print(f"{prefix} {msg}", flush=True)


def _build_state_dict(num_layers, hidden_size, num_moe_experts, vocab_size, dtype):
    """Build a deterministic GPT(MoE) state dict identical on every rank.

    Determinism (via fixed seed) lets every rank produce the same tensors so
    DCP's de-duplication writes a single coherent checkpoint. After load, we
    re-derive the same tensors on rank 0 to verify round-trip.
    """
    torch.manual_seed(0xC0FFEE)
    sd = OrderedDict()
    sd['embedding.word_embeddings.weight'] = torch.randn(vocab_size, hidden_size, dtype=dtype)

    for i in range(num_layers):
        p = f'decoder.layers.{i}.'
        sd[p + 'input_layernorm.weight'] = torch.randn(hidden_size, dtype=dtype)
        sd[p + 'self_attention.linear_qkv.weight'] = torch.randn(
            3 * hidden_size, hidden_size, dtype=dtype
        )
        sd[p + 'self_attention.linear_proj.weight'] = torch.randn(
            hidden_size, hidden_size, dtype=dtype
        )
        sd[p + 'pre_mlp_layernorm.weight'] = torch.randn(hidden_size, dtype=dtype)

        if num_moe_experts is None:
            sd[p + 'mlp.linear_fc1.weight'] = torch.randn(
                4 * hidden_size, hidden_size, dtype=dtype
            )
            sd[p + 'mlp.linear_fc2.weight'] = torch.randn(
                hidden_size, 4 * hidden_size, dtype=dtype
            )
        else:
            sd[p + 'mlp.router.weight'] = torch.randn(
                num_moe_experts, hidden_size, dtype=dtype
            )
            for j in range(num_moe_experts):
                ep = p + f'mlp.experts.local_experts.{j}.'
                sd[ep + 'linear_fc1.weight'] = torch.randn(
                    4 * hidden_size, hidden_size, dtype=dtype
                )
                sd[ep + 'linear_fc2.weight'] = torch.randn(
                    hidden_size, 4 * hidden_size, dtype=dtype
                )

    sd['decoder.final_layernorm.weight'] = torch.randn(hidden_size, dtype=dtype)
    sd['output_layer.weight'] = torch.randn(vocab_size, hidden_size, dtype=dtype)
    return sd


def _build_ckpt_args(num_layers, hidden_size, num_moe_experts):
    return SimpleNamespace(
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=4,
        ffn_hidden_size=hidden_size * 4,
        seq_length=256,
        max_position_embeddings=256,
        iteration=100,
        consumed_train_samples=0,
        consumed_valid_samples=0,
        train_iters=1000,
        train_samples=0,
        tokenizer_type='GPT2BPETokenizer',
        position_embedding_type='rope',
        params_dtype=torch.float32,
        fp16=False,
        bf16=False,
        num_moe_experts=num_moe_experts,
        moe_shared_expert_intermediate_size=None,
        moe_layer_freq=1,
    )


def _init_process_group(init_file):
    """Initialize via file:// rendezvous on a shared filesystem.

    RANK / WORLD_SIZE come from env (set by srun). MASTER_ADDR / MASTER_PORT
    are not needed — every rank just opens the same shared file. This avoids
    the SLURM CLI tools (e.g. scontrol) which are not always present inside
    container images.

    The init file MUST NOT pre-exist; rank 0 cleans up any stale leftover.
    """
    if dist.is_initialized():
        return
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    if rank == 0 and os.path.exists(init_file):
        os.remove(init_file)
    # Brief settle so other ranks don't race ahead of the cleanup.
    time.sleep(1)
    dist.init_process_group(
        backend='gloo',  # CPU-only synthetic save; no NCCL needed
        init_method=f'file://{init_file}',
        rank=rank,
        world_size=world_size,
    )


def _verify_round_trip(original, recovered, label):
    missing, mismatch = [], []
    for k, v in original.items():
        if k not in recovered:
            missing.append(k)
            continue
        if not torch.equal(v, recovered[k].to(v.dtype)):
            mismatch.append(k)

    leaked_ssm = [k for k in recovered if 'mixer.' in k]

    if missing or mismatch or leaked_ssm:
        print(f"FAIL [{label}]:")
        for k in missing[:5]:
            print(f"  MISSING: {k}")
        for k in mismatch[:5]:
            print(f"  MISMATCH: {k}")
        for k in leaked_ssm[:5]:
            print(f"  SSM LEAKED: {k}")
        raise AssertionError(
            f"{label}: {len(missing)} missing, {len(mismatch)} mismatched, "
            f"{len(leaked_ssm)} SSM keys leaked"
        )

    print(f"PASS [{label}]: {len(original)} keys round-tripped exactly")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tp', type=int, default=1)
    parser.add_argument('--pp', type=int, default=1)
    parser.add_argument('--fsdp', type=int, default=1)
    parser.add_argument('--ep', type=int, default=1)
    parser.add_argument('--label', type=str, required=True)
    parser.add_argument('--output-root', type=str, required=True,
                        help='Shared-filesystem path where this scenario writes its '
                             'checkpoints (must be visible from every rank).')
    parser.add_argument('--num-layers', type=int, default=3)
    parser.add_argument('--hidden-size', type=int, default=64)
    parser.add_argument('--vocab-size', type=int, default=512)
    parser.add_argument('--num-moe-experts', type=int, default=None,
                        help='If set, use the MoE GPT state-dict layout '
                             '(mlp.router + mlp.experts.local_experts.*).')
    parser.add_argument('--pattern', type=str, default=None,
                        help='Hybrid layer pattern. Defaults to "M*-M*-M*-" for '
                             'dense or "M*EM*EM*E" when --num-moe-experts is set.')
    args = parser.parse_args()

    expected_world = args.tp * args.pp * args.fsdp * args.ep
    pattern = args.pattern or (
        'M*EM*EM*E' if args.num_moe_experts is not None else 'M*-M*-M*-'
    )

    # Shared init file lives on the same shared FS we use for checkpoints, so
    # all ranks on all nodes see the same path.
    init_file = os.path.join(args.output_root, f'_pg_init_{args.label}')
    if int(os.environ.get('RANK', 0)) == 0:
        os.makedirs(args.output_root, exist_ok=True)
    _init_process_group(init_file)
    rank = dist.get_rank()
    world = dist.get_world_size()
    if world != expected_world:
        if rank == 0:
            print(f"FAIL [{args.label}]: world={world} but tp*pp*fsdp*ep={expected_world}")
        sys.exit(2)

    # Lazy imports after sys.path is set.
    from gpt_mamba_conversion import main as conversion_main
    from dist_checkpoint_io import (
        load_dist_checkpoint_full,
        save_dist_checkpoint_full,
        write_latest_iteration_marker,
    )

    if rank == 0:
        _log(
            f"label={args.label} tp={args.tp} pp={args.pp} fsdp={args.fsdp} "
            f"ep={args.ep} world={world} pattern={pattern} "
            f"num_moe_experts={args.num_moe_experts}",
            rank,
            args.label,
        )

    # Each rank builds the same full state dict — DCP de-duplicates writes
    # across ranks via its writer planner.
    state_dict = _build_state_dict(
        args.num_layers, args.hidden_size, args.num_moe_experts,
        args.vocab_size, torch.float32,
    )
    ckpt_args = _build_ckpt_args(
        args.num_layers, args.hidden_size, args.num_moe_experts,
    )

    scratch = os.path.join(args.output_root, args.label)
    src_dir = os.path.join(scratch, 'gpt_src')
    mid_dir = os.path.join(scratch, 'mamba_mid')
    dst_dir = os.path.join(scratch, 'gpt_dst')
    iter_subdir = os.path.join(src_dir, 'iter_0000100')

    if rank == 0:
        if os.path.isdir(scratch):
            shutil.rmtree(scratch, ignore_errors=True)
        os.makedirs(iter_subdir, exist_ok=True)
    dist.barrier()

    # --- Multi-rank DCP write of the source GPT checkpoint ---
    # dcp.save / dcp.load are both COLLECTIVE in the active process group, so
    # every rank in this PG must participate in every save and every load.
    # That includes the two conversion_main calls below, which internally call
    # load_dist_checkpoint_full + save_dist_checkpoint_full once each.
    # If a rank exits early its gloo socket closes and rank 0's reduce_scatter
    # dies with "Connection closed by peer".
    common_state = {
        'args': copy.deepcopy(ckpt_args),
        'checkpoint_version': 3.0,
        'iteration': 100,
    }
    save_dist_checkpoint_full(
        state_dict, common_state, iter_subdir,
        model_prefix='model.', backend='torch_dist',
    )
    if rank == 0:
        write_latest_iteration_marker(iter_subdir, 100)
    dist.barrier()

    # --- Convert GPT -> hybrid -> GPT (every rank participates collectively).
    common_kwargs = dict(
        hybrid_layer_pattern=pattern,
        d_model=args.hidden_size,
        mamba_version=2,
        mamba_d_state=16,
        mamba2_n_groups=2,
        mamba2_head_dim=32,
        d_conv=4,
        init_method_std=0.02,
        reset_iterations=False,
        input_format='auto',
        output_format='torch_dist',
    )

    # Silence non-rank-0 stdout to keep logs readable; collective behavior
    # is unaffected.
    if rank != 0:
        sys.stdout = open(os.devnull, 'w')

    t0 = time.time()
    conversion_main(argparse.Namespace(
        direction='gpt-to-mamba',
        load_dir=src_dir, save_dir=mid_dir,
        **common_kwargs,
    ))
    dist.barrier()
    conversion_main(argparse.Namespace(
        direction='mamba-to-gpt',
        load_dir=mid_dir, save_dir=dst_dir,
        **common_kwargs,
    ))
    dist.barrier()
    dt = time.time() - t0

    # Restore stdout for rank 0's verify message.
    if rank != 0:
        sys.stdout = sys.__stdout__

    # --- Load final + (rank 0 only) verify ---
    recovered, _, _, _, _ = load_dist_checkpoint_full(dst_dir)
    dist.barrier()

    if rank == 0:
        _log(f"conversion time: {dt:.2f}s", rank, args.label)
        _verify_round_trip(state_dict, recovered, args.label)
        shutil.rmtree(scratch, ignore_errors=True)
        if os.path.exists(init_file):
            os.remove(init_file)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
