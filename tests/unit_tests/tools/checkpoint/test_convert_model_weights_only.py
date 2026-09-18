# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
GPU unit test for ``convert_checkpoint(..., model_weights_only=True)``.

``tools/checkpoint/checkpoint_inspector.py``'s
``convert-torch-dist-to-fsdp-dtensor`` command converts a ``torch_dist``
checkpoint to ``fsdp_dtensor`` for Megatron-FSDP v2. M-FSDP v2 does not
implement optimizer checkpointing, so a converted checkpoint that still
carries ``optimizer.*`` state cannot be loaded. The ``--model-weights-only``
flag (``model_weights_only=True``) drops all optimizer state and emits only
model weights plus the non-optimizer common state (``args``, ``iteration``,
``checkpoint_version``).

This test builds a synthetic ``torch_dist`` checkpoint that genuinely
contains optimizer state (``optimizer.state.{exp_avg,exp_avg_sq}.<model_key>``
tensors plus ``optimizer.param_groups`` in ``common.pt``), runs the converter
twice, and asserts:

  * ``model_weights_only=True`` -> every model weight is present and
    element-wise identical, and **no** output key starts with ``optimizer.``.
  * ``model_weights_only=False`` (default) -> the same model weights
    round-trip, and the output still carries ``optimizer.*`` entries, proving
    the flag is what makes the difference.
  * the non-optimizer common state survives in both modes.

This is a GPU test: ``convert_checkpoint`` builds a CUDA ``DeviceMesh`` and
allocates CUDA DTensors, so an NCCL process group and at least one GPU are
required (the CLI's ``init_process_group`` also forces the NCCL backend). It
therefore cannot run on a CPU-only machine.

Launch it (all ranks participate in the DCP collectives; the default
``torch.distributed.run`` unit-test convention applies, and it is intended for
an H100-class GPU platform):

    uv run python -m torch.distributed.run --nproc-per-node 8 -m pytest -q \\
        tests/unit_tests/tools/checkpoint/test_convert_model_weights_only.py

or standalone on a shared filesystem (``--output-root`` must be visible from
every rank; single-node runs can use the default ``/tmp``-based root):

    torchrun --nproc_per_node=8 \\
        tests/unit_tests/tools/checkpoint/test_convert_model_weights_only.py \\
        --output-root /shared/scratch/ckpt_model_weights_only
"""

import argparse
import os
import shutil
import sys
import tempfile
from collections import OrderedDict
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import DefaultLoadPlanner, FileSystemReader
from torch.distributed.checkpoint.metadata import TensorStorageMetadata

# Make the conversion tool and helpers importable, matching the sibling tests.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.join(_THIS_DIR, '..', '..', '..', '..')
sys.path.insert(0, os.path.join(_REPO_ROOT, 'tools', 'checkpoint'))
sys.path.insert(0, _THIS_DIR)

# Override to point the shared output root at a shared filesystem (multi-node).
_SHARED_ROOT_ENV = 'MCORE_CKPT_MODEL_WEIGHTS_ONLY_ROOT'


def _log(rank, message):
    print(f"[rank={rank}] {message}", flush=True)


def _shared_root():
    """Directory shared by every rank (DCP saves/loads are collective)."""
    return os.environ.get(
        _SHARED_ROOT_ENV,
        os.path.join(tempfile.gettempdir(), 'mcore_ckpt_model_weights_only_test'),
    )


def _ensure_nccl_process_group():
    """Initialize the default NCCL process group if the harness has not.

    Returns ``True`` when a usable CUDA/NCCL group is available. A pre-existing
    non-NCCL default group (e.g. a gloo group initialized by an earlier test in
    the same pytest session) cannot back a CUDA ``DeviceMesh``, so it is
    reported as unusable rather than failing obscurely.
    """
    if dist.is_initialized():
        return dist.get_backend() == 'nccl'
    if 'RANK' not in os.environ or 'WORLD_SIZE' not in os.environ:
        return False
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))
    dist.init_process_group(backend='nccl')
    return True


# ---------------------------------------------------------------------------
# Synthetic checkpoint fixture
# ---------------------------------------------------------------------------


def _build_model_state_dict(num_layers, hidden_size, vocab_size, dtype):
    """Deterministic bare-key GPT state dict (identical on every rank)."""
    torch.manual_seed(0x5EED)
    sd = OrderedDict()
    sd['embedding.word_embeddings.weight'] = torch.randn(
        vocab_size, hidden_size, dtype=dtype
    )
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
        sd[p + 'mlp.linear_fc1.weight'] = torch.randn(
            4 * hidden_size, hidden_size, dtype=dtype
        )
        sd[p + 'mlp.linear_fc2.weight'] = torch.randn(
            hidden_size, 4 * hidden_size, dtype=dtype
        )
    sd['decoder.final_layernorm.weight'] = torch.randn(hidden_size, dtype=dtype)
    sd['output_layer.weight'] = torch.randn(vocab_size, hidden_size, dtype=dtype)
    return sd


def _build_optimizer_state_dict(model_state_dict):
    """Adam state keyed as Megatron shards it: ``optimizer.state.<slot>.<param>``.

    See ``tests/unit_tests/dist_checkpointing/test_optimizer.py`` (the
    ``optimizer.state.{exp_avg,exp_avg_sq}.<layer_name>`` layout) and
    ``megatron/core/optimizer/distrib_optimizer.py``.
    """
    torch.manual_seed(0x0F71)
    optimizer_sd = OrderedDict()
    for model_key, tensor in model_state_dict.items():
        for slot in ('exp_avg', 'exp_avg_sq'):
            optimizer_sd[f'optimizer.state.{slot}.{model_key}'] = torch.randn_like(tensor)
    return optimizer_sd


def _build_ckpt_args(num_layers, hidden_size, vocab_size):
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
        num_moe_experts=None,
        moe_shared_expert_intermediate_size=None,
        moe_layer_freq=1,
        vocab_size=vocab_size,
    )


def _build_common_state(num_layers, hidden_size, vocab_size, iteration=100):
    """common.pt contents: non-tensor state plus genuine optimizer param_groups.

    ``optimizer.param_groups`` matches the layout the converter reads via
    ``common_state["optimizer"]["param_groups"]``; then ``flatten`` turns it
    into ``optimizer.param_groups.<i>.<field>`` entries that the default
    conversion merges into the output.
    """
    return {
        'args': _build_ckpt_args(num_layers, hidden_size, vocab_size),
        'checkpoint_version': 3.0,
        'iteration': iteration,
        'optimizer': {
            'param_groups': [
                {'lr': 1.0e-4, 'weight_decay': 0.01, 'betas': [0.9, 0.95]}
            ]
        },
    }


# ---------------------------------------------------------------------------
# Conversion + verification
# ---------------------------------------------------------------------------


def _metadata_keys(ckpt_dir):
    """Keys recorded in a (raw) DCP checkpoint's metadata."""
    return set(FileSystemReader(ckpt_dir).read_metadata().state_dict_metadata.keys())


def _load_full_tensors(ckpt_dir):
    """Load a raw DCP checkpoint into full, gathered CPU tensors.

    ``convert_checkpoint`` writes a *raw* DCP checkpoint through
    ``torch.distributed.checkpoint`` and does not write Megatron's
    ``metadata.json``, so ``dist_checkpoint_io.load_dist_checkpoint_full``
    (which requires that config) cannot read it. This mirrors that helper's
    tensor-loading core, minus the config/prefix/filter logic.
    """
    reader = FileSystemReader(ckpt_dir)
    metadata = reader.read_metadata()
    state_dict = {}
    for key, md in metadata.state_dict_metadata.items():
        if not isinstance(md, TensorStorageMetadata):
            continue
        state_dict[key] = torch.empty(md.size, dtype=md.properties.dtype, device='cpu')
    dcp.load(state_dict, storage_reader=reader, planner=DefaultLoadPlanner())
    return state_dict


def run_case(
    label,
    model_weights_only,
    output_root,
    num_layers=2,
    hidden_size=32,
    vocab_size=64,
    dtype=torch.float32,
):
    """Build a torch_dist ckpt with optimizer state, convert, and verify."""
    from checkpoint_inspector import convert_checkpoint
    from dist_checkpoint_io import save_dist_checkpoint_full

    rank = dist.get_rank()
    case_dir = os.path.join(output_root, label)
    src_dir = os.path.join(case_dir, 'torch_dist_src', 'iter_0000100')
    dst_dir = os.path.join(case_dir, 'fsdp_dtensor_dst')

    if rank == 0 and os.path.isdir(case_dir):
        shutil.rmtree(case_dir, ignore_errors=True)
    dist.barrier()
    os.makedirs(os.path.dirname(src_dir), exist_ok=True)
    dist.barrier()

    model_sd = _build_model_state_dict(num_layers, hidden_size, vocab_size, dtype)
    full_sd = OrderedDict(model_sd)
    full_sd.update(_build_optimizer_state_dict(model_sd))

    # ``model_prefix=''`` keeps the bare model keys the converter expects and
    # leaves the optimizer.* keys exactly as Megatron shards them.
    save_dist_checkpoint_full(
        full_sd,
        _build_common_state(num_layers, hidden_size, vocab_size),
        src_dir,
        model_prefix='',
        backend='torch_dist',
    )
    dist.barrier()

    src_keys = _metadata_keys(src_dir)
    assert any(k.startswith('optimizer.state.') for k in src_keys), (
        f"[{label}] fixture is missing optimizer.state.* tensors: {sorted(src_keys)[:5]}"
    )

    # Every rank must participate: dcp.load / _save_state_dict are collective.
    convert_checkpoint(
        src_dir,
        dst_dir,
        False,
        process_group=dist.group.WORLD,
        model_weights_only=model_weights_only,
    )
    dist.barrier()

    dst_keys = _metadata_keys(dst_dir)
    optimizer_keys = sorted(k for k in dst_keys if k.startswith('optimizer.'))
    for expected_common_key in ('args', 'iteration', 'checkpoint_version'):
        assert expected_common_key in dst_keys, (
            f"[{label}] non-optimizer common state '{expected_common_key}' was dropped"
        )

    # Model weights must survive element-wise, without depending on load order.
    loaded = _load_full_tensors(dst_dir)
    model_prefix = 'model.module.'
    recovered = {
        k[len(model_prefix):]: v
        for k, v in loaded.items()
        if k.startswith(model_prefix)
    }
    missing = [k for k in model_sd if k not in recovered]
    mismatch = [
        k
        for k in model_sd
        if k in recovered and not torch.equal(model_sd[k], recovered[k].to(model_sd[k].dtype))
    ]
    assert not missing, f"[{label}] missing model weights: {missing[:5]}"
    assert not mismatch, f"[{label}] mismatched model weights: {mismatch[:5]}"

    if model_weights_only:
        assert not optimizer_keys, (
            f"[{label}] model-weights-only output still has optimizer keys: "
            f"{optimizer_keys[:5]}"
        )
        _log(
            rank,
            f"[{label}] PASS: {len(recovered)} model weights round-tripped, "
            f"0 optimizer keys",
        )
    else:
        assert optimizer_keys, (
            f"[{label}] default conversion unexpectedly dropped all optimizer keys"
        )
        _log(
            rank,
            f"[{label}] PASS: {len(recovered)} model weights round-tripped, "
            f"{len(optimizer_keys)} optimizer keys preserved",
        )

    dist.barrier()


# ---------------------------------------------------------------------------
# pytest entry points
# ---------------------------------------------------------------------------


@pytest.fixture(scope='module', autouse=True)
def _require_cuda_nccl():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required: convert_checkpoint builds a CUDA DeviceMesh.")
    if not _ensure_nccl_process_group():
        pytest.skip(
            "Requires an NCCL process group; launch via "
            "'torch.distributed.run --nproc-per-node <N> -m pytest' or torchrun."
        )
    yield
    if dist.is_initialized():
        dist.barrier()


def test_model_weights_only_drops_all_optimizer_state():
    """--model-weights-only emits model.* only; no optimizer.* key survives."""
    run_case('model_weights_only', True, _shared_root())


def test_default_conversion_keeps_optimizer_state():
    """The default path is unchanged and still carries optimizer.* entries."""
    run_case('default', False, _shared_root())


# ---------------------------------------------------------------------------
# Standalone entry point (torchrun)
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description='GPU test for checkpoint_inspector --model-weights-only.'
    )
    parser.add_argument(
        '--output-root',
        type=str,
        default=None,
        help='Shared-filesystem directory visible from every rank. Defaults '
        'to a /tmp-based directory (single node only).',
    )
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--hidden-size', type=int, default=32)
    parser.add_argument('--vocab-size', type=int, default=64)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        sys.exit('ERROR: CUDA is required (convert_checkpoint builds a CUDA DeviceMesh).')
    if not _ensure_nccl_process_group():
        sys.exit(
            'ERROR: could not initialize an NCCL process group; launch with '
            'torchrun/torch.distributed.run on a GPU node.'
        )

    output_root = os.path.abspath(args.output_root or _shared_root())
    size_kwargs = dict(
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        vocab_size=args.vocab_size,
    )
    run_case('model_weights_only', True, output_root, **size_kwargs)
    run_case('default', False, output_root, **size_kwargs)

    if dist.get_rank() == 0:
        _log(0, 'PASS: --model-weights-only drops all optimizer state; default keeps it.')
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
