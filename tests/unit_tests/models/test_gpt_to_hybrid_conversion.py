# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for the GPT -> Hybrid conversion path.

Covers three layers:

  * **Pure state-dict tests** (no GPU, no distributed init) —
    pattern parsing, layer-index mapping, GPT-compat validators, and the
    tensor-level ``convert_gpt_to_hybrid`` / ``initialize_ssm_layer_params``.

  * **End-to-end auto-convert test** — materializes a real GPT dist
    checkpoint on disk, invokes ``_autoconvert_gpt_to_hybrid_ckpt`` on a
    single rank, and asserts the rewritten directory round-trips through
    ``load_dist_checkpoint_full`` with attention/MLP bit-for-bit and the
    ``common_state`` reflecting the hybrid layout.

  * **Marker-hardening test** — pins the ``iter_XXXXXXX``-basename check in
    ``write_latest_iteration_marker`` so the original silent-no-op bug can't
    come back.
"""

import os
import sys
import tempfile
import types
from collections import OrderedDict

import pytest
import torch

# Make the offline ``dist_checkpoint_io`` module (only reachable via sys.path)
# importable so we can materialize a fake GPT dist-checkpoint on disk.
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
_TOOLS_CKPT = os.path.join(_REPO_ROOT, 'tools', 'checkpoint')
if _TOOLS_CKPT not in sys.path:
    sys.path.insert(0, _TOOLS_CKPT)

from dist_checkpoint_io import (
    load_dist_checkpoint_full,
    save_dist_checkpoint_full,
    write_latest_iteration_marker,
)

from megatron.core.models.hybrid.conversion import (
    build_layer_index_mapping,
    convert_gpt_to_hybrid,
    initialize_ssm_layer_params,
    parse_hybrid_layer_pattern,
    validate_pattern_gpt_compatible,
)
from megatron.training.checkpointing import _autoconvert_gpt_to_hybrid_ckpt

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_gpt_state_dict(num_layers: int, d_model: int = 32, ffn: int = 64):
    """Synthetic GPT state dict with attention + MLP per layer.

    Tensors are deterministic (torch.full with per-layer constants) so we can
    identity-check that each moves to the expected hybrid layer index.
    """
    sd = OrderedDict()
    sd['embedding.word_embeddings.weight'] = torch.arange(
        10 * d_model, dtype=torch.float32
    ).reshape(10, d_model)
    for i in range(num_layers):
        p = f'decoder.layers.{i}.'
        base = (i + 1) * 1000
        sd[p + 'input_layernorm.weight'] = torch.full((d_model,), float(base + 1))
        sd[p + 'self_attention.linear_qkv.weight'] = torch.full(
            (3 * d_model, d_model), float(base + 2)
        )
        sd[p + 'self_attention.linear_proj.weight'] = torch.full(
            (d_model, d_model), float(base + 3)
        )
        sd[p + 'pre_mlp_layernorm.weight'] = torch.full((d_model,), float(base + 4))
        sd[p + 'mlp.linear_fc1.weight'] = torch.full((ffn, d_model), float(base + 5))
        sd[p + 'mlp.linear_fc2.weight'] = torch.full((d_model, ffn), float(base + 6))
    sd['decoder.final_layernorm.weight'] = torch.full((d_model,), 99.0)
    sd['output_layer.weight'] = torch.full((10, d_model), 42.0)
    return sd


def _make_conv_args():
    """Minimal args object accepted by convert_gpt_to_hybrid."""
    return types.SimpleNamespace(
        d_model=32,
        mamba_d_inner=64,
        mamba_d_state=16,
        mamba2_n_groups=2,
        mamba2_n_heads=4,
        mamba2_head_dim=16,
        d_conv=4,
        init_method_std=0.02,
    )


def _write_gpt_ckpt(root, num_layers=3, d_model=32, ffn=64, iteration=5):
    """Persist a self-contained GPT dist checkpoint under ``root``.

    Returns the load-root (the directory containing
    ``latest_checkpointed_iteration.txt``).
    """
    iter_dir = os.path.join(root, f'iter_{iteration:07d}')
    model = _make_gpt_state_dict(num_layers, d_model=d_model, ffn=ffn)
    common = {
        'args': types.SimpleNamespace(
            num_layers=num_layers,
            hidden_size=d_model,
            hybrid_layer_pattern=None,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
        ),
        'iteration': iteration,
    }
    save_dist_checkpoint_full(model, common, iter_dir, model_prefix='model.')
    write_latest_iteration_marker(iter_dir, iteration)
    return root


# ---------------------------------------------------------------------------
# Pattern parsing / mapping
# ---------------------------------------------------------------------------


def test_parse_hybrid_layer_pattern_strips_mtp_and_pipe():
    assert parse_hybrid_layer_pattern('M*-|M*-') == list('M*-M*-')
    assert parse_hybrid_layer_pattern('M*-/MM/MM') == list('M*-')


def test_parse_hybrid_layer_pattern_rejects_bad_symbol():
    with pytest.raises(ValueError, match="Invalid layer symbol"):
        parse_hybrid_layer_pattern('M*Z-')


def test_build_layer_index_mapping_gpt_to_hybrid_pairs_attn_and_mlp():
    layer_types = parse_hybrid_layer_pattern('M*-M*-M*-')
    attn_map, mlp_map, ssm_indices = build_layer_index_mapping(layer_types, 'gpt-to-hybrid')
    # 3 GPT layers -> hybrid attn positions at indices 1, 4, 7 and mlp at 2, 5, 8.
    assert attn_map == {0: 1, 1: 4, 2: 7}
    assert mlp_map == {0: 2, 1: 5, 2: 8}
    assert ssm_indices == [0, 3, 6]


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------


def test_validate_pattern_rejects_gdn_symbol():
    with pytest.raises(ValueError, match="not GPT-compatible"):
        validate_pattern_gpt_compatible(list('M*-G'), 'gpt-to-hybrid')


def test_validate_pattern_rejects_dsa_symbol():
    # 'D' is a valid production symbol but not GPT-compatible; parse would
    # reject it too, so drive the validator directly with a hand-built list.
    with pytest.raises(ValueError, match="not GPT-compatible"):
        validate_pattern_gpt_compatible(['M', '*', '-', 'D'], 'gpt-to-hybrid')


def test_validate_pattern_rejects_mixed_dense_and_moe():
    with pytest.raises(ValueError, match="mixes '-'.*'E'"):
        validate_pattern_gpt_compatible(list('M*-M*E'), 'gpt-to-hybrid')


def test_validate_pattern_rejects_unequal_attn_and_mlp():
    with pytest.raises(ValueError, match="pair every attention layer"):
        # Two '*' but only one '-'.
        validate_pattern_gpt_compatible(list('M**-'), 'gpt-to-hybrid')


# ---------------------------------------------------------------------------
# Pure state-dict conversion
# ---------------------------------------------------------------------------


def test_convert_gpt_to_hybrid_maps_attention_and_mlp_bit_for_bit():
    d_model = 32
    ffn = 64
    src = _make_gpt_state_dict(num_layers=3, d_model=d_model, ffn=ffn)
    pattern = 'M*-M*-M*-'
    layer_types = parse_hybrid_layer_pattern(pattern)
    args = _make_conv_args()

    tgt = convert_gpt_to_hybrid(src, layer_types, args)

    # GPT layer i's attention lands at hybrid layer (3*i + 1) — the i-th '*'.
    # GPT layer i's MLP lands at hybrid layer (3*i + 2) — the i-th '-'.
    for i in range(3):
        h_attn = 3 * i + 1
        h_mlp = 3 * i + 2
        src_p = f'decoder.layers.{i}.'
        tgt_attn_p = f'decoder.layers.{h_attn}.'
        tgt_mlp_p = f'decoder.layers.{h_mlp}.'
        for suffix in (
            'input_layernorm.weight',
            'self_attention.linear_qkv.weight',
            'self_attention.linear_proj.weight',
        ):
            assert torch.equal(
                tgt[tgt_attn_p + suffix], src[src_p + suffix]
            ), f'attention tensor mismatch at layer {i} ({suffix})'
        for suffix in (
            'pre_mlp_layernorm.weight',
            'mlp.linear_fc1.weight',
            'mlp.linear_fc2.weight',
        ):
            assert torch.equal(
                tgt[tgt_mlp_p + suffix], src[src_p + suffix]
            ), f'mlp tensor mismatch at layer {i} ({suffix})'

    # Non-layer params: embedding kept as-is, final_layernorm renamed to final_norm.
    assert torch.equal(
        tgt['embedding.word_embeddings.weight'], src['embedding.word_embeddings.weight']
    )
    assert 'decoder.final_norm.weight' in tgt
    assert torch.equal(tgt['decoder.final_norm.weight'], src['decoder.final_layernorm.weight'])


def test_convert_gpt_to_hybrid_initializes_ssm_layers():
    src = _make_gpt_state_dict(num_layers=2, d_model=32, ffn=64)
    layer_types = parse_hybrid_layer_pattern('M*-M*-')
    tgt = convert_gpt_to_hybrid(src, layer_types, _make_conv_args())

    # Both SSM positions (layer 0 and layer 3) get fresh mixer.* params.
    for ssm_layer in (0, 3):
        for name in (
            'in_proj.weight',
            'in_proj.layer_norm_weight',
            'conv1d.weight',
            'conv1d.bias',
            'A_log',
            'D',
            'dt_bias',
            'norm.weight',
            'out_proj.weight',
        ):
            k = f'decoder.layers.{ssm_layer}.mixer.{name}'
            assert k in tgt, f'missing SSM tensor {k}'
            assert not torch.isnan(tgt[k]).any(), f'NaN in {k}'


def test_initialize_ssm_layer_params_has_expected_tensor_shapes():
    """The SSM initializer produces the exact keys and shapes MambaMixer
    expects, given a small set of Mamba dimensions."""
    d_model, d_inner, d_state, n_groups, n_heads, head_dim, d_conv = (32, 64, 16, 2, 4, 16, 4)
    conv_dim = d_inner + 2 * n_groups * d_state
    in_proj_out_dim = 2 * d_inner + 2 * n_groups * d_state + n_heads

    params = initialize_ssm_layer_params(
        layer_idx=0,
        d_model=d_model,
        mamba_d_inner=d_inner,
        mamba_d_state=d_state,
        mamba2_n_groups=n_groups,
        mamba2_n_heads=n_heads,
        mamba_head_dim=head_dim,
        d_conv=d_conv,
    )

    prefix = 'decoder.layers.0.mixer.'
    expected_shapes = {
        prefix + 'in_proj.weight': (in_proj_out_dim, d_model),
        prefix + 'in_proj.layer_norm_weight': (d_model,),
        prefix + 'conv1d.weight': (conv_dim, 1, d_conv),
        prefix + 'conv1d.bias': (conv_dim,),
        prefix + 'A_log': (n_heads,),
        prefix + 'D': (n_heads,),
        prefix + 'dt_bias': (n_heads,),
        prefix + 'norm.weight': (d_inner,),
        prefix + 'out_proj.weight': (d_model, d_inner),
    }
    assert set(params.keys()) == set(expected_shapes.keys())
    for k, shape in expected_shapes.items():
        assert tuple(params[k].shape) == shape, f'{k}: got {tuple(params[k].shape)}, want {shape}'
        assert not torch.isnan(params[k]).any(), f'NaN in {k}'


# ---------------------------------------------------------------------------
# End-to-end auto-convert
# ---------------------------------------------------------------------------


def test_autoconvert_produces_loadable_hybrid_ckpt(tmp_path, monkeypatch):
    """Full round trip: build a GPT ckpt, autoconvert, reload the rewrite.

    This exercises the whole rank-0 read / convert / write / marker path and
    is the regression guard for the original bug where
    ``write_latest_iteration_marker`` was called with ``tmp_root`` instead of
    ``iter_dir`` — the marker never landed, and the downstream re-preload
    silently returned no checkpoint.
    """
    if int(os.environ.get('WORLD_SIZE', '1')) > 1:
        pytest.skip('single-process variant; covered by the distributed test')

    src_root = _write_gpt_ckpt(str(tmp_path / 'src'), num_layers=3)

    runtime_args = types.SimpleNamespace(
        hybrid_layer_pattern='M*-M*-M*-', **vars(_make_conv_args())
    )
    ckpt_args = types.SimpleNamespace(num_layers=3, hidden_size=32, hybrid_layer_pattern=None)

    # Point tempfile.mkdtemp at pytest's tmp_path so the temp ckpt lands under it.
    monkeypatch.setenv('TMPDIR', str(tmp_path))
    tmp_root = _autoconvert_gpt_to_hybrid_ckpt(src_root, ckpt_args, runtime_args, iteration=5)

    # (1) Marker written next to the iter_ dir.
    tracker = os.path.join(tmp_root, 'latest_checkpointed_iteration.txt')
    assert os.path.exists(tracker), (
        f'auto-convert did not leave a tracker file at {tracker}; the '
        f'downstream load would silently start from scratch.'
    )
    assert open(tracker).read().strip() == '5'
    assert os.path.isdir(os.path.join(tmp_root, 'iter_0000005'))

    # (2) The rewritten dir round-trips through load_dist_checkpoint_full.
    model, common, prefix, backend, iteration = load_dist_checkpoint_full(tmp_root)

    # Attention lands at the '*' positions (indices 1, 4, 7).
    src_expected = _make_gpt_state_dict(3)
    for gpt_i, hyb_i in ((0, 1), (1, 4), (2, 7)):
        for suffix in (
            'input_layernorm.weight',
            'self_attention.linear_qkv.weight',
            'self_attention.linear_proj.weight',
        ):
            assert torch.equal(
                model[f'decoder.layers.{hyb_i}.{suffix}'],
                src_expected[f'decoder.layers.{gpt_i}.{suffix}'],
            ), f'attention drift at gpt layer {gpt_i}'

    # SSM tensors materialized at the 'M' positions (indices 0, 3, 6).
    for m_idx in (0, 3, 6):
        assert f'decoder.layers.{m_idx}.mixer.in_proj.weight' in model
        assert f'decoder.layers.{m_idx}.mixer.A_log' in model

    # (3) Rewritten args reflect the hybrid layout with num_layers recomputed
    # from the pattern (9 total positions), not left at the stale-3 GPT value.
    new_args = common.get('args')
    assert new_args is not None
    assert new_args.hybrid_layer_pattern == 'M*-M*-M*-'
    assert new_args.num_layers == 9


def test_autoconvert_distributed_rank_zero_rewrite(tmp_path, monkeypatch):
    """All ranks load one checkpoint that only rank 0 converts.

    This is intentionally a real multi-process test rather than two
    independent pytest executions. It covers the temporary-path broadcast,
    the post-write barrier, and the collective DCP reload. Nonzero ranks
    replace the pure converter with a function that raises, proving that they
    never execute the expensive conversion.
    """
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    if world_size < 2:
        pytest.skip('requires torchrun with at least two processes')

    initialized_here = not torch.distributed.is_initialized()
    if initialized_here:
        torch.distributed.init_process_group(backend='gloo')

    rank = torch.distributed.get_rank()
    try:
        # pytest gives each worker a different tmp_path. Rank 0 chooses the
        # node-shared root and broadcasts it before any checkpoint I/O.
        shared_paths = [str(tmp_path / 'distributed') if rank == 0 else None]
        torch.distributed.broadcast_object_list(shared_paths, src=0)
        shared_root = shared_paths[0]
        src_root = os.path.join(shared_root, 'src')
        conversion_tmp = os.path.join(shared_root, 'tmp')
        if rank == 0:
            os.makedirs(conversion_tmp, exist_ok=True)
        torch.distributed.barrier()

        # DCP save is collective, so every rank participates in creating the
        # one source checkpoint at the rank-0-selected shared path.
        _write_gpt_ckpt(src_root, num_layers=3)
        torch.distributed.barrier()

        runtime_args = types.SimpleNamespace(
            hybrid_layer_pattern='M*-M*-M*-', **vars(_make_conv_args())
        )
        ckpt_args = types.SimpleNamespace(num_layers=3, hidden_size=32, hybrid_layer_pattern=None)

        if rank != 0:

            def _conversion_must_only_run_on_rank_zero(*_args, **_kwargs):
                raise AssertionError('GPT-to-Hybrid conversion ran on a nonzero rank')

            monkeypatch.setattr(
                'megatron.core.models.hybrid.conversion.convert_gpt_to_hybrid',
                _conversion_must_only_run_on_rank_zero,
            )

        # tempfile caches its selected directory, so patch both the
        # environment and the cache to make the multi-rank location explicit.
        monkeypatch.setenv('TMPDIR', conversion_tmp)
        monkeypatch.setattr(tempfile, 'tempdir', conversion_tmp)
        converted_root = _autoconvert_gpt_to_hybrid_ckpt(
            src_root, ckpt_args, runtime_args, iteration=5
        )

        converted_roots = [None] * world_size
        torch.distributed.all_gather_object(converted_roots, converted_root)
        assert converted_roots == [converted_root] * world_size

        model, common, prefix, backend, iteration = load_dist_checkpoint_full(converted_root)
        assert prefix == 'model.'
        assert backend == 'torch_dist'
        assert iteration == 5
        assert common['args'].hybrid_layer_pattern == 'M*-M*-M*-'

        # Every rank must observe exactly the same persisted initialization,
        # including newly-created SSM weights, not merely the copied GPT data.
        keys = (
            'decoder.layers.1.self_attention.linear_qkv.weight',
            'decoder.layers.0.mixer.in_proj.weight',
            'decoder.layers.0.mixer.A_log',
        )
        for key in keys:
            gathered = [None] * world_size
            torch.distributed.all_gather_object(gathered, model[key])
            assert all(torch.equal(gathered[0], tensor) for tensor in gathered[1:]), key

        torch.distributed.barrier()
    finally:
        if initialized_here and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def test_autoconvert_raises_on_null_hybrid_pattern(tmp_path, monkeypatch):
    """The pattern-synthesis path lives in ``load_checkpoint``; the helper
    itself should raise (not silently no-op) when it receives an unset
    pattern — this catches accidental callers that skip the synthesis step.
    """
    src_root = _write_gpt_ckpt(str(tmp_path / 'src'), num_layers=3)
    runtime_args = types.SimpleNamespace(hybrid_layer_pattern=None)
    ckpt_args = types.SimpleNamespace(num_layers=3, hidden_size=32)
    monkeypatch.setenv('TMPDIR', str(tmp_path))
    with pytest.raises(RuntimeError, match='Automatic GPT-to-Hybrid conversion failed'):
        _autoconvert_gpt_to_hybrid_ckpt(src_root, ckpt_args, runtime_args, iteration=5)


def test_write_latest_iteration_marker_rejects_non_iter_dir(tmp_path):
    """Defense-in-depth: passing the ckpt root instead of the iter_ subdir
    must raise, not silently do nothing (the shape of the original bug)."""
    with pytest.raises(ValueError, match='iter_XXXXXXX'):
        write_latest_iteration_marker(str(tmp_path), 5)


# ---------------------------------------------------------------------------
# --use-legacy-model routing
# ---------------------------------------------------------------------------


def test_select_hybrid_model_cfg_routes_to_gpt_when_legacy_flag_set(monkeypatch):
    """``--use-legacy-model`` is the "keep training GPT" escape hatch: the
    Hybrid entrypoint must build a ``GPTModelConfig`` (not a ``HybridModelConfig``)
    so auto GPT->Hybrid conversion in ``load_checkpoint`` never fires.

    This test stubs both config builders — we're checking the routing wire,
    not exercising the real (import-heavy) config builders.
    """
    import pretrain_hybrid

    calls = []

    def fake_gpt_cfg(args):
        calls.append(('gpt', args))
        return object()

    def fake_hybrid_cfg(args):
        calls.append(('hybrid', args))
        return object()

    monkeypatch.setattr(pretrain_hybrid, 'gpt_config_from_args', fake_gpt_cfg)
    monkeypatch.setattr(pretrain_hybrid, 'hybrid_config_from_args', fake_hybrid_cfg)

    legacy_args = types.SimpleNamespace(use_legacy_model=True)
    pretrain_hybrid.select_hybrid_model_cfg(legacy_args)
    assert calls == [('gpt', legacy_args)], '--use-legacy-model must route to gpt_config_from_args'

    calls.clear()
    hybrid_args = types.SimpleNamespace(use_legacy_model=False)
    pretrain_hybrid.select_hybrid_model_cfg(hybrid_args)
    assert calls == [
        ('hybrid', hybrid_args)
    ], 'default (flag unset) must route to hybrid_config_from_args'

    calls.clear()
    unset_args = types.SimpleNamespace()  # attribute missing entirely
    pretrain_hybrid.select_hybrid_model_cfg(unset_args)
    assert calls == [
        ('hybrid', unset_args)
    ], 'missing flag must be treated as False and route to hybrid'


def test_build_auto_convert_args_prefers_runtime_over_ckpt_and_falls_back():
    """The Mamba shape resolver in ``_build_auto_convert_args`` reaches for
    runtime args first, then the source checkpoint, then a small default set.
    Guard the priority so ``d_model``-style regressions can't sneak in.
    """
    from megatron.training.checkpointing import _build_auto_convert_args

    # Runtime supplies d_model; checkpoint supplies d_conv; the rest come from
    # the fallback defaults inside the helper.
    runtime = types.SimpleNamespace(d_model=64)
    ckpt = types.SimpleNamespace(d_conv=8)
    conv = _build_auto_convert_args(runtime, ckpt)
    assert conv.d_model == 64
    assert conv.d_conv == 8
    # Defaults: d_inner = 2*d_model; head_dim = 64; n_heads = d_inner/head_dim
    assert conv.mamba_d_inner == 128
    assert conv.mamba2_head_dim == 64
    assert conv.mamba2_n_heads == 128 // 64
    assert conv.mamba_d_state == 128
    assert conv.mamba2_n_groups == 8

    # Canonical runtime names ("mamba_state_dim", ...) also feed the converter
    # namespace when the converter-side names are absent.
    runtime2 = types.SimpleNamespace(
        d_model=32, mamba_state_dim=16, mamba_num_groups=4, mamba_num_heads=2, mamba_head_dim=16
    )
    conv2 = _build_auto_convert_args(runtime2, types.SimpleNamespace())
    assert (
        conv2.mamba_d_state,
        conv2.mamba2_n_groups,
        conv2.mamba2_n_heads,
        conv2.mamba2_head_dim,
    ) == (16, 4, 2, 16)


def test_build_auto_convert_args_raises_without_d_model():
    """No hidden_size, no d_model, no Mamba-shaped source args — the helper
    must fail fast rather than silently produce zero-sized SSM tensors.
    """
    from megatron.training.checkpointing import _build_auto_convert_args

    with pytest.raises(RuntimeError, match='cannot infer d_model'):
        _build_auto_convert_args(types.SimpleNamespace(), types.SimpleNamespace())
