# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for loading GPT checkpoints into HybridModel runs.

Covers the load-time sharded state dict retargeting in
``megatron.core.models.hybrid.gpt_checkpoint_interop``:

* pure layer-map derivation and validation (no GPU state),
* pure key retargeting on synthetic sharded state dicts,
* end-to-end: save a GPTModel dist checkpoint under one (TP, PP, EP, ETP)
  layout, load it into a HybridModel under another layout, and verify
  attention/MLP weights round-trip bit-for-bit while layers without a GPT
  counterpart keep their fresh initialization.
"""

from functools import partial
from unittest import mock

import pytest
import torch

from megatron.core import parallel_state as ps
from megatron.core.dist_checkpointing import load, load_plain_tensors, save
from megatron.core.dist_checkpointing.dict_utils import diff
from megatron.core.dist_checkpointing.mapping import (
    LocalNonpersistentObject,
    ShardedObject,
    ShardedTensor,
    ShardedTensorFactory,
)
from megatron.core.dist_checkpointing.validation import StrictHandling
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.hybrid.gpt_checkpoint_interop import (
    gpt_compatible_layer_maps,
    retarget_sharded_state_dict_to_gpt_checkpoint,
)
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.arguments import parse_args
from megatron.training.checkpointing import load_checkpoint, save_checkpoint
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.dist_checkpointing.utils import (
    init_checkpointing_mock_args,
    setup_model_and_optimizer,
)
from tests.unit_tests.test_utilities import Utils


class TestGPTCompatLayerMaps:
    def test_pairs_positions_in_pattern_order(self):
        maps = gpt_compatible_layer_maps('M*-M*-')
        assert maps.attention_to_gpt == {1: 0, 4: 1}
        assert maps.mlp_to_gpt == {2: 0, 5: 1}
        assert maps.fresh_init == frozenset({0, 3})
        assert maps.num_gpt_layers == 2

    def test_pipeline_separators_are_ignored(self):
        assert gpt_compatible_layer_maps('M*-|M*-') == gpt_compatible_layer_maps('M*-M*-')

    def test_moe_positions_pair_like_dense_ones(self):
        maps = gpt_compatible_layer_maps('M*EM*E')
        assert maps.attention_to_gpt == {1: 0, 4: 1}
        assert maps.mlp_to_gpt == {2: 0, 5: 1}
        assert maps.num_gpt_layers == 2

    def test_attention_only_positions_can_precede_all_mlps(self):
        # Pairing is positional per type, not adjacency-based.
        maps = gpt_compatible_layer_maps('**--')
        assert maps.attention_to_gpt == {0: 0, 1: 1}
        assert maps.mlp_to_gpt == {2: 0, 3: 1}
        assert maps.fresh_init == frozenset()

    def test_rejects_empty_pattern(self):
        with pytest.raises(ValueError, match='empty'):
            gpt_compatible_layer_maps(None)

    def test_rejects_mtp_pattern(self):
        with pytest.raises(ValueError, match='MTP'):
            gpt_compatible_layer_maps('M*-M*-/MM/MM')

    def test_rejects_untranslatable_layer_types(self):
        with pytest.raises(ValueError, match='cannot be translated'):
            gpt_compatible_layer_maps('M*-G*-')
        # 'D' cannot be combined with '*' at all, so use a pattern the
        # production parser accepts and let the interop validation reject it.
        with pytest.raises(ValueError, match='cannot be translated'):
            gpt_compatible_layer_maps('MD-')

    def test_rejects_mixed_dense_and_moe(self):
        with pytest.raises(ValueError, match="one of '-' or 'E'"):
            gpt_compatible_layer_maps('M*-M*E')

    def test_rejects_unbalanced_attention_and_mlp(self):
        with pytest.raises(ValueError, match='equal, nonzero'):
            gpt_compatible_layer_maps('M**-')
        with pytest.raises(ValueError, match='equal, nonzero'):
            gpt_compatible_layer_maps('MMMM')


def _sharded_tensor(key):
    return ShardedTensor.from_rank_offsets(key, torch.ones(4))


class TestRetargetShardedStateDict:
    def test_keys_point_at_gpt_layout_and_fresh_layers_stay_local(self):
        # GPT checkpoints use the homogeneous layer format: numberless keys
        # with the layer index as the leading sharding axis.
        maps = gpt_compatible_layer_maps('M*-')
        mixer_weight = torch.full((4,), 7.0)
        sharded_sd = {
            'attn': _sharded_tensor('decoder.layers.1.self_attention.linear_qkv.weight'),
            'mlp': _sharded_tensor('decoder.layers.2.mlp.linear_fc1.layer_norm_weight'),
            'mixer': ShardedTensor.from_rank_offsets(
                'decoder.layers.0.mixer.in_proj.weight', mixer_weight
            ),
            'final_norm': _sharded_tensor('decoder.final_norm.weight'),
            'embedding': _sharded_tensor('embedding.word_embeddings.weight'),
            'nested': {
                'proj': _sharded_tensor('decoder.layers.1.self_attention.linear_proj.weight')
            },
        }

        retarget_sharded_state_dict_to_gpt_checkpoint(sharded_sd, maps)

        attn = sharded_sd['attn']
        assert attn.key == 'decoder.layers.self_attention.linear_qkv.weight'
        assert attn.prepend_axis_num == 1
        assert attn.global_shape == (1, 4)
        assert attn.global_offset == (0, 0)
        assert attn.axis_fragmentations == (1, 1)
        assert sharded_sd['mlp'].key == 'decoder.layers.mlp.linear_fc1.layer_norm_weight'
        assert sharded_sd['mlp'].global_offset == (0, 0)
        assert (
            sharded_sd['nested']['proj'].key == 'decoder.layers.self_attention.linear_proj.weight'
        )
        assert sharded_sd['final_norm'].key == 'decoder.final_layernorm.weight'
        assert sharded_sd['final_norm'].prepend_axis_num == 0
        assert sharded_sd['embedding'].key == 'embedding.word_embeddings.weight'
        assert isinstance(sharded_sd['mixer'], LocalNonpersistentObject)
        assert sharded_sd['mixer'].unwrap() is mixer_weight

    def test_extra_state_and_factory_entries_follow_the_layer_axis(self):
        maps = gpt_compatible_layer_maps('M*-M*-')  # 2 GPT layers
        extra_state = ShardedObject(
            'decoder.layers.5.mlp.linear_fc2._extra_state', None, (1,), (0,)
        )

        def build_fn(key, data, replica_id, flattened_range):
            return {
                'chunk': ShardedTensor.from_rank_offsets(
                    f'{key}_chunk', data, replica_id=replica_id
                )
            }

        factory = ShardedTensorFactory(
            'decoder.layers.2.mlp.linear_fc1.weight',
            torch.ones(4),
            build_fn,
            lambda sd: sd['chunk'],
        )
        sharded_sd = {'extra_state': extra_state, 'factory': factory}

        retarget_sharded_state_dict_to_gpt_checkpoint(sharded_sd, maps)

        # Hybrid layer 5 is the 2nd MLP position -> GPT layer 1.
        assert extra_state.key == 'decoder.layers.mlp.linear_fc2._extra_state'
        assert extra_state.global_shape == (2,)
        assert extra_state.global_offset == (1,)
        # Hybrid layer 2 is the 1st MLP position -> GPT layer 0; sub-tensors
        # built by the factory inherit the layer axis.
        assert factory.key == 'decoder.layers.mlp.linear_fc1.weight'
        built = factory.build()
        assert built['chunk'].key == 'decoder.layers.mlp.linear_fc1.weight_chunk'
        assert built['chunk'].prepend_axis_num == 1
        assert built['chunk'].global_shape == (2, 4)
        assert built['chunk'].global_offset == (0, 0)

    def test_layer_outside_pattern_raises(self):
        maps = gpt_compatible_layer_maps('M*-')
        sharded_sd = {'bad': _sharded_tensor('decoder.layers.7.self_attention.linear_qkv.weight')}
        with pytest.raises(ValueError, match='not part of the hybrid layer pattern'):
            retarget_sharded_state_dict_to_gpt_checkpoint(sharded_sd, maps)

    def test_optimizer_state_entries_retarget_like_the_model(self):
        # The distributed optimizer's model-space sharded state dict embeds the
        # model key under ``optimizer.state.<state>.<model_key>`` and mirrors the
        # model param's sharding, so the same retargeting must point the moments
        # and fp32 master params at the GPT checkpoint and keep fresh-layer
        # optimizer state local.
        maps = gpt_compatible_layer_maps('M*-')
        fresh_exp_avg = torch.full((4,), 3.0)
        optim_sd = {
            'param_state': {
                # attention position (hybrid layer 1 -> GPT layer 0)
                0: {
                    'exp_avg': _sharded_tensor(
                        'optimizer.state.exp_avg.decoder.layers.1.self_attention.linear_qkv.weight'
                    ),
                    'fp32_param': _sharded_tensor(
                        'optimizer.state.fp32_param.decoder.layers.1.self_attention.linear_qkv.weight'
                    ),
                },
                # MLP position (hybrid layer 2 -> GPT layer 0)
                1: {
                    'exp_avg_sq': _sharded_tensor(
                        'optimizer.state.exp_avg_sq.decoder.layers.2.mlp.linear_fc1.weight'
                    )
                },
                # fresh Mamba position (hybrid layer 0) -> stays local
                2: {
                    'exp_avg': ShardedTensor.from_rank_offsets(
                        'optimizer.state.exp_avg.decoder.layers.0.mixer.in_proj.weight',
                        fresh_exp_avg,
                    )
                },
            },
            'param_state_sharding_type': 'fully_sharded_model_space',
        }

        retarget_sharded_state_dict_to_gpt_checkpoint(optim_sd, maps)

        attn = optim_sd['param_state'][0]['exp_avg']
        assert attn.key == 'optimizer.state.exp_avg.decoder.layers.self_attention.linear_qkv.weight'
        assert attn.prepend_axis_num == 1
        assert attn.global_shape == (1, 4)
        assert attn.global_offset == (0, 0)
        master = optim_sd['param_state'][0]['fp32_param']
        assert (
            master.key
            == 'optimizer.state.fp32_param.decoder.layers.self_attention.linear_qkv.weight'
        )
        assert optim_sd['param_state'][1]['exp_avg_sq'].key == (
            'optimizer.state.exp_avg_sq.decoder.layers.mlp.linear_fc1.weight'
        )
        fresh = optim_sd['param_state'][2]['exp_avg']
        assert isinstance(fresh, LocalNonpersistentObject)
        assert fresh.unwrap() is fresh_exp_avg
        # Non-sharded bookkeeping is passed through untouched.
        assert optim_sd['param_state_sharding_type'] == 'fully_sharded_model_space'


def _base_config_kwargs(parallel, moe, glu):
    tp, pp, ep, etp = parallel
    config_kwargs = dict(
        num_attention_heads=8,
        # for Mamba: expand=2, headdim=64 -> nheads=8 (divisible by ngroups=8)
        hidden_size=256,
        use_cpu_initialization=True,
        pipeline_dtype=torch.bfloat16,
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=pp,
        sequence_parallel=(tp > 1 and ep > 1),
        # gated MLPs exercise the swiglu ShardedTensorFactory path
        gated_linear_unit=glu,
        add_bias_linear=not glu,
    )
    if moe:
        config_kwargs.update(
            num_moe_experts=8,
            moe_grouped_gemm=True,  # the hybrid moe spec is built with grouped GEMM experts
            add_bias_linear=False,
            moe_router_topk=2,
            expert_model_parallel_size=ep,
            expert_tensor_parallel_size=etp,
        )
    return config_kwargs


def initialize_gpt_model(seed, num_gpt_layers, parallel, moe, glu=False):
    torch.manual_seed(seed)
    model_parallel_cuda_manual_seed(seed)

    config = TransformerConfig(num_layers=num_gpt_layers, **_base_config_kwargs(parallel, moe, glu))
    if moe:
        layer_spec = get_gpt_decoder_block_spec(config, use_transformer_engine=True)
    else:
        layer_spec = get_gpt_layer_with_transformer_engine_spec()
    model = GPTModel(
        config=config,
        transformer_layer_spec=layer_spec,
        vocab_size=128,
        max_sequence_length=4,
        pre_process=ps.is_pipeline_first_stage(),
        post_process=ps.is_pipeline_last_stage(),
        position_embedding_type='rope',
        share_embeddings_and_output_weights=True,
    )
    with torch.no_grad():
        for param in model.parameters():
            param.random_()
    return model


def initialize_hybrid_model(seed, pattern, parallel, moe, glu=False):
    torch.manual_seed(seed)
    model_parallel_cuda_manual_seed(seed)

    num_layers = len(pattern.replace('|', ''))
    config = TransformerConfig(num_layers=num_layers, **_base_config_kwargs(parallel, moe, glu))
    return HybridModel(
        config=config,
        hybrid_stack_spec=hybrid_stack_spec,
        vocab_size=128,
        max_sequence_length=4,
        hybrid_layer_pattern=pattern,
        pre_process=ps.is_pipeline_first_stage(),
        post_process=ps.is_pipeline_last_stage(),
        position_embedding_type='rope',
        share_embeddings_and_output_weights=True,
    )


def _snapshot_fresh_layers(hybrid_model, layer_maps):
    """Clone all tensors of layers that must keep their fresh initialization."""
    snapshot = {}
    for layer in hybrid_model.decoder.layers:
        global_idx = layer.layer_number - 1
        if global_idx in layer_maps.fresh_init:
            snapshot[global_idx] = {
                name: tensor.detach().clone()
                for name, tensor in layer.state_dict().items()
                if isinstance(tensor, torch.Tensor)
            }
    return snapshot


def _assert_fresh_layers_untouched(hybrid_model, layer_maps, snapshot):
    for layer in hybrid_model.decoder.layers:
        global_idx = layer.layer_number - 1
        if global_idx not in layer_maps.fresh_init:
            continue
        for name, tensor in layer.state_dict().items():
            if not isinstance(tensor, torch.Tensor):
                continue
            assert torch.equal(
                tensor, snapshot[global_idx][name]
            ), f'fresh layer {global_idx} tensor {name} was overwritten by the GPT load'


def _drop_extra_state(plain_state_dict):
    return {k: v for k, v in plain_state_dict.items() if '_extra_state' not in k}


class TestGPTToHybridLoad:
    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    @pytest.mark.parametrize(
        ('src_parallel', 'dest_parallel', 'pattern', 'moe', 'glu'),
        [
            # (tp, pp, ep, etp) of the GPT save -> of the hybrid load.
            # Dense: TP/PP resharding.
            ((1, 1, 1, 1), (1, 1, 1, 1), 'M*-M*-M*-M*-', False, False),
            ((2, 1, 1, 1), (1, 2, 1, 1), 'M*-M*-M*-M*-', False, False),
            ((1, 2, 1, 1), (2, 1, 1, 1), 'M*-M*-M*-M*-', False, False),
            ((2, 2, 1, 1), (4, 1, 1, 1), 'M*-M*-M*-M*-', False, False),
            ((4, 1, 1, 1), (2, 4, 1, 1), 'M*-M*-M*-M*-', False, False),
            # Gated MLP exercises the swiglu factory path.
            ((2, 1, 1, 1), (1, 2, 1, 1), 'M*-M*-M*-M*-', False, True),
            # Pipeline stage boundaries given explicitly with '|'.
            ((1, 2, 1, 1), (1, 2, 1, 1), 'M*-M*-|M*-M*-', False, False),
            # MoE: EP/ETP resharding (ETP defaults to TP when 1).
            ((1, 1, 1, 1), (1, 1, 4, 1), 'M*EM*E', True, False),
            ((1, 1, 4, 1), (2, 1, 1, 2), 'M*EM*E', True, False),
            ((2, 1, 2, 2), (1, 1, 8, 1), 'M*EM*E', True, False),
            ((1, 1, 2, 1), (4, 1, 2, 4), 'M*EM*E', True, False),
            # MoE with PP as well.
            ((2, 1, 2, 2), (1, 2, 2, 1), 'M*EM*EM*EM*E', True, False),
        ],
    )
    def test_gpt_checkpoint_loads_into_hybrid_across_parallel_layouts(
        self, tmp_path_dist_ckpt, src_parallel, dest_parallel, pattern, moe, glu
    ):
        layer_maps = gpt_compatible_layer_maps(pattern)
        src_tp, src_pp, src_ep, src_etp = src_parallel
        dest_tp, dest_pp, dest_ep, dest_etp = dest_parallel

        Utils.initialize_model_parallel(
            src_tp, src_pp, expert_model_parallel_size=src_ep, expert_tensor_parallel_size=src_etp
        )
        with (
            TempNamedDir(tmp_path_dist_ckpt / 'gpt_hybrid_interop_gpt_src') as ckpt_dir_gpt,
            TempNamedDir(tmp_path_dist_ckpt / 'gpt_hybrid_interop_roundtrip') as ckpt_dir_back,
        ):
            # Save a GPT checkpoint under the source parallel layout.
            gpt_model = initialize_gpt_model(1, layer_maps.num_gpt_layers, src_parallel, moe, glu)
            save(gpt_model.sharded_state_dict(), ckpt_dir_gpt)
            Utils.destroy_model_parallel()

            # Load it into a hybrid model under the destination layout by
            # retargeting the hybrid sharded state dict, exactly as
            # load_checkpoint does for GPT checkpoints.
            Utils.initialize_model_parallel(
                dest_tp,
                dest_pp,
                expert_model_parallel_size=dest_ep,
                expert_tensor_parallel_size=dest_etp,
            )
            hybrid_model = initialize_hybrid_model(2, pattern, dest_parallel, moe, glu)
            fresh_snapshot = _snapshot_fresh_layers(hybrid_model, layer_maps)

            sharded_sd = hybrid_model.sharded_state_dict()
            retarget_sharded_state_dict_to_gpt_checkpoint(sharded_sd, layer_maps)
            state_dict, missing_keys, unexpected_keys = load(
                sharded_sd, ckpt_dir_gpt, strict=StrictHandling.RETURN_ALL
            )
            # Any mismatch beyond TE extra states means the retargeting missed keys.
            assert all('_extra_state' in k for k in missing_keys), missing_keys
            assert all('_extra_state' in k for k in unexpected_keys), unexpected_keys
            hybrid_model.load_state_dict(state_dict)

            _assert_fresh_layers_untouched(hybrid_model, layer_maps, fresh_snapshot)

            # Save the hybrid model back under GPT keys (fresh layers stay
            # local and are skipped) and compare both checkpoints tensorwise.
            sharded_sd_back = hybrid_model.sharded_state_dict()
            retarget_sharded_state_dict_to_gpt_checkpoint(sharded_sd_back, layer_maps)
            save(sharded_sd_back, ckpt_dir_back)
            Utils.destroy_model_parallel()

            Utils.initialize_model_parallel(1, 1)
            plain_gpt = _drop_extra_state(load_plain_tensors(ckpt_dir_gpt))
            plain_back = _drop_extra_state(load_plain_tensors(ckpt_dir_back))
            only_gpt, only_back, mismatch = diff(plain_gpt, plain_back)
            assert not only_back, f'roundtrip produced keys missing from the GPT ckpt: {only_back}'
            assert not only_gpt, f'GPT ckpt keys not covered by the hybrid load: {only_gpt}'
            assert not mismatch, f'weights changed by the GPT->hybrid->GPT roundtrip: {mismatch}'


# ---------------------------------------------------------------------------
# End-to-end optimizer loading through save_checkpoint / load_checkpoint.
# ---------------------------------------------------------------------------

_OPT_HIDDEN = 256
_OPT_HEADS = 8


def _opt_provider_config(num_layers, **config_kwargs):
    # get_model passes these through; they are not TransformerConfig fields.
    for extra in ('pg_collection', 'config', 'vp_stage'):
        config_kwargs.pop(extra, None)
    return TransformerConfig(
        num_layers=num_layers,
        hidden_size=_OPT_HIDDEN,
        num_attention_heads=_OPT_HEADS,
        use_cpu_initialization=True,
        add_bias_linear=True,
        gated_linear_unit=False,
        **config_kwargs,
    )


def gpt_provider_for_opt(pre_process=True, post_process=True, *, seed=0, num_gpt_layers, **kw):
    torch.manual_seed(seed)
    model_parallel_cuda_manual_seed(seed)
    return GPTModel(
        config=_opt_provider_config(num_gpt_layers, **kw),
        transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
        vocab_size=128,
        max_sequence_length=4,
        pre_process=pre_process,
        post_process=post_process,
        position_embedding_type='rope',
        share_embeddings_and_output_weights=True,
    )


def hybrid_provider_for_opt(pre_process=True, post_process=True, *, seed=0, pattern, **kw):
    torch.manual_seed(seed)
    model_parallel_cuda_manual_seed(seed)
    num_layers = len(pattern.replace('|', ''))
    return HybridModel(
        config=_opt_provider_config(num_layers, **kw),
        hybrid_stack_spec=hybrid_stack_spec,
        vocab_size=128,
        max_sequence_length=4,
        hybrid_layer_pattern=pattern,
        pre_process=pre_process,
        post_process=post_process,
        position_embedding_type='rope',
        share_embeddings_and_output_weights=True,
    )


def _inner_optimizers(optimizer):
    if hasattr(optimizer, 'chained_optimizers'):
        return [o for opt in optimizer.chained_optimizers for o in _inner_optimizers(opt)]
    inner = getattr(optimizer, 'optimizer', None)
    return [inner] if inner is not None else []


def _optimizer_moment_fingerprint(optimizer):
    """Sum of norms of every floating-point optimizer-state tensor (per rank)."""
    total = 0.0
    for inner in _inner_optimizers(optimizer):
        for state in inner.state.values():
            for value in state.values():
                if torch.is_tensor(value) and value.is_floating_point():
                    total += value.detach().double().norm().item()
    return total


class TestGPTToHybridOptimizerLoad:
    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    @pytest.mark.parametrize('pattern', ['*-*-', 'M*-M*-'])
    def test_gpt_optimizer_state_loads_into_hybrid(self, tmp_path_dist_ckpt, pattern):
        layer_maps = gpt_compatible_layer_maps(pattern)
        num_gpt_layers = layer_maps.num_gpt_layers

        Utils.initialize_model_parallel(1, 1)
        with TempNamedDir(tmp_path_dist_ckpt / 'gpt_hybrid_opt_interop') as ckpt_dir:
            mock_args = parse_args(ignore_unknown_args=True)
            with mock.patch(
                'megatron.training.checkpointing.get_args', new=lambda: mock_args
            ):
                # Build a GPT model + distributed optimizer whose Adam moments are
                # seeded to random values, then save a full checkpoint.
                gpt_model, gpt_optimizer = setup_model_and_optimizer(
                    seed=2,
                    tp=1,
                    pp=1,
                    initialize_fn=partial(
                        gpt_provider_for_opt, num_gpt_layers=num_gpt_layers
                    ),
                )
                init_checkpointing_mock_args(mock_args, ckpt_dir, fully_parallel=True)
                mock_args.use_distributed_optimizer = True
                mock_args.hidden_size = _OPT_HIDDEN
                mock_args.num_attention_heads = _OPT_HEADS
                mock_args.num_layers = num_gpt_layers
                save_checkpoint(10, gpt_model, gpt_optimizer, None, 0)
                Utils.destroy_model_parallel()

                # Build a hybrid model + optimizer (independently seeded moments) and
                # load the GPT checkpoint, translating model and optimizer state.
                Utils.initialize_model_parallel(1, 1)
                hybrid_model, hybrid_optimizer = setup_model_and_optimizer(
                    seed=4,
                    tp=1,
                    pp=1,
                    initialize_fn=partial(hybrid_provider_for_opt, pattern=pattern),
                )
                fresh_snapshot = _snapshot_fresh_layers(hybrid_model, layer_maps)
                moments_before = _optimizer_moment_fingerprint(hybrid_optimizer)

                init_checkpointing_mock_args(mock_args, ckpt_dir, fully_parallel=True)
                mock_args.use_distributed_optimizer = True
                mock_args.finetune = True  # required for a GPT->hybrid load
                mock_args.hybrid_layer_pattern = pattern
                mock_args.hidden_size = _OPT_HIDDEN
                mock_args.num_attention_heads = _OPT_HEADS
                mock_args.num_layers = len(pattern.replace('|', ''))

                iteration, _ = load_checkpoint(hybrid_model, hybrid_optimizer, None)

                # Finetune semantics: iteration restarts even though the optimizer
                # moments were warm-started.
                assert iteration == 0
                # The optimizer state was actually loaded (GPT-sourced moments
                # overwrite the freshly seeded ones).
                moments_after = _optimizer_moment_fingerprint(hybrid_optimizer)
                assert abs(moments_after - moments_before) > 1e-6, (
                    'optimizer state does not appear to have been loaded'
                )
                # Layers without a GPT counterpart keep their fresh weights.
                _assert_fresh_layers_untouched(hybrid_model, layer_maps, fresh_snapshot)
