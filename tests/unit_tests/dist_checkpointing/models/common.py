# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import math

import torch

from megatron.core import parallel_state
from megatron.core.dist_checkpointing import load, load_plain_tensors, save
from megatron.core.dist_checkpointing.dict_utils import diff
from megatron.core.dist_checkpointing.serialization import (
    get_default_load_sharded_strategy,
    get_default_save_sharded_strategy,
)
from megatron.core.dist_checkpointing.strategies.fully_parallel import (
    FullyParallelLoadStrategyWrapper,
    FullyParallelSaveStrategyWrapper,
)
from megatron.core.dist_checkpointing.validation import StrictHandling
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils


def common_test_simple_sharded_state_dict_save_load(
    initialize_model_fn, tmp_path_dist_ckpt, src_layer_spec_fn, dst_layer_spec_fn
):
    """Simple save and load sanity check, without any equality tests."""
    tp = 2
    pp = 4
    Utils.initialize_model_parallel(tp, pp)
    gpt_model = initialize_model_fn(
        1, src_layer_spec_fn, tensor_model_parallel_size=tp, pipeline_model_parallel_size=pp
    )
    with TempNamedDir(tmp_path_dist_ckpt / 'test_gpt_model') as ckpt_dir:
        # Save
        sharded_state_dict = gpt_model.sharded_state_dict()
        save(sharded_state_dict, ckpt_dir)

        # Load
        gpt_model = initialize_model_fn(
            2, dst_layer_spec_fn, tensor_model_parallel_size=tp, pipeline_model_parallel_size=pp
        )
        sharded_state_dict = gpt_model.sharded_state_dict()
        state_dict, missing_keys, unexpected_keys = load(
            sharded_state_dict, ckpt_dir, strict=StrictHandling.RETURN_ALL
        )
        # Potential mismatch is because of extra states which is ok
        assert all('_extra_state' in k for k in missing_keys)
        assert all('_extra_state' in k for k in unexpected_keys)
        gpt_model.load_state_dict(state_dict)
    Utils.destroy_model_parallel()


def common_test_parallel_reconfiguration_e2e(
    initialize_model_fn,
    tmp_path_dist_ckpt,
    src_tp_pp,
    dest_tp_pp,
    src_layer_spec_fn,
    dst_layer_spec_fn,
    use_fpsl,
    load_order="tp-dp-pp",
    store_order="tp-dp-pp",
    src_tp_pp_kwargs=None,
    dst_tp_pp_kwargs=None,
    src_model_init_kwargs=None,
    dst_model_init_kwargs=None,
):
    """Test model saving and loading with different TP/PP"""
    src_model_init_kwargs = src_model_init_kwargs or {}
    dst_model_init_kwargs = dst_model_init_kwargs or {}
    Utils.initialize_model_parallel(*src_tp_pp, **(src_tp_pp_kwargs or {}), order=load_order)
    with TempNamedDir(
        tmp_path_dist_ckpt / 'test_gpt_model_reconfiguration_model_A'
    ) as ckpt_dir_A, TempNamedDir(
        tmp_path_dist_ckpt / 'test_gpt_model_reconfiguration_model_B'
    ) as ckpt_dir_B:
        # Save checkpoint A
        gpt_model_A = initialize_model_fn(
            1,
            src_layer_spec_fn,
            tensor_model_parallel_size=src_tp_pp[0],
            pipeline_model_parallel_size=src_tp_pp[1],
            **src_model_init_kwargs,
        )
        save_strategy = get_default_save_sharded_strategy()
        if use_fpsl:
            save_strategy = FullyParallelSaveStrategyWrapper(
                save_strategy,
                parallel_state.get_data_parallel_group(with_context_parallel=True),
                True,
            )
        save(gpt_model_A.sharded_state_dict(), ckpt_dir_A, save_strategy)
        regular_state_dict_A = gpt_model_A.state_dict()
        Utils.destroy_model_parallel()

        # Load checkpoint A with different TP/PP and save as checkpoint B
        # No FPS this time, only FPL
        Utils.initialize_model_parallel(*dest_tp_pp, **(dst_tp_pp_kwargs or {}), order=store_order)
        gpt_model_B = initialize_model_fn(
            2,
            dst_layer_spec_fn,
            tensor_model_parallel_size=dest_tp_pp[0],
            pipeline_model_parallel_size=dest_tp_pp[1],
            **dst_model_init_kwargs,
        )
        if use_fpsl:
            load_strategy = get_default_load_sharded_strategy(ckpt_dir_A)
            load_strategy = FullyParallelLoadStrategyWrapper(load_strategy)
        else:
            load_strategy = None
        state_dict, missing_keys, unexpected_keys = load(
            gpt_model_B.sharded_state_dict(),
            ckpt_dir_A,
            load_strategy,
            strict=StrictHandling.RETURN_ALL,
        )
        # Potential mismatch is because of extra states which is ok
        assert all('_extra_state' in k for k in missing_keys)
        assert all('_extra_state' in k for k in unexpected_keys)
        gpt_model_B.load_state_dict(state_dict)
        save(gpt_model_B.sharded_state_dict(), ckpt_dir_B)
        regular_state_dict_B = gpt_model_A.state_dict()
        Utils.destroy_model_parallel()

        # Test both checkpoints are equal
        Utils.initialize_model_parallel(1, 1)
        plain_state_dict_A = load_plain_tensors(ckpt_dir_A)
        plain_state_dict_B = load_plain_tensors(ckpt_dir_B)
        diffs = diff(plain_state_dict_A, plain_state_dict_B)
        assert not any(map(bool, diffs)), diffs

        # Test both regular state dicts are equal, turning FP8 states to bytes first
        regular_state_dict_A = {
            k: v for k, v in regular_state_dict_A.items() if not k.endswith('_extra_state')
        }
        regular_state_dict_B = {
            k: v for k, v in regular_state_dict_B.items() if not k.endswith('_extra_state')
        }
        diffs = diff(regular_state_dict_A, regular_state_dict_B)
        assert not any(map(bool, diffs)), diffs
        Utils.destroy_model_parallel()


def common_test_state_dict_comparison(initialize_model_fn, tmp_path_dist_ckpt):
    tp = 2
    pp = 4
    Utils.initialize_model_parallel(tp, pp)
    with TempNamedDir(
        tmp_path_dist_ckpt / 'test_state_dict_comparison_A'
    ) as ckpt_dir_A, TempNamedDir(
        tmp_path_dist_ckpt / 'test_state_dict_comparison_B'
    ) as ckpt_dir_B:
        gpt_model_A = initialize_model_fn(
            1, tensor_model_parallel_size=tp, pipeline_model_parallel_size=pp
        )
        save(gpt_model_A.sharded_state_dict(), ckpt_dir_A)
        gpt_model_B = initialize_model_fn(
            2, tensor_model_parallel_size=tp, pipeline_model_parallel_size=pp
        )
        save(gpt_model_B.sharded_state_dict(), ckpt_dir_B)

        state_dict_A = load_plain_tensors(ckpt_dir_A)
        state_dict_A_dup = load_plain_tensors(ckpt_dir_A)
        state_dict_B = load_plain_tensors(ckpt_dir_B)

        # Test that A matches A
        diffs = diff(state_dict_A, state_dict_A_dup)
        assert not any(map(bool, diffs)), diffs

        # Test that A *keys* match B *keys*, but the tensors content is different
        only_left, only_right, mismatch = diff(state_dict_A, state_dict_B)
        assert not only_left and not only_right, (only_left, only_right)
        assert len(mismatch) == len(state_dict_A), (len(mismatch), (len(state_dict_A)))
    Utils.destroy_model_parallel()


def common_test_vocab_size_padding_change(
    initialize_model_fn, tmp_path_dist_ckpt, vocab_size_base, src_tp_pp, dest_tp_pp
):
    """Test model loading with different vocab size (caused by TP padding)."""

    def get_test_vocab_size(make_divisible_by=128):
        divisor = make_divisible_by * parallel_state.get_tensor_model_parallel_world_size()
        return int(math.ceil(vocab_size_base / divisor)) * divisor

    vocab_size_dependent_keys = {
        'output_layer.weight',
        'output_layer.bias',
        'embedding.word_embeddings.weight',
    }

    with TempNamedDir(
        tmp_path_dist_ckpt / 'test_vocab_size_padding_change_A'
    ) as ckpt_dir_A, TempNamedDir(
        tmp_path_dist_ckpt / 'test_vocab_size_padding_change_B'
    ) as ckpt_dir_B:
        # Save checkpoint A
        Utils.initialize_model_parallel(*src_tp_pp)
        gpt_model_A = initialize_model_fn(
            1,
            tensor_model_parallel_size=src_tp_pp[0],
            pipeline_model_parallel_size=src_tp_pp[1],
            vocab_size=get_test_vocab_size(),
        )
        save(gpt_model_A.sharded_state_dict(), ckpt_dir_A)
        Utils.destroy_model_parallel()

        # Load checkpoint A with different TP/PP and save as checkpoint B
        Utils.initialize_model_parallel(*dest_tp_pp)
        gpt_model_B = initialize_model_fn(
            2,
            tensor_model_parallel_size=dest_tp_pp[0],
            pipeline_model_parallel_size=dest_tp_pp[1],
            vocab_size=get_test_vocab_size(),
        )
        state_dict = load(gpt_model_B.sharded_state_dict(), ckpt_dir_A)
        gpt_model_B.load_state_dict(state_dict)
        save(gpt_model_B.sharded_state_dict(), ckpt_dir_B)
        Utils.destroy_model_parallel()

        # Test equality
        Utils.initialize_model_parallel(1, 1)
        plain_state_dict_A = load_plain_tensors(ckpt_dir_A)
        plain_state_dict_B = load_plain_tensors(ckpt_dir_B)
        # Test vocab size dependent keys are equal up to `vocab_size_base`
        for vocab_layer_key in vocab_size_dependent_keys:
            if vocab_layer_key in plain_state_dict_A:
                ten_A = plain_state_dict_A.pop(vocab_layer_key)
                ten_B = plain_state_dict_B.pop(vocab_layer_key)
                assert torch.all(
                    ten_A[:vocab_size_base] == ten_B[:vocab_size_base]
                ), vocab_layer_key

        # Test other tensors are equal
        diffs = diff(plain_state_dict_A, plain_state_dict_B)
        assert not any(map(bool, diffs)), diffs
        Utils.destroy_model_parallel()
