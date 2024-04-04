# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import math

import torch

from megatron.core.dist_checkpointing import save, load, load_plain_tensors
from megatron.core import parallel_state
from megatron.core.dist_checkpointing.dict_utils import diff
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils


def common_test_simple_sharded_state_dict_save_load(initialize_model_fn, tmp_path_dist_ckpt,
                                             src_layer_spec_fn, dst_layer_spec_fn):
    """ Simple save and load sanity check, without any equality tests. """
    Utils.initialize_model_parallel(2,4)
    gpt_model = initialize_model_fn(1, src_layer_spec_fn)
    with TempNamedDir(tmp_path_dist_ckpt / 'test_gpt_model') as ckpt_dir:
        # Save
        sharded_state_dict = gpt_model.sharded_state_dict()
        save(sharded_state_dict, ckpt_dir)

        # Load
        gpt_model = initialize_model_fn(2, dst_layer_spec_fn)
        sharded_state_dict = gpt_model.sharded_state_dict()
        state_dict = load(sharded_state_dict, ckpt_dir)
        gpt_model.load_state_dict(state_dict)
    Utils.destroy_model_parallel()


def common_test_parallel_reconfiguration_e2e(initialize_model_fn, tmp_path_dist_ckpt, src_tp_pp, dest_tp_pp,
                                      src_layer_spec_fn, dst_layer_spec_fn):
    """ Test model saving and loading with different TP/PP """
    with TempNamedDir(tmp_path_dist_ckpt / 'test_gpt_model_reconfiguration_model_A') as ckpt_dir_A, \
         TempNamedDir(tmp_path_dist_ckpt / 'test_gpt_model_reconfiguration_model_B') as ckpt_dir_B:
        # Save checkpoint A
        Utils.initialize_model_parallel(*src_tp_pp)
        gpt_model_A = initialize_model_fn(1, src_layer_spec_fn)
        save(gpt_model_A.sharded_state_dict(), ckpt_dir_A)
        regular_state_dict_A = gpt_model_A.state_dict()
        Utils.destroy_model_parallel()

        # Load checkpoint A with different TP/PP and save as checkpoint B
        Utils.initialize_model_parallel(*dest_tp_pp)
        gpt_model_B = initialize_model_fn(2, dst_layer_spec_fn)
        state_dict = load(gpt_model_B.sharded_state_dict(), ckpt_dir_A)
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
        regular_state_dict_A = {k: v for k, v in regular_state_dict_A.items()
                                if not k.endswith('_extra_state')}
        regular_state_dict_B = {k: v for k, v in regular_state_dict_B.items()
                                if not k.endswith('_extra_state')}
        diffs = diff(regular_state_dict_A, regular_state_dict_B)
        assert not any(map(bool, diffs)), diffs
        Utils.destroy_model_parallel()


def common_test_state_dict_comparison(initialize_model_fn, tmp_path_dist_ckpt):
    Utils.initialize_model_parallel(2, 4)
    with TempNamedDir(tmp_path_dist_ckpt / 'test_state_dict_comparison_A') as ckpt_dir_A, \
         TempNamedDir(tmp_path_dist_ckpt / 'test_state_dict_comparison_B') as ckpt_dir_B:
        gpt_model_A = initialize_model_fn(1)
        save(gpt_model_A.sharded_state_dict(), ckpt_dir_A)
        gpt_model_B = initialize_model_fn(2)
        save(gpt_model_B.sharded_state_dict(), ckpt_dir_B)

        state_dict_A = load_plain_tensors(ckpt_dir_A)
        state_dict_A_dup = load_plain_tensors(ckpt_dir_A)
        state_dict_B = load_plain_tensors(ckpt_dir_B)

        # Test that A matches A
        diffs = diff(state_dict_A, state_dict_A_dup)
        assert not any(map(bool, diffs)), diffs

        # Test that A *keys* match B *keys*, but the tensors content is different
        only_left, only_right, mismatch = diff(state_dict_A, state_dict_B)
        assert (not only_left and not only_right), (only_left, only_right)
        assert len(mismatch) == len(state_dict_A), (len(mismatch), (len(state_dict_A)))
    Utils.destroy_model_parallel()


def common_test_vocab_size_padding_change(initialize_model_fn, tmp_path_dist_ckpt, vocab_size_base, src_tp_pp, dest_tp_pp):
    """ Test model loading with different vocab size (caused by TP padding). """
    def get_test_vocab_size(make_divisible_by=128):
        divisor = make_divisible_by * parallel_state.get_tensor_model_parallel_world_size()
        return int(math.ceil(vocab_size_base / divisor)) * divisor

    vocab_size_dependent_keys = {
        'output_layer.weight',
        'output_layer.bias',
        'embedding.word_embeddings.weight',
    }

    with TempNamedDir(tmp_path_dist_ckpt / 'test_vocab_size_padding_change_A') as ckpt_dir_A, \
         TempNamedDir(tmp_path_dist_ckpt / 'test_vocab_size_padding_change_B') as ckpt_dir_B:
        # Save checkpoint A
        Utils.initialize_model_parallel(*src_tp_pp)
        gpt_model_A = initialize_model_fn(1, vocab_size=get_test_vocab_size())
        save(gpt_model_A.sharded_state_dict(), ckpt_dir_A)
        Utils.destroy_model_parallel()

        # Load checkpoint A with different TP/PP and save as checkpoint B
        Utils.initialize_model_parallel(*dest_tp_pp)
        gpt_model_B = initialize_model_fn(2, vocab_size=get_test_vocab_size())
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
                assert torch.all(ten_A[:vocab_size_base] == ten_B[:vocab_size_base]), vocab_layer_key

        # Test other tensors are equal
        diffs = diff(plain_state_dict_A, plain_state_dict_B)
        assert not any(map(bool, diffs)), diffs
        Utils.destroy_model_parallel()
