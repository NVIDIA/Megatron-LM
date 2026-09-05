# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

import megatron.core.models.common.language_module.language_module as language_module
import megatron.core.pipeline_parallel.utils as pipeline_utils
import megatron.training.models.dist_utils as dist_utils
import megatron.training.training as training
from megatron.core.models.common.language_module.language_module import (
    LanguageModule,
    defer_initial_embedding_sync,
)
from megatron.core.tokenizers.utils.build_tokenizer import vocab_size_with_padding
from megatron.training.checkpointing import save_grads
from megatron.training.global_vars import set_args
from megatron.training.training import build_train_valid_test_data_iterators
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils


def mock_train_valid_test_datasets_provider(train_val_test_num_samples):
    return iter([1]), iter([2]), iter([3])


class _LenDataloader:
    """Fake dataloader with __len__ (required by the full_validation path)
    and __iter__ (consumed via cyclic_iter)."""

    def __init__(self, data):
        self._data = list(data)

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)


def mock_multi_valid_full_datasets_provider(train_val_test_num_samples):
    return (iter([1]), [_LenDataloader([2, 2]), _LenDataloader([20, 20, 20])], iter([3]))


def create_test_args():
    # Set dummy values for the args.
    args = SimpleNamespace()
    args.iteration = 0
    args.train_samples = 1
    args.train_iters = 1
    args.eval_interval = 1
    args.eval_iters = 1
    args.global_batch_size = 1
    args.consumed_train_samples = 1
    args.consumed_valid_samples = 1
    args.dataloader_type = "external"
    args.skip_train = False
    args.start_eval_at_iter = None
    args.full_validation = False
    args.multiple_validation_sets = False
    args.perform_rl_step = False
    args.phase_transition_iterations = None

    return args


def _make_mtp_language_module():
    module = LanguageModule.__new__(LanguageModule)
    torch.nn.Module.__init__(module)
    module.config = SimpleNamespace(
        init_model_with_meta_device=False,
        mtp_num_layers=1,
        pipeline_model_parallel_size=2,
        use_mup=False,
    )
    module.share_embeddings_and_output_weights = False
    module.mtp_process = True
    module.pre_process = False
    module.post_process = False
    module.vp_stage = 1
    module.vp_size = 2
    module.pp_group = None
    module.embd_group = object()
    module.embedding = torch.nn.Module()
    module.embedding.word_embeddings = torch.nn.Embedding(4, 2)
    return module


def _model_construction_config():
    return SimpleNamespace(
        bf16=False,
        fp16=False,
        freeze_all_layers=False,
        init_model_with_meta_device=False,
        use_cpu_initialization=True,
        use_megatron_fsdp=False,
        use_torch_fsdp2=True,
        virtual_pipeline_model_parallel_size=3,
    )


def _model_construction_pg_collection():
    pp_group = mock.Mock()
    pp_group.size.return_value = 2
    return SimpleNamespace(cp=object(), dp=object(), gtp_remat=object(), pp=pp_group, tp=object())


def _record_deferred_scope(event_log):
    @contextmanager
    def context():
        event_log.append("enter-defer")
        try:
            yield
        finally:
            event_log.append("exit-defer")

    return context


def _make_sync_recording_model(event_log, vp_stage):
    model = torch.nn.Module()

    def record_sync():
        event_log.append(f"sync-{vp_stage}")

    setattr(model, "sync_initial_embeddings_and_output_layer", record_sync)
    return model


def test_mtp_embedding_sync_is_deferred_until_explicit_sync(monkeypatch):
    module = _make_mtp_language_module()
    shared_weight = mock.Mock()
    shared_weight.data.cuda.return_value = shared_weight.data
    module.shared_embedding_or_output_weight = mock.Mock(return_value=shared_weight)
    module._is_in_embd_group = mock.Mock(return_value=True)
    all_reduce = mock.Mock()
    monkeypatch.setattr(
        language_module, "is_vp_first_stage", lambda vp_stage, _vp_size: vp_stage == 0
    )
    monkeypatch.setattr(language_module, "is_pp_first_stage", lambda _group: False)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)

    with defer_initial_embedding_sync():
        module.setup_embeddings_and_output_layer()

    all_reduce.assert_not_called()
    module.sync_initial_embeddings_and_output_layer()
    all_reduce.assert_called_once_with(shared_weight.data, group=module.embd_group)

    all_reduce.reset_mock()
    module.setup_embeddings_and_output_layer()
    all_reduce.assert_called_once_with(shared_weight.data, group=module.embd_group)


def test_vpp_builders_sync_embeddings_after_all_chunks_are_built(monkeypatch):
    expected_events = [
        "enter-defer",
        "build-0",
        "build-1",
        "build-2",
        "exit-defer",
        "sync-0",
        "sync-1",
        "sync-2",
    ]

    legacy_events = []
    legacy_args = _model_construction_config()
    legacy_pg = _model_construction_pg_collection()
    monkeypatch.setattr(training, "get_args", lambda: legacy_args)
    monkeypatch.setattr(training, "get_pg_size", lambda group: 2 if group is legacy_pg.pp else 1)
    monkeypatch.setattr(training, "get_pg_rank", lambda _group: 1)
    monkeypatch.setattr(training, "is_pp_first_stage", lambda _group: True)
    monkeypatch.setattr(training, "is_pp_last_stage", lambda _group: False)
    monkeypatch.setattr(training, "correct_amax_history_if_needed", lambda _model: None)
    monkeypatch.setattr(training, "has_nvidia_modelopt", False)
    monkeypatch.setattr(
        training, "defer_initial_embedding_sync", _record_deferred_scope(legacy_events)
    )

    def legacy_model_provider(**kwargs):
        legacy_events.append(f"build-{kwargs['vp_stage']}")
        return _make_sync_recording_model(legacy_events, kwargs["vp_stage"])

    legacy_model = training.get_model(
        legacy_model_provider, wrap_with_ddp=False, pg_collection=legacy_pg
    )

    assert len(legacy_model) == 3
    assert legacy_events == expected_events

    builder_events = []
    builder_config = _model_construction_config()
    builder_pg = _model_construction_pg_collection()
    monkeypatch.setattr(pipeline_utils, "is_pp_first_stage", lambda _group: True)
    monkeypatch.setattr(pipeline_utils, "is_pp_last_stage", lambda _group: False)
    monkeypatch.setattr(
        pipeline_utils, "is_vp_first_stage", lambda vp_stage, vp_size: vp_stage == 0
    )
    monkeypatch.setattr(
        pipeline_utils, "is_vp_last_stage", lambda vp_stage, vp_size: vp_stage == vp_size - 1
    )
    monkeypatch.setattr(
        dist_utils, "defer_initial_embedding_sync", _record_deferred_scope(builder_events)
    )
    monkeypatch.setattr(
        dist_utils,
        "prepare_existing_model_chunks_for_distributed_training",
        lambda model, *_args, **_kwargs: model,
    )

    def build_model(_pg_collection, *, vp_stage, **_kwargs):
        builder_events.append(f"build-{vp_stage}")
        return _make_sync_recording_model(builder_events, vp_stage)

    builder_model = dist_utils.unimodal_build_distributed_models(
        build_model, builder_config, builder_pg, wrap_with_ddp=False
    )

    assert len(builder_model) == 3
    assert builder_events == expected_events


class TestTraining:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        args = create_test_args()
        set_args(args)

    def test_build_train_valid_test_data_iterators(self):
        train_iter, valid_iter, test_iter = build_train_valid_test_data_iterators(
            mock_train_valid_test_datasets_provider
        )
        train_data = next(train_iter)
        valid_data = next(valid_iter)
        test_data = next(test_iter)
        assert (train_data, valid_data, test_data) == (1, 2, 3)

    def test_build_train_valid_test_data_iterators_multi_full_validation(self):
        """multiple_validation_sets + full_validation builds a list of iterators
        (one per validation set) and sets args.eval_iters to the per-loader
        lengths MAX-reduced across DP ranks."""
        args = create_test_args()
        args.multiple_validation_sets = True
        args.full_validation = True
        set_args(args)
        _, valid_iters, _ = build_train_valid_test_data_iterators(
            mock_multi_valid_full_datasets_provider
        )
        assert isinstance(valid_iters, list)
        assert len(valid_iters) == 2
        assert next(valid_iters[0]) == 2
        assert next(valid_iters[1]) == 20
        # data_parallel_size=1, so MAX across DP ranks equals the local lengths
        assert args.eval_iters == [2, 3]

    def test_closed_formula_vocab_size_with_padding(self):
        def old_round_impl(after, multiple):
            while (after % multiple) != 0:
                after += 1
            return after

        args = SimpleNamespace()
        args.rank = 0
        args.tensor_model_parallel_size = 1

        for vocab in range(1, 600000, 1000):
            for mult in [1, 17, 32, 64, 128]:
                args.make_vocab_size_divisible_by = mult
                assert old_round_impl(vocab, mult) == vocab_size_with_padding(vocab, args, False), (
                    vocab,
                    mult,
                )

        for vocab in range(1, 10_000, 500):
            for mult in range(1, 1024 + 1):
                args.make_vocab_size_divisible_by = mult
                assert old_round_impl(vocab, mult) == vocab_size_with_padding(vocab, args, False), (
                    vocab,
                    mult,
                )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()


class TestGetModelBucketSizingPgCollection:
    """The DDP-bucket-sizing path in get_model must read world size / rank from the
    explicitly passed pg_collection (pg_collection.dp_cp / pg_collection.pp) rather
    than the mpu globals. With an explicit pg_collection the mpu globals must not be
    consulted at all."""

    def test_bucket_sizing_uses_explicit_pg_collection(self, monkeypatch):
        import megatron.training.training as training

        # Sentinel groups whose size()/rank() identify which group was read.
        class _Group:
            def __init__(self, size, rank):
                self._size = size
                self._rank = rank

            def size(self):
                return self._size

            def rank(self):
                return self._rank

        pg_collection = SimpleNamespace(dp_cp=_Group(size=7, rank=0), pp=_Group(size=4, rank=3))

        # The mpu globals replaced on the bucket-sizing path must never be called
        # when an explicit pg_collection is supplied.
        def _boom(*args, **kwargs):
            raise AssertionError("mpu global consulted on explicit pg_collection path")

        monkeypatch.setattr(training.mpu, "get_data_parallel_world_size", _boom)
        monkeypatch.setattr(training.mpu, "get_pipeline_model_parallel_rank", _boom)

        # get_pg_size/get_pg_rank return 1/0 unless torch.distributed is initialized,
        # so make them read directly off the sentinel groups for this host-only test.
        monkeypatch.setattr(training, "get_pg_size", lambda group: group.size())
        monkeypatch.setattr(training, "get_pg_rank", lambda group: group.rank())

        # Mirror the exact bucket-sizing expressions from get_model.
        bucket_size = max(40000000, 1000000 * training.get_pg_size(pg_collection.dp_cp))
        pp_rank = training.get_pg_rank(pg_collection.pp)

        # dp_cp size 7 -> 7_000_000 < 40_000_000, so the floor wins (default behavior).
        assert bucket_size == 40000000
        # pp rank is driven by pg_collection.pp, not the mpu global.
        assert pp_rank == 3


class TestSaveGrads:
    """Tests for the save_grads function."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_save_grads(self, tmp_path_dist_ckpt):
        """Test that save_grads creates the correct directory structure and saves
        state_dict correctly.

        With TP=1, PP=1 on 8 GPUs, we have 8 DP ranks. Only the rank with
        expert_data_parallel_rank==0 should save. All ranks verify the result.
        """
        save_dir = str(tmp_path_dist_ckpt / "test_save_grads")

        with TempNamedDir(save_dir, sync=True) as save_dir:
            # Create a mock state_dict with gradients (use deterministic values for reproducibility).
            state_dict = defaultdict(dict)
            state_dict["model_chunk0"]["layer.weight"] = torch.arange(16).reshape(4, 4).float()
            state_dict["model_chunk0"]["layer.bias"] = torch.arange(4).float()

            iteration = 100
            grad_label = "wgrads"

            # All ranks call save_grads, but only expert_data_parallel_rank==0 actually saves.
            save_grads(save_dir, dict(state_dict), iteration, grad_label)

            # Synchronize before checking results since only rank 0 saves.
            torch.distributed.barrier()

            # All ranks verify the file was created by rank 0.
            expected_dir = Path(save_dir) / grad_label / f"iter_{iteration:07d}"
            assert expected_dir.exists(), f"Expected directory {expected_dir} to exist"

            expected_file = expected_dir / "mp_rank_00.pth"
            assert expected_file.exists(), f"Expected file {expected_file} to exist"

            # Verify saved content.
            loaded = torch.load(expected_file)
            assert "model_chunk0" in loaded
            assert "layer.weight" in loaded["model_chunk0"]
            assert "layer.bias" in loaded["model_chunk0"]
            assert torch.equal(
                loaded["model_chunk0"]["layer.weight"], state_dict["model_chunk0"]["layer.weight"]
            )
            assert torch.equal(
                loaded["model_chunk0"]["layer.bias"], state_dict["model_chunk0"]["layer.bias"]
            )
