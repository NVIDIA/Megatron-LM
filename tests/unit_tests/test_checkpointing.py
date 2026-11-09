# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Note: --ckpt-format torch_dist has tests in tests/unit_tests/dist_checkpointing.
import contextlib
import os
from dataclasses import fields, replace
from types import SimpleNamespace
from typing import Optional
from unittest import mock

import pytest
import torch
import torch.distributed.checkpoint
import yaml

from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed.fsdp.mcore_fsdp_adapter import FullyShardedDataParallel
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.num_microbatches_calculator import (
    init_num_microbatches_calculator,
    unset_num_microbatches_calculator,
)
from megatron.core.parallel_state import get_data_parallel_world_size
from megatron.core.rerun_state_machine import RerunMode, RerunState, RerunStateMachine
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import MegatronModule, WideResidualConfig
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_torch_min_version
from megatron.training import checkpointing
from megatron.training.checkpointing import (
    CheckpointType,
    _build_sharded_state_dict_metadata,
    _load_base_checkpoint,
    _maybe_setup_gpt_to_hybrid_load,
    _read_checkpoint_run_config,
    check_checkpoint_args,
    get_checkpoint_tracker_filename,
    get_checkpoint_version,
    load_args_from_checkpoint,
    load_checkpoint,
    maybe_save_dataloader_state,
    read_metadata,
    save_checkpoint,
    set_checkpoint_version,
)
from megatron.training.config import (
    CheckpointConfig,
    InferenceConfigContainer,
    InferenceSetupConfig,
    ProfilingConfig,
)
from megatron.training.global_vars import get_args, get_run_config, set_args
from megatron.training.models import (
    BertModelConfig,
    GPTModelConfig,
    LLaVAModelConfig,
    MimoBuildConfig,
    T5ModelConfig,
)
from megatron.training.state import TrainState
from megatron.training.utils.checkpoint_utils import (
    get_checkpoint_run_config_filename,
    get_checkpoint_train_state_filename,
)
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils, sync_checkpoint_config

pytestmark = pytest.mark.usefixtures("run_config")


@pytest.mark.parametrize("ckpt_args", [None, SimpleNamespace()])
def test_model_only_checkpoint_does_not_trigger_gpt_hybrid_interop(ckpt_args):
    """Missing checkpoint args do not identify a model-only checkpoint as GPT."""

    args = SimpleNamespace(hybrid_layer_pattern="M*E")
    assert _maybe_setup_gpt_to_hybrid_load(
        args, ckpt_args, [object()], load_optim=True, saved_save_optim=True
    ) == (None, False)


@pytest.mark.parametrize("auto_detect", [False, True])
def test_inference_shards_reject_legacy_checkpoint(create_ckpt_load_args, monkeypatch, auto_detect):
    from megatron.training import checkpointing

    args = create_ckpt_load_args
    args.load = "unused"
    args.ckpt_format = "torch_dist" if auto_detect else "torch"
    args.auto_detect_ckpt_format = auto_detect
    args.inference_shards = "tp=1,role=prefill+tp=1,role=decode"
    monkeypatch.setattr(checkpointing, "get_args", lambda: args)
    monkeypatch.setattr(
        checkpointing,
        "_load_base_checkpoint",
        mock.Mock(return_value=({}, "unused", False, CheckpointType.LEGACY)),
    )
    with pytest.raises(ValueError, match="incompatible with legacy torch checkpoints"):
        sync_checkpoint_config(args)
        load_checkpoint([], None, None)


class MockModel(MegatronModule):
    """Dummy megatron model."""

    def __init__(self, config):
        super().__init__(config=config)
        self.l = torch.nn.Linear(1, 2)
        torch.nn.init.ones_(self.l.weight)
        torch.nn.init.zeros_(self.l.bias)
        self._called_metadata = []

    def sharded_state_dict(self, *args, metadata: Optional[dict] = None, **kwargs):
        self._called_metadata.append(metadata)
        return self.state_dict()


class MockState:
    def __init__(self, state_dict):
        self._state_dict = state_dict
        self.is_stub_optimizer = False
        self._called_metadata = []

        # Optimizers are expected to have this attribute for checkpointing.
        self.param_groups = []

    def state_dict(self, is_loading=False):
        return self._state_dict

    def load_state_dict(self, state_dict):
        self._state_dict = state_dict

    def save_parameter_state(self, *args, **kwargs):
        pass

    def load_parameter_state(self, *args, **kwargs):
        pass

    def sharded_state_dict(self, *args, metadata: Optional[dict] = None, **kwargs):
        self._called_metadata.append(metadata)
        return self.state_dict()


@pytest.mark.parametrize(
    ("runtime_mode", "checkpoint_args", "expected_mode"),
    [
        ("indices", SimpleNamespace(moe_hybridep_routing_map_mode="bool"), "indices"),
        (None, SimpleNamespace(moe_hybridep_routing_map_mode="bool"), "bool"),
        (None, SimpleNamespace(), None),
    ],
)
def test_load_args_preserves_runtime_hybridep_routing_map_mode(
    runtime_mode, checkpoint_args, expected_mode
):
    args = SimpleNamespace(
        load="checkpoint",
        iteration=0,
        moe_hybridep_routing_map_mode=runtime_mode,
        use_tokenizer_model_from_checkpoint_args=False,
        use_mp_args_from_checkpoint_args=False,
    )
    state_dict = {"args": checkpoint_args, "iteration": 12}

    with mock.patch(
        "megatron.training.checkpointing._load_base_checkpoint",
        return_value=(state_dict, "checkpoint", False, CheckpointType.LEGACY),
    ):
        restored_args, _ = load_args_from_checkpoint(args)

    assert restored_args.moe_hybridep_routing_map_mode == expected_mode


def test_maybe_save_dataloader_state_uses_explicit_process_groups(tmp_path):
    """Dataloader checkpoints use the supplied module groups and canonical model-parallel path.

    The data-parallel group here is the full data-distribution group (dp x gtp_remat), which is
    the axis the dataloader shards on; the replicate group would give gtp_remat peers one name.
    """
    groups = {
        "tp": SimpleNamespace(rank=0, size=2),
        "pp": SimpleNamespace(rank=0, size=2),
        "dp": SimpleNamespace(rank=3, size=4),
    }
    barriers = []
    saved = []
    iterator = SimpleNamespace(
        iterable=SimpleNamespace(save_state=lambda: {"global_sequence_id": 16})
    )

    with (
        mock.patch(
            "megatron.training.checkpointing.get_pg_rank", side_effect=lambda group: group.rank
        ),
        mock.patch(
            "megatron.training.checkpointing.get_pg_size", side_effect=lambda group: group.size
        ),
        mock.patch(
            "megatron.training.checkpointing.mpu.get_context_parallel_group", return_value=None
        ),
        mock.patch(
            "megatron.training.checkpointing.torch.distributed.barrier",
            side_effect=lambda group: barriers.append(group),
        ),
        mock.patch(
            "megatron.training.checkpointing.torch.save",
            side_effect=lambda state, path: saved.append((state, path)),
        ),
    ):
        maybe_save_dataloader_state(
            iterator,
            2,
            tmp_path,
            tp_group=groups["tp"],
            pp_group=groups["pp"],
            dp_group=groups["dp"],
        )

    assert barriers == [groups["dp"], groups["dp"]]
    assert saved[0][0] == {"dataloader_state_dict": {"global_sequence_id": 16}}
    assert saved[0][1] == str(
        tmp_path / "iter_0000002" / "mp_rank_00_000" / "train_dataloader_dprank003.pt"
    )


@pytest.mark.parametrize("dp_rank", [0, 3])
@pytest.mark.parametrize("cp_rank", [None, 0, 1])
def test_maybe_save_dataloader_state_rank_independent_writes_one_file(tmp_path, dp_rank, cp_rank):
    """Rank-independent state is written only by data rank 0 on the first CP rank."""
    groups = {
        "tp": SimpleNamespace(rank=0, size=1),
        "pp": SimpleNamespace(rank=0, size=1),
        "dp": SimpleNamespace(rank=dp_rank, size=4),
    }
    barriers = []
    saved = []
    save_state_calls = []
    cp_group = None if cp_rank is None else SimpleNamespace(rank=cp_rank, size=2)

    def save_state():
        save_state_calls.append(True)
        return {"global_sequence_id": 16}

    iterator = SimpleNamespace(
        iterable=SimpleNamespace(save_state=save_state, is_save_state_rank_independent=True)
    )

    with (
        mock.patch(
            "megatron.training.checkpointing.get_pg_rank", side_effect=lambda group: group.rank
        ),
        mock.patch(
            "megatron.training.checkpointing.get_pg_size", side_effect=lambda group: group.size
        ),
        mock.patch(
            "megatron.training.checkpointing.mpu.get_context_parallel_group", return_value=None
        ),
        mock.patch(
            "megatron.training.checkpointing.torch.distributed.barrier",
            side_effect=lambda group: barriers.append(group),
        ),
        mock.patch(
            "megatron.training.checkpointing.torch.save",
            side_effect=lambda state, path: saved.append((state, path)),
        ),
    ):
        maybe_save_dataloader_state(
            iterator,
            2,
            tmp_path,
            tp_group=groups["tp"],
            pp_group=groups["pp"],
            dp_group=groups["dp"],
            cp_group=cp_group,
        )

    if cp_rank == 1:
        assert barriers == []
        assert saved == []
        assert save_state_calls == []
        return

    # All DP ranks on the first CP rank join the barriers; only DP0 builds and writes state.
    assert barriers == [groups["dp"], groups["dp"]]
    if dp_rank == 0:
        assert saved[0][1] == str(
            tmp_path / "iter_0000002" / "mp_rank_00" / "train_dataloader_dprank000.pt"
        )
        assert save_state_calls == [True]
    else:
        assert saved == []
        assert save_state_calls == []


def test_maybe_save_dataloader_state_skips_empty_state_after_barriers(tmp_path):
    """Ranks without dataloader state participate in barriers but do not write a file."""
    group = SimpleNamespace(rank=0, size=1)
    iterator = SimpleNamespace(iterable=SimpleNamespace(save_state=lambda: None))
    barriers = []

    with (
        mock.patch(
            "megatron.training.checkpointing.get_pg_rank",
            side_effect=lambda process_group: process_group.rank,
        ),
        mock.patch(
            "megatron.training.checkpointing.get_pg_size",
            side_effect=lambda process_group: process_group.size,
        ),
        mock.patch(
            "megatron.training.checkpointing.mpu.get_context_parallel_group", return_value=None
        ),
        mock.patch(
            "megatron.training.checkpointing.torch.distributed.barrier",
            side_effect=lambda group: barriers.append(group),
        ),
        mock.patch("megatron.training.checkpointing.torch.save") as save,
    ):
        maybe_save_dataloader_state(
            iterator, 2, tmp_path, tp_group=group, pp_group=group, dp_group=group
        )

    assert barriers == [group, group]
    save.assert_not_called()


@pytest.mark.parametrize(
    "cp_rank,global_cp_rank,should_save",
    [
        (None, None, True),
        (None, 0, True),
        (None, 1, False),
        (0, None, True),
        (1, None, False),
        (0, 1, True),
        (1, 0, False),
    ],
)
def test_maybe_save_dataloader_state_context_parallel_groups(
    tmp_path, cp_rank, global_cp_rank, should_save
):
    """Explicit CP groups take precedence; an absent global group must not assert."""
    model_group = SimpleNamespace(rank=0, size=1)
    dp_group = SimpleNamespace(rank=1, size=2)
    cp_group = None if cp_rank is None else SimpleNamespace(rank=cp_rank, size=2)
    global_cp_group = (
        None if global_cp_rank is None else SimpleNamespace(rank=global_cp_rank, size=2)
    )
    iterator = SimpleNamespace(
        iterable=SimpleNamespace(save_state=mock.Mock(return_value={"position": 16}))
    )

    def get_context_parallel_group(check_initialized=True):
        if check_initialized and global_cp_group is None:
            raise AssertionError("context parallel group is not initialized")
        return global_cp_group

    with (
        mock.patch(
            "megatron.training.checkpointing.get_pg_rank",
            side_effect=lambda process_group: process_group.rank,
        ),
        mock.patch(
            "megatron.training.checkpointing.get_pg_size",
            side_effect=lambda process_group: process_group.size,
        ),
        mock.patch(
            "megatron.training.checkpointing.mpu.get_context_parallel_group",
            side_effect=get_context_parallel_group,
        ) as get_cp_group,
        mock.patch(
            "megatron.training.checkpointing.mpu.get_context_parallel_rank",
            side_effect=AssertionError("Must not require the global CP rank"),
        ),
        mock.patch("megatron.training.checkpointing.torch.distributed.barrier") as barrier,
        mock.patch("megatron.training.checkpointing.torch.save") as save,
    ):
        maybe_save_dataloader_state(
            iterator,
            2,
            tmp_path,
            tp_group=model_group,
            pp_group=model_group,
            dp_group=dp_group,
            cp_group=cp_group,
        )

    if cp_group is None:
        get_cp_group.assert_called_once_with(check_initialized=False)
    else:
        get_cp_group.assert_not_called()

    if should_save:
        iterator.iterable.save_state.assert_called_once_with()
        assert barrier.call_args_list == [mock.call(group=dp_group), mock.call(group=dp_group)]
        save.assert_called_once_with(
            {"dataloader_state_dict": {"position": 16}},
            str(tmp_path / "iter_0000002" / "mp_rank_00" / "train_dataloader_dprank001.pt"),
        )
    else:
        iterator.iterable.save_state.assert_not_called()
        barrier.assert_not_called()
        save.assert_not_called()


@pytest.mark.parametrize("save_path", [None, ""])
def test_maybe_save_dataloader_state_disabled_without_global_groups(save_path):
    """Text-only training without dataloader-state saving must not inspect CP groups."""
    with mock.patch(
        "megatron.training.checkpointing.mpu.get_context_parallel_group",
        side_effect=AssertionError("Must not inspect CP when dataloader saving is disabled"),
    ) as get_cp_group:
        maybe_save_dataloader_state(iter([1]), 2, save_path)
    get_cp_group.assert_not_called()


class MockOptParamScheduler(MockState):
    def __init__(self, state_dict):
        super().__init__(state_dict)
        self.num_steps = state_dict.get("num_steps", 0)
        self.step_calls = []

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.num_steps = state_dict.get("num_steps", self.num_steps)

    def step(self, increment=1):
        self.step_calls.append(increment)


class MockOptimizer(MockState):
    def state_dict(self, is_loading=False):
        state_dict = super().state_dict(is_loading=is_loading).copy()
        state_dict["param_groups"] = [param_group.copy() for param_group in self.param_groups]
        return state_dict

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.param_groups = [param_group.copy() for param_group in state_dict["param_groups"]]


@pytest.mark.parametrize(
    ("checkpoint_args", "configured_num_householder", "expected_num_householder"),
    [(SimpleNamespace(gdp_num_householder=5), 3, 5), (SimpleNamespace(), 5, 3)],
)
def test_load_args_restores_gdp_num_householder_from_checkpoint(
    checkpoint_args, configured_num_householder, expected_num_householder
):
    args = SimpleNamespace(
        load="checkpoint",
        iteration=0,
        gdp_num_householder=configured_num_householder,
        use_tokenizer_model_from_checkpoint_args=False,
        use_mp_args_from_checkpoint_args=False,
    )
    state_dict = {"args": checkpoint_args, "iteration": 12}

    with mock.patch(
        "megatron.training.checkpointing._load_base_checkpoint",
        return_value=(state_dict, "checkpoint", False, CheckpointType.LEGACY),
    ):
        restored_args, _ = load_args_from_checkpoint(args)

    assert restored_args.gdp_num_householder == expected_num_householder


def test_load_args_restores_quantile_balancing_from_checkpoint():
    """QB mode and histogram shape are reconstructed before the router is built."""
    checkpoint_args = SimpleNamespace(
        moe_router_load_balancing_type="quantile_balancing",
        moe_aux_loss_coeff=0.0,
        moe_router_quantile_balancing_estimation_scope="global_batch",
        moe_router_qb_num_bins=257,
    )
    args = SimpleNamespace(
        load="checkpoint",
        iteration=0,
        moe_router_load_balancing_type="aux_loss",
        moe_aux_loss_coeff=0.01,
        moe_router_quantile_balancing_estimation_scope="micro_batch",
        moe_router_qb_num_bins=64,
        use_tokenizer_model_from_checkpoint_args=False,
        use_mp_args_from_checkpoint_args=False,
    )
    state_dict = {"args": checkpoint_args, "iteration": 12}

    with mock.patch(
        "megatron.training.checkpointing._load_base_checkpoint",
        return_value=(state_dict, "checkpoint", False, CheckpointType.LEGACY),
    ):
        restored_args, _ = load_args_from_checkpoint(args)

    assert restored_args.moe_router_load_balancing_type == "quantile_balancing"
    assert restored_args.moe_aux_loss_coeff == 0.0
    assert restored_args.moe_router_quantile_balancing_estimation_scope == "global_batch"
    assert restored_args.moe_router_qb_num_bins == 257


@pytest.mark.parametrize(
    ("checkpoint_args", "configured_scale", "expected_scale"),
    [
        (SimpleNamespace(activation_func_tanh_clamp_scale=16.0), None, 16.0),
        (SimpleNamespace(activation_func_tanh_clamp_scale=16.0), 1.0, 16.0),
        (SimpleNamespace(), 4.0, 4.0),
    ],
    ids=["restored", "overrides-command-line", "absent-keeps-command-line"],
)
def test_load_args_restores_activation_func_tanh_clamp_scale_from_checkpoint(
    checkpoint_args, configured_scale, expected_scale
):
    """The checkpoint's clamp scale overrides the command line (force=True), like squared_relu."""
    args = SimpleNamespace(
        load="checkpoint",
        iteration=0,
        activation_func_tanh_clamp_scale=configured_scale,
        use_tokenizer_model_from_checkpoint_args=False,
        use_mp_args_from_checkpoint_args=False,
    )
    state_dict = {"args": checkpoint_args, "iteration": 12}

    with mock.patch(
        "megatron.training.checkpointing._load_base_checkpoint",
        return_value=(state_dict, "checkpoint", False, CheckpointType.LEGACY),
    ):
        restored_args, _ = load_args_from_checkpoint(args)

    assert restored_args.activation_func_tanh_clamp_scale == expected_scale


def test_load_args_restores_wide_residual_config_from_checkpoint():
    """Checkpoint arguments should reconstruct the flat wide-residual CLI settings."""
    checkpoint_args = SimpleNamespace(
        wide_residual_num_streams=3,
        wide_residual_streamwise_sigmoid_init_scale=0.02,
        wide_residual_learned_retention=True,
        wide_residual_retention_init=0.998,
        wide_residual_retention_max_forget=0.2,
    )
    args = SimpleNamespace(
        load='checkpoint',
        iteration=0,
        wide_residual_num_streams=None,
        wide_residual_streamwise_sigmoid_init_scale=0.01,
        wide_residual_learned_retention=False,
        wide_residual_retention_init=0.999,
        wide_residual_retention_max_forget=0.10,
        use_tokenizer_model_from_checkpoint_args=False,
        use_mp_args_from_checkpoint_args=False,
    )
    state_dict = {'args': checkpoint_args, 'iteration': 12}

    with mock.patch(
        'megatron.training.checkpointing._load_base_checkpoint',
        return_value=(state_dict, 'checkpoint', False, CheckpointType.LEGACY),
    ):
        restored_args, _ = load_args_from_checkpoint(args)

    assert restored_args.wide_residual_num_streams == 3
    assert restored_args.wide_residual_streamwise_sigmoid_init_scale == 0.02
    assert restored_args.wide_residual_learned_retention
    assert restored_args.wide_residual_retention_init == 0.998
    assert restored_args.wide_residual_retention_max_forget == 0.2


def test_check_checkpoint_args_rejects_wide_residual_mismatch(run_config):
    """Resuming must reject a checkpoint with different wide-residual geometry."""
    current_args = SimpleNamespace(
        num_layers=2,
        hidden_size=128,
        num_attention_heads=4,
        add_position_embedding=False,
        vocab_file=None,
        data_parallel_random_init=False,
        phase_transition_iterations=None,
        use_dist_ckpt=True,
        wide_residual_num_streams=3,
        wide_residual_streamwise_sigmoid_init_scale=0.01,
        wide_residual_learned_retention=False,
        wide_residual_retention_init=0.999,
        wide_residual_retention_max_forget=0.10,
    )
    checkpoint_args = SimpleNamespace(**vars(current_args))
    checkpoint_args.wide_residual_num_streams = 4
    run_config.model = GPTModelConfig(
        transformer=TransformerConfig(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=4,
            wide_residual=WideResidualConfig(num_streams=3),
        ),
        seq_length=16,
        position_embedding_type='rope',
    )

    with (
        mock.patch('megatron.training.checkpointing.get_args', return_value=current_args),
        mock.patch('megatron.training.checkpointing.get_checkpoint_version', return_value=3.0),
        pytest.raises(AssertionError, match='wide_residual_num_streams value from checkpoint'),
    ):
        check_checkpoint_args(checkpoint_args)


@pytest.mark.parametrize("model_cls", [GPTModelConfig, BertModelConfig, MimoBuildConfig])
def test_checkpoint_model_checks_use_config_not_stale_args(run_config, model_cls):
    kwargs = {"position_embedding_type": "rope"} if model_cls is GPTModelConfig else {}
    run_config.model = model_cls(
        transformer=TransformerConfig(num_layers=2, hidden_size=32, num_attention_heads=4),
        seq_length=64,
        vocab_size=128,
        **kwargs,
    )
    args = SimpleNamespace(
        vocab_file=None,
        use_dist_ckpt=True,
        phase_transition_iterations=None,
        add_position_embedding=True,
    )
    saved = run_config.to_dict()
    stale = SimpleNamespace(num_layers=99, hidden_size=99, num_attention_heads=99)
    with (
        mock.patch('megatron.training.checkpointing.get_args', return_value=args),
        mock.patch('megatron.training.checkpointing.get_checkpoint_version', return_value=3.0),
    ):
        check_checkpoint_args(stale, checkpoint_config=saved)
        saved['model']['transformer']['hidden_size'] = 64
        with pytest.raises(AssertionError, match='hidden_size value from checkpoint'):
            check_checkpoint_args(stale, checkpoint_config=saved)


@pytest.mark.parametrize("saved_config", [False, True])
def test_checkpoint_allows_rope_to_yarn(run_config, saved_config):
    args = SimpleNamespace(
        num_layers=2,
        hidden_size=32,
        num_attention_heads=4,
        add_position_embedding=True,
        vocab_file=None,
        use_dist_ckpt=True,
        phase_transition_iterations=None,
    )
    checkpoint_args = SimpleNamespace(**vars(args), position_embedding_type="rope")
    run_config.model.position_embedding_type = "rope"
    saved = run_config.to_dict() if saved_config else None
    run_config.model.position_embedding_type = "yarn"
    with (
        mock.patch('megatron.training.checkpointing.get_args', return_value=args),
        mock.patch('megatron.training.checkpointing.get_checkpoint_version', return_value=3.0),
    ):
        check_checkpoint_args(checkpoint_args, checkpoint_config=saved)
        checkpoint_args.add_position_embedding = False
        with pytest.raises(AssertionError, match="add_position_embedding"):
            check_checkpoint_args(checkpoint_args, checkpoint_config=saved)


@pytest.mark.parametrize("saved_config", [False, True])
def test_checkpoint_padded_vocabulary_uses_tokenizer_config(run_config, saved_config):
    run_config.checkpoint.ckpt_format = "torch"
    run_config.tokenizer.vocab_file = "vocab"
    run_config.tokenizer.tokenizer_type = "NullTokenizer"
    run_config.tokenizer.padded_vocab_size = 512
    run_config.model.vocab_size = 640
    run_config.model.should_pad_vocab = False
    args = SimpleNamespace(
        vocab_file="vocab",
        use_dist_ckpt=False,
        phase_transition_iterations=None,
        tokenizer_type="NullTokenizer",
        padded_vocab_size=999,
        add_position_embedding=True,
    )
    checkpoint_args = SimpleNamespace(
        num_layers=2,
        hidden_size=32,
        num_attention_heads=4,
        max_position_embeddings=16,
        make_vocab_size_divisible_by=128,
        tokenizer_type="NullTokenizer",
        padded_vocab_size=512,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
    )
    saved = run_config.to_dict() if saved_config else None
    with (
        mock.patch('megatron.training.checkpointing.get_args', return_value=args),
        mock.patch('megatron.training.checkpointing.get_checkpoint_version', return_value=3.0),
    ):
        check_checkpoint_args(checkpoint_args, checkpoint_config=saved)
        run_config.tokenizer.padded_vocab_size = 768
        with pytest.raises(AssertionError, match='padded_vocab_size value from checkpoint'):
            check_checkpoint_args(checkpoint_args, checkpoint_config=saved)
        check_checkpoint_args(
            checkpoint_args, skip_args={'padded_vocab_size'}, checkpoint_config=saved
        )


@pytest.mark.parametrize("restore_parallelism", [False, True])
def test_checkpoint_args_restore_preserves_mlm_policy(restore_parallelism):
    """CLI bootstrap retains MLM precedence; saved config does not replace launch settings."""
    args = SimpleNamespace(
        load='checkpoint',
        hidden_size=64,
        num_layers=None,
        seq_length=None,
        attention_dropout=0.3,
        tensor_model_parallel_size=2,
        recompute_granularity='selective',
        tp_comm_overlap=False,
        bf16=True,
        use_tokenizer_model_from_checkpoint_args=False,
        use_mp_args_from_checkpoint_args=restore_parallelism,
    )
    checkpoint_args = SimpleNamespace(
        hidden_size=32,
        num_layers=2,
        seq_length=128,
        spec=['custom_model', 'layer_spec'],
        attention_dropout=0.1,
        tensor_model_parallel_size=4,
        recompute_granularity='full',
        tp_comm_overlap=True,
        bf16=False,
    )
    # Native configuration is not implicitly merged into the CLI bootstrap path.
    saved_config = {'model': {'transformer': {'num_layers': 99, 'attention_dropout': 0.9}}}
    state = {'args': checkpoint_args, 'checkpoint_version': 3.0}
    with (
        mock.patch(
            'megatron.training.checkpointing._load_base_checkpoint',
            return_value=(state, 'checkpoint', False, CheckpointType.LEGACY),
        ),
        mock.patch(
            'megatron.training.checkpointing._read_checkpoint_run_config', return_value=saved_config
        ),
    ):
        load_args_from_checkpoint(args)
    assert args.hidden_size == 64  # Non-None CLI inputs retain precedence.
    assert args.num_layers == 2
    assert args.seq_length == 128
    assert args.spec == checkpoint_args.spec  # Keep the original CLI import path.
    assert args.attention_dropout == 0.1  # Existing MLM force=True behavior.
    assert args.tensor_model_parallel_size == (4 if restore_parallelism else 2)
    assert args.recompute_granularity == 'selective'
    assert args.tp_comm_overlap is False
    assert args.bf16 is True


@pytest.mark.parametrize(
    "section,name,current,other",
    [
        ("train", "global_batch_size", 8, 16),
        ("rng", "data_parallel_random_init", True, False),
        ("tokenizer", "tokenizer_type", "GPT2BPETokenizer", "SentencePieceTokenizer"),
    ],
)
@pytest.mark.parametrize(
    "saved_source",
    ["legacy", "missing_section", "null_section", "missing_field", "config", "config_only"],
)
def test_check_checkpoint_args_uses_config_owner(
    monkeypatch, run_config, section, name, current, other, saved_source
):
    """Config precedence, old checkpoint fallback and skips share one comparison path."""
    run_config.model = GPTModelConfig(
        transformer=TransformerConfig(num_layers=2, hidden_size=128, num_attention_heads=4),
        seq_length=16,
        vocab_size=128,
        position_embedding_type="rope",
    )
    args = SimpleNamespace(
        num_layers=2,
        hidden_size=128,
        num_attention_heads=4,
        add_position_embedding=False,
        vocab_file=None,
        phase_transition_iterations=[100] if section == "train" else None,
        use_dist_ckpt=True,
        **{name: other},
    )
    owner = getattr(run_config, section)
    if section == "tokenizer":
        run_config.tokenizer.vocab_file = "vocab.json"
        args.max_position_embeddings = run_config.model.seq_length
        args.make_vocab_size_divisible_by = run_config.model.make_vocab_size_divisible_by
    assert hasattr(owner, name)
    setattr(owner, name, current)
    checkpoint_args = SimpleNamespace(**vars(args))
    checkpoint_config = None
    if saved_source != "legacy":
        checkpoint_config = {} if saved_source == "missing_section" else {section: {}}
        if saved_source == "null_section":
            checkpoint_config[section] = None
    if saved_source in ("config", "config_only"):
        saved_owner = checkpoint_config[section]
        saved_owner[name] = current
        if saved_source == "config_only":
            delattr(checkpoint_args, name)
    else:
        saved_owner = vars(checkpoint_args)
        saved_owner[name] = current
    monkeypatch.setattr("megatron.training.checkpointing.get_args", lambda: args)
    monkeypatch.setattr("megatron.training.checkpointing.get_checkpoint_version", lambda: 3.0)

    check_checkpoint_args(checkpoint_args, checkpoint_config=checkpoint_config)
    for saved_value in (other, None):
        saved_owner[name] = saved_value
        with pytest.raises(AssertionError, match=f"{name} value from checkpoint"):
            check_checkpoint_args(checkpoint_args, checkpoint_config=checkpoint_config)
        check_checkpoint_args(
            checkpoint_args, checkpoint_config=checkpoint_config, skip_args={name}
        )


def create_checkpoint(load_path, ckpt_format):
    """Setup a dummy checkpoint directory."""
    iteration = 123
    ckpt_dir = load_path / "iter_{:07d}".format(iteration)
    tracker_path = get_checkpoint_tracker_filename(load_path)
    with open(tracker_path, "w") as f:
        f.write(str(iteration))

    state_dict = {"args": "dummy", "iteration": iteration}

    if ckpt_format == "torch":
        # Torch checkpoints use a specific directory structure.
        pt_dir = ckpt_dir / "mp_rank_00"
        pt_dir.mkdir(parents=True)
        torch.save(state_dict, pt_dir / "model_optim_rng.pt")
    elif ckpt_format == "torch_dcp" and is_torch_min_version("2.4.0"):
        torch.distributed.checkpoint.save(state_dict, checkpoint_id=ckpt_dir)


@pytest.fixture
def create_args(run_config):
    """Setup dummy args."""
    args = SimpleNamespace()
    args.finetune = False
    args.non_persistent_global_ckpt_dir = None
    args.non_persistent_ckpt_type = None
    args.non_persistent_save_interval = None
    args.exit_on_missing_checkpoint = True
    args.async_save = False
    args.async_strategy = "nvrx"
    args.data_parallel_random_init = False
    args.no_save_optim = False
    args.no_save_rng = False
    args.no_load_optim = False
    args.no_load_rng = False
    args.log_progress = False
    args.ckpt_fully_parallel_save = False
    args.dist_ckpt_optim_fully_reshardable = False
    args.distrib_optim_fully_reshardable_mem_efficient = False
    args.auto_detect_ckpt_format = False
    args.ckpt_convert_update_legacy_dist_opt_format = False
    args.ckpt_step = None
    args.override_opt_param_scheduler = False
    args.swiglu = True
    args.num_experts = 1
    args.verify_integrity = False
    run_config.model = GPTModelConfig(
        transformer=TransformerConfig(num_layers=1, hidden_size=2, num_attention_heads=1),
        seq_length=16,
        position_embedding_type='rope',
    )

    yield args


@pytest.fixture
def create_ckpt_load_args(create_args):
    """Setup dummy args allowing checkpoint load."""
    args = create_args
    args.auto_detect_ckpt_format = False
    args.num_layers = 1
    args.hidden_size = 2
    args.num_attention_heads = 1
    args.add_position_embedding = False
    args.tensor_model_parallel_size = 1
    args.pipeline_model_parallel_size = 1
    args.ckpt_assume_constant_structure = False
    args.ckpt_fully_parallel_save = False
    args.ckpt_fully_parallel_load = False
    args.ckpt_load_validate_sharding_integrity = True
    args.dist_ckpt_strictness = 'assume_ok_unexpected'
    args.use_megatron_fsdp = False
    args.strict_fsdp_dtensor_load = True
    args.phase_transition_iterations = None

    cfg = get_run_config()
    cfg.tokenizer.vocab_file = None

    yield args


@pytest.fixture
def init_model_parallel():
    """Init torch distributed."""
    Utils.initialize_model_parallel(1, 1)
    # Guard against leaked state: a prior test's teardown may have raised before
    # reaching `unset_num_microbatches_calculator()`, leaving the calculator
    # initialized for this test's setup.
    unset_num_microbatches_calculator()
    init_num_microbatches_calculator(
        rank=0, global_batch_size=1, micro_batch_size=1, data_parallel_size=1
    )
    model_parallel_cuda_manual_seed(123)
    yield  # Run the actual test.
    Utils.destroy_model_parallel()
    unset_num_microbatches_calculator()


@pytest.mark.parametrize("ckpt_format", ["torch_dcp"])
def test_load_base_checkpoint(
    init_model_parallel, create_ckpt_load_args, ckpt_format, tmp_path_dist_ckpt
):
    """Test _load_base_checkpoint."""

    if ckpt_format == "torch_dcp" and not is_torch_min_version("2.4.0"):
        pytest.skip("torch_dcp requires torch >= 2.4.0")

    # TempNamedDir uses the same directory for all ranks in a multi-GPU setup. Cleanup is handled.
    with TempNamedDir(tmp_path_dist_ckpt / "test_load_base_checkpoint", sync=True) as load_dir:
        create_checkpoint(load_dir, ckpt_format)
        args = create_ckpt_load_args
        args.ckpt_format = ckpt_format

        sync_checkpoint_config(args)
        state_dict, checkpoint_name, release, ckpt_type = _load_base_checkpoint(
            load_dir, args, rank0=True
        )

    assert state_dict["args"] == "dummy"
    assert state_dict["iteration"] == 123

    expected_ckpt_path = None
    if ckpt_format == "torch":
        expected_ckpt_path = str(load_dir / "iter_0000123" / "mp_rank_00" / "model_optim_rng.pt")
    elif ckpt_format == "torch_dcp":
        expected_ckpt_path = str(load_dir / "iter_0000123")

    assert checkpoint_name == expected_ckpt_path
    assert not release

    expected_ckpt_type = None
    if ckpt_format == "torch":
        expected_ckpt_type = CheckpointType.LEGACY
    elif ckpt_format == "torch_dcp":
        expected_ckpt_type = CheckpointType.TORCH_DCP

    assert ckpt_type == expected_ckpt_type


@pytest.mark.parametrize("ckpt_format", ["torch", "torch_dcp", "fsdp_dtensor"])
@pytest.mark.parametrize("iteration", [0, 123])
def test_save_checkpoint(
    init_model_parallel, create_args, tmp_path_dist_ckpt, ckpt_format, iteration, run_config
):
    """Test save_checkpoint."""
    args = create_args
    args.ckpt_format = ckpt_format
    profiling = ProfilingConfig(
        use_nsys_profiler=True, profile_ranks=[0], memory_snapshot_path="owned.pickle"
    )
    run_config.profiling = profiling
    # Runtime config owns these fields, even if the legacy namespace disagrees.
    args.profile = False
    args.profile_ranks = [99]
    args.memory_snapshot_path = "stale.pickle"

    if ckpt_format == "torch_dcp" and not is_torch_min_version("2.4.0"):
        pytest.skip("torch_dcp requires torch >= 2.4.0")

    args.use_distributed_optimizer = ckpt_format != "torch_dcp"
    args.use_dist_ckpt = ckpt_format != "torch"

    config = TransformerConfig(num_layers=1, kv_channels=1)
    model = MockModel(config)
    optimizer = MockState({"optimizer": "optimizer_state"})
    if ckpt_format == "fsdp_dtensor":
        model = FullyShardedDataParallel(
            config=config,
            ddp_config=DistributedDataParallelConfig(
                use_distributed_optimizer=True, use_megatron_fsdp=True
            ),
            module=model,
        )
        optimizer = MockState({"state": {}})
    opt_param_scheduler = MockState({"opt_param_scheduler": "scheduler_state"})
    num_floating_point_operations_so_far = 456

    with TempNamedDir(tmp_path_dist_ckpt / "test_save_checkpoint", sync=True) as save_dir:
        args.save = save_dir
        args.save_tokenizer_assets = False
        set_args(args)

        # Text-only checkpoint saves still bypass dataloader state, while passing CP through.
        cp_group = mock.sentinel.cp_group
        with mock.patch(
            "megatron.training.checkpointing.maybe_save_dataloader_state",
            wraps=maybe_save_dataloader_state,
        ) as save_dataloader_state:
            sync_checkpoint_config(args)
            save_checkpoint(
                iteration,
                [model],
                optimizer,
                opt_param_scheduler,
                num_floating_point_operations_so_far,
                cp_group=cp_group,
            )
        assert save_dataloader_state.call_args.kwargs["cp_group"] is cp_group

        with open(args.save / "latest_checkpointed_iteration.txt", "r") as f:
            assert iteration == int(f.read())

        ckpt_dir = args.save / f"iter_{iteration:07d}"

        expected_ckpt_path = None
        if ckpt_format == "torch":
            expected_ckpt_path = ckpt_dir / "mp_rank_00" / "model_optim_rng.pt"
        elif ckpt_format in ["torch_dcp", "fsdp_dtensor"]:
            expected_ckpt_path = ckpt_dir / ".metadata"

        assert os.path.exists(expected_ckpt_path)
        sync_checkpoint_config(args)
        state, _, _, _ = _load_base_checkpoint(args.save, args, rank0=True)
        # Legacy args remain unchanged; the run config records the effective policy.
        assert state["args"].profile is False and state["args"].profile_ranks == [99]
        assert state["args"].memory_snapshot_path == "stale.pickle"
        with open(ckpt_dir / "run_config.yaml") as f:
            saved_config = yaml.safe_load(f)
        for field in fields(profiling):
            assert saved_config["profiling"][field.name] == getattr(profiling, field.name)
        assert args.profile is False and args.profile_ranks == [99]
        assert args.memory_snapshot_path == "stale.pickle"


@pytest.mark.parametrize("ckpt_format", ["torch"])
@pytest.mark.parametrize(
    "model_cls", [GPTModelConfig, BertModelConfig, T5ModelConfig, LLaVAModelConfig, MimoBuildConfig]
)
@pytest.mark.parametrize("owned_rng", [False, True])
@pytest.mark.parametrize("iteration", [0, 123])
@pytest.mark.parametrize(
    "load_kwargs", [{}, {"restore_training_state": False}, {"restore_training_state": True}]
)
def test_load_checkpoint(
    init_model_parallel,
    create_ckpt_load_args,
    tmp_path_dist_ckpt,
    ckpt_format,
    owned_rng,
    iteration,
    run_config,
    load_kwargs,
    model_cls,
):
    """Test load_checkpoint."""
    args = create_ckpt_load_args
    args.padded_vocab_size = None
    run_config.tokenizer.padded_vocab_size = 128
    args.ckpt_format = ckpt_format
    args.use_distributed_optimizer = ckpt_format != "torch_dcp"
    args.use_dist_ckpt = ckpt_format != "torch"
    args.fp16 = False
    args.bf16 = False
    kwargs = {"encoder_num_layers": 1} if model_cls is T5ModelConfig else {}
    run_config.model = model_cls(transformer=run_config.model.transformer, seq_length=16, **kwargs)
    # Live runtime callbacks are not part of the saved model configuration.
    run_config.model.transformer.grad_scale_func = lambda value: value
    run_config.model.transformer.no_sync_func = [lambda: None]
    rng_config = run_config.rng
    if owned_rng:
        rng_config.seed = 987
        rng_config.data_parallel_random_init = True
        for name in (
            "seed",
            "te_rng_tracker",
            "inference_rng_tracker",
            "data_parallel_random_init",
        ):
            if hasattr(args, name):
                delattr(args, name)

    if ckpt_format == "torch_dcp" and not is_torch_min_version("2.4.0"):
        pytest.skip("torch_dcp requires torch >= 2.4.0")

    with TempNamedDir(tmp_path_dist_ckpt / "test_load_checkpoint", sync=True) as ckpt_dir:
        args.load = str(ckpt_dir)
        args.save = str(ckpt_dir)
        args.save_tokenizer_assets = False
        set_args(args)

        # Create and save a checkpoint first.
        config = TransformerConfig(num_layers=1, kv_channels=1)
        model = MockModel(config)

        optimizer = MockState({"optimizer": "optimizer_state"})
        opt_param_scheduler = MockState({"opt_param_scheduler": "scheduler_state"})
        num_floating_point_operations_so_far = 456

        # Match the save policy: one shared stream unless DP RNG is rank-specific.
        torch.manual_seed(rng_config.seed + (torch.distributed.get_rank() if owned_rng else 0))
        expected_cpu_rng = torch.get_rng_state().clone()
        with mock.patch("torch.save", wraps=torch.save) as serialized:
            sync_checkpoint_config(args)
            save_checkpoint(
                iteration,
                [model],
                optimizer,
                opt_param_scheduler,
                num_floating_point_operations_so_far,
            )
        # With TP=PP=1, rank 0 writes the checkpoint. Other ranks can still call
        # torch.save internally while serializing RNG tensors for all_gather_object.
        if torch.distributed.get_rank() == 0:
            # The persisted arguments are a copy: its checkpoint names follow the section (for a
            # namespace that agrees with it, identical) and its padded_vocab_size is the resolved
            # vocabulary (#8401); the live namespace is not written.
            saved_args = serialized.call_args.args[0]["args"]
            assert saved_args is not args
            assert vars(saved_args) == {**vars(args), "padded_vocab_size": 128}
        assert args.padded_vocab_size is None
        with open(ckpt_dir / f"iter_{iteration:07d}" / "run_config.yaml") as f:
            saved_config = yaml.safe_load(f)
        for field in fields(rng_config):
            assert saved_config["rng"][field.name] == getattr(rng_config, field.name)
        restored_config = type(run_config).from_yaml(
            ckpt_dir / f"iter_{iteration:07d}" / "run_config.yaml"
        )
        assert type(restored_config.model) is model_cls
        # The shared YAML comes from rank 0; distributed rank fields differ on other ranks.
        assert restored_config.to_dict()["model"] == run_config.to_dict()["model"]
        assert restored_config.model.transformer.grad_scale_func is None
        assert restored_config.model.transformer.no_sync_func is None

        # Create new model, optimizer, and scheduler instances to load into.
        new_model = MockModel(config)
        with torch.no_grad():
            new_model.l.weight.zero_()
            new_model.l.bias.fill_(1)
        new_optimizer = MockState({"optimizer": "dummy1"})
        new_opt_param_scheduler = MockState({"opt_param_scheduler": "dummy2"})

        # Load checkpoint
        sync_checkpoint_config(args)
        loaded_iter, loaded_flops = load_checkpoint(
            [new_model], new_optimizer, new_opt_param_scheduler, strict=True, **load_kwargs
        )

        assert torch.equal(torch.get_rng_state(), expected_cpu_rng)
        if owned_rng:
            assert not hasattr(args, "seed")
            assert not hasattr(args, "data_parallel_random_init")
        assert loaded_iter == iteration
        assert loaded_flops == num_floating_point_operations_so_far

        for k in model.state_dict():
            assert torch.equal(model.state_dict()[k], new_model.state_dict()[k])

        if load_kwargs.get("restore_training_state", False):
            assert new_optimizer.state_dict() == optimizer.state_dict()
            assert new_opt_param_scheduler.state_dict() == opt_param_scheduler.state_dict()
        else:
            assert new_optimizer.state_dict() == {"optimizer": "dummy1"}
            assert new_opt_param_scheduler.state_dict() == {"opt_param_scheduler": "dummy2"}


def test_load_checkpoint_stream_ckpt_dequant_refuses_optimizer(create_ckpt_load_args):
    """The streaming dequantize load is weights-only (inference): the loaded state dict holds
    the re-quantized params, so `load_checkpoint` must refuse any optimizer up front, before
    anything is read."""
    args = create_ckpt_load_args
    args.load = "/nonexistent"
    args.no_load_optim = True
    args.stream_ckpt_dequant = True
    set_args(args)

    with pytest.raises(ValueError, match="inference-only"):
        sync_checkpoint_config(args)
        load_checkpoint(
            [MockModel(TransformerConfig(num_layers=1, kv_channels=1))], MockState({}), None
        )


@pytest.mark.parametrize("ckpt_format", ["torch"])
def test_load_checkpoint_override_opt_param_scheduler(
    init_model_parallel, create_ckpt_load_args, tmp_path_dist_ckpt, ckpt_format, run_config
):
    """Test override_opt_param_scheduler behavior during checkpoint load."""
    args = create_ckpt_load_args
    args.ckpt_format = ckpt_format
    args.use_distributed_optimizer = False
    args.use_dist_ckpt = ckpt_format != "torch"
    run_config.scheduler.override_opt_param_scheduler = True
    run_config.optimizer.lr = 1.0
    run_config.optimizer.min_lr = 0.1
    run_config.optimizer.decoupled_lr = 0.5
    run_config.optimizer.decoupled_min_lr = 0.05
    train_state = TrainState(consumed_train_samples=42)

    with (
        mock.patch(
            "megatron.training.checkpointing.get_train_state", return_value=train_state
        ) as get_train_state,
        TempNamedDir(
            tmp_path_dist_ckpt / "test_load_checkpoint_override_opt_param_scheduler", sync=True
        ) as ckpt_dir,
    ):
        args.load = ckpt_dir
        args.save = ckpt_dir
        args.save_tokenizer_assets = False
        set_args(args)

        # Create and save a checkpoint first.
        iteration = 123
        config = TransformerConfig(num_layers=1, kv_channels=1)
        model = MockModel(config)

        optimizer = MockOptimizer({"optimizer": "optimizer_state"})
        optimizer.param_groups = [
            {"is_decoupled_lr": False, "max_lr": -1.0, "min_lr": -1.0},
            {"is_decoupled_lr": True, "max_lr": -1.0, "min_lr": -1.0},
        ]
        opt_param_scheduler = MockOptParamScheduler(
            {"opt_param_scheduler": "scheduler_state", "num_steps": 3}
        )
        num_floating_point_operations_so_far = 456

        sync_checkpoint_config(args)
        save_checkpoint(
            iteration, [model], optimizer, opt_param_scheduler, num_floating_point_operations_so_far
        )

        # Create new model, optimizer, and scheduler instances to load into.
        new_model = MockModel(config)
        new_optimizer = MockOptimizer({"optimizer": "dummy1"})
        new_optimizer.param_groups = [
            {"is_decoupled_lr": False, "max_lr": -2.0, "min_lr": -2.0},
            {"is_decoupled_lr": True, "max_lr": -2.0, "min_lr": -2.0},
        ]
        new_opt_param_scheduler = MockOptParamScheduler(
            {"opt_param_scheduler": "dummy2", "num_steps": 0}
        )

        # Load checkpoint and verify runtime overrides are restored.
        train_state = TrainState()
        get_train_state.return_value = train_state
        sync_checkpoint_config(args)
        loaded_iter, loaded_flops = load_checkpoint(
            [new_model],
            new_optimizer,
            new_opt_param_scheduler,
            strict=True,
            restore_training_state=True,
        )
        assert loaded_iter == iteration
        assert loaded_flops == num_floating_point_operations_so_far
        assert new_optimizer.param_groups[0]["max_lr"] == run_config.optimizer.lr
        assert new_optimizer.param_groups[0]["min_lr"] == run_config.optimizer.min_lr
        assert new_optimizer.param_groups[1]["max_lr"] == run_config.optimizer.decoupled_lr
        assert new_optimizer.param_groups[1]["min_lr"] == run_config.optimizer.decoupled_min_lr
        assert new_opt_param_scheduler.num_steps == train_state.consumed_train_samples == 42
        assert new_opt_param_scheduler.step_calls[-1] == 0

        # Ensure loading without optimizer/scheduler remains safe.
        sync_checkpoint_config(args)
        loaded_iter_none, loaded_flops_none = load_checkpoint(
            [new_model], None, None, strict=True, restore_training_state=True
        )
        assert loaded_iter_none == iteration
        assert loaded_flops_none == num_floating_point_operations_so_far


@pytest.mark.parametrize("load_rerun_mode", [RerunMode.DISABLED, RerunMode.VALIDATE_RESULTS])
def test_strict_load_idle_rerun_checkpoint(
    init_model_parallel, tmp_path_dist_ckpt, create_ckpt_load_args, load_rerun_mode
):
    """Load an idle rerun shard without treating it as a pending rerun."""
    args = create_ckpt_load_args
    # Startup normally fills these; the saved topology must match the real launch.
    args.world_size = torch.distributed.get_world_size()
    args.data_parallel_size = get_data_parallel_world_size()
    args.ckpt_format = "torch_dist"
    args.use_dist_ckpt = True
    args.use_distributed_optimizer = False
    args.dist_ckpt_strictness = "raise_all"
    args.no_save_rng = True
    args.no_load_rng = True
    args.no_save_optim = True
    args.no_load_optim = True
    args.fp16 = False
    args.bf16 = False
    args.save_tokenizer_assets = False
    args.finetune = False
    config = TransformerConfig(num_layers=1, kv_channels=1)
    model = MockModel(config)
    saved_machine = RerunStateMachine(mode=RerunMode.VALIDATE_RESULTS)

    with TempNamedDir(tmp_path_dist_ckpt / "test_strict_idle_rerun", sync=True) as ckpt_dir:
        args.save = ckpt_dir
        args.load = ckpt_dir
        set_args(args)
        with mock.patch(
            "megatron.training.checkpointing.get_rerun_state_machine", return_value=saved_machine
        ):
            sync_checkpoint_config(args)
            save_checkpoint(123, [model], None, None, 456)

        loaded_model = MockModel(config)
        with torch.no_grad():
            loaded_model.l.weight.zero_()
        loaded_machine = RerunStateMachine(mode=load_rerun_mode)
        with mock.patch(
            "megatron.training.checkpointing.get_rerun_state_machine", return_value=loaded_machine
        ):
            sync_checkpoint_config(args)
            iteration, flops = load_checkpoint(
                [loaded_model], None, None, strict=True, restore_training_state=True
            )

        assert iteration == 123
        assert flops == 456
        torch.testing.assert_close(loaded_model.l.weight, model.l.weight, rtol=0, atol=0)
        assert loaded_machine.state == RerunState.NOT_RUNNING_YET
        assert loaded_machine.mode == load_rerun_mode
        assert not loaded_machine.rerun_requested


@pytest.mark.parametrize(
    ('first', 'second', 'should_raise'),
    [
        (3.1, 3.0, False),  # 3.1 student then 3.0 teacher (distillation resume)
        (3.0, 3.1, False),
        (3.1, 3.1, False),
        (3.0, 2.0, True),  # major bump changes QKV layout handling
        (3.1, None, True),
    ],
)
def test_set_checkpoint_version_allows_minor_mismatch(first, second, should_raise):
    """Loading checkpoints that differ only in minor version in one process must not fail."""
    with mock.patch('megatron.training.checkpointing._CHECKPOINT_VERSION', None):
        set_checkpoint_version(first)
        assert get_checkpoint_version() == first
        if should_raise:
            with pytest.raises(AssertionError, match='checkpoint versions do not match'):
                set_checkpoint_version(second)
        else:
            set_checkpoint_version(second)
            assert get_checkpoint_version() == second


def test_dist_checkpoint_versioning(init_model_parallel, tmp_path_dist_ckpt, create_ckpt_load_args):
    """Test distributed checkpoint versioning."""
    args = create_ckpt_load_args
    args.ckpt_format = 'torch_dist'
    args.use_distributed_optimizer = True
    args.use_dist_ckpt = True

    with TempNamedDir(
        tmp_path_dist_ckpt / "test_dist_checkpoint_versioning", sync=True
    ) as ckpt_dir:
        args.load = ckpt_dir
        args.save = ckpt_dir
        args.save_tokenizer_assets = False
        set_args(args)

        # Create and save a checkpoint first.
        iteration = 123
        config = TransformerConfig(num_layers=1, kv_channels=1)
        model = MockModel(config)

        optimizer = MockState({"optimizer": "optimizer_state"})
        opt_param_scheduler = MockState({"opt_param_scheduler": "scheduler_state"})
        num_fp_ops = 456

        base_metadata = _build_sharded_state_dict_metadata()
        first_job_mock_metadata = {**base_metadata, 'metadata_A': 42, 'metadata_B_soon_removed': 43}
        with mock.patch(
            'megatron.training.checkpointing._build_sharded_state_dict_metadata',
            return_value=first_job_mock_metadata,
        ):
            sync_checkpoint_config(args)
            save_checkpoint(iteration, [model], optimizer, opt_param_scheduler, num_fp_ops)
        expected_loaded_metadata = {**first_job_mock_metadata, 'checkpoint_version': 3.1}

        second_job_mock_metadata = {
            **base_metadata,
            'metadata_A': 'changed_default_value',
            'metadata_C_new': {'nested': 'val'},
        }
        with mock.patch(
            'megatron.training.checkpointing._build_sharded_state_dict_metadata',
            return_value=second_job_mock_metadata,
        ):
            # Load checkpoint (into the same model, we don't check load correctness here)
            sync_checkpoint_config(args)
            load_checkpoint(
                [model], optimizer, opt_param_scheduler, strict=True, restore_training_state=True
            )
            assert optimizer._called_metadata[-1] == expected_loaded_metadata

            # Save the checkpoint again to check if the content metadata for the new checkpoint will be new
            sync_checkpoint_config(args)
            save_checkpoint(iteration, [model], optimizer, opt_param_scheduler, num_fp_ops)
            assert optimizer._called_metadata[-1] == second_job_mock_metadata

        assert optimizer._called_metadata == model._called_metadata
        assert optimizer._called_metadata == [
            first_job_mock_metadata,
            expected_loaded_metadata,
            second_job_mock_metadata,
        ]


@pytest.mark.parametrize("ckpt_format", ["torch", "torch_dist"])
@pytest.mark.parametrize("no_load_rng", [False, True])
@pytest.mark.parametrize("load_mode", ["default", "inference", "resume", "override", "converter"])
def test_checkpoint_load_without_microbatch_calculator(
    init_model_parallel,
    create_ckpt_load_args,
    tmp_path_dist_ckpt,
    ckpt_format,
    no_load_rng,
    load_mode,
    run_config,
    monkeypatch,
):
    """Checkpoint loading restores data, not the training runtime calculator."""
    args = create_ckpt_load_args
    args.ckpt_format = ckpt_format
    args.use_distributed_optimizer = False
    args.use_dist_ckpt = ckpt_format != "torch"
    args.world_size = torch.distributed.get_world_size()
    args.data_parallel_size = args.world_size
    args.global_batch_size = args.world_size
    args.micro_batch_size = 1
    run_config.train.global_batch_size = args.global_batch_size
    run_config.train.micro_batch_size = 1
    args.fp16 = False
    args.bf16 = False
    args.hidden_size = 32
    args.num_attention_heads = 4
    train_state = TrainState(consumed_train_samples=123 * args.global_batch_size)
    with (
        mock.patch(
            "megatron.training.checkpointing.get_train_state", return_value=train_state
        ) as get_train_state,
        TempNamedDir(tmp_path_dist_ckpt / "inference_checkpoint", sync=True) as ckpt_dir,
    ):
        args.load = ckpt_dir
        args.save = ckpt_dir
        args.save_tokenizer_assets = False
        set_args(args)
        config = TransformerConfig(
            num_layers=1, hidden_size=32, num_attention_heads=4, use_cpu_initialization=True
        )
        run_config.model = GPTModelConfig(transformer=config, seq_length=16, vocab_size=32)

        def build_model():
            # CPU-initialized DP replicas must match the weights saved by rank 0.
            torch.manual_seed(123)
            return (
                GPTModel(
                    config=config,
                    transformer_layer_spec=get_gpt_layer_local_spec(),
                    vocab_size=32,
                    max_sequence_length=16,
                )
                .cuda()
                .eval()
            )

        torch.manual_seed(123)
        model = build_model()
        sync_checkpoint_config(args)
        save_checkpoint(123, [model], None, None, 456)
        saved_rng = torch.get_rng_state().clone()
        saved_cuda_rng = torch.cuda.get_rng_state().clone()

        restored = build_model()
        with torch.no_grad():
            for parameter in restored.parameters():
                parameter.zero_()
        torch.manual_seed(987)
        before_load_rng = torch.get_rng_state().clone()
        before_load_cuda_rng = torch.cuda.get_rng_state().clone()
        args.no_load_rng = no_load_rng
        # These training options must not influence an inference load.
        args.override_ckpt_iteration = None if load_mode in ("resume", "converter") else 7
        is_inference = load_mode in ("default", "inference")
        args.phase_transition_iterations = [100] if is_inference else None
        args.no_load_optim = True
        train_state = None if is_inference else TrainState()
        get_train_state.return_value = train_state
        if is_inference:
            for name in ("global_batch_size", "micro_batch_size"):
                delattr(args, name)
        unset_num_microbatches_calculator()
        read_config = mock.Mock(wraps=_read_checkpoint_run_config)
        monkeypatch.setattr(
            'megatron.training.checkpointing._read_checkpoint_run_config', read_config
        )
        if is_inference or load_mode == "converter":
            from megatron.training.argument_utils import inference_cfg_container_from_args

            # Converters restore consumed-sample metadata without a training config.
            monkeypatch.setattr(
                "megatron.training.global_vars._GLOBAL_RUN_CONFIG",
                inference_cfg_container_from_args(
                    args, model_cfg=run_config.model, build_model_config=False
                ),
            )
        if is_inference:
            with mock.patch(
                "megatron.training.checkpointing.get_rerun_state_machine",
                side_effect=AssertionError("Inference must not restore training rerun state"),
            ):
                load_kwargs = {} if load_mode == "default" else {"restore_training_state": False}
                sync_checkpoint_config(args)
                metadata = load_checkpoint([restored], None, None, **load_kwargs)
            assert not hasattr(args, "consumed_train_samples")
            assert not hasattr(args, "skipped_train_samples")
            assert not hasattr(args, "consumed_valid_samples")
        else:
            sync_checkpoint_config(args)
            metadata = load_checkpoint([restored], None, None, restore_training_state=True)
            expected_iteration = 7 if load_mode == "override" else 123
            assert train_state.iteration == expected_iteration
            assert train_state.consumed_train_samples == expected_iteration * args.global_batch_size
        assert metadata == (7 if load_mode == "override" else 123, 456)
        read_config.assert_called_once()
        for expected, actual in zip(model.parameters(), restored.parameters()):
            assert torch.equal(expected, actual)
        assert torch.equal(torch.get_rng_state(), before_load_rng if no_load_rng else saved_rng)
        assert torch.equal(
            torch.cuda.get_rng_state(), before_load_cuda_rng if no_load_rng else saved_cuda_rng
        )
        tokens = torch.arange(8, device="cuda").unsqueeze(0)
        with torch.inference_mode():
            assert torch.equal(model(tokens, tokens, None), restored(tokens, tokens, None))


@pytest.mark.parametrize(
    "metadata_content,expected_iter,expected_release",
    [
        ("456", 456, False),  # Normal iteration
        ("release", 0, True),  # Release checkpoint should return iteration=1
        ("123", 123, False),  # Another normal iteration
    ],
)
def test_read_metadata_non_distributed(tmp_path, metadata_content, expected_iter, expected_release):
    """Test read_metadata without torch.distributed initialized."""
    test_dir = tmp_path / "test_read_metadata_non_distributed"
    test_dir.mkdir(parents=True, exist_ok=True)
    tracker_file = test_dir / "latest_checkpointed_iteration.txt"

    with open(tracker_file, "w") as f:
        f.write(metadata_content)

    with mock.patch('torch.distributed.is_initialized', return_value=False):
        max_iter, release = read_metadata(str(tracker_file))

    assert max_iter == expected_iter, f"Expected iteration {expected_iter}, got {max_iter}"
    assert release == expected_release, f"Expected release={expected_release}, got {release}"


def _describe_metadata_run(
    run_config,
    use_distributed_optimizer=False,
    use_layer_wise_distributed_optimizer=False,
    ckpt_format='torch_dist',
    dist_ckpt_optim_fully_reshardable=False,
    distrib_optim_fully_reshardable_mem_efficient=False,
):
    """The metadata builder reads the optimizer flags from the optimizer section and the
    checkpoint settings from the checkpoint section."""
    run_config.optimizer.use_distributed_optimizer = use_distributed_optimizer
    run_config.optimizer.use_layer_wise_distributed_optimizer = use_layer_wise_distributed_optimizer
    run_config.checkpoint = CheckpointConfig(
        ckpt_format=ckpt_format,
        dist_ckpt_optim_fully_reshardable=dist_ckpt_optim_fully_reshardable,
        distrib_optim_fully_reshardable_mem_efficient=(
            distrib_optim_fully_reshardable_mem_efficient
        ),
    )


class TestBuildShardedStateDictMetadata:
    """``_build_sharded_state_dict_metadata`` must set ``distrib_optim_sharding_type``
    whenever a real :class:`DistributedOptimizer` instance will be used at save
    time -- otherwise the DistOpt path falls through to the deprecated
    ``fully_sharded_model_space`` default whose ``flattened_range`` usage is
    rejected by ``ShardedTensor.validate_metadata_integrity`` post commit
    5ab481cb45.
    """

    DUMMY_GROUP = object()

    @pytest.mark.parametrize('ckpt_format', ['torch_dist', 'fsdp_dtensor'])
    def test_inference_container_keeps_shared_metadata(self, monkeypatch, run_config, ckpt_format):
        # No optimizer section: the shared metadata only, whatever the format.
        run_config.checkpoint.ckpt_format = ckpt_format
        cfg = InferenceConfigContainer(
            model=None, checkpoint=run_config.checkpoint, inference=InferenceSetupConfig()
        )
        monkeypatch.setattr('megatron.training.checkpointing.get_run_config', lambda: cfg)

        metadata = _build_sharded_state_dict_metadata(dp_cp_group=self.DUMMY_GROUP)

        assert metadata == {
            'singleton_local_shards': False,
            'chained_optim_avoid_prefix': True,
            'dp_cp_group': self.DUMMY_GROUP,
        }

    def test_distributed_optimizer_sets_dp_reshardable_default(self, run_config):
        _describe_metadata_run(run_config, use_distributed_optimizer=True)
        metadata = _build_sharded_state_dict_metadata(dp_cp_group=self.DUMMY_GROUP)
        assert metadata['distrib_optim_sharding_type'] == 'dp_reshardable'

    def test_distributed_optimizer_fully_reshardable_flag(self, run_config):
        _describe_metadata_run(
            run_config, use_distributed_optimizer=True, dist_ckpt_optim_fully_reshardable=True
        )
        metadata = _build_sharded_state_dict_metadata(dp_cp_group=self.DUMMY_GROUP)
        assert metadata['distrib_optim_sharding_type'] == 'fully_reshardable'
        assert metadata['distrib_optim_fully_reshardable_mem_efficient'] is False

    def test_distributed_optimizer_fsdp_dtensor(self, run_config):
        _describe_metadata_run(
            run_config, use_distributed_optimizer=True, ckpt_format='fsdp_dtensor'
        )
        metadata = _build_sharded_state_dict_metadata(dp_cp_group=self.DUMMY_GROUP)
        assert metadata['distrib_optim_sharding_type'] == 'fsdp_dtensor'

    def test_layer_wise_only_still_sets_sharding_type(self, run_config):
        # Arg parser flips ``use_distributed_optimizer`` off when Muon is in
        # use, but the LayerWise + DistOpt split path still has a DistOpt
        # sub-optimizer for non-Muon params, so the metadata is required.
        _describe_metadata_run(run_config, use_layer_wise_distributed_optimizer=True)
        metadata = _build_sharded_state_dict_metadata(dp_cp_group=self.DUMMY_GROUP)
        assert metadata['distrib_optim_sharding_type'] == 'dp_reshardable'

    def test_layer_wise_with_fully_reshardable(self, run_config):
        _describe_metadata_run(
            run_config,
            use_layer_wise_distributed_optimizer=True,
            dist_ckpt_optim_fully_reshardable=True,
        )
        metadata = _build_sharded_state_dict_metadata(dp_cp_group=self.DUMMY_GROUP)
        assert metadata['distrib_optim_sharding_type'] == 'fully_reshardable'

    def test_no_distributed_optimizer_no_sharding_type(self, run_config):
        _describe_metadata_run(run_config)
        metadata = _build_sharded_state_dict_metadata(dp_cp_group=self.DUMMY_GROUP)
        assert 'distrib_optim_sharding_type' not in metadata


def test_save_follows_the_section_not_the_namespace(
    init_model_parallel, create_args, tmp_path_dist_ckpt, run_config
):
    """A save goes where the registered section says, in its format, whatever the namespace says,
    and the persisted arguments describe the policy that produced the checkpoint."""
    args = create_args
    args.use_distributed_optimizer = False
    config = TransformerConfig(num_layers=1, kv_channels=1)
    model = MockModel(config)
    optimizer = MockState({"optimizer": "optimizer_state"})
    opt_param_scheduler = MockState({"opt_param_scheduler": "scheduler_state"})

    with TempNamedDir(tmp_path_dist_ckpt / "test_save_follows_the_section", sync=True) as save_dir:
        args.save = str(save_dir / "section")
        args.ckpt_format = "torch"
        args.use_dist_ckpt = False
        args.save_tokenizer_assets = False
        set_args(args)
        before = replace(sync_checkpoint_config(args))
        # The namespace now disagrees on every setting a save reads.
        args.save = str(save_dir / "namespace")
        args.ckpt_format = "torch_dist"
        args.use_dist_ckpt = True
        args.async_save = True
        args.no_save_optim = True

        save_checkpoint(7, [model], optimizer, opt_param_scheduler, 0)

        ckpt_file = save_dir / "section" / "iter_0000007" / "mp_rank_00" / "model_optim_rng.pt"
        assert os.path.exists(ckpt_file)
        assert not os.path.exists(save_dir / "namespace")
        state = torch.load(ckpt_file, weights_only=False)
        assert "optimizer" in state
        assert state["args"].save == str(save_dir / "section")
        assert state["args"].ckpt_format == "torch"
        assert state["args"].no_save_optim is False and state["args"].use_dist_ckpt is False
        # Neither the namespace nor the registered section was written.
        assert args.ckpt_format == "torch_dist" and args.no_save_optim is True
        assert run_config.checkpoint == before


@pytest.mark.parametrize("section_says_rng", [True, False, None])
def test_the_saved_section_decides_what_a_resume_requests(
    init_model_parallel, create_ckpt_load_args, tmp_path_dist_ckpt, run_config, section_says_rng
):
    """The load template requests what the checkpoint holds as its saved checkpoint section says;
    the arguments persisted with the checkpoint decide only when there is no saved section (None
    here: the YAML is removed, and those arguments say the RNG state was saved)."""
    args = create_ckpt_load_args
    args.world_size = torch.distributed.get_world_size()
    args.data_parallel_size = get_data_parallel_world_size()
    args.ckpt_format = "torch_dist"
    args.use_dist_ckpt = True
    args.use_distributed_optimizer = False
    args.fp16 = False
    args.bf16 = False
    args.save_tokenizer_assets = False
    config = TransformerConfig(num_layers=1, kv_channels=1)

    with TempNamedDir(tmp_path_dist_ckpt / "test_saved_section_decides", sync=True) as ckpt_dir:
        args.save = str(ckpt_dir)
        args.load = str(ckpt_dir)
        set_args(args)
        sync_checkpoint_config(args)
        save_checkpoint(5, [MockModel(config)], None, None, 0)

        config_filename = get_checkpoint_run_config_filename(str(ckpt_dir / "iter_0000005"))
        if torch.distributed.get_rank() == 0:
            if section_says_rng is None:
                os.remove(config_filename)
            elif section_says_rng is False:
                with open(config_filename) as handle:
                    saved = yaml.safe_load(handle)
                saved["checkpoint"]["save_rng"] = False
                with open(config_filename, "w") as handle:
                    yaml.safe_dump(saved, handle)
        torch.distributed.barrier()

        with mock.patch(
            "megatron.training.checkpointing.get_rng_state", wraps=checkpointing.get_rng_state
        ) as rng_template:
            iteration, _ = load_checkpoint([MockModel(config)], None, None, strict=True)
        assert iteration == 5
        # The RNG shard is requested unless the saved section says it is not there.
        assert rng_template.called is (section_says_rng is not False)


@pytest.mark.parametrize(
    "saved_section, saved_no_save_optim, expected",
    [
        ({"checkpoint": {"save_optim": True}}, True, True),  # the saved section wins
        ({"checkpoint": {"save_optim": False}}, False, False),  # in both directions
        (None, True, False),  # no saved section: the saved arguments decide
        (None, False, True),
    ],
)
def test_the_glu_optimizer_guard_follows_the_saved_checkpoint_section(
    init_model_parallel, monkeypatch, run_config, saved_section, saved_no_save_optim, expected
):
    """The guard that keeps optimizer rows from being loaded across a different GLU layout
    decides whether optimizer state will be loaded exactly as the load template does: from the
    saved checkpoint section, and from the saved arguments only when the checkpoint has none."""
    args = SimpleNamespace(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        world_size=1,
        data_parallel_size=1,
        swiglu=True,
        moe_mlp_glu_interleave_size=2,
        use_distributed_optimizer=False,
    )
    saved_args = SimpleNamespace(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        world_size=1,
        data_parallel_size=1,
        no_save_optim=saved_no_save_optim,
        swiglu=True,
        moe_mlp_glu_interleave_size=1,
    )
    state = {"args": saved_args, "model": {}}
    run_config.checkpoint.load = "/synthetic"
    run_config.checkpoint.load_rng = False
    monkeypatch.setattr(checkpointing, "get_args", lambda: args)
    monkeypatch.setattr(checkpointing, "unwrap_model", lambda value: value)
    monkeypatch.setattr(
        checkpointing, "_otel_managed_span", lambda *a, **k: contextlib.nullcontext()
    )
    monkeypatch.setattr(
        checkpointing,
        "_load_base_checkpoint",
        lambda *a, **k: (state, "/synthetic/iter_0000001", False, CheckpointType.GLOBAL),
    )
    monkeypatch.setattr(checkpointing, "_read_checkpoint_run_config", lambda *a, **k: saved_section)
    monkeypatch.setattr(
        checkpointing, "_maybe_setup_gpt_to_hybrid_load", lambda *a, **k: (None, False)
    )
    monkeypatch.setattr(checkpointing, "validate_glu_checkpoint_backend", mock.Mock())
    decisions = []

    class StopAtTheGuard(Exception):
        pass

    def guard(state_dict, run_args, *, loading_optimizer):
        decisions.append(loading_optimizer)
        raise StopAtTheGuard  # Before any tensor storage is read.

    monkeypatch.setattr(checkpointing, "validate_glu_optimizer_layout", guard)
    optimizer = SimpleNamespace(is_stub_optimizer=False)
    with pytest.raises(StopAtTheGuard):
        load_checkpoint([object()], optimizer, None, restore_training_state=True)
    assert decisions == [expected]


def test_the_pretrained_fallback_makes_the_load_weights_only(
    init_model_parallel, monkeypatch, run_config
):
    """A missing --load with --pretrained-checkpoint loads the pretrained directory weights-only:
    the base loader is called with finetune=True; neither the registered section nor the
    namespace is written."""
    args = SimpleNamespace(
        finetune=False,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        world_size=1,
        data_parallel_size=1,
        swiglu=False,
        use_distributed_optimizer=False,
    )
    run_config.checkpoint.load = "/missing"
    run_config.checkpoint.pretrained_checkpoint = "/pretrained"
    run_config.checkpoint.load_rng = False
    loads = []

    def base_load(load_dir, run_args, *a, finetune, **k):
        loads.append((load_dir, finetune))
        return None, "", False, None  # Nothing to load: the call ends after the fallback.

    monkeypatch.setattr(checkpointing, "get_args", lambda: args)
    monkeypatch.setattr(checkpointing, "checkpoint_exists", lambda path: path == "/pretrained")
    monkeypatch.setattr(checkpointing, "unwrap_model", lambda value: value)
    monkeypatch.setattr(
        checkpointing, "_otel_managed_span", lambda *a, **k: contextlib.nullcontext()
    )
    monkeypatch.setattr(checkpointing, "_load_base_checkpoint", base_load)
    assert load_checkpoint([object()], None, None) == (0, 0)
    assert loads and loads[0] == ("/pretrained", True)
    assert run_config.checkpoint.finetune is False and args.finetune is False


def test_the_load_directory_is_the_sections_unless_passed(
    init_model_parallel, monkeypatch, run_config
):
    """``load_dir`` omitted reads the run's load directory; ``None`` clears it, so the
    pretrained-checkpoint fallback applies (the reference weights of RL); a string names another
    directory (the distillation teacher's). None of them writes the section."""
    args = SimpleNamespace(
        finetune=False,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        world_size=1,
        data_parallel_size=1,
        swiglu=False,
        use_distributed_optimizer=False,
    )
    run_config.checkpoint.load = "/run"
    run_config.checkpoint.pretrained_checkpoint = "/pretrained"
    loads = []

    def base_load(load_dir, run_args, *a, finetune, **k):
        if not k.get("rank0"):  # The final read; the rank-0 metadata read carries the same policy.
            loads.append((load_dir, finetune))
        return None, "", False, None  # Nothing to load: the call ends after resolving the policy.

    monkeypatch.setattr(checkpointing, "get_args", lambda: args)
    monkeypatch.setattr(
        checkpointing, "checkpoint_exists", lambda path: path in ("/run", "/pretrained", "/teacher")
    )
    monkeypatch.setattr(checkpointing, "unwrap_model", lambda value: value)
    monkeypatch.setattr(
        checkpointing, "_otel_managed_span", lambda *a, **k: contextlib.nullcontext()
    )
    monkeypatch.setattr(checkpointing, "_load_base_checkpoint", base_load)

    load_checkpoint([object()], None, None)
    load_checkpoint([object()], None, None, load_dir=None, finetune=True, load_optim=False)
    load_checkpoint([object()], None, None, load_dir="/teacher", finetune=True)
    assert loads == [("/run", False), ("/pretrained", True), ("/teacher", True)]
    assert run_config.checkpoint.load == "/run" and run_config.checkpoint.finetune is False


@pytest.mark.parametrize("cleared", [True, False])
def test_the_pretrained_reference_load_requests_the_llm_only_metadata(
    init_model_parallel, monkeypatch, run_config, cleared
):
    """The reference-weights load of RL clears the load directory, so the pretrained fallback
    classifies it as a pretrained load: with --allow-llm-only-checkpoint the load template then
    carries ``load_from_llm_only_checkpoint``. Naming the pretrained directory directly is an
    ordinary load and does not."""
    args = SimpleNamespace(
        finetune=False,
        allow_llm_only_checkpoint=True,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        world_size=1,
        data_parallel_size=1,
        swiglu=False,
        use_distributed_optimizer=False,
    )
    saved_args = SimpleNamespace(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        world_size=1,
        data_parallel_size=1,
    )
    state = {"args": saved_args, "model": {}}
    run_config.checkpoint.load = "/missing"
    run_config.checkpoint.pretrained_checkpoint = "/pretrained"
    run_config.checkpoint.ckpt_format = "torch_dist"
    monkeypatch.setattr(checkpointing, "get_args", lambda: args)
    monkeypatch.setattr(checkpointing, "checkpoint_exists", lambda path: path == "/pretrained")
    monkeypatch.setattr(checkpointing, "unwrap_model", lambda value: value)
    monkeypatch.setattr(
        checkpointing, "_otel_managed_span", lambda *a, **k: contextlib.nullcontext()
    )
    monkeypatch.setattr(
        checkpointing,
        "_load_base_checkpoint",
        lambda *a, **k: (state, "/pretrained/iter_0000001", False, CheckpointType.GLOBAL),
    )
    monkeypatch.setattr(checkpointing, "_read_checkpoint_run_config", lambda *a, **k: None)
    monkeypatch.setattr(
        checkpointing, "_maybe_setup_gpt_to_hybrid_load", lambda *a, **k: (None, False)
    )
    monkeypatch.setattr(checkpointing, "validate_glu_checkpoint_backend", mock.Mock())
    monkeypatch.setattr(checkpointing, "validate_glu_optimizer_layout", mock.Mock())
    templates = []

    class StopAtTheTemplate(Exception):
        pass

    def capture(*a, model_sd_kwargs, **k):
        templates.append(model_sd_kwargs["metadata"])
        raise StopAtTheTemplate  # Before any tensor storage is read.

    monkeypatch.setattr(checkpointing, "generate_state_dict", capture)
    load_dir = None if cleared else "/pretrained"
    with pytest.raises(StopAtTheTemplate):
        load_checkpoint(
            [object()],
            None,
            None,
            restore_training_state=True,
            load_dir=load_dir,
            finetune=True,
            load_optim=False,
        )
    assert templates[0].get("load_from_llm_only_checkpoint", False) is cleared


def _loader_args(**overrides):
    values = dict(
        finetune=False,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        world_size=1,
        data_parallel_size=1,
        swiglu=False,
        use_distributed_optimizer=False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _stub_loader_to_the_policy(monkeypatch, args, existing, loads):
    """Stop every load right after its policy is resolved: nothing is read from storage."""

    def base_load(load_dir, run_args, *a, finetune, **k):
        if not k.get("rank0"):  # The final read; the rank-0 metadata read carries the same policy.
            loads.append((load_dir, finetune))
        return None, "", False, None

    monkeypatch.setattr(checkpointing, "get_args", lambda: args)
    monkeypatch.setattr(checkpointing, "checkpoint_exists", lambda path: path in existing)
    monkeypatch.setattr(checkpointing, "unwrap_model", lambda value: value)
    monkeypatch.setattr(
        checkpointing, "_otel_managed_span", lambda *a, **k: contextlib.nullcontext()
    )
    monkeypatch.setattr(checkpointing, "_load_base_checkpoint", base_load)


def test_only_the_primary_load_records_the_finetune_classification(
    init_model_parallel, monkeypatch, run_config
):
    """The primary load writes TrainState.effective_finetune, True or False; the auxiliary loads
    (the distillation teacher, RL's reference weights and reload) leave it alone. The namespace
    is not written."""
    args = _loader_args()
    train_state = TrainState()
    monkeypatch.setattr(checkpointing, "get_train_state", lambda: train_state)
    run_config.checkpoint.load = "/missing"
    run_config.checkpoint.pretrained_checkpoint = "/pretrained"
    loads = []
    _stub_loader_to_the_policy(monkeypatch, args, ("/pretrained", "/teacher"), loads)

    # The primary load takes the pretrained fallback: weights-only.
    load_checkpoint([object()], None, None, restore_training_state=True, record_finetune=True)
    assert train_state.effective_finetune is True
    # A teacher load and RL's reload are not the primary load.
    load_checkpoint([object()], None, None, load_dir="/teacher", finetune=False)
    load_checkpoint([object()], None, None, restore_training_state=True, load_optim=False)
    assert train_state.effective_finetune is True
    assert loads == [("/pretrained", True), ("/teacher", False), ("/pretrained", True)]
    assert args.finetune is False and not hasattr(args, "override_ckpt_iteration")

    # A later primary resume of a saved run resets the classification.
    run_config.checkpoint.load = "/saved"
    monkeypatch.setattr(checkpointing, "checkpoint_exists", lambda path: path == "/saved")
    load_checkpoint([object()], None, None, restore_training_state=True, record_finetune=True)
    assert train_state.effective_finetune is False
    assert run_config.checkpoint.finetune is False


@pytest.mark.parametrize("progress_iteration", [0, 250])
def test_the_frozen_layer_load_is_weights_only_at_any_data_position(
    init_model_parallel, monkeypatch, run_config, progress_iteration
):
    """--freeze-all-layers loads the model weights-only and resumes the data position from the
    run's own progress tracker: the classification is recorded on the train state, the position
    stays the load's own business, and the namespace receives neither."""
    args = _loader_args(freeze_all_layers=True)
    train_state = TrainState()
    monkeypatch.setattr(checkpointing, "get_train_state", lambda: train_state)
    run_config.checkpoint.load = "/frozen"
    run_config.checkpoint.save = "/out"
    progress_reads = []

    def read_progress(save_dir):
        progress_reads.append(save_dir)
        return progress_iteration

    monkeypatch.setattr(checkpointing, "read_frozen_resume_iteration", read_progress)
    loads = []
    _stub_loader_to_the_policy(monkeypatch, args, ("/frozen",), loads)

    load_checkpoint([object()], None, None, restore_training_state=True, record_finetune=True)
    assert loads == [("/frozen", True)]
    assert progress_reads == ["/out"]
    assert train_state.effective_finetune is True
    assert not hasattr(args, "override_ckpt_iteration") and args.finetune is False
    assert run_config.checkpoint.override_ckpt_iteration is None


@pytest.mark.parametrize(
    "case",
    ["both", "train_state_only", "tracker_only", "ckpt_step", "release", "pretrained_fallback"],
)
def test_the_fsdp_dtensor_reader_follows_the_train_state_tracker(
    init_model_parallel, monkeypatch, run_config, tmp_path, case
):
    """The directory handed to the fsdp_dtensor reader comes from the same resolution as the base
    load: latest_train_state.pt when present, the legacy text tracker otherwise, --ckpt-step over
    both, the release flag honoured, and the pretrained fallback's directory when --load has no
    checkpoint."""

    def tracked(directory, tracker=None, train_state=None, release=False):
        directory.mkdir(parents=True, exist_ok=True)
        if tracker is not None:
            with open(get_checkpoint_tracker_filename(str(directory)), "w") as f:
                f.write("release" if release and train_state is None else str(tracker))
        if train_state is not None:
            torch.save(
                TrainState(iteration=train_state, release=release).state_dict(),
                get_checkpoint_train_state_filename(str(directory), prefix="latest"),
            )

    load_dir, pretrained_dir = tmp_path / "load", tmp_path / "pretrained"
    expected = {
        "both": (dict(tracker=10, train_state=20), "iter_0000020"),
        "train_state_only": (dict(train_state=20), "iter_0000020"),
        "tracker_only": (dict(tracker=10), "iter_0000010"),
        "ckpt_step": (dict(tracker=10, train_state=20), "iter_0000030"),
        "release": (dict(tracker=10, train_state=20, release=True), "release"),
        "pretrained_fallback": (None, "iter_0000020"),
    }
    files, leaf = expected[case]
    if files is not None:
        tracked(load_dir, **files)
        wanted = str(load_dir / leaf)
    else:
        load_dir.mkdir()  # no checkpoint under --load: the pretrained directory is read
        tracked(pretrained_dir, tracker=10, train_state=20)
        wanted = str(pretrained_dir / leaf)
        run_config.checkpoint.pretrained_checkpoint = str(pretrained_dir)
    run_config.checkpoint.load = str(load_dir)
    run_config.checkpoint.ckpt_format = "fsdp_dtensor"
    if case == "ckpt_step":
        run_config.checkpoint.ckpt_step = 30

    args = _loader_args(use_megatron_fsdp=True)
    saved_args = SimpleNamespace(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        world_size=1,
        data_parallel_size=1,
    )
    state = {"args": saved_args, "model": {}}
    monkeypatch.setattr(checkpointing, "get_args", lambda: args)
    monkeypatch.setattr(checkpointing, "unwrap_model", lambda value: value)
    monkeypatch.setattr(
        checkpointing, "_otel_managed_span", lambda *a, **k: contextlib.nullcontext()
    )
    monkeypatch.setattr(
        checkpointing,
        "_load_base_checkpoint",
        lambda *a, **k: (state, wanted, False, CheckpointType.FSDP_DTENSOR),
    )
    monkeypatch.setattr(checkpointing, "_read_checkpoint_run_config", lambda *a, **k: None)
    monkeypatch.setattr(
        checkpointing, "_maybe_setup_gpt_to_hybrid_load", lambda *a, **k: (None, False)
    )
    monkeypatch.setattr(checkpointing, "validate_glu_checkpoint_backend", mock.Mock())
    monkeypatch.setattr(checkpointing, "validate_glu_optimizer_layout", mock.Mock())
    readers = []

    class StopAtTheReader(Exception):
        pass

    def reader(path):
        readers.append(path)
        raise StopAtTheReader  # Before any metadata is read from storage.

    monkeypatch.setattr(checkpointing, "FileSystemReader", reader)
    with pytest.raises(StopAtTheReader):
        load_checkpoint([object()], None, None, restore_training_state=True)
    assert readers == [wanted]
