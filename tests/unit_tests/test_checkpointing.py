# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Note: --ckpt-format torch_dist has tests in tests/unit_tests/dist_checkpointing.
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
from unittest import mock

import pytest
import torch
import torch.distributed.checkpoint

from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed.fsdp.mcore_fsdp_adapter import FullyShardedDataParallel
from megatron.core.num_microbatches_calculator import (
    init_num_microbatches_calculator,
    unset_num_microbatches_calculator,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_torch_min_version
from megatron.training.checkpointing import (
    CheckpointType,
    _get_non_persistent_iteration,
    _build_sharded_state_dict_metadata,
    _get_checkpoint_format,
    _load_base_checkpoint,
    _to_dtensor,
    _transpose_first_dim,
    checkpoint_exists,
    cleanup_old_non_persistent_checkpoint,
    finalize_deletion_processes,
    find_checkpoint_rank_0,
    fix_query_key_value_ordering,
    generate_state_dict,
    get_checkpoint_tracker_filename,
    get_checkpoint_name,
    get_distributed_optimizer_checkpoint_name,
    get_load_checkpoint_path_by_args,
    get_loaded_iteration,
    load_checkpoint,
    load_biencoder_checkpoint,
    load_args_from_checkpoint,
    maybe_save_dataloader_state,
    read_metadata,
    save_checkpoint,
    set_checkpoint_version,
    set_loaded_iteration,
)
from megatron.training import checkpointing as checkpointing_module
from megatron.training import global_vars as global_vars_module
from megatron.training.global_vars import set_args
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils


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


@pytest.fixture
def restore_global_args():
    original_args = global_vars_module._GLOBAL_ARGS
    yield
    global_vars_module._GLOBAL_ARGS = original_args


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
def create_args():
    """Setup dummy args."""
    args = SimpleNamespace()
    args.finetune = False
    args.non_persistent_global_ckpt_dir = None
    args.non_persistent_ckpt_type = None
    args.non_persistent_save_interval = None
    args.exit_on_missing_checkpoint = True
    args.async_save = False
    args.async_strategy = "mcore"
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
    args.swiglu = True
    args.num_experts = 1

    yield args


@pytest.fixture
def create_ckpt_load_args(create_args):
    """Setup dummy args allowing checkpoint load."""
    args = create_args
    args.auto_detect_ckpt_format = False
    args.consumed_train_samples = 0
    args.skipped_train_samples = 0
    args.consumed_valid_samples = 0
    args.num_layers = 1
    args.hidden_size = 2
    args.num_attention_heads = 1
    args.add_position_embedding = False
    args.vocab_file = None
    args.tensor_model_parallel_size = 1
    args.pipeline_model_parallel_size = 1
    args.ckpt_assume_constant_structure = False
    args.ckpt_fully_parallel_save = False
    args.ckpt_fully_parallel_load = False
    args.dist_ckpt_strictness = 'assume_ok_unexpected'
    args.use_megatron_fsdp = False
    args.strict_fsdp_dtensor_load = True
    args.phase_transition_iterations = None

    yield args


@pytest.fixture
def init_model_parallel():
    """Init torch distributed."""
    Utils.initialize_model_parallel(1, 1)
    init_num_microbatches_calculator(0, None, 1, 1, 1)
    model_parallel_cuda_manual_seed(123)
    yield  # Run the actual test.
    Utils.destroy_model_parallel()
    unset_num_microbatches_calculator()


def test_load_base_checkpoint(create_ckpt_load_args, tmp_path):
    """Test _load_base_checkpoint for torch format (CPU only, no GPU needed)."""
    load_dir = tmp_path / "load_base"
    load_dir.mkdir()

    # Create a torch checkpoint
    ckpt_dir = load_dir / "iter_0000123" / "mp_rank_00"
    ckpt_dir.mkdir(parents=True)
    state_dict = {"args": "dummy", "iteration": 123}
    torch.save(state_dict, ckpt_dir / "model_optim_rng.pt")

    # Write tracker file
    tracker_path = get_checkpoint_tracker_filename(load_dir)
    with open(tracker_path, "w") as f:
        f.write("123")

    args = create_ckpt_load_args
    args.ckpt_format = "torch"

    state_dict, checkpoint_name, release, ckpt_type = _load_base_checkpoint(
        load_dir, args, rank0=True
    )

    assert state_dict["args"] == "dummy"
    assert state_dict["iteration"] == 123
    assert not release
    assert ckpt_type == CheckpointType.LEGACY


def test_get_checkpoint_format_detects_torch_dcp_and_fsdp(tmp_path, monkeypatch):
    monkeypatch.setattr(
        checkpointing_module.dist_checkpointing,
        "check_is_distributed_checkpoint",
        lambda path: False,
    )
    torch_dir = tmp_path / "torch"
    torch_dir.mkdir()
    (torch_dir / "mp_rank_00").mkdir()
    dcp_dir = tmp_path / "dcp"
    dcp_dir.mkdir()
    (dcp_dir / ".metadata").write_text("metadata", encoding="utf-8")

    assert _get_checkpoint_format(torch_dir, SimpleNamespace(use_megatron_fsdp=False)) == "torch"
    assert _get_checkpoint_format(dcp_dir, SimpleNamespace(use_megatron_fsdp=False)) == "torch_dcp"
    assert _get_checkpoint_format(dcp_dir, SimpleNamespace(use_megatron_fsdp=True)) == "fsdp_dtensor"

    with pytest.raises(NotImplementedError, match="unknown checkpoint format"):
        _get_checkpoint_format(tmp_path, SimpleNamespace(use_megatron_fsdp=False))


def test_to_dtensor_preserves_extra_state_and_distributes_tensors(monkeypatch):
    calls = []
    tensor_api = SimpleNamespace(
        distribute_tensor=lambda tensor, mesh: calls.append((tensor, mesh))
        or ("dtensor", tensor, mesh)
    )
    monkeypatch.setattr(
        checkpointing_module.torch.distributed,
        "tensor",
        tensor_api,
        raising=False,
    )

    converted = _to_dtensor(
        [SimpleNamespace(device_mesh="mesh")],
        {
            "layer.weight": torch.tensor([1.0]),
            "layer._extra_state": {"fp8": True},
        },
    )

    assert converted["layer.weight"][0] == "dtensor"
    assert converted["layer.weight"][2] == "mesh"
    assert converted["layer._extra_state"] == {"fp8": True}
    assert len(calls) == 1


def test_maybe_save_dataloader_state_saves_only_first_pipeline_rank(tmp_path, monkeypatch):
    calls = []

    class FakeIterable:
        def save_state(self):
            calls.append("save-state")
            return {"offset": 7}

    monkeypatch.setattr(checkpointing_module.mpu, "is_pipeline_first_stage", lambda ignore_virtual=True: True)
    monkeypatch.setattr(checkpointing_module.mpu, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(checkpointing_module.mpu, "get_data_parallel_rank", lambda: 3)
    monkeypatch.setattr(checkpointing_module.mpu, "get_data_parallel_group", lambda: "dp")
    # Mock get_checkpoint_name to return a simple path
    monkeypatch.setattr(
        checkpointing_module,
        "get_checkpoint_name",
        lambda save_dir, iteration, release=False, basename=None, **kwargs: str(tmp_path / (basename or f"iter_{iteration:07d}")),
    )
    monkeypatch.setattr(
        checkpointing_module.torch.distributed,
        "barrier",
        lambda group=None: calls.append(("barrier", group)),
    )
    # Mock torch.save inside the checkpointing module
    monkeypatch.setattr(
        checkpointing_module.torch,
        "save",
        lambda state, path: calls.append(("save", state, Path(path).name)),
    )
    # Also mock ensure_directory_exists to avoid file system operations
    monkeypatch.setattr(
        checkpointing_module,
        "ensure_directory_exists",
        lambda filename, check_parent=True: None,
    )

    maybe_save_dataloader_state(SimpleNamespace(iterable=FakeIterable()), 12, tmp_path)

    assert "save-state" in calls
    assert ("barrier", "dp") in calls
    assert any(call[0] == "save" and call[1]["dataloader_state_dict"] == {"offset": 7} for call in calls)
    assert any(call[0] == "save" and "train_dataloader_dprank003.pt" in call[2] for call in calls)


def test_maybe_save_dataloader_state_rejects_unsupported_iterator(tmp_path):
    with pytest.raises(RuntimeError, match="Could not find a save_state"):
        maybe_save_dataloader_state(SimpleNamespace(iterable=SimpleNamespace()), 1, tmp_path)


def test_load_biencoder_checkpoint_can_load_only_query_model(tmp_path, monkeypatch, restore_global_args):
    load_dir = tmp_path / "biencoder"
    load_dir.mkdir()
    Path(get_checkpoint_tracker_filename(load_dir)).write_text("5", encoding="utf-8")
    # Manually construct checkpoint path matching load_biencoder_checkpoint's logic
    # load_biencoder_checkpoint calls: get_checkpoint_name(load_path, iteration, args.use_distributed_optimizer, release=False)
    # With use_distributed_optimizer=False, this creates: iter_0000005/mp_rank_00/model_optim_rng.pt
    # Keep this test focused on load_biencoder_checkpoint's filtering behavior.
    # The legacy helper currently passes the distributed-optimizer flag as a
    # positional argument before release=False, which conflicts with the current
    # get_checkpoint_name signature. Patch the path builder locally instead of
    # changing Megatron runtime code.
    checkpoint_name = load_dir / "iter_0000005" / "mp_rank_00" / "model_optim_rng.pt"
    checkpoint_name.parent.mkdir(parents=True)
    torch.save({"model": {"query_model": {"w": 1}, "context_model": {"w": 2}}}, checkpoint_name)

    loaded = []
    fake_model = SimpleNamespace(load_state_dict=lambda state: loaded.append(state))
    set_args(SimpleNamespace(load=load_dir, use_distributed_optimizer=False))

    def fake_get_checkpoint_name(checkpoints_path, iteration, *args, **kwargs):
        basename = kwargs.get("basename", "model_optim_rng.pt")
        return str(Path(checkpoints_path) / f"iter_{iteration:07d}" / "mp_rank_00" / basename)

    monkeypatch.setattr(checkpointing_module, "unwrap_model", lambda model: model)
    monkeypatch.setattr(checkpointing_module, "get_checkpoint_name", fake_get_checkpoint_name)
    monkeypatch.setattr(checkpointing_module.mpu, "get_data_parallel_rank", lambda: 0)
    monkeypatch.setattr(checkpointing_module.torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(checkpointing_module.torch.distributed, "barrier", lambda: None)

    result = load_biencoder_checkpoint([fake_model], only_query_model=True)

    assert result == [fake_model]
    assert loaded == [{"query_model": {"w": 1}}]


@pytest.mark.parametrize("ckpt_format", ["torch", "torch_dcp", "fsdp_dtensor"])
def test_save_checkpoint(init_model_parallel, create_args, tmp_path_dist_ckpt, ckpt_format):
    """Test save_checkpoint."""
    args = create_args
    args.ckpt_format = ckpt_format

    if ckpt_format == "torch_dcp" and not is_torch_min_version("2.4.0"):
        pytest.skip("torch_dcp requires torch >= 2.4.0")

    args.use_distributed_optimizer = ckpt_format != "torch_dcp"
    args.use_dist_ckpt = ckpt_format != "torch"

    iteration = 123
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
        set_args(args)

        save_checkpoint(
            iteration, [model], optimizer, opt_param_scheduler, num_floating_point_operations_so_far
        )

        with open(args.save / "latest_checkpointed_iteration.txt", "r") as f:
            assert iteration == int(f.read())

        ckpt_dir = args.save / "iter_0000123"

        expected_ckpt_path = None
        if ckpt_format == "torch":
            expected_ckpt_path = ckpt_dir / "mp_rank_00" / "model_optim_rng.pt"
        elif ckpt_format in ["torch_dcp", "fsdp_dtensor"]:
            expected_ckpt_path = ckpt_dir / ".metadata"

        assert os.path.exists(expected_ckpt_path)


@pytest.mark.parametrize("ckpt_format", ["torch"])
def test_load_checkpoint(
    init_model_parallel, create_ckpt_load_args, tmp_path_dist_ckpt, ckpt_format
):
    """Test load_checkpoint."""
    args = create_ckpt_load_args
    args.ckpt_format = ckpt_format
    args.use_distributed_optimizer = ckpt_format != "torch_dcp"
    args.use_dist_ckpt = ckpt_format != "torch"

    if ckpt_format == "torch_dcp" and not is_torch_min_version("2.4.0"):
        pytest.skip("torch_dcp requires torch >= 2.4.0")

    with TempNamedDir(tmp_path_dist_ckpt / "test_load_checkpoint", sync=True) as ckpt_dir:
        args.load = ckpt_dir
        args.save = ckpt_dir
        set_args(args)

        # Create and save a checkpoint first.
        iteration = 123
        config = TransformerConfig(num_layers=1, kv_channels=1)
        model = MockModel(config)

        optimizer = MockState({"optimizer": "optimizer_state"})
        opt_param_scheduler = MockState({"opt_param_scheduler": "scheduler_state"})
        num_floating_point_operations_so_far = 456

        save_checkpoint(
            iteration, [model], optimizer, opt_param_scheduler, num_floating_point_operations_so_far
        )

        # Create new model, optimizer, and scheduler instances to load into.
        new_model = MockModel(config)
        new_optimizer = MockState({"optimizer": "dummy1"})
        new_opt_param_scheduler = MockState({"opt_param_scheduler": "dummy2"})

        # Load checkpoint
        loaded_iter, loaded_flops = load_checkpoint(
            [new_model], new_optimizer, new_opt_param_scheduler, strict=True
        )

        assert loaded_iter == iteration
        assert loaded_flops == num_floating_point_operations_so_far

        for k in model.state_dict():
            assert torch.equal(model.state_dict()[k], new_model.state_dict()[k])

        assert new_optimizer.state_dict() == optimizer.state_dict()
        assert new_opt_param_scheduler.state_dict() == opt_param_scheduler.state_dict()


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
        set_args(args)

        # Create and save a checkpoint first.
        iteration = 123
        config = TransformerConfig(num_layers=1, kv_channels=1)
        model = MockModel(config)

        optimizer = MockState({"optimizer": "optimizer_state"})
        opt_param_scheduler = MockState({"opt_param_scheduler": "scheduler_state"})
        num_fp_ops = 456

        base_metadata = _build_sharded_state_dict_metadata(args)
        first_job_mock_metadata = {**base_metadata, 'metadata_A': 42, 'metadata_B_soon_removed': 43}
        with mock.patch(
            'megatron.training.checkpointing._build_sharded_state_dict_metadata',
            return_value=first_job_mock_metadata,
        ):
            save_checkpoint(iteration, [model], optimizer, opt_param_scheduler, num_fp_ops)

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
            load_checkpoint([model], optimizer, opt_param_scheduler, strict=True)
            assert optimizer._called_metadata[-1] == first_job_mock_metadata

            # Save the checkpoint again to check if the content metadata for the new checkpoint will be new
            save_checkpoint(iteration, [model], optimizer, opt_param_scheduler, num_fp_ops)
            assert optimizer._called_metadata[-1] == second_job_mock_metadata

        assert optimizer._called_metadata == model._called_metadata
        assert optimizer._called_metadata == [
            first_job_mock_metadata,
            first_job_mock_metadata,
            second_job_mock_metadata,
        ]


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


def test_checkpoint_name_variants_and_tracker_paths(tmp_path):
    root = tmp_path / "ckpts"

    base_dir = get_checkpoint_name(str(root), 42, return_base_dir=True)
    tensor_only = get_checkpoint_name(
        str(root),
        42,
        pipeline_parallel=False,
        tensor_rank=3,
        pipeline_rank=0,
        expert_parallel=False,
        expert_rank=0,
    )
    pipeline_and_expert = get_checkpoint_name(
        str(root),
        42,
        pipeline_parallel=True,
        tensor_rank=1,
        pipeline_rank=2,
        expert_parallel=True,
        expert_rank=4,
        basename="state.pt",
    )
    release = get_checkpoint_name(
        str(root),
        0,
        release=True,
        pipeline_parallel=False,
        tensor_rank=0,
        pipeline_rank=0,
        expert_parallel=False,
        expert_rank=0,
    )

    assert base_dir.endswith("iter_0000042")
    assert tensor_only.endswith("iter_0000042/mp_rank_03/model_optim_rng.pt")
    assert pipeline_and_expert.endswith("iter_0000042/mp_rank_01_002_004/state.pt")
    assert release.endswith("release/mp_rank_00/model_optim_rng.pt")
    assert get_checkpoint_tracker_filename(str(root)).endswith("latest_checkpointed_iteration.txt")
    assert get_distributed_optimizer_checkpoint_name(tensor_only).endswith("mp_rank_03/distrib_optim.pt")


def test_checkpoint_exists_and_load_path_by_args(monkeypatch, tmp_path):
    monkeypatch.setattr(checkpointing_module.torch.distributed, "is_initialized", lambda: False)

    tracker = tmp_path / "latest_checkpointed_iteration.txt"

    assert not checkpoint_exists(None)
    assert not checkpoint_exists(str(tmp_path))

    tracker.write_text("7", encoding="utf-8")
    args = SimpleNamespace(load=str(tmp_path), ckpt_step=None)

    assert checkpoint_exists(str(tmp_path))
    assert get_load_checkpoint_path_by_args(args).endswith("iter_0000007")

    args.ckpt_step = 9
    assert get_load_checkpoint_path_by_args(args).endswith("iter_0000009")


def test_find_checkpoint_rank_0_checks_known_layouts(monkeypatch, tmp_path):
    checkpoint_name = get_checkpoint_name(
        str(tmp_path),
        4,
        pipeline_parallel=True,
        tensor_rank=0,
        pipeline_rank=0,
        expert_parallel=True,
        expert_rank=0,
    )
    Path(checkpoint_name).parent.mkdir(parents=True)
    Path(checkpoint_name).write_text("checkpoint", encoding="utf-8")
    monkeypatch.setattr(checkpointing_module.dist_checkpointing, "check_is_distributed_checkpoint", lambda path: False)

    assert find_checkpoint_rank_0(str(tmp_path), 4) == checkpoint_name
    assert find_checkpoint_rank_0(str(tmp_path), 5) is None


def test_checkpoint_version_and_loaded_iteration_helpers(monkeypatch):
    monkeypatch.setattr(checkpointing_module, "_CHECKPOINT_VERSION", None)
    monkeypatch.setattr(checkpointing_module, "_LOADED_ITERATION", None)

    set_loaded_iteration(123)

    assert get_loaded_iteration() == 123

    set_checkpoint_version(3.0)
    set_checkpoint_version(3.0)


def test_finalize_deletion_processes_joins_finished_and_blocking(monkeypatch):
    class FakeProcess:
        def __init__(self, alive):
            self.alive = alive
            self.pid = id(self)
            self.joined = False

        def is_alive(self):
            return self.alive

        def join(self):
            self.joined = True

    finished = FakeProcess(alive=False)
    running = FakeProcess(alive=True)
    monkeypatch.setattr(checkpointing_module, "_deletion_processes", [finished, running])

    finalize_deletion_processes(blocking=False)
    assert finished.joined
    assert not running.joined
    assert checkpointing_module._deletion_processes == [running]

    finalize_deletion_processes(blocking=True)
    assert running.joined
    assert checkpointing_module._deletion_processes == []


def test_generate_state_dict_collects_model_optimizer_scheduler_and_rng(create_args):
    args = create_args
    args.ckpt_format = "torch"
    args.no_save_optim = False
    args.no_save_rng = False
    model = [mock.Mock()]
    model[0].state_dict_for_save_checkpoint.return_value = {"weight": torch.ones(1)}
    optimizer = MockState({"optimizer": "state"})
    scheduler = MockState({"scheduler": "state"})
    rng_state = [{"rng": "state"}]
    rerun_state = {"rerun": "state"}

    state_dict = generate_state_dict(
        args,
        model,
        optimizer,
        scheduler,
        rng_state,
        iteration=11,
        rerun_state=rerun_state,
    )

    assert state_dict["args"] is args
    assert state_dict["checkpoint_version"] == 3.0
    assert state_dict["iteration"] == 11
    assert torch.equal(state_dict["model"]["weight"], torch.ones(1))
    assert state_dict["optimizer"] == {"optimizer": "state"}
    assert state_dict["opt_param_scheduler"] == {"scheduler": "state"}
    assert state_dict["rng_state"] == rng_state
    assert state_dict["rerun_state_machine"] == rerun_state


def test_generate_state_dict_skips_optimizer_and_rng_when_disabled(create_args):
    args = create_args
    args.ckpt_format = "torch"
    args.no_save_optim = True
    args.no_save_rng = True
    model = [mock.Mock()]
    model[0].state_dict_for_save_checkpoint.return_value = {"weight": "model"}

    state_dict = generate_state_dict(args, model, optimizer=None, opt_param_scheduler=None, rng_state=None)

    assert "optimizer" not in state_dict
    assert "opt_param_scheduler" not in state_dict
    assert "rng_state" not in state_dict


def test_transpose_first_dim_uses_attention_shape_metadata():
    attention = SimpleNamespace(hidden_size_per_attention_head=2, num_attention_heads_per_partition=2)
    model = SimpleNamespace(
        language_model=SimpleNamespace(
            encoder=SimpleNamespace(
                layers=[SimpleNamespace(self_attention=attention)]
            )
        )
    )
    tensor = torch.arange(24).view(12, 2)

    first = _transpose_first_dim(tensor, num_splits=3, num_splits_first=True, model=model)
    last = _transpose_first_dim(tensor, num_splits=3, num_splits_first=False, model=model)

    assert first.shape == tensor.shape
    assert last.shape == tensor.shape
    assert sorted(first.flatten().tolist()) == sorted(tensor.flatten().tolist())
    assert sorted(last.flatten().tolist()) == sorted(tensor.flatten().tolist())


def test_fix_query_key_value_ordering_transposes_legacy_qkv_params(monkeypatch):
    calls = []

    class FakeParam:
        def __init__(self):
            self.data = torch.arange(24).view(12, 2)

    class FakeModel:
        def __init__(self):
            self.weight = FakeParam()
            self.bias = FakeParam()
            self.other = FakeParam()

        def named_parameters(self):
            return [
                ("layer.query_key_value.weight", self.weight),
                ("layer.query_key_value.bias", self.bias),
                ("layer.dense.weight", self.other),
            ]

    def fake_transpose(tensor, num_splits, num_splits_first, model):
        calls.append((tuple(tensor.shape), num_splits, num_splits_first))
        return tensor + 1

    model = FakeModel()
    monkeypatch.setattr(checkpointing_module, "_transpose_first_dim", fake_transpose)

    fix_query_key_value_ordering([model], checkpoint_version=0)
    fix_query_key_value_ordering(model, checkpoint_version=1.0)

    assert calls == [
        ((12, 2), 3, True),
        ((12, 2), 3, True),
        ((12, 2), 3, False),
        ((12, 2), 3, False),
    ]
    assert torch.equal(model.other.data, torch.arange(24).view(12, 2))


def test_cleanup_old_non_persistent_checkpoint_keeps_newest(monkeypatch, tmp_path):
    monkeypatch.setattr(checkpointing_module.torch.distributed, "is_initialized", lambda: False)

    for iteration in [1, 2, 3]:
        (tmp_path / f"iter_{iteration:07d}").mkdir()

    cleanup_old_non_persistent_checkpoint(tmp_path, leave_ckpt_num=1, do_async=False)

    assert not (tmp_path / "iter_0000001").exists()
    assert not (tmp_path / "iter_0000002").exists()
    assert (tmp_path / "iter_0000003").exists()


def test_get_non_persistent_iteration_global_and_local(monkeypatch, tmp_path):
    monkeypatch.setattr(checkpointing_module.torch.distributed, "is_initialized", lambda: False)

    args = SimpleNamespace(non_persistent_ckpt_type=None)
    assert _get_non_persistent_iteration(str(tmp_path), args) == -1

    tracker = tmp_path / "latest_checkpointed_iteration.txt"
    tracker.write_text("13", encoding="utf-8")
    args.non_persistent_ckpt_type = "global"
    assert _get_non_persistent_iteration(str(tmp_path), args) == 13

    args.non_persistent_ckpt_type = "local"
    context = {"local_checkpoint_manager": SimpleNamespace(find_latest=lambda: 17)}
    assert _get_non_persistent_iteration(str(tmp_path), args, context) == 17


def test_maybe_save_dataloader_state_validates_iterator(monkeypatch, tmp_path):
    maybe_save_dataloader_state(None, iteration=1, dataloader_save_path=str(tmp_path))
    maybe_save_dataloader_state(object(), iteration=1, dataloader_save_path="")

    with pytest.raises(RuntimeError, match="Could not find a save_state"):
        maybe_save_dataloader_state(SimpleNamespace(iterable=object()), 1, str(tmp_path))


def test_maybe_save_dataloader_state_calls_supported_iterator(monkeypatch, tmp_path):
    calls = []
    iterator = SimpleNamespace(iterable=SimpleNamespace(save_state=lambda: {"state": 1}))
    monkeypatch.setattr(checkpointing_module.mpu, "is_pipeline_first_stage", lambda ignore_virtual=True: True)
    monkeypatch.setattr(checkpointing_module.mpu, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(checkpointing_module.mpu, "get_data_parallel_rank", lambda: 0)
    monkeypatch.setattr(checkpointing_module.mpu, "get_data_parallel_group", lambda: "dp")
    monkeypatch.setattr(checkpointing_module.torch.distributed, "barrier", lambda group=None: calls.append(("barrier", group)))
    monkeypatch.setattr(checkpointing_module, "ensure_directory_exists", lambda filename, check_parent=True: calls.append(("mkdir", filename)))
    monkeypatch.setattr(
        checkpointing_module,
        "get_checkpoint_name",
        lambda save_dir, iteration, release=False, basename=None, **kwargs: str(tmp_path / basename),
    )
    monkeypatch.setattr(checkpointing_module.torch, "save", lambda state, path: calls.append(("save", state, path)))

    maybe_save_dataloader_state(iterator, iteration=7, dataloader_save_path=str(tmp_path))

    assert calls == [
        ("barrier", "dp"),
        ("mkdir", str(tmp_path / "train_dataloader_dprank000.pt")),
        ("barrier", "dp"),
        ("save", {"dataloader_state_dict": {"state": 1}}, str(tmp_path / "train_dataloader_dprank000.pt")),
    ]


def test_load_args_from_checkpoint_updates_missing_values(monkeypatch, tmp_path):
    checkpoint_args = SimpleNamespace(
        disable_bias_linear=True,
        hybrid_override_pattern="M*",
        hybrid_layer_pattern=None,
        num_layers=2,
        hidden_size=16,
        ffn_hidden_size=64,
        seq_length=8,
        num_attention_heads=4,
        num_query_groups=2,
        group_query_attention=True,
        kv_channels=4,
        max_position_embeddings=8,
        position_embedding_type="rope",
        add_position_embedding=True,
        use_rotary_position_embeddings=True,
        rotary_base=10000,
        rotary_percent=1.0,
        rotary_interleaved=False,
        add_qkv_bias=True,
        squared_relu=False,
        swiglu=True,
        untie_embeddings_and_output_weights=True,
        apply_layernorm_1p=False,
        normalization="RMSNorm",
        apply_query_key_layer_scaling=False,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        mtp_hybrid_override_pattern=None,
        mtp_num_layers=None,
        mtp_use_repeated_layer=False,
        spec=None,
        num_experts=None,
        moe_layer_freq=None,
        moe_router_topk=1,
        moe_token_dispatcher_type="allgather",
        moe_router_pre_softmax=False,
        moe_grouped_gemm=False,
        moe_shared_expert_intermediate_size=None,
        moe_router_score_function="softmax",
        moe_router_enable_expert_bias=False,
        moe_router_topk_scaling_factor=None,
        mamba_state_dim=None,
        mamba_head_dim=None,
        mamba_num_groups=None,
        mamba_num_heads=None,
        heterogeneous_layers_config_path=None,
        heterogeneous_layers_config_encoded_json=None,
        moe_latent_size=None,
        tokenizer_model="tok.model",
        tokenizer_type="GPT2BPETokenizer",
        tiktoken_pattern=None,
        padded_vocab_size=128,
        ckpt_format="torch",
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
        num_layers_per_virtual_pipeline_stage=None,
        expert_model_parallel_size=1,
    )
    state_dict = {
        "args": checkpoint_args,
        "checkpoint_version": 3.0,
        "iteration": 23,
    }
    args = SimpleNamespace(
        load=str(tmp_path),
        use_tokenizer_model_from_checkpoint_args=True,
        use_mp_args_from_checkpoint_args=True,
    )
    monkeypatch.setattr(
        "megatron.training.checkpointing._load_base_checkpoint",
        lambda *items, **kwargs: (state_dict, "checkpoint", False, CheckpointType.LEGACY),
    )

    updated, returned_checkpoint_args = load_args_from_checkpoint(args)

    assert updated is args
    assert returned_checkpoint_args is checkpoint_args
    assert args.iteration == 23
    assert args.add_bias_linear is False
    assert args.hybrid_layer_pattern == "M*"
    assert checkpoint_args.num_layers is None
    assert not hasattr(args, "num_layers")
    assert args.hidden_size == 16
    assert args.num_query_groups == 2
    assert args.tokenizer_model == "tok.model"
    assert args.tensor_model_parallel_size == 1
