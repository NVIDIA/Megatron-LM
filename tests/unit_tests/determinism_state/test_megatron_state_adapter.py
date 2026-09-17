# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU checks of local-state extraction and observation contracts, not GPU proof."""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tools.determinism.megatron_state import (
    capture_adam_state,
    capture_mock_loader_state,
    capture_model,
)
from tools.determinism.megatron_state_worker import TrainingCapture, recipe_arguments
from tools.determinism.training_state import (
    UnverifiedState,
    read_checkpoint_record,
    record_checkpoint_directory,
)


def trained_adam():
    model = torch.nn.Linear(2, 2)
    inner = torch.optim.AdamW(model.parameters(), lr=0.01)
    model(torch.ones(1, 2)).square().sum().backward()
    inner.step()
    wrapper = SimpleNamespace(
        optimizer=inner,
        grad_scaler=None,
        get_loss_scale=lambda: torch.tensor([1.0]),
        state_dict=lambda: pytest.fail("Outer distributed-optimizer state omits moments"),
    )
    return model, wrapper


def test_optimizer_includes_both_moments_and_live_master_parameters():
    _, wrapper = trained_adam()
    state, precision = capture_adam_state(wrapper)
    for identifier, parameter in zip(
        state["state"]["param_groups"][0]["params"], wrapper.optimizer.param_groups[0]["params"]
    ):
        assert torch.equal(
            state["state"]["state"][identifier]["exp_avg"],
            wrapper.optimizer.state[parameter]["exp_avg"],
        )
        assert torch.equal(
            state["state"]["state"][identifier]["exp_avg_sq"],
            wrapper.optimizer.state[parameter]["exp_avg_sq"],
        )
    assert (
        precision["master_parameters"][0][0]["value"]
        is wrapper.optimizer.param_groups[0]["params"][0]
    )
    assert (
        precision["master_parameters"][0][0]["grad"]
        is wrapper.optimizer.param_groups[0]["params"][0].grad
    )
    assert precision["grad_scaler"] == "disabled"


@pytest.mark.parametrize("missing", ["exp_avg", "exp_avg_sq", "parameter"])
def test_incomplete_optimizer_state_is_rejected(missing):
    _, wrapper = trained_adam()
    parameter = wrapper.optimizer.param_groups[0]["params"][0]
    if missing == "parameter":
        del wrapper.optimizer.state[parameter]
    else:
        del wrapper.optimizer.state[parameter][missing]
    with pytest.raises(UnverifiedState, match="missing"):
        capture_adam_state(wrapper)


def test_all_model_chunks_and_main_gradients_are_captured():
    chunks = [torch.nn.Linear(2, 2), torch.nn.Linear(2, 1)]
    for chunk in chunks:
        for parameter in chunk.parameters():
            parameter.main_grad = torch.full_like(parameter, 3.0)
    model, gradients = capture_model(chunks)
    assert set(model) == set(gradients) == {"0", "1"}
    assert torch.equal(gradients["1"]["weight"]["main_grad"], chunks[1].weight.main_grad)
    assert gradients["0"]["weight"]["grad"] is None


def sampler_class():
    # Execute the real sampler class without importing the CUDA-only MCore package.
    source = Path(__file__).resolve().parents[3] / "megatron/training/datasets/data_samplers.py"
    node = next(
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "MegatronPretrainingSampler"
    )
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["MegatronPretrainingSampler"]


class DatasetFixture(torch.utils.data.Dataset):
    """Explicit CPU fixture for the extraction helper, not a claimed MockGPT run."""

    def __init__(self):
        self.unique_description = "CPU contract fixture"
        self.indices = self.document_index = self.shuffle_index = np.arange(32)
        self.sample_index = np.arange(64).reshape(32, 2)
        self.dataset = SimpleNamespace(
            sequence_lengths=np.ones(32, dtype=np.int32), vocab_size=8, eod_token=7
        )
        self.masks_and_position_ids_are_cacheable = True
        self.masks_and_position_ids_are_cached = True
        self.cached_attention_mask = None
        self.cached_loss_mask = torch.ones(2)
        self.cached_position_ids = torch.arange(2)

    def __len__(self):
        return 32

    def __getitem__(self, index):
        return torch.tensor([index])


def make_loader(consumed):
    sampler = sampler_class()(32, consumed, 1, 1, 2)
    loader = torch.utils.data.DataLoader(
        DatasetFixture(),
        batch_sampler=sampler,
        num_workers=0,
        generator=torch.Generator().manual_seed(123),
    )
    iterator = SimpleNamespace(
        iterable=iter(loader), replaying=False, replay_pos=0, saved_microbatches=[]
    )
    return loader, iterator


def test_sampler_resume_cursor_matches_actual_yields_and_next_batch():
    continuous, a = make_loader(0)
    resumed, b = make_loader(4)
    for _ in range(3):
        next(a.iterable)
    next(b.iterable)
    left = capture_mock_loader_state(continuous, a, consumed_samples=6)
    right = capture_mock_loader_state(resumed, b, consumed_samples=6)
    assert left["next_global_sample"] == right["next_global_sample"] == 6
    assert left["sampler"] == right["sampler"]
    assert torch.equal(left["loader_generator"], right["loader_generator"])
    assert left["iterator_base_seed"] == right["iterator_base_seed"]
    assert torch.equal(next(a.iterable), next(b.iterable))
    with pytest.raises(UnverifiedState, match="counters disagree"):
        capture_mock_loader_state(resumed, b, consumed_samples=6)


def test_loader_rejects_buffered_replay_state():
    loader, iterator = make_loader(0)
    iterator.saved_microbatches = [torch.ones(1)]
    with pytest.raises(UnverifiedState, match="Buffered reruns"):
        capture_mock_loader_state(loader, iterator, consumed_samples=0)


def checkpoint_tree(tmp_path):
    root = tmp_path / "reference"
    checkpoint = root / "megatron-checkpoints/iter_0000002"
    checkpoint.mkdir(parents=True)
    (checkpoint / "common.pt").write_bytes(b"common-state")
    (checkpoint / "__0.distcp").write_bytes(b"tensor-state")
    (checkpoint / ".metadata").write_bytes(b"metadata")
    record = record_checkpoint_directory(
        root, checkpoint_directory=checkpoint, step=2, rank=0, run_id="reference-run"
    )
    return root, checkpoint, record


@pytest.mark.parametrize("change", ["modify", "remove", "add", "symlink"])
def test_real_checkpoint_directory_changes_invalidate_identity(tmp_path, change):
    root, checkpoint, record = checkpoint_tree(tmp_path)
    assert read_checkpoint_record(root, step=2, rank=0) == record
    path = checkpoint / "__0.distcp"
    if change == "modify":
        path.write_bytes(b"different")
    elif change == "remove":
        path.unlink()
    elif change == "add":
        (checkpoint / "another.distcp").write_bytes(b"new")
    else:
        path.unlink()
        path.symlink_to(checkpoint / "common.pt")
    with pytest.raises(UnverifiedState):
        read_checkpoint_record(root, step=2, rank=0)


def make_capture(monkeypatch, tmp_path):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "4")
    root, _, record = checkpoint_tree(tmp_path)
    args = SimpleNamespace(
        resume=root,
        checkpoint_step=2,
        steps=4,
        stop_step=None,
        pipeline_size=1,
        omit_restore="rng",
        output=tmp_path / "resume",
        run_id="resume-run",
    )
    training_args = SimpleNamespace(load=str(root / "megatron-checkpoints"), no_load_rng=False)
    training = SimpleNamespace(get_args=lambda: training_args)
    return TrainingCapture(args, training, None), training_args, record


def test_omitted_rng_control_uses_actual_load_flag_and_restores_it(monkeypatch, tmp_path):
    capture, args, record = make_capture(monkeypatch, tmp_path)

    def load():
        assert args.no_load_rng
        return 2, 0.0

    assert capture.wrap_load(load)() == (2, 0.0)
    assert capture.resumed == record
    assert args.no_load_rng is False


def test_wrong_restored_iteration_cannot_produce_resume_evidence(monkeypatch, tmp_path):
    capture, args, _ = make_capture(monkeypatch, tmp_path)
    with pytest.raises(UnverifiedState, match="requested iteration"):
        capture.wrap_load(lambda: (4, 0.0))()
    assert capture.resumed is None
    assert args.no_load_rng is False


def test_failed_load_restores_flag_and_never_records_success(monkeypatch, tmp_path):
    capture, args, _ = make_capture(monkeypatch, tmp_path)

    def load():
        raise RuntimeError("failed restore")

    with pytest.raises(RuntimeError, match="failed restore"):
        capture.wrap_load(load)()
    assert capture.resumed is None
    assert args.no_load_rng is False


def test_post_step_observer_preserves_callback_order_and_return(monkeypatch, tmp_path):
    capture, _, _ = make_capture(monkeypatch, tmp_path)
    events = []
    monkeypatch.setattr(capture, "capture", lambda *args: events.append(("capture", args)))

    def callback(*args):
        events.append(("callback", args))
        return 17

    assert capture.wrap_post_step(callback)("model", "optimizer", "scheduler", 3, "prof") == 17
    assert events == [
        ("callback", ("model", "optimizer", "scheduler", 3, "prof")),
        ("capture", ("model", "optimizer", "scheduler", 3)),
    ]


def test_incomplete_training_cannot_publish_completion(monkeypatch, tmp_path):
    capture, _, _ = make_capture(monkeypatch, tmp_path)
    capture.captured_steps = [3, 4]
    with pytest.raises(UnverifiedState, match="Missing training completion"):
        capture.finish()
    assert not capture.args.output.exists()


def test_stop_point_keeps_original_training_and_schedule_horizon():
    full = recipe_arguments(4, 5)
    stopped = recipe_arguments(4, 5, 3)
    assert stopped == [*full, "--exit-interval", "3"]
    assert stopped[stopped.index("--train-iters") + 1] == "5"
    assert stopped[stopped.index("--lr-decay-iters") + 1] == "5"
    assert stopped[stopped.index("--eval-interval") + 1] == "6"


def test_stop_point_runs_original_callbacks_without_earlier_capture(monkeypatch, tmp_path):
    capture, _, _ = make_capture(monkeypatch, tmp_path)
    capture.args.stop_step = 3
    events = []

    def callback(model, optimizer, scheduler, iteration):
        events.append(("original", iteration))
        return iteration

    monkeypatch.setattr(capture, "capture", lambda *args: events.append(("capture", args[-1])))
    wrapped = capture.wrap_post_step(callback)
    for step in (1, 2, 3):
        assert wrapped(None, None, None, step) == step
    assert events == [("original", 1), ("original", 2), ("original", 3), ("capture", 3)]


@pytest.mark.parametrize("step,decision", [(2, True), (3, False), (4, True)])
def test_wrong_exit_decision_cannot_complete_stop_point(monkeypatch, tmp_path, step, decision):
    capture, _, _ = make_capture(monkeypatch, tmp_path)
    capture.args.stop_step = 3

    def decide(iteration):
        return decision

    with pytest.raises(UnverifiedState, match="exit decision"):
        capture.wrap_decide_exit(decide)(iteration=step)
    assert not capture.stop_requested


@pytest.mark.parametrize("failure", [None, "nonzero_exit", "no_decision", "no_capture", "returned"])
def test_only_successful_target_exit_completes_training(monkeypatch, tmp_path, failure):
    capture, _, record = make_capture(monkeypatch, tmp_path)
    capture.args.stop_step = 3
    capture.resumed = record
    monkeypatch.setattr(capture, "validate_configuration", lambda: None)
    monkeypatch.setattr(capture, "runtime_provenance", lambda: {"fixture": True})
    error = SystemExit(1 if failure == "nonzero_exit" else 0)

    def train(train_data_iterator):
        if failure != "no_capture":
            capture.captured_steps = [3]
        if failure != "no_decision":
            assert capture.wrap_decide_exit(lambda iteration: True)(3)
        if failure == "returned":
            return 3, 0.0
        raise error

    if failure is None:
        with pytest.raises(SystemExit) as raised:
            capture.wrap_train(train)("live-iterator")
        assert raised.value is error is capture.expected_exit
        assert capture.train_completed
        assert capture.iterator == "live-iterator"
    else:
        with pytest.raises(UnverifiedState):
            capture.wrap_train(train)("live-iterator")
        assert not capture.train_completed
        assert capture.expected_exit is None


@pytest.mark.parametrize("world_size", [4, 8])
@pytest.mark.parametrize("pipeline_size", [1, 2])
def test_pipeline_recipe_preserves_training_horizon_and_global_batch(world_size, pipeline_size):
    command = recipe_arguments(world_size, 5, 3, pipeline_size=pipeline_size)
    for option, value in (
        ("--tensor-model-parallel-size", "2"),
        ("--pipeline-model-parallel-size", str(pipeline_size)),
        ("--global-batch-size", str(world_size)),
        ("--train-iters", "5"),
        ("--lr-decay-iters", "5"),
        ("--exit-interval", "3"),
    ):
        assert command[command.index(option) + 1] == value


@pytest.mark.parametrize("change", [None, "wrong_pp", "vpp", "p2p_overlap", "deferred_wgrad"])
def test_pipeline_configuration_rejects_uncovered_variants(monkeypatch, tmp_path, change):
    capture, args, _ = make_capture(monkeypatch, tmp_path)
    capture.args.pipeline_size = 2
    args.__dict__.update(
        bf16=True,
        use_distributed_optimizer=True,
        deterministic_mode=True,
        ckpt_format="torch_dist",
        dataloader_type="single",
        num_workers=0,
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=2,
        context_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
    )
    if change == "wrong_pp":
        args.pipeline_model_parallel_size = 1
    elif change == "vpp":
        args.virtual_pipeline_model_parallel_size = 2
    elif change == "p2p_overlap":
        args.overlap_p2p_comm = True
    elif change == "deferred_wgrad":
        args.defer_embedding_wgrad_compute = True
    if change is None:
        capture.validate_configuration()
    else:
        with pytest.raises(UnverifiedState):
            capture.validate_configuration()


@pytest.mark.parametrize("pipeline_size,pipeline_rank", [(1, 0), (2, 0), (2, 1)])
@pytest.mark.parametrize("change", [None, "endpoint", "missing_layer", "extra_chunk"])
def test_pipeline_partition_requires_its_layer_and_endpoint(
    monkeypatch, tmp_path, pipeline_size, pipeline_rank, change
):
    capture, _, _ = make_capture(monkeypatch, tmp_path)
    capture.args.pipeline_size = pipeline_size
    capture.provenance = {"rank_layout": {"PP": pipeline_rank}}
    chunk = SimpleNamespace(
        pre_process=pipeline_rank == 0,
        post_process=pipeline_rank == pipeline_size - 1,
        decoder=SimpleNamespace(layers=[object() for _ in range(2 // pipeline_size)]),
    )
    chunks = [chunk]
    if change == "endpoint":
        chunk.pre_process = not chunk.pre_process
    elif change == "missing_layer":
        chunk.decoder.layers = []
    elif change == "extra_chunk":
        chunks.append(chunk)
    if change is None:
        capture.validate_partition(chunks)
    else:
        with pytest.raises(UnverifiedState, match="partition"):
            capture.validate_partition(chunks)
