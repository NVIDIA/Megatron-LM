# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for chunked optimizer state and master-weight offload."""

import dataclasses
import logging
import sys
from contextlib import nullcontext
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import torch.nn as nn

from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.optimizer import (
    ChainedOptimizer,
    FP32Optimizer,
    OptimizerConfig,
    get_megatron_optimizer,
)
from megatron.core.optimizer.cpu_offloading.chunked_optimizer_state_offload import (
    ChunkedOptimizerStateOffloader,
)
from megatron.core.transformer import TransformerConfig
from megatron.core.utils import is_te_min_version
from megatron.training.arguments import parse_args, validate_args
from tests.unit_tests.test_utilities import Utils

try:
    from transformer_engine.pytorch.optimizers import FusedAdam  # noqa: F401

    TE_FUSED_ADAM_AVAILABLE = True
except ImportError:
    TE_FUSED_ADAM_AVAILABLE = False


class SimpleModel(nn.Module):
    """Small model with enough parameters to exercise multiple state chunks."""

    def __init__(self, hidden_size: int = 256, num_layers: int = 2) -> None:
        super().__init__()
        assert num_layers >= 2
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.hidden_layers = nn.ModuleList(
            nn.Linear(hidden_size, hidden_size) for _ in range(num_layers - 2)
        )
        self.fc2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run the test MLP."""

        output = torch.relu(self.fc1(inputs))
        for layer in self.hidden_layers:
            output = torch.relu(layer(output))
        return self.fc2(output)


@pytest.fixture
def model_parallel_context():
    """Always tear down model-parallel state, including on assertion failure or skip."""

    try:
        Utils.initialize_model_parallel()
        yield
    finally:
        Utils.destroy_model_parallel()


def test_legacy_flag_warns_and_enables_chunked_offload():
    """OptimizerConfig retains and normalizes the deprecated compatibility field."""

    with pytest.warns(FutureWarning, match='offload_optimizer_states is deprecated'):
        config = OptimizerConfig(
            optimizer='adam', use_distributed_optimizer=True, offload_optimizer_states=True
        )

    assert not config.offload_optimizer_states
    assert config.chunked_optimizer_state_offload

    with mock.patch("megatron.core.optimizer.optimizer_config.warnings.warn") as warn:
        copied_config = dataclasses.replace(config)

    warn.assert_not_called()
    assert not copied_config.offload_optimizer_states
    assert copied_config.chunked_optimizer_state_offload


def test_legacy_flag_uses_user_facing_optimizer_error():
    """An unsupported legacy invocation names the deprecated CLI alias in its failure."""

    with (
        pytest.warns(FutureWarning, match='offload_optimizer_states is deprecated'),
        pytest.raises(AssertionError, match='deprecated --offload-optimizer-states alias'),
    ):
        OptimizerConfig(optimizer='sgd', offload_optimizer_states=True)


def test_chunked_optimizer_state_offload_config_accepts_supported_optimizers():
    """OptimizerConfig accepts the supported Adam and compact LayerWise Muon modes."""

    OptimizerConfig(
        optimizer='adam', use_distributed_optimizer=True, chunked_optimizer_state_offload=True
    )
    OptimizerConfig(
        optimizer='muon',
        bf16=True,
        use_layer_wise_distributed_optimizer=True,
        use_layer_wise_param_layout=False,
        chunked_optimizer_state_offload=True,
    )


def test_chunked_optimizer_state_offload_fraction_zero_is_a_noop():
    """The feature flag may remain set while a zero fraction disables mode restrictions."""

    config = OptimizerConfig(
        optimizer='sgd', chunked_optimizer_state_offload=True, optimizer_state_offload_fraction=0.0
    )

    assert config.chunked_optimizer_state_offload
    assert config.optimizer_state_offload_fraction == 0.0


@pytest.mark.parametrize(
    ('overrides', 'error'),
    [
        ({'optimizer': 'sgd'}, 'currently supports Adam and Muon'),
        ({'use_distributed_optimizer': False}, 'requires use_distributed_optimizer'),
        (
            {'optimizer': 'muon', 'use_layer_wise_distributed_optimizer': True},
            'currently requires bf16',
        ),
        (
            {'use_layer_wise_distributed_optimizer': True},
            'Adam optimizer state offload does not support',
        ),
        ({'optimizer': 'muon', 'bf16': True}, 'requires LayerWiseDistributedOptimizer'),
        (
            {
                'optimizer': 'muon',
                'bf16': True,
                'use_layer_wise_distributed_optimizer': True,
                'use_layer_wise_param_layout': True,
            },
            '--use-layer-wise-param-layout to be disabled',
        ),
        ({'optimizer_cpu_offload': True}, 'optimizer_cpu_offload are mutually exclusive'),
        ({'optimizer_state_offload_chunk_size_mb': -1}, 'must be non-negative'),
        ({'optimizer_state_offload_fraction': 1.1}, 'must be in'),
        ({'optimizer_cuda_graph': True}, 'does not support optimizer CUDA graphs'),
    ],
)
def test_chunked_optimizer_state_offload_config_rejects_unsupported_modes(overrides, error):
    """Optimizer-owned offload constraints are enforced by OptimizerConfig."""

    config_kwargs = {
        'optimizer': 'adam',
        'use_distributed_optimizer': True,
        'chunked_optimizer_state_offload': True,
    }
    config_kwargs.update(overrides)
    with pytest.raises(AssertionError, match=error):
        OptimizerConfig(**config_kwargs)


@pytest.mark.parametrize(
    ("flag_name", "chunk_size_mb", "expects_unbounded_window_warning"),
    [("chunked_optimizer_state_offload", 1, False), ("offload_optimizer_states", 0, True)],
)
def test_training_args_reject_async_save_with_chunked_state_offload(
    monkeypatch, flag_name, chunk_size_mb, expects_unbounded_window_warning
):
    """Reusable canonical CPU buffers cannot be handed to a background checkpoint writer."""

    monkeypatch.setattr(sys, 'argv', ['test_optimizer_state_offloading.py'])
    args = parse_args()
    args.num_layers = 2
    args.vocab_size = 256
    args.hidden_size = 64
    args.num_attention_heads = 4
    args.max_position_embeddings = 128
    args.seq_length = 128
    args.micro_batch_size = 1
    args.train_iters = 1
    args.lr = 1e-4
    args.tokenizer_type = 'NullTokenizer'
    args.optimizer = 'adam'
    args.use_distributed_optimizer = True
    setattr(args, flag_name, True)
    args.optimizer_state_offload_chunk_size_mb = chunk_size_mb
    args.ckpt_format = 'torch_dist'
    args.async_save = True
    args.use_persistent_ckpt_worker = True

    with mock.patch("megatron.training.arguments.warn_rank_0") as warn:
        with pytest.raises(AssertionError, match="does not support --async-save"):
            validate_args(args)

    warning_messages = [call.args[0] for call in warn.call_args_list if call.args]
    assert any("enabled with chunk size 0" in message for message in warning_messages) is (
        expects_unbounded_window_warning
    )


def test_training_args_treat_zero_offload_fraction_as_disabled(monkeypatch):
    """A zero fraction bypasses checkpoint restrictions during startup validation."""

    monkeypatch.setattr(sys, 'argv', ['test_optimizer_state_offloading.py'])
    args = parse_args()
    args.num_layers = 2
    args.vocab_size = 256
    args.hidden_size = 64
    args.num_attention_heads = 4
    args.max_position_embeddings = 128
    args.seq_length = 128
    args.micro_batch_size = 1
    args.train_iters = 1
    args.lr = 1e-4
    args.tokenizer_type = 'NullTokenizer'
    args.optimizer = 'adam'
    args.use_distributed_optimizer = True
    args.chunked_optimizer_state_offload = True
    args.optimizer_state_offload_fraction = 0.0
    args.ckpt_format = 'torch'
    args.use_dist_ckpt = False
    args.async_save = True
    args.use_persistent_ckpt_worker = True

    validated_args = validate_args(args)

    assert validated_args.async_save


def test_training_args_reject_muon_offload_without_distributed_optimizer(monkeypatch):
    """Muon offload must route non-Muon groups to a sibling DistributedOptimizer."""

    monkeypatch.setattr(sys, 'argv', ['test_optimizer_state_offloading.py'])
    args = parse_args()
    args.num_layers = 2
    args.vocab_size = 256
    args.hidden_size = 64
    args.num_attention_heads = 4
    args.max_position_embeddings = 128
    args.seq_length = 128
    args.micro_batch_size = 1
    args.train_iters = 1
    args.lr = 1e-4
    args.tokenizer_type = 'NullTokenizer'
    args.optimizer = 'muon'
    args.use_distributed_optimizer = False
    args.chunked_optimizer_state_offload = True
    args.ckpt_format = 'torch_dist'

    with pytest.raises(AssertionError, match="requires the LayerWise distributed optimizer"):
        validate_args(args)


def test_training_args_accept_deprecated_dist_muon_offload(monkeypatch):
    """The deprecated dist_muon spelling normalizes to the supported LayerWise mode."""

    monkeypatch.setattr(sys, 'argv', ['test_optimizer_state_offloading.py'])
    args = parse_args()
    args.num_layers = 2
    args.vocab_size = 256
    args.hidden_size = 64
    args.num_attention_heads = 4
    args.max_position_embeddings = 128
    args.seq_length = 128
    args.micro_batch_size = 1
    args.train_iters = 1
    args.lr = 1e-4
    args.tokenizer_type = 'NullTokenizer'
    args.optimizer = 'dist_muon'
    args.use_distributed_optimizer = False
    args.chunked_optimizer_state_offload = True
    args.ckpt_format = 'torch_dist'

    validated_args = validate_args(args)

    assert validated_args.optimizer == 'muon'
    assert validated_args.use_layer_wise_distributed_optimizer
    assert not validated_args.use_distributed_optimizer


@pytest.mark.parametrize(
    ("ckpt_format", "async_save", "no_load_optim"),
    [("torch", False, True), ("torch_dist", True, False)],
)
def test_training_args_narrow_checkpoint_restrictions_to_optimizer_io(
    monkeypatch, ckpt_format, async_save, no_load_optim
):
    """Model-only saves skip offload restrictions not needed by optimizer checkpoint I/O."""

    monkeypatch.setattr(sys, 'argv', ['test_optimizer_state_offloading.py'])
    args = parse_args()
    args.num_layers = 2
    args.vocab_size = 256
    args.hidden_size = 64
    args.num_attention_heads = 4
    args.max_position_embeddings = 128
    args.seq_length = 128
    args.micro_batch_size = 1
    args.train_iters = 1
    args.lr = 1e-4
    args.tokenizer_type = 'NullTokenizer'
    args.optimizer = 'adam'
    args.use_distributed_optimizer = True
    args.chunked_optimizer_state_offload = True
    args.ckpt_format = ckpt_format
    args.use_dist_ckpt = ckpt_format == "torch_dist"
    args.async_save = async_save
    args.use_persistent_ckpt_worker = True
    args.no_save_optim = True
    args.no_load_optim = no_load_optim

    validated_args = validate_args(args)

    assert validated_args.ckpt_format == ckpt_format
    assert validated_args.async_save is async_save


def test_standard_chained_optimizer_shares_state_offload_streams():
    """Ordinary chained Adam-style siblings share transfer ordering, not only LayerWise."""

    class FakeManager:
        def __init__(self):
            self.transfer_streams = (object(), object())

        def use_transfer_streams(self, d2h_stream, h2d_stream):
            self.transfer_streams = (d2h_stream, h2d_stream)

    class FakeOptimizer:
        def __init__(self, config, manager):
            self.config = config
            self._optimizer_state_offloader = manager

    config = object()
    managers = [FakeManager(), FakeManager()]
    ChainedOptimizer([FakeOptimizer(config, manager) for manager in managers])

    assert managers[1].transfer_streams[0] is managers[0].transfer_streams[0]
    assert managers[1].transfer_streams[1] is managers[0].transfer_streams[1]


def test_chained_optimizer_forwards_master_residency_assertion():
    """Chain assertions reach every child while inherited lifecycle defaults remain safe."""

    calls = []

    class FakeOptimizer:
        def __init__(self, config, name):
            self.config = config

            self.name = name

        def assert_master_weights_resident(self, operation):
            calls.append((self.name, operation))

    config = object()
    optimizer = ChainedOptimizer(
        [FakeOptimizer(config, 'layerwise'), FakeOptimizer(config, 'distributed')]
    )

    assert optimizer._optimizer_state_offloader is None
    optimizer.assert_master_weights_resident("test")
    assert calls == [('layerwise', 'test'), ('distributed', 'test')]
    optimizer.set_optimizer_state_offload_deferred_lifecycle(
        state_prefetch_to_step=False, master_offload_for_param_sync=False
    )
    assert not optimizer._defer_optimizer_state_prefetch_to_step
    assert not optimizer._defer_optimizer_master_offload_for_param_sync


def test_chained_master_restore_targets_only_pre_forward_sync_children():
    """A compact Muon pre-forward restore must not pull sibling DistOpt masters back."""

    events = []

    class FakeOptimizer:
        def __init__(self, config, name, requires_sync):
            self.config = config
            self.name = name
            self.requires_sync = requires_sync

        def optimizer_state_offload_requires_pre_forward_param_sync(self):
            checks.append(self.name)
            return self.requires_sync

        def ensure_master_weights_for_pre_forward_param_sync(self):
            if self.optimizer_state_offload_requires_pre_forward_param_sync():
                events.append(self.name)

    config = object()
    checks = []
    optimizer = ChainedOptimizer(
        [FakeOptimizer(config, "layerwise", True), FakeOptimizer(config, "distributed", False)]
    )

    optimizer.ensure_master_weights_for_pre_forward_param_sync()

    assert events == ["layerwise"]
    assert checks == ["layerwise", "distributed"]


def test_chained_general_master_restore_targets_all_children():
    """The general chain API preserves the leaf contract by restoring every child."""

    events = []

    class FakeOptimizer:
        def __init__(self, config, name):
            self.config = config
            self.name = name

        def ensure_master_weights_for_param_sync(self):
            events.append(self.name)

    config = object()
    optimizer = ChainedOptimizer(
        [FakeOptimizer(config, "layerwise"), FakeOptimizer(config, "distributed")]
    )

    optimizer.ensure_master_weights_for_param_sync()

    assert events == ["layerwise", "distributed"]


def test_pre_forward_sync_requirement_requires_subset_implementation():
    """A leaf declaring pre-forward sync must implement its bucket-subset dispatch."""

    optimizer = object.__new__(FP32Optimizer)
    optimizer._defer_optimizer_master_offload_for_param_sync = True

    with pytest.raises(NotImplementedError, match="must override"):
        optimizer.start_param_sync_for_bucket_group_subset(force_sync=True)


def test_fp32_ready_grad_step_rejects_chunked_state_offloader():
    """The unsupported common-base FP32 entry point must not bypass its manager."""

    optimizer = object.__new__(FP32Optimizer)
    optimizer.is_stub_optimizer = False
    optimizer._optimizer_state_offloader = object()

    with pytest.raises(RuntimeError, match="FP32Optimizer.step_with_ready_grads"):
        optimizer.step_with_ready_grads()


def test_offloader_rejects_two_master_storage_schemes():
    """One wrapped optimizer cannot alias explicit and optimizer-owned master slots."""

    param = nn.Parameter(torch.ones(2))
    optimizer = torch.optim.SGD([param], lr=0.1)
    optimizer.master_weights = True
    with pytest.raises(ValueError, match="both MCore-owned master parameters"):
        ChunkedOptimizerStateOffloader(
            optimizer=optimizer,
            master_params=[param],
            chunk_size_bytes=0,
            offload_fraction=1.0,
            state_dtypes=(torch.float32,),
        )


def test_oversized_atomic_state_chunks_log_once():
    """A soft chunk target overflow is visible without splitting optimizer parameters."""

    manager = object.__new__(ChunkedOptimizerStateOffloader)
    manager.chunk_size_bytes = 1024**2
    manager._state_bytes_per_param = torch.float32.itemsize
    manager.optimizer = SimpleNamespace()
    params = tuple(nn.Parameter(torch.empty(numel)) for numel in (300_000, 400_000, 10))

    with mock.patch(
        "megatron.core.optimizer.cpu_offloading.chunked_optimizer_state_offload.log_single_rank"
    ) as log:
        chunks = manager._build_chunks(params)

    assert [[id(param) for param in chunk.params] for chunk in chunks] == [
        [id(param)] for param in params
    ]
    log.assert_called_once()
    _, level, _, target_mib, oversized_count, largest_mib, optimizer_name = log.call_args.args
    assert level == logging.WARNING
    assert target_mib == 1.0
    assert oversized_count == 2
    assert largest_mib == pytest.approx(400_000 * torch.float32.itemsize / 1024**2)
    assert optimizer_name == "SimpleNamespace"


def test_h2d_targets_record_the_transfer_stream():
    """State and master allocations remain live if an update exits before its H2D completes."""

    class FakeTensor:
        def __init__(self, *, is_cuda=False, pinned=False):
            self.is_cuda = is_cuda
            self.pinned = pinned
            self.recorded_streams = []

        def is_pinned(self):
            return self.pinned

        def copy_(self, source, non_blocking=False):
            return self

        def record_stream(self, stream):
            self.recorded_streams.append(stream)

    class FakeEvent:
        def __init__(self):
            self.recorded_stream = None

        def record(self, stream):
            self.recorded_stream = stream

    h2d_stream = object()
    state_param = object()
    state_cpu = FakeTensor(pinned=True)
    state_gpu = FakeTensor(is_cuda=True)
    state_manager = object.__new__(ChunkedOptimizerStateOffloader)
    state_manager._h2d_stream = h2d_stream
    state_manager._state_staging_views = lambda chunk: [
        (state_param, "moment", state_cpu, state_gpu)
    ]
    state_manager._order_h2d_after_source_streams = lambda: None
    state_manager.optimizer = SimpleNamespace(state={state_param: {}})

    class FakeParam:
        pass

    master_param = FakeParam()
    master_cpu = FakeTensor(pinned=True)
    master_gpu = FakeTensor(is_cuda=True)
    master_param.data = master_cpu
    master_manager = object.__new__(ChunkedOptimizerStateOffloader)
    master_manager._selected_params = (master_param,)
    master_manager._master_h2d_event = None
    master_manager._explicit_master_param_ids = {id(master_param)}
    master_manager._master_weights_resident = True
    master_manager._param_devices = {master_param: object()}
    master_manager._cpu_master = {}
    master_manager._h2d_stream = h2d_stream
    master_manager._validate_master_storage = lambda: None
    master_manager._order_h2d_after_source_streams = lambda: None
    master_manager.optimizer = SimpleNamespace(state={master_param: {}})

    with (
        mock.patch.object(torch.cuda, "stream", side_effect=lambda stream: nullcontext()),
        mock.patch.object(torch.cuda, "Event", side_effect=FakeEvent),
        mock.patch.object(torch, "empty_like", return_value=master_gpu),
    ):
        state_manager._prefetch_state(object())
        master_manager._schedule_master_h2d()

    assert state_gpu.recorded_streams == [h2d_stream]
    assert master_gpu.recorded_streams == [h2d_stream]


def test_initialize_state_for_loading_requires_an_initializer():
    """A missing external optimizer initializer must fail instead of silently skipping state."""

    manager = object.__new__(ChunkedOptimizerStateOffloader)

    with pytest.raises(RuntimeError, match="requires an optimizer state initializer"):
        manager.initialize_state_for_loading(None, object())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_transfer_stream_adoption_is_idempotent_after_first_use():
    """Re-adopting the current stream pair is safe even after canonical CPU state exists."""

    param = nn.Parameter(torch.zeros(8, device='cuda'))
    optimizer = torch.optim.SGD([param], lr=0.1)
    manager = ChunkedOptimizerStateOffloader(
        optimizer=optimizer,
        master_params=[],
        chunk_size_bytes=0,
        offload_fraction=1.0,
        state_dtypes=(torch.float32,),
    )
    d2h_stream, h2d_stream = manager.transfer_streams
    manager._cpu_state[param]['moment'] = torch.empty(param.shape, device='cpu', pin_memory=True)

    manager.use_transfer_streams(d2h_stream, h2d_stream)

    with pytest.raises(RuntimeError, match="cannot change after first use"):
        manager.use_transfer_streams(torch.cuda.Stream(), h2d_stream)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_chunked_step_releases_staging_slots_when_optimizer_raises():
    """An optimizer failure must not leave reusable staging buffers owned by the manager."""

    class FailingOptimizer(torch.optim.Optimizer):
        def __init__(self, params):
            super().__init__(params, defaults={})

        def step(self, closure=None):
            raise RuntimeError("optimizer failure")

    param = nn.Parameter(torch.zeros(8, device='cuda'))
    optimizer = FailingOptimizer([param])
    manager = ChunkedOptimizerStateOffloader(
        optimizer=optimizer,
        master_params=[],
        chunk_size_bytes=0,
        offload_fraction=1.0,
        state_dtypes=(torch.float32,),
    )
    cpu_state = torch.empty(param.shape, device='cpu', pin_memory=True)
    optimizer.state[param]['moment'] = cpu_state
    manager._cpu_state[param]['moment'] = cpu_state

    with pytest.raises(RuntimeError, match="optimizer failure"):
        manager.step()

    assert all(not slot.buffers for slot in manager._state_staging_slots)
    manager.synchronize_for_checkpoint()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_chunked_step_propagates_group_writes_and_clones_tensor_step():
    """Group metadata is committed once without rescanning original group parameters."""

    class ParamsThatMustNotBeIterated(list):
        def __iter__(self):
            raise AssertionError("chunk execution rescanned the full optimizer parameter group")

    class GroupStateOptimizer(torch.optim.Optimizer):
        def __init__(self, params):
            super().__init__(
                params,
                defaults={"step": torch.tensor(0), "updates": 0, "removed_during_step": True},
            )

        @torch.no_grad()
        def step(self, closure=None):
            for group in self.param_groups:
                group["step"].add_(1)
                group["updates"] += 1
                group["new_field"] = "preserved"
                group.pop("removed_during_step")
                for param in group["params"]:
                    param.add_(1)

    params = [nn.Parameter(torch.zeros(256, device="cuda")) for _ in range(3)]
    optimizer = GroupStateOptimizer(params)
    manager = ChunkedOptimizerStateOffloader(
        optimizer=optimizer,
        master_params=[],
        chunk_size_bytes=1024,
        offload_fraction=1.0,
        state_dtypes=(torch.float32,),
    )
    assert len(manager.chunks) == 3
    optimizer.param_groups[0]["params"] = ParamsThatMustNotBeIterated(
        optimizer.param_groups[0]["params"]
    )

    manager.step()

    group = optimizer.param_groups[0]
    assert group["step"].item() == 1
    assert group["updates"] == 1
    assert group["new_field"] == "preserved"
    assert "removed_during_step" not in group
    assert all(torch.equal(param, torch.ones_like(param)) for param in params)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_state_staging_views_are_aligned_and_non_overlapping():
    """Flat staging pools must preserve a distinct aligned region for every state tensor."""

    params = [nn.Parameter(torch.zeros(size, device="cuda")) for size in (3, 5, 7)]
    optimizer = torch.optim.SGD(params, lr=0.1)
    manager = ChunkedOptimizerStateOffloader(
        optimizer=optimizer,
        master_params=[],
        chunk_size_bytes=0,
        offload_fraction=1.0,
        state_dtypes=(torch.float32,),
    )
    for param in params:
        cpu_state = torch.empty(param.shape, dtype=torch.float32, device="cpu", pin_memory=True)
        optimizer.state[param]["moment"] = cpu_state
        manager._cpu_state[param]["moment"] = cpu_state

    staging_views = manager._state_staging_views(manager.chunks[0])
    byte_ranges = []
    for _, _, _, gpu_tensor in staging_views:
        start = gpu_tensor.data_ptr()
        assert start % 256 == 0
        byte_ranges.append((start, start + gpu_tensor.numel() * gpu_tensor.element_size()))

    byte_ranges.sort()
    assert all(left[1] <= right[0] for left, right in zip(byte_ranges, byte_ranges[1:]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_state_h2d_waits_for_compute_stream_before_reusing_staging_buffer():
    """H2D must not overwrite compute-stream storage while prior work still uses it."""

    param = nn.Parameter(torch.zeros(4096, device="cuda"))
    optimizer = torch.optim.SGD([param], lr=0.1)
    manager = ChunkedOptimizerStateOffloader(
        optimizer=optimizer,
        master_params=[],
        chunk_size_bytes=0,
        offload_fraction=1.0,
        state_dtypes=(torch.float32,),
    )
    cpu_state = torch.zeros(param.shape, dtype=torch.float32, device="cpu", pin_memory=True)
    optimizer.state[param]["moment"] = cpu_state
    manager._cpu_state[param]["moment"] = cpu_state

    compute_stream = torch.cuda.Stream()
    with torch.cuda.stream(compute_stream):
        # Allocate the reusable slot on the compute stream, then make its last compute use
        # finish well after the tiny CPU-to-GPU copy would finish without stream ordering.
        staging_views = manager._state_staging_views(manager.chunks[0])
        manager._next_state_staging_slot = 0
        torch.cuda._sleep(50_000_000)
        staging_views[0][3].fill_(7)
        prefetch = manager._prefetch_state(manager.chunks[0])

    assert prefetch.event is not None
    prefetch.event.synchronize()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        optimizer.state[param]["moment"], torch.zeros_like(optimizer.state[param]["moment"])
    )
    manager.synchronize_for_checkpoint()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_master_h2d_allocates_on_compute_stream():
    """A restored master window must return to the compute stream's allocator pool."""

    param = nn.Parameter(torch.ones(4096, device="cuda"))
    optimizer = torch.optim.SGD([param], lr=0.1)
    manager = ChunkedOptimizerStateOffloader(
        optimizer=optimizer,
        master_params=[param],
        chunk_size_bytes=0,
        offload_fraction=1.0,
        state_dtypes=(torch.float32,),
    )
    manager.synchronize_for_checkpoint()
    assert param.device.type == "cpu"

    allocation_streams = []
    compute_stream = torch.cuda.Stream()
    empty_like = torch.empty_like

    def record_allocation_stream(*args, **kwargs):
        allocation_streams.append(torch.cuda.current_stream().cuda_stream)
        tensor = empty_like(*args, **kwargs)
        # If master H2D does not wait for this allocation/owner stream, the delayed
        # sentinel write can race after the CPU master copy and remain visible.
        torch.cuda._sleep(50_000_000)
        tensor.fill_(7)
        return tensor

    with (
        torch.cuda.stream(compute_stream),
        mock.patch.object(torch, "empty_like", side_effect=record_allocation_stream),
    ):
        manager.prefetch_master_for_step()

    manager.ensure_master_for_param_sync()
    torch.cuda.synchronize()
    assert allocation_streams == [compute_stream.cuda_stream]
    assert param.is_cuda
    torch.testing.assert_close(param, torch.ones_like(param))
    manager.synchronize_for_checkpoint()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_state_eligibility_is_identical_for_d2h_and_cpu_adoption():
    """A scalar tensor is not state for a one-element parameter unless its shape matches."""

    param = nn.Parameter(torch.zeros(1, device="cuda"))
    optimizer = torch.optim.SGD([param], lr=0.1)
    manager = ChunkedOptimizerStateOffloader(
        optimizer=optimizer,
        master_params=[],
        chunk_size_bytes=0,
        offload_fraction=1.0,
        state_dtypes=(torch.float32,),
    )
    optimizer.state[param]["moment"] = torch.ones_like(param)
    optimizer.state[param]["step"] = torch.ones((), device="cuda")
    optimizer.state[param]["found_inf"] = torch.ones_like(param)

    manager.synchronize_for_checkpoint()

    assert optimizer.state[param]["moment"].device.type == "cpu"
    assert optimizer.state[param]["step"].is_cuda
    assert optimizer.state[param]["found_inf"].is_cuda
    assert "moment" in manager._cpu_state.get(param, {})
    assert "step" not in manager._cpu_state.get(param, {})
    assert "found_inf" not in manager._cpu_state.get(param, {})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_load_releases_cpu_buffers_from_an_old_state_schema():
    """Replacing checkpoint state must not retain obsolete pinned canonical buffers."""

    param = nn.Parameter(torch.zeros(8, device="cuda"))
    optimizer = torch.optim.SGD([param], lr=0.1)
    manager = ChunkedOptimizerStateOffloader(
        optimizer=optimizer,
        master_params=[],
        chunk_size_bytes=0,
        offload_fraction=1.0,
        state_dtypes=(torch.float32,),
    )
    old_state = torch.empty_like(param, device="cpu", pin_memory=True)
    new_state = torch.empty_like(param, device="cpu", pin_memory=True)
    optimizer.state[param]["old_moment"] = old_state
    manager._cpu_state[param]["old_moment"] = old_state
    state_dict = optimizer.state_dict()
    state_dict["state"][0] = {"new_moment": new_state}

    manager.load_state_dict_without_device_cast(state_dict)

    assert "old_moment" not in manager._cpu_state[param]
    assert manager._cpu_state[param]["new_moment"] is new_state


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_partial_load_moves_resident_tensor_state_back_to_cuda():
    """Fully-parallel CPU loads keep selected state canonical but restore resident moments."""

    params = [nn.Parameter(torch.zeros(8, device="cuda")) for _ in range(2)]
    optimizer = torch.optim.SGD(params, lr=0.1)
    manager = ChunkedOptimizerStateOffloader(
        optimizer=optimizer,
        master_params=[],
        chunk_size_bytes=0,
        offload_fraction=0.5,
        state_dtypes=(torch.float32,),
    )
    state_dict = optimizer.state_dict()
    state_dict["state"] = {
        param_id: {
            "moment": torch.full(param.shape, index + 1.0, dtype=param.dtype, device="cpu"),
            "step": torch.tensor(index + 1),
        }
        for index, (param_id, param) in enumerate(
            zip(state_dict["param_groups"][0]["params"], params)
        )
    }

    manager.load_state_dict_without_device_cast(state_dict)

    selected_param = manager.selected_params[0]
    resident_param = manager._resident_params[0]
    assert optimizer.state[selected_param]["moment"].device.type == "cpu"
    assert optimizer.state[selected_param]["moment"].is_pinned()
    assert optimizer.state[resident_param]["moment"].is_cuda
    assert optimizer.state[resident_param]["step"].device.type == "cpu"
    torch.testing.assert_close(
        optimizer.state[resident_param]["moment"], torch.full_like(resident_param, 2.0)
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_byte_planning_uses_state_dtypes_and_prioritizes_master_bundles():
    """A partial budget uses real moment widths and spends bytes on removable masters first."""

    native_param = nn.Parameter(torch.zeros(128, device="cuda"))
    master_param = nn.Parameter(torch.zeros(128, device="cuda"))
    optimizer = torch.optim.SGD([native_param, master_param], lr=0.1)
    manager = ChunkedOptimizerStateOffloader(
        optimizer=optimizer,
        master_params=[master_param],
        chunk_size_bytes=0,
        offload_fraction=0.5,
        state_dtypes=(torch.bfloat16, torch.bfloat16),
    )

    assert manager._estimated_state_bytes(native_param) == 128 * 2 * torch.bfloat16.itemsize
    assert len(manager.selected_params) == 1
    assert manager.selected_params[0] is master_param


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_precision_aware_master_planning_uses_real_storage():
    """Precision-aware planner accounts for every declared master using its real dtype."""

    low_precision_param = nn.Parameter(torch.zeros(128, dtype=torch.bfloat16, device="cuda"))
    native_fp32_param = nn.Parameter(torch.zeros(128, dtype=torch.float32, device="cuda"))
    optimizer = torch.optim.SGD([native_fp32_param, low_precision_param], lr=0.1)
    optimizer.master_weights = True
    manager = ChunkedOptimizerStateOffloader(
        optimizer=optimizer,
        master_params=[],
        chunk_size_bytes=0,
        offload_fraction=1.0,
        state_dtypes=(torch.bfloat16, torch.bfloat16),
        optimizer_owned_master_dtypes={
            native_fp32_param: torch.float32,
            low_precision_param: torch.int16,
        },
    )

    state_bytes = 128 * 2 * torch.bfloat16.itemsize
    assert manager._estimated_bundle_bytes(native_fp32_param) == (
        state_bytes + 128 * torch.float32.itemsize
    )
    assert manager._estimated_bundle_bytes(low_precision_param) == (
        state_bytes + 128 * torch.int16.itemsize
    )


def create_model_and_optimizer(
    hidden_size: int = 256,
    num_layers: int = 2,
    chunked_optimizer_state_offload: bool = True,
    chunk_size_mb: int = 1,
    offload_fraction: float = 1.0,
    include_native_fp32_param: bool = False,
    **optimizer_kwargs,
):
    """Create a bf16 DDP model and DistributedOptimizer."""

    model = SimpleModel(hidden_size=hidden_size, num_layers=num_layers).bfloat16().cuda()
    if include_native_fp32_param:
        model.native_fp32_param = nn.Parameter(torch.zeros(hidden_size, device='cuda'))
    ddp_config = DistributedDataParallelConfig(use_distributed_optimizer=True)
    model = DistributedDataParallel(
        TransformerConfig(num_attention_heads=1, num_layers=1), ddp_config, model
    )

    default_config = dict(
        optimizer='adam',
        bf16=True,
        lr=0.001,
        use_distributed_optimizer=True,
        chunked_optimizer_state_offload=chunked_optimizer_state_offload,
        optimizer_state_offload_chunk_size_mb=chunk_size_mb,
        optimizer_state_offload_fraction=offload_fraction,
    )
    default_config.update(optimizer_kwargs)
    optimizer = get_megatron_optimizer(OptimizerConfig(**default_config), [model])
    return model, optimizer


@pytest.mark.skipif(
    not TE_FUSED_ADAM_AVAILABLE or not is_te_min_version("2.1.0.dev0"),
    reason="Requires TE FusedAdam with precision-aware optimizer support",
)
@pytest.mark.usefixtures("model_parallel_context")
def test_distributed_optimizer_plans_native_fp32_optimizer_owned_master():
    """DistOpt tells the planner that TE owns a master for native-FP32 shards too."""

    model, optimizer = create_model_and_optimizer(
        use_precision_aware_optimizer=True, include_native_fp32_param=True, offload_fraction=0.5
    )
    dist_optimizer = optimizer.chained_optimizers[0]
    manager = dist_optimizer._optimizer_state_offloader
    native_fp32_shards = [param for group in dist_optimizer.shard_fp32_groups for param in group]

    assert native_fp32_shards
    assert manager is not None
    assert all(manager._param_has_master(param) for param in native_fp32_shards)
    assert all(
        manager._optimizer_owned_master_dtypes[id(param)] == dist_optimizer.config.main_params_dtype
        for param in native_fp32_shards
    )
    del model, optimizer


def test_group_metadata_comparison_accepts_bool_convertible_equality():
    """Non-bool scalar equality results are interpreted through their truth value."""

    class TruthValue:
        def __bool__(self):
            return True

    class Metadata:
        def __eq__(self, other):
            return TruthValue()

    assert ChunkedOptimizerStateOffloader._group_values_equal(Metadata(), Metadata())


def offload_before_forward(optimizer) -> None:
    """Move selected state/master bundles to CPU and release their CUDA tensors."""

    optimizer.offload_optimizer_state_for_forward()
    optimizer.zero_grad()


def run_forward_backward_step(
    model, optimizer, hidden_size: int = 256, input_tensor: torch.Tensor | None = None
) -> None:
    """Run forward/backward and one chunked optimizer update."""

    if input_tensor is None:
        input_tensor = torch.randn(8, hidden_size, dtype=torch.bfloat16, device='cuda')
    output = model(input_tensor)
    output.sum().backward()
    optimizer.prefetch_optimizer_state_for_step()
    optimizer.step()
    optimizer.zero_grad()


@pytest.mark.skipif(not TE_FUSED_ADAM_AVAILABLE, reason="Requires TE FusedAdam")
@pytest.mark.usefixtures("model_parallel_context")
def test_chunk_plan_and_initial_master_offload():
    """The manager builds bounded chunks and moves masters to CPU before the first forward."""

    model, optimizer = create_model_and_optimizer(hidden_size=512, chunk_size_mb=1)
    dist_optimizer = optimizer.chained_optimizers[0]
    manager = dist_optimizer._optimizer_state_offloader

    assert manager is not None
    # Direct manager tests cover exact chunk splitting independently of DP sharding. This
    # integration test only needs a valid plan before checking the initial master offload.
    assert manager.chunks
    assert manager.selected_params

    expected_masters = {
        id(param): param.detach().clone()
        for group in dist_optimizer.shard_fp32_from_float16_groups
        for param in group
        if manager.is_param_offloaded(param)
    }
    offload_before_forward(optimizer)
    optimizer.synchronize_optimizer_state_for_checkpoint()

    for group in dist_optimizer.shard_fp32_from_float16_groups:
        for param in group:
            if manager.is_param_offloaded(param):
                assert param.device.type == 'cpu'
                torch.testing.assert_close(param, expected_masters[id(param)].cpu())

    # Public reload entry points restore an offloaded master window before writing it.
    dist_optimizer.reload_model_params()
    for group in dist_optimizer.shard_fp32_from_float16_groups:
        for param in group:
            if manager.is_param_offloaded(param):
                assert param.is_cuda
                torch.testing.assert_close(param, expected_masters[id(param)])

    del model, optimizer


@pytest.mark.skipif(not TE_FUSED_ADAM_AVAILABLE, reason="Requires TE FusedAdam")
@pytest.mark.usefixtures("model_parallel_context")
def test_chunk_size_zero_uses_one_full_state_window():
    """Chunk size zero preserves full temporary GPU restore semantics."""

    model, optimizer = create_model_and_optimizer(chunk_size_mb=0)
    manager = optimizer.chained_optimizers[0]._optimizer_state_offloader
    assert manager is not None
    assert len(manager.chunks) == 1
    del model, optimizer


@pytest.mark.skipif(not TE_FUSED_ADAM_AVAILABLE, reason="Requires TE FusedAdam")
@pytest.mark.usefixtures("model_parallel_context")
def test_step_leaves_state_and_master_cpu_canonical():
    """A completed update leaves selected moments and masters CPU-resident before forward."""

    model, optimizer = create_model_and_optimizer(hidden_size=512, chunk_size_mb=1)
    dist_optimizer = optimizer.chained_optimizers[0]
    manager = dist_optimizer._optimizer_state_offloader

    offload_before_forward(optimizer)
    run_forward_backward_step(model, optimizer, hidden_size=512)
    optimizer.offload_optimizer_state_for_forward()

    for param in manager.selected_params:
        for key, value in manager.optimizer.state[param].items():
            if isinstance(value, torch.Tensor) and value.numel() == param.numel():
                assert value.device.type == 'cpu', f"{key} was not offloaded"

    for group in dist_optimizer.shard_fp32_from_float16_groups:
        for param in group:
            if manager.is_param_offloaded(param):
                assert param.device.type == 'cpu'

    del model, optimizer


@pytest.mark.skipif(not TE_FUSED_ADAM_AVAILABLE, reason="Requires TE FusedAdam")
@pytest.mark.usefixtures("model_parallel_context")
def test_state_can_offload_before_master_for_mxfp8_staging():
    """The delayed-master path keeps masters on CUDA until parameter staging completes."""

    model, optimizer = create_model_and_optimizer(hidden_size=512, chunk_size_mb=1)
    dist_optimizer = optimizer.chained_optimizers[0]
    manager = dist_optimizer._optimizer_state_offloader

    offload_before_forward(optimizer)
    run_forward_backward_step(model, optimizer, hidden_size=512)
    optimizer.offload_optimizer_state_for_forward(offload_master=False)

    for param in manager.selected_params:
        for key, value in manager.optimizer.state[param].items():
            if key != "master_param" and isinstance(value, torch.Tensor):
                if value.numel() == param.numel():
                    assert value.device.type == "cpu"
    for group in dist_optimizer.shard_fp32_from_float16_groups:
        for param in group:
            if manager.is_param_offloaded(param):
                assert param.is_cuda

    optimizer.offload_optimizer_state_for_forward()
    for group in dist_optimizer.shard_fp32_from_float16_groups:
        for param in group:
            if manager.is_param_offloaded(param):
                assert param.device.type == "cpu"

    optimizer.synchronize_optimizer_state_for_checkpoint()
    del model, optimizer


@pytest.mark.skipif(not TE_FUSED_ADAM_AVAILABLE, reason="Requires TE FusedAdam")
@pytest.mark.usefixtures("model_parallel_context")
def test_offload_fraction_applies_to_state_master_bundles():
    """Partial offload selects matching state/master bundles deterministically."""

    model, optimizer = create_model_and_optimizer(
        hidden_size=512, chunk_size_mb=1, offload_fraction=0.5
    )
    dist_optimizer = optimizer.chained_optimizers[0]
    manager = dist_optimizer._optimizer_state_offloader
    assert manager is not None
    total_bytes = sum(manager._estimated_bundle_bytes(param) for param in manager._params)
    selected_bytes = sum(
        manager._estimated_bundle_bytes(param) for param in manager.selected_params
    )
    assert manager.selected_params
    assert selected_bytes >= (total_bytes + 1) // 2

    selected_ids = {id(param) for param in manager.selected_params}
    resident_master = any(
        id(param) in manager._explicit_master_param_ids and id(param) not in selected_ids
        for param in manager._params
    )
    selected_without_master = any(
        id(param) not in manager._explicit_master_param_ids for param in manager.selected_params
    )
    assert not (resident_master and selected_without_master)

    offload_before_forward(optimizer)
    for group in dist_optimizer.shard_fp32_from_float16_groups:
        for param in group:
            if id(param) in selected_ids:
                assert param.device.type == 'cpu'
            else:
                assert param.is_cuda

    del model, optimizer


@pytest.mark.skipif(not TE_FUSED_ADAM_AVAILABLE, reason="Requires TE FusedAdam")
@pytest.mark.usefixtures("model_parallel_context")
def test_partial_offload_keeps_data_parallel_ranks_in_sync():
    """Rank-local partial selections must still produce identical gathered parameters."""

    if torch.distributed.get_world_size() < 2:
        pytest.skip("Requires multiple data-parallel ranks")

    torch.manual_seed(42)
    model, optimizer = create_model_and_optimizer(
        hidden_size=512, chunk_size_mb=1, offload_fraction=0.5
    )
    torch.manual_seed(123)
    for _ in range(2):
        inputs = torch.randn(8, 512, dtype=torch.bfloat16, device="cuda")
        offload_before_forward(optimizer)
        run_forward_backward_step(model, optimizer, hidden_size=512, input_tensor=inputs)

    optimizer.offload_optimizer_state_for_forward()
    for param in model.parameters():
        gathered = [torch.empty_like(param) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(gathered, param)
        for other in gathered[1:]:
            torch.testing.assert_close(param, other, atol=0, rtol=0)

    optimizer.synchronize_optimizer_state_for_checkpoint()
    del model, optimizer


@pytest.mark.skipif(not TE_FUSED_ADAM_AVAILABLE, reason="Requires TE FusedAdam")
@pytest.mark.usefixtures("model_parallel_context")
def test_chunked_step_advances_group_step_once():
    """Calling the external optimizer for several chunks advances logical step only once."""

    # DistOpt shards the flat parameter buffer across DP ranks. Give every rank about three
    # small layers so each local shard contains several bundles and must span multiple chunks.
    num_layers = 3 * torch.distributed.get_world_size()
    model, optimizer = create_model_and_optimizer(num_layers=num_layers, chunk_size_mb=1)
    dist_optimizer = optimizer.chained_optimizers[0]
    manager = dist_optimizer._optimizer_state_offloader
    assert manager is not None and len(manager.chunks) > 1

    for _ in range(3):
        offload_before_forward(optimizer)
        run_forward_backward_step(model, optimizer)

    steps = [
        group['step']
        for group in manager.optimizer.param_groups
        if group['params'] and 'step' in group
    ]
    assert steps and all(step == 3 for step in steps)

    del model, optimizer


@pytest.mark.skipif(not TE_FUSED_ADAM_AVAILABLE, reason="Requires TE FusedAdam")
@pytest.mark.usefixtures("model_parallel_context")
def test_training_matches_non_offloaded_optimizer():
    """Chunked updates remain numerically equivalent to a full FusedAdam update."""

    torch.manual_seed(42)
    model_offload, optimizer_offload = create_model_and_optimizer(
        chunked_optimizer_state_offload=True, chunk_size_mb=1, lr=0.01
    )
    torch.manual_seed(42)
    model_reference, optimizer_reference = create_model_and_optimizer(
        chunked_optimizer_state_offload=False, lr=0.01
    )

    torch.manual_seed(123)
    for _ in range(5):
        inputs = torch.randn(8, 256, dtype=torch.bfloat16, device='cuda')

        offload_before_forward(optimizer_offload)
        run_forward_backward_step(model_offload, optimizer_offload, input_tensor=inputs)

        output = model_reference(inputs)
        output.sum().backward()
        optimizer_reference.step()
        optimizer_reference.zero_grad()

    optimizer_offload.offload_optimizer_state_for_forward()
    for (name_offload, param_offload), (name_reference, param_reference) in zip(
        model_offload.named_parameters(), model_reference.named_parameters()
    ):
        assert name_offload == name_reference
        torch.testing.assert_close(
            param_offload,
            param_reference,
            atol=1e-5,
            rtol=1e-4,
            msg=f"Parameter {name_offload} differs from the full-step reference",
        )

    del model_offload, optimizer_offload, model_reference, optimizer_reference
