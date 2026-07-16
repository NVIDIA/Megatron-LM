# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the Megatron-FSDP v2 ``fully_shard`` API, ``FSDPModule``,
and checkpoint (``get_state_dict``).

Covers:
- Basic fully_shard: class mutation, hooks, reshard on init
- Multi-layer LLM-style nesting (embedding → transformer layers → lm_head)
- Multimodal-style: separate encoders with partial freezing
- Partially frozen training (requires_grad=False on some params)
- Nested FSDP (expert-in-layer pattern)
- Mixed precision policies (fp32 main params, fp32 grad reduce)
- ignored_params
- enable_unshard_prefetch / enable_async_reduce_grad feature flags
- Forward/backward lifecycle correctness
- get_state_dict / preprocess_state_dict_for_uneven_dtensor
- Double-shard safety (reject re-wrap)

Run with:
    torchrun --nproc_per_node=2 -m pytest \\
        tests/unit_tests/distributed/megatron_fsdp/v2/test_fully_shard.py -v

Single-GPU tests:
    pytest tests/unit_tests/distributed/megatron_fsdp/v2/test_fully_shard.py -v \\
        -k "test_double_shard_rejected or test_no_params_module"
"""

import shutil
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import StateDictOptions
from torch.distributed.checkpoint.state_dict import get_state_dict as torch_get_state_dict
from torch.distributed.checkpoint.state_dict import set_state_dict as torch_set_state_dict
from torch.distributed.tensor import DeviceMesh

sys.path.insert(0, str(Path(__file__).parents[2]))
from megatron.core.distributed.fsdp.src.megatron_fsdp.uneven_dtensor import (
    get_state_dict,
    preprocess_state_dict_for_uneven_dtensor,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.v2.fsdp_module import FSDPModule
from megatron.core.distributed.fsdp.src.megatron_fsdp.v2.fully_shard import fully_shard
from megatron.core.distributed.fsdp.src.megatron_fsdp.v2.hooks import mfsdp_forward_pre_hook
from megatron.core.distributed.fsdp.src.megatron_fsdp.v2.mixed_precision import MixedPrecisionPolicy

SHARED_TMP_DIR = "/tmp/pytest-shared-tmp"

# ------------------------------------------------------------------ #
#  Distributed environment (NCCL session-scoped)
# ------------------------------------------------------------------ #


@pytest.fixture(scope="session", autouse=True)
def dist_env():
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")
    rank = torch.distributed.get_rank()
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    torch.cuda.set_device(device)
    yield
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def _rank():
    return torch.distributed.get_rank()


def _world_size():
    return torch.distributed.get_world_size()


def _device():
    return torch.device(f"cuda:{_rank() % torch.cuda.device_count()}")


def _build_hsdp_mesh():
    world_size = _world_size()
    if world_size < 4 or world_size % 2 != 0:
        pytest.skip("HSDP checkpoint coverage requires an even world size >= 4")

    mesh = torch.arange(world_size, dtype=torch.int).reshape(2, world_size // 2)
    return DeviceMesh(_device().type, mesh, mesh_dim_names=("dp_outer", "dp"))


# ------------------------------------------------------------------ #
#  Mock models for different application scenarios
# ------------------------------------------------------------------ #


class SimpleMLP(nn.Module):
    """Single linear layer with optional bias."""

    def __init__(self, hidden=64, bias=True):
        super().__init__()
        self.fc = nn.Linear(hidden, hidden, bias=bias)

    def forward(self, x):
        return self.fc(x)


class MixedDtypeBuffers(nn.Module):
    """Module whose FSDP unit owns communication buffers with different dtypes."""

    def __init__(self, hidden=64):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(hidden, hidden))
        nn.init.normal_(self.weight)
        self.uint8_weight = nn.Parameter(
            torch.arange(hidden * hidden, dtype=torch.uint8).reshape(hidden, hidden),
            requires_grad=False,
        )

    def forward(self, x):
        return x @ self.weight


class TinyLLM(nn.Module):
    """Simulates an LLM: embedding → block of layers → lm_head.

    Structure::
        embedding (nn.Embedding) → layers (nn.ModuleList of SimpleMLP) → lm_head (nn.Linear)
    """

    def __init__(self, vocab=128, hidden=64, num_layers=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab, hidden)
        self.layers = nn.ModuleList([SimpleMLP(hidden) for _ in range(num_layers)])
        self.lm_head = nn.Linear(hidden, vocab)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x):
        h = self.embedding(x)
        for layer in self.layers:
            h = layer(h)
        return self.lm_head(self.norm(h))


class MultimodalModel(nn.Module):
    """Simulates a multimodal model with separate vision/text encoders.

    Structure::
        vision_encoder (nn.Linear) — may be frozen
        text_encoder (nn.Linear) — trainable
        fusion (nn.Linear) — trainable
    """

    def __init__(self, hidden=64):
        super().__init__()
        self.vision_encoder = nn.Linear(hidden, hidden)
        self.text_encoder = nn.Linear(hidden, hidden)
        self.fusion = nn.Linear(hidden * 2, hidden)

    def forward(self, img, txt):
        v = self.vision_encoder(img)
        t = self.text_encoder(txt)
        return self.fusion(torch.cat([v, t], dim=-1))


class ExpertBlock(nn.Module):
    """Simulates an MoE expert: two linear layers."""

    def __init__(self, hidden=64, ffn_hidden=128):
        super().__init__()
        self.fc1 = nn.Linear(hidden, ffn_hidden)
        self.fc2 = nn.Linear(ffn_hidden, hidden)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class MOETransformerLayer(nn.Module):
    """Simulates a transformer layer with MoE: attention → MoE experts.

    Structure::
        attn (nn.Linear) → experts (ExpertBlock) → norm (nn.LayerNorm)
    """

    def __init__(self, hidden=64, ffn_hidden=128):
        super().__init__()
        self.attn = nn.Linear(hidden, hidden)
        self.experts = ExpertBlock(hidden, ffn_hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x):
        h = self.attn(x)
        h = self.experts(h)
        return self.norm(h + x)


# ------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------ #


def _set_last_backward(model, is_last_backward: bool = True):
    """Mark the next FSDP v2 backward as the optimizer-step boundary."""
    if hasattr(model, "set_is_last_backward"):
        model.set_is_last_backward(is_last_backward)


def _forward_backward(model, x):
    """Run forward + backward and return loss."""
    out = model(x)
    loss = out.sum()
    loss.backward()
    return loss.item()


def _assert_dtensor_params(module):
    """Assert all parameters in the module (and any nested FSDPModules) are DTensors."""
    from torch.distributed.tensor import DTensor

    for name, p in module.named_parameters():
        assert isinstance(p, DTensor), (
            f"Parameter '{name}' should be a DTensor after fully_shard, " f"got {type(p).__name__}"
        )


def _assert_original_params_unchanged(module, originals):
    """After fully_shard, the original (pre-fully_shard) param OBJECTS should
    still be the same Python objects (identity check), but their .data may have
    been freed (empty tensor)."""
    for name, p in module.named_parameters():
        assert (
            p is originals[name]
        ), f"Original param object for '{name}' was replaced; expected identity match."


def _count_fsdp_modules(module):
    """Return number of FSDPModule instances in the module tree."""
    return sum(1 for m in module.modules() if isinstance(m, FSDPModule))


# ------------------------------------------------------------------ #
#  1. Basic fully_shard — class mutation, hooks, reshard on init
# ------------------------------------------------------------------ #


class TestFullyShardBasic:
    def test_module_class_becomes_fsdp(self):
        """fully_shard should dynamically convert the module class to a FSDPModule mixin."""
        torch.manual_seed(42)
        model = SimpleMLP(64).to(_device())
        original_cls = model.__class__
        wrapped = fully_shard(model)
        assert wrapped is model  # returns same object
        assert isinstance(wrapped, FSDPModule)
        assert FSDPModule in type(wrapped).__mro__
        assert original_cls in type(wrapped).__mro__

    def test_params_are_dtensor_after_reshard(self):
        """After fully_shard, module.reshard() is called, so params must be DTensors."""
        torch.manual_seed(42)
        model = SimpleMLP(64).to(_device())
        fully_shard(model)
        _assert_dtensor_params(model)

    def test_forward_without_errors(self):
        """A simple forward pass after fully_shard should succeed."""
        torch.manual_seed(42)
        model = SimpleMLP(64).to(_device())
        fully_shard(model)
        x = torch.randn(2, 64, device=_device())
        out = model(x)
        assert out.shape == (2, 64)

    def test_forward_backward_no_nan(self):
        """Forward + backward should produce finite loss and gradients."""
        torch.manual_seed(42)
        model = SimpleMLP(64).to(_device())
        fully_shard(model)
        x = torch.randn(2, 64, device=_device())
        loss = _forward_backward(model, x)
        assert not torch.isnan(torch.tensor(loss)), "Loss is NaN"
        assert not torch.isinf(torch.tensor(loss)), "Loss is Inf"

    def test_no_shard_forward_backward_finish_grad_sync(self):
        """no_shard keeps full replicated buffers and all-reduces at grad sync."""
        torch.manual_seed(42)
        model = SimpleMLP(64).to(_device())
        fully_shard(model, sharding_strategy="no_shard", enable_async_reduce_grad=False)

        x = torch.randn(2, 64, device=_device())
        _set_last_backward(model)
        loss = _forward_backward(model, x)
        assert not torch.isnan(torch.tensor(loss)), "Loss is NaN"
        model.finish_grad_sync()

        for param_group in model._fsdp_param_groups:
            assert param_group.model_weight_buffer is not None
            assert not param_group.model_weight_buffer.inner_sharded
            assert param_group.main_grad_buffer is not None
            assert not param_group.main_grad_buffer.inner_sharded
            for dist_grad in param_group.dist_grads:
                if dist_grad is None:
                    continue
                local_grad = dist_grad._local_tensor
                gathered = [torch.empty_like(local_grad) for _ in range(_world_size())]
                torch.distributed.all_gather(gathered, local_grad)
                for replica in gathered:
                    torch.testing.assert_close(replica, local_grad)

    @pytest.mark.parametrize(
        "sharding_strategy", ["no_shard", "optim", "optim_grads", "optim_grads_params"]
    )
    @pytest.mark.parametrize(
        ("model_dtype", "main_grad_dtype"),
        [
            pytest.param(torch.float32, None, id="fp32"),
            pytest.param(torch.bfloat16, torch.float32, id="bf16-param-fp32-grad"),
        ],
    )
    def test_cuda_graph_accumulates_microbatches(
        self, sharding_strategy, model_dtype, main_grad_dtype
    ):
        """Accumulate one eager and one replayed M-FSDP microbatch.

        :param sharding_strategy: M-FSDP gradient sharding strategy.
        :type sharding_strategy: str
        :param model_dtype: Compute parameter dtype.
        :type model_dtype: torch.dtype
        :param main_grad_dtype: Optimizer gradient dtype.
        :type main_grad_dtype: Optional[torch.dtype]
        """
        model = SimpleMLP(4, bias=True).to(_device(), dtype=model_dtype)
        fully_shard(
            model,
            sharding_strategy=sharding_strategy,
            mp_policy=MixedPrecisionPolicy(
                main_params_dtype=main_grad_dtype, main_grads_dtype=main_grad_dtype
            ),
            enable_unshard_prefetch=False,
            enable_async_reduce_grad=False,
            enable_cuda_graph=True,
        )

        for value in (2.0, 3.0):
            sample = torch.full(
                (2, 4), value, device=_device(), dtype=model_dtype, requires_grad=True
            )
            model(sample).sum().backward()
        model.finish_grad_sync()
        torch.cuda.synchronize()

        for param_names, param_group in model._named_param_groups:
            for name, dist_grad in zip(param_names, param_group.dist_grads):
                if dist_grad is None:
                    continue
                local_expected = 10.0 if name.endswith("weight") else 4.0
                expected = local_expected * _world_size()
                torch.testing.assert_close(
                    dist_grad.to_local(), torch.full_like(dist_grad.to_local(), expected)
                )

    @pytest.mark.parametrize(
        "enable_unshard_prefetch,enable_async_reduce_grad",
        [(False, False), (False, True), (True, False), (True, True)],
    )
    def test_feature_flags(self, enable_unshard_prefetch, enable_async_reduce_grad):
        """All combinations of overlap flags should work."""
        torch.manual_seed(42)
        model = SimpleMLP(64).to(_device())
        fully_shard(
            model,
            enable_unshard_prefetch=enable_unshard_prefetch,
            enable_async_reduce_grad=enable_async_reduce_grad,
        )
        x = torch.randn(2, 64, device=_device())
        out = model(x)
        loss = out.sum()
        loss.backward()
        assert not torch.isnan(torch.tensor(loss.item()))

    def test_unshard_coalescing_keeps_mixed_dtypes_separate(self, monkeypatch):
        """Coalesced all-gathers should not group buffers with different dtypes."""
        from megatron.core.distributed.fsdp.src.megatron_fsdp.v2 import (
            fsdp_module as fsdp_module_mod,
        )

        torch.manual_seed(42)
        model = MixedDtypeBuffers(16).to(_device())
        fully_shard(model, enable_unshard_prefetch=True, enable_async_reduce_grad=False)

        captured_run_dtypes = []

        def capture_unshard(
            outer_dp_group,
            inner_dp_group,
            weight_buffers,
            *,
            async_op,
            stream,
            caller_stream,
        ):
            del outer_dp_group, inner_dp_group, async_op, caller_stream
            captured_run_dtypes.append(
                tuple(weight_buffer.dtype for weight_buffer in weight_buffers)
            )
            for weight_buffer in weight_buffers:
                weight_buffer.unshard(
                    unshard_dim=1,
                    bind_params=True,
                    stream=stream,
                )

        monkeypatch.setattr(fsdp_module_mod, "_unshard_weight_buffers", capture_unshard)

        try:
            model.unshard(async_op=True)
            assert captured_run_dtypes
            assert {
                dtype for run_dtypes in captured_run_dtypes for dtype in run_dtypes
            } == {torch.float32, torch.uint8}
            assert len(captured_run_dtypes) >= 2
            for run_dtypes in captured_run_dtypes:
                assert len(set(run_dtypes)) == 1, (
                    "Unshard coalescing must keep mixed dtype buffers in separate runs, "
                    f"got {run_dtypes}"
                )
        finally:
            model.reshard()

    def test_prefetch_defers_post_unshard_to_caller_stream(self, monkeypatch):
        """Prefetch should allocate now but run post-processing when consumed."""
        torch.manual_seed(42)
        model = TinyLLM(vocab=32, hidden=16, num_layers=1).to(_device())
        layer = model.layers[0]
        fully_shard(layer, enable_unshard_prefetch=True, enable_async_reduce_grad=False)
        fully_shard(model, enable_unshard_prefetch=True, enable_async_reduce_grad=False)

        ctx = model._fsdp_root_context
        caller_stream = torch.cuda.current_stream()
        allocation_streams = []
        post_streams = []

        allocator = ctx.bucket_allocator
        original_allocate = allocator.allocate

        def capture_allocate(*args, **kwargs):
            allocation_streams.append(torch.cuda.current_stream())
            return original_allocate(*args, **kwargs)

        monkeypatch.setattr(allocator, "allocate", capture_allocate)
        for param_group in layer._fsdp_param_groups:
            original_post_unshard = param_group.post_unshard

            def capture_post_unshard(bwd_pass=False, *, _original=original_post_unshard):
                post_streams.append(torch.cuda.current_stream())
                return _original(bwd_pass=bwd_pass)

            monkeypatch.setattr(param_group, "post_unshard", capture_post_unshard)

        try:
            model.unshard(async_op=True)
            assert False in ctx.unshard_pending_post[id(layer)]
            assert not post_streams, "Prefetch must not run post-unshard processing"
            assert allocation_streams
            assert all(stream == caller_stream for stream in allocation_streams)

            model.reshard()
            layer.unshard(async_op=True)

            assert False not in ctx.unshard_pending_post[id(layer)]
            assert post_streams
            assert all(stream == caller_stream for stream in post_streams)
        finally:
            model.reshard()
            layer.reshard()

    @pytest.mark.parametrize("outer_strategy", ["no_shard", "optim"])
    def test_weight_unshard_coalesces_outer_before_inner(
        self, monkeypatch, outer_strategy
    ):
        """Outer runs should finish before inner AGs; no_shard outer is a no-op."""
        from megatron.core.distributed.fsdp.src.megatron_fsdp.v2 import (
            fsdp_module as fsdp_module_mod,
        )
        from megatron.core.distributed.fsdp.src.megatron_fsdp.v2.dp_buffer import (
            DataParallelBuffer,
        )

        torch.manual_seed(42)
        model = SimpleMLP(16).to(_device())
        model.fc.bias.requires_grad_(False)
        fully_shard(
            model,
            mesh=_build_hsdp_mesh(),
            sharding_strategy="optim_grads_params",
            outer_dp_sharding_strategy=outer_strategy,
            enable_unshard_prefetch=True,
            enable_async_reduce_grad=False,
        )

        original_unshard = DataParallelBuffer.unshard
        original_coalescing_manager = fsdp_module_mod._coalescing_manager
        original_all_gather = torch.distributed.all_gather_into_tensor
        manager_groups = []
        active_manager_groups = []
        unshard_calls = []
        collective_groups = []
        weight_buffers = []

        for param_group in model._fsdp_param_groups:
            weight_buffers.extend(param_group.weight_buffers_for_unshard())

        assert len(weight_buffers) == 2
        first_buffer, second_buffer = weight_buffers
        assert first_buffer.outer_dp_group is second_buffer.outer_dp_group
        assert first_buffer.inner_dp_group is second_buffer.inner_dp_group
        assert first_buffer.dtype == second_buffer.dtype
        assert first_buffer.device == second_buffer.device
        outer_dp_group = first_buffer.outer_dp_group
        inner_dp_group = first_buffer.inner_dp_group

        if outer_strategy == "optim":
            for weight_buffer in weight_buffers:
                # Force the post-optimizer/checkpoint state so dim 0 launches AG.
                weight_buffer._outer_dirty = True

        @contextmanager
        def capture_coalescing_manager(group, *args, **kwargs):
            manager_groups.append(group)
            with original_coalescing_manager(group, *args, **kwargs) as event:
                active_manager_groups.append(group)
                try:
                    yield event
                finally:
                    active_manager_groups.pop()

        def capture_unshard(buffer, *args, **kwargs):
            unshard_dim = kwargs.get("unshard_dim", args[0] if args else 1)
            active_group = (
                active_manager_groups[-1] if active_manager_groups else None
            )
            unshard_calls.append((id(buffer), unshard_dim, active_group))
            return original_unshard(buffer, *args, **kwargs)

        def capture_all_gather(*args, **kwargs):
            collective_groups.append(kwargs["group"])
            return original_all_gather(*args, **kwargs)

        monkeypatch.setattr(DataParallelBuffer, "unshard", capture_unshard)
        monkeypatch.setattr(
            fsdp_module_mod,
            "_coalescing_manager",
            capture_coalescing_manager,
        )
        monkeypatch.setattr(
            torch.distributed,
            "all_gather_into_tensor",
            capture_all_gather,
        )

        try:
            model.unshard(async_op=True)
            assert len(manager_groups) == 2
            assert all(
                actual is expected
                for actual, expected in zip(
                    manager_groups,
                    [outer_dp_group, inner_dp_group],
                )
            )
            expected_unshard_calls = [
                (id(first_buffer), 0, outer_dp_group),
                (id(second_buffer), 0, outer_dp_group),
                (id(first_buffer), 1, inner_dp_group),
                (id(second_buffer), 1, inner_dp_group),
            ]
            assert len(unshard_calls) == len(expected_unshard_calls)
            assert all(
                actual_buffer == expected_buffer
                and actual_dim == expected_dim
                and actual_group is expected_group
                for (actual_buffer, actual_dim, actual_group), (
                    expected_buffer,
                    expected_dim,
                    expected_group,
                ) in zip(unshard_calls, expected_unshard_calls)
            )
            expected_collective_groups = (
                [outer_dp_group, outer_dp_group]
                if outer_strategy == "optim"
                else []
            )
            expected_collective_groups.extend([inner_dp_group, inner_dp_group])
            assert len(collective_groups) == len(expected_collective_groups)
            assert all(
                actual is expected
                for actual, expected in zip(
                    collective_groups,
                    expected_collective_groups,
                )
            )
            assert all(not weight_buffer._outer_dirty for weight_buffer in weight_buffers)
        finally:
            model.reshard()

    def test_skipped_prefetch_waits_before_reshard(self, monkeypatch):
        """A skipped prefetched module must join its AG before freeing buffers."""
        torch.manual_seed(42)
        model = TinyLLM(vocab=32, hidden=16, num_layers=1).to(_device())
        layer = model.layers[0]
        fully_shard(layer, enable_unshard_prefetch=True, enable_async_reduce_grad=False)
        fully_shard(model, enable_unshard_prefetch=True, enable_async_reduce_grad=False)

        ctx = model._fsdp_root_context
        model.unshard(async_op=True)
        model.reshard()

        real_event = ctx.unshard_done_events[id(layer)]
        assert real_event is not None
        real_event.synchronize()

        order = []

        class CompletedEvent:
            def wait(self):
                order.append("wait")

        ctx.unshard_done_events[id(layer)] = CompletedEvent()
        for param_group in layer._fsdp_param_groups:
            original_reshard = param_group.reshard

            def capture_reshard(*, _original=original_reshard):
                order.append("reshard")
                return _original()

            monkeypatch.setattr(param_group, "reshard", capture_reshard)

        try:
            layer.reshard()
            assert order
            assert order[0] == "wait"
            assert False not in ctx.unshard_pending_post[id(layer)]
            assert ctx.unshard_done_events[id(layer)] is None
        finally:
            model.reshard()
            layer.reshard()


# ------------------------------------------------------------------ #
#  2. Scenarios — LLM-style nesting (embedding + layers + lm_head)
# ------------------------------------------------------------------ #


class TestLLMScenario:
    def test_llm_full_shard_root(self):
        """Shard the root TinyLLM — all params go into one FSDP module."""
        torch.manual_seed(42)
        model = TinyLLM(vocab=128, hidden=64, num_layers=2).to(_device())
        fully_shard(model)
        assert _count_fsdp_modules(model) == 1
        _assert_dtensor_params(model)

        x = torch.randint(0, 128, (4, 8), device=_device())
        loss = _forward_backward(model, x)
        assert not torch.isnan(torch.tensor(loss))

    def test_llm_per_layer_shard(self):
        """Shard each transformer layer individually (typical FSDP setup)."""
        torch.manual_seed(42)
        model = TinyLLM(vocab=128, hidden=64, num_layers=2).to(_device())
        # Shard each child layer separately
        for layer in model.layers:
            fully_shard(layer)
        # Shard the root (embedding + lm_head + norm covered)
        fully_shard(model)

        assert _count_fsdp_modules(model) == 3  # 2 layers + root
        _assert_dtensor_params(model)

        x = torch.randint(0, 128, (4, 8), device=_device())
        loss = _forward_backward(model, x)
        assert not torch.isnan(torch.tensor(loss))


# ------------------------------------------------------------------ #
#  3. Multimodal scenario — separate encoders + partial freezing
# ------------------------------------------------------------------ #


class TestMultimodalScenario:
    def test_frozen_vision_encoder(self):
        """Freeze vision encoder params; they should NOT be sharded (no grad)."""
        torch.manual_seed(42)
        model = MultimodalModel(hidden=64).to(_device())
        # Freeze vision encoder
        for p in model.vision_encoder.parameters():
            p.requires_grad = False
        fully_shard(model)

        x_img = torch.randn(2, 64, device=_device(), requires_grad=True)
        x_txt = torch.randn(2, 64, device=_device(), requires_grad=True)
        out = model(x_img, x_txt)
        loss = out.sum()
        loss.backward()

        # Frozen params should have no grad after reduce_grad
        for name, p in model.named_parameters():
            if "vision_encoder" in name:
                assert not p.requires_grad, f"Frozen param {name} should have requires_grad=False"

        assert not torch.isnan(torch.tensor(loss.item()))

    def test_frozen_params_in_own_group(self):
        """Frozen params are included but grouped separately (different requires_grad)."""
        torch.manual_seed(42)
        model = MultimodalModel(hidden=64).to(_device())
        for p in model.vision_encoder.parameters():
            p.requires_grad = False
        fully_shard(model)

        # Frozen params should be in a group with requires_grad=False
        has_frozen_group = False
        for param_group in model._fsdp_param_groups:
            if not param_group.requires_grad:
                has_frozen_group = True
                for p in param_group.params:
                    assert (
                        not p.requires_grad
                    ), "Param in frozen group should have requires_grad=False"
        assert has_frozen_group, "Frozen params should be in their own param group"

    def test_mixed_frozen_and_trainable(self):
        """Some parts frozen, some trainable — all sharded together."""
        torch.manual_seed(42)
        model = MultimodalModel(hidden=64).to(_device())
        # Freeze only vision, text and fusion stay trainable
        for p in model.vision_encoder.parameters():
            p.requires_grad = False
        fully_shard(model)

        x_img = torch.randn(2, 64, device=_device(), requires_grad=True)
        x_txt = torch.randn(2, 64, device=_device(), requires_grad=True)

        # Should run without error
        out = model(x_img, x_txt)
        loss = out.sum()
        loss.backward()
        assert not torch.isnan(torch.tensor(loss.item()))


# ------------------------------------------------------------------ #
#  4. Nested FSDP — expert-in-layer (EDP pattern)
# ------------------------------------------------------------------ #


class TestNestedFSDP:
    def test_nested_expert_in_layer(self):
        """Shard experts inside layer, then shard layer, then root — EDP pattern."""
        torch.manual_seed(42)
        device = _device()

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = MOETransformerLayer(64, 128)
                self.head = nn.Linear(64, 10)

            def forward(self, x):
                return self.head(self.layer(x))

        model = Model().to(device)

        # Step 1: shard the expert (nested FSDP)
        model.layer.experts = fully_shard(model.layer.experts)
        # Step 2: shard the layer (will detect nested FSDP)
        model.layer = fully_shard(model.layer)
        # Step 3: shard the root
        model = fully_shard(model)

        assert _count_fsdp_modules(model) == 3  # experts, layer, root

        x = torch.randn(2, 64, device=device, requires_grad=True)
        loss = _forward_backward(model, x)
        assert not torch.isnan(torch.tensor(loss))

    def test_nested_ignored_params_are_skipped(self):
        """Nested FSDP module params must be in the parent's ignored_params."""
        torch.manual_seed(42)
        device = _device()
        model = MOETransformerLayer(64, 128).to(device)

        expert = fully_shard(model.experts)
        model.experts = expert  # rebind (fully_shard returns same object)
        model = fully_shard(model)

        # The outer FSDP (model) should have a parameter group that does NOT
        # include the inner FSDP's (expert) params
        expert_param_ids = set(id(p) for p in expert.parameters())
        for _, param_group in model._named_param_groups:
            for p in param_group.params:
                assert (
                    id(p) not in expert_param_ids
                ), "Nested FSDP param leaked into parent param group"

    def test_nested_forward_backward(self):
        """Nested FSDP forward+backward produces correct loss pattern."""
        torch.manual_seed(42)
        device = _device()
        model = MOETransformerLayer(64, 128).to(device)

        model.experts = fully_shard(model.experts)
        model = fully_shard(model)

        # Run twice — second pass should work correctly (buffer reuse)
        for _ in range(2):
            x = torch.randn(2, 64, device=device, requires_grad=True)
            out = model(x)
            loss = out.sum()
            loss.backward()
            assert not torch.isnan(torch.tensor(loss.item()))


# ------------------------------------------------------------------ #
#  5. Mixed precision policies
# ------------------------------------------------------------------ #


class TestMixedPrecision:
    def test_main_params_fp32(self):
        """With fp32 main params, main_weight_buffer should be created."""
        torch.manual_seed(42)
        mp_policy = MixedPrecisionPolicy(main_params_dtype=torch.float32)
        model = SimpleMLP(64).to(_device()).bfloat16()
        fully_shard(model, mp_policy=mp_policy)

        # Verify main_weight_buffer exists and is fp32
        for param_group in model._fsdp_param_groups:
            if param_group.main_weight_buffer is not None:
                assert (
                    param_group.main_weight_buffer.dtype == torch.float32
                ), f"Expected fp32 main weight buffer, got {param_group.main_weight_buffer.dtype}"

    def test_main_params_none(self):
        """With no main_params_dtype, no main_weight_buffer should be created."""
        torch.manual_seed(42)
        mp_policy = MixedPrecisionPolicy(main_params_dtype=None)
        model = SimpleMLP(64).to(_device())
        fully_shard(model, mp_policy=mp_policy)

        for param_group in model._fsdp_param_groups:
            assert (
                param_group.main_weight_buffer is None
            ), "main_weight_buffer should be None when main_params_dtype is None"

    def test_fp32_grad_reduce(self):
        """grad_reduce_in_fp32=True should use fp32 gradient communication."""
        torch.manual_seed(42)
        mp_policy = MixedPrecisionPolicy(grad_comm_dtype=torch.float32)
        model = SimpleMLP(64).to(_device()).bfloat16()
        fully_shard(model, mp_policy=mp_policy)

        x = torch.randn(2, 64, device=_device(), dtype=torch.bfloat16, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        assert not torch.isnan(torch.tensor(loss.item()))


# ------------------------------------------------------------------ #
#  6. ignored_params
# ------------------------------------------------------------------ #


class TestIgnoredParams:
    def test_ignored_params_excluded_from_groups(self):
        """Params passed as ignored_params should not appear in FSDP groups."""
        torch.manual_seed(42)
        model = SimpleMLP(64).to(_device())
        # Pre-identify the param to ignore before wrapping
        ignored = {model.fc.weight}
        fully_shard(model, ignored_params=ignored)

        for param_group in model._fsdp_param_groups:
            for p in param_group.params:
                assert p is not model.fc.weight, "Ignored param leaked into group"

    def test_ignored_param_stays_on_module(self):
        """Ignored param should remain as a regular nn.Parameter on the module."""
        torch.manual_seed(42)
        model = SimpleMLP(64).to(_device())
        original_weight = model.fc.weight
        ignored = {model.fc.weight}
        fully_shard(model, ignored_params=ignored)

        # After fully_shard and reshard, the ignored weight should still be
        # the original nn.Parameter (not a DTensor) on the module
        from torch.distributed.tensor import DTensor

        assert not isinstance(
            model.fc.weight, DTensor
        ), "Ignored param should not be converted to DTensor"
        assert model.fc.weight is original_weight, "Ignored param identity changed"


# ------------------------------------------------------------------ #
#  7. Forward/backward lifecycle correctness
# ------------------------------------------------------------------ #


class TestLifecycle:
    def test_params_unsharded_during_forward(self):
        """During forward, model parameters should be in unsharded state (full tensors)."""
        torch.manual_seed(42)
        model = SimpleMLP(64).to(_device())

        captured_shapes = []

        def hook(module, inp):
            # At this point, the pre-forward hook has already unsharded
            for name, p in module.named_parameters():
                captured_shapes.append((name, p.data.shape))

        model.register_forward_pre_hook(hook)
        fully_shard(model)

        # Before forward, params should be DTensors (reshard called at init)
        from torch.distributed.tensor import DTensor

        for _, p in model.named_parameters():
            assert isinstance(p, DTensor), "Params should be DTensors after init reshard"

        x = torch.randn(2, 64, device=_device())
        model(x)

        # Inside the forward pre-hook, params should be full tensors
        for name, shape in captured_shapes:
            assert shape == torch.Size([64, 64]) or shape == torch.Size(
                [64]
            ), f"Param {name} has wrong shape during forward: {shape}"

    def test_params_resharded_after_forward(self):
        """After forward, model parameters should be resharded back to DTensors."""
        torch.manual_seed(42)
        model = SimpleMLP(64).to(_device())
        fully_shard(model)

        from torch.distributed.tensor import DTensor

        x = torch.randn(2, 64, device=_device())
        model(x)

        for _, p in model.named_parameters():
            assert isinstance(p, DTensor), "Params should be DTensors after forward reshard"

    def test_params_unsharded_during_backward(self):
        """During backward, model parameters should be unsharded."""
        torch.manual_seed(42)
        model = SimpleMLP(64).to(_device())
        fully_shard(model)

        from torch.distributed.tensor import DTensor

        captured_dtensor = []

        def grad_hook(grad):
            for _, p in model.named_parameters():
                captured_dtensor.append(isinstance(p, DTensor))

        x = torch.randn(2, 64, device=_device(), requires_grad=True)
        out = model(x)
        out.register_hook(grad_hook)
        loss = out.sum()
        loss.backward()

        # During backward (grad_hook fires during backward pass), params should NOT be DTensor
        for is_dt in captured_dtensor:
            assert not is_dt, "Params should be unsharded during backward"

    def test_params_resharded_after_backward(self):
        """After full backward pass, params should be DTensors again."""
        torch.manual_seed(42)
        model = SimpleMLP(64).to(_device())
        fully_shard(model)

        from torch.distributed.tensor import DTensor

        x = torch.randn(2, 64, device=_device(), requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()

        for _, p in model.named_parameters():
            assert isinstance(p, DTensor), "Params should be DTensors after backward reshard"


# ------------------------------------------------------------------ #
#  8. Activation checkpointing
# ------------------------------------------------------------------ #


class MLPWithCheckpointing(nn.Module):
    """A multi-layer MLP that supports activation checkpointing on its blocks."""

    def __init__(self, hidden=64, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
                for _ in range(num_layers)
            ]
        )
        self._use_activation_checkpointing = False

    def forward(self, x):
        for layer in self.layers:
            if self._use_activation_checkpointing:
                x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        return x

    def enable_activation_checkpointing(self):
        self._use_activation_checkpointing = True


class LargePerLayerModel(nn.Module):
    """Multi-layer model with individually wrapped FSDP layers and optional
    activation checkpointing support."""

    def __init__(self, hidden=256, num_layers=4):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
                for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TestActivationCheckpointing:
    def test_recompute_successor_uses_updated_weight_after_optimizer_step(self):
        """A recompute-prefetched successor must not reuse pre-step weights.

        Layer 1 finishes backward before layer 0 is recomputed. A normal
        forward-prefetch from that recompute can incorrectly resurrect layer
        1's full model-weight buffer after its post-backward reshard. Under
        outer ``no_shard``, copying the optimizer's FP32 main shard updates
        only persistent BF16 storage, so the resurrected full buffer would be
        stale on the next forward.

        Use per-layer FSDP units (and an FSDP root), but no nested expert unit:
        this isolates successor prefetch from the separate nested post-forward
        lifecycle path.
        """
        torch.manual_seed(42)
        device = _device()
        mesh = _build_hsdp_mesh()

        class TwoLayerCheckpointModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.Linear(32, 32), nn.GELU(), nn.Linear(32, 32)
                        )
                        for _ in range(2)
                    ]
                )

            def forward(self, x):
                for layer in self.layers:
                    x = torch.utils.checkpoint.checkpoint(
                        layer, x, use_reentrant=False
                    )
                return x

        model = TwoLayerCheckpointModel().to(device=device, dtype=torch.bfloat16)
        shard_kwargs = dict(
            mesh=mesh,
            sharding_strategy="optim_grads_params",
            outer_dp_sharding_strategy="no_shard",
            mp_policy=MixedPrecisionPolicy(
                main_params_dtype=torch.float32,
                main_grads_dtype=torch.float32,
                grad_comm_dtype=torch.float32,
            ),
            enable_unshard_prefetch=True,
            enable_async_reduce_grad=True,
        )
        for index, layer in enumerate(model.layers):
            model.layers[index] = fully_shard(layer, **shard_kwargs)
        model = fully_shard(model, **shard_kwargs)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.25)

        successor = model.layers[1]
        param_group = successor._fsdp_param_groups[0]
        model_buffer = param_group.model_weight_buffer
        main_buffer = param_group.main_weight_buffer
        assert main_buffer is not None
        assert model_buffer.storage_shard_layout == (0, 1)
        assert main_buffer.storage_shard_layout == (0, 1)

        _set_last_backward(model)
        x = torch.randn(4, 32, device=device, dtype=torch.bfloat16, requires_grad=True)
        with torch.utils.checkpoint.set_checkpoint_early_stop(False):
            loss = model(x).float().square().mean()
            loss.backward()
        model.finish_grad_sync()

        main_before = main_buffer.data.detach().clone()
        optimizer.step()
        assert not torch.equal(main_buffer.data, main_before), "SGD did not update successor"
        model._copy_main_weights_to_model_weights()
        assert torch.equal(model_buffer.data, main_buffer.data.to(model_buffer.dtype))

        expected_full = torch.empty(
            model_buffer.buffer_index.bucket_meta.size,
            dtype=model_buffer.dtype,
            device=device,
        )
        torch.distributed.all_gather_into_tensor(
            expected_full,
            model_buffer.data,
            group=model_buffer.inner_dp_group,
        )

        observed_full = []

        def capture_successor_full_buffer(_module, _args):
            assert model_buffer._unsharded_buffer is not None
            observed_full.append(model_buffer._unsharded_buffer.detach().clone())

        handle = successor.register_forward_pre_hook(capture_successor_full_buffer)
        try:
            x_next = torch.randn(4, 32, device=device, dtype=torch.bfloat16)
            model(x_next)
        finally:
            handle.remove()

        assert len(observed_full) == 1
        for item_id in range(len(param_group.params)):
            start, end = model_buffer.buffer_index._get_item_global_range(item_id)
            torch.testing.assert_close(
                observed_full[0][start:end],
                expected_full[start:end],
                rtol=0,
                atol=0,
            )

    def test_recompute_forward_self_unshard_disables_prefetch(self, monkeypatch):
        """Recompute may unshard itself but must not advance forward prefetch."""
        torch.manual_seed(42)
        model = TinyLLM(vocab=32, hidden=16, num_layers=1).to(_device())
        target = model.layers[0]
        fully_shard(
            target,
            enable_unshard_prefetch=True,
            enable_async_reduce_grad=False,
        )
        fully_shard(
            model,
            enable_unshard_prefetch=True,
            enable_async_reduce_grad=False,
        )

        assert not target._fsdp_state._is_root
        ctx = model._fsdp_root_context
        ctx.backward_phase = True
        ctx.backward_module = id(target)

        calls = []

        def capture_unshard(
            async_op=False,
            bwd_pass=False,
            prefetch=True,
        ):
            calls.append((async_op, bwd_pass, prefetch))

        monkeypatch.setattr(target, "unshard", capture_unshard)
        mfsdp_forward_pre_hook(target, (), {})

        assert calls == [
            (True, True, True),
            (True, False, False),
        ]

    def test_activation_checkpointing_forward_backward(self):
        """Forward + backward with activation checkpointing should produce finite loss."""
        torch.manual_seed(42)
        device = _device()
        model = MLPWithCheckpointing(hidden=64, num_layers=4).to(device)
        model.enable_activation_checkpointing()

        for layer in model.layers:
            fully_shard(layer)
        fully_shard(model)
        _assert_dtensor_params(model)

        x = torch.randn(2, 64, device=device, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()

        assert not torch.isnan(torch.tensor(loss.item())), "Loss is NaN"
        assert not torch.isinf(torch.tensor(loss.item())), "Loss is Inf"

    def test_activation_checkpointing_multi_step(self):
        """Multiple forward+backward steps with activation checkpointing should be stable."""
        torch.manual_seed(42)
        device = _device()
        model = MLPWithCheckpointing(hidden=64, num_layers=4).to(device)
        model.enable_activation_checkpointing()

        for layer in model.layers:
            fully_shard(layer)
        fully_shard(model)

        losses = []
        for step in range(4):
            torch.manual_seed(step)
            x = torch.randn(2, 64, device=device, requires_grad=True)
            out = model(x)
            loss = out.sum()
            loss.backward()
            losses.append(loss.item())

        for i, loss_val in enumerate(losses):
            assert not torch.isnan(torch.tensor(loss_val)), f"Loss at step {i} is NaN"
            assert not torch.isinf(torch.tensor(loss_val)), f"Loss at step {i} is Inf"

    def test_activation_checkpointing_with_overlap(self):
        """Activation checkpointing should work with unshard_prefetch and async_reduce_grad."""
        torch.manual_seed(42)
        device = _device()
        model = MLPWithCheckpointing(hidden=128, num_layers=4).to(device)
        model.enable_activation_checkpointing()

        for layer in model.layers:
            fully_shard(layer, enable_unshard_prefetch=True, enable_async_reduce_grad=True)
        fully_shard(model, enable_unshard_prefetch=True, enable_async_reduce_grad=True)

        x = torch.randn(2, 128, device=device, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()

        assert not torch.isnan(torch.tensor(loss.item()))

    def test_activation_checkpointing_nested_fsdp(self):
        """Activation checkpointing with nested FSDP (expert-in-layer) should work."""
        torch.manual_seed(42)
        device = _device()

        class NestedCheckpointModel(nn.Module):
            def __init__(self, hidden=64):
                super().__init__()
                self.attn = nn.Linear(hidden, hidden)
                self.experts = nn.Sequential(
                    nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden)
                )
                self.norm = nn.LayerNorm(hidden)

            def forward(self, x):
                h = self.attn(x)
                if self._use_activation_checkpointing:
                    h = torch.utils.checkpoint.checkpoint(self.experts, h, use_reentrant=False)
                else:
                    h = self.experts(h)
                return self.norm(h + x)

        model = NestedCheckpointModel(hidden=64).to(device)
        model._use_activation_checkpointing = True
        model.experts = fully_shard(model.experts)
        model = fully_shard(model)

        x = torch.randn(2, 64, device=device, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()

        assert not torch.isnan(torch.tensor(loss.item()))

    def test_activation_checkpointing_disabled_vs_enabled_same_loss(self):
        """With same inputs and no parameter updates, checkpointed and non-checkpointed
        forward should produce the same output (checkpoint recompute is numerically transparent)."""
        torch.manual_seed(42)
        device = _device()

        model = MLPWithCheckpointing(hidden=64, num_layers=3).to(device)

        for layer in model.layers:
            fully_shard(layer)
        fully_shard(model)

        x = torch.randn(2, 64, device=device, requires_grad=True)

        # Forward without activation checkpointing
        model._use_activation_checkpointing = False
        out_no_ckpt = model(x)

        # Forward with activation checkpointing
        torch.manual_seed(42)
        x2 = torch.randn(2, 64, device=device, requires_grad=True)
        model._use_activation_checkpointing = True
        out_ckpt = model(x2)

        assert torch.allclose(
            out_no_ckpt, out_ckpt, atol=1e-5
        ), "Checkpointing changed forward output (same inputs)"

    def test_activation_checkpointing_per_layer_shard_with_ckpt(self):
        """Per-layer FSDP with activation checkpointing on each layer — full training step."""
        torch.manual_seed(42)
        device = _device()
        model = LargePerLayerModel(hidden=256, num_layers=6).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        for layer in model.layers:
            fully_shard(layer)

        fully_shard(model)

        # Use checkpoint on every other layer to test mixed use
        def ckpt_forward(x):
            for i, layer in enumerate(model.layers):
                if i % 2 == 0:
                    x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
                else:
                    x = layer(x)
            return x

        x = torch.randn(4, 256, device=device, requires_grad=True)
        out = ckpt_forward(x)
        loss = out.sum()
        loss.backward()
        optimizer.step()

        assert not torch.isnan(torch.tensor(loss.item()))


# ------------------------------------------------------------------ #
#  9. Safety — double-shard rejection
# ------------------------------------------------------------------ #


class TestSafety:
    def test_double_shard_rejected(self):
        """Calling fully_shard on an already-wrapped module should raise ValueError."""
        torch.manual_seed(42)
        model = SimpleMLP(64).to(_device())
        fully_shard(model)
        with pytest.raises(ValueError, match="already been fully sharded"):
            fully_shard(model)

    def test_no_params_module_ok(self):
        """fully_shard on a module with no parameters should succeed (no-op)."""
        model = nn.Sequential().to(_device())
        wrapped = fully_shard(model)
        assert isinstance(wrapped, FSDPModule)
        assert _count_fsdp_modules(wrapped) == 1


# ------------------------------------------------------------------ #
# 10. Checkpoint — get_state_dict and preprocess_state_dict_for_uneven_dtensor
# ------------------------------------------------------------------ #


class TestCheckpoint:
    def test_get_state_dict_returns_dicts(self):
        """get_state_dict should return model and optimizer state dicts."""
        torch.manual_seed(42)
        model = SimpleMLP(64).to(_device())
        fully_shard(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # Run one step to populate optimizer state
        x = torch.randn(2, 64, device=_device())
        out = model(x)
        loss = out.sum()
        loss.backward()
        optimizer.step()

        model_sd, opt_sd = get_state_dict(model, optimizer)
        assert isinstance(model_sd, dict)
        assert isinstance(opt_sd, dict)
        assert len(model_sd) > 0, "Model state dict should not be empty"

    def test_get_state_dict_nested_fsdp(self):
        """get_state_dict should work with nested FSDP modules."""
        torch.manual_seed(42)
        device = _device()
        model = MOETransformerLayer(64, 128).to(device)
        model.experts = fully_shard(model.experts)
        model = fully_shard(model)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        x = torch.randn(2, 64, device=device)
        out = model(x)
        loss = out.sum()
        loss.backward()
        optimizer.step()

        model_sd, opt_sd = get_state_dict(model, optimizer)
        assert len(model_sd) > 0

    @pytest.mark.skip(reason="Hangs. Debug in progress.")
    def test_preprocess_state_dict_adds_metadata(self):
        """preprocess_state_dict_for_uneven_dtensor should add chunk metadata."""
        torch.manual_seed(42)
        model = SimpleMLP(64).to(_device())
        fully_shard(model)
        opt = torch.optim.SGD(model.parameters(), lr=0.0)

        # Build a raw state dict via torch's state_dict
        sd = torch_get_state_dict(
            model, opt, options=StateDictOptions(full_state_dict=True, cpu_offload=True)
        )
        preprocess_state_dict_for_uneven_dtensor(sd)

        # Check that the state dict still contains parameter data
        assert len(sd) > 0

    def test_get_state_dict_strict_all_dtensor(self):
        """get_state_dict should assert all params are DTensors."""
        torch.manual_seed(42)
        model = SimpleMLP(64).to(_device())
        # DON'T call fully_shard — params are NOT DTensors
        optimizer = torch.optim.AdamW(model.parameters())

        with pytest.raises(AssertionError, match="Expected all parameters to be DTensors"):
            get_state_dict(model, optimizer)

    def test_get_state_dict_llm_scenario(self):
        """Full LLM forward-backward-checkpoint cycle should work."""
        torch.manual_seed(42)
        device = _device()
        model = TinyLLM(vocab=128, hidden=64, num_layers=2).to(device)
        for layer in model.layers:
            fully_shard(layer)
        fully_shard(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        x = torch.randint(0, 128, (4, 8), device=device)
        out = model(x)
        loss = out.sum()
        loss.backward()
        optimizer.step()

        model_sd, opt_sd = get_state_dict(model, optimizer)
        assert len(model_sd) > 0
        assert len(opt_sd) > 0

    def test_get_state_dict_with_frozen_params(self):
        """get_state_dict should work with mixed frozen/trainable params."""
        torch.manual_seed(42)
        device = _device()
        model = MultimodalModel(hidden=64).to(device)
        for p in model.vision_encoder.parameters():
            p.requires_grad = False
        fully_shard(model)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)

        x_img = torch.randn(2, 64, device=device, requires_grad=True)
        x_txt = torch.randn(2, 64, device=device, requires_grad=True)
        out = model(x_img, x_txt)
        loss = out.sum()
        loss.backward()
        optimizer.step()

        model_sd, opt_sd = get_state_dict(model, optimizer)
        assert len(model_sd) > 0

    def test_get_state_dict_hsdp_outer_optim(self):
        """HSDP outer-optim checkpoint state should survive a DCP roundtrip."""
        from torch.distributed.tensor import DTensor
        from torch.distributed.tensor.placement_types import Shard

        def build_model_and_optimizer(seed):
            torch.manual_seed(seed)
            model = SimpleMLP(64).to(device)
            fully_shard(
                model,
                mesh=mesh,
                sharding_strategy="optim_grads_params",
                outer_dp_sharding_strategy="optim",
                mp_policy=MixedPrecisionPolicy(
                    main_params_dtype=torch.float32,
                    main_grads_dtype=torch.float32,
                ),
                enable_async_reduce_grad=False,
            )
            return model, torch.optim.AdamW(model.parameters(), lr=1e-3)

        def run_one_step(model, optimizer, seed):
            torch.manual_seed(seed)
            x = torch.randn(2, 64, device=device)
            _set_last_backward(model)
            loss = model(x).sum()
            loss.backward()
            model.finish_grad_sync()
            optimizer.step()

        def clone_dtensor_values(state_dict):
            return {
                name: value.to_local().detach().clone()
                for name, value in state_dict.items()
                if isinstance(value, DTensor)
            }

        def clone_optimizer_dtensor_values(state_dict):
            values = {}
            for name, state_tensors in state_dict.get("state", {}).items():
                values[name] = {
                    key: value.to_local().detach().clone()
                    for key, value in state_tensors.items()
                    if isinstance(value, DTensor) and value.to_local().dim() > 0
                }
            return {name: tensors for name, tensors in values.items() if tensors}

        def assert_hsdp_dtensor_metadata(dtensor):
            assert len(dtensor.placements) == 2
            assert isinstance(dtensor.placements[0], Shard)
            assert isinstance(dtensor.placements[1], Shard)
            assert hasattr(dtensor._local_tensor, "__create_chunk_list__")
            assert hasattr(dtensor._local_tensor, "__create_write_items__")

        device = _device()
        mesh = _build_hsdp_mesh()
        model, optimizer = build_model_and_optimizer(seed=42)
        run_one_step(model, optimizer, seed=43)

        model_sd, opt_sd = get_state_dict(model, optimizer)
        expected_model = clone_dtensor_values(model_sd)
        expected_optim = clone_optimizer_dtensor_values(opt_sd)
        assert expected_model, "HSDP model checkpoint should contain DTensor params"
        assert expected_optim, "HSDP optimizer checkpoint should contain DTensor state"

        for dtensor in (value for value in model_sd.values() if isinstance(value, DTensor)):
            assert_hsdp_dtensor_metadata(dtensor)
        for state_tensors in opt_sd.get("state", {}).values():
            for value in state_tensors.values():
                if isinstance(value, DTensor) and value.to_local().dim() > 0:
                    assert_hsdp_dtensor_metadata(value)

        ckpt_dir = Path(SHARED_TMP_DIR) / "test_get_state_dict_hsdp_outer_optim"
        if _rank() == 0:
            shutil.rmtree(ckpt_dir, ignore_errors=True)
            ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.distributed.barrier()

        dcp.save({"model": model_sd, "optimizer": opt_sd}, checkpoint_id=str(ckpt_dir))
        torch.distributed.barrier()

        load_model, load_optimizer = build_model_and_optimizer(seed=123)
        run_one_step(load_model, load_optimizer, seed=124)
        load_model_sd, load_opt_sd = get_state_dict(load_model, load_optimizer)
        dcp.load({"model": load_model_sd, "optimizer": load_opt_sd}, checkpoint_id=str(ckpt_dir))
        torch_set_state_dict(
            load_model,
            load_optimizer,
            model_state_dict=load_model_sd,
            optim_state_dict=load_opt_sd,
            options=StateDictOptions(strict=False),
        )

        loaded_model_sd, loaded_opt_sd = get_state_dict(load_model, load_optimizer)
        loaded_model = clone_dtensor_values(loaded_model_sd)
        loaded_optim = clone_optimizer_dtensor_values(loaded_opt_sd)

        assert loaded_model.keys() == expected_model.keys()
        for name, expected in expected_model.items():
            assert torch.allclose(loaded_model[name], expected), name

        assert loaded_optim.keys() == expected_optim.keys()
        for name, expected_tensors in expected_optim.items():
            assert loaded_optim[name].keys() == expected_tensors.keys()
            for key, expected in expected_tensors.items():
                assert torch.allclose(loaded_optim[name][key], expected), f"{name}.{key}"

        if _rank() == 0:
            shutil.rmtree(ckpt_dir, ignore_errors=True)
        torch.distributed.barrier()
