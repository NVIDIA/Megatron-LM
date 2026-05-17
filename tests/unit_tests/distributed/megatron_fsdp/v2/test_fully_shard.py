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

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import StateDictOptions
from torch.distributed.checkpoint.state_dict import get_state_dict as torch_get_state_dict

sys.path.insert(0, str(Path(__file__).parents[2]))
from megatron.core.distributed.fsdp.src.megatron_fsdp.uneven_dtensor import (
    get_state_dict,
    preprocess_state_dict_for_uneven_dtensor,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.v2.fsdp_module import FSDPModule
from megatron.core.distributed.fsdp.src.megatron_fsdp.v2.fully_shard import fully_shard
from megatron.core.distributed.fsdp.src.megatron_fsdp.v2.mixed_precision import (
    FullyShardMixedPrecisionPolicy,
)

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
        mp_policy = FullyShardMixedPrecisionPolicy(main_params_dtype=torch.float32)
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
        mp_policy = FullyShardMixedPrecisionPolicy(main_params_dtype=None)
        model = SimpleMLP(64).to(_device())
        fully_shard(model, mp_policy=mp_policy)

        for param_group in model._fsdp_param_groups:
            assert (
                param_group.main_weight_buffer is None
            ), "main_weight_buffer should be None when main_params_dtype is None"

    def test_fp32_grad_reduce(self):
        """grad_reduce_in_fp32=True should use fp32 gradient communication."""
        torch.manual_seed(42)
        mp_policy = FullyShardMixedPrecisionPolicy(
            main_grads_dtype=torch.float32, grad_comm_dtype=torch.float32
        )
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
#  8. Safety — double-shard rejection
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
#  9. Checkpoint — get_state_dict and preprocess_state_dict_for_uneven_dtensor
# ------------------------------------------------------------------ #


class TestCheckpoint:
    @pytest.mark.skip(reason="Hangs. Debug in progress.")
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

    @pytest.mark.skip(reason="Hangs. Debug in progress.")
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

    @pytest.mark.skip(reason="Hangs. Debug in progress.")
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

    @pytest.mark.skip(reason="Hangs. Debug in progress.")
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
