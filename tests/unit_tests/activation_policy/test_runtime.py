# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Unit tests for the activation-strategy runtime on a small owner module."""

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from megatron.core.activation_policy import (
    CheckpointStrategy,
    CheckpointWithoutOutputStrategy,
    Event,
    Strategy,
    has_strategy,
    install_activation_strategies,
    installed_strategies,
)
from megatron.core.activation_policy import runtime as runtime_module
from megatron.core.activation_policy.runtime import (
    _ModulePointHook,
    call_without_activation_strategies,
)
from megatron.core.tensor_parallel.random import initialize_rng_tracker
from megatron.core.transformer.module import MegatronModule


class _Owner(nn.Module):
    """norm -> linear branch -> residual add, the shape of a transformer sub-block."""

    def __init__(self, hidden=16):
        super().__init__()
        self.config = _Config()
        self.pg_collection = SimpleNamespace(tp=None)
        self.norm = nn.LayerNorm(hidden)
        self.linear = nn.Linear(hidden, hidden)

    def _bda(self, branch_out, residual):
        return branch_out + residual

    def forward(self, x):
        normed = self.norm(x)
        branch = self.linear(normed)
        return self._bda(branch, x)


def _run(owner, x):
    x = x.detach().clone().requires_grad_(True)
    out = owner(x)
    out.sum().backward()
    grads = {n: p.grad.detach().clone() for n, p in owner.named_parameters()}
    return out.detach(), x.grad.detach(), grads


@pytest.fixture
def rng_tracker():
    # The checkpoint primitives save and restore the CUDA RNG tracker's states.
    initialize_rng_tracker(force_reset=True)


def _install(owner, *strategies):
    install_activation_strategies(owner, strategies)


def _recompute_norm(owner):
    return CheckpointWithoutOutputStrategy(
        region=owner.norm, release_event=owner._bda, release_on_exit=False
    )


class TestInstall:
    def test_without_strategies_nothing_is_hooked(self):
        owner = _Owner()
        x = torch.randn(2, 16)
        torch.testing.assert_close(owner(x), owner.linear(owner.norm(x)) + x)
        assert installed_strategies(owner) == {}
        assert "_bda" not in owner.__dict__ and "_call_impl" not in owner.norm.__dict__

    def test_strategy_observes_raw_arguments_and_return_value(self):
        seen = []

        class Observer(Strategy):
            def observes(self):
                return (owner._bda,)

            def run(self, owner, defer, forward, args, kwargs):
                defer(Event.enter(owner._bda), self, seen.append)
                defer(Event.exit(owner._bda), self, seen.append)
                return forward(*args, **kwargs)

        owner = _Owner()
        _install(owner, Observer(owner.norm))
        owner.norm(torch.zeros(16))
        assert owner._bda(2, residual=3) == 5
        assert seen == [((2,), {"residual": 3}), 5]

    def test_failed_replacement_keeps_previous_hooks(self):
        owner = _Owner()
        original = _recompute_norm(owner)
        _install(owner, original)
        hook = owner.norm._call_impl
        with pytest.raises(ValueError, match="method of its owner"):
            _install(owner, CheckpointStrategy(_Owner()._bda), CheckpointStrategy(owner.linear))
        assert owner.norm._call_impl is hook
        assert "_call_impl" not in owner.linear.__dict__
        assert runtime_module.installed_strategy(owner, owner.norm) is original

    def test_method_points_are_hooked_transparently(self):
        owner = _Owner()
        bound = owner._bda
        _install(owner, CheckpointStrategy(owner._bda))
        hook = owner.__dict__["_bda"]
        # The hook stands for the bound method: the same point before and after install.
        assert hook == bound and hash(hook) == hash(bound)
        assert has_strategy(owner, owner._bda, CheckpointStrategy)
        assert has_strategy(owner, bound, CheckpointStrategy)
        assert owner._bda(torch.ones(2), torch.ones(2)).tolist() == [2.0, 2.0]
        runtime_module._uninstall(owner)
        assert "_bda" not in owner.__dict__

    def test_install_wraps_named_children_and_uninstall_restores(self):
        owner = _Owner()
        _install(owner, _recompute_norm(owner))
        assert "_call_impl" in owner.norm.__dict__
        assert "_call_impl" not in owner.linear.__dict__  # not observed
        runtime_module._uninstall(owner)
        assert "_call_impl" not in owner.norm.__dict__
        assert installed_strategies(owner) == {}

    def test_points_given_by_name_are_rejected(self):
        owner = _Owner()
        with pytest.raises(TypeError, match="child module or a method"):
            _install(owner, CheckpointStrategy("_bda"))

    def test_deepcopy_remaps_owner(self):
        owner = _Owner()
        _install(owner, _recompute_norm(owner))
        clone = copy.deepcopy(owner)
        wrapper = clone.norm.__dict__["_call_impl"]
        assert wrapper.owner is clone and wrapper.module is clone.norm

    def test_deepcopy_of_child_alone_gets_plain_forward(self):
        owner = _Owner()
        _install(owner, _recompute_norm(owner))
        norm = copy.deepcopy(owner.norm)
        assert not isinstance(norm.__dict__["_call_impl"], _ModulePointHook)
        x = torch.randn(2, 16)
        assert torch.equal(norm(x), owner.norm(x))

    def test_uninstall_restores_instance_forward(self):
        owner = _Owner()
        patched = owner.norm._call_impl
        owner.norm._call_impl = patched
        _install(owner, _recompute_norm(owner))
        runtime_module._uninstall(owner)
        assert owner.norm.__dict__["_call_impl"] is patched

    def test_conflicting_strategies_raise(self):
        owner = _Owner()
        with pytest.raises(ValueError):
            _install(owner, _recompute_norm(owner), CheckpointStrategy(owner.norm))

    def test_update_strategy_replaces_fields_and_rehooks(self):
        owner = _Owner()
        _install(owner, _recompute_norm(owner), CheckpointStrategy(owner._bda))
        original = runtime_module.installed_strategy(owner, owner.norm)
        updated = runtime_module.update_strategy(
            owner,
            owner.norm,
            CheckpointWithoutOutputStrategy,
            release_on=Event(owner.linear, "exit"),
        )
        assert updated is original
        assert installed_strategies(owner)[owner.norm] == (updated,)
        assert updated.release_on.region == owner.linear
        assert "_call_impl" in owner.linear.__dict__  # the new release point is observed
        assert has_strategy(owner, owner._bda, CheckpointStrategy)  # other points untouched

    def test_update_strategy_without_a_match(self):
        owner = _Owner()
        _install(owner, _recompute_norm(owner))
        assert runtime_module.update_strategy(owner, owner.linear, release_on=None) is None
        with pytest.raises(AttributeError, match="no field"):
            runtime_module.update_strategy(owner, owner.norm, not_a_field=1)

    def test_install_replaces_previous_strategies(self):
        owner = _Owner()
        _install(owner, _recompute_norm(owner))
        install_activation_strategies(owner, [CheckpointStrategy(owner.linear)])
        assert "_call_impl" not in owner.norm.__dict__
        assert "_call_impl" in owner.linear.__dict__
        assert list(installed_strategies(owner)) == [owner.linear]


@pytest.mark.usefixtures("rng_tracker")
class TestPolicies:
    def _reference(self, x):
        torch.manual_seed(0)
        owner = _Owner().cuda()
        return owner, _run(owner, x)

    @pytest.mark.parametrize("discard_output", [False, True])
    def test_module_hooks_match_on_forward_and_replay(self, discard_output):
        owner = _Owner().cuda()
        owner.linear.register_forward_pre_hook(lambda module, args: (2 * args[0],))
        owner.linear.register_forward_hook(lambda module, args, out: 3 * out)
        x = torch.randn(4, 16, device="cuda")
        expected, dx, grads = _run(owner, x)
        owner.zero_grad()
        strategy = (
            CheckpointWithoutOutputStrategy(owner.linear, release_event=owner._bda)
            if discard_output
            else CheckpointStrategy(owner.linear)
        )
        _install(owner, strategy)
        actual, actual_dx, actual_grads = _run(owner, x)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual_dx, dx)
        for name, grad in grads.items():
            torch.testing.assert_close(actual_grads[name], grad)

    def test_recompute_releases_output_and_matches(self):
        x = torch.randn(8, 4, 16, device="cuda")
        owner, (ref_out, ref_dx, ref_grads) = self._reference(x)

        _install(owner, _recompute_norm(owner))
        for p in owner.parameters():
            p.grad = None

        seen = {}
        original = owner.norm.forward

        def spy(*args, **kwargs):
            out = original(*args, **kwargs)
            seen.setdefault("out", out)
            return out

        owner.norm.forward = spy
        xi = x.detach().clone().requires_grad_(True)
        out = owner(xi)
        # The norm output was released when the branch output reached the residual add.
        assert seen["out"].untyped_storage().size() == 0
        out.sum().backward()

        assert torch.equal(out.detach(), ref_out)
        assert torch.equal(xi.grad, ref_dx)
        for name, grad in ref_grads.items():
            assert torch.equal(dict(owner.named_parameters())[name].grad, grad), name
        assert not runtime_module._installed(owner).pending

    def test_updated_release_event_matches(self):
        x = torch.randn(8, 4, 16, device="cuda")
        owner, (ref_out, ref_dx, ref_grads) = self._reference(x)
        strategy = CheckpointWithoutOutputStrategy(region=owner.norm, release_event=owner.linear)
        strategy.release_on_enter(owner._bda)
        _install(owner, strategy)
        strategy.release_on_exit(owner.linear)
        assert runtime_module.installed_strategy(owner, owner.norm) is strategy

        for p in owner.parameters():
            p.grad = None
        xi = x.detach().clone().requires_grad_(True)
        normed = owner.norm(xi)
        owner._bda(xi, xi)  # Visiting the old boundary must leave the norm output intact.
        assert normed.untyped_storage().size() > 0
        branch = owner.linear(normed)
        assert normed.untyped_storage().size() == 0
        out = owner._bda(branch, xi)
        out.sum().backward()
        assert torch.equal(out.detach(), ref_out)
        assert torch.equal(xi.grad, ref_dx)
        for name, grad in ref_grads.items():
            assert torch.equal(dict(owner.named_parameters())[name].grad, grad), name
        assert not runtime_module._installed(owner).pending

    def test_release_on_exit_of_an_enclosing_method(self):
        class Enclosing(_Owner):
            def _branch(self, x):
                return self.linear(self.norm(x))

            def forward(self, x):
                return self._bda(self._branch(x), x)

        torch.manual_seed(0)
        owner = Enclosing().cuda()
        x = torch.randn(8, 4, 16, device="cuda")
        ref_out, ref_dx, ref_grads = _run(owner, x)
        # Released when the method that runs the norm returns (nothing is pending when it starts).
        _install(
            owner, CheckpointWithoutOutputStrategy(region=owner.norm, release_event=owner._branch)
        )
        for p in owner.parameters():
            p.grad = None
        out, dx, grads = _run(owner, x)
        assert not runtime_module._installed(owner).pending
        assert torch.equal(out, ref_out) and torch.equal(dx, ref_dx)
        for name, grad in ref_grads.items():
            assert torch.equal(grads[name], grad), name

    def test_call_without_activation_strategies_bypasses_the_point(self):
        owner = _Owner().cuda()
        _install(owner, _recompute_norm(owner))
        x = torch.randn(8, 4, 16, device="cuda", requires_grad=True)
        out = call_without_activation_strategies(owner.norm, x)
        # Not checkpointed: nothing is scheduled and the output keeps its storage.
        assert not runtime_module._installed(owner).pending
        owner._bda(out, x).sum().backward()
        assert out.untyped_storage().size() > 0
        owner.norm(x)
        assert runtime_module._installed(owner).pending  # the strategy applies again

    @pytest.mark.parametrize("separate_trigger", [False, True])
    def test_keyword_inputs_and_parallel_gradient_triggers(self, separate_trigger):
        class Branches(MegatronModule):
            def norm(self, *, inputs):
                x, scale = inputs
                return x * scale

            def branches(self, y):
                return {"unused": y.cos(), "used": y.square()}

            def combine(self, *, branches):
                return branches["used"]

            def forward(self, x):
                scale = x.sin()
                self.saved_output = self.norm(inputs=(x, scale))
                branches = self.branches(self.saved_output)
                return self.combine(branches=branches) + scale.square()

        owner = Branches(None)
        x = torch.randn(16, device="cuda", requires_grad=True)
        owner(x).sum().backward()
        expected = x.grad.clone()
        x.grad = None
        install_activation_strategies(
            owner,
            [
                CheckpointWithoutOutputStrategy(
                    region=owner.norm,
                    release_event=owner.combine,
                    release_on_exit=False,
                    recompute_on=Event.exit(owner.branches) if separate_trigger else None,
                )
            ],
        )
        output = owner(x)
        assert owner.saved_output.untyped_storage().size() == 0
        assert not runtime_module._installed(owner).pending
        output.sum().backward()
        torch.testing.assert_close(x.grad, expected, rtol=0, atol=0)

    def test_input_alias_keeps_storage_when_output_is_released(self):
        class AliasOwner(MegatronModule):
            def project(self, x):
                return x.square(), x.view_as(x)

            def consume(self, projected):
                squared, alias = projected
                return squared * alias

        owner = AliasOwner(None)
        install_activation_strategies(
            owner,
            [CheckpointWithoutOutputStrategy(region=owner.project, release_event=owner.consume)],
        )
        x = torch.randn(16, device="cuda", requires_grad=True)
        expected = x.detach().clone()
        squared, alias = owner.project(x)
        result = owner.consume((squared, alias))
        assert squared.untyped_storage().size() == 0
        torch.testing.assert_close(alias, expected)
        result.sum().backward()
        torch.testing.assert_close(x.grad, 3 * expected.square())

    def test_checkpoint_with_tensor_kwargs_matches(self):
        class KwOwner(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(16, 16)

            config = _Config()
            pg_collection = SimpleNamespace(tp=None)

            def _branch(self, x, *, scale, mode):
                assert mode == "mul"
                return self.linear(x) * scale

            def forward(self, x, scale):
                return self._branch(x, scale=scale, mode="mul")

        torch.manual_seed(0)
        owner = KwOwner().cuda()
        x = torch.randn(4, 16, device="cuda", requires_grad=True)
        scale = torch.randn(4, 16, device="cuda", requires_grad=True)
        owner(x, scale).sum().backward()
        ref = (x.grad.clone(), scale.grad.clone(), owner.linear.weight.grad.clone())
        x.grad = scale.grad = owner.linear.weight.grad = owner.linear.bias.grad = None

        install_activation_strategies(owner, [CheckpointStrategy(owner._branch)])
        owner(x, scale).sum().backward()
        assert torch.equal(x.grad, ref[0])
        assert torch.equal(scale.grad, ref[1])
        assert torch.equal(owner.linear.weight.grad, ref[2])


class _Config:
    fp8 = fp4 = None


class _Layer(MegatronModule):
    """A Megatron module that declares its strategies instead of installing them."""

    def __init__(self, recompute: bool):
        super().__init__(_Config())
        self.recompute = recompute
        self.linear = nn.Linear(16, 16)

    def activation_strategies(self):
        if self.recompute:
            self.linear_recompute = self.register_activation_strategy(
                CheckpointStrategy(self.linear)
            )

    def forward(self, x):
        return self.linear(x)


class _Subclass(_Layer):
    def __init__(self, recompute: bool):
        super().__init__(recompute)
        self.extra = nn.Identity()

    def activation_strategies(self):
        super().activation_strategies()
        assert installed_strategies(self) == {}
        self.unregistered = CheckpointStrategy(self.extra)
        self.register_activation_strategy(CheckpointStrategy(self.extra))


class _StandIn(_Layer):
    """Builds only part of the layer, bypassing ``_Layer.__init__``."""

    def __init__(self):
        MegatronModule.__init__(self, _Config())
        self.recompute = True
        self.linear = nn.Linear(16, 16)


class TestMegatronModuleInstall:
    def test_default_module_installs_nothing(self):
        module = MegatronModule(_Config())
        assert installed_strategies(module) == {}

    def test_strategies_install_after_construction(self):
        layer = _Layer(recompute=True)
        assert has_strategy(layer, layer.linear, CheckpointStrategy)
        assert installed_strategies(_Layer(recompute=False)) == {}

    def test_subclass_inherits_strategies(self):
        layer = _Subclass(recompute=True)
        assert runtime_module.installed_strategy(layer, layer.linear) is layer.linear_recompute
        extra = runtime_module.installed_strategy(layer, layer.extra)
        assert isinstance(extra, CheckpointStrategy) and extra is not layer.unregistered

    def test_stand_in_that_bypasses_the_constructor_gets_none(self):
        assert installed_strategies(_StandIn()) == {}
