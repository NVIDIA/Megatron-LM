# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import warnings

import pytest
import torch

from megatron.core.extensions import transformer_engine as te_ext
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_te_min_version

pytestmark = [
    pytest.mark.skipif(not te_ext.HAVE_TE, reason="Transformer Engine is not available"),
    pytest.mark.skipif(
        not is_te_min_version("1.13.0"),
        reason="TE fused ops wrappers require Transformer Engine >= 1.13.0",
    ),
]


def _make_rmsnorm() -> torch.nn.Module:
    return te_ext.TEFusedResidualRMSNorm(normalized_shape=16, dtype=torch.float32, device="cpu")


def _make_fused_mlp_shell() -> torch.nn.Module:
    module = te_ext.TEFusedMLP.__new__(te_ext.TEFusedMLP)
    torch.nn.Module.__init__(module)
    module.linear_fc1 = torch.nn.Linear(2, 2)
    module.linear_fc2 = torch.nn.Linear(2, 2)
    return module


def test_rmsnorm_fused_impl_aliases_source_weight():
    module = _make_rmsnorm()

    fused_impl = module._get_fused_impl()

    assert fused_impl[1].weight is module.weight


def test_fused_impl_is_cached_and_resettable():
    module = _make_rmsnorm()

    first_impl = module._get_fused_impl()

    assert module._get_fused_impl() is first_impl

    module._reset_fused_impl()

    assert module._fused_impl is None

    second_impl = module._get_fused_impl()

    assert second_impl is not first_impl
    assert second_impl[1].weight is module.weight


def test_fused_impl_is_not_registered_as_module_or_state_dict_source():
    module = _make_rmsnorm()
    expected_state_keys = set(module.state_dict().keys())
    expected_module_keys = tuple(module._modules.keys())

    fused_impl = module._get_fused_impl()

    assert set(module.state_dict().keys()) == expected_state_keys
    assert "weight" in expected_state_keys
    assert tuple(module._modules.keys()) == expected_module_keys
    assert "_fused_impl" not in module._modules
    assert all(child is not fused_impl for child in module.modules())


def test_mcore_te_linear_adapter_rejects_plain_te_linear():
    plain_linear = te_ext.te.pytorch.Linear(16, 16, device="meta")

    with pytest.raises(ValueError) as exc_info:
        te_ext._make_te_ops_basic_linear_from_mcore_te_linear(
            plain_linear, module_name="plain_linear"
        )

    message = str(exc_info.value)
    assert "plain_linear" in message
    assert plain_linear.__class__.__name__ in message
    assert "config.tp_comm_overlap" in message


def test_mcore_te_linear_adapter_aliases_source_weight():
    config = TransformerConfig(
        num_layers=1,
        hidden_size=16,
        num_attention_heads=1,
        use_cpu_initialization=True,
        params_dtype=torch.float32,
    )
    linear = te_ext.TEColumnParallelLinear(
        16,
        32,
        config=config,
        init_method=torch.nn.init.zeros_,
        gather_output=False,
        bias=True,
        skip_bias_add=True,
        is_expert=False,
        tp_comm_buffer_name="fc1",
    )

    op = te_ext._make_te_ops_basic_linear_from_mcore_te_linear(
        linear, module_name="linear", output_features=linear.weight.size(0)
    )

    assert isinstance(op, te_ext.te.pytorch.ops.BasicLinear)
    assert op.weight is linear.weight


def test_fused_mlp_forwards_current_submodule_pre_hooks():
    module = _make_fused_mlp_shell()
    fused_impl = torch.nn.Identity()

    # The real training lifecycle lazily constructs the fused implementation while
    # DDP parameter-gather hooks are disabled for the first iteration.
    module._register_hooks_on_fused_impl(fused_impl)

    events = []

    def old_hook(submodule, _inputs):
        events.append(("old", submodule, tuple(submodule.parameters(recurse=False))))

    old_handles = [
        module.linear_fc1.register_forward_pre_hook(old_hook),
        module.linear_fc2.register_forward_pre_hook(old_hook),
    ]
    module.register_forward_pre_hook(old_hook)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fused_impl(torch.ones(1, 2))

    assert [event[0] for event in events] == ["old", "old"]
    assert [event[1] for event in events] == [module.linear_fc1, module.linear_fc2]
    assert events[0][2][0] is module.linear_fc1.weight
    assert events[1][2][0] is module.linear_fc2.weight

    for handle in old_handles:
        handle.remove()
    events.clear()

    def replacement_hook(submodule, _inputs, kwargs):
        events.append(("replacement", submodule, kwargs))

    module.linear_fc2.register_forward_pre_hook(replacement_hook, with_kwargs=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fused_impl(torch.ones(1, 2))

    assert events == [("replacement", module.linear_fc2, {})]


def test_fused_mlp_wrapper_hooks_execute_once():
    module = _make_fused_mlp_shell()
    fused_impl = torch.nn.Identity()
    module.forward = lambda inputs: fused_impl(inputs)

    events = []
    module.register_forward_pre_hook(lambda _module, _inputs: events.append("forward-pre"))
    module.register_forward_hook(lambda _module, _inputs, _output: events.append("forward-post"))
    module.register_full_backward_pre_hook(
        lambda _module, _grad_output: events.append("backward-pre")
    )
    module.register_full_backward_hook(
        lambda _module, _grad_input, _grad_output: events.append("backward-post")
    )
    module._register_hooks_on_fused_impl(fused_impl)

    output = module(torch.ones(1, 2, requires_grad=True))
    output.sum().backward()

    assert events == ["forward-pre", "forward-post", "backward-pre", "backward-post"]


def test_fused_mlp_partitions_hooks_across_execution_boundaries():
    module = _make_fused_mlp_shell()
    module.linear_fc1.inner = torch.nn.Identity()
    events = []

    class RecordingIdentity(torch.nn.Module):

        def __init__(self, name):
            super().__init__()
            self.name = name

        def forward(self, inputs):
            events.append(self.name)
            return inputs

    norm_impl = RecordingIdentity("norm")
    fused_impl = RecordingIdentity("fused")
    fc1_submodules = tuple(module.linear_fc1.modules())
    fc1_submodule_ids = {id(submodule) for submodule in fc1_submodules}
    remaining_submodules = tuple(
        submodule
        for submodule in module.modules()
        if submodule is not module and id(submodule) not in fc1_submodule_ids
    )

    def post_hook(name):
        return lambda _module, _inputs, _output: events.append(name)

    module.linear_fc1.register_forward_hook(post_hook("fc1-post"))
    module.linear_fc1.inner.register_forward_hook(post_hook("fc1-inner-post"))
    module.linear_fc2.register_forward_hook(post_hook("fc2-post"))

    with pytest.warns(UserWarning, match="post-forward hook"):
        module._register_hooks_on_fused_impl(
            fused_impl, pre_forward_submodules=remaining_submodules
        )
    module._register_forward_pre_hooks_on_fused_impl(norm_impl, fc1_submodules)

    def pre_hook(name):
        return lambda _module, _inputs: events.append(name)

    pre_hook_handles = [
        module.linear_fc1.register_forward_pre_hook(pre_hook("fc1-pre")),
        module.linear_fc1.inner.register_forward_pre_hook(pre_hook("fc1-inner-pre")),
        module.linear_fc2.register_forward_pre_hook(pre_hook("fc2-pre")),
    ]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        norm_impl(torch.ones(1, 2))
        fused_impl(torch.ones(1, 2))

    assert events == [
        "fc1-pre",
        "fc1-inner-pre",
        "norm",
        "fc2-pre",
        "fused",
        "fc1-post",
        "fc1-inner-post",
        "fc2-post",
    ]

    for handle in pre_hook_handles:
        handle.remove()
    events.clear()

    def replacement_hook(name):
        return lambda _module, _inputs, kwargs: events.append((name, kwargs))

    module.linear_fc1.register_forward_pre_hook(
        replacement_hook("fc1-replacement"), with_kwargs=True
    )
    module.linear_fc2.register_forward_pre_hook(
        replacement_hook("fc2-replacement"), with_kwargs=True
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        norm_impl(torch.ones(1, 2))
        fused_impl(torch.ones(1, 2))

    assert events == [
        ("fc1-replacement", {}),
        "norm",
        ("fc2-replacement", {}),
        "fused",
        "fc1-post",
        "fc1-inner-post",
        "fc2-post",
    ]


@pytest.mark.parametrize("invalid_source", ["duplicate", "non-descendant"])
def test_fused_mlp_pre_hook_sources_are_validated(invalid_source):
    module = _make_fused_mlp_shell()
    if invalid_source == "duplicate":
        source_submodules = (module.linear_fc1, module.linear_fc1)
        error_match = "must not contain duplicates"
    else:
        source_submodules = (torch.nn.Linear(2, 2),)
        error_match = "must be descendants"

    with pytest.raises(ValueError, match=error_match):
        module._register_forward_pre_hooks_on_fused_impl(torch.nn.Identity(), source_submodules)


@pytest.mark.parametrize("hook_kind", ["pre", "post"])
def test_fused_mlp_partition_still_rejects_fc1_backward_hooks(hook_kind):
    module = _make_fused_mlp_shell()
    if hook_kind == "pre":
        module.linear_fc1.register_full_backward_pre_hook(lambda _module, _grad_output: None)
        error_match = "pre-backward hooks"
    else:
        module.linear_fc1.register_full_backward_hook(
            lambda _module, _grad_input, _grad_output: None
        )
        error_match = "post-backward hooks"

    with pytest.raises(RuntimeError, match=error_match):
        module._register_hooks_on_fused_impl(
            torch.nn.Identity(), pre_forward_submodules=(module.linear_fc2,)
        )


def test_fused_mlp_rejects_input_modifying_submodule_hook_added_after_construction():
    module = _make_fused_mlp_shell()
    fused_impl = torch.nn.Identity()
    module._register_hooks_on_fused_impl(fused_impl)

    module.linear_fc1.register_forward_pre_hook(lambda _module, inputs: inputs)

    with pytest.warns(UserWarning, match="pre-forward hook"):
        with pytest.raises(RuntimeError, match="modifies input tensor"):
            fused_impl(torch.ones(1, 2))
