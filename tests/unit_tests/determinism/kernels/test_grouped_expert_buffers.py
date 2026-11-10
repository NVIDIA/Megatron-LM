# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# License for these changes: BSD-3-Clause.

"""Compare native TE forward, dgrad and accumulated FP32 wgrad with shared buffers."""

import gc

import pytest
import torch

from megatron.core.ops.moe.grouped_expert_buffers import substitute_forward
from tests.unit_tests.determinism.kernels.harness import assert_replays_bit_exact


def _relative_error(actual, expected):
    return (actual.double() - expected.double()).norm() / expected.double().norm().clamp_min(1e-30)


def test_shared_buffer_backward_replays_bit_exact():
    """Replaying the adapter must not retain stale gradient storage."""
    from transformer_engine.pytorch.module.grouped_linear import GroupedLinear

    module = GroupedLinear(
        num_gemms=128,
        in_features=32,
        out_features=64,
        bias=False,
        params_dtype=torch.bfloat16,
        use_grouped_tensor=False,
    )
    splits = torch.tensor([0 if i % 7 == 0 else 16 for i in range(128)])
    source = torch.randn(sum(splits.tolist()), 32, device="cuda", dtype=torch.bfloat16)
    gradient = torch.randn(source.shape[0], 64, device="cuda", dtype=torch.bfloat16)

    def run(value):
        module.zero_grad(set_to_none=True)
        value = value.detach().requires_grad_()
        output = module(value, splits)
        output.backward(gradient)
        return (
            output.detach().clone(),
            value.grad.clone(),
            *(p.grad.clone() for p in module.parameters()),
        )

    with substitute_forward():
        assert_replays_bit_exact(
            run, (source,), replays=3, backward=False, what="shared grouped expert buffers"
        )


@pytest.mark.parametrize('groups,width,out_width', [(8, 32, 64), (128, 32, 64)])
def test_grouped_buffers_preserve_native_gradients(groups, width, out_width):
    from transformer_engine.pytorch.module.grouped_linear import GroupedLinear

    torch.manual_seed(2901)
    options = dict(
        num_gemms=groups,
        in_features=width,
        out_features=out_width,
        bias=False,
        params_dtype=torch.bfloat16,
        fuse_wgrad_accumulation=True,
        use_grouped_tensor=False,
    )
    reference, candidate = GroupedLinear(**options), GroupedLinear(**options)
    hook_counts = [0, 0]

    def hook(index):
        def clear_dummy(parameter):
            assert parameter.grad_added_to_main_grad
            parameter.grad = None
            hook_counts[index] += 1

        return clear_dummy

    with torch.no_grad():
        for a, b in zip(reference.parameters(), candidate.parameters(), strict=True):
            b.copy_(a)
            a.main_grad = torch.randn_like(a, dtype=torch.float32)
            b.main_grad = a.main_grad.clone()
        for index, module in enumerate((reference, candidate)):
            for parameter in module.parameters():
                parameter.grad_added_to_main_grad = False
                parameter.zero_out_wgrad = True
                parameter.register_post_accumulate_grad_hook(hook(index))
    counts = [0 if i % 7 == 0 else 16 for i in range(groups)]
    splits = torch.tensor(counts, dtype=torch.int64)
    source = torch.randn((sum(counts), width), device='cuda', dtype=torch.bfloat16)
    dy = torch.randn((sum(counts), out_width), device='cuda', dtype=torch.bfloat16)
    original_source, original_dy = source.clone(), dy.clone()
    rng = torch.cuda.get_rng_state().clone()
    retained = []
    memory = []
    for update in range(3):
        x = source.detach().requires_grad_(True)
        expected = reference(x, splits, is_first_microbatch=update == 0)
        expected.backward(dy)
        dx = x.grad.clone()
        x = source.detach().requires_grad_(True)
        with substitute_forward() as calls:
            actual = candidate(x, splits, is_first_microbatch=update == 0)
            actual.backward(dy)
        assert calls['fallback'] == 0
        assert calls['forward'] == (groups + 63) // 64
        assert calls['native_backward'] == (groups + 63) // 64
        assert _relative_error(actual, expected) <= 0.005
        assert _relative_error(x.grad, dx) <= 0.005
        for a, b in zip(reference.parameters(), candidate.parameters(), strict=True):
            assert torch.equal(a, b)
            assert _relative_error(b.main_grad, a.main_grad) <= 0.005
        # A retained output also retains its leaf through AccumulateGrad. Clear the
        # leaf-owned gradient before measuring temporary adapter storage.
        x.grad = None
        retained.append(actual)  # Live graph nodes must not pin temporary dgrad storage.
        del x, expected, dx
        gc.collect()
        torch.cuda.synchronize()
        memory.append(
            torch.cuda.memory_allocated() - sum(t.numel() * t.element_size() for t in retained)
        )
    assert memory[-1] - memory[0] < max(source.numel() * source.element_size(), 1024**2)
    assert hook_counts[0] == hook_counts[1] == groups * 3
    assert torch.equal(source, original_source) and torch.equal(dy, original_dy)
    assert torch.equal(rng, torch.cuda.get_rng_state())
