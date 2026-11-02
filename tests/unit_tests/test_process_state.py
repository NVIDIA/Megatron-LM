# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Process-state restoration must obey PyTorch's frozen backend flag policy."""

import inspect

import pytest
import torch

from tests.unit_tests.test_utilities import _restore_torch_settings, _snapshot_torch_settings


@pytest.mark.parametrize("frozen", [False, True], ids=["unlocked", "locked"])
@pytest.mark.parametrize("changed", [False, True], ids=["unchanged", "changed"])
def test_restore_torch_settings_preserves_cudnn_policy_and_uncaptured_flags(frozen, changed):
    original = _snapshot_torch_settings()
    original_policy = torch.backends.flags_frozen()
    supports_precision = (
        "_fp32_precision" in inspect.signature(torch.backends.cudnn.set_flags).parameters
    )
    precision_context = {"fp32_precision": "ieee"} if supports_precision else {}
    try:
        # PyTorch's own context restores the prior mutation policy, including
        # when a failed assertion interrupts either the locked or unlocked case.
        with torch.backends.__allow_nonbracketed_mutation():
            with torch.backends.cudnn.flags(
                enabled=False,
                deterministic=True,
                benchmark=False,
                benchmark_limit=7,
                allow_tf32=False,
                **precision_context,
            ):
                if frozen:
                    torch.backends.disable_global_flags()
                assert torch.backends.flags_frozen() is frozen
                captured = _snapshot_torch_settings()
                uncaptured = (
                    torch.backends.cudnn.enabled,
                    torch.backends.cudnn.benchmark_limit,
                    torch.backends.cudnn.fp32_precision if supports_precision else None,
                )
                # Exercise restoration both with changed values and when the
                # snapshot already matches, as in an ordinary fixture teardown.
                mode, warn_only = captured["deterministic"]
                torch.use_deterministic_algorithms(
                    not mode if changed else mode, warn_only=not warn_only if changed else warn_only
                )
                fill_memory = captured["fill_uninitialized_memory"]
                torch.utils.deterministic.fill_uninitialized_memory = (
                    not fill_memory if changed else fill_memory
                )
                deterministic, benchmark, allow_tf32 = captured["cudnn"]
                precision_setter = {"_fp32_precision": None} if supports_precision else {}
                torch.backends.cudnn.set_flags(
                    _deterministic=not deterministic if changed else deterministic,
                    _benchmark=not benchmark if changed else benchmark,
                    _allow_tf32=not allow_tf32 if changed else allow_tf32,
                    **precision_setter,
                )
                matmul = torch.backends.cuda.matmul
                allow_tf32, allow_bf16, allow_fp16 = captured["matmul"]
                matmul.allow_tf32 = not allow_tf32 if changed else allow_tf32
                matmul.allow_bf16_reduced_precision_reduction = (
                    not allow_bf16 if changed else allow_bf16
                )
                matmul.allow_fp16_reduced_precision_reduction = (
                    not allow_fp16 if changed else allow_fp16
                )
                assert (_snapshot_torch_settings() != captured) is changed
                if frozen:
                    # The old restoration's direct assignment fails under this
                    # real policy; the fix must restore values without unlocking it.
                    with pytest.raises(RuntimeError, match="not allowed to set"):
                        torch.backends.cudnn.deterministic = deterministic

                _restore_torch_settings(captured)
                assert _snapshot_torch_settings() == captured
                assert (
                    torch.backends.cudnn.enabled,
                    torch.backends.cudnn.benchmark_limit,
                    torch.backends.cudnn.fp32_precision if supports_precision else None,
                ) == uncaptured
                assert torch.backends.flags_frozen() is frozen
    finally:
        # Also clean up correctly when running this regression on the unfixed
        # baseline: restoration there needs PyTorch's sanctioned mutable scope.
        with torch.backends.__allow_nonbracketed_mutation():
            _restore_torch_settings(original)
    assert _snapshot_torch_settings() == original
    assert torch.backends.flags_frozen() is original_policy
