# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Test Triton's real Autotuner, with focused CUDA and distributed coverage."""

import json
import logging
import os
from dataclasses import asdict
from unittest.mock import Mock

import pytest

from megatron.core.tuning import interception, selection
from megatron.core.tuning.policy import AutotunePolicy

triton = pytest.importorskip("triton")
import triton.language as tl
from triton.runtime.autotuner import Autotuner


def make_tuner(*, prune=None, fail=False, module="mamba_ssm.ops.triton.test"):
    """Use a fake launcher underneath the unmodified Triton autotuner."""
    hooks = []

    class Kernel:
        def __init__(self):
            self.fn = lambda *args, **kwargs: None
            self.fn.__module__ = module
            self.fn.__name__ = "test_kernel"

        def run(self, *args, **kwargs):
            if fail:
                raise RuntimeError("launch failed")
            return kwargs["BLOCK_SIZE"]

    configs = [
        triton.Config({"BLOCK_SIZE": size}, pre_hook=lambda args: hooks.append(args["BLOCK_SIZE"]))
        for size in (32, 128)
    ]
    tuner = Autotuner(
        Kernel(),
        ["x", "size"],
        configs,
        ["size"],
        reset_to_zero=None,
        restore_value=None,
        prune_configs_by={"early_config_prune": prune} if prune else None,
    )
    return tuner, hooks


def forbid_benchmark(*args, **kwargs):
    """A pinned launch must never time even a single candidate."""
    pytest.fail("pinned execution benchmarked a config")


def test_install_is_cuda_lazy(isolated_policy, monkeypatch):
    def unexpected_device_query():
        pytest.fail("install queried CUDA before rank-local device selection")

    monkeypatch.setattr(selection, "arch_tag", unexpected_device_query)
    assert interception.install(AutotunePolicy(mode="pinned"))


def test_install_pins_in_tree_tuner_created_before_install(isolated_policy, monkeypatch):
    import torch

    from megatron.core.tuning import policy as tuning_policy

    monkeypatch.setattr(tuning_policy, "_deterministic_override", None)
    monkeypatch.setattr(torch, "are_deterministic_algorithms_enabled", lambda: False)
    tuner, hooks = make_tuner(module="megatron.core.ssm.ops.gdp.chunk_h")
    original_configs = tuner.configs
    assert selection.autotune_configs(original_configs) is original_configs
    assert len(original_configs) == 2
    monkeypatch.setattr(tuner, "_bench", forbid_benchmark)
    policy = AutotunePolicy()
    supplied_settings = asdict(policy)

    interception.install(policy, deterministic=True)

    assert tuner.run(None, 128) == 32
    assert tuner.run(None, 128) == 32
    assert hooks == [32, 32]
    assert tuner.configs is original_configs
    assert tuner.nargs is None
    # Installation and cached launches keep results outside the caller's settings.
    assert asdict(policy) == supplied_settings
    assert policy.mode is None
    assert interception.active_policy().mode == "pinned"
    assert interception.choice_log()


def test_pinning_honours_pruning_per_shape_and_preserves_hooks(isolated_policy, monkeypatch):
    def prune(configs, named_args, **kwargs):
        size = kwargs.get("size", named_args.get("size"))
        return [config for config in configs if config.kwargs["BLOCK_SIZE"] == size]

    tuner, hooks = make_tuner(prune=prune)
    original_configs = tuner.configs
    monkeypatch.setattr(tuner, "_bench", forbid_benchmark)
    interception.install(AutotunePolicy(mode="pinned"))
    assert tuner.run(None, 128) == 128
    assert tuner.run(None, size=32) == 32
    assert hooks == [128, 32]
    assert tuner.configs is original_configs
    assert tuner.nargs is None


@pytest.mark.parametrize("chaos", [False, True])
def test_repeated_pinning_caches_pruning_and_selection(isolated_policy, monkeypatch, chaos):
    def prune(configs, named_args, **kwargs):
        size = kwargs.get("size", named_args.get("size"))
        return [config for config in configs if config.kwargs["BLOCK_SIZE"] == size]

    tuner, hooks = make_tuner(prune=prune)
    original_configs = tuner.configs
    pruning = Mock(wraps=tuner.prune_configs)
    choice_name = "chaos_choice" if chaos else "deterministic_choice"
    choosing = Mock(wraps=getattr(selection, choice_name))
    monkeypatch.setattr(tuner, "prune_configs", pruning)
    monkeypatch.setattr(selection, choice_name, choosing)
    monkeypatch.setattr(tuner, "_bench", forbid_benchmark)
    interception.install(AutotunePolicy(mode="pinned", chaos=chaos))

    assert tuner.run(None, 128) == 128
    assert tuner.run(None, size=128) == 128
    assert tuner.run(None, 32) == 32
    assert tuner.run(None, 128) == 128
    assert pruning.call_count == choosing.call_count == 2
    assert hooks == [128, 128, 32, 128]
    assert tuner.configs is original_configs
    assert tuner.nargs is None


def test_pinned_cache_distinguishes_dtype_and_architecture(isolated_policy, monkeypatch):
    import torch

    tuner, hooks = make_tuner()
    pruning = Mock(wraps=tuner.prune_configs)
    choosing = Mock(wraps=selection.deterministic_choice)
    monkeypatch.setattr(tuner, "prune_configs", pruning)
    monkeypatch.setattr(selection, "deterministic_choice", choosing)
    monkeypatch.setattr(tuner, "_bench", forbid_benchmark)
    interception.install(AutotunePolicy(mode="pinned"))
    fp32 = torch.empty(0, dtype=torch.float32)
    fp16 = torch.empty(0, dtype=torch.float16)

    assert tuner.run(fp32, 128) == 32
    assert tuner.run(fp32, 128) == 32
    assert pruning.call_count == choosing.call_count == 1
    assert tuner.run(fp16, 128) == 32
    assert pruning.call_count == choosing.call_count == 2
    monkeypatch.setattr(selection, "arch_tag", lambda: "other_arch")
    assert tuner.run(fp32, 128) == 32
    assert tuner.run(fp32, 128) == 32
    assert pruning.call_count == choosing.call_count == 3
    monkeypatch.setattr(selection, "arch_tag", lambda: "test_arch")
    assert tuner.run(fp32, 128) == 32
    assert pruning.call_count == choosing.call_count == 3
    assert hooks == [32] * 6


def test_pinned_cache_keeps_live_configs_per_tuner(isolated_policy, monkeypatch):
    first, first_hooks = make_tuner(prune=lambda configs, *args, **kwargs: configs[1:])
    second, second_hooks = make_tuner(prune=lambda configs, *args, **kwargs: configs[:1])
    monkeypatch.setattr(first, "_bench", forbid_benchmark)
    monkeypatch.setattr(second, "_bench", forbid_benchmark)
    interception.install(AutotunePolicy(mode="pinned"))
    assert selection.kernel_name(first) == selection.kernel_name(second)

    for _ in range(2):
        assert first.run(None, 128) == 128
        assert second.run(None, 128) == 32
        assert first.best_config is first.configs[1]
        assert second.best_config is second.configs[0]
    assert first_hooks == [128, 128]
    assert second_hooks == [32, 32]


def test_pinned_cache_invalidates_only_when_policy_changes(isolated_policy, monkeypatch):
    tuner, hooks = make_tuner()
    pruning = Mock(wraps=tuner.prune_configs)
    choosing = Mock(side_effect=tuner.configs)
    monkeypatch.setattr(tuner, "prune_configs", pruning)
    monkeypatch.setattr(selection, "deterministic_choice", choosing)
    monkeypatch.setattr(tuner, "_bench", forbid_benchmark)
    interception.install(AutotunePolicy(mode="pinned"))
    assert tuner.run(None, 128) == 32

    interception.install(AutotunePolicy(mode="pinned"))
    assert tuner.run(None, 128) == 32
    assert pruning.call_count == choosing.call_count == 1
    interception.install(AutotunePolicy(mode="pinned", on_miss="error"))
    assert tuner.run(None, 128) == 128
    assert tuner.run(None, 128) == 128
    assert pruning.call_count == choosing.call_count == 2
    assert hooks == [32, 32, 128, 128]


def test_pinned_cache_invalidates_when_block_sizes_change(isolated_policy, monkeypatch):
    tuner, hooks = make_tuner()
    choosing = Mock(wraps=selection.deterministic_choice)
    monkeypatch.setattr(selection, "deterministic_choice", choosing)
    monkeypatch.setattr(tuner, "_bench", forbid_benchmark)
    interception.install(AutotunePolicy(mode="pinned", block_sizes=(("BLOCK_SIZE", 32),)))
    assert tuner.run(None, 128) == 32
    assert tuner.run(None, 128) == 32
    assert choosing.call_count == 1

    interception.install(AutotunePolicy(mode="pinned", block_sizes=(("BLOCK_SIZE", 128),)))
    assert tuner.run(None, 128) == 128
    assert tuner.run(None, 128) == 128
    assert choosing.call_count == 2
    assert hooks == [32, 32, 128, 128]


def test_pinning_restores_state_after_launch_failure(isolated_policy, monkeypatch):
    tuner, _ = make_tuner(fail=True)
    original_configs = tuner.configs
    choosing = Mock(wraps=selection.deterministic_choice)
    monkeypatch.setattr(selection, "deterministic_choice", choosing)
    interception.install(AutotunePolicy(mode="pinned"))
    for _ in range(2):
        with pytest.raises(RuntimeError, match="launch failed"):
            tuner.run(None, 128)
        assert tuner.configs is original_configs
        assert tuner.nargs is None
        assert interception.choice_log() == {}
    assert choosing.call_count == 2


def test_cached_pinning_restores_state_after_launch_failure(isolated_policy, monkeypatch):
    tuner, hooks = make_tuner()
    original_configs = tuner.configs
    choosing = Mock(wraps=selection.deterministic_choice)
    monkeypatch.setattr(selection, "deterministic_choice", choosing)
    interception.install(AutotunePolicy(mode="pinned"))
    assert tuner.run(None, 128) == 32
    choices = interception.choice_log()

    def fail(*args, **kwargs):
        raise RuntimeError("launch failed")

    monkeypatch.setattr(tuner.fn, "run", fail)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="launch failed"):
            tuner.run(None, 128)
        assert tuner.configs is original_configs
        assert tuner.nargs is None
        assert interception.choice_log() == choices
    assert choosing.call_count == 1
    assert hooks == [32, 32, 32]


def test_install_can_upgrade_an_observer_to_pinning(isolated_policy, monkeypatch):
    tuner, _ = make_tuner()
    interception.install(AutotunePolicy(mode="auto", verify_every=1))
    interception.install(AutotunePolicy(mode="pinned"))
    monkeypatch.setattr(tuner, "_bench", forbid_benchmark)
    assert tuner.run(None, 128) == 32
    assert interception.active_policy().mode == "pinned"


def test_install_replaces_the_previous_policy(isolated_policy, nondeterministic_torch):
    interception.install(AutotunePolicy(mode="pinned", modules=("my_kernels",), verify_every=3))
    interception.install(AutotunePolicy(mode="auto"))
    assert interception.active_policy() == AutotunePolicy(mode="auto")
    interception.install(AutotunePolicy(mode="pinned"))
    interception.install()
    assert interception.active_policy() == AutotunePolicy(mode="auto")


def test_install_derives_an_omitted_mode(isolated_policy, nondeterministic_torch):
    policy = AutotunePolicy(modules=("my_kernels",), verify_every=3)
    interception.install(policy, deterministic=True)
    assert interception.active_policy() == policy.resolve(deterministic=True)
    assert interception.active_policy().mode == "pinned"
    interception.install(policy)
    assert interception.active_policy().mode == "auto"
    # An explicit mode wins over the deterministic request.
    interception.install(AutotunePolicy(mode="auto"), deterministic=True)
    assert interception.active_policy().mode == "auto"


def test_in_tree_selection_uses_explicit_block_sizes(isolated_policy, monkeypatch):
    from megatron.core.tuning.policy import set_deterministic_mode

    tuner, _ = make_tuner()
    interception.install(AutotunePolicy(mode="pinned", block_sizes=(("BLOCK_SIZE", 128),)))
    monkeypatch.setenv("TRITON_AUTOTUNE_BLOCK_SIZE", "32")
    set_deterministic_mode(True)
    try:
        assert selection.autotune_configs(tuner.configs) == [tuner.configs[1]]
    finally:
        set_deterministic_mode(None)


def test_module_scope_respects_package_boundaries(isolated_policy, monkeypatch):
    tuner, _ = make_tuner(module="mamba_ssm_extra")
    monkeypatch.setattr(
        tuner, "_bench", lambda *args, config, **kwargs: -config.kwargs["BLOCK_SIZE"]
    )
    interception.install(AutotunePolicy(mode="pinned"))
    assert tuner.run(None, 128) == 128


def test_record_mode_keeps_autotuning_and_captures_the_winner(
    isolated_policy, monkeypatch, tmp_path
):
    tuner, _ = make_tuner()
    monkeypatch.setattr(
        tuner, "_bench", lambda *args, config, **kwargs: -config.kwargs["BLOCK_SIZE"]
    )
    # The rank is read while recording, when the process group still exists.
    monkeypatch.setattr(interception, "safe_get_rank", lambda: 3)
    interception.install(AutotunePolicy(mode="record", record_path=str(tmp_path / "rec")))
    assert tuner.run(None, 128) == 128
    assert tuner.run(None, 128) == 128
    kernel = "mamba_ssm.ops.triton.test.test_kernel"
    record = interception._tune_records["test_arch"][kernel]["size=128"]
    assert record["kwargs"]["BLOCK_SIZE"] == 128
    monkeypatch.setattr(interception, "safe_get_rank", lambda: 0)
    interception._dump_records()
    recorded = json.loads((tmp_path / "rec.rank3.json").read_text())
    assert recorded["test_arch"][kernel]["size=128"] == record
    assert not list(tmp_path.glob("*.tmp.*"))


def test_table_distinguishes_all_launch_options(isolated_policy):
    from megatron.core.tuning.table import TunedTable

    ordinary = triton.Config({"BLOCK_SIZE": 32})
    clustered = triton.Config({"BLOCK_SIZE": 32}, num_ctas=2, maxnreg=64)
    assert selection.config_signature(ordinary) != selection.config_signature(clustered)
    table = TunedTable("test_arch", {"test_kernel": {"*": selection.config_data(clustered)}})
    assert table.lookup("test_kernel", "size=128", [ordinary, clustered]) is clustered


@pytest.mark.parametrize(
    "maps, agrees",
    [
        ([{"shared": "a", "stage0": "b"}, {"shared": "a", "stage1": "c"}], True),
        ([{"shared": "a"}, {}], True),
        ([{"shared": "a"}, {"shared": "b"}], False),
    ],
)
def test_verification_compares_only_observed_choices(isolated_policy, monkeypatch, maps, agrees):
    import torch

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: len(maps))

    def gather(output, value, group=None):
        output[:] = maps

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    monkeypatch.setattr(interception, "_policy", AutotunePolicy(verify_strict=True))
    if agrees:
        assert interception.verify_choices()
    else:
        with pytest.raises(RuntimeError, match="Ranks disagree on 1 autotune choice"):
            interception.verify_choices()


def test_verification_exchanges_only_new_choices(isolated_policy, monkeypatch, caplog):
    import torch

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: 0)
    sent = []
    other_rank: dict = {}

    def gather(output, value, group=None):
        sent.append(dict(value))
        output[:] = [value, dict(other_rank)]

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    monkeypatch.setattr(interception, "_policy", AutotunePolicy(verify_every=1))
    tuner, _ = make_tuner()
    interception.install(AutotunePolicy(mode="pinned", verify_every=1))
    assert tuner.run(None, 128) == 32
    assert interception.verify_choices()
    assert len(sent[-1]) == 1
    # Steady state: nothing new was chosen, so nothing is exchanged.
    assert tuner.run(None, 128) == 32
    assert interception.verify_choices()
    assert sent[-1] == {}
    # A later report that contradicts an agreed choice is still caught.
    (entry,) = interception.choice_log()
    other_rank[entry] = "different;pinned"
    with caplog.at_level(logging.WARNING, logger=interception.__name__):
        assert interception.verify_choices() is False
    assert "Ranks disagree on 1 autotune choice" in caplog.text


def test_pinned_cuda_kernel_skips_benchmarks(isolated_policy, monkeypatch):
    import torch

    from tests.unit_tests.test_utilities import Utils

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    Utils.initialize_distributed()

    @triton.autotune(
        configs=[triton.Config({"BLOCK_SIZE": size}) for size in (32, 128)], key=["size"]
    )
    @triton.jit
    def double(x, out, size, BLOCK_SIZE: tl.constexpr):
        index = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        value = tl.load(x + index, mask=index < size, other=0)
        tl.store(out + index, value * 2, mask=index < size)

    monkeypatch.setattr(double, "_bench", forbid_benchmark)
    interception.install(AutotunePolicy(mode="pinned", modules=(__name__,)))
    for size in (17, 257):
        values = torch.arange(size, device="cuda", dtype=torch.float32)
        output = torch.empty_like(values)
        double[lambda meta: (triton.cdiv(size, meta["BLOCK_SIZE"]),)](values, output, size)
        torch.testing.assert_close(output, values * 2, rtol=0, atol=0)
    assert len(double.configs) == 2
    assert double.best_config.kwargs["BLOCK_SIZE"] == 32


def test_verification_with_real_process_group(isolated_policy, monkeypatch):
    import torch

    from tests.unit_tests.test_utilities import Utils

    if not torch.cuda.is_available() or Utils.world_size < 2:
        pytest.skip("requires at least two CUDA ranks")
    Utils.initialize_distributed()
    rank = torch.distributed.get_rank()
    monkeypatch.setattr(interception, "_policy", AutotunePolicy(verify_strict=True))
    interception._unverified.update({"shared": "same", f"stage{rank}": "local"})
    assert interception.verify_choices()
    interception._unverified["shared"] = str(rank % 2)
    with pytest.raises(RuntimeError, match="Ranks disagree on 1 autotune choice"):
        interception.verify_choices()


@pytest.fixture
def nondeterministic_torch(monkeypatch):
    import torch

    from megatron.core.tuning import policy as tuning_policy

    monkeypatch.setattr(tuning_policy, "_deterministic_override", None)
    monkeypatch.setattr(torch, "are_deterministic_algorithms_enabled", lambda: False)


def test_install_converts_mappings_and_rejects_other_types(isolated_policy, nondeterministic_torch):
    interception.install({"mode": "pinned", "table_path": None})
    assert interception.active_policy() == AutotunePolicy(mode="pinned")
    with pytest.raises(TypeError, match="AutotunePolicy or a mapping"):
        interception.install(7)
    # The rejected value never reached the adapter.
    assert interception.active_policy() == AutotunePolicy(mode="pinned")


def test_pinned_launches_log_a_choice_only_when_it_is_made(isolated_policy, monkeypatch):
    tuner, hooks = make_tuner()
    monkeypatch.setattr(tuner, "_bench", forbid_benchmark)
    interception.install(AutotunePolicy(mode="pinned"))
    assert tuner.run(None, 128) == 32
    assert len(interception.choice_log()) == 1
    recorded = Mock(wraps=interception._record_choice)
    monkeypatch.setattr(interception, "_record_choice", recorded)
    for _ in range(3):
        assert tuner.run(None, 128) == 32
    recorded.assert_not_called()
    assert hooks == [32] * 4


def test_out_of_scope_autotuners_are_enumerated_but_not_logged(
    isolated_policy, monkeypatch, caplog
):
    tuner, _ = make_tuner(module="other_package.kernels")
    monkeypatch.setattr(
        tuner, "_bench", lambda *args, config, **kwargs: -config.kwargs["BLOCK_SIZE"]
    )
    interception.install(AutotunePolicy(mode="pinned", verify_every=1, enumerate_autotuners=True))
    with caplog.at_level(logging.WARNING, logger=interception.__name__):
        for size in range(100, 110):
            assert tuner.run(None, size) == 128
    assert interception.choice_log() == {}
    assert "UNPINNED other_package.kernels.test_kernel (2 configs)" in caplog.text


def test_config_invariant_kernels_keep_timed_choice(isolated_policy, monkeypatch, caplog):
    tuner, _ = make_tuner()
    timed = Mock(side_effect=lambda *args, config, **kwargs: -config.kwargs["BLOCK_SIZE"])
    monkeypatch.setattr(tuner, "_bench", timed)
    interception.install(
        AutotunePolicy(
            mode="pinned",
            enumerate_autotuners=True,
            config_invariant=("mamba_ssm.ops.triton.test.test_kernel",),
        )
    )
    with caplog.at_level(logging.WARNING, logger=interception.__name__):
        assert tuner.run(None, 128) == 128
    assert timed.call_count == 2
    assert interception.choice_log() == {}
    assert "TIMED    mamba_ssm.ops.triton.test.test_kernel" in caplog.text


def test_record_mode_does_not_record_forced_singletons(isolated_policy, monkeypatch, tmp_path):
    tuner, _ = make_tuner()
    tuner.configs = tuner.configs[:1]
    interception.install(AutotunePolicy(mode="record", record_path=str(tmp_path / "rec")))
    assert tuner.run(None, 128) == 32
    assert interception._tune_records == {}


def test_record_path_is_checked_when_installed(isolated_policy, monkeypatch, tmp_path):
    interception.install(AutotunePolicy(mode="pinned"))
    monkeypatch.setattr(os, "access", lambda path, mode: False)
    with pytest.raises(PermissionError, match="Cannot write Triton autotune recordings"):
        interception.install(AutotunePolicy(mode="record", record_path=str(tmp_path / "rec")))
    # A policy that could not be installed changes nothing.
    assert interception.active_policy().mode == "pinned"


def test_failed_record_dump_is_logged_not_raised(isolated_policy, monkeypatch, tmp_path, caplog):
    tuner, _ = make_tuner()
    monkeypatch.setattr(
        tuner, "_bench", lambda *args, config, **kwargs: -config.kwargs["BLOCK_SIZE"]
    )
    interception.install(AutotunePolicy(mode="record", record_path=str(tmp_path / "rec")))
    assert tuner.run(None, 128) == 128

    def refuse(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("builtins.open", refuse)
    with caplog.at_level(logging.ERROR, logger=interception.__name__):
        interception._dump_records()
    assert "Could not write Triton autotune recording" in caplog.text
