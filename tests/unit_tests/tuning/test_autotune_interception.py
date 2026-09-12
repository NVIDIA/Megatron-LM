# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Test Triton's real Autotuner, with focused CUDA and distributed coverage."""

import inspect

import pytest

from megatron.core.tuning import interception, selection
from megatron.core.tuning.policy import AutotunePolicy

triton = pytest.importorskip("triton")
import triton.language as tl
from triton.runtime.autotuner import Autotuner


@pytest.fixture
def isolated_policy(monkeypatch):
    """Keep the process-wide patch and diagnostics local to each test."""
    monkeypatch.setattr(Autotuner, "run", inspect.unwrap(Autotuner.run))
    monkeypatch.setattr(interception, "_installed", False)
    monkeypatch.setattr(interception, "_policy", None)
    monkeypatch.setattr(interception, "_explicit_policy", False, raising=False)
    monkeypatch.setattr(interception, "_tables", {}, raising=False)
    monkeypatch.setattr(interception, "_choice_log", {})
    monkeypatch.setattr(interception, "_tune_records", {})
    monkeypatch.setattr(interception, "_enumerated", set())
    monkeypatch.setattr(selection, "_untuned_kernels_warned", set())
    monkeypatch.setattr(selection, "arch_tag", lambda: "test_arch")
    monkeypatch.setattr(interception.atexit, "register", lambda *_: None)
    for name in ("MCORE_AUTOTUNE_MODE", "MCORE_AUTOTUNE_RECORD", "MCORE_DET_TUNE_RECORD"):
        monkeypatch.delenv(name, raising=False)


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


def test_pinning_restores_state_after_launch_failure(isolated_policy):
    tuner, _ = make_tuner(fail=True)
    original_configs = tuner.configs
    interception.install(AutotunePolicy(mode="pinned"))
    with pytest.raises(RuntimeError, match="launch failed"):
        tuner.run(None, 128)
    assert tuner.configs is original_configs
    assert tuner.nargs is None
    assert interception.choice_log() == {}


def test_install_can_upgrade_an_observer_to_pinning(isolated_policy, monkeypatch):
    tuner, _ = make_tuner()
    interception.install(AutotunePolicy(mode="auto", verify_every=1))
    interception.install(AutotunePolicy(mode="pinned"))
    monkeypatch.setattr(tuner, "_bench", forbid_benchmark)
    assert tuner.run(None, 128) == 32
    assert interception.active_policy().mode == "pinned"


def test_environment_install_respects_explicit_policy(isolated_policy, monkeypatch):
    policy = AutotunePolicy(mode="pinned", modules=("my_kernels",))
    interception.install(policy)
    monkeypatch.setenv("MCORE_AUTOTUNE_MODE", "auto")
    interception.install_from_env()
    assert interception.active_policy() == policy


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
    interception.install(AutotunePolicy(mode="record", record_path=str(tmp_path / "rec")))
    assert tuner.run(None, 128) == 128
    record = interception._tune_records["test_arch"]["test_kernel"]["size=128"]
    assert record["kwargs"]["BLOCK_SIZE"] == 128


def test_table_distinguishes_all_launch_options(isolated_policy):
    from megatron.core.tuning.table import TunedTable

    ordinary = triton.Config({"BLOCK_SIZE": 32})
    clustered = triton.Config({"BLOCK_SIZE": 32}, num_ctas=2, maxnreg=64)
    assert selection.config_signature(ordinary) != selection.config_signature(clustered)
    table = TunedTable("test_arch", {"test_kernel": {"*": selection.config_data(clustered)}})
    assert table.lookup("test_kernel", "size=128", [ordinary, clustered]) is clustered


def test_transformer_config_requests_pinning(isolated_policy, monkeypatch):
    import torch

    from megatron.core.transformer.transformer_config import TransformerConfig

    monkeypatch.setenv("MAMBA_DETERMINISTIC", "0")
    monkeypatch.setattr(torch, "are_deterministic_algorithms_enabled", lambda: False)
    TransformerConfig(num_layers=1, hidden_size=16, num_attention_heads=1, deterministic_mode=True)
    assert interception.active_policy().mode == "pinned"
    TransformerConfig(num_layers=1, hidden_size=16, num_attention_heads=1)
    assert interception.active_policy().mode == "pinned"
    monkeypatch.setenv("MCORE_AUTOTUNE_MODE", "auto")
    interception.install_from_env()
    assert interception.active_policy().mode == "auto"


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
    gathers = iter([list(range(len(maps))), maps])

    def gather(output, value, group=None):
        output[:] = next(gathers)

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    monkeypatch.setattr(interception, "_policy", AutotunePolicy(verify_strict=True))
    if agrees:
        assert interception.verify_choices()
    else:
        with pytest.raises(RuntimeError, match="Ranks disagree on 1 autotune choice"):
            interception.verify_choices()


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
    interception._choice_log.update({"shared": "same", f"stage{rank}": "local"})
    assert interception.verify_choices()
    interception._choice_log["shared"] = str(rank % 2)
    with pytest.raises(RuntimeError, match="Ranks disagree on 1 autotune choice"):
        interception.verify_choices()
