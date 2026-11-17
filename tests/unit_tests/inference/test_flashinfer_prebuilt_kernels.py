# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""`missing_flashinfer_prebuilt_kernels` against a stand-in for FlashInfer's loader state."""

import hashlib
import sys
from types import ModuleType, SimpleNamespace
from unittest import mock

import pytest

from megatron.core.inference import utils as inference_utils

CUBIN_DIR = "fmha/trtllm-gen/"
CUBINS = ["include/flashInferMetaInfo.h", "fmhaSm100Kernel_a.cubin", "fmhaSm100Kernel_b.cubin"]
# FlashInfer's manifest: each file's SHA-256 and path. FlashInfer pins the manifest's own hash.
MANIFEST = "".join(f"{hashlib.sha256(name.encode()).hexdigest()} {name}\n" for name in CUBINS)


def _compatible(binary_target, device):
    """FlashInfer's rule for the targets used here: an ``a`` target runs only on its own SM."""
    return (3, 0) if binary_target.rstrip("af") == device else None


def _flashinfer(cubin_dir, provider_architectures=("sm100a",), aot_modules=("fmha_gen",)):
    """Modules standing in for flashinfer and flashinfer.jit.env as FlashInfer resolved them."""
    env = SimpleNamespace(
        FLASHINFER_AOT_PROVIDERS=tuple(
            SimpleNamespace(cuda_architectures=frozenset([architecture]))
            for architecture in provider_architectures
        ),
        FLASHINFER_CUBIN_DIR=cubin_dir,
        cuda_binary_target_compatibility_score=_compatible,
        get_aot_artifacts=lambda module: ("artifact",) if module in aot_modules else (),
    )
    artifacts = ModuleType("flashinfer.artifacts")
    artifacts.CheckSumHash = SimpleNamespace(
        map_checksums={f"{CUBIN_DIR}checksums.txt": hashlib.sha256(MANIFEST.encode()).hexdigest()}
    )
    flashinfer = ModuleType("flashinfer")
    flashinfer.__version__ = "0.7.2.post1"
    jit = ModuleType("flashinfer.jit")
    jit.env = env
    flashinfer.jit = jit
    flashinfer.artifacts = artifacts
    return {
        "flashinfer": flashinfer,
        "flashinfer.artifacts": artifacts,
        "flashinfer.jit": jit,
        "flashinfer.jit.env": env,
    }


def _write_cubins(root, names, manifest=MANIFEST):
    """FlashInfer's manifest and the listed files ``names``, each holding its own name."""
    directory = root / CUBIN_DIR
    (directory / "include").mkdir(parents=True)
    (directory / "checksums.txt").write_text(manifest)
    for name in names:
        (directory / name).write_bytes(name.encode())


def _missing(modules):
    with mock.patch.dict(sys.modules, modules):
        return inference_utils._missing_flashinfer_prebuilt_kernels.__wrapped__(
            "sm100", ("fmha_gen",), (CUBIN_DIR,)
        )


def test_complete_install_is_prebuilt(tmp_path):
    """A provider for the device, the module, the pinned manifest and every cubin it lists at
    its SHA-256: nothing is missing. The cubin directory is the one FlashInfer resolved:
    flashinfer-cubin, FLASHINFER_CUBIN_DIR or a cache that `flashinfer download-cubin` filled."""
    _write_cubins(tmp_path, CUBINS)
    assert _missing(_flashinfer(tmp_path)) is None


@pytest.mark.parametrize(
    ("provider_architectures", "aot_modules", "cubins", "expected"),
    [
        # An H100-only image (`download-kernels --sm sm90a`) on a GB200, or a shim installed
        # without providers.
        pytest.param(
            ("sm90a",), ("fmha_gen",), CUBINS, "no flashinfer-jit-cache provider", id="sm90a-only"
        ),
        pytest.param(
            (), ("fmha_gen",), CUBINS, "no flashinfer-jit-cache provider", id="no-provider"
        ),
        pytest.param(("sm100a",), (), CUBINS, "no prebuilt fmha_gen", id="no-module"),
        # A download cache that holds only the cubins of earlier on-demand calls.
        pytest.param(("sm100a",), ("fmha_gen",), CUBINS[:2], "incomplete", id="partial-cubins"),
        pytest.param(("sm100a",), ("fmha_gen",), None, "not in", id="no-cubins"),
    ],
)
def test_missing_piece_is_named(tmp_path, provider_architectures, aot_modules, cubins, expected):
    """FlashInfer would compile or download at its first call, so the check names what is
    missing."""
    if cubins is not None:
        _write_cubins(tmp_path, cubins)
    modules = _flashinfer(tmp_path, provider_architectures, aot_modules)
    missing = _missing(modules)
    assert missing is not None and expected in missing


def test_stale_cubin_is_named(tmp_path):
    """A cubin whose SHA-256 is not the manifest's: FlashInfer's loader rejects it and downloads
    a replacement, which fails offline, so the check names it."""
    _write_cubins(tmp_path, CUBINS)
    (tmp_path / CUBIN_DIR / CUBINS[1]).write_bytes(b"a cubin from another build")
    missing = _missing(_flashinfer(tmp_path))
    assert missing is not None and CUBINS[1] in missing and "SHA-256" in missing


@pytest.mark.parametrize(
    "manifest",
    [
        pytest.param("", id="empty"),
        # Every file it lists is present and matches, but FlashInfer pins the whole manifest.
        pytest.param(MANIFEST.splitlines(keepends=True)[0], id="truncated"),
        pytest.param("not a manifest\n", id="unparseable"),
    ],
)
def test_manifest_flashinfer_does_not_pin_is_named(tmp_path, manifest):
    """FlashInfer's loader takes checksums.txt only at the hash it pins and otherwise downloads
    it, so the check names a manifest with any other content."""
    _write_cubins(tmp_path, CUBINS, manifest)
    missing = _missing(_flashinfer(tmp_path))
    assert missing is not None and "checksums.txt" in missing and "pins" in missing


def test_flashinfer_not_installed():
    with mock.patch.dict(sys.modules, {"flashinfer": None, "flashinfer.jit": None}):
        missing = inference_utils._missing_flashinfer_prebuilt_kernels.__wrapped__("sm100", (), ())
    assert missing == "FlashInfer is not installed"


def test_moe_check_raises_with_the_missing_piece():
    """The FlashInfer MoE backend's check uses the same rule and names what is missing."""
    with mock.patch.object(
        inference_utils, "missing_flashinfer_prebuilt_kernels", return_value="nothing at all"
    ):
        with pytest.raises(RuntimeError, match="nothing at all"):
            inference_utils.check_flashinfer_jit_cache_installed()
    with mock.patch.object(
        inference_utils, "missing_flashinfer_prebuilt_kernels", return_value=None
    ):
        inference_utils.check_flashinfer_jit_cache_installed()
