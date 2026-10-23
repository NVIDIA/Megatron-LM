# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

HELPER = Path(__file__).resolve().parents[2] / "docker/common/prepare_optional_backends.py"
SPEC = importlib.util.spec_from_file_location("prepare_optional_backends", HELPER)
assert SPEC is not None and SPEC.loader is not None
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)
HOOK = "nvidia_cutlass_dsl_packages.pth"


@pytest.fixture
def packages(tmp_path, monkeypatch):
    venv, system = tmp_path / "venv", tmp_path / "system"
    source, base = venv / "flash_attn/cute", system / "flash_attn"
    files = {
        source / "interface.py": "new FA4",
        base / "__init__.py": "base FA2",
        base / "flash_attn_interface.py": "compiled FA2 interface",
        base / "cute/stale.py": "old FA2 cute",
        venv / HOOK: "dsl_packages",
        system / HOOK: "dsl_packages",
        system / "unrelated.pth": "keep unrelated",
        system / "nvidia_cutlass_dsl_unrecorded.pth": "keep unowned",
    }
    for path, content in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    def distribution(root, name, entries=()):
        return SimpleNamespace(
            metadata={"Name": name}, version="test", files=entries, locate_file=root.__truediv__
        )

    distributions = {
        venv: [
            distribution(venv, "flash-attn-4"),
            distribution(venv, "nvidia-cutlass-dsl", [HOOK]),
        ],
        system: [
            distribution(system, "nvidia-cutlass-dsl", [HOOK]),
            distribution(system, "unrelated", ["unrelated.pth"]),
        ],
    }
    fa2_spec = SimpleNamespace(origin=str(base / "__init__.py"))
    # Replace only the helper's module references, leaving pytest's import machinery intact.
    monkeypatch.setattr(
        prepare,
        "importlib",
        SimpleNamespace(
            metadata=SimpleNamespace(distributions=lambda *, path: distributions[Path(path[0])]),
            util=SimpleNamespace(find_spec=lambda name: fa2_spec),
        ),
    )
    monkeypatch.setattr(prepare, "sysconfig", SimpleNamespace(get_path=lambda name: str(venv)))
    monkeypatch.setattr(prepare, "site", SimpleNamespace(getsitepackages=lambda: [venv, system]))
    return SimpleNamespace(venv=venv, system=system, source=source, base=base, fa2_spec=fa2_spec)


def test_replaces_only_fa4_and_owned_system_path_hook(packages):
    prepare.main()
    assert (packages.base / "cute/interface.py").read_text() == "new FA4"
    assert not (packages.base / "cute/stale.py").exists()
    assert (packages.base / "__init__.py").read_text() == "base FA2"
    assert (packages.base / "flash_attn_interface.py").read_text() == "compiled FA2 interface"
    assert (packages.system / "unrelated.pth").read_text() == "keep unrelated"
    assert (packages.system / "nvidia_cutlass_dsl_unrecorded.pth").read_text() == "keep unowned"
    assert not (packages.system / HOOK).exists()
    assert (packages.system / (HOOK + ".disabled-by-mcore")).read_text() == "dsl_packages"
    assert (packages.venv / HOOK).read_text() == "dsl_packages"


def test_repeated_preparation_is_idempotent(packages):
    prepare.main()
    before = {path: path.read_bytes() for path in packages.system.rglob("*") if path.is_file()}
    prepare.main()
    after = {path: path.read_bytes() for path in packages.system.rglob("*") if path.is_file()}
    assert after == before


def test_same_site_preserves_fa4_source(packages):
    packages.fa2_spec.origin = str(packages.venv / "flash_attn/__init__.py")
    (packages.venv / "flash_attn/flash_attn_interface.py").write_text("FA2")
    prepare.main()
    assert (packages.source / "interface.py").read_text() == "new FA4"


@pytest.mark.parametrize(
    "invalid, message",
    [
        ("missing_fa4", "Missing locked"),
        ("symlinked_target", "symlinked FA2"),
        ("unexpected_hook", "Unexpected system"),
    ],
)
def test_rejects_unsafe_or_incomplete_package_layout(packages, invalid, message):
    if invalid == "missing_fa4":
        (packages.source / "interface.py").unlink()
    elif invalid == "symlinked_target":
        (packages.base / "cute/stale.py").unlink()
        (packages.base / "cute").rmdir()
        (packages.base / "cute").symlink_to(packages.source, target_is_directory=True)
    else:
        (packages.system / HOOK).write_text("unrecognized hook")
    with pytest.raises(RuntimeError, match=message):
        prepare.main()
