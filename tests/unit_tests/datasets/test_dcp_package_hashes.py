# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Recorded file identities of the runtime-scheduler package against the files on disk."""

import hashlib
import json
import re
from pathlib import Path

from megatron.core.datasets import dcp_runtime_scheduler
from megatron.core.datasets.dcp_runtime_scheduler.models import full35b, qwen3_h100_p2p

ROOT = Path(dcp_runtime_scheduler.__file__).parent
BENCHMARK = ROOT / "models" / "_benchmark"
# A Python file of the vendored tree whose entry lists only this change is the source file
# behind the vendor prefix.
PREFIX_ONLY = [
    "Prepend the copyright header and vendor-only lint comments; original code body unchanged"
]


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _files(directory, pattern="*"):
    return {
        path.relative_to(directory).as_posix()
        for path in directory.rglob(pattern)
        if path.is_file() and "__pycache__" not in path.parts
    }


def _digests(node, trail=()):
    """Key path of every sha256 value in a JSON document."""
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _digests(value, trail + (key,))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _digests(value, trail + (index,))
    elif isinstance(node, str) and re.fullmatch(r"[0-9a-f]{64}", node):
        yield trail


def test_origin_records_match_the_packaged_files():
    origin = json.loads((ROOT / "ORIGIN.json").read_text())
    parameters = origin["runtime_parameters_sha256"]
    # Listed and present agree: every JSON file of the package except the record itself.
    assert set(parameters) == _files(ROOT, "*.json") - {"ORIGIN.json"}
    for name, digest in parameters.items():
        assert _sha256(ROOT / name) == digest, name
    verified = {("runtime_parameters_sha256", name) for name in parameters}

    predictors = origin["benchmark_predictors"]
    assert predictors["vendor_manifest_sha256"] == _sha256(BENCHMARK / "manifest.json")
    assert predictors["source_manifest_sha256"] == _sha256(BENCHMARK / "UPSTREAM_MANIFEST.sha256")
    verified |= {
        ("benchmark_predictors", "vendor_manifest_sha256"),
        ("benchmark_predictors", "source_manifest_sha256"),
    }
    # The original DynamicUSP table is packaged unchanged under another name.
    table = ("dynamic_usp", "source_files_sha256", "profiles/calib_table_v1.json")
    assert origin["dynamic_usp"]["source_files_sha256"][table[-1]] == _sha256(
        ROOT / "models" / "usp_calib_v1.json"
    )
    candidate = ("full_qwen35", "source_candidate_sha256")
    packaged = json.loads((ROOT / origin["full_qwen35"]["packaged_manifest"]).read_text())
    assert origin["full_qwen35"][candidate[-1]] == packaged["source_candidate_sha256"]
    verified |= {table, candidate}

    # Every other digest names a Python file of a source repository, which was adapted here
    # and is therefore not a file of this package.
    remaining = set(_digests(origin)) - verified
    assert {trail[:-1] for trail in remaining} == {
        ("source_python_sha256",),
        ("dynamic_usp", "source_python_sha256"),
    }


def test_benchmark_manifest_matches_the_vendored_tree():
    manifest = json.loads((BENCHMARK / "manifest.json").read_text())
    files = manifest["files"]
    # Listed and present agree: every vendored file except the manifest itself.
    assert set(files) == _files(BENCHMARK) - {"manifest.json"}
    prefix = manifest["post_export_adaptation"]["python_prefix_utf8"].encode()
    for name, entry in files.items():
        raw = (BENCHMARK / name).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == entry["vendor_sha256"], name
        assert len(raw) == entry["vendor_bytes"], name
        source = (entry["source_sha256"], entry["source_bytes"])
        if not entry["changes"]:
            assert source == (entry["vendor_sha256"], entry["vendor_bytes"]), name
        elif entry["changes"] == PREFIX_ONLY:
            assert raw.startswith(prefix), name
            body = raw[len(prefix) :]
            assert source == (hashlib.sha256(body).hexdigest(), len(body)), name
        else:
            assert entry["source_sha256"] != entry["vendor_sha256"], name

    snapshot = manifest["source_snapshot"]
    assert snapshot["manifest_sha256"] == _sha256(BENCHMARK / "UPSTREAM_MANIFEST.sha256")
    assert snapshot["provenance_sha256"] == _sha256(BENCHMARK / "UPSTREAM_PROVENANCE.json")
    assert not snapshot["source_directory"].startswith(("/", "~"))
    rebound = manifest["hash_rebindings"]["integer_base_artifact_new_sha256"]
    assert rebound in {entry["vendor_sha256"] for entry in files.values()}

    # The source manifest: packaged entries carry its digests, and every other entry is
    # declared as not packaged.
    upstream = {}
    for line in (BENCHMARK / "UPSTREAM_MANIFEST.sha256").read_text().splitlines():
        digest, name = line.split(None, 1)
        upstream[name.strip()] = digest
    packaged = {entry["source_path"]: entry["source_sha256"] for entry in files.values()}
    for name, digest in packaged.items():
        if name in upstream:
            assert digest == upstream[name], name
    not_packaged = manifest["post_export_adaptation"]["source_files_not_packaged"]
    omitted = not_packaged["omitted_by_the_export"]
    prefixes = tuple(omitted["source_path_prefixes"])
    declared = set(omitted["source_paths"]) | set(
        not_packaged["removed_after_the_export"]["source_paths"]
    )
    assert declared <= set(upstream) and not declared & set(packaged)
    assert not any(name.startswith(prefixes) for name in packaged if name)
    assert all(any(name.startswith(prefix) for name in upstream) for prefix in prefixes)
    assert {name for name in upstream if name not in packaged} == declared | {
        name for name in upstream if name.startswith(prefixes)
    }


def test_loader_pins_match_the_parameter_files():
    origin = json.loads((ROOT / "ORIGIN.json").read_text())["runtime_parameters_sha256"]
    directory = ROOT / "models" / "full35b"
    assert full35b.PARAMETERS_SHA256 == _sha256(directory / "parameters.json")
    assert full35b.PARAMETERS_SHA256 == origin["models/full35b/parameters.json"]
    packaged = json.loads((directory / "manifest.json").read_text())["packaged_sha256"]
    assert packaged["parameters.json"] == full35b.PARAMETERS_SHA256
    for name, digest in packaged.items():
        assert _sha256(directory / name) == digest, name
    assert qwen3_h100_p2p.PACKAGED_SHA256 == _sha256(ROOT / "models" / "qwen3_h100_p2p.json")
    assert qwen3_h100_p2p.PACKAGED_SHA256 == origin["models/qwen3_h100_p2p.json"]
