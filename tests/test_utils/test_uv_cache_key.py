# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU-only cache compatibility tests; run with unittest, without a GPU runtime."""

import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "docker/common/uv_cache_key.py"
SPEC = importlib.util.spec_from_file_location("uv_cache_key", SCRIPT)
assert SPEC is not None
assert SPEC.loader is not None
uv_cache_key = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(uv_cache_key)


class BuildEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.project = {
            "project": {"name": "example-project"},
            "dependency-groups": {"build": ["compiler", "torch"]},
            "tool": {
                "uv": {
                    "no-build-isolation-package": ["transformer-engine"],
                    "sources": {"transformer-engine": {"git": "https://example/te", "rev": "a"}},
                    "dependency-metadata": [{"name": "transformer-engine", "version": "1"}],
                }
            },
        }
        self.lock = {
            "requires-python": ">=3.12",
            "package": [
                {
                    "name": "example-project",
                    "source": {"editable": "."},
                    "dev-dependencies": {"build": [{"name": "compiler"}, {"name": "torch"}]},
                },
                {
                    "name": "compiler",
                    "version": "1",
                    "dependencies": [{"name": "shared-tool"}],
                    "optional-dependencies": {"extra": [{"name": "optional-tool"}]},
                    "build-dependencies": [{"name": "bootstrap"}],
                },
                {"name": "shared-tool", "version": "1"},
                {"name": "shared-tool", "version": "2", "marker": "sys_platform == 'win32'"},
                {"name": "optional-tool", "version": "1"},
                {"name": "bootstrap", "version": "1"},
                {"name": "transformer-engine", "source": {"git": "https://example/te#a"}},
                {"name": "requests", "version": "1"},
            ],
        }

    def environment(self):
        return copy.deepcopy(uv_cache_key.build_environment(self.project, self.lock))

    def test_runtime_and_te_updates_preserve_other_cached_wheels(self):
        before = self.environment()
        self.project["tool"]["uv"]["sources"]["transformer-engine"]["rev"] = "b"
        self.project["tool"]["uv"]["dependency-metadata"][0]["version"] = "2"
        self.lock["package"][-2]["source"]["git"] = "https://example/te#b"
        self.lock["package"][-1]["version"] = "2"
        self.assertEqual(before, self.environment())

    def test_all_transitive_marker_optional_and_build_dependencies_are_tracked(self):
        before = self.environment()
        for index in (2, 3, 4, 5):
            with self.subTest(package=self.lock["package"][index]):
                original = copy.deepcopy(self.lock)
                self.lock["package"][index]["version"] = "changed"
                self.assertNotEqual(before, self.environment())
                self.lock = original

    def test_lock_order_is_irrelevant(self):
        before = self.environment()
        self.lock["package"].reverse()
        self.assertEqual(before, self.environment())

    def test_build_flags_invalidate(self):
        before = self.environment()
        self.project["tool"]["uv"]["config-settings"] = {"build-option": "--debug"}
        self.assertNotEqual(before, self.environment())

    def test_extra_build_dependencies_are_tracked(self):
        self.project["tool"]["uv"]["extra-build-dependencies"] = {
            "transformer-engine": [{"requirement": "requests", "match-runtime": True}]
        }
        before = self.environment()
        self.lock["package"][-1]["version"] = "2"
        self.assertNotEqual(before, self.environment())

    def test_missing_build_dependency_fails_closed(self):
        self.lock["package"] = [p for p in self.lock["package"] if p["name"] != "bootstrap"]
        with self.assertRaisesRegex(ValueError, "bootstrap.*missing"):
            self.environment()

    def test_missing_build_group_fails_closed(self):
        del self.lock["package"][0]["dev-dependencies"]["build"]
        with self.assertRaisesRegex(ValueError, "no resolved build"):
            self.environment()

    def test_package_name_normalization(self):
        self.lock["package"][2]["name"] = "Shared_Tool"
        self.assertIn("Shared_Tool", [p["name"] for p in self.environment()["packages"]])


class CacheKeyTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.repository = Path(self.directory.name)
        for name in uv_cache_key.RECIPE_FILES:
            path = self.repository / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("recipe version 1\n")
        (self.repository / "pyproject.toml").write_text(
            '[project]\nname = "example"\n[dependency-groups]\nbuild = ["compiler"]\n'
        )
        (self.repository / "uv.lock").write_text(
            '[[package]]\nname = "example"\nsource = { editable = "." }\n'
            '[package.dev-dependencies]\nbuild = [{ name = "compiler" }]\n'
            '[[package]]\nname = "compiler"\nversion = "1"\n'
        )
        self.base_image = "registry.example/base@sha256:" + "a" * 64

    def key(self, **overrides):
        arguments = {
            "repository": self.repository,
            "base_image": self.base_image,
            "architecture": "amd64",
            "cuda_archs": "80;90;100",
        }
        arguments.update(overrides)
        return uv_cache_key.cache_key(**arguments)

    def test_base_digest_and_architecture_partition_cache(self):
        before = self.key()
        self.assertNotEqual(before, self.key(architecture="arm64"))
        self.assertNotEqual(before, self.key(cuda_archs="90"))
        self.assertNotEqual(before, self.key(base_image="example@sha256:" + "b" * 64))
        self.assertRegex(before, r"^[0-9a-f]{32}$")

    def test_each_recipe_and_helper_change_invalidates(self):
        before = self.key()
        for name in uv_cache_key.RECIPE_FILES:
            with self.subTest(file=name):
                path = self.repository / name
                original = path.read_text()
                path.write_text("recipe version 2\n")
                self.assertNotEqual(before, self.key())
                path.write_text(original)

    def test_mutable_image_tag_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "pinned by sha256"):
            self.key(base_image="registry.example/base:latest")

    def test_host_dependent_cuda_architecture_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "explicit semicolon-separated"):
            self.key(cuda_archs="native")


if __name__ == "__main__":
    unittest.main()
