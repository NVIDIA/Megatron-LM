#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Exercise the CI shell and pytest guard without importing Megatron or using GPUs."""

import base64
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import types
import unittest
from pathlib import Path
from unittest import mock

import yaml
from click.testing import CliRunner

REPO_ROOT = Path(__file__).resolve().parents[2]


def _encode(files: object) -> str:
    return base64.urlsafe_b64encode(json.dumps(files).encode()).decode()


class TestSelectiveTestRuntime(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.repo_root = Path(directory.name).resolve() / "repo"
        self.repo_root.mkdir()
        self.bin_dir = self.repo_root / "bin"
        self.bin_dir.mkdir()
        self.command_log = self.repo_root / "commands.jsonl"
        self._write("tests/__init__.py")
        self._write("tests/unit_tests/__init__.py")
        self._write(
            "tests/unit_tests/conftest.py",
            """
            def pytest_addoption(parser):
                parser.addoption('--experimental', action='store_true')
            """,
        )
        self._write(
            "pytest.ini",
            """
            [pytest]
            markers =
                experimental: experimental suite
                internal: unavailable in legacy
                flaky: unavailable in lts
                flaky_in_dev: unavailable in dev
                launch_on_gb200: enabled on gb200
            """,
        )
        self._write("tests/unit_tests/test_selected.py", "def test_selected(): pass\n")
        self._write("tests/unit_tests/test_unselected.py", "def test_unselected(): assert False\n")
        shutil.copyfile(
            REPO_ROOT / "tests/unit_tests/selective_test_guard.py",
            self.repo_root / "tests/unit_tests/selective_test_guard.py",
        )
        # Change only the container mount path; execute the real runner logic.
        runner = (REPO_ROOT / "tests/unit_tests/run_ci_test.sh").read_text()
        runner = runner.replace("/opt/megatron-lm-legacy/", str(self.repo_root))
        runner = runner.replace("/opt/megatron-lm", str(self.repo_root))
        self.runner = self._write("tests/unit_tests/run_ci_test.sh", runner)
        self._write(
            "tests/unit_tests/find_test_cases.py",
            """
            import os
            import sys

            if os.environ.get('FAIL_BUCKET_LOOKUP'):
                raise SystemExit(3)
            print('--ignore=tests/unit_tests/test_unselected.py')
            """,
        )
        # Keep the exact pytest arguments and use real pytest. Simulate torchrun's
        # nonzero-worker -> launcher-exit-1 mapping, including empty collection.
        self._executable(
            "uv",
            """
            import json
            import os
            import subprocess
            import sys

            with open(os.environ['COMMAND_LOG'], 'a') as stream:
                stream.write(json.dumps(sys.argv[1:]) + '\\n')
            pytest_args = sys.argv[sys.argv.index('pytest') + 1:]
            result = subprocess.run([sys.executable, '-m', 'pytest', *pytest_args])
            raise SystemExit(0 if result.returncode == 0 else 1)
            """,
        )
        self._executable(
            "coverage",
            """
            import json
            import os
            import sys

            with open(os.environ['COMMAND_LOG'], 'a') as stream:
                stream.write(json.dumps(['coverage', *sys.argv[1:]]) + '\\n')
            """,
        )
        (self.bin_dir / "python").symlink_to(sys.executable)
        # macOS ships Bash 3, while CI runs Bash 5 (mapfile requires Bash 4).
        self.bash = shutil.which("bash")
        if sys.platform == "darwin" and Path("/opt/homebrew/bin/bash").exists():
            self.bash = "/opt/homebrew/bin/bash"

    def _write(self, relative_path: str, contents: str = "") -> Path:
        path = self.repo_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(contents))
        return path

    def _executable(self, name: str, contents: str) -> None:
        path = self._write(f"bin/{name}", f"#!{sys.executable}\n" + textwrap.dedent(contents))
        path.chmod(0o755)

    def _run(
        self,
        files: object = None,
        *,
        payload: str | None = None,
        tag: str = "latest",
        environment: str = "dev",
        platform: str = "h100",
        repeat: int = 1,
        extra_env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess:
        env = {
            **os.environ,
            "PATH": f"{self.bin_dir}{os.pathsep}{os.environ['PATH']}",
            "PYTHONPATH": str(self.repo_root),
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "COMMAND_LOG": str(self.command_log),
            "UNIT_TEST_FILES_B64": payload if payload is not None else _encode(files),
            **(extra_env or {}),
        }
        return subprocess.run(
            [
                self.bash,
                str(self.runner),
                "--tag",
                tag,
                "--environment",
                environment,
                "--platform",
                platform,
                "--bucket",
                "tests/unit_tests/**/*.py",
                "--unit-test-repeat",
                str(repeat),
                "--log-dir",
                str(self.repo_root / "logs"),
            ],
            cwd=self.repo_root,
            env=env,
            text=True,
            capture_output=True,
            timeout=30,
        )

    def _commands(self) -> list[list[str]]:
        if not self.command_log.exists():
            return []
        return [json.loads(line) for line in self.command_log.read_text().splitlines()]

    def test_exact_files_reach_both_marker_phases_and_every_repeat(self) -> None:
        spaced_file = "tests/unit_tests/test_path with spaces.py"
        self._write(
            spaced_file,
            """
            import pytest

            @pytest.mark.experimental
            def test_experimental(): pass
            """,
        )
        selected = ["tests/unit_tests/test_selected.py", spaced_file]
        result = self._run(selected, repeat=2)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        commands = self._commands()
        self.assertEqual(len(commands), 5)
        for index, command in enumerate(commands[:-1]):
            self.assertEqual(command[-2:], selected)
            self.assertIn("tests.unit_tests.selective_test_guard", command)
            self.assertEqual("--experimental" in command, index % 2 == 1)
            self.assertEqual("coverage" in command, index % 2 == 0)
            self.assertFalse(any(arg.startswith("--ignore=") for arg in command))
        self.assertEqual(commands[-1], ["coverage", "combine", "-q"])

    def test_experimental_only_selection_survives_empty_production_phase(self) -> None:
        self._write(
            "tests/unit_tests/test_selected.py",
            """
            import pytest

            @pytest.mark.experimental
            def test_selected(): pass
            """,
        )
        result = self._run(["tests/unit_tests/test_selected.py"])
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(len(self._commands()), 3)

    def test_marker_filtered_selection_fails_even_on_gb200(self) -> None:
        for platform in ("h100", "gb200"):
            with self.subTest(platform=platform):
                self._write(
                    "tests/unit_tests/test_selected.py",
                    """
                    import pytest

                    @pytest.mark.flaky_in_dev
                    def test_selected(): pass
                    """,
                )
                result = self._run(["tests/unit_tests/test_selected.py"], platform=platform)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("No selectively chosen tests survived", result.stdout)

    def test_invalid_or_stale_payload_fails_before_launching_pytest(self) -> None:
        for files in (
            [],
            None,
            {},
            [42],
            ["tests/unit_tests/test_missing.py"],
            ["tests/unit_tests/test_selected.py"] * 2,
            ["tests/unit_tests/../test_outside.py"],
            ["/tests/unit_tests/test_selected.py"],
            ["tests/unit_tests/test_selected.py\n"],
            ["tests/unit_tests/conftest.py"],
        ):
            with self.subTest(files=files):
                result = self._run(files)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("invalid or stale", result.stdout)
                self.assertEqual(self._commands(), [])
        result = self._run(payload="not-base64!")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self._commands(), [])

    def test_symlink_outside_unit_suite_is_rejected(self) -> None:
        outside = self._write("test_outside.py", "def test_outside(): pass\n")
        (self.repo_root / "tests/unit_tests/test_symlink.py").symlink_to(outside)
        result = self._run(["tests/unit_tests/test_symlink.py"])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("outside tests/unit_tests", result.stderr)
        self.assertEqual(self._commands(), [])

    def test_pytest_failures_and_collection_errors_propagate(self) -> None:
        for contents in ("def test_selected(): assert False\n", "this is invalid python!\n"):
            with self.subTest(contents=contents):
                self.command_log.unlink(missing_ok=True)
                self._write("tests/unit_tests/test_selected.py", contents)
                result = self._run(["tests/unit_tests/test_selected.py"])
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(len(self._commands()), 1)

    def test_full_bucket_keeps_exclusions_coverage_and_marker_splits(self) -> None:
        self._write(
            "tests/unit_tests/test_experimental.py",
            """
            import pytest

            @pytest.mark.experimental
            def test_experimental(): pass
            """,
        )
        result = self._run(payload="")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        commands = self._commands()
        self.assertEqual(len(commands), 3)
        for command in commands[:-1]:
            self.assertEqual(command[-1], "tests/unit_tests")
            self.assertIn("--ignore=tests/unit_tests/test_unselected.py", command)
            self.assertNotIn("tests.unit_tests.selective_test_guard", command)
        self.assertIn("coverage", commands[0])
        self.assertNotIn("coverage", commands[1])
        self.assertEqual(commands[-1], ["coverage", "combine", "-q"])

    def test_legacy_lts_selection_preserves_marker_filters(self) -> None:
        result = self._run(["tests/unit_tests/test_selected.py"], tag="legacy", environment="lts")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        commands = self._commands()
        self.assertEqual(len(commands), 2)
        self.assertIn("not experimental and not internal and not flaky", commands[0])
        self.assertNotIn("--experimental", commands[0])

    def test_full_bucket_discovery_failure_stops_the_job(self) -> None:
        result = self._run(payload="", extra_env={"FAIL_BUCKET_LOOKUP": "1"})
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self._commands(), [])


class TestSelectiveWorkloadTransport(unittest.TestCase):
    def test_action_passes_the_encoded_selection_and_stops_on_setup_failure(self) -> None:
        action = yaml.safe_load((REPO_ROOT / ".github/actions/action.yml").read_text())
        step = next(
            step
            for step in action["runs"]["steps"]
            if step["name"] == "Create run-script (unit test)"
        )
        payload = _encode(["tests/unit_tests/test_path with spaces.py"])
        values = {
            "test_case": "tests/unit_tests/**/*.py",
            "platform": "dgx_h100",
            "tag": "latest",
            "container-image": "test-image",
        }
        script = step["run"]
        for key, value in values.items():
            script = script.replace("${{ inputs." + key + " }}", value)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            command_log = root / "commands.jsonl"
            uv = root / "uv"
            uv.write_text(f"#!{sys.executable}\n" + textwrap.dedent("""
                    import json
                    import os
                    import sys

                    with open(os.environ['COMMAND_LOG'], 'a') as stream:
                        stream.write(json.dumps(sys.argv[1:]) + '\\n')
                    if os.environ.get('FAIL_SYNC') and sys.argv[1] == 'sync':
                        raise SystemExit(17)
                    """))
            uv.chmod(0o755)
            environment = {
                **os.environ,
                "PATH": f"{root}{os.pathsep}{os.environ['PATH']}",
                "COMMAND_LOG": str(command_log),
                "UNIT_TEST_FILES_B64": payload,
            }
            subprocess.run(
                ["bash", "-euo", "pipefail", "-c", script],
                cwd=root,
                env=environment,
                capture_output=True,
                text=True,
                check=True,
            )
            for fail_setup in (False, True):
                with self.subTest(fail_setup=fail_setup):
                    command_log.unlink(missing_ok=True)
                    result = subprocess.run(
                        ["bash", "job.sh"],
                        cwd=root,
                        env={**environment, "FAIL_SYNC": "1" if fail_setup else ""},
                        capture_output=True,
                        text=True,
                    )
                    commands = [json.loads(line) for line in command_log.read_text().splitlines()]
                    if fail_setup:
                        self.assertEqual(result.returncode, 17, result.stderr)
                        self.assertEqual(commands[-1][0], "sync")
                    else:
                        self.assertEqual(result.returncode, 0, result.stderr)
                        launch = commands[-1]
                        self.assertEqual(launch[launch.index("--unit-test-files") + 1], payload)
                        self.assertEqual(
                            launch[launch.index("--test-case") + 1], "tests/unit_tests/**/*.py"
                        )

    def test_cli_payload_reaches_docker_environment_unchanged(self) -> None:
        recipe_parser = types.ModuleType("recipe_parser")
        recipe_parser.load_workloads = mock.Mock(
            return_value=[
                types.SimpleNamespace(
                    type="basic",
                    spec={
                        "name": "unit-tests",
                        "test_case": "tests/unit_tests/**/*.py",
                        "script": "echo selected",
                    },
                )
            ]
        )
        scripts = types.ModuleType("tests.test_utils.python_scripts")
        scripts.recipe_parser = recipe_parser
        nemo_run = mock.MagicMock()
        experiment = nemo_run.Experiment.return_value.__enter__.return_value
        experiment.status.return_value = {"task": {"status": "SUCCEEDED"}}
        modules = {
            "nemo_run": nemo_run,
            "tests": types.ModuleType("tests"),
            "tests.test_utils": types.ModuleType("tests.test_utils"),
            "tests.test_utils.python_scripts": scripts,
        }
        module_path = REPO_ROOT / "tests/test_utils/python_scripts/launch_nemo_run_workload.py"
        spec = importlib.util.spec_from_file_location("selective_runtime_launcher", module_path)
        module = importlib.util.module_from_spec(spec)
        with mock.patch.dict(sys.modules, modules):
            spec.loader.exec_module(module)

        for payload in ("", _encode(["tests/unit_tests/test_path with spaces.py"])):
            with self.subTest(payload=payload):
                args = [
                    "--scope",
                    "unit-tests",
                    "--model",
                    "unit-tests",
                    "--test-case",
                    "tests/unit_tests/**/*.py",
                    "--environment",
                    "dev",
                    "--platform",
                    "dgx_h100",
                    "--container-image",
                    "test-image",
                ]
                if payload:
                    args.extend(["--unit-test-files", payload])
                result = CliRunner().invoke(module.main, args)
                self.assertEqual(result.exit_code, 0, result.output or repr(result.exception))
                environment = nemo_run.DockerExecutor.call_args.kwargs["env_vars"]
                self.assertEqual(environment["UNIT_TEST_FILES_B64"], payload)


if __name__ == "__main__":
    unittest.main()
