# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def converter_plugins(tmp_path):
    plugins = tmp_path / "plugins"
    plugins.mkdir()
    (plugins / "loader_fixture.py").write_text(
        "def add_arguments(parser):\n"
        "    pass\n"
        "def load_checkpoint(queue, args):\n"
        "    queue.put({'name': 'fixture', 'value': [1, 2, 3]})\n"
    )
    (plugins / "saver_fixture.py").write_text(
        "import json\n"
        "from pathlib import Path\n"
        "def add_arguments(parser):\n"
        "    pass\n"
        "def save_checkpoint(queue, args):\n"
        "    payload = queue.get()\n"
        "    (Path(args.save_dir) / 'weights.json').write_text(json.dumps(payload))\n"
    )
    return plugins


def _run_converter(plugins: Path, output: Path) -> subprocess.CompletedProcess[str]:
    repository = Path(__file__).resolve().parents[4]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(plugins), str(repository), environment.get("PYTHONPATH", "")]
    )
    return subprocess.run(
        [
            sys.executable,
            str(repository / "tools/checkpoint/convert.py"),
            "--model-type",
            "GPT",
            "--loader",
            "fixture",
            "--saver",
            "fixture",
            "--load-dir",
            str(plugins),
            "--save-dir",
            str(output),
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def test_converter_reports_saver_write_failure(tmp_path, converter_plugins) -> None:
    output = tmp_path / "not-a-directory"
    output.write_text("existing file")

    result = _run_converter(converter_plugins, output)

    assert "NotADirectoryError" in result.stderr
    assert result.returncode != 0, result.stdout + result.stderr
    assert output.read_text() == "existing file"


def test_converter_succeeds_when_saver_completes(tmp_path, converter_plugins) -> None:
    output = tmp_path / "checkpoint"
    output.mkdir()

    result = _run_converter(converter_plugins, output)

    assert result.returncode == 0, result.stdout + result.stderr
    assert (output / "weights.json").is_file()
