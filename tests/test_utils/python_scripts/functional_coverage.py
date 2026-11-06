# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Collect coverage for one functional workload inside its disposable CI container."""

import argparse
import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
import sysconfig
import time
from pathlib import Path

import coverage

logger = logging.getLogger(__name__)

CONFIG = """[run]
source = megatron
branch = true
parallel = true
relative_files = true
concurrency = thread,multiprocessing
data_file = ${FUNCTIONAL_COVERAGE_DIR}/.coverage
context = ${FUNCTIONAL_COVERAGE_CONTEXT}|rank=${RANK-unset}|repeat=${REPEAT-unset}|phase=${RUN_NUMBER-unset}|restart=${TORCHELASTIC_RESTART_COUNT-0}
"""


def run_workload(script: Path, output: Path, metadata_path: Path) -> int:
    """Instrument subprocesses, preserve workload failures, and retain raw rank data."""
    output.mkdir(parents=True, exist_ok=False)
    metadata = json.loads(metadata_path.read_text())
    metadata.update(
        schema_version=1,
        coverage_version=coverage.__version__,
        python_version=platform.python_version(),
        machine=platform.machine(),
        attempt=output.name,
        collection_status="running",
        workload_exit_code=None,
        script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
    )
    model_config = (
        Path("tests/functional_tests/test_cases")
        / metadata["model"]
        / metadata["test_case"]
        / "model_config.yaml"
    )
    if model_config.is_file():
        metadata["model_config_sha256"] = hashlib.sha256(model_config.read_bytes()).hexdigest()
        (output / "model_config.yaml").write_bytes(model_config.read_bytes())

    config = output / "coverage.ini"
    config.write_text(CONFIG)
    manifest = output / "metadata.json"
    manifest.write_text(json.dumps(metadata, indent=2) + "\n")
    environment = {
        **os.environ,
        "COVERAGE_PROCESS_START": str(config.resolve()),
        "FUNCTIONAL_COVERAGE_DIR": str(output.resolve()),
        "FUNCTIONAL_COVERAGE_CONTEXT": f"{metadata['platform']}/{metadata['model']}/{metadata['test_case']}",
    }
    # A .pth hook preserves any image-provided sitecustomize. Every worker and
    # Python subprocess using this interpreter starts coverage before imports.
    # This helper runs only in the workload's disposable container, not the host.
    hook = Path(sysconfig.get_path("purelib")) / "_mcore_functional_coverage.pth"
    with hook.open("x") as stream:
        stream.write("import coverage; coverage.process_startup()\n")

    started = time.monotonic()
    try:
        result = subprocess.run(["bash", str(script.resolve())], env=environment, check=False)
        metadata["workload_exit_code"] = result.returncode
    finally:
        hook.unlink()
        metadata["elapsed_seconds"] = time.monotonic() - started
        manifest.write_text(json.dumps(metadata, indent=2) + "\n")

    # Reporting must not inherit worker startup or its rank/phase environment.
    # Pass the same run options explicitly rather than expanding worker config.
    collector = coverage.Coverage(
        data_file=str(output / ".coverage"), config_file=False, branch=True, source=["megatron"]
    )
    collector.set_option("run:relative_files", True)
    try:
        metadata["raw_data_files"] = sorted(path.name for path in output.glob(".coverage.*"))
        collector.combine(data_paths=[str(output)], strict=True, keep=True)
        collector.save()
        contexts = sorted(collector.get_data().measured_contexts())
        metadata["contexts"] = contexts
        metadata["observed_ranks"] = sorted(
            {
                int(part[5:])
                for context in contexts
                for part in context.split("|")
                if part.startswith("rank=") and part[5:].isdigit()
            }
        )
        collector.json_report(outfile=str(output / "coverage.json"), show_contexts=True)
        metadata["collection_status"] = "collected"
    except coverage.CoverageException as error:
        logger.exception("Could not collect functional coverage in %s", output)
        metadata["collection_status"] = "error"
        metadata["collection_error"] = str(error)
    finally:
        manifest.write_text(json.dumps(metadata, indent=2) + "\n")

    # Failed/timed-out attempts remain diagnostic evidence, never a successful
    # baseline. A passing workload with missing coverage also fails collection.
    return result.returncode or (0 if metadata["collection_status"] == "collected" else 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    sys.exit(run_workload(args.script, args.output, args.metadata))


if __name__ == "__main__":
    main()
