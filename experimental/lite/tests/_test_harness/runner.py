# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Internal orchestrator for the public MLite local-test entry."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import signal
import subprocess
import sys
import tempfile
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

EXIT_SUCCESS = 0
EXIT_FAILURE = 1
EXIT_UNSUPPORTED_HARDWARE = 2
EXIT_EMPTY_PROFILE = 3

HARNESS_ROOT = Path(__file__).resolve().parent
TEST_ROOT = HARNESS_ROOT.parent
LITE_ROOT = TEST_ROOT.parent
REPO_ROOT = TEST_ROOT.parents[2]
WORKER_PATH = HARNESS_ROOT / "pytest_worker.py"
if str(HARNESS_ROOT) not in sys.path:
    sys.path.insert(0, str(HARNESS_ROOT))

from markers import (  # noqa: E402
    ARCHITECTURE_CAPABILITIES,
    ARCHITECTURE_ORDER,
    ENVIRONMENT_VARIABLES,
    architecture_supports,
)

TEST_ROOTS = (
    "experimental/lite/tests/model",
    "experimental/lite/tests/primitive",
    "experimental/lite/tests/runtime",
    "experimental/lite/tests/examples",
)
EXCLUDE_GLOBS = ("experimental/lite/tests/examples/test_verl_*.py",)
PYTEST_ARGS = (
    "-o",
    "addopts=",
    "-q",
    "-ra",
    "--strict-markers",
    "--capture=fd",
    "--disable-warnings",
    "-p",
    "no:cacheprovider",
)
HARDWARE_PROFILES = {
    "standard": (8, "hopper"),
    "blackwell": (4, "blackwell"),
}
COLLECTION_TIMEOUT_SECONDS = 300

_OUTCOME_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_DISTRIBUTED_ENV = {
    "GROUP_RANK",
    "LOCAL_RANK",
    "LOCAL_WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
    "RANK",
    "ROLE_RANK",
    "ROLE_WORLD_SIZE",
    "TORCHELASTIC_RESTART_COUNT",
    "TORCHELASTIC_RUN_ID",
    "WORLD_SIZE",
}
_TEST_CONTROL_ENV = {
    "CUBLAS_WORKSPACE_CONFIG",
    "NVTE_ALLOW_NONDETERMINISTIC_ALGO",
    "NVTE_FLASH_ATTN",
    "NVTE_FUSED_ATTN",
    "NVTE_UNFUSED_ATTN",
}
_TEST_CONTROL_PREFIXES = ("MEGATRON_LITE_", "MLITE_")


class CollectionError(RuntimeError):
    pass


class DependencyError(RuntimeError):
    pass


class SelectionError(ValueError):
    pass


class UnsupportedHardware(RuntimeError):
    pass


@dataclass(frozen=True)
class HardwareSelection:
    profile: str
    architecture: str | None
    gpu_count: int
    compute_capability: tuple[int, int] | None


@dataclass(frozen=True)
class TestCase:
    nodeid: str
    gpus: int
    min_architecture: str | None
    environment: tuple[tuple[str, str | None], ...]
    timeout_seconds: int
    optional: bool

    @property
    def path(self) -> str:
        return self.nodeid.split("::", 1)[0]


@dataclass(frozen=True)
class Suite:
    name: str
    gpus: int
    environment: tuple[tuple[str, str | None], ...]
    timeout_seconds: int
    targets: tuple[str, ...]


@dataclass(frozen=True)
class SuiteResult:
    name: str
    gpus: int
    duration_seconds: float
    status: str
    ranks_reported: int
    passed: int
    failed: int
    skipped: int
    xfailed: int
    xpassed: int
    errors: int


def _test_roots() -> list[Path]:
    roots = [(REPO_ROOT / value).resolve() for value in TEST_ROOTS]
    if not all(path.is_dir() for path in roots):
        raise SelectionError("the MLite test layout is incomplete")
    return roots


def resolve_targets(arguments: list[str], caller_cwd: Path) -> list[str]:
    roots = _test_roots()
    if not arguments:
        return list(TEST_ROOTS)

    selected: list[Path] = []
    for argument in arguments:
        if not argument or argument.startswith("-") or "::" in argument:
            raise SelectionError("test targets must be file or directory paths")
        raw_path = Path(argument)
        candidate = raw_path if raw_path.is_absolute() else caller_cwd / raw_path
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as exc:
            raise SelectionError("test target does not exist") from exc
        if not any(resolved == root or root in resolved.parents for root in roots):
            raise SelectionError(
                "test target is outside the four MLite test directories"
            )
        if resolved.is_file():
            if not resolved.name.startswith("test_") or resolved.suffix != ".py":
                raise SelectionError("test file targets must be named test_*.py")
        elif not resolved.is_dir():
            raise SelectionError("test target must be a file or directory")
        selected.append(resolved)

    deduplicated: list[Path] = []
    for path in sorted(
        set(selected), key=lambda value: (len(value.parts), value.as_posix())
    ):
        if any(path == parent or parent in path.parents for parent in deduplicated):
            continue
        deduplicated.append(path)
    return [path.relative_to(REPO_ROOT).as_posix() for path in deduplicated]


def _probe_hardware(torch_module, profile: str) -> HardwareSelection:
    if not torch_module.cuda.is_available():
        raise UnsupportedHardware("CUDA is unavailable")
    gpu_count = int(torch_module.cuda.device_count())
    if gpu_count <= 0:
        raise UnsupportedHardware("no GPUs are visible")

    capabilities = {
        tuple(int(value) for value in torch_module.cuda.get_device_capability(index))
        for index in range(gpu_count)
    }
    if len(capabilities) != 1:
        raise UnsupportedHardware("visible GPUs have mixed compute capabilities")
    capability = capabilities.pop()
    architecture = next(
        (
            name
            for name, expected_capability in ARCHITECTURE_CAPABILITIES.items()
            if expected_capability == capability
        ),
        None,
    )
    if architecture is None:
        raise UnsupportedHardware("visible GPU architecture is not supported")
    return HardwareSelection(profile, architecture, gpu_count, capability)


def detect_default_hardware(torch_module) -> HardwareSelection:
    detected = _probe_hardware(torch_module, "unselected")
    for profile, (gpu_count, architecture) in HARDWARE_PROFILES.items():
        if (
            detected.gpu_count == gpu_count
            and detected.architecture == architecture
        ):
            return HardwareSelection(
                profile,
                detected.architecture,
                detected.gpu_count,
                detected.compute_capability,
            )
    raise UnsupportedHardware("visible GPU topology is not a supported default profile")


def detect_subset_hardware(torch_module) -> HardwareSelection:
    return _probe_hardware(torch_module, "subset")


def _load_torch():
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"The pynvml package is deprecated\..*",
                category=FutureWarning,
            )
            import torch
    except Exception as exc:
        raise DependencyError("PyTorch with CUDA support is unavailable") from exc
    return torch


def _sanitized_environment(environ: dict[str, str] | None = None) -> dict[str, str]:
    base = dict(os.environ if environ is None else environ)
    for key in tuple(base):
        if (
            key in _DISTRIBUTED_ENV
            or key in _TEST_CONTROL_ENV
            or key in ENVIRONMENT_VARIABLES
            or key.startswith(_TEST_CONTROL_PREFIXES)
        ):
            base.pop(key)
    base.pop("PYTEST_ADDOPTS", None)
    base.pop("PYTEST_PLUGINS", None)
    base["PYTHONPATH"] = os.pathsep.join((str(REPO_ROOT), str(LITE_ROOT)))
    base["PYTHONDONTWRITEBYTECODE"] = "1"
    base["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    base["PYTHONHASHSEED"] = "0"
    base["PYTHONUNBUFFERED"] = "1"
    return base


def _decode_collected_case(raw: object) -> TestCase:
    if not isinstance(raw, dict):
        raise CollectionError("collected test entry must be an object")
    try:
        nodeid = raw["nodeid"]
        if not isinstance(nodeid, str):
            raise TypeError
        path = nodeid.split("::", 1)[0]
        if not any(path == root or path.startswith(f"{root}/") for root in TEST_ROOTS):
            raise ValueError
        environment = raw["environment"]
        return TestCase(
            nodeid=nodeid,
            gpus=raw["gpus"],
            min_architecture=raw["min_architecture"],
            environment=tuple(sorted(environment.items())),
            timeout_seconds=raw["timeout_seconds"],
            optional=raw["optional"],
        )
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise CollectionError("pytest collection plan is invalid") from exc


def collect_tests(
    targets: list[str],
    temporary_root: Path,
    environ: dict[str, str] | None = None,
) -> list[TestCase]:
    plan_path = temporary_root / "collection-plan.json"
    environment = _sanitized_environment(environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["MLITE_TEST_HARNESS"] = "1"
    environment["MLITE_TEST_PLAN_PATH"] = str(plan_path)

    command = [
        sys.executable,
        "-m",
        "pytest",
        f"--rootdir={REPO_ROOT}",
        "--collect-only",
        *PYTEST_ARGS,
        *(f"--ignore-glob={pattern}" for pattern in EXCLUDE_GLOBS),
        *targets,
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=COLLECTION_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise CollectionError("pytest collection timed out") from exc
    if completed.returncode not in {0, 5} or not plan_path.is_file():
        if completed.stdout:
            print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n")
        if completed.stderr:
            print(completed.stderr, end="" if completed.stderr.endswith("\n") else "\n")
        raise CollectionError("pytest collection failed")

    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CollectionError("cannot load pytest collection plan") from exc
    if not isinstance(plan, dict) or not isinstance(plan.get("tests"), list):
        raise CollectionError("pytest collection plan is invalid")

    cases = [_decode_collected_case(raw) for raw in plan["tests"]]
    nodeids = [case.nodeid for case in cases]
    if len(nodeids) != len(set(nodeids)):
        raise CollectionError("pytest collection returned duplicate nodeids")
    return cases


def select_cases(
    cases: list[TestCase],
    selection: HardwareSelection,
    explicit_targets: bool,
) -> list[TestCase]:
    selected: list[TestCase] = []
    for case in cases:
        if case.optional and not explicit_targets:
            continue
        if not explicit_targets:
            if selection.profile == "blackwell":
                if case.gpus == 0 or case.min_architecture != "blackwell":
                    continue
            elif case.gpus and case.min_architecture != selection.architecture:
                continue
        if case.gpus:
            if selection.architecture is None or case.min_architecture is None:
                raise SelectionError("selected GPU test has no usable architecture")
            if not architecture_supports(selection.architecture, case.min_architecture):
                raise SelectionError("selected test requires a newer GPU architecture")
            if case.gpus > selection.gpu_count:
                raise SelectionError(
                    "selected test requires more GPUs than are visible"
                )
        selected.append(case)
    return selected


def _suite_group(case: TestCase) -> tuple[object, ...]:
    execution = (
        case.gpus,
        case.min_architecture,
        case.environment,
        case.timeout_seconds,
    )
    return (*execution, "cpu" if case.gpus == 0 else case.path)


def _suite_name(key: tuple[object, ...], cases: list[TestCase]) -> str:
    if (
        cases[0].gpus == 0
        and cases[0].timeout_seconds == 1800
        and not cases[0].environment
    ):
        return "cpu"
    stem = Path(cases[0].path).stem.removeprefix("test_")
    readable = re.sub(r"[^a-z0-9]+", "-", f"{cases[0].gpus}gpu-{stem}".lower())
    digest = hashlib.sha256(repr(key).encode("utf-8")).hexdigest()[:8]
    return f"{readable[:54].strip('-')}-{digest}"


def build_suites(cases: list[TestCase]) -> list[Suite]:
    grouped: dict[tuple[object, ...], list[TestCase]] = {}
    for case in sorted(cases, key=lambda value: value.nodeid):
        grouped.setdefault(_suite_group(case), []).append(case)

    suites = [
        Suite(
            name=_suite_name(key, group),
            gpus=group[0].gpus,
            environment=group[0].environment,
            timeout_seconds=group[0].timeout_seconds,
            targets=tuple(case.nodeid for case in group),
        )
        for key, group in grouped.items()
    ]
    return sorted(suites, key=lambda suite: (suite.gpus, suite.name))


def _visible_devices(gpu_count: int, environ: dict[str, str]) -> str:
    configured = environ.get("CUDA_VISIBLE_DEVICES")
    if configured:
        entries = [entry.strip() for entry in configured.split(",") if entry.strip()]
        if len(entries) >= gpu_count:
            return ",".join(entries[:gpu_count])
    return ",".join(str(index) for index in range(gpu_count))


def build_suite_environment(
    suite: Suite,
    report_dir: Path,
    environ: dict[str, str] | None = None,
) -> dict[str, str]:
    original = dict(os.environ if environ is None else environ)
    visible_devices = _visible_devices(suite.gpus, original) if suite.gpus else ""
    base = _sanitized_environment(original)
    base["CUDA_VISIBLE_DEVICES"] = visible_devices
    base["MLITE_TEST_HARNESS"] = "1"
    base["MLITE_TEST_REPORT_DIR"] = str(report_dir)
    base["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    base["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] = "0"

    for name, value in suite.environment:
        if value is None:
            base.pop(name, None)
        else:
            base[name] = value
    return base


def build_suite_command(suite: Suite) -> list[str]:
    pytest_args = [f"--rootdir={REPO_ROOT}", *PYTEST_ARGS, *suite.targets]
    worker = str(WORKER_PATH)
    if suite.gpus == 0:
        return [sys.executable, worker, *pytest_args]
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={suite.gpus}",
        worker,
        *pytest_args,
    ]


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _run_command(
    command: list[str], env: dict[str, str], timeout: int
) -> tuple[int, bool]:
    try:
        process = subprocess.Popen(
            command, cwd=REPO_ROOT, env=env, start_new_session=True
        )
    except OSError:
        return EXIT_FAILURE, False
    try:
        return process.wait(timeout=timeout), False
    except subprocess.TimeoutExpired:
        _terminate_process_group(process)
        return EXIT_FAILURE, True
    except KeyboardInterrupt:
        _terminate_process_group(process)
        raise


def _load_rank_reports(report_dir: Path) -> list[dict[str, object]]:
    reports: list[dict[str, object]] = []
    for path in sorted(report_dir.glob("rank-*.json")):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(report, dict):
            reports.append(report)
    return reports


def _safe_count(counts: object, key: str) -> int:
    if not isinstance(counts, dict):
        return 0
    try:
        return int(counts.get(key, 0))
    except (TypeError, ValueError):
        return 0


def evaluate_suite_reports(
    suite: Suite,
    reports: list[dict[str, object]],
    process_exit_code: int,
    timed_out: bool,
    duration_seconds: float = 0.0,
) -> SuiteResult:
    expected_ranks = max(1, suite.gpus)
    valid = process_exit_code == 0 and not timed_out and len(reports) == expected_ranks
    ranks: set[int] = set()
    outcome_digests: set[str] = set()
    rank_zero: dict[str, object] | None = None

    for report in reports:
        rank = report.get("rank")
        digest = report.get("outcome_digest")
        counts = report.get("counts")
        if isinstance(rank, bool) or not isinstance(rank, int):
            valid = False
        else:
            ranks.add(rank)
            if rank == 0:
                rank_zero = report
        if report.get("status") != "PASS":
            valid = False
        if not isinstance(counts, dict):
            valid = False
        if not isinstance(digest, str) or not _OUTCOME_DIGEST.fullmatch(digest):
            valid = False
        else:
            outcome_digests.add(digest)

    if ranks != set(range(expected_ranks)) or len(outcome_digests) != 1:
        valid = False
    counts = rank_zero.get("counts") if rank_zero is not None else None
    return SuiteResult(
        name=suite.name,
        gpus=suite.gpus,
        duration_seconds=duration_seconds,
        status="PASS" if valid else "FAIL",
        ranks_reported=len(reports),
        passed=_safe_count(counts, "passed"),
        failed=_safe_count(counts, "failed"),
        skipped=_safe_count(counts, "skipped"),
        xfailed=_safe_count(counts, "xfailed"),
        xpassed=_safe_count(counts, "xpassed"),
        errors=_safe_count(counts, "error"),
    )


def run_suite(suite: Suite, temporary_root: Path) -> SuiteResult:
    report_dir = temporary_root / suite.name
    report_dir.mkdir(parents=True, exist_ok=False)
    command = build_suite_command(suite)
    environment = build_suite_environment(suite, report_dir)

    print(f"suite={suite.name} gpus={suite.gpus} status=RUNNING", flush=True)
    started = time.monotonic()
    process_exit_code, timed_out = _run_command(
        command, environment, suite.timeout_seconds
    )
    duration = time.monotonic() - started
    return evaluate_suite_reports(
        suite,
        _load_rank_reports(report_dir),
        process_exit_code,
        timed_out,
        duration,
    )


def _source_revision() -> str:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
    return f"{revision}+dirty" if dirty else revision


def _safe_version(value: object) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9.+_-]", "_", str(value))
    return sanitized[:80] or "unknown"


def _print_header(selection: HardwareSelection, torch_module) -> None:
    print("MLite local validation")
    print(f"profile={selection.profile}")
    print(f"source_revision={_source_revision()}")
    if selection.compute_capability is None:
        print(
            "hardware_gpu_count=not_required "
            "hardware_compute_capability=not_required "
            "hardware_architecture=not_required"
        )
    else:
        capability = (
            f"sm_{selection.compute_capability[0]}{selection.compute_capability[1]}"
        )
        print(
            f"hardware_gpu_count={selection.gpu_count} "
            f"hardware_compute_capability={capability} "
            f"hardware_architecture={selection.architecture}"
        )
    torch_version = "not_required"
    cuda_version = "not_required"
    if torch_module is not None:
        torch_version = _safe_version(torch_module.__version__)
        cuda_version = _safe_version(torch_module.version.cuda)
    print(
        f"software_python={_safe_version(platform.python_version())} "
        f"software_torch={torch_version} software_cuda={cuda_version}"
    )


def _print_summary(
    profile_name: str,
    results: list[SuiteResult],
    overall: str,
    exit_code: int,
    reason: str | None = None,
) -> None:
    print("MLite validation summary")
    print(f"profile={profile_name}")
    for result in results:
        print(
            f"suite={result.name} gpus={result.gpus} status={result.status} "
            f"duration_seconds={result.duration_seconds:.1f} ranks_reported={result.ranks_reported} "
            f"passed={result.passed} failed={result.failed} skipped={result.skipped} "
            f"xfailed={result.xfailed} xpassed={result.xpassed} errors={result.errors}"
        )
    suffix = f" reason={reason}" if reason else ""
    print(f"overall={overall} exit_code={exit_code}{suffix}")


def _stop(
    profile: str,
    overall: str,
    exit_code: int,
    reason: str | None = None,
    results: list[SuiteResult] | None = None,
    message: str | None = None,
) -> int:
    if message:
        print(message)
    _print_summary(profile, results or [], overall, exit_code, reason)
    return exit_code


def _hardware_failure(exc: Exception, profile: str) -> int:
    if isinstance(exc, DependencyError):
        return _stop(
            profile,
            "FAIL",
            EXIT_FAILURE,
            "mandatory_dependency_missing",
            message=f"dependency_error={exc}",
        )
    if isinstance(exc, UnsupportedHardware):
        return _stop(
            profile,
            "UNSUPPORTED",
            EXIT_UNSUPPORTED_HARDWARE,
            message=f"hardware_error={exc}",
        )
    return _stop(
        profile,
        "FAIL",
        EXIT_FAILURE,
        "hardware_probe_failed",
        message=f"hardware_error={type(exc).__name__}",
    )


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        targets = resolve_targets(arguments, Path.cwd())
    except SelectionError as exc:
        return _stop(
            "unselected",
            "FAIL",
            EXIT_UNSUPPORTED_HARDWARE,
            "invalid_test_target",
            message=f"selection_error={exc}",
        )

    explicit_targets = bool(arguments)
    selection: HardwareSelection | None = None
    torch_module = None
    if not explicit_targets:
        try:
            torch_module = _load_torch()
            selection = detect_default_hardware(torch_module)
        except Exception as exc:
            return _hardware_failure(exc, "unselected")
        _print_header(selection, torch_module)

    results: list[SuiteResult] = []
    with tempfile.TemporaryDirectory(prefix="mlite-local-validation-") as temporary:
        temporary_root = Path(temporary)
        try:
            collected = collect_tests(targets, temporary_root)
        except CollectionError as exc:
            return _stop(
                selection.profile if selection is not None else "unselected",
                "FAIL",
                EXIT_FAILURE,
                "collection_failed",
                message=f"collection_error={exc}",
            )

        if explicit_targets:
            if any(case.gpus for case in collected):
                try:
                    torch_module = _load_torch()
                    selection = detect_subset_hardware(torch_module)
                except Exception as exc:
                    return _hardware_failure(exc, "subset")
            else:
                selection = HardwareSelection("subset", None, 0, None)
            _print_header(selection, torch_module)

        assert selection is not None
        try:
            cases = select_cases(collected, selection, explicit_targets)
        except SelectionError as exc:
            return _stop(
                selection.profile,
                "UNSUPPORTED",
                EXIT_UNSUPPORTED_HARDWARE,
                "selected_tests_require_unsupported_hardware",
                message=f"selection_error={exc}",
            )

        suites = build_suites(cases)
        if not suites:
            reason = "no_selected_tests" if explicit_targets else "no_registered_tests"
            return _stop(
                selection.profile, "NOT_RUN", EXIT_EMPTY_PROFILE, reason
            )

        if any(suite.gpus for suite in suites):
            distributed = getattr(torch_module, "distributed", None)
            if (
                distributed is None
                or not distributed.is_available()
                or not distributed.is_nccl_available()
            ):
                return _stop(
                    selection.profile,
                    "FAIL",
                    EXIT_FAILURE,
                    "distributed_nccl_unavailable",
                )

        for suite in suites:
            try:
                result = run_suite(suite, temporary_root)
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                return _stop(
                    selection.profile,
                    "FAIL",
                    EXIT_FAILURE,
                    "runner_error",
                    results,
                    f"runner_error={type(exc).__name__}",
                )
            results.append(result)
            if result.status != "PASS":
                return _stop(
                    selection.profile, "FAIL", EXIT_FAILURE, results=results
                )

    _print_summary(selection.profile, results, "PASS", EXIT_SUCCESS)
    return EXIT_SUCCESS


if __name__ == "__main__":
    raise SystemExit(main())
