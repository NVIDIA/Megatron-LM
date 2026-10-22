# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""pytest configuration for the ``examples/multimodal_dev`` test suite.

This suite runs as its own CI bucket (see
``tests/test_utils/recipes/h100/unit-tests.yaml``), so
``tests/unit_tests/conftest.py`` is not an ancestor of the collected files
and none of its hooks apply here.  The hooks and fixtures the CI runner
(``tests/unit_tests/run_ci_test.sh``) depends on are reproduced below:

* ``pytest_configure`` — the NCCL unit-test defaults moved out of the shell
  runner into the main conftest; this independent bucket needs them too, plus
  nonblocking communicators for graceful CP ring shutdown in the CI container;
* ``--experimental`` — the runner invokes pytest a second time with this
  flag, which would otherwise be rejected as an unknown option;
* ``pytest_sessionfinish`` — that second invocation collects nothing in
  this bucket (there are no experimental tests), and pytest's "no tests
  collected" exit code 5 must not fail the job.  Unlike the unit-test
  suite's copy, the remap here is restricted to the ``--experimental``
  pass so an empty *production* collection still fails;
* ``cleanup`` — without it the NCCL process group is still alive at
  interpreter exit, which torch reports as a resource leak and which can
  hang teardown in the CI container.

Three fixtures are *deliberately* not reproduced:

* ``set_env`` — ``tests/unit_tests`` forces ``NVTE_FLASH_ATTN=0`` and
  ``NVTE_FUSED_ATTN=0`` on every test.  This suite needs the fused and
  flash attention backends *enabled*: the THD/packed-sequence paths under
  test are only reachable through them, and pinning them off would make
  the parity tests silently exercise the unfused fallback instead.
* ``reset_env_vars`` — the only environment mutation in this suite comes from
  ``Utils`` unsetting ``NVTE_FLASH_ATTN`` / ``NVTE_FUSED_ATTN`` /
  ``NVTE_UNFUSED_ATTN`` (``tests/unit_tests/test_utilities.py``), which leaves
  exactly the backend selection this bucket wants; nothing here sets them.
* ``ensure_test_data`` — no test in this bucket reads dataset assets.

(``tmp_path_dist_ckpt`` is not autouse upstream, so not reproducing it changes
nothing here.)

Because ``--experimental`` is registered both here and in
``tests/unit_tests/conftest.py``, the two directories cannot be passed to a
single pytest invocation -- pytest registers initial conftests in command-line
argument order and rejects the duplicate option. CI never does this; it passes
one bucket per run. Run them as separate commands locally.
"""

import os
import sys

import pytest
import torch
import torch.distributed

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from megatron.core import config  # noqa: E402


def _report_stderr(message):
    sys.stderr.write(message + "\n")
    sys.stderr.flush()


def pytest_configure(config):
    """Set the suite's NCCL defaults before any process group is created."""
    os.environ.setdefault("NCCL_MAX_NCHANNELS", "1")
    os.environ.setdefault("NCCL_NVLS_ENABLE", "0")
    # CP ring traffic creates several peer communicators in a process group.
    # Nonblocking finalization lets all peers start shutting down before waiting;
    # blocking finalization can hang when ranks visit those peers in different orders.
    # The CI image explicitly sets this to 0, so setdefault would leave the hang.
    # Pin only this test suite's mode; keep the normal destroy_process_group path.
    os.environ["TORCH_NCCL_USE_COMM_NONBLOCKING"] = "1"


def pytest_addoption(parser):
    """Mirror the ``--experimental`` option from tests/unit_tests/conftest.py."""
    parser.addoption(
        '--experimental',
        action='store_true',
        help="pass that argument to enable experimental flag during testing (DEFAULT: False)",
    )


@pytest.fixture(autouse=True)
def experimental(request):
    """Simple fixture setting the experimental flag [CPU | GPU]"""
    config.ENABLE_EXPERIMENTAL = request.config.getoption("--experimental") is True


@pytest.fixture(scope="session", autouse=True)
def cleanup():
    """Tear the process group down before exit, as the unit-test suite does."""
    yield
    if torch.distributed.is_initialized():
        rank = os.environ.get("RANK", "?")
        _report_stderr(f"[rank {rank}] session cleanup: barrier")
        try:
            if torch.cuda.is_available():
                torch.distributed.barrier(device_ids=[torch.cuda.current_device()])
            else:
                torch.distributed.barrier()
        except Exception:
            return
        _report_stderr(f"[rank {rank}] session cleanup: destroy process group")
        torch.distributed.destroy_process_group()
        _report_stderr(f"[rank {rank}] session cleanup: complete")


# Tests that must actually execute -- not merely be collected -- whenever the
# world size allows it.  Keyed by the module basename, valued by the minimum
# world size the module needs.
_MUST_RUN_MODULES = {"test_pp_support.py": 2}

_skipped_must_run = []


def pytest_runtest_logreport(report):
    """Report failures before distributed cleanup and track required-module skips."""
    if report.failed:
        rank = os.environ.get("RANK", "?")
        _report_stderr(f"\n[rank {rank}] {report.nodeid} ({report.when})\n{report.longreprtext}")
    if report.when != "setup" or not report.skipped:
        return
    module = os.path.basename(report.nodeid.split("::")[0])
    min_world_size = _MUST_RUN_MODULES.get(module)
    if min_world_size is None:
        return
    if int(os.environ.get("WORLD_SIZE", "1")) >= min_world_size:
        _skipped_must_run.append(report.nodeid)


def pytest_sessionfinish(session, exitstatus):
    """Treat "no tests collected" as success for the ``--experimental`` pass only.

    ``tests/unit_tests/conftest.py`` remaps exit code 5 unconditionally, but there
    the production pass collects hundreds of tests so the remap never fires for it.
    Here the empty collection is routine (no experimental tests exist in this
    bucket), so an unconditional remap would also hide a *production* pass that
    collected nothing -- a moved directory or a stale glob would report success
    having run no tests, which is precisely the failure this bucket exists to catch.

    Corollary: if every test here were ever marked ``flaky_in_dev``, the
    production pass would deselect everything and the job would go red with
    "no tests collected" rather than quietly skipping -- which is the right
    outcome for a bucket whose job is to prove this example still runs.
    """
    if exitstatus == 5 and session.config.getoption("--experimental"):
        session.exitstatus = 0

    # A skip is invisible in a green run, so a guard that silently stops
    # matching the CI topology (an "== 2 ranks" check under an 8-rank runner,
    # say) would take the pipeline-parallel coverage to zero without anything
    # going red. Fail loudly instead: if the world size was large enough for
    # these tests to run and they skipped anyway, the bucket is not testing
    # what it claims to.
    if _skipped_must_run:
        _report_stderr(
            "\nERROR: tests that must run in this topology were skipped "
            f"(WORLD_SIZE={os.environ.get('WORLD_SIZE', '1')}):\n  "
            + "\n  ".join(_skipped_must_run)
        )
        session.exitstatus = 1
