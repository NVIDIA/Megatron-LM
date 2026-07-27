# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Top-level conftest for the opt-in end-to-end (integration) test tree.

Its only job is to register the ``--run-e2e`` command-line flag *once*, at the
initial-conftest level, so that invoking a suite by path (the intended opt-in
entrypoint) makes the flag available during collection. The actual opt-in gate
— and everything expensive — lives in the per-suite leaf conftest so that a bare
``pytest tests`` run neither imports nor executes any of it.

Equivalent to ``--run-e2e``: set ``MCORE_CHECKPOINT_E2E=1`` in the environment
(the env var is the robust primary switch; the flag is convenience).
"""


def pytest_addoption(parser):
    parser.addoption(
        "--run-e2e",
        action="store_true",
        default=False,
        help=(
            "Opt in to the end-to-end GPU integration suites under "
            "tests/integration_tests (equivalently set MCORE_CHECKPOINT_E2E=1)."
        ),
    )
