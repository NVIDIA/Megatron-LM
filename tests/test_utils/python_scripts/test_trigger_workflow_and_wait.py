# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
from pathlib import Path
from unittest import mock

import pytest

MODULE_PATH = Path(__file__).parents[3] / ".github/scripts/trigger_workflow_and_wait.py"
SPEC = importlib.util.spec_from_file_location("trigger_workflow_and_wait", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def config(**overrides):
    values = {
        "app_id": "123",
        "private_key": "-----BEGIN PRIVATE KEY-----\nfixture\n-----END PRIVATE KEY-----",
        "mcore_ref": "a" * 40,
        "test_suite": "L1",
        "poll_timeout_seconds": 12600,
        "caller_repository": "NVIDIA/Megatron-LM",
        "run_id": "456",
        "run_attempt": "2",
        "server_url": "https://github.com",
    }
    values.update(overrides)
    return MODULE.Config(**values)


def test_read_config_accepts_workflow_contract():
    result = MODULE.read_config(
        {
            "BOT_ID": "123",
            "BOT_KEY": "-----BEGIN PRIVATE KEY-----\nfixture\n-----END PRIVATE KEY-----",
            "MBRIDGE_MCORE_REF": "a" * 40,
            "MBRIDGE_TEST_SUITE": "L1",
            "MBRIDGE_POLL_TIMEOUT_SECONDS": "12600",
            "GITHUB_REPOSITORY": "NVIDIA/Megatron-LM",
            "GITHUB_RUN_ID": "456",
            "GITHUB_RUN_ATTEMPT": "2",
            "GITHUB_SERVER_URL": "https://github.com",
        }
    )
    assert result == config()


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"app_id": "not-numeric"}, "App ID"),
        ({"private_key": "missing"}, "private key"),
        ({"mcore_ref": "main"}, "full lowercase commit SHA"),
        ({"test_suite": "unsupported"}, "Unsupported"),
        ({"poll_timeout_seconds": 1}, "between 60 and 14400"),
        ({"caller_repository": "attacker/fork"}, "only accepts calls"),
        ({"run_id": "not-numeric"}, "run ID and attempt"),
        ({"server_url": "https://example.test"}, "Only github.com"),
    ],
)
def test_validate_config_rejects_untrusted_inputs(override, message):
    with pytest.raises(ValueError, match=message):
        MODULE.validate_config(config(**override))


def test_create_app_jwt_is_short_lived_and_invokes_openssl():
    completed = mock.Mock(stdout=b"signature")
    with mock.patch.object(MODULE.subprocess, "run", return_value=completed) as run:
        jwt = MODULE.create_app_jwt("123", config().private_key, now_ms=1_800_000)
    assert jwt.count(".") == 2
    run.assert_called_once()
    assert run.call_args.args[0][:4] == ["openssl", "dgst", "-sha256", "-sign"]
    assert run.call_args.kwargs["pass_fds"]
    assert run.call_args.args[0][4].startswith("/proc/self/fd/")


def test_poll_run_refreshes_actions_read_tokens():
    credentials = iter(
        [MODULE.Credential("token-1", 15 * 60 * 1000), MODULE.Credential("token-2", 60 * 60 * 1000)]
    )
    minted = []

    def mint(permissions):
        minted.append(permissions)
        return next(credentials)

    now = iter([0, 10 * 60 * 1000, 10 * 60 * 1000, 10 * 60 * 1000])
    responses = iter(
        [
            {"status": "in_progress", "conclusion": None},
            {"status": "completed", "conclusion": "success"},
        ]
    )
    MODULE.poll_run(
        77,
        12600,
        mint,
        request=lambda *_: next(responses),
        now=lambda: next(now),
        sleep=lambda _: None,
    )
    assert minted == [{"actions": "read"}, {"actions": "read"}]


def test_poll_run_fails_closed_on_timeout():
    credential = MODULE.Credential("token", 60 * 60 * 1000)
    now = iter([0, 0, 1_000, 1_000])
    with pytest.raises(TimeoutError, match="workflow 77"):
        MODULE.poll_run(
            77,
            1,
            lambda _: credential,
            request=lambda *_: {"status": "in_progress", "conclusion": None},
            now=lambda: next(now),
            sleep=lambda _: None,
        )


def test_orchestrate_scopes_tokens_and_cleans_up():
    minted = []
    requests = []
    credentials = iter(
        [
            MODULE.Credential("branch", 60 * 60 * 1000),
            MODULE.Credential("trigger", 60 * 60 * 1000),
            MODULE.Credential("poll", 60 * 60 * 1000),
            MODULE.Credential("cleanup", 60 * 60 * 1000),
        ]
    )

    def mint(permissions):
        minted.append(permissions)
        return next(credentials)

    run_queries = 0

    def request(token, method, path, body):
        nonlocal run_queries
        requests.append((token, method, path, body))
        if path.endswith("/git/ref/heads/main"):
            return {"object": {"sha": "b" * 40}}
        if path.endswith("/runs?event=workflow_dispatch&branch=mcore-testing-456-2&per_page=100"):
            count = sum(1 for request_entry in requests if request_entry[2] == path)
            if count == 1:
                return {"workflow_runs": []}
            return {
                "workflow_runs": [
                    {
                        "id": 77,
                        "event": "workflow_dispatch",
                        "head_branch": "mcore-testing-456-2",
                        "created_at": "1970-01-01T00:00:01Z",
                    }
                ]
            }
        if path.endswith("/actions/runs/77"):
            return {"status": "completed", "conclusion": "success"}
        return None

    times = iter([1_000, 1_000, 1_000, 1_000, 1_000, 1_000, 1_000])
    MODULE.orchestrate(
        config(), request=request, now=lambda: next(times), sleep=lambda _: None, mint_token=mint
    )
    assert minted == [
        {"contents": "write"},
        {"actions": "write"},
        {"actions": "read"},
        {"contents": "write"},
    ]
    assert any(
        method == "DELETE" and path.endswith("/mcore-testing-456-2")
        for _, method, path, _ in requests
    )


def test_orchestrate_preserves_test_failure_when_cleanup_fails():
    credentials = iter(
        [
            MODULE.Credential("branch", 60 * 60 * 1000),
            MODULE.Credential("trigger", 60 * 60 * 1000),
            MODULE.Credential("poll", 60 * 60 * 1000),
            MODULE.Credential("cleanup", 60 * 60 * 1000),
        ]
    )

    run_queries = 0

    def request(token, method, path, body):
        nonlocal run_queries
        if path.endswith("/git/ref/heads/main"):
            return {"object": {"sha": "b" * 40}}
        if "/runs?" in path:
            run_queries += 1
            if run_queries == 1:
                return {"workflow_runs": []}
            return {
                "workflow_runs": [
                    {
                        "id": 77,
                        "event": "workflow_dispatch",
                        "head_branch": "mcore-testing-456-2",
                        "created_at": "1970-01-01T00:00:01Z",
                    }
                ]
            }
        if path.endswith("/actions/runs/77"):
            return {"status": "completed", "conclusion": "failure"}
        if method == "DELETE":
            raise RuntimeError("cleanup failed")
        return None

    with pytest.raises(RuntimeError, match="concluded with failure"):
        MODULE.orchestrate(
            config(),
            request=request,
            now=lambda: 1_000,
            sleep=lambda _: None,
            mint_token=lambda _: next(credentials),
        )
