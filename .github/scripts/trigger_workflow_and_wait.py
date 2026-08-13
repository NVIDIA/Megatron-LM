# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Trigger and wait for Megatron-Bridge validation with refreshable App credentials."""

from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

API_ROOT = "https://api.github.com"
TARGET_OWNER = "NVIDIA-NeMo"
TARGET_REPO = "Megatron-Bridge"
TARGET_WORKFLOW = "cicd-main.yml"
TARGET_BASE_REF = "main"
ALLOWED_CALLER = "NVIDIA/Megatron-LM"
ALLOWED_SUITES = {"L1", "unit-only"}
TOKEN_REFRESH_MARGIN_MS = 10 * 60 * 1000
DISCOVERY_TIMEOUT_MS = 10 * 60 * 1000
POLL_INTERVAL_MS = 60 * 1000

JsonObject = dict[str, Any]
ApiRequest = Callable[[str, str, str, JsonObject | None], JsonObject | None]
Now = Callable[[], int]
Sleep = Callable[[int], None]


@dataclass(frozen=True)
class Config:
    """Validated inputs for one MBridge orchestration attempt."""

    app_id: str
    private_key: str
    mcore_ref: str
    test_suite: str
    poll_timeout_seconds: int
    caller_repository: str
    run_id: str
    run_attempt: str
    server_url: str


@dataclass(frozen=True)
class Credential:
    """One scoped, expiring installation token."""

    token: str
    expires_at_ms: int


class ApiError(RuntimeError):
    """A bounded GitHub API failure."""

    def __init__(self, method: str, path: str, status: int, response: str) -> None:
        super().__init__(f"GitHub API {method} {path} failed with HTTP {status}: {response[:1000]}")
        self.status = status


def _base64_url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def create_app_jwt(app_id: str, private_key: str, now_ms: int | None = None) -> str:
    """Create a short-lived GitHub App JWT using the runner's OpenSSL."""

    current_ms = int(time.time() * 1000) if now_ms is None else now_ms
    issued_at = current_ms // 1000 - 60
    header = _base64_url(json.dumps({"alg": "RS256", "typ": "JWT"}).encode())
    payload = _base64_url(
        json.dumps({"iat": issued_at, "exp": issued_at + 9 * 60, "iss": app_id}).encode()
    )
    signing_input = f"{header}.{payload}".encode()
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8") as key_file:
        key_file.write(private_key)
        key_file.flush()
        signature = subprocess.run(
            ["openssl", "dgst", "-sha256", "-sign", key_file.name],
            input=signing_input,
            check=True,
            capture_output=True,
        ).stdout
    return f"{header}.{payload}.{_base64_url(signature)}"


def _parse_timeout(raw_timeout: str) -> int:
    try:
        value = float(raw_timeout)
    except ValueError as error:
        raise ValueError("poll_timeout_seconds must be an integer between 60 and 14400") from error
    if not value.is_integer() or value < 60 or value > 4 * 60 * 60:
        raise ValueError("poll_timeout_seconds must be an integer between 60 and 14400")
    return int(value)


def read_config(env: dict[str, str] | None = None) -> Config:
    """Read and validate the bounded workflow-call contract."""

    values = os.environ if env is None else env
    config = Config(
        app_id=values.get("BOT_ID", ""),
        private_key=values.get("BOT_KEY", ""),
        mcore_ref=values.get("MBRIDGE_MCORE_REF", ""),
        test_suite=values.get("MBRIDGE_TEST_SUITE", ""),
        caller_repository=values.get("GITHUB_REPOSITORY", ""),
        poll_timeout_seconds=_parse_timeout(values.get("MBRIDGE_POLL_TIMEOUT_SECONDS", "12600")),
        run_id=values.get("GITHUB_RUN_ID", ""),
        run_attempt=values.get("GITHUB_RUN_ATTEMPT", ""),
        server_url=values.get("GITHUB_SERVER_URL", "https://github.com"),
    )
    validate_config(config)
    return config


def validate_config(config: Config) -> None:
    """Fail closed on any untrusted caller-controlled input."""

    if re.fullmatch(r"\d+", config.app_id) is None:
        raise ValueError("The GitHub App ID must be numeric")
    if re.search(r"-----BEGIN (?:RSA )?PRIVATE KEY-----", config.private_key) is None:
        raise ValueError("The GitHub App private key is missing or malformed")
    if re.fullmatch(r"[0-9a-f]{40}", config.mcore_ref) is None:
        raise ValueError("mcore_ref must be a full lowercase commit SHA")
    if config.test_suite not in ALLOWED_SUITES:
        raise ValueError(f"Unsupported MBridge test suite: {config.test_suite}")
    if config.caller_repository != ALLOWED_CALLER:
        raise ValueError(f"This workflow only accepts calls from {ALLOWED_CALLER}")
    if not 60 <= config.poll_timeout_seconds <= 4 * 60 * 60:
        raise ValueError("poll_timeout_seconds must be an integer between 60 and 14400")
    if (
        re.fullmatch(r"\d+", config.run_id) is None
        or re.fullmatch(r"\d+", config.run_attempt) is None
    ):
        raise ValueError("GitHub run ID and attempt must be numeric")
    if config.server_url != "https://github.com":
        raise ValueError("Only github.com callers are supported")


def testing_branch(config: Config) -> str:
    return f"mcore-testing-{config.run_id}-{config.run_attempt}"


def triggered_by(config: Config) -> str:
    return f"{config.server_url}/{config.caller_repository}/actions/runs/{config.run_id}"


def api_request(
    token: str, method: str, path: str, body: JsonObject | None = None
) -> JsonObject | None:
    """Make one bounded GitHub REST request."""

    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(
        f"{API_ROOT}{path}",
        data=data,
        method=method,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            response_text = response.read().decode()
            return json.loads(response_text) if response_text else None
    except urllib.error.HTTPError as error:
        response_text = error.read().decode(errors="replace")
        raise ApiError(method, path, error.code, response_text) from error


def _parse_timestamp(value: str) -> int:
    return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1000)


def create_token_minter(
    config: Config, *, request: ApiRequest = api_request, now: Now = lambda: int(time.time() * 1000)
) -> Callable[[dict[str, str]], Credential]:
    """Return a minter that scopes every token to one operation."""

    installation_id: int | None = None

    def mint_token(permissions: dict[str, str]) -> Credential:
        nonlocal installation_id
        jwt = create_app_jwt(config.app_id, config.private_key, now())
        if installation_id is None:
            installation = request(
                jwt, "GET", f"/repos/{TARGET_OWNER}/{TARGET_REPO}/installation", None
            )
            if installation is None:
                raise RuntimeError("GitHub returned an empty installation response")
            installation_id = int(installation["id"])
        response = request(
            jwt,
            "POST",
            f"/app/installations/{installation_id}/access_tokens",
            {"repositories": [TARGET_REPO], "permissions": permissions},
        )
        if response is None:
            raise RuntimeError("GitHub returned an empty token response")
        return Credential(
            token=str(response["token"]),
            expires_at_ms=_parse_timestamp(str(response["expires_at"])),
        )

    return mint_token


def runs_path(branch: str) -> str:
    query = urllib.parse.urlencode(
        {"event": "workflow_dispatch", "branch": branch, "per_page": "100"}
    )
    workflow = urllib.parse.quote(TARGET_WORKFLOW, safe="")
    return f"/repos/{TARGET_OWNER}/{TARGET_REPO}/actions/workflows/{workflow}/runs?{query}"


def select_dispatched_run(
    runs: list[JsonObject], old_run_ids: set[int], branch: str, started_at_ms: int
) -> JsonObject | None:
    candidates = [
        run
        for run in runs
        if int(run["id"]) not in old_run_ids
        and run.get("event") == "workflow_dispatch"
        and run.get("head_branch") == branch
        and _parse_timestamp(str(run["created_at"])) >= started_at_ms - 2 * 60 * 1000
    ]
    return min(candidates, key=lambda run: _parse_timestamp(str(run["created_at"])), default=None)


def discover_run(
    token: str,
    old_run_ids: set[int],
    branch: str,
    started_at_ms: int,
    *,
    request: ApiRequest = api_request,
    now: Now = lambda: int(time.time() * 1000),
    sleep: Sleep = lambda milliseconds: time.sleep(milliseconds / 1000),
) -> int:
    deadline = now() + DISCOVERY_TIMEOUT_MS
    while now() < deadline:
        response = request(token, "GET", runs_path(branch), None)
        if response is None:
            raise RuntimeError("GitHub returned an empty workflow-runs response")
        run = select_dispatched_run(
            list(response["workflow_runs"]), old_run_ids, branch, started_at_ms
        )
        if run is not None:
            return int(run["id"])
        sleep(POLL_INTERVAL_MS)
    raise TimeoutError("Timed out waiting for the dispatched MBridge workflow run to appear")


def poll_run(
    run_id: int,
    timeout_seconds: int,
    mint_token: Callable[[dict[str, str]], Credential],
    *,
    request: ApiRequest = api_request,
    now: Now = lambda: int(time.time() * 1000),
    sleep: Sleep = lambda milliseconds: time.sleep(milliseconds / 1000),
) -> None:
    deadline = now() + timeout_seconds * 1000
    credential: Credential | None = None
    while True:
        if credential is None or credential.expires_at_ms - now() <= TOKEN_REFRESH_MARGIN_MS:
            credential = mint_token({"actions": "read"})
        run = request(
            credential.token,
            "GET",
            f"/repos/{TARGET_OWNER}/{TARGET_REPO}/actions/runs/{run_id}",
            None,
        )
        if run is None:
            raise RuntimeError("GitHub returned an empty workflow-run response")
        print(f"MBridge workflow status={run['status']} conclusion={run['conclusion']}")
        if run["status"] == "completed":
            if run["conclusion"] != "success":
                raise RuntimeError(f"MBridge workflow {run_id} concluded with {run['conclusion']}")
            return
        remaining = deadline - now()
        if remaining <= 0:
            raise TimeoutError(f"Timed out waiting for MBridge workflow {run_id} to complete")
        sleep(min(POLL_INTERVAL_MS, remaining))


def delete_branch(
    mint_token: Callable[[dict[str, str]], Credential],
    branch: str,
    *,
    request: ApiRequest = api_request,
) -> None:
    credential = mint_token({"contents": "write"})
    branch_name = urllib.parse.quote(branch, safe="")
    try:
        request(
            credential.token,
            "DELETE",
            f"/repos/{TARGET_OWNER}/{TARGET_REPO}/git/refs/heads/{branch_name}",
            None,
        )
    except ApiError as error:
        if error.status != 404:
            raise


def orchestrate(
    config: Config,
    *,
    request: ApiRequest = api_request,
    now: Now = lambda: int(time.time() * 1000),
    sleep: Sleep = lambda milliseconds: time.sleep(milliseconds / 1000),
    mint_token: Callable[[dict[str, str]], Credential] | None = None,
) -> None:
    """Create, dispatch, monitor, and clean up one isolated MBridge run."""

    token_minter = mint_token or create_token_minter(config, request=request, now=now)
    branch = testing_branch(config)
    branch_created = False
    failure: Exception | None = None
    try:
        branch_credential = token_minter({"contents": "write"})
        base_ref = request(
            branch_credential.token,
            "GET",
            f"/repos/{TARGET_OWNER}/{TARGET_REPO}/git/ref/heads/{TARGET_BASE_REF}",
            None,
        )
        if base_ref is None:
            raise RuntimeError("GitHub returned an empty base-ref response")
        request(
            branch_credential.token,
            "POST",
            f"/repos/{TARGET_OWNER}/{TARGET_REPO}/git/refs",
            {"ref": f"refs/heads/{branch}", "sha": base_ref["object"]["sha"]},
        )
        branch_created = True
        print(f"Created temporary MBridge branch {branch}")

        trigger_credential = token_minter({"actions": "write"})
        prior_runs = request(trigger_credential.token, "GET", runs_path(branch), None)
        if prior_runs is None:
            raise RuntimeError("GitHub returned an empty workflow-runs response")
        old_run_ids = {int(run["id"]) for run in prior_runs["workflow_runs"]}
        started_at_ms = now()
        workflow = urllib.parse.quote(TARGET_WORKFLOW, safe="")
        request(
            trigger_credential.token,
            "POST",
            f"/repos/{TARGET_OWNER}/{TARGET_REPO}/actions/workflows/{workflow}/dispatches",
            {
                "ref": branch,
                "inputs": {
                    "mcore_ref": config.mcore_ref,
                    "test_suite": config.test_suite,
                    "triggered_by": triggered_by(config),
                },
            },
        )
        run_id = discover_run(
            trigger_credential.token,
            old_run_ids,
            branch,
            started_at_ms,
            request=request,
            now=now,
            sleep=sleep,
        )
        print(f"Discovered MBridge workflow run {run_id}")
        poll_run(
            run_id, config.poll_timeout_seconds, token_minter, request=request, now=now, sleep=sleep
        )
    except Exception as error:  # noqa: BLE001 - cleanup must run for every operational failure
        failure = error
    finally:
        if branch_created:
            try:
                delete_branch(token_minter, branch, request=request)
                print(f"Deleted temporary MBridge branch {branch}")
            except Exception as cleanup_error:  # noqa: BLE001 - preserve the primary failure
                if failure is None:
                    failure = cleanup_error
                else:
                    print(f"MBridge branch cleanup also failed: {cleanup_error}", file=sys.stderr)
    if failure is not None:
        raise failure


def main() -> None:
    orchestrate(read_config())


if __name__ == "__main__":
    main()
