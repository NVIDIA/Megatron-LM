#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trigger a GitHub workflow and wait with refreshable GitHub App authentication."""

import argparse
import base64
import json
import os
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime

API_VERSION = "2022-11-28"
TOKEN_REFRESH_MARGIN_SECONDS = 5 * 60


class GitHubApiError(RuntimeError):
    """A failed GitHub API request."""

    def __init__(self, status: int, body: str):
        super().__init__(f"GitHub API request failed with HTTP {status}: {body}")
        self.status = status


def _base64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def create_app_jwt(app_id: str, private_key: str, now: int) -> str:
    """Create a short-lived GitHub App JWT without persisting the private key."""
    header = _base64url(json.dumps({"alg": "RS256", "typ": "JWT"}, separators=(",", ":")).encode())
    payload = _base64url(
        json.dumps(
            {"iat": now - 60, "exp": now + 9 * 60, "iss": app_id}, separators=(",", ":")
        ).encode()
    )
    unsigned_token = f"{header}.{payload}"

    key_path = ""
    try:
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as key_file:
            key_file.write(private_key)
            key_path = key_file.name
        os.chmod(key_path, 0o600)
        signature = subprocess.run(
            ["openssl", "dgst", "-sha256", "-sign", key_path],
            input=unsigned_token.encode(),
            check=True,
            capture_output=True,
        ).stdout
    finally:
        if key_path:
            os.unlink(key_path)

    return f"{unsigned_token}.{_base64url(signature)}"


def request_json(
    api_url: str, method: str, path: str, token: str, payload: dict | None = None
) -> dict | None:
    """Send one authenticated GitHub API request and decode its JSON response."""
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        f"{api_url.rstrip('/')}{path}",
        data=data,
        method=method,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "X-GitHub-Api-Version": API_VERSION,
        },
    )
    try:
        with urllib.request.urlopen(request) as response:
            body = response.read()
    except urllib.error.HTTPError as error:
        raise GitHubApiError(error.code, error.read().decode(errors="replace")) from error
    return json.loads(body) if body else None


class InstallationAuth:
    """Mint and refresh repository-scoped GitHub App installation tokens."""

    def __init__(
        self, app_id, private_key, owner, repo, api_url, clock=time.time, request=request_json
    ):
        self.app_id = app_id
        self.private_key = private_key
        self.owner = owner
        self.repo = repo
        self.api_url = api_url
        self.clock = clock
        self.request = request
        self._token = ""
        self._expires_at = 0.0

    def invalidate(self) -> None:
        """Force the next request to mint a new token."""
        self._token = ""
        self._expires_at = 0.0

    def _revoke(self, token: str) -> None:
        """Revoke one installation token without masking the workflow result."""
        try:
            self.request(self.api_url, "DELETE", "/installation/token", token)
        except GitHubApiError as error:
            print(f"Warning: failed to revoke installation token (HTTP {error.status})")

    def token(self) -> str:
        """Return a token with at least five minutes of remaining validity."""
        now = self.clock()
        if self._token and self._expires_at - now > TOKEN_REFRESH_MARGIN_SECONDS:
            return self._token

        if self._token:
            self._revoke(self._token)
            self.invalidate()

        app_jwt = create_app_jwt(self.app_id, self.private_key, int(now))
        installation = self.request(
            self.api_url, "GET", f"/repos/{self.owner}/{self.repo}/installation", app_jwt
        )
        minted = self.request(
            self.api_url,
            "POST",
            f"/app/installations/{installation['id']}/access_tokens",
            app_jwt,
            {"repositories": [self.repo]},
        )
        self._token = minted["token"]
        self._expires_at = datetime.fromisoformat(
            minted["expires_at"].replace("Z", "+00:00")
        ).timestamp()
        return self._token

    def close(self) -> None:
        """Revoke the current installation token."""
        if not self._token:
            return
        self._revoke(self._token)
        self.invalidate()


def installation_request(
    auth: InstallationAuth, method: str, path: str, payload: dict | None = None
) -> dict | None:
    """Make an installation request, refreshing once after an expired token."""
    for attempt in range(2):
        try:
            return auth.request(auth.api_url, method, path, auth.token(), payload)
        except GitHubApiError as error:
            if error.status != 401 or attempt:
                raise
            auth.invalidate()
    raise AssertionError("unreachable")


def workflow_runs_path(owner: str, repo: str, workflow: str, ref: str, created_after: str) -> str:
    query = urllib.parse.urlencode(
        {
            "event": "workflow_dispatch",
            "branch": ref,
            "created": f">={created_after}",
            "per_page": 100,
        }
    )
    return f"/repos/{owner}/{repo}/actions/workflows/{urllib.parse.quote(workflow, safe='')}/runs?{query}"


def append_output(name: str, value: str) -> None:
    if output_path := os.getenv("GITHUB_OUTPUT"):
        with open(output_path, "a", encoding="utf-8") as output_file:
            output_file.write(f"{name}={value}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--workflow", required=True)
    parser.add_argument("--ref", required=True)
    parser.add_argument("--client-payload", default="{}")
    parser.add_argument("--wait-interval", type=int, default=60)
    parser.add_argument("--discovery-timeout", type=int, default=10 * 60)
    parser.add_argument("--timeout", type=int, default=6 * 60 * 60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client_payload = json.loads(args.client_payload)
    if not isinstance(client_payload, dict):
        raise ValueError("--client-payload must be a JSON object")

    api_url = os.getenv("GITHUB_API_URL", "https://api.github.com")
    auth = InstallationAuth(
        os.environ["GITHUB_APP_ID"],
        os.environ["GITHUB_APP_PRIVATE_KEY"],
        args.owner,
        args.repo,
        api_url,
    )
    started_at = time.time()
    created_after = (
        datetime.fromtimestamp(started_at - 120).astimezone().isoformat(timespec="seconds")
    )
    runs_path = workflow_runs_path(args.owner, args.repo, args.workflow, args.ref, created_after)

    try:
        old_runs_response = installation_request(auth, "GET", runs_path)
        assert old_runs_response is not None
        old_run_ids = {run["id"] for run in old_runs_response["workflow_runs"]}
        workflow = urllib.parse.quote(args.workflow, safe="")
        print(f"Triggering {args.owner}/{args.repo} workflow {args.workflow} on {args.ref}")
        installation_request(
            auth,
            "POST",
            f"/repos/{args.owner}/{args.repo}/actions/workflows/{workflow}/dispatches",
            {"ref": args.ref, "inputs": client_payload},
        )

        discovery_deadline = time.time() + args.discovery_timeout
        run = None
        while time.time() < discovery_deadline:
            runs_response = installation_request(auth, "GET", runs_path)
            assert runs_response is not None
            runs = runs_response["workflow_runs"]
            new_runs = [candidate for candidate in runs if candidate["id"] not in old_run_ids]
            if new_runs:
                run = min(new_runs, key=lambda candidate: candidate["created_at"])
                break
            time.sleep(args.wait_interval)
        if run is None:
            raise TimeoutError("Timed out waiting for the dispatched workflow run to appear")

        run_id = run["id"]
        server_url = os.getenv("GITHUB_SERVER_URL", "https://github.com")
        workflow_url = f"{server_url}/{args.owner}/{args.repo}/actions/runs/{run_id}"
        append_output("workflow_id", str(run_id))
        append_output("workflow_url", workflow_url)
        print(f"Waiting for workflow {workflow_url}")

        deadline = time.time() + args.timeout
        while time.time() < deadline:
            run = installation_request(
                auth, "GET", f"/repos/{args.owner}/{args.repo}/actions/runs/{run_id}"
            )
            assert run is not None
            print(f"Workflow status={run['status']} conclusion={run['conclusion']}")
            if run["status"] == "completed":
                append_output("conclusion", str(run["conclusion"]))
                if run["conclusion"] != "success":
                    raise RuntimeError(f"Downstream workflow concluded with {run['conclusion']}")
                return
            time.sleep(args.wait_interval)
        raise TimeoutError(f"Timed out waiting for workflow {workflow_url}")
    finally:
        auth.close()


if __name__ == "__main__":
    main()
