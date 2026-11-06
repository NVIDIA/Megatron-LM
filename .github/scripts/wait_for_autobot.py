# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

"""Receive a request-bound Autobot result through GitHub, without an Autobot endpoint."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

CHECK_NAME = "Autobot CI selection"
REQUEST_JOB = "Autobot CI request (hello)"


def pages(get, path, collection, **query):
    """Read complete bounded collections, including earlier same-SHA attempts."""
    for page in range(1, 11):
        values = get(f"{path}?{urllib.parse.urlencode({**query, 'per_page': 100, 'page': page})}")[
            collection
        ]
        yield from values
        if len(values) < 100:
            return
    raise RuntimeError("GitHub result discovery exceeded its page limit")


def receive(
    get,
    repository,
    run_id,
    attempt,
    head_sha,
    app_id,
    *,
    timeout=600,
    now=time.monotonic,
    sleep=time.sleep,
):
    """Wait only for this run's request job and the authorized App's matching result."""
    deadline = now() + timeout
    request = None

    def bounded_get(path):
        if now() >= deadline:
            raise TimeoutError("Autobot result discovery deadline reached")
        return get(path)

    while now() < deadline:
        try:
            if request is None:
                jobs = [
                    job
                    for job in pages(
                        bounded_get,
                        f"/repos/{repository}/actions/runs/{run_id}/attempts/{attempt}/jobs",
                        "jobs",
                    )
                    if job.get("name") == REQUEST_JOB
                ]
                if len(jobs) > 1:
                    raise RuntimeError("Multiple Autobot request jobs in this attempt")
                if jobs:
                    job = jobs[0]
                    if (
                        job.get("run_id") != run_id
                        or job.get("run_attempt") != attempt
                        or job.get("head_sha") != head_sha
                        or job.get("status") != "completed"
                        or job.get("conclusion") != "success"
                        or type(job.get("id")) is not int
                        or job["id"] <= 0
                    ):
                        raise RuntimeError("Autobot request job identity/status mismatch")
                    branch = re.fullmatch(
                        r"pull-request/([1-9][0-9]*)", str(job.get("head_branch") or "")
                    )
                    if not branch:
                        raise RuntimeError("Autobot request is not on a PR mirror")
                    request = {
                        "repository": repository,
                        "run_id": run_id,
                        "run_attempt": attempt,
                        "job_id": job["id"],
                        "head_sha": head_sha,
                        "pull_request_number": int(branch[1]),
                        "request_id": f"autobot-ci:v1:{repository}:{run_id}:{attempt}:{job['id']}",
                    }
            if request:
                checks = [
                    check
                    for check in pages(
                        bounded_get,
                        f"/repos/{repository}/commits/{head_sha}/check-runs",
                        "check_runs",
                        check_name=CHECK_NAME,
                        filter="all",
                    )
                    if check.get("name") == CHECK_NAME
                    and check.get("head_sha") == head_sha
                    and check.get("app", {}).get("id") == app_id
                    and check.get("external_id") == request["request_id"]
                ]
                if len(checks) > 1:
                    raise RuntimeError("Multiple Autobot checks for this request")
                if checks and checks[0].get("status") == "completed":
                    check = checks[0]
                    if check.get("conclusion") != "success":
                        raise RuntimeError(f"Autobot result failed: {check.get('conclusion')}")
                    result = json.loads(check.get("output", {}).get("text") or "")
                    target = result.get("request", {})
                    if (
                        type(result.get("schema_version")) is not int
                        or result["schema_version"] != 1
                        or result.get("kind") != "hello"
                        or {key: target.get(key) for key in request} != request
                        or set(target) != {*request, "base_sha"}
                        or not re.fullmatch(r"[0-9a-f]{40}", str(target.get("base_sha") or ""))
                        or result.get("result") != {"message": "Hello from Autobot"}
                    ):
                        raise RuntimeError("Autobot result payload does not match this request")
                    return result
        except urllib.error.HTTPError as error:
            throttled = error.code == 403 and (
                error.headers.get("Retry-After")
                or error.headers.get("X-RateLimit-Remaining") == "0"
            )
            retry_after = error.headers.get("Retry-After", "10")
            error.close()
            if error.code != 429 and error.code < 500 and not throttled:
                raise
            delay = float(retry_after) if retry_after.isdigit() else 10
        except (urllib.error.URLError, TimeoutError):
            delay = 10
        else:
            delay = 10
        sleep(min(max(1, delay), max(0, deadline - now())))
    raise TimeoutError("Timed out waiting for this run's Autobot CI result")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repository = os.environ["GITHUB_REPOSITORY"]
    run_id, attempt = int(os.environ["GITHUB_RUN_ID"]), int(os.environ["GITHUB_RUN_ATTEMPT"])
    head_sha = os.environ["GITHUB_SHA"]
    app_id = int(os.environ["AUTOBOT_CI_APP_ID"])
    if (
        repository != "NVIDIA/Megatron-LM"
        or min(run_id, attempt, app_id) <= 0
        or not re.fullmatch(r"[0-9a-f]{40}", head_sha)
    ):
        raise ValueError("Invalid Autobot CI caller identity or publisher App ID")

    def get(path):
        request = urllib.request.Request(
            f"https://api.github.com{path}",
            headers={
                "Authorization": f"Bearer {os.environ['GH_TOKEN']}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        with urllib.request.urlopen(request, timeout=15) as response:
            return json.load(response)

    result = receive(get, repository, run_id, attempt, head_sha, app_id)
    text = json.dumps(result, sort_keys=True, separators=(",", ":"))
    args.output.write_text(text + "\n")
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as output:
            output.write(f"result={text}\n")
    print(result["result"]["message"])


if __name__ == "__main__":
    main()
