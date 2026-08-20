#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Cancel active workflow runs for superseded generations of one queued PR."""

import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request

_ACTIVE_STATUSES = ("requested", "queued", "in_progress", "waiting", "pending")
_MERGE_GROUP_REF = re.compile(r"^gh-readonly-queue/.+/pr-(\d+)-[0-9a-f]+$", re.IGNORECASE)


class GitHubApiError(RuntimeError):
    """Report a failed GitHub REST API request."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


class GitHubClient:
    """Minimal GitHub Actions REST API client."""

    def __init__(self, api_url: str, repository: str, token: str) -> None:
        owner, name = repository.split("/", maxsplit=1)
        self._repository_path = f"repos/{urllib.parse.quote(owner)}/{urllib.parse.quote(name)}"
        self._api_url = api_url.rstrip("/")
        self._headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    def _request(self, path: str, method: str = "GET") -> dict[str, object] | None:
        for attempt in range(5):
            request = urllib.request.Request(
                f"{self._api_url}/{self._repository_path}/{path}",
                data=b"" if method == "POST" else None,
                headers=self._headers,
                method=method,
            )
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    payload = response.read()
                return json.loads(payload) if payload else None
            except urllib.error.HTTPError as error:
                details = error.read().decode("utf-8", errors="replace")
                if error.code != 429 and error.code < 500:
                    raise GitHubApiError(
                        error.code, f"GitHub API {method} {path} failed: {details}"
                    ) from error
                if attempt == 4:
                    raise GitHubApiError(
                        error.code, f"GitHub API {method} {path} failed after retries: {details}"
                    ) from error
                delay = min(int(error.headers.get("Retry-After", 2**attempt)), 30)
            except (TimeoutError, urllib.error.URLError) as error:
                if attempt == 4:
                    raise RuntimeError(f"GitHub API {method} {path} failed after retries: {error}") from error
                delay = 2**attempt

            print(f"GitHub API {method} {path} failed transiently; retrying in {delay}s")
            time.sleep(delay)

        raise AssertionError("unreachable")

    def _list_runs(self, endpoint: str, parameters: dict[str, str]) -> list[dict[str, object]]:
        runs: list[dict[str, object]] = []
        page = 1
        while True:
            query = urllib.parse.urlencode({**parameters, "per_page": "100", "page": str(page)})
            response = self._request(f"{endpoint}?{query}")
            if not isinstance(response, dict) or not isinstance(response.get("workflow_runs"), list):
                raise RuntimeError(f"GitHub API returned an invalid workflow-runs response for {endpoint}")

            page_runs = response["workflow_runs"]
            runs.extend(page_runs)
            if len(page_runs) < 100:
                return runs
            page += 1

    def list_trigger_workflow_runs(
        self, workflow_id: int, head_branches: set[str]
    ) -> list[dict[str, object]]:
        """List merge-group runs that establish the ordering of active generations."""

        runs: list[dict[str, object]] = []
        for head_branch in sorted(head_branches):
            runs.extend(
                self._list_runs(
                    f"actions/workflows/{workflow_id}/runs",
                    {"branch": head_branch, "event": "merge_group"},
                )
            )
        return runs

    def list_active_merge_group_runs(self) -> list[dict[str, object]]:
        """List all non-terminal merge-group workflow runs in the repository."""

        runs_by_id: dict[int, dict[str, object]] = {}
        for status in _ACTIVE_STATUSES:
            for run in self._list_runs("actions/runs", {"event": "merge_group", "status": status}):
                runs_by_id[int(run["id"])] = run
        return list(runs_by_id.values())

    def cancel_run(self, run_id: int) -> None:
        """Request cancellation of one workflow run."""

        self._request(f"actions/runs/{run_id}/cancel", method="POST")


def _pr_number(head_branch: object) -> int | None:
    match = _MERGE_GROUP_REF.fullmatch(str(head_branch))
    return int(match.group(1)) if match else None


def _run_order(run: dict[str, object]) -> tuple[str, int]:
    return str(run["created_at"]), int(run["id"])


def _select_superseded_runs(
    active_runs: list[dict[str, object]],
    trigger_workflow_runs: list[dict[str, object]],
    current_run: dict[str, object],
) -> list[dict[str, object]]:
    current_head = str(current_run["head_branch"])
    current_pr = _pr_number(current_head)
    if current_pr is None:
        raise ValueError(f"Unexpected merge-group head branch: {current_head}")

    generation_order: dict[str, tuple[str, int]] = {}
    for run in trigger_workflow_runs:
        head_branch = str(run.get("head_branch", ""))
        if _pr_number(head_branch) != current_pr:
            continue
        run_order = _run_order(run)
        previous_order = generation_order.get(head_branch)
        if previous_order is None or run_order < previous_order:
            generation_order[head_branch] = run_order

    current_order = _run_order(current_run)
    generation_order[current_head] = min(generation_order.get(current_head, current_order), current_order)
    if any(head != current_head and order > current_order for head, order in generation_order.items()):
        print(f"A newer merge-group generation exists for PR #{current_pr}; this cleanup is stale")
        return []

    superseded = []
    for run in active_runs:
        head_branch = str(run.get("head_branch", ""))
        if head_branch == current_head or _pr_number(head_branch) != current_pr:
            continue

        old_generation_order = generation_order.get(head_branch)
        if old_generation_order is None:
            print(f"Skipping run {run.get('id')} because its merge-group generation was not found")
            continue
        if old_generation_order < current_order:
            superseded.append(run)

    return sorted(superseded, key=_run_order)


def main() -> None:
    """Cancel active runs belonging to older merge-group generations of the triggering PR."""

    current_run = {
        "id": int(os.environ["TRIGGER_RUN_ID"]),
        "head_branch": os.environ["TRIGGER_HEAD_BRANCH"],
        "created_at": os.environ["TRIGGER_CREATED_AT"],
    }
    client = GitHubClient(
        api_url=os.environ.get("GITHUB_API_URL", "https://api.github.com"),
        repository=os.environ["GITHUB_REPOSITORY"],
        token=os.environ["GITHUB_TOKEN"],
    )

    active_runs = client.list_active_merge_group_runs()
    current_pr = _pr_number(current_run["head_branch"])
    if current_pr is None:
        raise ValueError(f"Unexpected merge-group head branch: {current_run['head_branch']}")
    active_heads = {
        str(run.get("head_branch", "")) for run in active_runs if _pr_number(run.get("head_branch")) == current_pr
    }
    active_heads.add(str(current_run["head_branch"]))
    trigger_runs = client.list_trigger_workflow_runs(int(os.environ["TRIGGER_WORKFLOW_ID"]), active_heads)
    superseded_runs = _select_superseded_runs(active_runs, trigger_runs, current_run)

    canceled = 0
    for run in superseded_runs:
        run_id = int(run["id"])
        try:
            client.cancel_run(run_id)
        except GitHubApiError as error:
            if error.status == 409:
                print(f"Run {run_id} became terminal before cancellation")
                continue
            raise
        print(f"Canceled superseded {run.get('name', 'workflow')} run {run_id} ({run.get('head_branch')})")
        canceled += 1

    print(f"Canceled {canceled} superseded merge-group run(s)")


if __name__ == "__main__":
    main()
