# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import importlib.util
from pathlib import Path

import pytest


def load_utils_module():
    module_path = Path(__file__).parents[2] / ".github" / "scripts" / "github_slack_utils.py"
    spec = importlib.util.spec_from_file_location("github_slack_utils", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


@pytest.fixture
def email_lookup(monkeypatch):
    def configure(profile, commits):
        module = load_utils_module()
        requests_seen = []

        class FakeRequests:
            @staticmethod
            def get(url, headers, timeout):
                requests_seen.append((url, headers, timeout))
                return FakeResponse(200, profile if "/users/" in url else commits)

        monkeypatch.setenv("GH_TOKEN", "token")
        monkeypatch.setattr(module, "requests", FakeRequests)
        return module, requests_seen

    return configure


def make_commit(
    message="", login="alice", name="Alice", email="12345+alice@users.noreply.github.com"
):
    return {
        "author": {"login": login} if login is not None else None,
        "commit": {"author": {"name": name, "email": email}, "message": message},
    }


def test_get_user_email_uses_signed_off_by_fallback(email_lookup):
    module, requests_seen = email_lookup(
        {"email": None}, [make_commit("Subject\n\nSigned-off-by: Alice <alice@nvidia.com>")]
    )

    assert module.get_user_email("alice") == "alice@nvidia.com"
    assert requests_seen[0][1]["Authorization"] == "Bearer token"
    assert requests_seen[0][1]["Accept"] == "application/vnd.github+json"
    assert requests_seen[0][1]["X-GitHub-Api-Version"] == "2022-11-28"
    assert requests_seen[0][2] == 30


def test_get_user_email_prefers_profile_and_caches_result(email_lookup):
    module, requests_seen = email_lookup(
        {"email": "alice@nvidia.com"}, [make_commit(email="old-alice@nvidia.com")]
    )

    assert module.get_user_email("alice") == "alice@nvidia.com"
    assert module.get_user_email("alice") == "alice@nvidia.com"
    assert len(requests_seen) == 1


def test_get_user_email_uses_primary_author_email_beyond_latest_three_commits(email_lookup):
    username = "shanmugamr1992"
    commits = [
        make_commit(
            "Signed-off-by: Shanmugam Ramasamy <old-address@nvidia.com>",
            login=username,
            name="Shanmugam Ramasamy",
            email="shanmugamr1992@gmail.com",
        )
        for _ in range(3)
    ]
    commits.append(make_commit(login=username, email="shanmugamr@nvidia.com"))
    module, requests_seen = email_lookup({"email": None, "name": None}, commits)

    assert module.get_user_email(username) == "shanmugamr@nvidia.com"
    assert "per_page=10" in requests_seen[1][0]


@pytest.mark.parametrize(
    "extra_commits,expected", [(1, "shanmugamr@nvidia.com"), (0, "shanmugamr1992@gmail.com")]
)
def test_get_user_email_resolves_squashed_pr_from_signoff_frequency(
    email_lookup, extra_commits, expected
):
    module, requests_seen = email_lookup(
        {"email": None, "name": None},
        [
            make_commit(
                "Signed-off-by: shanmugamr1992 <shanmugamr1992@gmail.com>\n"
                "Signed-off-by: William Dykas <wdykas@nvidia.com>\n"
                "Signed-off-by: Shanmugam Ramasamy <shanmugamr@nvidia.com>\n"
                "Co-authored-by: William Dykas <wdykas@nvidia.com>",
                login="shanmugamr1992",
                name="Shanmugam Ramasamy",
                email="shanmugamr1992@gmail.com",
            )
        ]
        + [
            make_commit(
                "Signed-off-by: Shanmugam <shanmugamr@nvidia.com>",
                login="shanmugamr1992",
                email="shanmugamr1992@gmail.com",
            )
            for _ in range(extra_commits)
        ],
    )

    assert module.get_user_email("shanmugamr1992") == expected
    assert module.get_user_email("shanmugamr1992") == expected
    assert len(requests_seen) == 2


def test_get_user_email_uses_most_common_signoff_across_varying_names(email_lookup):
    module, _ = email_lookup(
        {"name": "Alice Current"},
        [
            make_commit("Signed-off-by: Alice Current <latest@nvidia.com>"),
            make_commit("Signed-off-by: Alice Before <alice@nvidia.com>", name="Different Name"),
            make_commit("Signed-off-by: A. Example <alice@nvidia.com>", name=None),
        ],
    )

    assert module.get_user_email("alice") == "alice@nvidia.com"


@pytest.mark.parametrize(
    "preferred_count,expected", [(2, "alice@nvidia.com"), (1, "alice@users.noreply.github.com")]
)
def test_get_user_email_ranks_author_addresses_before_signoffs(
    email_lookup, preferred_count, expected
):
    module, _ = email_lookup(
        {},
        [make_commit(email="old@nvidia.com")]
        + [
            make_commit("Signed-off-by: Alice <alice@nvidia.com>", email="alice@nvidia.com")
            for _ in range(preferred_count)
        ],
    )

    assert module.get_user_email("alice") == expected


@pytest.mark.parametrize(
    "profile_name,author_name,signer",
    [
        (None, None, "Different Person"),
        ("Display One", "Display Two", "Third Person"),
        (None, "Earlier Name", "Later Name"),
    ],
)
def test_get_user_email_does_not_depend_on_display_names(
    email_lookup, profile_name, author_name, signer
):
    module, _ = email_lookup(
        {"name": profile_name},
        [
            make_commit(
                f"Signed-off-by: {signer} <alice@nvidia.com>", login="ALICE", name=author_name
            )
        ],
    )

    assert module.get_user_email("alice") == "alice@nvidia.com"


@pytest.mark.parametrize("public_email", [None, "alice@example.com"])
def test_get_user_email_ignores_coauthor_trailers(email_lookup, public_email):
    module, _ = email_lookup(
        {"email": public_email},
        [make_commit("Co-authored-by: Alice <alice@nvidia.com>") for _ in range(3)],
    )

    assert module.get_user_email("alice") == (public_email or "alice@users.noreply.github.com")


def test_get_user_email_counts_duplicate_trailers_once_and_leaves_ties_unresolved(email_lookup):
    module, _ = email_lookup(
        {"email": "alice@example.com"},
        [
            make_commit("Signed-off-by: Alice <first@nvidia.com>\n" * 3),
            make_commit("Signed-off-by: Alice <second@nvidia.com>"),
        ],
    )

    assert module.get_user_email("alice") == "alice@example.com"


@pytest.mark.parametrize("primary_author", ["bob", None])
def test_get_user_email_ignores_commits_without_matching_primary_author(
    email_lookup, primary_author
):
    module, _ = email_lookup(
        {},
        [
            make_commit(
                "Signed-off-by: Alice <wrong@nvidia.com>\n"
                "Co-authored-by: Alice <alice@nvidia.com>",
                login=primary_author,
                email="bob@nvidia.com",
            ),
            make_commit(email="alice@example.com"),
        ],
    )

    assert module.get_user_email("alice") == "alice@example.com"


def test_get_headers_requires_gh_token_without_github_token_fallback(monkeypatch):
    module = load_utils_module()

    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.setenv("GITHUB_TOKEN", "github-token")

    with pytest.raises(SystemExit):
        module.get_headers()


def test_get_headers_uses_requested_token_env(monkeypatch):
    module = load_utils_module()

    monkeypatch.setenv("ISSUE_COMMENT_TOKEN", "comment-token")

    headers = module.get_headers("ISSUE_COMMENT_TOKEN")

    assert headers["Authorization"] == "Bearer comment-token"


def test_get_slack_user_id_uses_lookup_by_email():
    module = load_utils_module()

    class FakeSlackClient:
        def users_lookupByEmail(self, email):
            assert email == "alice@nvidia.com"
            return {"user": {"id": "U123"}}

    assert module.get_slack_user_id(FakeSlackClient(), "alice@nvidia.com") == "U123"
