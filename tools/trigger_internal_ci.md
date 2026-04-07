# Trigger Internal CI

:warning: This is only useful to NVIDIANs.

Pushes the current branch to the internal GitLab remote and triggers a CI
pipeline — without touching the GitLab UI.

## Prerequisites

**1. Add the internal GitLab as a git remote** (skip if you already have one configured):

```bash
git remote add gitlab git@<gitlab-hostname>:ADLR/Megatron-LM.git
```

To check existing remotes: `git remote -v`

**The name of the origin will be required later!**

**2. Obtain a personal access token:**

1. Open your internal GitLab profile: **User menu → Edit profile → Access tokens**.
2. Click **Add new token**, give it a description, set an expiry, and select the **`api`** scope.
3. Click **Create personal access token** and copy the generated token (starts with `glpat-`).
4. Store it in your environment to avoid passing it on every invocation:

```bash
export GITLAB_TOKEN=glpat-<your-token>
```

**Tip: Store this in your .env or .bashrc file**

## Usage

```bash
python -m pip install python-gitlab
python tools/trigger_internal_ci.py \
  --gitlab-origin gitlab \
  [--access-token glpat-<your-token>] \
  [--functional-test-scope mr] \
  [--functional-test-repeat 5] \
  [--functional-test-cases all] \
  [--dry-run]
```

| Argument | Default | Description |
|---|---|---|
| `--gitlab-origin` | *(required)* | Git remote name for the internal GitLab |
| `--access-token` | `$GITLAB_TOKEN` | Personal access token with `api` scope |
| `--functional-test-scope` | `mr` | `FUNCTIONAL_TEST_SCOPE` pipeline variable |
| `--functional-test-repeat` | `5` | `FUNCTIONAL_TEST_REPEAT` pipeline variable |
| `--functional-test-cases` | `all` | `FUNCTIONAL_TEST_CASES` pipeline variable |
| `--dry-run` | off | Print what would happen without pushing or triggering |

## Example

```bash
# Dry run — no push, no trigger
python tools/trigger_internal_ci.py --gitlab-origin gitlab --dry-run

# Real run — uses token from environment
python tools/trigger_internal_ci.py --gitlab-origin gitlab
```

## Expected behavior

```
Current branch: my-feature-branch
Everything up-to-date
Triggering pipeline on https://<gitlab-hostname> project 19378 @ pull-request/my-feature-branch
Pipeline triggered: https://<gitlab-hostname>/<namespace>/<project>/-/pipelines/123456
```

1. The current branch is detected from git.
2. The branch is force-pushed to the GitLab remote as `pull-request/<branch>`.
3. A pipeline is triggered on that ref with the configured test variables.
4. The URL of the newly created pipeline is printed.
