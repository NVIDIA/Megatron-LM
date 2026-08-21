---
name: git-credentials-setup
description: Set up and load GitHub and GitLab credentials for Megatron-LM tooling. Loads GITHUB_TOKEN / GH_TOKEN / GITLAB_TOKEN from the environment, falling back to an env file, and prompts the user to create the env file the first time it is missing. Use whenever a GitHub or GitLab command fails with an authentication, authorization, 401, 403, or "token" error, or when the user mentions configuring git tokens.
when_to_use: A gh/gitlab/git/curl command failed with auth/401/403/permission/"bad credentials"/"token" errors; before running any GitHub or GitLab API call that needs auth; 'set up my git tokens', 'gitlab token', 'github token', 'why is gh unauthenticated'.
user_invocable: true
---

# GitHub & GitLab Credentials Setup

Loads `GITHUB_TOKEN`, `GH_TOKEN`, and `GITLAB_TOKEN` so GitHub (`gh`, GitHub
API) and GitLab (`tools/trigger_internal_ci.py`, GitLab API) tooling can
authenticate. Resolution order is: current environment → env file → prompt the
user to create the env file.

## When to use

Invoke this skill whenever a GitHub or GitLab operation fails with an auth-style
error, for example:

- `gh: ... HTTP 401: Bad credentials` / `gh auth status` shows logged out
- `HTTP 403` / `401 Unauthorized` from a GitHub or GitLab API call
- `fatal: Authentication failed` on `git push`/`git fetch`
- `tools/trigger_internal_ci.py` reporting a missing/invalid `GITLAB_TOKEN`

Run the loader first, then retry the failing command.

## Env file

Default path (override with `MEGATRON_ENV_FILE`):

```
~/.config/megatron-lm/credentials.env
```

It must define:

```bash
export GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxxxxxx   # GitHub personal access token
export GITLAB_TOKEN=glpat-xxxxxxxxxxxxxxxxxxxx     # Internal GitLab PAT (api scope)
```

`GH_TOKEN` is derived from `GITHUB_TOKEN` automatically for the `gh` CLI.

## Workflow

### 1. Load credentials

Source the loader so the tokens are exported into the current shell:

```bash
source skills/git-credentials-setup/scripts/load_git_credentials.sh
```

- If both tokens are already in the environment, it confirms and exits.
- If not, it sources the env file and exports the tokens.
- If the env file does not exist, it prints creation instructions and returns
  non-zero — go to step 2.

### 2. First-time setup (env file missing)

This is the only step that needs the user. **Do not invent or guess token
values.** Ask the user to provide them, then create the file:

1. Tell the user the env file is missing and ask them to supply a GitHub PAT
   and a GitLab PAT (or to paste them into the file themselves). Point them at
   the token pages:
   - GitHub: https://github.com/settings/tokens (scopes: `repo`, `read:org`)
   - GitLab: internal GitLab → **Edit profile → Access tokens** (`api` scope)
2. Create the file with `0600` permissions:

```bash
mkdir -p ~/.config/megatron-lm
cat > ~/.config/megatron-lm/credentials.env <<'ENV'
export GITHUB_TOKEN=PASTE_GITHUB_TOKEN_HERE
export GITLAB_TOKEN=PASTE_GITLAB_TOKEN_HERE
ENV
chmod 600 ~/.config/megatron-lm/credentials.env
```

3. Re-run step 1.

If the user prefers, persist the export lines in `~/.bashrc` / `~/.zshrc`
instead — the loader picks up anything already in the environment.

### 3. Validate

```bash
# GitHub
gh auth status

# GitLab (replace host; expects HTTP 200)
curl -fsS --header "PRIVATE-TOKEN: $GITLAB_TOKEN" \
  "https://<gitlab-hostname>/api/v4/user" >/dev/null && echo "GitLab OK"
```

A `401`/`403` here means the token is wrong, expired, or missing the required
scope — have the user regenerate it and update the env file.

## Notes

- Never print full token values in logs or commit them. The env file is
  git-ignored by living outside the repo (`~/.config`).
- The loader only reads from the environment and the env file; it never writes
  tokens anywhere.
