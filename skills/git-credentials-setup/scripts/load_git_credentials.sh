#!/usr/bin/env bash
# Load GitHub/GitLab tokens for Megatron-LM tooling.
#
# Resolution order:
#   1. Tokens already present in the environment.
#   2. Tokens defined in the env file ($MEGATRON_ENV_FILE or the default below).
#   3. Otherwise: print instructions to create the env file and fail.
#
# Usage:
#   source skills/git-credentials-setup/scripts/load_git_credentials.sh
#     -> exports GITHUB_TOKEN, GH_TOKEN, GITLAB_TOKEN into the current shell.
#
# Exit/return code: 0 when both tokens are available, 1 otherwise.

ENV_FILE="${MEGATRON_ENV_FILE:-$HOME/.config/megatron-lm/credentials.env}"

# Return on `source`, exit on direct execution.
_gc_fail() {
    return 1 2>/dev/null || exit 1
}

# Load the env file if either token is missing from the environment.
if [ -z "${GITHUB_TOKEN:-}" ] || [ -z "${GITLAB_TOKEN:-}" ]; then
    if [ -f "$ENV_FILE" ]; then
        set -a
        # shellcheck disable=SC1090
        . "$ENV_FILE"
        set +a
    else
        cat >&2 <<EOF
[git-credentials] Env file not found: $ENV_FILE

First-time setup: ask the user for a GitHub PAT and a GitLab PAT, then create
the file (do NOT guess token values):

  mkdir -p "$(dirname "$ENV_FILE")"
  cat > "$ENV_FILE" <<'ENV'
  export GITHUB_TOKEN=PASTE_GITHUB_TOKEN_HERE
  export GITLAB_TOKEN=PASTE_GITLAB_TOKEN_HERE
  ENV
  chmod 600 "$ENV_FILE"

Then re-run this loader.
EOF
        _gc_fail
    fi
fi

# gh CLI reads GH_TOKEN; derive it from GITHUB_TOKEN when unset.
if [ -n "${GITHUB_TOKEN:-}" ]; then
    export GH_TOKEN="${GH_TOKEN:-$GITHUB_TOKEN}"
fi
export GITHUB_TOKEN GITLAB_TOKEN

missing=""
[ -z "${GITHUB_TOKEN:-}" ] && missing="$missing GITHUB_TOKEN"
[ -z "${GITLAB_TOKEN:-}" ] && missing="$missing GITLAB_TOKEN"

if [ -n "$missing" ]; then
    echo "[git-credentials] Missing:$missing (add them to $ENV_FILE)" >&2
    _gc_fail
fi

echo "[git-credentials] GITHUB_TOKEN, GH_TOKEN, and GITLAB_TOKEN are loaded."
