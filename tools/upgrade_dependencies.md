# upgrade_dependencies.sh

Updates the `uv` lockfile, optionally upgrading all dependencies to their latest allowed versions.

## Prerequisites

- Docker
- `GITLAB_ENDPOINT` environment variable (see [Setup](#setup))

## Setup

The script pulls the `adlr/megatron-lm/mcore_ci_dev:main` image from the internal GitLab container registry for the ADLR/Megatron-LM project. Before running, export the registry hostname (no scheme, no trailing slash):

```bash
export GITLAB_ENDPOINT=<internal-gitlab-hostname>
```

Ask a team member or check your local environment / CI secrets for the correct value.

## Usage

```bash
# Update lockfile only (default)
./tools/upgrade_dependencies.sh

# Upgrade all dependencies and update lockfile
./tools/upgrade_dependencies.sh --upgrade
```

## Options

| Flag | Description |
|------|-------------|
| `--upgrade` | Upgrade dependencies in addition to updating the lockfile |
| `--help` | Show usage information |
