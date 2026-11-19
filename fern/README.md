# Fern documentation CI

The Fern site publishes the content in `docs/` as **Nightly** and links to
previous developer guides through **Archives**. Publishing tags select a Git
revision; they do not create version snapshots or change the version selector.

Use Node.js 22 and the CLI version from `fern.config.json`. From the repository
root, install the CLI and run the same checks as CI:

```bash
npm install -g "fern-api@$(jq -er .version fern/fern.config.json)"
cd fern
fern check
fern docs md check
fern docs broken-links --strict
```

## Workflows

| Workflow | Trigger and behavior |
| --- | --- |
| `fern-docs-ci.yml` | Every PR, merge-queue groups, pushes to `main` and to `pull-request/*` and `deploy-release/*` branches, manual runs, and calls from production. Checks configuration, MDX, and broken links. |
| `fern-docs-preview-request.yml` | Docs changes pushed by the existing synchronization process to `pull-request/<number>`. Runs the project's pinned preflight without Fern credentials. |
| `fern-docs-preview.yml` | After a successful preview-request run on a synchronized PR branch. Validates the revision, waits for `fern-preview` approval, then updates a stable preview and one bot comment, including for fork PRs. |
| `fern-docs-staging.yml` | Docs changes on `main`, or manual runs on `main`. Updates the shared `megatron-lm-staging` preview and links it in the run summary. |
| `fern-docs-publish.yml` | `docs/v*` tag pushes, or manual runs on `main` or a tag. Runs validation on that revision before publishing production. |
| `fern-docs-version-bump.yml` | Final `core_vX.Y.Z` releases, or a manually supplied version. Opens a draft PR adding the release to `versions/archives.mdx`. Existing entries are left unchanged. |

Staging runs independently of validation. PR previews require their own validation
of the exact synchronized revision before requesting deployment approval. Each
preview ID is specific to Megatron-LM to avoid collisions with other repositories using the
same Fern site. Previews remain available after PRs close; automatic preview
cleanup is not configured. Fern preview URLs alone do not provide private access.

Validation receives no Fern token, including when called by production. This
keeps fork and trusted PR checks equivalent; authenticated missing-redirect
checks are not included. Preview and publish commands receive their token only
in the step that deploys the site.

## Preview approval flow

Authors keep their PR branches in forks. Once the existing copy-pr-bot process
synchronizes an eligible PR into `pull-request/<number>`, documentation changes
trigger the secretless preview-request workflow and its shared CI preflight.
Draft PRs are excluded, matching the repository's ready-for-review process.

A successful request triggers the preview workflow from the default branch.
It verifies the requesting workflow ID, the current synchronized branch SHA,
the open PR's head ancestry, and the PR's changed files through GitHub's API.
It does not consume requester artifacts or execute the PR's actions or scripts.
The preview helper uses Python's standard library and the runner's authenticated
GitHub CLI, including pagination; no additional Python packages are required.
The synchronized commit can include a merge with the PR base; both its SHA and
the original fork head SHA appear in the run summary and preview comment.

Validation and deployment use the workflow code and CLI pin from the trusted
workflow revision on `main`. The PR's `docs/` and `fern/` trees are checked out
separately at the immutable synchronized SHA. Symlinks in those trees are rejected,
and `fern.config.json` is replaced with the trusted CLI and organization settings.
Changes to the CLI pin therefore take effect in previews after merging to `main`.

After validation, a reviewer approves the `fern-preview` deployment in Actions.
The workflow rechecks the PR and synchronized branch after approval and only then
publishes. New commits supersede old requests and require a new approval; closed,
draft, or superseded requests are skipped. If the PR changes during publishing,
the workflow leaves its preview comment unchanged. Rerun the preview-request
workflow for the current synchronized commit to request another preview.

Both workflows and their helper must first be merged to `main` before this flow
can run. The existing test-queue approval is not used: it skips docs-only PRs.

## Repository configuration

- Set the `FERN_TOKEN` repository secret for previews, staging, and production in the NVIDIA
  Fern organization.
- Create the `fern-preview` environment with required reviewers, allowed refs
  restricted to `main`. Preview publishing uses the same `FERN_TOKEN` repository
  secret as staging and production. The environment name in YAML alone does not
  enforce approval.
- Configure the `fern-publish` environment with required reviewers and allowed
  refs matching `main` and `docs/v*` tags. The workflow references this environment;
  its approval and ref rules must be configured in GitHub settings. Manual tag
  runs can select tags outside the automatic `docs/v*` filter, so environment
  rules must enforce the intended tag policy.
- Require the `Fern docs (check)` workflow's `check` job through the repository's
  branch protection or ruleset if documentation validation should gate merging.
- For archive-update PRs, set `FERN_DOCS_BOT_FORK` to a personal fork in
  `owner/repository` format. Install the app identified by `BOT_ID` and `BOT_KEY`
  on both repositories, with contents read and pull requests write on upstream,
  and contents write on the fork. The workflow uses separate installation tokens,
  pushes only to that fork, and creates signed, signed-off draft PRs.

Merge an archive-link PR once the versioned guide is available at the archive
URL. For a pre-Fern release, select `publish-docs` when running the engineering
release workflow to build its Sphinx archive. After merging the archive link,
update the Fern site with a `docs/v*` tag or manual dispatch.
