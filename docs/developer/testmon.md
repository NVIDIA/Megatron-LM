# Selective unit tests in GitHub CI

The `Run selective unit tests` label opts an eligible synthetic PR push into
Testmon selection. The PR restores a compatible database recorded on `main`,
selects affected files, and runs their eligible tests through the normal pytest
and coverage path. PRs never record or publish Testmon databases.

## Produce the shared baseline

`.github/workflows/populate-build-cache.yml` builds the dev containers and then
records all unit-test buckets using the same source SHA and platform matrix.
It runs at 00:17, 06:17, 12:17, and 18:17 UTC, on relevant main-branch pushes,
and on manual dispatch. GitHub schedules can be delayed. Concurrent refreshes
are serialized without cancelling the active producer.

After merging the workflow into `main`, establish the first generation:

```bash
gh workflow run populate-build-cache.yml --repo NVIDIA/Megatron-LM --ref main
gh run list --repo NVIDIA/Megatron-LM --workflow populate-build-cache.yml --limit 5
```

Each enabled platform/bucket records the complete production and experimental
phases. Only successful, validated databases are saved. Publication is checked
with an exact cache lookup. A failed bucket leaves its older generation
available; the cache set is not an atomic snapshot of all buckets.

## Restore on labeled PRs

Producer and consumer use the same versioned compatibility prefix. It separates
platforms and buckets and includes the immutable image identity and the relevant
build, dependency, and test-execution configuration. General Python source/test
changes and the source commit SHA do not form part of that compatibility prefix.
The producer appends its run ID and attempt to create a new immutable cache key.

PRs restore the newest accessible generation matching the complete prefix.
There is no broader fallback across platforms or incompatible environments.
The new namespace excludes the earlier PR-scoped experimental databases.
GitHub's `cache-hit: false` can still mean a successful prefix restore; the
matched key and database validation determine whether selection is enabled.

The cache manifest records the producer SHA, creation time, compatibility
identity and phase metadata. SQLite integrity and recorded runtime identity are
checked before use. Selection operates on private copies and does not modify
the restored databases. An older compatible source baseline can select tests
for a newer commit; the comparison includes all changes since that baseline.

| Condition | Unit-test behavior |
| --- | --- |
| No selective label | Full bucket |
| Label and valid compatible baseline | Select affected files |
| Missing, incompatible or invalid baseline | Full bucket; no recording or save |
| Selection failure | Full bucket; no recording or save |
| Successful empty selection | No test execution for that phase |
| `Run tests`, `Run functional tests`, `force-run-all`, LTS, or merge group | Full unit-test path |

Apply the label before the next synthetic PR push, or rerun CI after applying
it. Adding the label alone does not trigger `cicd-main.yml`. The first shared
producer supports `main`; PRs without a matching supported target baseline run
fully.

## Validate the rollout

1. Confirm the producer publishes every enabled bucket under `refs/heads/main`.
2. Run two separate labeled PRs and verify both restore those main generations.
   Confirm neither PR records a baseline or saves a Testmon cache.
3. Refresh the producer and confirm subsequent PR runs restore the newer keys.
4. Exercise a missing/corrupt cache, a changed runtime, an unlabeled PR, and a
   full-test override. Each must take the full path without PR recording.
5. Compare selective and exhaustive results at the same commit and environment.
   Include a regression in code reached exclusively on a nonzero rank.

Keep exhaustive merge validation required during the pilot. Recording currently
observes rank 0; unioning selections across ranks does not establish dependency
coverage for code executed only by other ranks. Non-Python assets and external
services also require an explicit full-test decision when they affect tests.

Use the job summaries to inspect cache keys, baseline source/time, selection and
fallback reasons; use final pytest summaries for executed outcomes. Ordinary
pytest skips are not Testmon omissions. Include collection, generation and
retries when measuring elapsed-time or resource savings.

Investigate failed refreshes and baselines older than two refresh intervals.
To stop selection, remove the selective label or apply `Run tests` and rerun CI.
Preserve previous compatible generations during producer failures. A bad cache
protocol can be retired by changing its namespace; no PR database needs recovery.
