# Review Threshold

A Lite PR is ready for review when it has:

- A narrow stated scope.
- Local import coverage.
- A validation command that reviewers can run.
- Clear non-goals for behavior deferred to later slices.

If a change needs GPU evidence, collect it separately and include job IDs in the
PR evidence. Do not treat skipped tests as passing evidence.
