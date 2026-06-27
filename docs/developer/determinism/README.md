# Determinism (developer docs)

Developer reference for bit-exact deterministic training in Megatron-Core: the
current status, the per-op catalog, and the training-step branch map. This is the
foundation artifact for the determinism roadmap (validation, performance, tooling).

> User-facing "how do I turn it on" guide: `docs/user-guide/deterministic-training.md`.

## Contents

1. **[`status.md`](./status.md)** — start here. Definition of bitwise determinism,
   why it's hard, the perf targets (~15% → <10% → 5%), the control plane
   (`--deterministic-mode`, `determinism.py`, env vars), enforced limitations, the
   determinism branch surface, validation status, and known gaps.
2. **[`training-path.md`](./training-path.md)** — a forward→backward→optimizer walk
   that flags every point where determinism enters or is decided, with file:line
   refs and a 🟢/🔵/🟡/🔴 status for each.
3. **[`op-catalog.md`](./op-catalog.md)** — the per-operation catalog table
   (det? / det path / non-det path / how selected / evidence / perf Δ / gap), plus
   the perf hotspot priority list and the verification backlog.

## Maintenance

Keep the catalog **evidence-based**: classify each op via PyTorch/TE/NCCL docs, an
explicit code branch, or a `BitExactRunner` result — and fill perf deltas from the
nsys det-vs-nondet leaderboard, never by estimate. See "How this catalog is
maintained" in [`op-catalog.md`](./op-catalog.md).
