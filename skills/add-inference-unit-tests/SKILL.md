---
name: add-inference-unit-tests
description: Author NEW unit tests for Megatron-LM inference code (megatron/core/inference, tests/unit_tests/inference) — decide whether a test is even worth writing, where it belongs, what to test vs. what reviewers reject, then run it. Pairs with run-inference-unit-tests, which only knows how to execute an already-written suite. - Adding a unit test for new or changed inference code; deciding whether a new test file is justified; 'add a unit test', 'write tests for this inference change', 'cover this code path', 'should I test this'.
when_to_use: Writing new unit tests for inference code; covering a new code path before pushing; deciding whether a test or a whole test file is even worth adding; 'add inference unit test', 'write tests for', 'test coverage for inference', 'should I test this'.
---

# Add new Megatron-LM inference unit tests

This skill is about **authoring** unit tests for inference code under
`megatron/core/inference/` (tests live in `tests/unit_tests/inference/`).
The runner skill [[run-inference-unit-tests]] knows how to *execute* an
existing suite on the cluster; this skill decides *what to write in the
first place* and where it goes.

The hard part of a good unit test is not the mechanics — it's **judgment
about what deserves a test at all**. A reviewer will reject a
mechanically-correct test that locks down a language guarantee, a
logging string, or a one-line list wrapper. Most of this skill is that
judgment.

> ## 🌱 This is an evolving skill — keep it that way.
>
> Every code review teaches a new "never test this" or "this is the
> load-bearing method you missed." **When a reviewer rejects a test you
> wrote, or asks for a test you didn't think to write — after you fix
> it — add the lesson here** (a row in the anti-patterns table, or a
> note in the relevant step). The value of this skill is the
> accumulated review feedback; a skill that doesn't grow re-earns the
> same review comments every PR.
>
> Concretely:
> - Reviewer says "never test X" → add it to **§3 Anti-patterns**.
> - Reviewer says "you should have tested the big method Y, not the
>   small blocks" → reinforce **§2 Decide what to test**.
> - You discover a fixture / marker / distributed-init gotcha → add it
>   to **§5 Pitfalls**.
> - The skill turned out to be wrong about a step → fix the prose,
>   don't work around it.

---

## Workflow at a glance

1. **§1 — Read the code under test.** No test before you understand the
   public API and what actually breaks if it regresses.
2. **§2 — Decide what to test (and whether a new file is justified).**
   The HARD STOPS gate. Most rejected tests die here.
3. **§3 — Avoid the anti-patterns.** The concrete "never write this"
   list from past reviews.
4. **§4 — Write the test.** Layout, fixtures, markers, distributed init.
5. **§5 — Run it.** Defers entirely to [[run-inference-unit-tests]].
6. **§6 — Lint & finalize.**

---

## §1 — Read the code under test first

Before writing anything:

- Read the source file. Identify the **public, load-bearing method** —
  the one other code actually calls, the one that combines the small
  helpers into something useful.
- Note what would genuinely break a caller if it regressed. That is your
  test target. Private helpers and getters that do nothing on their own
  are *not* targets — they're exercised transitively by the public method.
- **Find every sister test file before writing a new one.** A test that
  duplicates an existing one is the most common rejection. Run:

  ```bash
  grep -rl "<ClassUnderTest>\|<method_under_test>" tests/unit_tests/
  ```

  Read every hit. If an existing file already exercises the behavior
  (especially a `*_lifecycle` test), judge whether yours adds anything.
  Reviewers cite this explicitly: *"`test_data_parallel_inference_coordinator.py`
  has `test_control_logic_lifecycle` — judge whether your test offers
  anything on top of that pre-existing test."*

---

## §2 — Decide what to test — HARD STOPS

Before writing a new test file, answer the questions a senior reviewer
will ask. **If the answer to any of these is "no," do NOT write a new
test file.** Instead: skip the source file, add a case to an existing
test file, or invest in a higher-level functional test that exercises
the code naturally.

- **Is there a public, load-bearing method here that a caller depends
  on?** If the only public entry point can't be unit-tested (needs a
  CUDA-extension compile, a real `ProcessGroup` rendezvous, a live
  server), **don't write a separate test file at all.** The small
  internal blocks are already tested by anyone who calls the real public
  method. Adding tests for the small blocks while leaving the big method
  untested is backwards.

  > Reviewer on `unified_memory.py`: *"The most important method is
  > `create_unified_mempool`. And that is precisely the method that is
  > never tested. This is backwards: we should not be adding extra tests
  > for small individual blocks of code that do nothing on their own,
  > but fail to test the big block of code that combines them."*

- **Does an existing test already cover this?** (See §1 grep.) If a
  `*_lifecycle` test already drives this path, a new "unit" test is
  usually redundant.

- **Would this test fail only if someone changes the code's *intent*,
  not if they break it?** A test that breaks on a harmless rename, a log
  reformat, or a comment change is noise. Skip it.

- **Is the behavior a language/library guarantee rather than your
  logic?** Enum uniqueness, subclass relationships, `list.__getitem__` —
  Python already guarantees these. Skip it (see §3).

**Prefer ONE lifecycle test over N micro-tests.** For complex async
services (e.g. `InferenceClient`, coordinators), a single lifecycle test
that drives connect → use → stop catches ordering bugs that isolated
`test_connect_*`, `test_stop_*`, `test_recv_*` unit tests never will.
Reviewers actively ask for the lifecycle test *instead of* the unit
tests.

---

## §3 — Anti-patterns: never write these

Concrete patterns reviewers have rejected. Each one tests Python, a
logging string, or an unintended behavior — not your code.

| # | Don't write | Why |
|---|---|---|
| 1 | **Enum distinctness / member existence.** `len({s.value for s in MyEnum}) == len(list(MyEnum))`, `test_engine_state_members_distinct`. | The `Enum` metaclass enforces uniqueness — the test just restates the source. Reviewer: *"should specifically be documented … as an example of what to never do under any circumstances whatsoever."* |
| 2 | **Exception class hierarchy.** `assert issubclass(MyError, Exception)`; `with pytest.raises(MyError): raise MyError("x")`. | Tests Python's class system, not your code. |
| 3 | **Each dataclass `__post_init__` assertion individually.** | Collapse to ONE parameterized test, or leave it to an integration test that exercises the dataclass naturally. |
| 4 | **`__str__` / `__repr__` formatting when it's only used for logging.** | Format changes break nothing; the test just blocks harmless edits. Don't test it AT ALL in this case. |
| 5 | **deserialize-accepts-unknown-keys.** | This is unintended permissiveness, not a feature. Reviewer: *"This test is pointless and harmful. This functionality is entirely unintended and should not be tested."* |
| 6 | **`from_request`-style constructors that wrap a 1-line list init.** `record = Record.from_request(req); assert record.requests == [req]`. | Only tests `list.__init__`. |
| 7 | **Indexing that's `dataclass.__getitem__ = list.__getitem__`.** `record[0] is req1`. | Only tests `list[0]`. |
| 8 | **A new test file before checking sister files.** | See §1 — most-cited rejection. `grep -rl "<class>\|<method>" tests/unit_tests/` first. |
| 9 | **Extra tests for small blocks while the load-bearing method stays untested.** | Backwards. Test the big method that combines them, or nothing. See §2. |
| 10 | **Mocking enums.** `MagicMock()` for `EngineState.RUNNING`. | Enums are constants — just import them. A mocked enum's `.is_set()` returns another MagicMock, so the test passes without testing anything. |
| 11 | **N isolated `test_connect_*` / `test_stop_*` / `test_recv_*` for an async service.** | One lifecycle test beats them — it catches ordering bugs the unit tests miss. See §2. |
| 12 | **`add_attributes(**kwargs)`-style dynamic field setters.** | They're escape hatches the public API discourages; testing them locks the escape hatch open. |

When in doubt, ask: *"If a teammate refactored the implementation but
kept the public behavior, would this test still pass?"* If no, you're
testing the implementation, not the contract — reconsider.

---

## §4 — Write the test

Layout and conventions (from the [[testing]] skill):

1. Create `tests/unit_tests/inference/<category>/test_<name>.py`, mirroring
   the source layout under `megatron/core/inference/`.
2. Reuse fixtures from `tests/unit_tests/conftest.py` and any
   `tests/unit_tests/inference/conftest.py` — don't hand-roll distributed
   setup.
3. Inference unit tests initialize a `torch.distributed` group, so they
   need GPU access and run under `torch.distributed.run` (the runner
   skill handles this). Use the existing model-parallel init helpers
   (`Utils.initialize_model_parallel` / `Utils.destroy_model_parallel`)
   rather than re-implementing rendezvous.
4. Apply markers as needed:
   - `@pytest.mark.internal` — skipped on the `legacy` tag
   - `@pytest.mark.flaky_in_dev` — skipped in `dev` (disable a flaky test
     without blocking the standard pipeline)
   - `@pytest.mark.flaky` — skipped in `lts`
   - `@pytest.mark.experimental` — `latest` tag only
5. Parameterize related cases with `@pytest.mark.parametrize` instead of
   copy-pasting near-identical test functions (esp. for the dataclass
   assertion case in §3 #3).
6. If the test needs its own CI bucket, add an entry to
   `tests/test_utils/recipes/h100/unit-tests.yaml`. Most inference tests
   ride the existing inference bucket — only add a bucket if the suite
   genuinely needs isolation.

---

## §5 — Run the test

**Do not re-derive how to run it — use [[run-inference-unit-tests]].**
That skill owns the cog submit recipe (8 GPUs, 1 node, single torchrun
launcher with `--nproc-per-node 8`), the marker filters, the
`--ignore=…/test_dynamic_engine.py` workaround, and result parsing.

Iterating on a single new test? Don't `cog submit` in a loop — start one
`cog session` and `cog session exec` the pytest command per iteration.
See the "Iterating" callout in [[run-inference-unit-tests]] and
[[cog-setup-and-help]].

A quick single-file invocation (the runner skill wraps this for cluster
submission):

```bash
python -m torch.distributed.run --nproc-per-node 8 -m pytest \
  -v -o addopts= \
  tests/unit_tests/inference/<category>/test_<name>.py
```

---

## §6 — Lint & finalize

- After editing imports, run `uv run isort` on the new/changed files
  (mandatory per repo CLAUDE.md) before committing.
- See [[linting-and-formatting]] for the full autoformat flow.

---

## §5b — Pitfalls (grow this section)

Each row: symptom → cause → fix. Add a row whenever you hit a fresh
authoring/run gotcha that wasn't covered above.

| Symptom | Cause | Fix |
|---|---|---|
| Whole suite SIGABRTs mid-run with `delete_cuda_graphs` in the dump | `test_dynamic_engine.py`'s teardown SIGABRTs the rank; torchrun then kills peers | The runner skill already passes `--ignore=…/test_dynamic_engine.py`. Don't model new teardown on that file's cuda-graph cleanup. |
| `pytest-cov` dies at `_has_torch_function already has a docstring` | torch C-extension init collides with coverage on the current image | Coverage is opt-in and currently broken — see [[run-inference-unit-tests]] §2b. Don't add `--cov` to your local loop. |

---

## Cross-references

- [[run-inference-unit-tests]] — how to *run* the suite on the cluster.
  Owns the cog submit recipe and result parsing. This skill defers there
  for everything execution-related.
- [[testing]] — full test layout, recipe YAML schema, markers, fixtures,
  CI parity, golden values (functional only).
- [[cog-setup-and-help]] — cog install, cluster registration, session
  vs. submit. Canonical in-tree cog reference.
- [[linting-and-formatting]] — isort/ruff/black flow to run before
  committing.
- [[add-inference-performance-test]] — sibling "authoring" skill for the
  *performance* suite (baselines, not goldens).
