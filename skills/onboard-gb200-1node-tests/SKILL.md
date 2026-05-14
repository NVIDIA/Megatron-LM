---
name: onboard-gb200-1node-tests
description: Onboard 1-node GitHub MR functional tests for GB200 from existing mr-scoped 2-node tests.
when_to_use: Adding GB200 github-mr tests; creating single-node variants of existing tests; expanding CI coverage for GB200; 'add GB200 MR tests', 'onboard GB200 1-node', 'create single-node variant'.
user_invocable: true
argument: "[model-yaml]  # optional: gpt, moe, or both (default: both)"
---

# Onboard GB200 1-Node GitHub MR Tests

Create 1-node (`mr-github`) variants of existing 2-node (`mr`-scoped) GB200 functional tests.
Each GB200 node has **4 GPUs**. A 2-node test uses 8 GPUs total; the 1-node variant uses 4.

---

## Background

GB200 functional tests live in `tests/test_utils/recipes/gb200/`:

| Recipe file | Notes |
|-------------|-------|
| `gpt.yaml` | GPT dense tests, `nodes: 2, gpus: 4` (8 total) |
| `moe.yaml` | MoE tests, `nodes: 2, gpus: 4` (8 total) |
| `moe-1node.yaml` | Existing 1-node MoE tests, `nodes: 1, gpus: 4` (4 total) |
| `gpt-1node.yaml` | 1-node GPT tests (create if not present) |

Model configs live at:
`tests/functional_tests/test_cases/{model}/{test_case}/model_config.yaml`

1-node test cases use the `_1node` suffix:
`tests/functional_tests/test_cases/{model}/{test_case}_1node/model_config.yaml`

---

## Workflow

### Step 1 — Find candidate tests

Scan the `products:` block in `gpt.yaml` and `moe.yaml` for entries with `scope: [mr, ...]` or `scope: [mr-slim, ...]`. These are the 2-node tests that need 1-node `mr-github` counterparts.

Ignore tests already covered in `*-1node.yaml` files, and ignore `nightly`, `weekly`, `mr-broken` scopes.

### Step 2 — Read each model config

For each candidate, read its `model_config.yaml` and extract the key parallelism arguments:

```
--tensor-model-parallel-size   (TP)
--pipeline-model-parallel-size (PP)
--expert-model-parallel-size   (EP)
--expert-tensor-parallel-size  (ETP)
--context-parallel-size        (CP)
--global-batch-size
--micro-batch-size
```

### Step 3 — Classify: trivial copy vs. needs adaptation

The world size formula is: `world_size = TP × PP × DP` where `DP ≥ EP`.

Going from 8 GPUs → 4 GPUs:

| Condition | Action |
|-----------|--------|
| `TP × PP ≤ 4` | **Trivial copy.** Config unchanged; DP is halved automatically. |
| `TP × PP = 8` (e.g. tp4 pp2) | **Reduce PP.** Set `PP = PP / 2` (e.g. pp2→1). Verify `TP × PP_new ≤ 4`. |
| `EP > 4` (e.g. ep8 with tp1 pp1) | **Reduce EP.** Set `EP = 4`. Experts stay at `num-experts` (each EP rank holds more experts). |
| `EP > 4` **and** `TP × PP > 4` | Reduce both PP and EP as above. |
| ETP test (ep × etp ≤ TP × DP) | Check `EP × ETP ≤ TP × DP_new` after PP reduction. Usually satisfied when pp→1. |

**Do not change GBS** — let gradient accumulation absorb the reduced DP.

### Step 4 — Create `_1node` model config directories

```bash
# Trivial copy
mkdir -p tests/functional_tests/test_cases/{model}/{test_case}_1node
cp tests/functional_tests/test_cases/{model}/{test_case}/model_config.yaml \
   tests/functional_tests/test_cases/{model}/{test_case}_1node/model_config.yaml

# Then apply any parallelism changes (EP or PP) with Edit tool
```

### Step 5 — Create or update recipe files

**For GPT tests** — create `tests/test_utils/recipes/gb200/gpt-1node.yaml` (if absent) by cloning `gpt.yaml`'s spec block with `nodes: 1`. Use this template for the spec:

```yaml
type: basic
format_version: 1
maintainers: [mcore]
loggers: [stdout]
spec:
  name: "{test_case}_{environment}_{platforms}"
  model: gpt          # or moe
  build: mcore-pyt-{environment}
  nodes: 1
  gpus: 4
  n_repeat: 5
  platforms: dgx_gb200
  script_setup: |    # copy verbatim from gpt.yaml / moe.yaml
    ...
  script: |-         # copy verbatim from gpt.yaml / moe.yaml
    ...
```

**For MoE tests** — append entries to the existing `moe-1node.yaml`.

### Step 6 — Add products entries

Scope convention:
- **1–2 most representative tests** per recipe: `scope: [mr-github, mr-github-slim]`
- **All other tests**: `scope: [mr-github]`

```yaml
products:
  - test_case: [<test_case>_1node]
    products:
      - environment: [dev]
        scope: [mr-github, mr-github-slim]   # or [mr-github]
        platforms: [dgx_gb200]
```

---

## Quick parallelism reference

| Original (8 GPUs) | 1-node config (4 GPUs) | Notes |
|-------------------|----------------------|-------|
| tp1 pp1 ep1 → dp8 | tp1 pp1 ep1 → dp4 | trivial |
| tp2 pp1 ep1 → dp4 | tp2 pp1 ep1 → dp2 | trivial |
| tp1 pp2 ep1 → dp4 | tp1 pp2 ep1 → dp2 | trivial |
| tp4 pp1 ep1 → dp2 | tp4 pp1 ep1 → dp1 | trivial |
| tp1 pp4 ep1 → dp2 | tp1 pp4 ep1 → dp1 | trivial |
| tp1 pp1 ep8 → dp8 | tp1 pp1 ep4 → dp4 | ep 8→4 |
| tp4 pp2 ep2 etp2 → dp1 | tp4 pp1 ep2 etp2 → dp1 | pp 2→1 |

---

## Checklist

- [ ] Identified all `mr`-scoped tests in `gpt.yaml` and `moe.yaml` not yet in `*-1node.yaml`
- [ ] Read model config for each candidate
- [ ] Classified trivial vs. adaptation needed
- [ ] Created `_1node/model_config.yaml` for each test
- [ ] Applied EP or PP reductions where needed
- [ ] Created/updated recipe YAML with `nodes: 1, gpus: 4`
- [ ] Assigned `mr-github` scope (+ `mr-github-slim` for 1–2 representative tests per recipe)
- [ ] Verified no `mr-github-slim` overload (slim suite should stay small)
