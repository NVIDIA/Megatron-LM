# Megatron Lite Skills

This directory defines agent-agnostic skills for maintaining Megatron Lite.
A skill is an operational contract for agents: what context to load, what
invariants to protect, how to make a change, and what evidence to leave behind.

These files are not Codex, Claude, or tool-specific skills. Do not add
agent-specific frontmatter, prompt wrappers, or runtime assumptions here.

## Function Model

Treat a skill as a function:

- `Schema` is the signature and summary.
- `imports` are skills that must be loaded before execution, like Python imports.
- `calls` are every skill this function explicitly calls in the body.
- the body is Python-like pseudocode with bounded exits.

An agent should be able to route work from the `Schema` alone. It should read
the body only when executing that skill.

## File Layout

Each skill file has three regions:

1. Before `MLITE_SKILL_SCHEMA_BEGIN`: a short human-facing title or note. Agents
   should not depend on this region for routing.
2. Between `MLITE_SKILL_SCHEMA_BEGIN` and `MLITE_SKILL_SCHEMA_END`: the compact
   schema used for retrieval, routing, imports, and call planning.
3. After `MLITE_SKILL_SCHEMA_END`: the skill body. This must be Python-like
   pseudocode, not free-form prose.

The pseudocode is a contract, not executable Python. It should still be precise:
clear inputs, explicit outputs, explicit skill calls, and finite exits.

Skills are organized like Python modules. A skill is a single `.md` file;
directories are namespaces, not skill bodies. Do not create `skill-name/README.md`
for individual skills.
Map schema names to paths by replacing `.` with `/` and `_` with `-`, then
adding `.md`: `model_compose.config_mapping` lives at
`model-compose/config-mapping.md`.

## Loading Model

Agents should load skills progressively:

1. Read this file.
2. Read exactly one leaf skill for the current work type.
3. Read linked Megatron Lite docs or source files only when the leaf skill asks
   for them.

Keep the active context small. A skill should point to durable source files and
validation commands instead of copying long design background.

## Defined Skills

`basic`:
- `basic.constitution`
- `basic.find_reference`
- `basic.align_precision`
- `basic.align_e2e_precision`
- `basic.lint_skill`
- `basic.construct_proxy_task`
- `basic.review_threshold`

`primitive`:
- `primitive.contract`
- `primitive.principle`
- `primitive.select_for_compose`
- `primitive.design`
- `primitive.validate`
- `primitive.fuse`
- `primitive.process_group`
- `primitive.parallel.tp`
- `primitive.parallel.ep`
- `primitive.parallel.pp`
- `primitive.parallel.cp`
- `primitive.optimizer.fsdp`
- `primitive.optimizer.distopt`
- `primitive.module.moe`
- `primitive.module.gqa`
- `primitive.module.thd`

Primitive work has two reusable meta layers: `primitive.principle` defines what
must be true, and `primitive.select_for_compose` decides when a primitive belongs
in a model composition.

`model_compose`:
- `model_compose.config_mapping`
- `model_compose.weight_mapping`
- `model_compose.build_model`
- `model_compose.qwen`

`application`:
- `application.runtime`
- `application.verl`
- `application.bench`

`perf`:
- `perf.measure`
- `perf.memory`
- `perf.fusion`
- `perf.optimize`

`insight`:
- `insight.progressive_model_support`
- `insight.progressive_primitive_design`
- `insight.scope_control`

## Skill Contract

Each skill file contains a delimited schema block near the top. Agents may read
only this block for routing, then read the body only when executing the skill.

````text
<!-- MLITE_SKILL_SCHEMA_BEGIN -->
```python
schema = Skill(
    "basic.align_precision", kind="state_machine", purpose="align precision",
    imports=["basic.constitution"], calls=["basic.constitution", "basic.find_reference"],
    inputs=["task", "files", "reference", "budget"],
    outputs=["patch", "evidence", "risks"], exits=["done", "blocked", "out_of_scope"],
)
```
<!-- MLITE_SKILL_SCHEMA_END -->
````

Use the markers instead of fixed line counts. Keep the schema short enough to
scan without reading the body; prefer compact lists and short field names over
prose.

After the schema, each skill body is Python-like pseudocode. The top-level
function should match the schema inputs and return one of the declared exits.
Use prose only inside short comments.

Required sections:

- one top-level function matching the schema name;
- explicit `return done(...)`, `return blocked(...)`, or
  `return out_of_scope(...)`;
- explicit calls to skills as `namespace.skill(...)`.

`imports` and `calls` are separate. `imports` says what an agent should preload.
`calls` says what the body invokes. If a loaded skill is directly invoked, list
it in `calls` too.

Every loop must have a progress measure, a maximum attempt count, or a blocking
condition. A skill that can spin forever is invalid.

## Skill Lint

Every new or changed skill should pass `basic.lint_skill` before review. Lint is
structural: it checks that schema can be extracted, imports and calls resolve,
the file path matches the schema name, body calls match declared calls, exits
are explicit, and loops are bounded. Scenario dry-runs are optional but
preferred for complex skills; they use mocked inputs to verify the pseudocode
can reach a declared exit.

## Boundaries

- Skills describe how agents work; `docs/` describes Megatron Lite for users and
  reviewers.
- Skills must not replace tests. They should name the right validation surface.
- Skills must not store task history, temporary job logs, or one-off debugging
  transcripts.
- Skills should stay stable across agents. If a rule depends on one agent tool,
  keep it outside this directory.
