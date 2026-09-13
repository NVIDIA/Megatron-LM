# Lint Skill

Validate a Megatron Lite skill file before review.

## Schema

<!-- MLITE_SKILL_SCHEMA_BEGIN -->
```python
schema = Skill(
    "basic.lint_skill", kind="state_machine", purpose="validate skill structure and dry-runs",
    imports=[], calls=[],
    inputs=["skill_file", "registry", "scenarios", "budget"],
    outputs=["lint", "dry_runs", "risks"], exits=["done", "blocked", "out_of_scope"],
)
```
<!-- MLITE_SKILL_SCHEMA_END -->

```python
def lint_skill(skill_file, registry, scenarios, budget):
    if not skill_file.path.startswith("experimental/lite/skills/"):
        return out_of_scope("not an MLite skill")

    schema = extract_schema_block(skill_file)
    if schema is None:
        return blocked("missing schema markers")

    spec = parse_skill_schema(schema)
    expected_path = module_name_to_path(spec.name)
    if skill_file.path != expected_path:
        return blocked("schema name does not match file path", evidence=[spec.name, skill_file.path])

    lint = [
        require_single_md_file(skill_file),
        reject_readme_skill_body(skill_file),
        require_python_like_body(skill_file),
        require_top_level_function(skill_file, name=spec.name.split(".")[-1], inputs=spec.inputs),
        require_declared_exits(skill_file, exits=spec.exits),
        require_bounded_loops(skill_file),
        require_resolved_imports(spec.imports, registry),
        require_resolved_calls(spec.calls, registry),
    ]
    body_calls = extract_skill_calls(skill_file.body, registry=registry)
    lint.extend([
        require_body_calls_declared(body_calls, declared=spec.calls),
        require_declared_calls_used(spec.calls, body_calls, allow_recursive=True),
    ])
    if any(check.fail for check in lint):
        return blocked("skill lint failed", lint=lint)

    dry_runs = []
    for scenario in scenarios[:budget.max_scenarios]:
        dry_runs.append(trace_pseudocode(skill_file, scenario, max_steps=budget.max_steps))
        if dry_runs[-1].exit not in spec.exits:
            return blocked("scenario reached undeclared exit", dry_runs=dry_runs)

    risks = ["dry-runs are contract checks, not executable unit tests"]
    return done(lint=lint, dry_runs=dry_runs, risks=risks)
```
