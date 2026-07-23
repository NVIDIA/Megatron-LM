# Skill Lint

Validate an Agent Compose skill before review.

<!-- MLITE_SKILL_SCHEMA_BEGIN -->
```python
schema = Skill(
    "basic.lint_skill", kind="state_machine", purpose="validate skill structure",
    imports=[], calls=[],
    inputs=["skill_file", "registry", "budget"],
    outputs=["lint", "risks"], exits=["done", "blocked", "out_of_scope"],
)
```
<!-- MLITE_SKILL_SCHEMA_END -->

```python
def lint_skill(skill_file, registry, budget):
    root = "experimental/agent_compose/skills/"
    if not skill_file.path.startswith(root):
        return out_of_scope("not an Agent Compose skill")

    schema = extract_schema_block(skill_file)
    if schema is None:
        return blocked("missing schema markers")
    spec = parse_skill_schema(schema)
    expected_path = root + module_name_to_path(spec.name)
    if skill_file.path != expected_path:
        return blocked("schema name does not match file path")

    checks = [
        require_python_like_body(skill_file),
        require_top_level_function(skill_file, spec.name.split(".")[-1], spec.inputs),
        require_declared_exits(skill_file, spec.exits),
        require_bounded_loops(skill_file, max_steps=budget.max_steps),
        require_resolved_imports(spec.imports, registry),
        require_resolved_calls(spec.calls, registry),
        require_body_calls_declared(skill_file.body, spec.calls),
    ]
    if any(check.fail for check in checks):
        return blocked("skill lint failed", lint=checks)
    return done(lint=checks, risks=["structural lint does not replace executable tests"])
```
