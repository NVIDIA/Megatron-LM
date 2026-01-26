# Generating Docs Locally

To generate docs locally, use the following commands:

```
cd docs
uv run --only-group docs sphinx-autobuild . _build/html --port 8080 --host 127.0.0.1
```

Docs will be generated at <http://localhost:8080/>.

**Recommended:** set the environment variable `SKIP_AUTODOC=true` when generating docs 
to skip the generation of `apidocs`.