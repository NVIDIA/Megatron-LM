# Repository CI source-build assets

This directory is materialized from the canonical Megatron-LM v2.5 release assets so the isolated
repository CI builder can evaluate `COPY assets/ /opt/data/` in `docker/Dockerfile.ci.dev`. The builder
clones tracked Git content only and cannot run the GitHub Actions asset-preparation step before the
Docker build.

Generated with:

```bash
uv run --no-project --with click --with requests \
  python tests/test_utils/python_scripts/download_unit_tests_dataset.py --assets-dir ./assets
```

Release inputs:

- `datasets.zip`: `sha256:2dda736d6daa6ed32c5f866aec7e7915c380e353cd39b05ba2dcb13039225661`
- `tokenizers.zip`: `sha256:e58ed690b48958e31e3b1453a9bccb592cf4bb7a3b3ac3938da13dd52d694f9c`

Source: <https://github.com/NVIDIA/Megatron-LM/releases/tag/v2.5>
