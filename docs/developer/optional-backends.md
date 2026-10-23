# Optional GPU backends

Megatron Core publishes optional backend requirements as package extras. The
`inference` and `nvshmem` extras target Linux with CUDA 13. Install them inside a
compatible container; the extras do not select a GPU, install a driver, or
configure distributed networking.

| Extra | Purpose | Runtime requirements |
| --- | --- | --- |
| `inference` | Inference memory management, FlashAttention 2 and 4, and TileLang | FlashAttention 4 requires a supported Hopper or Blackwell GPU. |
| `nvshmem` | NVSHMEM model resharding, including its Python and CuPy dependencies | CUDA 13, a compatible NVIDIA driver, and NVSHMEM 3.6.5 or later within the declared version range. |

For example, from a checkout inside a CUDA 13 development container:

```bash
uv pip install --no-build-isolation -e ".[inference,nvshmem]"
```

Install the build requirements and the container-compatible PyTorch distribution
before building native extensions. The `te` and `ssm` extras remain separate;
request them when the model needs Transformer Engine or state-space layers.
Installing `nvshmem` does not change the selected model-resharding backend.

For a package installation, resolve FlashAttention 2 and 4 together using the
declared version ranges: both distributions use the `flash_attn` namespace.
The dev NGC image retains its prebuilt FlashAttention 2 and installs FlashAttention
4 with its CUDA 13 dependencies. That container installation also depends on the
pinned base image; it is not reproduced by installing arbitrary FA2 and FA4 wheels.

## Downstream dependency resolution

The extras in `project.optional-dependencies` become the wheel's `Requires-Dist`
metadata. A downstream library such as Megatron Bridge can request
`megatron-core[inference,nvshmem]` to propagate these requirements to its users.
Bridge must explicitly request the extras it supports; updating Megatron Core
alone does not enable an unrequested extra.

Version ranges describe package compatibility. Each downstream application still
owns its resolved environment and lockfile. Megatron Core's `uv.lock`, development
source mappings, and root uv overrides are not inherited as wheel requirements.
Downstream overrides can replace inherited requirements, so remove or reconcile
conflicting pins before refreshing the application's lockfile. See uv's
[dependency sources](https://docs.astral.sh/uv/concepts/projects/dependencies/#dependency-sources)
and [override semantics](https://docs.astral.sh/uv/reference/settings/#override-dependencies).

## CI coverage

The dev image installs these extras. Its H100 resharding bucket sets
`MCORE_REQUIRE_NVSHMEM=1`, making missing NVSHMEM dependencies a test failure.
The GB200 inference bucket sets `MCORE_REQUIRE_FA4=1` and selects the existing
one-token prefill test's FA4 parameter, which runs paged attention and compares
the result with a reference implementation. Local runs without these flags retain
optional-backend skips. Testmon selection still applies to these buckets, with
both backend test files required when `pyproject.toml`, `uv.lock`, the dev
Dockerfile, or its optional-backend preparation helper changes.
