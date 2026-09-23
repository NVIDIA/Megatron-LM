# Native dependency wheel cache

The development image's `uv sync` already builds packages concurrently. Its
BuildKit cache mount is local to a builder, so the CI workflow also restores and
publishes that mount as a separate ECR image. The final test image does
not contain this snapshot.

`docker/common/uv_cache_key.py` partitions snapshots by the immutable base image,
CPU architecture, TE CUDA targets, Docker/NCCL recipe, uv configuration and
resolved build-tool dependencies. uv itself matches individual package versions
and Git commits. Unrelated runtime dependency changes therefore retain usable
wheels. Recipe changes conservatively invalidate the whole wheel cache.

Maintainer PR builds first look for their own compatible snapshot, then the
target branch's baseline. They publish only their own namespace. The existing
`populate-build-cache.yml` workflow publishes shared baseline snapshots. The
first build for a new compatibility key compiles normally; subsequent builds can
reuse its wheels. ARM and x86 snapshots are separate.

Restore and export run as separate Docker targets so snapshot changes do not
invalidate `main`'s Docker layers. A failed restore selects fresh cache mounts
before the source build. Failed or empty exports do not fail the test image
build. Export runs `uv cache prune --ci` to retain source-built wheels and remove
downloaded wheels. NCCL EP runtime headers are preserved separately by TE revision
for source builds whose wheels do not package them.

## Verification

Run `python -m unittest discover -s tests/test_utils -p test_uv_cache_key.py` to
check compatibility changes and reuse across unrelated dependency updates.

For an end-to-end check, build once and check that **Publish native wheel cache**
succeeds. On a fresh builder, restore the reported snapshot with target
`uv-cache-restore` and `--no-cache-filter uv-cache-restore`. Run the image build
with the same base-image digest, `UV_CACHE_ID`, and `NVTE_CUDA_ARCHS` arguments,
adding `--no-cache-filter main` to force dependency installation to execute.
Unchanged native packages should install from cached wheels without `Building`
messages. Check both the dependency duration and snapshot transfer time, then
run the normal GPU tests. An ordinary Docker layer hit alone does not verify
wheel-cache reuse.

The CI job summary records the compatibility key, selected snapshot and cache
destination. Full native-wheel reuse and elapsed-time savings must be verified
on the CI runners; the local cache-transfer smoke test does not measure them.
