# Packaging Verification

Executed on an allocated CPU node in a Slurm batch job.
Python 3.12, NumPy 2.5.4 (legacy only), setuptools 84.2.1, wheel 0.48.0.
Both distributions were built, installed as wheels, and tested after installation.

| Check | Result |
| --- | --- |
| API and dependency timeline contracts | 16 tests passed |
| Native CP1 export vs original frozen source | 72 predictions, exact equality |
| Native scope | 2 GPUs x 6 geometries x 3 padding/storage cases x F/B |
| Explicit legacy wrapper vs frozen estimator | 18 calls, exact equality |
| Full timeline vs original engine | 48 configurations, all fields/operations equal |
| Timeline coverage | PP1/2/4, VPP1/2, EP1/2/4, barrier on/off, recompute on/off |
| Installed import from an unrelated working directory | Passed |
| Native/E2E with NumPy, Torch, ORTools and original repo imports blocked | Passed |

Local Python 3.11 also passed the 16 small contract tests. No local search,
calibration, training or GPU benchmark was executed.

The full CPU output is retained in `verification/package-check.log`.
`tools/make_release.py` additionally requires every delivered Python/JSON source
file to match the built wheel byte-for-byte before making the release archive.

This verifies packaging, software equivalence, and API contracts. It does NOT
prove hardware timing accuracy, finish native multi-CP calibration, validate
PP/VPP against a new training run, or approve the diagnostic CP1 coefficients.
No held benchmark was released and no GPU work was submitted for packaging.

The source capsule and untrimmed reference engine are build-only inputs, not
runtime dependencies. To repeat the CPU harness, supply their paths through
`SOURCE_CAPSULE` and `TIMELINE_REFERENCE_PARENT`; see `PROVENANCE.json` for
their exact source identity. Ordinary contract tests need neither reference.
