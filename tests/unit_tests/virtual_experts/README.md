# Virtual-expert regression suite

Run from the repository root in the dev container on one NVLink node:

```bash
# Full suite on four Blackwell GPUs (including MXFP8).
bash tests/unit_tests/run_virtual_expert_tests.sh

# Non-MXFP8 cases on all eight H100 GPUs, as in H100 CI.
GPUS_PER_NODE=8 bash tests/unit_tests/run_virtual_expert_tests.sh -m 'not launch_on_gb200'
```

The runner budgets 240 seconds for torchrun, pytest startup, compilation and test
execution. It returns failure on timeout, with ten seconds allowed for shutdown.
The dev container must provide compatible TE/cuDNN frontend/CuTe DSL versions;
selective recomputation requires the real fused SReLU kernel.
It uses synthetic inputs and no training dataset or checkpoint files. All cases
carry `internal`; only the three MXFP8 compute cases carry `launch_on_gb200` and
skip on pre-Blackwell GPUs. Configuration checks mentioning MXFP8 do not execute
FP8 kernels and run on H100 too. The H100 recipe runs the other 29 cases on eight
ranks; the GB200 recipe selects the three MXFP8 cases on four ranks. Both use a
dedicated dev/latest bucket, excluded from their catch-all. The GTP and planner
cases keep four-rank groups (two independent groups on H100), and BF16 parity
keeps EP2. Other world sizes skip the suite. H100 needs a dev container with
compatible HybridEP and TE grouped-tensor support; the installed TE's BF16
Hopper path requires cuBLASLt 13.4 or newer. `HYBRID_EP_CACHE_DIR` selects the
writable kernel-cache directory; the default is `/tmp/megatron-virtual-expert-hybridep`. The integration fixture
sets HybridEP's `load_cached_kernels=True`: the first model compiles real kernels,
and subsequent models reload them instead of recompiling after buffer teardown.
The installed backend isolates these files by process ID, so the first HybridEP
compilation still counts toward every run's budget.

## Measured runtime

The 32 cases passed on all four ranks in **178 seconds end to end**, including
torchrun/pytest startup and coverage collection, on the available four-GPU GB300
node (2026-09-17). The two training variants took approximately 108 seconds and
10 seconds, respectively, with the second reusing the first's compiled kernels.
The fixed-weight BF16 case added approximately four seconds.
Run with `VIRTUAL_EXPERT_TEST_COVERAGE=1` to include coverage.
This measurement used existing Triton/CuTe compilation caches and compatible
native cuDNN frontend/CuTe DSL packages. It includes the first HybridEP
compilation, but is not an entirely cold-cache or GB200 measurement. Both CI
buckets enforce the same 240-second test budget. The eight-H100 and four-GB200
runs still need validation on their target hardware. The original exhaustive
additions are reduced from 153 parameterized cases to 32.

## What the small suite protects

| Coverage | Regression caught |
| --- | --- |
| Independent CPU planner oracle, three small shapes | Wrong expert ownership, dropped/duplicated routes, ties, strided inputs, stale workspace after changing load |
| 512-expert, top-10 sigmoid/quantile routing | Wrong compact IDs, scores or gradients; broken dense fallback; duplicate quantile accumulation during recomputation |
| Configuration boundaries | Unsupported storage, topology, dropping or graph scopes accepted silently |
| Tiny real four-rank GTP collectives | Peek returning stale weights, duplicate gathers, premature scratch reuse, missing gradient contributions or nonzero padding |
| Small `HybridModel`, `EE/E/E`, two layouts | EP2 with unsharded experts and a BF16 MTP parameter override; EP2 × EGTP2 with native MXFP8 decoder/MTP experts. Both cover repeated MTP, HSM, latent/shared experts, grouped-MLP offload, dense GTP4, BF16 gradients with FP32 reduction, FP8 parameter gathering and distributed Adam updates; parity with ordinary HybridEP over two changed batches |
| Fixed-weight BF16 routing parity | Bitwise outputs, input/probability gradients and router gradients, with only expert wgrads allowed accumulation-order error; two uses of the same layer before backward, opposite load skews and an expert split between its owner and a virtual copy |
| Direct expert-MLP parity | MXFP8 selective `moe_act` recomputation preserves outputs, input/probability gradients and all expert weight gradients without another transport setup |

Each integration layout uses the same seeded model and batches with virtual
experts disabled as its numerical reference. The EGTP case asserts that all
decoder and MTP expert weights occupy half their full size and belong to the
actual two-rank expert-GTP group; gradients and updates are compared shard by
shard on every rank. Both variants use EP2 with two local experts to reuse
HybridEP and grouped-GEMM compilation; four-rank planner and GTP collective behavior are checked separately.
Both layouts compare losses and every parameter gradient, require
nonzero router gradients and an active virtual expert, and check that offloading
actually transfers bytes. Repeated MTP plans must remain independent until
backward.

The separate BF16 case disables FP8 globally, verifies identical starting
weights, and compares full MoE outputs before any optimizer update. It requires
bitwise equality (including signed zero) of router IDs/probabilities, outputs,
input/probability gradients and router weight gradients. Each use must split the
hot expert between native and virtual execution, and the two plans must own
distinct storage until backward. Expert wgrads accumulate in FP32 and receive a
small summation-order tolerance. This separates fixed-weight routing invariants
from quantization error and optimizer drift in the two-step MXFP8 training cases.
Numerical bounds are:

| Comparison | Relative L2 limit | Observed maximum L2 | Peak error / reference peak limit |
| --- | --- | --- | --- |
| BF16 outputs, input/probability/router gradients | Bitwise | Bitwise | Bitwise |
| BF16 expert weight gradients (FP32 accumulation) | 0.001% | 0.000006% | 0.001% |
| Training losses | 0.01% | 0% | 0.01% |
| Every training parameter gradient | 1% | 0.339% | 2% |
| Optimizer parameter updates | 2% | 1.899% | 210% |
| Recomputation: FC2 weight gradients | 2.5% | 1.824% | 6% |
| Recomputation: outputs and all other gradients | Bitwise | Bitwise | Bitwise |

The cases make 1,176 numerical and 84 bitwise tensor comparisons across the four
ranks. The EGTP variant retains the same limits (observed gradient L2 below
0.283%, update L2 below 1.9%, reproduced in standalone and full-suite runs). Each
L2 bound applies separately to each tensor, on every rank and training step.
Adam can amplify isolated sign changes in near-zero gradients, hence the looser
peak bound for optimizer updates; their aggregate L2 bound still applies.
Requantizing recomputed MXFP8 activations introduces larger FC2 weight-gradient
error. The EGTP addition retains the previously tightened limits. This covers
representative combinations without sweeping every topology or internal
implementation detail.

## Relationship to `run_nt4_nano.sh`

The tests reduce model width, tokens, depth and EP size to run on four GPUs. The
512-expert/top-10 routing semantics are still tested independently. The large
GDP/attention stack, eight-node transport, Muon, NVLS user buffers, checkpoint
save/resume and throughput are outside this suite.

The supplied launcher targets a different checkout. This branch does not expose
`activation_func_tanh_clamp_scale`, quantile-balancing estimation scope/bin count,
wide residuals, MoE shortcuts, or MTP loss normalization by main tokens. Its
quantile router rejects `moe_router_fusion=True`, so quantile integration uses the
unfused router and the compact fused route is checked separately with seq-aux
routing. TE's BF16 squared-ReLU path rejects selective `moe_act` recomputation;
the direct MXFP8 expert case covers recomputation separately from mixed MTP.

The BF16 MTP override currently guarantees parameter storage only: the fused
expert path calls TE ops directly, bypassing the grouped-linear wrapper's
precision context. With EGTP this combination fails in ordinary HybridEP
backward (`Tensor` has no `_columnwise_data`) when the fused FP8 kernel receives
re-gathered BF16 weights. The EGTP variant therefore uses native MXFP8 throughout,
including MTP. Mixed BF16-parameter/FP8-compute MTP remains covered without EGTP.
The fixed-weight case checks true BF16 expert compute without EGTP. The initial
standalone BF16 + EGTP2 probe's backward `KeyError: None` was traced to its
prefetch-chain layout, rather than a general precision restriction. Classifying a
bare `MoELayer` gives names like `experts.linear_fc1.weight0`; the classifier only
splits routed FC1/FC2 chains for names containing `.mlp.experts.`. Giving the probe
normal model names makes its repeated BF16 + EGTP2 execution and parity pass on
all four ranks without a production change.

The mixed chain exposes a duplicate-prefetch bug: VE peeks both FC weights before
either GEMM, then consuming FC1 prefetches the already-ready FC2 again. FC2 consumes
the drained marker but leaves that new forward handle pending. Backward mistakes
it for a backward gather and looks up the unset backward ticket. A two-weight GTP
probe reproduces this without MoE/TE kernels. A broad guard that skips already
available neighbor prefetches breaks dense training, because readiness does not
distinguish forward from backward. The targeted fix is to drain a newer pending
handle even when consuming an earlier drained marker. A test-process prototype
passes the original probe and all 32 suite cases (171 seconds with coverage on
four GB300 GPUs); that prefetch fix is not included in this branch.
Full BF16 MTP compute under the mixed-precision recipe still requires the
precision-policy fix above.

## Branch layout

- `jiemingz/replica_hybridep`: production changes together with this reduced
  32-case regression suite, replacing the original exhaustive test additions.
- `archive/replica-hybridep-exhaustive-tests-20260917`: original branch tip
  `bc5a4cb806` and all comprehensive tests, preserved for deeper investigations.
- `jiemingz/replica-hybridep-tests`: retained test-development branch whose
  changes have been fast-forwarded onto `jiemingz/replica_hybridep`.

The original branch now contains both the production code and its reduced tests.
The test cleanup leaves the production implementation unchanged.
