# CPU-only runtime DCP planning

This package contains the lightweight planner, physical-capacity and rank-row
contracts, CP/EP forward/backward timeline, and the frozen timing-model kernels
and parameters. It does not need the external `dcp-scheduler` repository.
`ORIGIN.json` identifies the source commit, original parameter artifacts and
hashes of the three runtime parameter files. Only selected formulas and consumed
coefficients are packaged; historical heads, calibration reports and detailed
prediction traces stay in the source repository. Coefficients are unchanged;
the scalar evaluator shares geometry/dispatch validation and contains no
offline composer or fitting diagnostics.

The two solvers have the same structure: an injected cost model plus a search
algorithm. Model and solver selection are independent; neither solver inherits
the other's search or owns a private copy of a timing model.

```text
models/                     # all cost-model code and coefficient files
  attention/                # FA model, frozen F/B formulas and GB200/GB300 data
  qwen35.py, qwen35.json     # mixed FA/GDN model
  usp_calib_v1.py, *.json    # temporary upstream U8 cost model
  base.py, cache.py         # shared model interface and memoized pack costs
runtime/                    # scheduling only
  lightweight.py            # arbitrary-CP candidate search
  dynamic_usp.py, hot_switch.py  # DynamicUSP allocation/count search
  planner.py, timeline.py   # common validation, rank rows and complete-step scoring
```

The worker runs this package directly by file path, without importing the
Megatron/PyTorch parent packages. A persistent leader-only CPU process solves one
batch ahead; the training process owns data reads and distributed calls. The
solver's time budget is cooperative, not a hard deadline. Late results are waited
for at consumption. Use `--no-dynamic-cp-runtime-async` for a same-solver control.

Available in-tree models:

- `calibrated_attention`: frozen GB200/GB300 FA core forward/backward formulas.
  Supply the query/KV heads, head dimension, layer count and measured
  non-attention times in model options. `allow_provisional=true` is required.
  Currently restricted to a single NVL72 domain; these core times exclude FA
  projections, and CUDA-graph/NCCL calibration transferred to eager/native is
  provisional, not native/eager accuracy evidence.
- `qwen35_gb200`: the frozen W32/EP4/local16K mixed model with paired FA
  remainders, GDN and non-attention parameters. It preserves the a55 model and
  its explicit mixed-stack/padding-transfer limitations, not a general Qwen
  configuration or a precision guarantee. The global partition must remain
  contiguous and the tail policy `extend_last` for that recipe.
- `usp_calib_v1`: temporary use of DynamicUSP's original default U8 all-to-all
  plus ring cost profile, including its **synthetic 44-layer shape**, MoE,
  replay, tier factors and global constant. The formulas and calibration table
  are unchanged, not refitted for Qwen or native P2P. `allow_provisional=true`
  is required. The model does not enable Ulysses or all-to-all communication.

The training adapter uses the existing classic CP ProcessGroups and TE path.
It restricts candidate widths to power-of-two divisors of DPxCP and group starts
to multiples of their width. Both solvers still optimize over these candidates;
the baseline-only control is disabled by default. Generic planner geometry can
also describe arbitrary widths and unaligned starts for other runtime adapters.

Example extra flags for a standard-FA NVL72 run (alongside the DCP,
varlen, packed-padding and parallelism arguments):

```bash
--dynamic-cp-runtime-model calibrated_attention \
--dynamic-cp-runtime-model-options '{"hardware":"gb200","query_heads":32,"kv_heads":4,"head_dim":128,"layer_count":48,"allow_provisional":true}' \
--dynamic-cp-runtime-solver-options '{"time_budget_s":0.8,"max_scored_candidates":4,"max_guide_candidates":4}'
```

The Python backend needs no additional numerical packages. To use the optional
CPU JIT backend, make Numba available in the training environment and add
`"kernel_backend":"numba"` to model options. No CUDA kernel is launched by these
predictors. Caller-provided `module:factory` model, solver and historical powers
baseline hooks remain supported. `--dynamic-cp-runtime-force-powers` bypasses
the solver search using the configured validated powers-only baseline.

With `dynamic_cp_runtime_model` configured, an optional generic hot-switch solver
is selected independently of the model. Solver/options flags alone do not enable
the runtime path:

```bash
--dynamic-cp-runtime-solver dynamic_usp \
--dynamic-cp-runtime-solver-options '{"candidate_count":4,"outlier_threshold":1.0}'
```

It ports DynamicUSP `solve_gbs` Stage 1: long-sample group allocation, short
prepacking/LPT, empty-group repair, rank rotation, tier affinity, heavy-middle
ordering and bounded microbatch-count search. Allowed power-of-two widths are
taken from the existing geometry. Samples and physical sequence-count/padding
limits are validated by the same rank-row contract as the default planner.
The injected model supplies scheduling work and complete-step candidate scores;
no heads, attention implementation or communication backend is selected by this
solver. The configured incumbent is retained if no candidate is predicted faster.

The temporary upstream-cost recipe keeps CP powers of two with a minimum of 8:

```bash
--min-dynamic-context-parallel-size 8 \
--dynamic-cp-runtime-model usp_calib_v1 \
--dynamic-cp-runtime-model-options '{"allow_provisional":true}' \
--dynamic-cp-runtime-solver dynamic_usp \
--dynamic-cp-runtime-solver-options '{"candidate_count":4,"outlier_threshold":1.0}'
```

For a 32-rank DPxCP domain, both candidates and the incumbent are restricted to
CP8/16/32; CP24 is not supported. This recipe currently accepts DPxCP=8/16/32.
Execution continues to use the existing TE P2P path (`cp_comm_type=p2p`),
not the modeled U8 all-to-all. This intentional mismatch is a temporary scheduling
proxy pending upstream updates: predictions are **not** P2P latency or speedup
evidence. Worker/consume telemetry records the model name, source hashes and
this scope. Switching to an updated model does not require changing the worker,
prefetcher or communication implementation.

Only the default `usp_calib_v1` formulas and its original table are imported;
custom profile options, optional profile-specific fixed-topology Stage 2,
PP/VPP simulation, CLI and calibration tools are not imported. Existing
PP1/VPP1 restrictions, the CPU worker, async lookahead and sync control are
unchanged. No external `hot_switch_solver` installation is needed. Upstream
revision and license are recorded in `DYNAMIC_USP_NOTICE.txt`.
