# CPU-only runtime DCP planning

This package contains two CPU planners, physical-capacity and rank-row
contracts, CP/EP forward/backward timeline, and the frozen timing-model kernels
and parameters. It does not need the external `dcp-scheduler` repository.
`ORIGIN.json` identifies the source commit, original parameter artifacts and
hashes of packaged parameter files. Only runtime formulas and consumed
coefficients are packaged; imported models retain their original coefficients
and attribution. Profile metadata records calibration provenance and validation
scope. Offline fitting and detailed prediction traces remain outside the runtime.

The two solvers have the same structure: an injected cost model plus a search
algorithm. Model and solver selection are independent; neither solver inherits
the other's search or owns a private copy of a timing model.

```text
models/                     # all cost-model code and coefficient files
  qwen3_h100_p2p.py, *.json  # recipe-specific H100 P2P attention and EP costs
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

- `qwen3_h100_p2p`: the full Qwen3-30B-A3B H100 recipe described below.
  Classic and native P2P have independent timing tables. Each table supplies
  per-layer self-attention forward/backward costs, including projections and
  exposed CP communication, plus valid-token EP contributions, microbatch
  overhead and one iteration tail. Calibration and validation summaries are
  exposed through model metadata.
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

The training adapter uses the existing classic CP ProcessGroups and TE path
unless automatic native transport selection succeeds. Classic groups restrict
candidate widths to power-of-two divisors of DPxCP and starts to multiples of
their width. Native transport relaxes those constraints for the lightweight
solver; DynamicUSP retains its original aligned power-of-two allocation policy.
Both solvers optimize over legal candidates; the baseline-only control is
disabled by default.

The primary example adds runtime flags to a matching **full Qwen3-30B-A3B**
training configuration: BF16, 48 layers, hidden size 2048, 32 query heads,
4 KV heads, head dimension 128, QK layer normalization and RMSNorm, 128 experts,
top-8 routing and expert FFN size 768. Launch 32 H100s across four eight-GPU
nodes with TP1/PP1/EP8. Use fused attention, eager execution without activation
recomputation, and the flex/HybridEP dispatcher with variable-token padding.
Use global batch size 256, microbatch size 1 and a maximum sample length of 16384.

This profile requires `moe_router_force_load_balancing=True`. It describes a
forced-balanced benchmark configuration; it does not model ordinary training
with learned routing. The runtime checks model, precision, backend, padding and
parallelism settings against the profile's declared domain.

The current artifact contains fixed-layout module calibration. Its bootstrap
step overhead uses controlled CP8/DP4 input; the CP1 production input path and
candidate-count choices remain under real-data validation. The asset records
these input and timing adaptations explicitly.

Append these flags to that configuration and its varlen data arguments:

```bash
--attention-backend fused --cuda-graph-impl none \
--moe-token-dispatcher-type flex --moe-flex-dispatcher-backend hybridep \
--moe-hybridep-pad-variable-tokens --moe-router-force-load-balancing \
--calculate-per-token-loss --cross-entropy-loss-fusion \
--cross-entropy-fusion-impl native \
--micro-batch-size 1 --global-batch-size 256 \
--seq-length 16384 --max-position-embeddings 16384 \
--cp-comm-type p2p --context-parallel-size 1 \
--dynamic-context-parallel --min-dynamic-context-parallel-size 1 \
--sequence-packing-scheduler default_dynamic_cp \
--max-seqlen-per-dp-cp-rank 4096 --pad-packed-seq-alignment 4096 \
--thd-tail-padding-policy extend_last --thd-max-packed-sequences 64 \
--num-workers 0 \
--dynamic-cp-runtime-model qwen3_h100_p2p \
--dynamic-cp-runtime-solver dynamic_usp \
--dynamic-cp-runtime-solver-options '{"candidate_count":4,"outlier_threshold":1.0}'
```

Use the execution environment recorded by the calibration:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUDA_DEVICE_MAX_CONNECTIONS=1 NCCL_NVLS_ENABLE=0
export NVTE_FLASH_ATTN=0 NVTE_FUSED_ATTN=1 NVTE_ALLOW_NONDETERMINISTIC_ALGO=1
export NVTE_NORM_FWD_USE_CUDNN=1 NVTE_NORM_BWD_USE_CUDNN=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

This recipe restricts both solvers to CP1/2/4/8 and canonical size-aligned groups
inside one eight-H100 NVLink domain. The worker records an otherwise valid default
CP16/32 incumbent as unavailable and searches the profile's legal domain.
Each rank has a fixed 4096-token buffer;
`extend_last` determines attention's padded lengths. MoE cost uses valid tokens,
because HybridEP excludes padding rows. Its per-rank contribution is
`bias + linear * sum(logical_lengths) / (4096 * CP)`. The timeline waits for each
EP8 block, then averages the contributions over its eight ranks. This models
forced-balanced expert work; its accuracy on mixed layouts must be established
by the profile's validation. Microbatch overhead is charged for every candidate
microbatch, and iteration-tail overhead is charged once per step.

The CPU worker requires the profile's declared global sample count for a full-step
plan: this artifact uses 256. Per-pack attention queries remain independent of
that batch-level check.

The adapter selects the `classic_p2p` or `native_p2p` table from the execution
transport. Model selection does not enable native transport. A missing table or
an out-of-domain configuration is rejected; tables are not reused across
transports. With the native functional path, this model keeps the same declared
CP and placement limits. Other models and adapters retain their own CP domains,
including arbitrary widths when supported. Logical lengths and pack counts must
also remain within the profile's declared bounds.

The Python predictors need no additional numerical packages. The frozen
GB200/GB300 attention predictors also support an optional CPU JIT backend: make
Numba available and set `"kernel_backend":"numba"` in those model options.
No CUDA kernel is launched by these predictors. Caller-provided `module:factory`
model and solver factories and historical powers baseline hooks remain supported.
`--dynamic-cp-runtime-force-powers` bypasses
the solver search using the configured validated powers-only baseline.

With `dynamic_cp_runtime_model` configured, an optional generic hot-switch solver
is selected independently of the model. Solver/options flags alone do not enable
the runtime path.

`dynamic_usp` ports DynamicUSP `solve_gbs` Stage 1: long-sample group allocation, short
prepacking/LPT, empty-group repair, rank rotation, tier affinity, heavy-middle
ordering and bounded microbatch-count search. Allowed power-of-two widths are
taken from the existing geometry. Samples and physical sequence-count/padding
limits are validated by the same rank-row contract as the default planner.
The injected model supplies scheduling work and complete-step candidate scores;
no heads, attention implementation or communication backend is selected by this
solver. The configured incumbent is retained if no candidate is predicted faster.

The optional historical upstream reference proxy retains its original U8 profile
and CP powers of two with a minimum of 8:

```bash
--min-dynamic-context-parallel-size 8 \
--dynamic-cp-runtime-model usp_calib_v1 \
--dynamic-cp-runtime-model-options '{"allow_provisional":true}' \
--dynamic-cp-runtime-solver dynamic_usp \
--dynamic-cp-runtime-solver-options '{"candidate_count":4,"outlier_threshold":1.0}'
```

For a 32-rank DPxCP domain, both candidates and the incumbent are restricted to
CP8/16/32; CP24 is not supported. This recipe currently accepts DPxCP=8/16/32.
Execution continues to use the TE P2P path (`cp_comm_type=p2p`), while the
reference profile models U8 all-to-all plus ring communication. Its synthetic
44-layer shape, formulas, calibration asset, provenance and license are unchanged.
It remains a scheduling proxy for historical comparisons; its predictions are
**not** P2P latency or speedup evidence. Worker/consume telemetry records the
model name, source hashes and this scope.

Only the default `usp_calib_v1` formulas and its original table are imported;
custom profile options, optional profile-specific fixed-topology Stage 2,
PP/VPP simulation, CLI and calibration tools are not imported. Existing
PP1/VPP1 restrictions, the CPU worker, async lookahead and sync control are
unchanged. No external `hot_switch_solver` installation is needed. Upstream
revision and license are recorded in `DYNAMIC_USP_NOTICE.txt`.
