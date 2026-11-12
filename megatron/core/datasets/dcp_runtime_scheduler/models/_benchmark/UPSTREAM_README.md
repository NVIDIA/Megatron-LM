# Attention And Model E2E Predictors

Two independent Python distributions, no scheduler dependency:

- `dcp-attention-predictor`: ordinary causal GQA core attention, no projections.
- `dcp-model-e2e`: consumes attention predictors and explicitly supplied fixed
  non-attention measurements; it never fits the non-attention part.

This is a research snapshot, not a claim that the current calibration satisfies
the independent per-case 5% target. Experimental CP1 and historical multi-CP
calibrations are separate, explicitly selected profiles. No benchmark or held
GPU job is resumed by installing or using this bundle.

## Installation

Python >=3.10, no GPU required. From this directory:

```bash
python -m pip install ./packages/attention-predictor ./packages/model-e2e
# Optional dependency, only for explicitly selected historical profiles:
python -m pip install './packages/attention-predictor[legacy]'
```

`dist/` contains two wheels and a source bundle. Installed packages do not need
the original experiment directory. No license or redistribution rights are
invented by this export; existing source ownership continues to apply.

## Attention API

```python
from attention_predictor import AttentionPredictor

attention = AttentionPredictor(
    hardware="h100", query_heads=32, kv_heads=4, head_dim=128,
    sequence_alignment=64, pack_alignment=4096,
    allow_provisional=True,
)
result = attention.predict([1000, 2000], cp_size=1, phase="training")
print(result.time_ms)                  # ms, forward + backward
print(result.components_ms)            # independently modeled F/B
print(result.pack.storage_lengths)     # (1024, 3072)
```

Raw lengths are positive integers, with no minimum beyond 1. First align each
sequence to `sequence_alignment`, then round the total storage to
`pack_alignment`. Extend only the last real sequence's storage with the tail.
Do not change logical lengths or introduce a positive-length dummy sequence.
Unused slots in static65 metadata repeat the last endpoint.

### Experimental Native CP1

The default `experimental_cp1_bounded` profile exports diagnostic coefficients,
**not approved/adopted runtime parameters**:

- H100 and GB200, BF16 eager ordinary causal GQA core only. No projections,
  GDN, or CUDA Graph prediction.
- Geometries `(query_heads, kv_heads, head_dim)`: `(32,4,128)`, `(64,4,128)`,
  `(16,2,256)`, `(32,2,256)`, `(32,8,128)`, `(64,8,128)`.
- Final storage 4096, 8192, or 16384; at most 64 real sequences per pack.
- Separate coefficients for GPU, geometry and direction, common functional form:

$$T=c_0+c_1 B_{valid}+c_2 B_{padding}+c_3 R_{source}$$

`B_valid/B_padding` are compulsory tensor-footprint features in MB, not claims
of measured HBM traffic. `R_source` is a source-job resource critical span with
explicit resource weights and dependencies; worker ownership and the resource
envelope remain hypotheses. Exact prediction-only formula bodies are in
`attention_predictor/_native/`. No formula was refit by this packaging task.

The parameter reports contain 1200 H100 and 984 GB200 known development/control
measurements, **not independent holdout counts**. Artifact/output metadata keeps
`independent_accuracy_passed=False`. Explicit `allow_provisional=True` is required;
no mean or maximum 5% guarantee is made. Short or unseen input combinations do
not automatically acquire an accuracy guarantee.

Native multi-CP is unfinished. CP>1 raises an error, never silently falling
back to old coefficients.

### Explicit Legacy Multi-CP

```python
legacy = AttentionPredictor(
    "h100", 32, 4, 128, sequence_alignment=128, pack_alignment=16384,
    profile="legacy_frozen_20260922", allow_provisional=True,
)
estimate = legacy.predict([12000], cp_size=4, phase="forward", ranks=(0, 1, 2, 3))
```

Requires NumPy. The frozen registry determines which combinations are actually
available: CP<=64 does not mean all shapes/lengths/CP values were calibrated.
The H100 integer extension is explicitly selected by
`legacy_model_family="h100_qwen3_integer"` and stops at CP32. Alignment must be
divisible by `2*CP`. Actual physical rank IDs are required for multi-CP; H100
NVL8 / GB200 NVL72 placement is not guessed. Changed-domain backward placements
without an applicable calibration are rejected, not estimated from forward.

These are old TE/NCCL fits. `legacy_transport_backend="native"` is only a proxy,
not a new native calibration. This wrapper marks all legacy outputs provisional.
Native H100 CP128 and native arbitrary-CP acceptance remain unfinished. No
paused GPU work was resumed by packaging.

## Model E2E API

```python
from model_e2e import ModelEvaluator, NonAttentionTimes, PackGroup, qwen3_30b

# f_ms/b_ms must come from measurements of the same execution configuration.
non_attention = NonAttentionTimes(
    forward_ms_per_layer=f_ms,
    backward_ms_per_layer=b_ms,
    source="measurement record path or identifier",
)
model = ModelEvaluator(
    qwen3_30b(world_size=2, ep_size=2),
    non_attention, predictors={"gqa": attention},
)
schedule = [[PackGroup((1000, 2000), (0,)), PackGroup((4096,), (1,))]]
prediction = model.predict(schedule)
print(prediction.time_ms)
print(prediction.timeline.ep_sync_wait_gpu_ms)
```

Non-attention inputs are fixed **milliseconds per layer, per microbatch, per
rank**, independently supplied for F/B. Include projections, MLP/MoE, EP
communication and other excluded work; exclude predicted core attention to
avoid double counting. A scalar applies everywhere; heterogeneous layers may
use `{"gqa": ..., "gdn": ...}`; each value may instead be a list of fixed
measurements, one per physical PP stage. No automatic benchmark defaults,
token-dependent regression, or E2E residual are added. HybridEP valid-token
work variations remain an explicitly reported limitation of fixed inputs.
`iteration_tail_ms` is only for known schedule-independent measured work,
not a place to absorb unexplained residuals.

`world_size` counts DCP ranks **per PP stage**; total GPUs are
`world_size * pp_size`. Every microbatch must occupy exactly that many ranks
once each, not <=world_size. Groups must be contiguous ascending physical rank
intervals; non-power-of-two CP is allowed. Noncontiguous groups are rejected,
not silently relabeled. Global IDs including the PP stage offset reach predictors.

There is no cross-CP-group barrier after every microbatch. Each CP group waits
only for its own ranks. Fixed contiguous EP groups rendezvous each layer.
`first_forward_dp_cp_sync=True` retains the agreed first-forward barrier
approximation for parameter gathering; it can be explicitly disabled.
PP uses warmup/1F1B/cooldown; VPP uses interleaved, chunk-major dependencies.
These are modeled timelines, not hardware-validated PP performance guarantees.
The output includes full operations and wait accounting.

Forward orders core attention then EP/non-attention; backward reverses them.
`recompute_forward=True` explicitly adds whole-layer recomputation. It does
not double count FlashAttention backward's internal recomputation.

Presets only define layer layouts: `qwen3_30b` has 48 GQA layers;
`qwen35_35b` has the 40 decoder layers and the MTP layer of Qwen3.5-35B-A3B
(30 GDN and 11 GQA). The latter requires a separate GDN **core** predictor;
module-scope GDN times are not masqueraded as core costs.
Use `ModelConfig(layer_types=(...), ...)` for other layouts.

Custom predictors implement `predict(lengths, cp_size, *, phase, ranks)` and
return an object or mapping with `time_ms` and `scope="core_attention"`.
`model_e2e` imports neither `attention_predictor` nor a solver; an independent
predictor can be injected without changing the E2E package.

## Verification And Provenance

```bash
python -m unittest discover -s tests -v
python tools/verify_export.py /path/to/frozen/native-source.tar.gz
```

Tests cover padding, separate F/B, explicit non-attention input, projection
scope, EP/CP dependencies, PP/VPP, arbitrary CP and exact rank occupancy.
Source equivalence checks are software tests, **not new GPU accuracy tests**.
See `VALIDATION.md` for executed checks.

`PROVENANCE.json` records original commits, hashes, diagnostic parameter
sources and mechanical source pruning. Original reports are in `references/`.
Original repo/experiments/unopened holdout timings and held jobs are unchanged.
No commit or push was performed.

```text
packages/attention-predictor/  # component 1, parameters + explicit legacy
packages/model-e2e/           # component 2, fixed measurements + timeline
examples/                    # usage; E2E measurement inputs have no defaults
tests/                       # contract and timeline tests
tools/                       # mechanical export and equivalence verification
references/                  # original diagnostic reports, not acceptance
dist/                        # wheels and source bundle
```
