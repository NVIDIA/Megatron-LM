# Financial time-series data pipeline

**Turn a parquet store of market data bars into training-ready PyTorch tensors for patched time-series transformers.**

`preprocessing.py` reads per-bar market data from parquet, groups rows into
*patches* (the model's tokens), slides leak-free context windows over each
instrument, materialises future return/volume targets at multiple horizons, and
writes one tensor file per split (`train.pt` / `val.pt` / `test.pt`) plus a
`meta.pt` descriptor — ready to feed a PatchTST-style decoder.

| File | Role |
|---|---|
| [`preprocessing.py`](preprocessing.py) | parquet bars → windowed patch tensors (standalone; needs only pandas, pyarrow, torch) |
| [`ts_dataset.py`](ts_dataset.py) | Megatron data layer: map-style dataset over the tensors, `train_valid_test_datasets_provider`, TP-aware `get_batch` |
| [`ts_mlp_model.py`](ts_mlp_model.py) | `TSMlpModel`: the smallest `MegatronModule` that fits the tensor contract; a placeholder for a real architecture |
| [`pretrain_ts.py`](pretrain_ts.py) | `pretrain()` entry point wiring the three together |
| [`../../tests/unit_tests/data/test_financial_timeseries_preprocessing.py`](../../tests/unit_tests/data/test_financial_timeseries_preprocessing.py) | unit tests for targets, splits and leakage checks |


## Table of Contents

- [Environment](#environment)
- [Quick Start](#quick-start)
- [Training with Megatron](#training-with-megatron)
- [Command-Line Reference](#command-line-reference)
- [Input Format](#input-format)
- [Output Format](#output-format)
- [Sizing Notes](#sizing-notes)

## Environment

`preprocessing.py` needs only `pandas`, `pyarrow`, `psutil`, `tqdm` and `torch`; the training
example needs a Megatron-LM dev container (see the
[container setup guide](../../skills/mcore-build-and-dependency/SKILL.md)). Run every command
below from the repository root.

## Quick Start

```bash
# minimal — only --data-dir is required
python examples/financial_timeseries/preprocessing.py --data-dir /path/to/parquet_store
```

```bash
# realistic example: 8-row patches, 64-patch context, cumulative targets
# over the next 5/15/30 time units, no overnight exposure
python examples/financial_timeseries/preprocessing.py \
    --data-dir  /path/to/parquet_store \
    --out-dir   /path/to/tensors \
    --patch-len 8 --context-len 64 --window-stride 16 \
    --horizons 5,15,30 --horizon-unit time_units --target-mode cumulative \
    --no-allow-overnight-targets
```

The run logs each phase (store build, per-split window counts and expected disk
size, leakage check) and writes four files into `--out-dir`:

```
tensors/
├── train.pt    # dict of stacked tensors (see Output Format)
├── val.pt
├── test.pt
└── meta.pt     # dataset descriptor + train-only normalization stats
```

## Training with Megatron

`pretrain_ts.py` wires the tensors into Megatron's `pretrain()` loop with a deliberately minimal
model, `TSMlpModel` in `ts_mlp_model.py`: each patch is flattened, embedded, combined with a learned
instrument embedding and a time-of-day embedding, and mapped directly to the return and volume heads
at every horizon. It has no attention and never looks at neighbouring patches, so it cannot exploit
the context window. This is the smallest model that fits the data contract. It gives a fast
end-to-end check that preprocessing, the dataloader, the target mask and the loss agree, and it
marks where a real architecture goes.

The dataset layer in `ts_dataset.py` memory-maps each split, checks that `--seq-length` equals the
preprocessing `--context-len`, and builds the target mask from the `NaN` targets at load time.
Patch length, horizons and the time-of-day bucket count are read from `meta.pt`, so the trainer takes
no flags for them; `--seq-length` is the one geometry flag Megatron itself needs.

```bash
torchrun --nproc_per_node=4 -m examples.financial_timeseries.pretrain_ts \
    --ts-data-path /path/to/tensors --dataloader-type cyclic \
    --num-layers 2 --hidden-size 256 --num-attention-heads 4 \
    --seq-length 64 --max-position-embeddings 64 \
    --micro-batch-size 64 --global-batch-size 256 --train-iters 2000 \
    --lr 3e-4 --min-lr 3e-5 --lr-decay-style cosine --lr-warmup-iters 100 \
    --tokenizer-type NullTokenizer --bf16 \
    --log-interval 50 --eval-interval 500 --eval-iters 20 \
    --save /path/to/ckpt --save-interval 2000
```

Example-specific flags: `--ts-data-path` (required), `--ts-no-mmap`, `--loss-lambda-volume`
(weight of the volume term, default 1.0), `--loss-lambda-l1` (L1 fraction of the regression
loss) and `--ts-horizon-weights` (one weight per horizon). `--eval-interval`, `--eval-iters` and
`--log-interval` have no defaults in Megatron and must be given.

### Swapping in your own model

Replacing the MLP leaves the data layer untouched. `forward_step` calls the model with `patches`
(B, L, P, F), `inst_id` (B,) and `patch_tod` (B, L); `loss_func` expects a dict with `return` and
`volshare` tensors of shape (B, L, H), one forecast per context position and horizon. Any
`MegatronModule` that takes those three inputs and returns that dict can replace it with a one-line
change in `model_provider`. Subclassing `MegatronModule` (rather than a plain `nn.Module`) gives the
training loop the hooks it relies on: `model.config`, `set_input_tensor` for the pipeline schedule,
and `state_dict_for_save_checkpoint` for checkpointing; `ts_mlp_model.py` shows the minimum.

Target scaling is the model's responsibility. `meta.pt` carries per-horizon target statistics
computed from the training split only, so a model that standardises its targets can reuse them at
validation and test time without future data determining the scaling; `load_target_stats` in
`ts_dataset.py` reads and validates them. The example trainer leaves targets unscaled.

## Command-Line Reference

| Flag | Default | Description |
|---|---|---|
| `--data-dir` | *(required)* | Directory holding the parquet files. Hive partitions are understood; files must satisfy the ordering and grouping requirements under Input Format. |
| `--out-dir` | `<data-dir>/../tensors` | Output directory for `train.pt` / `val.pt` / `test.pt` / `meta.pt`. |
| `--tmp-dir` | `<out-dir>/.tmp_<split>` | Scratch space for temporary memmaps (≈ the size of the final `.pt` files; removed after each split is saved). |
| `--patch-len` | `8` | **P** — rows (time units) per patch. One patch becomes one model token. |
| `--context-len` | `32` | **L** — patches per training window. |
| `--window-stride` | `8` | Patches between consecutive window starts. `stride < context_len` ⇒ overlapping windows (more samples); `stride = context_len` ⇒ disjoint windows. |
| `--horizons` | `1,5,15,60` | Comma-separated forecast horizons, in `--horizon-unit` units. Sets **H**. |
| `--horizon-unit` | `time_units` | Unit for horizons: `time_units` (rows — minutes for 1-minute bars; `time-units` / `minutes` are accepted aliases) or `patches`. |
| `--target-mode` | `point` | `point`: target the single step at *t + h*. `cumulative`: target the whole interval *t + 1 … t + h* (summed return, total volume). |
| `--allow-overnight-targets` / `--no-allow-overnight-targets` | allow | Allowed: target intervals may cross the session close (cumulative returns then include the close-to-open move). Disallowed: targets ending on another trading day are stored as `NaN` (masked in the loss); every horizon must then fit inside one session (day). |
| `--within-day-context` | off | Reject windows whose *context* crosses a day boundary (targets may still reach into later days). |
| `--split-fracs` | `0.6,0.2,0.2` | Train/val/test fractions **by date**, summing to 1. Splits are contiguous in time: train earliest, test latest. |
| `--embargo-days` | `2` | Trading days dropped from the end of the training split to create a gap before validation. Applies only to the train/validation boundary; validation and test remain adjacent. |
| `--time-units-per-day` | `390` | Rows per trading day (390 = 1-minute bars, 09:30–16:00; 78 for 5-minute bars). Sets patches/day (`time_units_per_day // patch_len`; leftover rows at day end are dropped) and the intraday-horizon check. |
| `--features` | `log_return,log_volume` | Per-row features (**F** per step). The first two must remain `log_return,log_volume` (derived from Close/Volume; downstream losses rely on their positions). Additional names are read as parquet columns of the same name and passed through unchanged (e.g. a market return). |

> [!NOTE]
> **Patches vs. windows:** Each instrument/day is divided into non-overlapping
> patches of `patch_len` rows. Overlapping context windows reuse these whole
> patches; their step size is `window_stride` patches. Trailing rows that cannot
> form a complete patch are dropped without padding (390 rows with `patch_len=8`
> yield 48 patches and drop the last 6 rows).

## Input Format

A directory of parquet files; each row is one **bar**, i.e. the trades of one instrument
aggregated over one time unit (one minute by default). Only the columns below are read; any
other OHLC fields are ignored.

| Column | Type | Description |
|---|---|---|
| `Symbol` | string | Instrument identifier. |
| `Date` | date / datetime | Trading day. Rows are grouped by (`Symbol`, `Date`). |
| `Timestamp` | datetime | Bar time; orders rows within a day. |
| `Close` | float | :warning: **Last price of the *bar*, not of the day.** Each row carries the close of its own time unit, e.g. the 1-minute close. Must be > 0 (log returns are taken). |
| `Volume` | float / int | Volume traded in the *bar* (same time unit as `Close`), not the daily total. |
| *extra feature* | float | Optional — any column named in `--features` beyond the first two is passed through as an input feature. |

> [!NOTE]
> **Rows are steps, whatever their spacing.** A patch is simply `patch_len`
> consecutive rows of one (`Symbol`, `Date`). Gaps are **not** filled — if your
> steps should be equally spaced (e.g. a 1-minute grid), forward-fill upstream.
> On a regular grid, `patch_tod` (a patch's position within its day) corresponds
> to clock time.

Additional contract details:

- Per (`Symbol`, `Date`), only the first `(n_rows // patch_len) × patch_len` rows are used; a day with fewer than `patch_len` rows contributes nothing.
- `log_return` is computed within the day from consecutive bar closes (`log(Close_t) − log(Close_{t−1})`, first row of each day = 0), so the overnight gap never appears in the **inputs**; it can appear in cumulative **targets** when overnight targets are allowed.
- `log_volume` is `log1p(Volume)` of the bar, so zero-volume bars are well defined.
- All dates found in the store define the split calendar.

### File organization and ordering

Prepare the source files upstream so they satisfy these requirements:

- Keep all rows for each (`Symbol`, `Date`) together in a single parquet file. Different instruments for the same date may be stored in separate files.
- Files are processed in lexicographic order of their full paths. For each instrument, the dates in each subsequent file must come after its dates in the preceding file. Use sortable, zero-padded date names or partitions, such as `2026-01.parquet`, `2026-02.parquet`.

Files may cover a day, a month, or another date range provided these requirements
hold. Within each file, the script groups rows by instrument and date and sorts
each group by timestamp, so rows within a file may be unsorted. Feature
construction and patching happen independently per file; the resulting patches
for each instrument are then concatenated in file order. Splitting an
instrument's day across files resets its intraday returns and patch positions at
each file boundary, while files in the wrong order produce incorrect sequences
and future targets.

## Output Format

### Split files — `train.pt`, `val.pt`, `test.pt`

Each file is a `dict[str, torch.Tensor]` with a leading sample axis **N**
(windows in that split). With **L** = `context_len`, **P** = `patch_len`,
**F** = features per step, **H** = number of horizons:

| Key | Shape | dtype | Content |
|---|---|---|---|
| `patches` | `(N, L, P, F)` | `float32` | Model input: raw per-row features grouped into L patches of P rows. Unnormalized. |
| `inst_id` | `(N,)` | `int64` | Instrument index (see `sym_to_id` in meta) for a learned instrument embedding. |
| `patch_tod` | `(N, L)` | `int64` | Time-of-day index of each patch: position within the trading day, `0 … patches_per_day − 1`. |
| `target_return` | `(N, L, H)` | `float32` | Future **log return** at every context position × horizon. `point`: the single step at *t + h*; `cumulative`: the summed log return over *t + 1 … t + h* (including the overnight move where allowed). |
| `target_volshare` | `(N, L, H)` | `float32` | Future **volume** target on the same intervals: `log1p(volume)` of the step (`point`) or of the summed volume (`cumulative`). *(Key name is legacy.)* |

> [!IMPORTANT]
> Targets that cannot be formed (an intraday-only interval crossing the session
> close) are stored as **`NaN`**. No mask tensor is written — reconstruct it at
> load time as
> `target_mask = isfinite(target_return) & isfinite(target_volshare)`
> and zero-fill the NaNs before computing the loss.

Windows are stacked grouped by instrument, with ascending start positions within
each instrument.

### `meta.pt`

A `dict` with everything a training run needs to configure itself consistently:

| Key | Content |
|---|---|
| `n_instruments`, `symbols`, `sym_to_id` | Instrument count, ticker list, ticker → `inst_id` mapping. |
| `features` | Feature names, in `patches`' last-axis order. |
| `patch_len`, `context_len` | P and L used to build the windows. |
| `horizons`, `horizon_unit`, `horizon_minutes` | The H horizons, their unit, and their time-unit equivalents. |
| `target_mode`, `allow_overnight_targets` | How targets were constructed. |
| `target_stats` | **Train-split-only** per-horizon statistics (`return_mean/std/count`, `volume_mean/std/count`, each length H) — normalize targets with these to avoid val/test leakage. |
| `valid_target_counts` | Non-`NaN` return/volume target counts per split and horizon. |
| `input_token_shape` | `(patch_len, n_features)`; the flattened per-token input size is their product. |
| `time_units_per_day`, `patches_per_day`, `num_tod_buckets` | Session geometry; a time-of-day embedding needs ≥ `num_tod_buckets` entries. |
| `window_counts` | Number of windows (N) per split. |
| `split_fracs`, `embargo_days` | How the date splits were drawn. |


## Sizing Notes

Output size is dominated by `patches`: roughly `N × L × P × F × 4` bytes, where
`N` grows with the number of instruments and days, and with `1 / window_stride`
(overlap). The run logs the expected memmap size per split before materialising
it, so you can abort early if the configuration is too large for your disk.
