# Indexer top-k selection in Megatron Lite

The sparse attention layers of the native `glm5` model (DSA) and the compressed sparse attention
layers with compression ratio 4 of the native `deepseek_v4` model (CSA) select, for every query
token, the `index_topk` keys its attention reads: an indexer scores the keys the token sees and
keeps the best ones. By default every layer runs its upstream selector. `ImplConfig.indexer_topk`
installs an indexer top-k binding on every such layer instead; in eval mode with gradients
disabled, the binding selects with:

- the matched-precision reference selector: the indexer queries and keys are quantized to the
  model's indexer format (DSA: FP8 E4M3 rows with float32 scales; CSA: indexer MXFP4), DeepGEMM
  `fp8_fp4_mqa_logits` scores every visible key in float32, and a top-k kernel keeps the best
  keys of each row. Rows are scored in chunks whose scores fit a byte budget, so a long prompt
  never holds the scores of all its rows at once;
- LiteTopK, an external CUDA selector loaded as a plugin, on the rows of long sequences that its
  plan gives it. The reference selector selects every other row and recomputes the rows the
  plugin reports as overflowing or failed.

Megatron Lite owns the bindings, the reference selector, the planning and the plugin loader
(`megatron.lite.primitive.kernels.indexer_topk` and
`megatron.lite.primitive.modules.attention.indexer_topk`). The LiteTopK plugins, the exact-tie
top-k package and DeepGEMM are optional dependencies that are not installed with Megatron Lite;
see [Dependencies](#dependencies).

## Enabling it

Set `indexer_topk` in the model's `impl_cfg`. The runtime forwards it to the model's `ImplConfig`
unchanged:

```python
from megatron.lite.runtime import MegatronLiteConfig, RuntimeConfig, create_runtime
from megatron.lite.runtime.contracts import ParallelConfig

backend_cfg = MegatronLiteConfig(
    model_name="glm5",
    hf_path="/models/GLM-5.2",  # build_model reads the checkpoint path from here
    parallel=ParallelConfig(cp=8, ep=8),
    impl_cfg={
        "use_thd": True,
        "optimizer": None,
        "indexer_topk": {
            "backend": "litetopk",
            "precision": "exact",
            "litetopk": {
                "source": "/deps/plugins/glm-litetopk-raw32-abi1",
                "expected_source_id": "83669db87b20",
                "expected_adapter_sha256": (
                    "46db6d898e4b4487e502f0687df7bb22ce075869667b13bf0c43a8e2a0e27bd2"
                ),
                # The plugin's own file name (see Dependencies). Or build_dir and
                # deepgemm_include_dir for a JIT build of the plugin sources.
                "prebuilt_extension": (
                    "/deps/build/raw32-abi1/"
                    "sglang_litetopk_dsa_b200_production_83669db87b20.so"
                ),
            },
            "exact_topk": {"source": "/deps/native-exact-tie"},
        },
    },
)
runtime = create_runtime(
    RuntimeConfig(backend="mlite", hf_path="/models/GLM-5.2", backend_cfg=backend_cfg)
)
handle = runtime.build_model()
with runtime.eval_mode(handle):  # model chunks in eval mode, gradients disabled
    ...  # forward passes here select through the bindings
```

DeepSeek-V4 takes the same field. The qualified MXFP4 LiteTopK plugin does not select exactly, so its LiteTopK
configuration uses `precision="fast"`; the exact-tie top-k remains optional there and makes the
reference rows choose among keys tied at a row's cutoff score deterministically (lowest key id
first):

```python
impl_cfg = {
    "use_thd": True,
    "optimizer": None,
    "indexer_topk": {
        "backend": "litetopk",
        "precision": "fast",
        "litetopk": {
            "source": "/deps/plugins/dsv4-litetopk-ac1c-abi1",
            "expected_source_id": "ac1c7f51b362",
            "expected_adapter_sha256": (
                "69f3e88a0a6ce49a532a0932a95e99db02b42974b4465fda7b9b80615e380318"
            ),
            "prebuilt_extension": (
                "/deps/build/ac1c-abi1/sglang_litetopk_dsa_b200_production_ac1c7f51b362.so"
            ),
        },
        "exact_topk": {"source": "/deps/native-exact-tie"},
    },
}
```

For deterministic exact MXFP4 selection, use `backend="reference"`, `precision="exact"`
and configure `exact_topk`. This ranks the quantized MXFP4 operands exactly. The fast slab
plugin folds its score epilogue and stores a truncated 24-bit score code; setting `exact_topk`
only controls its reference rows and does not make its LiteTopK rows exact. Requesting
`backend="litetopk"`, `precision="exact"` with that plugin fails at model construction.

Without the runtime, pass the field to the protocol's `ImplConfig` (a mapping or an
`IndexerTopKConfig`):

```python
from megatron.lite.model.glm5.lite import protocol

# In an initialized torch.distributed job, as the runtime does.
model_cfg = protocol.build_model_config("/models/GLM-5.2")
impl_cfg = protocol.ImplConfig(
    optimizer=None, indexer_topk={"backend": "reference", "precision": "fast"}
)
bundle = protocol.build_model(model_cfg, impl_cfg=impl_cfg)
installation = bundle.extras["indexer_topk"]  # IndexerTopKInstallation
```

| Field | Default | Meaning |
| --- | --- | --- |
| `backend` | `"default"` | `default` binds nothing: every layer keeps its upstream selector. `reference` selects every row with the reference selector. `litetopk` selects with the plugin where its plan allows and with the reference selector elsewhere |
| `precision` | `"exact"` | `exact` or `fast`, see [Precision](#precision) |
| `litetopk` | None | The plugin directory and its optional pins (`LiteTopKPluginConfig`: `source`, `expected_source_id`, `expected_adapter_sha256`, `prebuilt_extension`, `prebuilt_extension_sha256`, `build_dir`, `deepgemm_include_dir`). Required by `backend="litetopk"` |
| `exact_topk` | None | The exact-tie top-k package (`ExactTopKConfig`: `source`, and `expected_sha256`, per-file pins). Required by `precision="exact"` unless the backend is `default` |
| `head_padding` | `False` | Let LiteTopK serve a layer whose indexer head count its route has no kernels for, on the route's next larger head count with zero heads appended; see [Head counts](#head-counts-and-zero-head-padding) |

- `ImplConfig` validates the field when it is built: unknown keys and invalid values or
  combinations raise `IndexerTopKConfigError` (a `ValueError`). It keeps the value as given.
- `build_model` hands the field to `configure_indexer_topk` right after it builds the model
  chunks, with the model's indexer format (GLM-5: `native_format="fp8"`; DeepSeek-V4:
  `native_format="mxfp4"`), and puts the result in `ModelBundle.extras["indexer_topk"]`: an
  `IndexerTopKInstallation`, or None for `backend="default"`. Unset, `build_model` does not touch
  the layers, adds no `extras` key and imports nothing of the indexer top-k package.
- `configure_indexer_topk` binds every DSA layer that selects its own top-k (IndexShare shared
  layers reuse the top-k of their source layer and stay unbound) and every CSA layer with
  compression ratio 4. It validates every layer before it binds any: it negotiates the layer's
  head counts with the reference score kernel, which it probes, and, for `backend="litetopk"`,
  loads the plugin and negotiates its route (see
  [Head counts](#head-counts-and-zero-head-padding)), so an unusable configuration fails when the
  model is built.
- The model configuration carries no tuning values. The selection plan (tile length, start
  position, seeding, memory budgets, plugin settings) is derived per layer by
  `resolve_indexer_topk_tuning`, the single policy seam in
  `megatron.lite.primitive.kernels.indexer_topk`; harnesses and tests can pass an
  `IndexerTopKTuning` to `configure_indexer_topk` directly.

## Dependencies

### LiteTopK plugins

A plugin is a directory holding the adapter module `litetopk.py` and its CUDA sources
`litetopk_kernels/{dsa_litetopk.cu, sm100_dsa_litetopk.cuh, dense_topk_litetopk.cuh}`. Its CUDA
extension is a prebuilt shared library (`prebuilt_extension`) or a JIT build of those sources
(optionally into `build_dir`, with the DeepGEMM headers of `deepgemm_include_dir`). A prebuilt
extension must keep the file name the plugin builds,
`sglang_litetopk_dsa_b200_production_<source id>.so` (after resolving symbolic links): every
plugin below checks it and fails to load a file of another name. Megatron Lite loads a plugin only
from the configured path, never from environment variables, and compares its pins before any
plugin code runs:

- `expected_source_id`: the first 12 hex digits of the SHA-256 over the name and then the bytes of
  each of the three CUDA files, in the order above (the plugins' own source id);
- `expected_adapter_sha256`: the SHA-256 of `litetopk.py`, which the source id does not cover;
- `prebuilt_extension_sha256`: the SHA-256 of the prebuilt extension.

Every pin is optional. Unpinned values are computed, logged once as a warning and recorded in
`IndexerTopKInstallation.plugin_info`. The plugins this integration was validated with:

| Plugin | Source id | `litetopk.py` SHA-256 | Route | Use |
| --- | --- | --- | --- | --- |
| `glm-litetopk-raw32-abi1` | `83669db87b20` | `46db6d898e4b4487e502f0687df7bb22ce075869667b13bf0c43a8e2a0e27bd2` | `fp8_paged`, exact | GLM-5 (32 FP8 indexer heads), exact |
| `glm-litetopk-raw32h64-abi1` | `7e5eb835fb7f` | `65951346a8c60d44945f1e7f03c5701e9c7e034f27d5274ee727a91d1a64aaac` | `fp8_paged`, exact | FP8 indexers with 32 or 64 heads, exact |
| `dsv4-litetopk-ac1c-abi1` | `ac1c7f51b362` | `69f3e88a0a6ce49a532a0932a95e99db02b42974b4465fda7b9b80615e380318` | `fp4_slab`, fast | DeepSeek-V4 |
| `glm-litetopk-996e-abi1` | `996e735c52df` | `37ff4143292871e0293c6e388ee96c90549c4fe60edf24dc9498e0bc165c2051` | `fp8_paged`, fast | GLM-5, the previous integration's selector |

These plugins and the exact-tie top-k package below are external (ABI version 1) and not yet
publicly released. Until they are, the only configuration that runs with public dependencies is
`{"backend": "reference", "precision": "fast"}` (the reference selector with DeepGEMM and the
cuDNN frontend radix top-k), and the GPU tests that need a plugin or the exact-tie package skip.

TODO(litetopk-public-link): the public location and license of the LiteTopK plugins.

### Exact-tie top-k

`exact_topk.source` names a directory holding `block_scan.py`, `indexer_top_k_varlen_util.py` and
`indexer_top_k_decode_varlen.py`: a cuDNN frontend CuTe DSL radix top-k modified to order equal
scores by ascending key id. Megatron Lite imports it as a private package (no `sys.path` change);
it needs the cuDNN frontend and the CUTLASS DSL. `precision="exact"` requires it. With
`precision="fast"` it is optional: without it the reference selector uses the cuDNN frontend radix
top-k (`cudnn.DSA`), which resolves keys tied at a row's cutoff score with atomics, so which of
them a row selects can differ from run to run. The validated files (pins for
`exact_topk.expected_sha256`):

| File | SHA-256 |
| --- | --- |
| `block_scan.py` | `f8ca2e276c9257637846e79e8c19f727e012b97363ffdb7748168319a9bfd184` |
| `indexer_top_k_decode_varlen.py` | `0c8accdee61cb6e7dc4c2dc4aeaf1a7e26b26435fb8f420e7b01d3f2dba53c09` |
| `indexer_top_k_varlen_util.py` | `a7b8eee675ca2702351d152f9561ede220db78f7cc7a8faef272d3e2e6e56c92` |

TODO(litetopk-public-link): the public location and license of the exact-tie top-k package.

### DeepGEMM

The reference selector of every backend other than `default` scores with
`deep_gemm.fp8_fp4_mqa_logits`, and the plugin loader requires the `deep_gemm` package (the
plugins build and run against it). DeepGEMM is not a Megatron Lite dependency; this integration
was validated with `sgl-deep-gemm` 0.1.3. The raw32 plugins advertise their exact route only when
the installed DeepGEMM is the build they were qualified against (`sgl-deep-gemm` 0.1.3, identified
by file hashes when the plugin is imported); with another DeepGEMM their route reports
`exact=False` and `precision="exact"` fails when the model is built.

### cuDNN frontend

The cuDNN frontend (`nvidia-cudnn-frontend`, a Megatron-LM dev dependency) provides the radix
top-k of the reference selector without `exact_topk`, and the exact-tie package imports its
compiler options. The optional GPU tests of this integration ran with cuDNN frontend 1.27.

### Hardware

LiteTopK needs an SM100 (Blackwell) GPU. The reference selector runs where DeepGEMM supports the
indexer's format and head count. Everything here was validated on B200 GPUs.

## Plugin ABI

Megatron Lite talks to a LiteTopK plugin through ABI version 1; a module whose
`LITETOPK_ABI_VERSION` is not `1` is rejected. The adapter exposes `plugin_info()`,
`load_extension(...)`, `production_min_s(use_fp4)`, `carry_vote_rows()`,
`begin_call(device, hot_key, sequence_length)`, `prepare_permuted_gather(...)`,
`try_large_exact_once_chunk(...)`, `stash_carry(...)`, `drop_carry(device, hot_key)` and
`release(device, *, release_scratch=True)`; keyword names are part of the ABI, and the loader
checks the signatures against the calls Megatron Lite makes.

- `plugin_info()` returns exactly the keys `abi`, `source_id`, `routes`, `effective_config`,
  `launch_time_env_keys`, `tie_policy` and `score_policy`. Each route declares its name
  (`fp8_paged` or `fp4_slab`), format, head counts, head dimensions, top-k sizes, query tile
  lengths, key range, HOT prefix, whether it selects exactly, and its tie and score policies.
- ABI v1 plugins read their configuration from `SGLANG_LITETOPK*` environment keys: the adapter
  snapshots them when it is imported and the CUDA extension reads a few of them on every launch.
  Megatron Lite renders its `LiteTopKPluginSettings` into these keys before the import, keeps
  them for the life of the process, and checks that the adapter snapshotted exactly the rendered
  values. A rendered key already set to another value fails the load, and so does any other
  `SGLANG_LITETOPK*` key (or known launch-time key) that no loaded plugin rendered, except the
  diagnostic keys. The launch-time keys are compared with their load-time values before every
  selection.
- One plugin source loads with one settings profile per process. The default settings of FP8
  (GLM-5) and MXFP4 (DeepSeek-V4) layers differ, so the two models select with LiteTopK in
  separate processes.
- Every tile writes one status code per row on the device: 0 valid, 1 or 2 a row whose candidates
  overflowed (that row is recomputed by the reference selector), 3 a failure (the whole tile is
  recomputed). A call reads the statuses once, after all its tiles.

## Precision

`precision="exact"`: every row selects the exact top-k of the matched-precision scores, the
float32 scores DeepGEMM `fp8_fp4_mqa_logits` computes from the quantized operands, ranked by score
descending and, for equal scores, by ascending key id. The reference selector meets this with the
exact-tie top-k. A LiteTopK route meets it only when it advertises exact selection: the raw32
plugins keep the unrounded float32 score of every candidate (as an order-preserving 32-bit code)
with its key id and rank the candidates by (score descending, key id ascending) themselves, on the
same operand bytes the reference selector scores. Exact FP8 selection also needs the plugin
settings `tie_policy="logical-id"` and `score_policy="native-fp32"`, which the policy seam derives
for it. Rows a plugin reports as overflowing or failed are recomputed by the reference selector,
so every row is exact.

`precision="fast"`: LiteTopK rows may differ from the exact selection among nearly tied keys, and
unless `exact_topk` is set, which of the keys tied at a reference row's cutoff score are selected
can differ from run to run. The optional GPU tests bound every key a fast route swaps to a relative
distance of 1e-3 (FP8) or 1.1e-3 (MXFP4) from the row's cutoff score. The MXFP4 slab route can swap
different near ties from run to run. The previous LiteTopK integration selected with fast routes
only; `glm-litetopk-996e-abi1` and `dsv4-litetopk-ac1c-abi1` carry its CUDA selectors.

With either precision every output row lists ascending key ids, followed by -1 for the missing
keys of a row that sees fewer than top-k keys. Neither precision reproduces the upstream
selectors bit for bit: those score the BF16 indexer operands, so selections can differ among keys
whose scores are close.

## Capabilities

The plugin routes declare their capabilities in `plugin_info()`; Megatron Lite hard-codes none:

| Plugin | Route | Indexer format | Heads | Head dim | Top-k | Keys per sequence | Exact |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `glm-litetopk-raw32-abi1` | `fp8_paged` | FP8 | 32 | 128 | 2048 | 196608 to 1048576 | yes |
| `glm-litetopk-raw32h64-abi1` | `fp8_paged` | FP8 | 32, 64 | 128 | 2048 | 196608 to 1048576 | yes |
| `glm-litetopk-996e-abi1` | `fp8_paged` | FP8 | 32 | 128 | 2048 | 196608 to 1048576 | no |
| `dsv4-litetopk-ac1c-abi1` | `fp4_slab` | MXFP4 | 32, 64 | 128 | 1 to 2048 | 65536 to 1048576 compressed keys | no |

- A layer whose head count the route has no kernels for fails when the model is built, unless
  `head_padding=True` lets it run on a larger head count of the route (see
  [Head counts](#head-counts-and-zero-head-padding)); so does a layer whose head dimension the
  route does not serve. Use `backend="reference"` for such a layer. A layer whose top-k the route
  does not select is selected by the reference selector alone.
- Within a sequence that has the route's minimum key count, LiteTopK tiles start at a fixed causal
  position, measured per operand format and kernel head count: FP8 tiles of the 32-head kernels
  8192 positions before the route's minimum key count (position 188416 with the routes above);
  FP8 tiles of the 64-head kernels nowhere by default, because no start was measured to keep
  LiteTopK at least as fast as the reference backend at every prompt length it would serve (see
  [Performance](#performance)); MXFP4 tiles at the position whose row sees 45056 compressed keys
  (position 180224). These are the starts
  that gave the fastest calls in the measurements behind the defaults; FP8 tiles hold 1776 rows
  (on 148 SMs) with 32 and with 64 kernel heads. Earlier rows, shorter sequences and the tiles the
  plugin declines go to the reference selector. `IndexerTopKInstallation.stats()` counts the rows
  each selector took and why a segment got no LiteTopK tile.
- Reference selector: `configure_indexer_topk` probes DeepGEMM for the layer's head count when the
  model is built and pads an unsupported count with zero query heads, which add exact zeros to
  every score. With `sgl-deep-gemm` 0.1.3 the score kernel accepts 16, 32 and 64 heads, for FP8
  and for MXFP4 operands. The exact-tie top-k selects up to 2048 keys per row among up to 2**20
  keys.

## Head counts and zero-head padding

An indexer ranks keys by a weighted sum over all its heads, so every selector of a layer scores
all of them, and score kernels exist for some head counts only. Megatron Lite hard-codes none:
when the model is built, `configure_indexer_topk` negotiates, for every layer, the head counts its
selectors score with, from the heads the plugin route declares in `plugin_info()` and from a probe
of the reference score kernel (one small DeepGEMM call per format, head count and device
architecture).

- LiteTopK runs a head count of its route as is. With `head_padding=True` it runs a head count
  that is a multiple of four and below the route's largest one on the route's next larger head
  count, with zero heads appended (16 heads on 32-head kernels, 48 on 64). Any other head count,
  or such a head count without `head_padding`, fails when the model is built; the message gives
  the padded head count and its cost.
- The reference selector pads where DeepGEMM has no kernel for the head count, to the next head
  count it has, with or without `head_padding`. While LiteTopK pads, the reference selector scores
  the plugin's padded operands, so that the rows it selects, recomputes and votes seeds with are
  scored from exactly the plugin's bytes, which `precision="exact"` relies on; with
  `precision="fast"` and a head count DeepGEMM supports, it keeps the layer's own head count. When
  the plan gives LiteTopK no row, it scores as the reference backend does.

With the FP8 route of `glm-litetopk-raw32h64-abi1` or the MXFP4 route of `dsv4-litetopk-ac1c-abi1`
(kernels for 32 and 64 heads) and `sgl-deep-gemm` 0.1.3 (16, 32 and 64 heads):

| Indexer heads | LiteTopK, `head_padding=False` | LiteTopK, `head_padding=True` | Reference backend |
| --- | --- | --- | --- |
| 4, 8, 12 | error, suggests padding to 32 | 32 heads | 16 heads |
| 16 | error, suggests padding to 32 | 32 heads | 16 heads |
| 20 | error, suggests padding to 32 | 32 heads | 32 heads |
| 32 | 32 heads | 32 heads | 32 heads |
| 48 | error, suggests padding to 64 | 64 heads | 64 heads |
| 64 | 64 heads | 64 heads | 64 heads |
| 96, 128 | error (the reference selector cannot score them either) | error | error |

`glm-litetopk-raw32-abi1` and `glm-litetopk-996e-abi1` have FP8 kernels for 32 heads only: they
serve 4 to 28 heads with padding, and no head count above 32.

Zero heads are appended after quantization: FP8 query codes 0 and folded weights 0; MXFP4 query
codes 0, group scales 127 (the UE8M0 code of 1.0; 255 encodes NaN) and weights 0. They change no
score value. The score kernels sum the weighted heads in four float32 FMA chains that start at +0
(heads `j` with equal `j % 4`, in head order, rounded to nearest), the zero heads end every chain,
and adding a zero leaves a partial sum bit for bit unchanged unless it is -0. A chain holds -0 only
after a negative product below half the smallest float32 subnormal rounded to zero, which needs a
product below 2**-102 in magnitude; then a score of -0 can become +0, an equal value with another
bit pattern.
The CPU tests check this with an exact simulation of the chains; the GPU tests show that DeepGEMM
scores 16 heads padded to 32 or 64 bit for bit like 16 heads and selects the same keys.

Padding multiplies the scoring work by the kernel heads over the layer's heads (2.00x for 16
heads on 32-head kernels, 1.33x for 48 on 64); the key reads do not change. Measured cold on one
B200 with 16 heads at 512K tokens: the reference selector took 673 ms with DeepGEMM's 16-head
kernel and 872 ms padded to 32 heads (+29.7%; its top-k does not grow), and LiteTopK on 32-head
kernels (tiles from position 188416, its reference rows padded as well) took 702 ms, 4.7% more
than the 16-head reference backend; all three selected the same keys.

The start positions of LiteTopK tiles above were measured against a reference selector that
scores as many heads as LiteTopK. When LiteTopK pads and the reference backend scores the layer
with fewer heads (16 heads on 32-head kernels, as DeepGEMM has 16-head kernels), LiteTopK has the
more work, as measured above, and the default plan gives LiteTopK no row of the layer;
`IndexerTopKTuning.startup_position` enables it. Padded layers whose reference selector pads to
the same head count (20 heads on 32, 48 on 64) take the start position of their kernel heads,
carried over from the unpadded measurements (both selectors score the same heads there too); these
padded head counts were not timed. A tuning with `required=True` fails when the model is built if
the default plan gives a layer no LiteTopK row.

The negotiated head counts are recorded in `IndexerTopKBinding.heads`, an `IndexerHeads` (the
indexer heads, the LiteTopK and reference kernel heads, the head count the reference selector
needs on its own, and whether LiteTopK and the reference selector pad), and per layer in
`IndexerTopKInstallation.heads()`; `IndexerTopKStats` counts the rows each selector scored with
zero heads (`padded_litetopk_rows`, `padded_reference_rows`), and every binding logs its kernel
head counts with its plan at its first selection on a device.

## Context parallelism

Selection issues no collective: under context parallelism every rank selects its local rows
against the keys the model has already gathered, so a rank whose plugin declines or fails
recomputes locally and every rank issues the same collectives as before. A call is planned per
segment (a run of local rows of one sequence) on the host, from the rank's layout: DSA native
context parallelism (contiguous layout) describes the rank's rows as a contiguous slice of one
sequence or of the packed sequences, and builds no dense causal mask for a bound layer; the CSA
THD path describes them as packed segments with sequence-relative ids. Each rank therefore plans
its own LiteTopK tiles; ranks whose rows lie before the start position select with the reference
selector only. With `dsa_cp_mode="legacy_gather_all"` every rank gathers the whole sequence and
selects all of its rows. There is no context-parallel configuration.

CSA's THD binding returns sequence-relative compressed-key ids. It does not add the BSHD
KV offset: Core's existing `build_attention_indices` maps those ids into the rank-major KV
layout after compression. Each completed group of four tokens exposes one compressed key;
the first three tokens of a sequence therefore have no compressed key. Bound CSA bypasses
only `compute_cp_indexer_topk`, retaining its boundary exchange, gathers and attention-index
construction. Compression ratios other than four have no indexer binding.

## Lifecycle and memory

The reference radix selector reads DeepGEMM's padded score views directly when the data pointer
and row stride are 32-byte aligned and columns are contiguous. It copies incompatible views
before calling cuDNN. This removes the extra full score-buffer copy in the common aligned case
without changing scores, visible lengths or selected keys.

- A selection call is self-contained: there is no request scope, and the plugin state a call
  creates (its seed carries) is dropped when the call ends, also on error.
- `IndexerTopKInstallation.release(device=None)` (or `release_indexer_topk_workspaces(device)`)
  frees the pooled key caches and the scratch memory of every loaded plugin; later selections
  allocate them again. Neither loads anything.
- Loaded plugins and the environment keys rendered for them stay for the life of the process.
- Calling `configure_indexer_topk` again on the model chunks replaces every binding; passing None
  or `backend="default"` unbinds every layer. Bindings are not modules: they hold no parameter,
  never enter a `state_dict`, and copies of a model share them.
- `IndexerTopKInstallation.stats()` and `reset_stats()` give per-layer counters; `plugin_info`
  records what was loaded (location, hashes, rendered environment, `plugin_info()`).

## Performance

Measured on one B200 with cold per-call timing (L2 flushed before every call), on identical
operands, the reference backend's selection time divided by the LiteTopK backend's, with the
default plan:

- GLM-5.2 indexer inputs, FP8, `precision="exact"`, `glm-litetopk-raw32-abi1`: 1.108 at 512K and
  1.278 at 1M tokens (layer 0: real indexer weights on a reconstruction of its input); 1.094
  summed over six layers captured in the model at 256K, of which layer 0 alone is 0.984; 0.769 to
  0.822 on four 256K inputs with real weights and embedding-level proxy activations, which give
  LiteTopK more candidates per row (a mean of 9.2K to 10.9K, against 3.3K to 7.3K in the captured
  layers).
- DeepSeek-V4 indexer inputs, MXFP4, `precision="fast"`, `dsv4-litetopk-ac1c-abi1`: 1.009 to 1.024
  at 256K tokens (two layers captured in the model, two synthetic inputs) and 1.083 to 1.098 at
  512K tokens (two synthetic inputs).
- FP8 indexers with 64 heads, `precision="exact"`, `glm-litetopk-raw32h64-abi1`, on synthetic
  inputs (structured, strict-gap) and on GLM-5.2 layer 0 with real weights and its 32 heads
  duplicated, with explicit start positions (the default plan gives these layers no LiteTopK
  row): tiles from position 188416, 0.986 (real weights) and 1.024 (structured) at 512K tokens,
  1.035 at 768K (structured) and 1.011 to 1.059 at 1M; tiles from position 524288 or 655360,
  0.973 to 1.025 at 768K (strict-gap below 1) and 1.025 to 1.054 at 1M; tiles from position
  786432 (none at 768K), 1.035 to 1.041 at 1M. The reference backend scores fewer rows per
  DeepGEMM call as sequences get longer (whole score-kernel waves within its 2 GiB budget of
  float32 scores: 888 rows up to 604575 keys, 592 up to 906862, 296 above). Its rows therefore
  cost 1.5% to 6.2% more at 1M than at the same positions of a 768K call, and the LiteTopK backend
  scores its reference rows in a narrower call with more rows per DeepGEMM call; much of
  LiteTopK's advantage at 1M comes from that. Up to 906862 tokens the reference backend scores
  as at 768K, where the strict-gap tiles took 5.7% to 6.2% longer than the reference rows they
  replace; prompts of 786433 to 906862 tokens were not measured. For 1M-token prompts
  `IndexerTopKTuning.startup_position` (for example 524288) enables LiteTopK.

`precision="exact"` costs more than the fast selection of the previous LiteTopK integration
(historical fast: `glm-litetopk-996e-abi1` with that integration's plan and settings), measured the
same way on the GLM-5.2 inputs above: the LiteTopK backend took 5.2% longer summed over the six
captured 256K layers (3.8% to 6.5% per layer), 4.9% to 8.0% longer on synthetic 256K and 512K
inputs, 2.3% to 3.9% on the proxy inputs, 2.5% and 1.2% on real-weight layer 0 at 512K and 1M,
and 3.0% on a synthetic 1M input. Most of this is the ascending sort of every output row that
exact selection needs (15 ms per 256K call, about 30 ms at 512K, about 65 ms at 1M); the
reference backend sorts as well, which makes its calls 3.3% to 7.2% slower at 256K and 512K and
1.6% to 1.7% slower at 1M than reference calls without the sort.

## Limitations

- Bindings select only in eval mode with gradients disabled. Training forwards always use the
  upstream selectors, also when they run without autograd: Lite's reentrant activation recompute
  runs a training forward under `torch.no_grad()` and again with gradients in the backward pass,
  and both runs must select the same top-k. The indexer loss therefore never sees a binding.
- The native GLM-5 and DeepSeek-V4 protocols reject tensor parallelism (`tp > 1`, `etp > 1`); a
  binding needs every indexer head of a layer, because the top-k ranks the sum over all heads.
- Selection under CUDA graph capture raises: tiles are planned on the host.
- A dense batch of several sequences (a batch dimension above 1) keeps the upstream selector;
  packed THD batches select through the bindings.
- LiteTopK serves long prompts only (see [Capabilities](#capabilities)); its benefit is for
  long-context prefill.

## Errors

| Error | When | Remedy |
| --- | --- | --- |
| `IndexerTopKConfigError: indexer_topk has unknown keys [...]` (also `indexer_topk.litetopk` and `indexer_topk.exact_topk`) | `ImplConfig` | Use only the listed keys; tuning values are not model configuration |
| `IndexerTopKConfigError: indexer_topk.backend must be one of 'default', 'reference', 'litetopk'` (or `precision`) | `ImplConfig` | Fix the value |
| `IndexerTopKConfigError: indexer_topk.litetopk.source is required: LiteTopK kernels are not bundled with Megatron Lite` | `ImplConfig` | Set `litetopk.source` (see [Dependencies](#dependencies)) or use `backend="reference"` |
| `IndexerTopKConfigError: indexer_topk.exact_topk.source is required for precision='exact'` | `ImplConfig` | Set `exact_topk.source` or use `precision="fast"` |
| `IndexerTopKConfigError: LiteTopK source ... has no fp8_paged route` (or `fp4_slab`) | `build_model` | Use the plugin of the model's format (FP8: GLM-5, MXFP4: DeepSeek-V4) |
| `IndexerTopKConfigError: LiteTopK route ... supports indexer heads [...], head_dim [...], topk ...; layer ... has H=..., D=..., K=.... Set indexer_topk.head_padding=True to pad heads to ... (...x scoring work) or use backend='reference'.` | `build_model` | `head_padding=True` (see [Head counts](#head-counts-and-zero-head-padding)) or `backend="reference"`; without the padding sentence, `backend="reference"`, or `backend="default"` when the message says the reference score kernel cannot score the head count either |
| `IndexerTopKConfigError: precision='exact' needs a LiteTopK route that advertises exact selection` | `build_model` | Use an exact plugin (raw32) with the qualified DeepGEMM, or `precision="fast"` |
| `IndexerTopKConfigError: precision='exact' scores the rows LiteTopK does not cover on the plugin's operands, but the reference score kernel cannot score ... heads` | `build_model` | `precision="fast"`, or a plugin whose head counts DeepGEMM scores |
| `IndexerTopKConfigError: IndexerTopKTuning.required needs LiteTopK rows, but the default plan gives LiteTopK none ...` | `build_model` | Set `IndexerTopKTuning.startup_position`, or drop `required` |
| `IndexerTopKConfigError: DeepGEMM fp8_fp4_mqa_logits supports no head count from ...` | `build_model` | A DeepGEMM that supports the head count |
| `IndexerTopKRuntimeError: the matched-precision indexer top-k reference selector needs DeepGEMM` | `build_model` | Install DeepGEMM or keep `backend="default"` |
| `IndexerTopKRuntimeError: the indexer top-k reference selector without an exact_topk package needs the cuDNN frontend ...` | `build_model` | Install the cuDNN frontend or set `exact_topk` |
| `IndexerTopKPluginError: LiteTopK plugin at ... cannot be loaded: ...` | `build_model` | Install what is listed (plugin files, `deep_gemm`, prebuilt extension) |
| `IndexerTopKPluginError: LiteTopK plugin at ...: load_extension() failed: RuntimeError: LiteTopK prebuilt basename must be sglang_litetopk_dsa_b200_production_<source id>.so, got ...` | `build_model` | Keep the plugin's file name for its prebuilt extension (see [LiteTopK plugins](#litetopk-plugins)) |
| `IndexerTopKPluginError: LiteTopK plugin at ...: source id is ..., expected ...` (or `adapter sha256`, `prebuilt extension sha256`) | `build_model` | The plugin differs from its pins: use the pinned files or update the pins |
| `IndexerTopKPluginError: LiteTopK plugin at ... exposes ABI ...` | `build_model` | Use an ABI v1 plugin (see [Plugin ABI](#plugin-abi)) |
| `IndexerTopKPluginError: SGLANG_LITETOPK_...=... is already set in the process but the plugin settings need ...`, or `the process sets ..., which the plugin settings do not render` | `build_model` | Unset the `SGLANG_LITETOPK*` keys; Megatron Lite renders them |
| `IndexerTopKPluginError: LiteTopK source ... is already loaded in this process with different settings` | `build_model` | One settings profile per plugin source per process; run GLM-5 and DeepSeek-V4 LiteTopK jobs in separate processes |
| `IndexerTopKPluginError: exact top-k package at ...` | `build_model` | Fix the path or the pins; install the cuDNN frontend and the CUTLASS DSL it imports |
| `IndexerTopKRuntimeError: indexer top-k binding cannot run under CUDA graph capture` | forward | Run the forward outside graph capture |
| `IndexerTopKRuntimeError: LiteTopK requires an SM100 (Blackwell) GPU` | first forward | Use a Blackwell GPU or `backend="reference"` |
| `IndexerTopKRuntimeError: SGLANG_LITETOPK_... changed from ... to ... after LiteTopK source ... was loaded` | forward | Leave the plugin's environment keys unchanged for the life of the process |

## Optional tests

The CPU tests run in the standard workflow (`experimental/lite/tests/run_tests.sh`). The GPU tests
are marked `optional` and run only when their paths are given. The tests that need a plugin or the
exact-tie package skip unless its location is given as JSON, inline or as the path of a JSON file:

- `LITETOPK_TEST_SELECTORS`: a list of `{"native_format": "fp8" | "mxfp4", "precision": ...,
  "heads": ..., "topk": ..., "litetopk": {<LiteTopKPluginConfig fields>}, "plugin_settings":
  {<LiteTopKPluginSettings fields>}}` (`plugin_settings` optional);
- `LITETOPK_TEST_EXACT_TOPK`: `{<ExactTopKConfig fields>, "pythonpath": [...]}`, where the optional
  `pythonpath` entries are prepended in the test process (for example a cuDNN frontend);
- `LITETOPK_TEST_PLUGINS`: a list of `{<LiteTopKPluginConfig fields>, "settings": {...}}`;
- `LITETOPK_TEST_CP_INPUTS`: optional real FP8 operands for the CP tests, a list of
  `{"path": ..., "softmax_scale": ...}`.

```bash
LITETOPK_TEST_SELECTORS=/path/to/selectors.json \
LITETOPK_TEST_EXACT_TOPK=/path/to/exact_topk.json \
experimental/lite/tests/run_tests.sh \
    experimental/lite/tests/smoke/primitive/indexer_topk/test_selector_gpu.py \
    experimental/lite/tests/smoke/primitive/indexer_topk/test_cp_sim_gpu.py
```

| Tests (`experimental/lite/tests/`) | GPUs | Needs |
| --- | --- | --- |
| `unit/primitive/kernels/indexer_topk/`, `unit/primitive/modules/attention/test_indexer_topk_dispatch_unit.py`, `unit/primitive/test_dsa_cp_indexer_topk_unit.py`, `unit/model/test_indexer_topk_impl_config.py` | CPU | nothing (fake plugins and kernels) |
| `smoke/primitive/indexer_topk/test_quant_order_gpu.py` | 1 | Triton |
| `smoke/primitive/indexer_topk/test_reference_gpu.py` | 1 | DeepGEMM; `LITETOPK_TEST_EXACT_TOPK` for its exact-tie tests; the cuDNN frontend for its radix top-k test |
| `smoke/primitive/indexer_topk/test_dispatch_gpu.py`, `test_dsa_native_cp_gpu.py` | 1 | DeepGEMM, cuDNN frontend 1.27 |
| `smoke/model/glm5/lite/test_glm5_lite_cp_smoke.py` (the `indexer_topk` case) | 2 | DeepGEMM, cuDNN frontend |
| `smoke/primitive/indexer_topk/test_plugins_gpu.py` | 1 | `LITETOPK_TEST_PLUGINS`; `LITETOPK_TEST_EXACT_TOPK` for its exact-tie test |
| `smoke/primitive/indexer_topk/test_selector_gpu.py`, `test_cp_sim_gpu.py` | 1 | `LITETOPK_TEST_SELECTORS`, `LITETOPK_TEST_EXACT_TOPK` |
| `smoke/primitive/indexer_topk/test_heads_gpu.py` | 1 | DeepGEMM, `LITETOPK_TEST_EXACT_TOPK`; the FP8 LiteTopK tests need an exact FP8 entry of `LITETOPK_TEST_SELECTORS` and the MXFP4 padding test a fast MXFP4 entry; each skips an entry whose route lacks the kernels it needs (64-head FP8 kernels as in `glm-litetopk-raw32h64-abi1`, the 32-head FP8 kernels that 16 heads pad to, or the 64-head MXFP4 kernels that 48 heads pad to) |
| `smoke/primitive/indexer_topk/test_cp_modules_gpu.py` | 2 and 4 | an exact FP8 entry of `LITETOPK_TEST_SELECTORS`, `LITETOPK_TEST_EXACT_TOPK` |
| `smoke/primitive/indexer_topk/test_csa_cp_modules_gpu.py` | 4 | an MXFP4 entry of `LITETOPK_TEST_SELECTORS`, `LITETOPK_TEST_EXACT_TOPK` |
| `smoke/primitive/indexer_topk/test_csa_protocol_gpu.py` | 2 | DeepGEMM, cuDNN frontend; builds the DeepSeek-V4 protocol with C4 and C128 layers |

The single-GPU plugin tests run every entry in a fresh interpreter, because plugin settings are
process-wide.
