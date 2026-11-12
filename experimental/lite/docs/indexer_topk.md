# Indexer top-k selection in Megatron Lite

The sparse attention layers of the native `glm5` model (DSA) select, for every query token, the
`index_topk` keys its attention reads: an indexer scores the keys the token sees and keeps the
best ones. By default every layer runs its upstream selector. `ImplConfig.indexer_topk` installs
an indexer top-k binding on every such layer instead; in eval mode with gradients disabled, the
binding selects with:

- the matched-precision reference selector: the indexer queries and keys are quantized to the
  model's indexer format (FP8 E4M3 rows with float32 scales), DeepGEMM `fp8_fp4_mqa_logits`
  scores every visible key in float32, and a top-k kernel keeps the best keys of each row. Rows
  are scored in chunks whose scores fit a byte budget, so a long prompt never holds the scores of
  all its rows at once;
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

- `ImplConfig` validates the field when it is built: unknown keys and invalid values or
  combinations raise `IndexerTopKConfigError` (a `ValueError`). It keeps the value as given.
- `build_model` hands the field to `configure_indexer_topk` right after it builds the model
  chunks, with the model's indexer format (GLM-5: `native_format="fp8"`), and puts the result
  in `ModelBundle.extras["indexer_topk"]`: an `IndexerTopKInstallation`, or None for
  `backend="default"`. Unset, `build_model` does not touch the layers, adds no `extras` key and
  imports nothing of the indexer top-k package.
- `configure_indexer_topk` binds every DSA layer that selects its own top-k (IndexShare shared
  layers reuse the top-k of their source layer and stay unbound). It validates every layer before
  it binds any: it probes the reference score kernel for the layer's head count and, for
  `backend="litetopk"`, loads the plugin and negotiates its route, so an unusable configuration
  fails when the model is built.
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
| `glm-litetopk-996e-abi1` | `996e735c52df` | `37ff4143292871e0293c6e388ee96c90549c4fe60edf24dc9498e0bc165c2051` | `fp8_paged`, fast | GLM-5, the previous integration's selector |

The previous integration, in this document, is the earlier out-of-tree integration of LiteTopK
into a Megatron-LM fork; it selected with fast FP8 routes only, and `glm-litetopk-996e-abi1`
carries its CUDA selector.

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
  (`fp8_paged`, the only route name this version accepts), format, head counts, head
  dimensions, top-k sizes, query tile lengths, key range, HOT prefix, whether it selects exactly,
  and its tie and score policies.
- ABI v1 plugins read their configuration from `SGLANG_LITETOPK*` environment keys: the adapter
  snapshots them when it is imported and the CUDA extension reads a few of them on every launch.
  Megatron Lite renders its `LiteTopKPluginSettings` into these keys before the import, keeps
  them for the life of the process, and checks that the adapter snapshotted exactly the rendered
  values. A rendered key already set to another value fails the load, and so does any other
  `SGLANG_LITETOPK*` key (or known launch-time key) that no loaded plugin rendered, except the
  diagnostic keys. The launch-time keys are compared with their load-time values before every
  selection.
- One plugin source loads with one settings profile per process.
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
distance of 1e-3 from the row's cutoff score. The previous LiteTopK integration selected with
fast routes only; `glm-litetopk-996e-abi1` carries its CUDA selector.

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

- A layer whose head count or head dimension the route does not serve fails when the model is
  built; use `backend="reference"` for it. A layer whose top-k the route does not select is
  selected by the reference selector alone.
- Within a sequence that has the route's minimum key count, LiteTopK tiles start at a fixed causal
  position, measured for 32 kernel heads: FP8 tiles 8192 positions before the route's minimum
  key count (position 188416 with the routes above), the start that gave the fastest calls in
  the measurements behind this default. For any other kernel head count (64 heads with
  `glm-litetopk-raw32h64-abi1`) the default plan gives LiteTopK no row, because no start was
  measured for it (see [Performance](#performance)); `IndexerTopKTuning.startup_position`
  enables it. Earlier rows, shorter sequences and the tiles the plugin declines go to the
  reference selector. `IndexerTopKInstallation.stats()` counts the rows each selector took and
  why a segment got no LiteTopK tile.
- Reference selector: `configure_indexer_topk` probes DeepGEMM for the layer's head count when the
  model is built and pads an unsupported count with zero-weight query heads, which add exact zeros
  to every score. With `sgl-deep-gemm` 0.1.3 the FP8 score kernel accepts 16, 32 and 64 heads.
  The exact-tie top-k selects up to 2048 keys per row among up to 2**20 keys.

## Context parallelism

Selection issues no collective: under context parallelism every rank selects its local rows
against the keys the model has already gathered, so a rank whose plugin declines or fails
recomputes locally and every rank issues the same collectives as before. A call is planned per
segment (a run of local rows of one sequence) on the host, from the rank's layout: DSA native
context parallelism (contiguous layout) describes the rank's rows as a contiguous slice of one
sequence or of the packed sequences, and builds no dense causal mask for a bound layer. Each rank
therefore plans its own LiteTopK tiles; ranks whose rows lie before the start position select
with the reference selector only. With `dsa_cp_mode="legacy_gather_all"` every rank gathers the
whole sequence and selects all of its rows. There is no context-parallel configuration.

## Lifecycle and memory

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
- FP8 indexers with 64 heads, `precision="exact"`, `glm-litetopk-raw32h64-abi1`, with the start
  position of the 32-head kernels set explicitly (the default plan gives these layers no
  LiteTopK row; tiles of three score-kernel waves, 888 rows on 148 SMs, from position
  188416), on GLM-5.2 layer 0 with real weights and its 32 heads duplicated and
  on synthetic inputs: 0.967 (real weights, so LiteTopK is slower) and 1.022 (synthetic) at 512K
  tokens, 1.032 at 768K (synthetic) and 1.005 to 1.058 at 1M; 256K was not measured. In the
  plugin's own harness, which times only the LiteTopK tiles (1776 rows) against the reference
  selector on the same rows, the tiles took 0.5% to 11.4% longer at 256K and 512K tokens on
  synthetic inputs.

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
- The native GLM-5 protocol rejects tensor parallelism (`tp > 1`, `etp > 1`); a binding needs
  every indexer head of a layer, because the top-k ranks the sum over all heads.
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
| `IndexerTopKConfigError: LiteTopK source ... has no fp8_paged route` | `build_model` | Use a plugin with an `fp8_paged` route |
| `IndexerTopKConfigError: LiteTopK route ... supports indexer heads [...], head_dim [...]; layer ... has H=..., D=....` | `build_model` | `backend="reference"` |
| `IndexerTopKConfigError: precision='exact' needs a LiteTopK route that advertises exact selection` | `build_model` | Use an exact plugin (raw32) with the qualified DeepGEMM, or `precision="fast"` |
| `IndexerTopKConfigError: IndexerTopKTuning.required needs LiteTopK rows, but the default plan gives LiteTopK none ...` | `build_model` | Set `IndexerTopKTuning.startup_position`, or drop `required` |
| `IndexerTopKConfigError: DeepGEMM fp8_fp4_mqa_logits supports no head count from ...` | `build_model` | A DeepGEMM that supports the head count |
| `IndexerTopKRuntimeError: the matched-precision indexer top-k reference selector needs DeepGEMM` | `build_model` | Install DeepGEMM or keep `backend="default"` |
| `IndexerTopKRuntimeError: the indexer top-k reference selector without an exact_topk package needs the cuDNN frontend ...` | `build_model` | Install the cuDNN frontend or set `exact_topk` |
| `IndexerTopKPluginError: LiteTopK plugin at ... cannot be loaded: ...` | `build_model` | Install what is listed (plugin files, `deep_gemm`, prebuilt extension) |
| `IndexerTopKPluginError: LiteTopK plugin at ...: load_extension() failed: RuntimeError: LiteTopK prebuilt basename must be sglang_litetopk_dsa_b200_production_<source id>.so, got ...` | `build_model` | Keep the plugin's file name for its prebuilt extension (see [LiteTopK plugins](#litetopk-plugins)) |
| `IndexerTopKPluginError: LiteTopK plugin at ...: source id is ..., expected ...` (or `adapter sha256`, `prebuilt extension sha256`) | `build_model` | The plugin differs from its pins: use the pinned files or update the pins |
| `IndexerTopKPluginError: LiteTopK plugin at ... exposes ABI ...` | `build_model` | Use an ABI v1 plugin (see [Plugin ABI](#plugin-abi)) |
| `IndexerTopKPluginError: LiteTopK plugin at ...: plugin_info()['routes'][...]['name'] must be one of ['fp8_paged']` | `build_model` | The plugin declares a route this version does not serve: use a plugin whose routes are `fp8_paged` |
| `IndexerTopKPluginError: SGLANG_LITETOPK_...=... is already set in the process but the plugin settings need ...`, or `the process sets ..., which the plugin settings do not render` | `build_model` | Unset the `SGLANG_LITETOPK*` keys; Megatron Lite renders them |
| `IndexerTopKPluginError: LiteTopK source ... is already loaded in this process with different settings` | `build_model` | One settings profile per plugin source per process |
| `IndexerTopKPluginError: exact top-k package at ...` | `build_model` | Fix the path or the pins; install the cuDNN frontend and the CUTLASS DSL it imports |
| `IndexerTopKRuntimeError: indexer top-k binding cannot run under CUDA graph capture` | forward | Run the forward outside graph capture |
| `IndexerTopKRuntimeError: LiteTopK requires an SM100 (Blackwell) GPU` | first forward | Use a Blackwell GPU or `backend="reference"` |
| `IndexerTopKRuntimeError: SGLANG_LITETOPK_... changed from ... to ... after LiteTopK source ... was loaded` | forward | Leave the plugin's environment keys unchanged for the life of the process |

## Optional tests

The CPU tests run in the standard workflow (`experimental/lite/tests/run_tests.sh`). The GPU tests
are marked `optional` and run only when their paths are given. The tests that need a plugin or the
exact-tie package skip unless its location is given as JSON, inline or as the path of a JSON file:

- `LITETOPK_TEST_SELECTORS`: a list of `{"native_format": "fp8", "precision": ...,
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
| `smoke/primitive/indexer_topk/test_cp_modules_gpu.py` | 2 and 4 | an exact FP8 entry of `LITETOPK_TEST_SELECTORS`, `LITETOPK_TEST_EXACT_TOPK` |

The single-GPU plugin tests run every entry in a fresh interpreter, because plugin settings are
process-wide.
