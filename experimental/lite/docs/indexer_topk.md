# Indexer top-k selection in Megatron Lite

The sparse attention layers of the native `glm5` model (DSA) select, for every query token, the
`index_topk` keys its attention reads: an indexer scores the keys the token sees and keeps the
best ones. The package `megatron.lite.primitive.kernels.indexer_topk` holds the building blocks
of optional indexer top-k selectors for these layers:

- the FP8 quantizer of the indexer operands and the row-order kernels (`quant.py`, `order.py`),
  so that every selector consumes byte-identical operands and returns its rows in a
  deterministic order;
- the matched-precision reference selector (`reference.py`, `reference_topk`): the indexer
  queries and keys are quantized to FP8 E4M3 rows with float32 scales, DeepGEMM
  `fp8_fp4_mqa_logits` scores every visible key in float32, and a top-k kernel keeps the best
  keys of each row. Rows are scored in chunks whose scores fit a byte budget, so a long prompt
  never holds the scores of all its rows at once;
- the query layouts of a selection call (`layout.py`) and the tile planner (`planner.py`) of
  LiteTopK, an external CUDA selector, with the tuning policy seam
  `resolve_indexer_topk_tuning` (`config.py`);
- the explicit-path loaders of the LiteTopK plugins and of the exact-tie top-k package
  (`plugins/`), and the configuration and error types of these dependencies (`config.py`).

Nothing in Megatron Lite imports the package yet, so it changes no model. The LiteTopK plugins,
the exact-tie top-k package and DeepGEMM are optional dependencies that are not installed with
Megatron Lite; see [Dependencies](#dependencies).

## Dependencies

### LiteTopK plugins

A plugin is a directory holding the adapter module `litetopk.py` and its CUDA sources
`litetopk_kernels/{dsa_litetopk.cu, sm100_dsa_litetopk.cuh, dense_topk_litetopk.cuh}`. Its CUDA
extension is a prebuilt shared library (`prebuilt_extension`) or a JIT build of those sources
(optionally into `build_dir`, with the DeepGEMM headers of `deepgemm_include_dir`). A prebuilt
extension must keep the file name the plugin builds,
`sglang_litetopk_dsa_b200_production_<source id>.so` (after resolving symbolic links): every
plugin below checks it and fails to load a file of another name. `load_litetopk_plugin` loads a
plugin only from the path of its `LiteTopKPluginConfig`, never from environment variables, and
compares its pins before any plugin code runs:

- `expected_source_id`: the first 12 hex digits of the SHA-256 over the name and then the bytes of
  each of the three CUDA files, in the order above (the plugins' own source id);
- `expected_adapter_sha256`: the SHA-256 of `litetopk.py`, which the source id does not cover;
- `prebuilt_extension_sha256`: the SHA-256 of the prebuilt extension.

Every pin is optional. Unpinned values are computed, logged once as a warning and recorded in the
load provenance (`LoadedLiteTopKPlugin.provenance()`). The plugins these building blocks were
validated with:

| Plugin | Source id | `litetopk.py` SHA-256 | Route | Use |
| --- | --- | --- | --- | --- |
| `glm-litetopk-raw32-abi1` | `83669db87b20` | `46db6d898e4b4487e502f0687df7bb22ce075869667b13bf0c43a8e2a0e27bd2` | `fp8_paged`, exact | GLM-5 (32 FP8 indexer heads), exact |
| `glm-litetopk-raw32h64-abi1` | `7e5eb835fb7f` | `65951346a8c60d44945f1e7f03c5701e9c7e034f27d5274ee727a91d1a64aaac` | `fp8_paged`, exact | FP8 indexers with 32 or 64 heads, exact |
| `glm-litetopk-996e-abi1` | `996e735c52df` | `37ff4143292871e0293c6e388ee96c90549c4fe60edf24dc9498e0bc165c2051` | `fp8_paged`, fast | GLM-5, the previous integration's selector |

The previous integration, in this document, is the earlier out-of-tree integration of LiteTopK
into a Megatron-LM fork; it selected with fast FP8 routes only, and `glm-litetopk-996e-abi1`
carries its CUDA selector.

These plugins and the exact-tie top-k package below are external (ABI version 1) and not yet
publicly released. Until they are, the reference selector with the cuDNN frontend radix top-k
(no exact-tie package) is the only selector of this package that runs with public
dependencies, and the GPU tests that need a plugin or the exact-tie package skip.

TODO(litetopk-public-link): the public location and license of the LiteTopK plugins.

### Exact-tie top-k

`ExactTopKConfig.source` names a directory holding `block_scan.py`, `indexer_top_k_varlen_util.py`
and `indexer_top_k_decode_varlen.py`: a cuDNN frontend CuTe DSL radix top-k modified to order
equal scores by ascending key id. `load_exact_topk` imports it as a private package (no
`sys.path` change); it needs the cuDNN frontend and the CUTLASS DSL. With it, the reference
selector's rows are the exact top-k of its float32 scores (score descending, lower key id first
on equal scores). Without it the reference selector uses the cuDNN frontend radix top-k
(`cudnn.DSA`), which resolves keys tied at a row's cutoff score with atomics, so which of them a
row selects can differ from run to run. The validated files (pins for
`ExactTopKConfig.expected_sha256`):

| File | SHA-256 |
| --- | --- |
| `block_scan.py` | `f8ca2e276c9257637846e79e8c19f727e012b97363ffdb7748168319a9bfd184` |
| `indexer_top_k_decode_varlen.py` | `0c8accdee61cb6e7dc4c2dc4aeaf1a7e26b26435fb8f420e7b01d3f2dba53c09` |
| `indexer_top_k_varlen_util.py` | `a7b8eee675ca2702351d152f9561ede220db78f7cc7a8faef272d3e2e6e56c92` |

TODO(litetopk-public-link): the public location and license of the exact-tie top-k package.

### DeepGEMM

The reference selector scores with `deep_gemm.fp8_fp4_mqa_logits`, and the plugin loader requires
the `deep_gemm` package (the plugins build and run against it). DeepGEMM is not a Megatron Lite
dependency; these building blocks were validated with `sgl-deep-gemm` 0.1.3. The raw32 plugins
advertise their exact route only when the installed DeepGEMM is the build they were qualified
against (`sgl-deep-gemm` 0.1.3, identified by file hashes when the plugin is imported); with
another DeepGEMM their route reports `exact=False`.

### cuDNN frontend

The cuDNN frontend (`nvidia-cudnn-frontend`, a Megatron-LM dev dependency) provides the radix
top-k of the reference selector without the exact-tie package, and the exact-tie package imports
its compiler options. The optional GPU tests ran with cuDNN frontend 1.27.

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

## Optional tests

The CPU tests run in the standard workflow (`experimental/lite/tests/run_tests.sh`). The GPU tests
are marked `optional`. The tests that need a plugin or the exact-tie package skip unless its
location is given as JSON, inline or as the path of a JSON file:

- `LITETOPK_TEST_PLUGINS`: a list of `{<LiteTopKPluginConfig fields>, "settings": {...}}`;
- `LITETOPK_TEST_EXACT_TOPK`: `{<ExactTopKConfig fields>, "pythonpath": [...]}`, where the optional
  `pythonpath` entries are prepended in the test process (for example a cuDNN frontend).

| Tests (`experimental/lite/tests/`) | GPUs | Needs |
| --- | --- | --- |
| `unit/primitive/kernels/indexer_topk/` | CPU | nothing (fake plugins and kernels) |
| `smoke/primitive/indexer_topk/test_quant_order_gpu.py` | 1 | Triton |
| `smoke/primitive/indexer_topk/test_reference_gpu.py` | 1 | DeepGEMM; `LITETOPK_TEST_EXACT_TOPK` for its exact-tie tests; the cuDNN frontend for its radix top-k test |
| `smoke/primitive/indexer_topk/test_plugins_gpu.py` | 1 | `LITETOPK_TEST_PLUGINS`; `LITETOPK_TEST_EXACT_TOPK` for its exact-tie test |

The single-GPU plugin tests run every entry in a fresh interpreter, because plugin settings are
process-wide.
