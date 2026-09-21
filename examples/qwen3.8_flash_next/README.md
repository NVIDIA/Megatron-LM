# Qwen3.8-Flash-Next: verify the implementation, then train it

This directory is the runnable companion to
[`docs/models/qwen3.8_flash_next/`](../../docs/models/qwen3.8_flash_next/). The docs record what
was measured; everything here you can execute.

Four things, in the order they are useful:

| | What | Needs |
|---|---|---|
| [1](#1-numerical-parity-against-hugging-face) | **Numerical parity** of this implementation against Hugging Face `qwen4_exp` | 1 GPU, ~5 minutes |
| [2](#2-train-the-proxy) | **Train the proxy** — every per-layer dimension real, 8 layers | 4 GPUs |
| [3](#3-convert-the-released-checkpoint) | **Convert the released checkpoint** HF → Megatron | CPU node + ~700 GiB |
| [4](#4-train-the-full-model) | **Train the full 48-layer model** | 64 GPUs |

Start with 1. It is cheap, it needs nothing downloaded, and it is the step that answers "does
this implementation actually compute the same function as the reference".

## Prerequisites

A container with Megatron's usual dependencies plus, for this model:

* **`fla`** — the gated-delta-rule kernels. Megatron's GDN imports it.
* **Triton** and a recent **TransformerEngine**.
* For step 1 only, a **`transformers` build that provides `qwen4_exp`**. At the time of writing
  that is not in a released wheel; install it into a side prefix and point `QWEN4_TRANSFORMERS`
  at it, so the container's own `transformers` is left alone:

  ```bash
  pip install --target /some/prefix 'transformers @ git+https://github.com/huggingface/transformers@<rev-with-qwen4_exp>'
  export QWEN4_TRANSFORMERS=/some/prefix
  ```

  `parity/run_parity.sh` fails with a clear message if the import is missing, so you find out in
  seconds rather than halfway through a run.

Everything below runs from the Megatron-LM root. On a scheduler, submit each command as its own
job; do not hold an interactive allocation open around them.

## 1. Numerical parity against Hugging Face

The comparison builds the **same 4-layer proxy on both sides** — gated residual over 4 streams,
GDN with a sigmoid output gate, QSA with its indexer, MoE with a gated shared expert, and the
PLE n-gram memory — gives them the same weights and the same tokens, and compares in fp32.

```bash
export QWEN4_TRANSFORMERS=/some/prefix     # see Prerequisites
./examples/qwen3.8_flash_next/parity/run_parity.sh /tmp/qwen4-parity --steps 20
```

It generates a random-initialised HF model, saves it, converts the saved tensors with the mapping
in `parity/hf_to_mcore.py`, loads them into Megatron's `HybridModel`, and reports three things:

| Check | What it catches | Expected |
|---|---|---|
| **forward** | wrong weight mapping, wrong layer order, a norm convention mismatch | `logits max_abs ≈ 2.7e-7`, `allclose(1e-6, 1e-5)`, identical argmax |
| **gradients** | a forward that is right for the wrong reason; backward-only bugs | 161 parameters compared, 0 outside `rtol 1e-5 / atol 1e-4` |
| **trajectory** | error that only compounds — a wrong optimizer-visible shape, a drifting state | 20 AdamW steps, `max abs(Δloss)` in the 2e-4 … 3e-4 band, accumulating smoothly |

A run that passes prints `[done] PASS=True` and writes `out/report.json` with every number.

### Two settings that decide whether the comparison means anything

`run_parity.sh` exports both; if you build your own harness, do not omit them:

```bash
export NVIDIA_TF32_OVERRIDE=0
export TRITON_F32_DEFAULT=ieee
```

TransformerEngine requests `CUBLAS_COMPUTE_32F_FAST_TF32` for every pure-fp32 GEMM, and Triton's
`tl.dot` defaults to TF32 for fp32 inputs. Either one puts roughly **1e-3** of noise under every
linear layer — a thousand times the quantity being measured, so the comparison would report
"agreement" at loose tolerances while saying nothing. They matter on **both** sides: when `fla`
is importable, HF's gated-delta-rule dispatches to the same Triton kernel, and missing that once
cost half a day chasing an 8.5e-4 "divergence" in a GDN layer that was pure TF32 noise.

### What the harness is made of

| File | Role |
|---|---|
| `parity/proxy_config.py` | The proxy, described once: the HF config, and the same model as Megatron arguments with each non-obvious correspondence explained |
| `parity/hf_to_mcore.py` | The **weight mapping**. Worth reading on its own — it is the most compact statement of how the two implementations line up |
| `parity/make_fixture.py` | Random-initialises the HF proxy and saves it, plus the token batches |
| `parity/run_parity.py` | Builds both models, maps, compares |
| `parity/run_parity.sh` | Environment + the two steps above |

The mapping is deliberately explicit rather than clever. Six rules are not plain renames, and each
says why in the source: the GDN input projection is fused differently, QSA's QKV is interleaved
per query group with its output gate, the shared expert stacks gate and up, the GDN output norm is
zero-centered on one side and not the other, the n-gram table is split per hash head, and HF keeps
three derived tensors that Megatron recomputes.

Two layout differences are handled outside the rules, in `hf_to_mcore.py`, because they are
properties of *how a checkpoint was written* rather than of the model:

* `repack_experts()` — `transformers` 5.x `save_pretrained` writes per-expert
  `experts.{e}.gate_proj.weight`, while the **released** checkpoints pack
  `experts.gate_up_proj [E, 2I, H]`. The rules target the released layout.
* `ple_source="sharded" | "single"` — on disk the n-gram table is split into
  `split_ngram_parts` row shards; in memory it is one `nn.Embedding`.

One subtlety worth knowing if you adapt this: the GDN output norm maps as `w_mcore = w_hf - 1`,
but **gradients are not offset** — a constant shift does not survive differentiation. Applying
the offset to a gradient produces a clean `1.0` discrepancy on exactly the zero-centered norms,
which is what it looks like when this is got wrong (`apply_rules(..., for_gradients=True)`).

### Scope

EP1. The parity harness used during development also runs at EP4 with the same result, since
expert parallelism moves weights between ranks without changing which experts a token traverses —
the forward is bit-identical. MTP is excluded: HF has no MTP implementation to compare against.

## 2. Train the proxy

Eight layers, 32 experts, every per-layer and per-expert dimension identical to the full model, on
mock data with `NullTokenizer` — nothing to download:

```bash
./examples/qwen3.8_flash_next/train_proxy_1node.sh
```

4 GPUs, EP4, BF16. Per-rank shapes match the 64-GPU model (8 local experts either way), which is
what makes it useful for recipe work. It is **not** a memory proxy for EP64 — it fits one node
with a large margin. Details and the dimension-by-dimension rationale:
[`training/proxy_single_node.md`](../../docs/models/qwen3.8_flash_next/training/proxy_single_node.md).

## 3. Convert the released checkpoint

Needed only for continued pretraining from the published weights. The released checkpoint is
131 shards / 335 GiB, and the conversion writes a native Megatron distributed checkpoint, so plan
for roughly 700 GiB of scratch and a CPU node.

The procedure, the rule set, the streaming treatment of the 51 B-parameter n-gram table and the
`--skip-ngram` / `--skip-mtp` switches are documented in
[`checkpoint/hf_weights_and_resume.md`](../../docs/models/qwen3.8_flash_next/checkpoint/hf_weights_and_resume.md).
The mapping those tools apply is the same one `parity/hf_to_mcore.py` states compactly here, so
reading it is the fastest way to understand what the converter does.

Measured once on the full 48 layers: strict conversion in 2 h 09 on a CPU node with 41 GiB peak
RSS, then a native EP64 checkpoint that a fresh 64-GPU model loads with all 109,632 parameters
equal.

## 4. Train the full model

```bash
MASTER_ADDR=<rank0-host> LOAD=<converted_ckpt_dir> \
    ./examples/qwen3.8_flash_next/train_full_model_64gpu.sh
```

One instance per node, 16 nodes × 4 GPUs, EP64, BF16. Omit `LOAD` to train from scratch. Resolve
`MASTER_ADDR` outside the container — `scontrol` usually is not installed inside it.

**The argument list is the one that was run** (50 steps, 16 s/step, 204.4 GiB peak per rank,
29 GiB headroom); **the wrapper script itself has not been executed**, so treat its first run as a
validation run. Why EP64 and not EP32, and the numbers behind that:
[`training/full_model_64gpu.md`](../../docs/models/qwen3.8_flash_next/training/full_model_64gpu.md).

## Known traps

Each of these cost at least one job to find.

* **Never set `NVTE_NORM_FWD_USE_CUDNN` / `NVTE_NORM_BWD_USE_CUDNN`** on this model — measured at
  5–10× step time. Recipes tuned for sibling dense-MoE models do set them.
* `--eval-interval` must stay set even with `--eval-iters 0`: the iteration-based data sizing
  divides by it.
* `--mock-data` needs `--moe-router-force-load-balancing` for meaningful timing; random tokens
  produce a badly skewed router. Drop both together when moving to real data.
* Packed sequences go through `--sft`; the packing schedulers selected by `--use-varlen-dataset`
  are rejected by the n-gram memory.
* VPP works, but not together with `--overlap-param-gather`.
* CUDA graphs: MoE-scoped capture works and is worth −6.2 % step time; wider scopes are refused by
  the sparse-attention and memory modules.

The full list of what runs and what is refused, with the exact flags:
[`validation/support_matrix.md`](../../docs/models/qwen3.8_flash_next/validation/support_matrix.md).
