# Arming the tracer

The method in `SKILL.md` needs one thing from the code: an ordered, per-rank
stream of fingerprinted events. This file covers where that instrumentation
lives, how to turn it on, what it costs, and what to do when it does not exist
where you need it.

## 1. Find the interface before assuming one

A tracer is four pieces. Module names and paths have differed across branches and
between repos, so **search for them rather than trusting a path from any
document, including this one**:

| Piece | What it does | Search for |
|---|---|---|
| Digest | tensor → one comparable value | `hash_tensor`, `signature`, `digest`, `fingerprint` |
| Semantic layer | records collectives / recompute / optimizer boundaries | `collective_trace`, `record_collective` |
| Op layer | fingerprints every ATen op output | `TorchDispatchMode`, `op_trace` |
| Comparator | offline first-divergence diff | `diff_streams`, `compare_traces` |

```bash
# What exists, and how is it switched on?
grep -rln "TorchDispatchMode\|hash_tensor" --include=*.py megatron/ src/ tools/ scripts/ 2>/dev/null
grep -rn "DET_TRACE\|determinism.trace" --include=*.py megatron/ src/ tools/ scripts/ 2>/dev/null | head -40
```

Expect to find nothing. At the time of writing no Megatron repo ships this
tracer on its main branch — it has existed only on feature branches — so the
normal case is that you supply it. Three options, cheapest first:

1. **Skip tracing entirely.** If the break reproduces in
   `tests/unit_tests/determinism/`, you have a faster loop and a regression test
   in one; take it.
2. **Port a tracer onto your branch** from wherever the current implementation
   lives — §2 and §3 describe what it has to do.
3. **Inject it without touching the source tree**, via `PYTHONPATH` (§7). This is
   the option for a read-only container.

## 2. Capture — what goes into the stream

> **Capture and hashing are separate concerns.** This section decides *what* is
> recorded, *where*, and under *what identity* — that is the method, and it is
> where the debugging value lives. §3 decides how a tensor becomes one comparable
> value; it has a clear default (`torch.hash_tensor`) and you should not spend
> design effort there. The two are genuinely decoupled — the same capture layer
> sits above either digest — but decoupled does not mean equally worth your time.

The snippets in §2–§4 are minimal but composable: a writer, an op layer, a
digest (§3) and a comparator (§4) are a working tracer. Build that first, confirm
it finds a divergence you already know about, then add semantic probes (§6).

### The stream writer

Everything else feeds this. Note what it does *not* do: no collectives, no
cross-rank coordination, and no `.item()` per tensor.

```python
SCHEMA_VERSION = 1          # bump on ANY digest change; the comparator refuses
                            # to diff across it (§4)
_STAGED = re.compile(r"__staged_(\d+)__")

class Trace:
    """One rank's stream for one iteration. Append-only JSONL."""

    def __init__(self, out_dir, rank, iteration, flush_every=256):
        self.path = Path(out_dir) / f"rank{rank:05d}_iter{iteration:06d}.jsonl"
        self.fh = self.path.open("w")
        self.rank, self.iteration = rank, iteration
        self.seq, self.events, self.staged = 0, [], []
        self.flush_every = flush_every
        self.is_open = True             # the op layer checks this before recording

    def stage(self, t):
        """Keep the digest on device; return a placeholder to substitute at flush."""
        self.staged.append(fingerprint(t))          # device tensor, NOT .item()
        return f"__staged_{len(self.staged) - 1}__"

    def record(self, kind, name, payload):
        self.events.append({"schema_version": SCHEMA_VERSION, "seq": self.seq,
                            "rank": self.rank, "iteration": self.iteration,
                            "kind": kind, "name": name, "payload": payload})
        self.seq += 1
        # Bound BOTH lists: one event can stage many tensors, so events alone
        # is not a bound on how much device state is pending.
        if len(self.events) >= self.flush_every or len(self.staged) >= 4 * self.flush_every:
            self.flush()

    def flush(self):
        if not self.events:
            return
        # The ONE host sync: every pending digest resolves in a single transfer.
        values = torch.stack(self.staged).tolist() if self.staged else []
        blob = "\n".join(json.dumps(e, separators=(",", ":")) for e in self.events)
        # One pass. Substituting in a loop rescans the whole blob per digest,
        # which turns a larger flush_every into a quadratic cost.
        blob = _STAGED.sub(lambda m: f"{values[int(m[1])] & 0xFFFFFFFFFFFFFFFF:016x}", blob)
        self.fh.write(blob + "\n")
        self.fh.flush()                 # a crashed run still has everything prior
        self.events, self.staged = [], []

    def close(self):
        self.flush()                    # the tail is lost without this
        self.is_open = False
        self.fh.close()
```

Helpers the snippets assume you supply: `active_trace()` (returns the trace for
the current rank/iteration, or `None`), `_tensors(x)` (flatten a tensor / tuple /
list to its tensors), `signature(trace, t)` (`shape`/`dtype`/`numel` plus
`trace.stage(t)`), and `_op_identity(trace, name)` (a per-`(trace, name)`
occurrence counter). None are subtle; all must be consistent across every layer
that writes into one stream.

`SCHEMA_VERSION` is not decoration — bump it whenever the digest definition
changes, and have the comparator refuse to diff across it (§4).

### The op layer, distilled

```python
class OpTraceMode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        if _SUSPENDED.value or (trace := active_trace()) is None or not trace.is_open:
            return out
        name = str(func)
        if any(tok in name for tok in ("empty", "c10d")):
            return out              # uninitialized memory; collectives belong to
                                    # the semantic layer (async = in-flight buffer)
        _SUSPENDED.value = True     # the tracer itself runs tensor ops
        try:
            sigs = [signature(trace, t) for t in _tensors(out) if t.numel() > 0]
            if sigs:
                with _LOCK:                        # backward ops arrive from
                    i = _op_identity(trace, name)  # autograd engine threads
                trace.record("op", name, {
                    "id": f"aten:{name}:{i}",   # run-independent identity; the
                                                # comparator (§4) keys on this
                    "scope": current_scope(),   # decoder.layers.7.mlp -- see below
                    "out": sigs,                # "in" too, if you fingerprint inputs
                })
        finally:
            _SUSPENDED.value = False
        return out

# signature() = shape/dtype/numel plus trace.stage(t) for the digest placeholder.
# Carry shape/dtype/numel alongside the digest: it is what distinguishes an
# all-zero tensor from the empty-tensor sentinel (§3).
```

The per-`(trace, op-name)` occurrence counter is the load-bearing part: it gives
every record an identity that does not depend on arrival order, which is what
lets the offline comparator align two runs semantically. Signatures cover
**every** dtype — integer and bool divergences (routing indices, argmax results)
are real non-determinism and must not be dropped. Oversize outputs record
`digest: "skipped_oversize"` with shape and dtype intact, so a divergence
surfaces at the first under-cap tensor downstream instead of vanishing.

### Name scope — where to attach it

Without this, a divergence reads `aten.slice#4417`, which tells you nothing. With
it, it reads `decoder.layers.7.mlp / aten.slice#12`, which tells you where to
look. The scope is cheap and it is the difference between an actionable finding
and a record id.

Hook it on the **module**, not the op — `nn.Module` forward hooks are the only
place that knows the name:

```python
_SCOPE = contextvars.ContextVar("scope", default=())   # NOT a global: backward
                                                       # runs on autograd threads

def register(root, prefix=""):
    for name, mod in root.named_modules():
        if not name:
            continue
        label = prefix + name
        # Pre-hook pushes, post-hook pops -> the stack is exact even with
        # nesting, early returns, and recursion.
        mod.register_forward_pre_hook(
            lambda m, a, l=label: _SCOPE.set(_SCOPE.get() + (l,)))
        mod.register_forward_hook(
            lambda m, a, o: _SCOPE.set(_SCOPE.get()[:-1]))

def current_scope():
    return "/".join(_SCOPE.get())      # record this on every event
```

Three rules that are easy to get wrong:

- **Prefix per model chunk.** With VPP or any multi-chunk model, two chunks have
  modules with identical relative names (`decoder.layers.0...`). Without a
  `chunk{i}.` prefix the offline diff cannot tell them apart and alignment
  silently breaks.
- **Use a `ContextVar`, not a module-level global.** Backward executes on
  autograd engine threads; a plain global gives you one thread's scope stamped
  onto another thread's records.
- **Do not extend scope into backward by hooking module outputs.** Backward
  hooks that alias module outputs break pipeline-parallel output deallocation.
  Backward divergences are still captured — they just carry the forward scope
  rather than a `[bwd]` label. If you want true backward labels, keep it to
  TP-only debugging runs.

Two interface styles exist. Check which one you have:

- **Config/argument-driven** — the training loop opens a trace for chosen
  iterations. A context-manager API makes the wiring a few lines at the top of
  the step:

  ```python
  with trace_iteration(trace_dir, iteration, hash_tensors=True):
      with op_trace_mode(enable_op_layer):   # rung 2; omit for rung 1 only
          ...                                # the existing train_step body
  ```

  If your checkout has the core module but no call site, that wiring is the
  piece you add. Keep it gated so `trace_dir=None` is a no-op.

- **Environment-driven** — the tracer self-installs and reads env vars. The
  names below are **illustrative, not an interface that exists in this repo** —
  they show which knobs are worth having, not what to type:

  | Env var | Effect |
  |---|---|
  | `DET_TRACE_OUT_DIR` | Stream directory; **setting it enables the tracer**. Use a shared FS. |
  | `DET_TRACE_ITERS` | Window: `all`, `a-b`, or a list `1,5,40-44`. Default `1`. |
  | `DET_TRACE_OPS=1` | Rung 2 — fingerprint every ATen op output. |
  | `DET_TRACE_OP_MAXNUMEL=N` | Skip outputs above N elements (0 = no cap). |
  | `DET_TRACE_FLUSH_EVERY=N` | Digest resolve/flush batch size. |

  Repo-specific probes may add more (`DET_TRACE_GEMM`, `DET_TRACE_GRADS`,
  `DET_TRACE_MOE`, …) — see §6.

### The call-site lifecycle

Whichever style you have, the wiring the training loop owns is the same four
moves. Condensed from the Megatron-Bridge `train.py` wiring:

```python
# --- once, before the loop: install capture, inert if the env var is unset ---
if out_dir := os.environ.get("DET_TRACE_OUT_DIR"):
    collective_trace.enable(out_dir=out_dir)
    for i, chunk in enumerate(model):
        # Per-chunk prefix, or VPP / multi-chunk runs collide on identical
        # relative module names and the offline diff cannot align them.
        module_scope.register(chunk.module, prefix=f"chunk{i}." if len(model) > 1 else "")
    if os.environ.get("DET_TRACE_OPS"):
        op_trace.enable()                      # rung 2

# --- per iteration: scope the window ---
on = trace_iters is None or step in trace_iters
if on:
    collective_trace.set_active(True, window=step)
try:
    train_step(...)
finally:
    if on:
        collective_trace.set_active(False)     # must be try/finally: a recovered
                                               # exception would otherwise leave
                                               # tracing armed for later iterations

# --- after the loop: flush streams and restore the patched call sites ---
op_trace.disable(); module_scope.unregister(); collective_trace.disable()
trace.close()        # flushes the tail; without it you lose the last partial batch
```

The three details that are easy to get wrong and expensive to debug: the
**per-chunk prefix** (alignment), the **`try`/`finally`** (a leaked active window
silently changes what later iterations capture), and **teardown order** (op layer
first, stream last, so the last records still have somewhere to go).

## 3. The hash function — use `torch.hash_tensor`

Independent of §2. Capture decides what to record; this decides how a tensor
becomes one comparable value. **Use the native op.** The comparison below is why.

### Why an xor reduction at all

The digest reduces millions of lanes on the GPU, and the reduction tree order is
not fixed — it varies with block count, occupancy, chunk size and device. So the
combining operation must be **associative and commutative in exact arithmetic**.
That one requirement eliminates most candidates before any quality argument:

| Reduction | Order-independent? | Verdict |
|---|---|---|
| **Floating-point sum** | **No** — fp addition is not associative | Disqualified. This is the exact bug you are hunting; the instrument would manufacture its own false divergences. |
| **XOR** | Yes — exact, no carries | What `torch.hash_tensor` uses. Cheapest possible; every bit independent, so it tree-reduces trivially. |
| **Integer sum mod 2^64** | Yes — wraps exactly | Also valid, and can be made permutation-sensitive by weighting each lane by its position first. Costs real time and memory (measured below). |
| **Multiply mod 2^64** | Yes, but zero-absorbing | Useless — one zero lane destroys the digest. |
| **SHA-256 / crypto hash** | **No** — needs a canonical byte order | Requires a host copy per tensor, which makes it **orders of magnitude slower** than any device-side digest. Reserve for settling a suspected collision; never the hot path. |

### The recommendation

```python
def fingerprint(t):
    x = t.detach()
    if x.is_complex():
        x = torch.view_as_real(x)     # hash_tensor has no complex support
    x = x.contiguous()
    # xor_sum has no UNSIGNED CUDA kernel (uint8/16/32/64 -> "xor_sum_cuda not
    # implemented for UInt64"). Bitcast unsigned -> signed of the SAME width:
    # identical bytes, and both jobs bitcast the same way, so still stable.
    x = x.view(_UINT_TO_INT.get(x.dtype, x.dtype))
    return torch.hash_tensor(x)       # uint64 TENSOR on x's device -> stageable
```

It returns a `uint64` **tensor**, not a host int, so the caller stages it and
defers the single `.item()` to the step boundary — nothing but an 8-byte scalar
crosses to host, and never mid-iteration.

Do **not** use `hash(tensor.numpy().tobytes())`: it is salted by
`PYTHONHASHSEED`, so it differs between processes and is silently wrong for
exactly the cross-launch comparison this method rests on.

### What each variant costs

Benchmarked on a single Blackwell GPU across tensors from tens to hundreds of
megabytes. Absolute timings are not portable and are omitted deliberately — the
**ordering** is the durable result, and it held at every size tested.

| Variant | Speed | Scratch memory | Catches permutations |
|---|---|---|---|
| whole tensor, native xor | fastest | none measurable | no |
| chunked `dim=-1` | same as whole tensor | negligible | across granules |
| two-level (chunk + position-weighted reduce) | slightly slower | negligible | across granules |
| full custom digest over raw bytes | ~an order of magnitude slower | **3x the input tensor** | within a granule too |

Two things drive the recommendation:

- **Chunking is free.** Reducing per granule costs the same as reducing the whole
  tensor, so permutation coverage at granule scope is not a performance trade at
  all. Its real price is trace size, since you store a vector of digests.
- **The full custom digest's cost is memory, not time.** It materialises a
  position ramp and several intermediates totalling 3x the input tensor. On a
  memory-tight run during the live-activation forward, that is what OOMs a rank —
  and it is the reason to reach for chunking or the two-level form first.

> **Beware microbenchmarks near the floor.** A digest of a small tensor is
> dominated by kernel-launch overhead, not compute, and at that scale the
> variants are indistinguishable — a finer-grained digest can even measure
> *faster* than a coarse one. Compare them on tensors large enough to clear that
> floor, or you will draw the wrong conclusion.

### What xor costs you

Measured on the same run — both blind spots are real, not theoretical:

| Property | `hash_tensor` | position-weighted sum |
|---|---|---|
| all-zero tensor digests to 0 | **yes** | no |
| all-zero len 4096 == all-zero len 8192 | **yes** | no |
| `[a, a]` collides with `[b, b]` | **yes** | no |
| detects a permutation | **no** | yes |
| detects a 1-ULP change | yes | yes |

XOR is involutive (`x ^ x == 0`), so duplicate lanes annihilate in pairs — and
this workload is full of repeats: routing maps, masks, one-hot rows,
zero-initialised buffers. It is also linear over GF(2)⁶⁴, so its collisions are
algebraically describable rather than merely rare.

**Mitigate rather than replace.** Compare `shape`, `dtype` and `numel` alongside
the digest — the comparators already do, which is exactly how an all-zero tensor
is told apart from the empty-tensor sentinel that shares its digest. That closes
the length and dtype collisions for free.

### Permutations: fix the granularity, not the hash function

The obvious reading of that table is "xor is permutation-blind, so swap the hash
function." **That is the wrong lever.** The blindness is a property of the
*scope* of the reduction, not of xor: `hash_tensor` accepts `dim=`, so reducing
per-row or per-chunk makes any permutation that moves values **across** granules
visible immediately. Measured on a 64K permutation array and a `[4096, 128]` MoE
dispatch tensor:

| Case | whole-tensor | chunked (`dim=-1`) | full custom digest |
|---|---|---|---|
| Permutation array `0..n-1` | **blind** | detected at every granule 4096→8 | detected |
| MoE dispatch, rows permuted | **blind** | detected per-row (G=128) | detected |
| Routing map, swap 10↔3000 | **blind** | detected (G≤16) | detected |
| Routing map, **adjacent** swap | **blind** | **blind at every G, incl. G=2** | detected |
| 1-ULP value change | detected | detected | detected |

So the rule is: **granularity bounds the blind spot; it does not remove it.**
A permutation entirely *within* one granule stays invisible to xor at any
granule size — only a position-weighted reduction catches that.

Why the whole-tensor row is so stark: **xor over any permutation of `0..n-1` is
identically zero.** Three unrelated permutations of `0..65535` all digest to
`0000000000000000`, colliding with the all-zero tensor and the empty-tensor
sentinel at once. And hashing the index tensor alongside the values does not
rescue it — an index array is itself a permutation, so xor is invariant to any
change in it.

Chunking is free in time (see the measurement table above), which is what makes
it the right first move. Its price is **trace size** — you now store a vector of
digests rather than one value. When that matters, use the **two-level** form:
native xor per granule, then a position-weighted reduce over the small digest
vector. That restores a single stored value for a small, near-flat time cost and
negligible scratch, where the full raw-byte digest costs both.

The full raw-byte custom digest only earns its keep when you need
**within-granule** permutation sensitivity.

### The position-weighted digest

A 128-bit, position-weighted, raw-byte digest that is permutation-sensitive and
does not cancel duplicates. Its shape, condensed:

```python
lanes = x.reshape(-1).view(torch.uint8).view(torch.int64)   # raw bytes, any dtype
idx   = torch.arange(lanes.numel(), device=lanes.device)    # absolute position
acc  += torch.stack((                       # two accumulators, both value- AND
    _mix64(lanes ^ _mix64(idx + 1)).sum(),   # position-sensitive, so a permutation
    _mix64(lanes ^ _mix64(idx + K)).sum()    # must collide in both to hide
                                             # (K = any second distinct constant)
))   # integer adds wrap mod 2**64 exactly -> chunk size, GPU and rank irrelevant
```

Price of that property, from the table above: roughly an order of magnitude more
time, and **3x the input tensor in scratch**. That is why you reach for chunking
or the two-level form first, and scope this one to the tensors that need
within-granule sensitivity. Do not hand-roll a variant — a digest change makes old and new
traces incomparable, which is why records carry a schema version the comparator
refuses to diff across.

### Resolution — the seam back to capture

The one place the two topics touch. Capture records a
`__staged_digest_N__` placeholder; `finalize_digests` stacks every pending
accumulator into **one host transfer per device** and the trace substitutes real
values at flush time (default every 256 events). This is the only point the
digest path touches the host, which is what keeps per-tensor fingerprinting off
the critical path — and it bounds crash loss to the trailing unflushed events.

## 4. Offline comparison

This is the payoff step, and it is about 20 lines. Align **by identity, not by
arrival order** — that is what the per-op occurrence counter in §2 bought you:

```python
def first_divergence(dir_a, dir_b):
    """Walk A in execution order; return the first record whose output differs."""
    for fa, fb in zip(sorted(Path(dir_a).glob("*.jsonl")),
                      sorted(Path(dir_b).glob("*.jsonl"))):   # pair by rank+iter
        # Stream A; index B lazily. Loading both files whole would parse tens of
        # GB to answer a question that is usually settled in the first thousand
        # records -- the opposite of what "stop at the first divergence" means.
        by_id_b, lines_b = {}, (json.loads(line) for line in fb.open())
        for a in (json.loads(line) for line in fa.open()):
            if a["schema_version"] != SCHEMA_VERSION:
                raise ValueError("schema bump: digests are not comparable")
            key = (a["name"], a["payload"].get("id"))
            while key not in by_id_b:               # advance B only as far as needed
                e = next(lines_b, None)
                if e is None:
                    break
                by_id_b[(e["name"], e["payload"].get("id"))] = e
            b = by_id_b.get(key)
            if b is None:
                # No counterpart at all -> control flow diverged, not just values.
                return {"kind": "missing_in_b", "file": fa.name, "record": a}
            if a["payload"]["out"] != b["payload"]["out"]:
                return {"file": fa.name, "seq": a["seq"], "name": a["name"],
                        # True  -> this op is the root cause.
                        # False -> cause is upstream; keep walking back.
                        "inputs_matched": a["payload"].get("in") == b["payload"].get("in"),
                        "a": a, "b": b}
    return None          # every output matched
```

A missing counterpart is a *different* finding from a value mismatch: it means
the two runs took different code paths, so compare configs before reading it as
numerical nondeterminism.

Two tools, whatever they are called in your implementation:

```bash
compare_traces <trace_dir_A> <trace_dir_B>    # first divergence, causal order
certify_traces <trace_dir_A> [<trace_dir_B>]  # trace invariants hold?
```

`compare_traces` walks both trees by semantic event identity and reports
divergences in causal order. `certify_traces` checks the invariants a
determinism run assumes — deterministic algorithms enabled, recompute matching
its forward, contiguous sequences, no collective begun without an end, no
pending collectives at window close, expected rank/iteration coverage. **Certify
one tree before you trust a diff between two.**

> **Gotcha — schema version.** The writer stamps `schema_version` on every record
> and the comparator accepts exactly one value. When they drift apart the
> comparator rejects every record, which looks like a broken tool — check the
> constant on both sides first. Never diff across a schema bump: the digest
> definition changed, so *every* record differs and the "first divergence" is
> meaningless.

## 5. Cost and scale

The design goal is a tool that survives a 3,000+ GPU campaign. The properties
that make that work, and the knobs you control:

- **Rank-local JSONL.** Each rank writes its own file. Tracing adds **no
  collectives and no cross-rank ordering** to the step being observed, so it
  cannot itself perturb the thing you are measuring.
- **Digests computed on the GPU.** Bytes are reduced on the tensor's own device
  with an order-independent reduction; only a scalar per tensor reaches the host
  (two, if you use the 128-bit position-weighted variant of §3). Digests stage as device accumulators and resolve one batch per
  flush, so the step is never stalled to read a hash back. Host-side byte
  hashing (`sha256`) is hundreds of times more expensive (§3) — use it only to
  settle a digest-collision question.
- **Scoped capture.** Trace chosen steps (start / end / every N-th) and chosen
  ranks (all, or a list like `0,3,8-15`). A bounded window is the difference
  between a 40 GB campaign and an unusable one.
- **Budget caps.** Skip oversize tensors (`DET_TRACE_OP_MAXNUMEL`), cap event
  counts, or run summary-only. A skipped output still records shape/dtype, so a
  divergence surfaces at the first under-cap tensor downstream.
- **Quantized-aware.** MXFP8 / NVFP4 tensors are digested through their raw data
  plus scale buffers, so a low-precision operand is a first-class fingerprint
  rather than an opaque `uint8` blob.

Operational consequences:

- Raise `--distributed-timeout-minutes`; traced iterations are slower and the
  NCCL watchdog does not know that.
- Rung 2 runs Python per ATen op and **cannot run inside `torch.cuda.graph`
  capture**. Disable CUDA graphs for the traced run.
- Collective outputs must be fingerprinted at the consumer's **wait**, never at
  the launch point, or the digest races the NCCL kernel still writing the
  buffer. If you add a collective probe yourself, respect this.

## 6. When a boundary is not covered

The op layer only sees the torch dispatcher. Anything that bypasses it — a
Transformer Engine GEMM (a pybind call), a fused wgrad writing `main_grad`,
cuDNN fused attention, Triton/FLA kernels, DeepEP/HybridEP token dispatch — is a
blind spot. A divergence inside one surfaces one hop later (see
`reading-traces.md`).

To close a blind spot, add a probe that writes into the **same ordered stream**.
That is the whole contract — a probe is about 20 lines:

```python
# Wrap the extension entry point; record operands and result as one event.
import transformer_engine.pytorch.cpp_extensions.gemm as te_gemm

_orig = te_gemm.general_gemm
_seen = collections.Counter()

def _traced(*args, **kwargs):
    out = _orig(*args, **kwargs)
    trace = active_trace()
    if trace is not None:
        # Name it by the axes you will want in a discriminator table later
        # (layout, accumulate, grad vs forward, output dtype) -- that naming is
        # what turns step 4 of SKILL.md into a table lookup instead of a new job.
        name = f"gemm:{kwargs.get('layout')}:acc{int(bool(kwargs.get('accumulate')))}"
        _seen[name] += 1
        trace.record("op", name, {
            "id": f"{name}:{_seen[name]}",                     # same identity rule as §2
            "in":  [signature(trace, t) for t in _tensors(args) if t.numel() > 0],
            "out": [signature(trace, t) for t in _tensors(out) if t.numel() > 0],
        })
    return out

te_gemm.general_gemm = _traced
```

Name events so instances are distinguishable by the axes you will later want in
a discriminator table (layout, accumulate, grad/forward, output dtype). That
naming is what lets step 4 of `SKILL.md` be a table lookup instead of a new job.

Probes that have paid for themselves: **TE `general_gemm`** (operand bits,
accumulator, output), **per-parameter gradient at `optimizer.step()` entry**
(catches fused wgrad writes the dispatcher never sees), **MoE dispatch/combine
boundaries**.

## 7. Inject the tracer; do not rebuild the image

This is a determinism constraint, not a convenience. **Rebuilding the image to
add instrumentation changes the artifact under test.** A different build of TE,
cuBLAS, NCCL or torch can change numerics on its own, so a traced run from a new
image is not comparable to the untraced run that showed the break, and you can
spend days chasing a divergence you introduced by instrumenting. Keep the image
bit-identical across both arms and both runs; add the tracer beside it.

CPython auto-imports `sitecustomize` from `PYTHONPATH` at interpreter startup, so
the bootstrap goes there, installs a post-import hook on the training module, and
wraps `train_step`:

```bash
PYTHONPATH=/path/to/dettrace_bootstrap:$PYTHONPATH \
DET_TRACE_OUT_DIR=/lustre/.../streams/run_A \
DET_TRACE_ITERS=1-8 DET_TRACE_OPS=1 \
  <your existing launch command>
```

The bootstrap must be **inert unless the trace directory is set** — it runs in
every Python process in the container, including ones you did not mean to trace.

This is also how you port a tracer between repos: the capture modules are plain
Python with no Megatron dependency beyond the wrap points, so a tracer developed
in one repo can be dropped into another as a standalone package on `PYTHONPATH`.

## 8. Below the dispatcher (rung 4)

The trace stops at the last line of code you can read. When the first divergence
is inside a closed kernel, these are the only instruments left — and you are not
profiling for speed, you are hunting **the reduction mechanism**, because
reduction order is what makes a kernel order-dependent.

Only after the discriminator table points at a closed kernel:

```bash
CUBLASLT_LOG_LEVEL=5   # which algo was selected, and the problem descriptor.
                       # Look for reductionScheme / numSplitsK: a split-K
                       # reduction is a nondeterminism mechanism you can name.
                       # Their ABSENCE is also a finding — it rules out the
                       # classic split-K explanation and the workspace knob.

NVTE_NVTX_ENABLED=1 nsys profile --trace=cuda,nvtx …
                       # the kernel SYMBOL, attributed to a library NVTX range
                       # so you know which call site it belongs to.

ncu …                  # atomics / reduction metrics — the only way to explain
                       # *why* a kernel is order-dependent, once it is named.
```

Two rules that matter more than the commands:

- **Separate selection from execution.** If two runs select the *same* algo and
  still disagree, the nondeterminism is inside the kernel, not in autotune or
  heuristic choice — which is a completely different bug report. Log the
  selection on both runs before blaming either.
- **Require two independent instruments to agree** before naming a kernel — e.g.
  the cuBLASLt log's dispatch count and nsys's launch count for the same
  workload. One instrument is an assertion; two is evidence.

Know the ceiling. From outside a closed binary you can reach the kernel symbol,
the algo descriptor, the exact problem descriptor, and a flag that toggles the
behaviour. *Why* the kernel is order-dependent internally needs `ncu` or the
owning library team — say that explicitly rather than speculating.
