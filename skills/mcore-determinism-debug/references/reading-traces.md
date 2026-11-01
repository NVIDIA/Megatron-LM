# Reading a divergence

A correct trace plus a careless read produces a confident wrong answer. This
file is the checklist between "I have a first divergence" and "I know the site".

## Anatomy of a record

```
seq=2547  gemm:NT:acc0:grad1   inputs=MATCH
  in : uint8(6144,9216)  uint8(6144,1152)  uint8(18432,3072)
       uint8(6144,9216)  uint8(18432,384)  uint8(6144,1152)    all SAME
       float32(6144,6144) digest 0000…0000                     SAME
  out: float32(6144,6144)   A be0de2f5…   B 804b6328…          DIFFERENT
```

Read it in this order:

1. **`inputs=MATCH`?** If no, you are downstream of the cause — keep walking
   back. Only an inputs-matched / output-differs record is a root-cause
   candidate.
2. **Are the inputs actually fingerprinted, or trivially empty?** A record with
   zero input entries "matches" vacuously. Confirm real digests on real shapes.
3. **What is before it?** Report the count: "2,445 ATen ops, 93 GEMMs, 9
   collectives, all bit-identical." That number is the claim's strength.
4. **Does it reproduce across arm pairs?** Same first divergence in 0v1, 0v2,
   0v3 is a property of the code. One pair is an anecdote.
5. **Is this op plausibly nondeterministic?** If not, see "one hop later".

## One hop later — the most common misread

The op layer only sees the torch dispatcher. These bypass it entirely:

| Bypass | Surfaces at |
|---|---|
| TE `general_gemm` (pybind → cuBLASLt) | first ATen op consuming the GEMM output |
| Fused wgrad writing `main_grad` | the gradient at `optimizer.step()`, or a later reduce |
| cuDNN / TE fused attention | validated example: the RoPE-backward `aten.slice` that reads the attention gradients |
| Triton / FLA kernels | first ATen consumer |
| DeepEP / HybridEP `fused_a2a`, `hybrid_ep_cpp` | first ATen op reading the dispatched tokens |

**Rule:** when the first divergent record is an op that cannot itself be
nondeterministic — a `slice`, `view`, `cat`, elementwise add — with matched
inputs, do not report it. Read *up* to what produced its input and add a rung-3
probe there. The trace is telling you the truth; it just cannot see the
producer.

Conversely, when the first divergent record *is* a heavy fused compute op with
matched inputs, that is the site.

## Known probe artifacts

These read as divergences and are not. Exclude them before acting:

- **`aten.empty*` family.** Returns uninitialized memory; its contents differ
  run-to-run by definition. Good tracers skip this family — if yours reports it,
  it is an artifact.
- **`c10d` collective ops seen by the dispatch mode.** For async collectives the
  dispatcher sees an in-flight buffer. Collectives belong to the semantic layer,
  which records them at the consumer's `wait`.
- **`aten._to_copy` of a small int64 count tensor where one side reads all-zero.**
  This is the tracer catching a `maybe_move_tensor_to_cpu(non_blocking=True)`
  mid-DMA. An observation error, not a divergence in training state.

General test for an artifact: *would a difference here actually propagate into
the next iteration's weights?* If the value is never read, or is read after a
sync that the tracer preempted, it is an artifact.

## What the fingerprint guarantees

Know your digest before you argue from it:

- **Order independence** is what makes a digest identical across processes, GPUs
  and topologies — the property a cross-job key needs. Every recommended digest
  has it; they differ in what else they catch.
- A plain xor-style reduction (`torch.hash_tensor`) is order-independent and
  1-ULP sensitive but **permutation-invariant** — see `tracing-setup.md` §3 for
  what that does and does not cover.
- **Byte-viewing rather than upcasting** keeps the digest dtype-agnostic, so
  fp8/fp4 payloads, int64 routing maps and uint8 scale buffers are all covered.
- **Never compare digests across schema versions.** A reweighted digest makes
  every record look divergent.

## Turning records into a discriminator table

The stream already contains many instances of the divergent call under different
configurations. Group them and read off the axis that discriminates:

```
gemm:TN:acc0:grad0   forward   D=bf16    CLEAN
gemm:NN:acc0:grad1   dgrad     D=bf16    CLEAN
gemm:NT:acc0:grad1   wgrad     D=fp32    DIVERGENT
gemm:NT:acc1:grad1   wgrad     D=fp32    DIVERGENT
```

Here `accumulate` is ruled out (both `acc0` and `acc1` diverge) and operand
dtype is ruled out (identical in all four rows). The discriminator is **fp32
output on the NT wgrad**. That conclusion cost zero additional jobs.

Then cross it against the other operand family to separate "selects the kernel
family" from "breaks the kernel":

```
              D = bf16   D = fp32
FP4 operands  clean      BREAKS
BF16 operands clean      clean
```

Only the intersection breaks — so neither factor alone is the cause, and any fix
that changes only one of them is incidental.

## Before you report

- [ ] Certification passed on at least one trace tree.
- [ ] First divergence has `inputs=MATCH` with non-empty, real input digests.
- [ ] It is not on the artifact list.
- [ ] It is not a trivially-deterministic op standing in for an uninstrumented
      producer.
- [ ] It reproduces across all arm pairs.
- [ ] The discriminator table names the axis, and you can say what each
      candidate factor was *ruled out* by.
- [ ] Any kernel name comes from two agreeing instruments, or is stated as not
      yet named.
