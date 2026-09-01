# MagiAttention in Megatron Lite

Megatron Lite can use [MagiAttention](https://github.com/SandAI-org/MagiAttention)
as the distributed core-attention backend for the native `qwen3_moe` model.
Lite continues to own embeddings, QKV/output projections, Q/K normalization,
RoPE, MoE, and the training runtime. MagiAttention owns the load-balanced token
dispatch and context-parallel attention communication.

MagiAttention is an optional, architecture-specific CUDA dependency and is not
installed with Megatron Lite. Install it for the target GPU before selecting the
backend.

In the GA=8 Qwen3-30B-A3B heavy-tail THD benchmark, MagiAttention improved
end-to-end training throughput by 31.88% over Transformer Engine.

```python
from megatron.lite.runtime import MegatronLiteConfig, RuntimeConfig, create_runtime
from megatron.lite.runtime.contracts import ParallelConfig

backend_cfg = MegatronLiteConfig(
    model_name="qwen3_moe",
    attention_backend_override="magi",
    parallel=ParallelConfig(tp=2, cp=4),
    impl_cfg={"use_thd": True},
)
runtime = create_runtime(
    RuntimeConfig(backend="mlite", hf_path="/models/Qwen3-MoE", backend_cfg=backend_cfg)
)
handle = runtime.build_model()
```

There are deliberately no user-facing tuning knobs for this backend: the
only user decision is ``attention_backend_override="magi"``. Chunk sizing and
overlap staging are decided automatically (magi derives the chunk size and
its dynamic overlap solver picks the staging per microbatch), and any locally
calibrated policy belongs in ``resolve_magi_attention_config`` — the single
policy seam in the backend primitive. Future attention backends must follow
the same rule: backend tuning lives in the backend's primitive, never as
parameters on model constructors or user config surfaces.

The batch protocol pads each sequence to Lite's TP/CP THD alignment and then
applies one MagiAttention dispatch plan to token IDs, labels, loss masks, and
position IDs before embedding and QKV projection. The runtime key stays with
that microbatch for activation recomputation. Q/K RoPE is indexed with the
dispatched positions, and MagiAttention output remains in the same local order
for the output projection and loss.

A chunk size chosen by magi (or by a calibration policy) acts as a cap: if
it would make MagiAttention add a second tail pad, Lite lowers it to the
nearest padding-free divisor of the per-CP-rank token count, keeping every CP
shard equal with no extra loss-bearing tokens.

## Hot-swapping backends

The te and magi core-attention backends are interchangeable on a built model:
``model.set_attention_backend("magi"|"te")`` swaps them in place. Both
backends are parameter-free, so the swap leaves every parameter and buffer —
and therefore the optimizer state — untouched; te-trained checkpoints resume
under magi and vice versa. The only state_dict difference is TE's internal
``_extra_state`` metadata entries (empty of trainable content in bf16), which
exist only while the te backend is selected; cross-backend restore must
tolerate their presence or absence. The batch protocol re-reads the backend for every
microbatch, so the swap takes effect on the next step; only switch at step
boundaries (never between a forward and its recompute/backward). Switching to
magi revalidates its scope (CP>1, PP=1, no MTP/MRoPE).

The initial supported scope is bf16 Qwen3-MoE training with packed THD batches,
full causal GQA, static CP, TP, and activation recomputation. It requires
`CP > 1`, `PP = VPP = 1`, and `use_thd=True`.

Qwen3.5, MTP, MRoPE, pipeline/virtual-pipeline parallelism, custom attention
masks, and inference are rejected or outside the initial integration. Qwen3.5
is excluded because its linear-attention layers require an order-preserving CP
layout that is incompatible with MagiAttention's load-balanced permutation.

## Reproducible test environment (Hopper and Blackwell)

The Lite test helpers build MagiAttention v1.1.1 in an isolated, persistent
venv and run an upstream kernel reference case, the operator-level attention
test (`tests/smoke/primitive/test_magi_attention_operator.py`: identical
global varlen Q/K/V through dispatch → core attention → undispatch versus a
per-sequence fp32 SDPA causal reference, forward and dQ/dK/dV, with
tolerances anchored to the bf16 SDPA noise floor), and the Lite Qwen3-MoE
forward/backward/undispatch E2E. The compute capability is auto-detected
(override with `MAGI_ATTENTION_BUILD_COMPUTE_CAPABILITY=90|100`): sm100
(Blackwell) builds the FA4 `flash_attn_cute` kernel backend, while sm90
(Hopper, e.g. H100) prebuilds MagiAttention's native FFA kernels and the
runner selects the matching `MAGI_ATTENTION_KERNEL_BACKEND` automatically:

```bash
export MAGI_ATTENTION_SOURCE=/path/to/MagiAttention-v1.1.1
export MAGI_ATTENTION_VENV=/path/to/magi_attention_test_env_v1.1.1/venv
experimental/lite/tests/setup_magi_attention_env.sh

# Run this inside a single-node 4-GPU allocation.
MLITE_MAGI_NPROC=4 experimental/lite/tests/run_magi_attention_e2e.sh all
```

The runner also accepts `upstream` or `lite` to execute one half of the suite.
