# MoE auxiliary normalization with masked SFT targets

With `--calculate-per-token-loss`, training sums main-loss gradients and divides
all model gradients by the global number of scored targets, `T`. The router
also seeds an auxiliary gradient multiplied by its input-token count. Without
an independent correction, this divides a summed input-token auxiliary gradient
by `T`, rather than by the number of real input tokens, `N`.

If the intended objective is a scored-token mean main loss plus an input-token
mean routing regularizer, the auxiliary coefficient therefore changes by `N/T`.
Masking more prompt targets can change this coefficient even with identical
inputs, router probabilities, and routing decisions. This is separate from
excluding padding from router statistics, CP/TP multiplicity, and cumulative
`global_aux_loss` counts.

For example, two DP replicas each process 200 real input tokens. Their scored
target counts are 60 and 140. An input-token regularizer's gradient should be
`(g0 + g1)/2`; the uncorrected per-token path instead gives `g0 + g1`.

## Opt-in correction for the standard GPT SFT path

Add `--moe-aux-loss-input-token-normalization` alongside `--sft` and
`--calculate-per-token-loss`. The training loop buffers one optimizer step of
CPU batches, counts `T` and `N` across DP, and uses the existing
`moe_grad_scale_func` hook to multiply the auxiliary seed by `max(T,1)/max(N,1)`.
The common finalizer then leaves the auxiliary gradient divided by `N`, without
changing the main gradient. Existing AMP/custom MoE scales are preserved. The
normalization choice is opt-in to preserve existing training behavior.

Counts include every microbatch in the step. A local or per-microbatch ratio
would target-weight the routing objective and does not implement input-token
normalization when target fractions vary. The scale remains constant during
the schedule, including recomputation. On a rerun, prefetch reads from the
rewound original iterator. Prefetch does not retain GPU forward graphs.

Only padding is excluded from routing: prompt tokens remain included even when
their next-token loss is masked. The GPT forward passes a padding mask derived
from a distinct PAD token ID. PAD and EOD sharing an ID is rejected because
token IDs alone then cannot distinguish real EOD tokens from padding.

This initial integration requires PP=1, the standard GPT SFT iterator (before CP
slicing), and no dynamic sequence scheduler or GTP weight rematerialization.
Unsupported combinations fail explicitly. The helper uses explicit TP/DP groups;
it supports fixed CP by counting complete sequences and reducing across DP
without double-counting CP replicas. This does not change the definition of
sequence/global load-balancing statistics or fix independent router issues.

Applications with their own training loop can use
`prepare_moe_loss_normalization` and `input_token_moe_loss_scaling`, provided the
counted input mask exactly matches the router's mask and the main finalizer uses
the counted target mask. The hook was introduced by
[PR #3956](https://github.com/NVIDIA/Megatron-LM/pull/3956); it alone does not
compute these counts or install the correction in the standard training loop.

## Regression test

From the repository's supported development container/environment:

```bash
uv run python -m torch.distributed.run --nproc-per-node=4 -m pytest \
  tests/unit_tests/training/test_moe_loss_normalization.py -q
```

The test uses the actual router and scheduling scale helper. It compares
auxiliary gradients against an independently input-weighted reference, with
uneven DP/microbatch target fractions, padding, CP=1/2, auxiliary/z losses, and
steps with no scored targets. It also verifies the uncorrected gradient's
`N/max(T,1)` bias and checks the main gradient against its scored-token mean.
These focused tests are not a full-model convergence or performance benchmark.

A small full-training smoke test (four GPUs, two optimizer steps) is:

```bash
uv run python -m torch.distributed.run --nproc-per-node=4 pretrain_gpt.py \
  --num-layers 1 --hidden-size 64 --ffn-hidden-size 128 --num-attention-heads 4 \
  --seq-length 128 --max-position-embeddings 128 --num-experts 4 --moe-router-topk 2 \
  --moe-aux-loss-coeff 0.1 --moe-z-loss-coeff 0.01 --moe-grouped-gemm \
  --moe-token-dispatcher-type alltoall --transformer-impl transformer_engine \
  --disable-bias-linear --bf16 --calculate-per-token-loss --micro-batch-size 1 \
  --global-batch-size 8 --train-iters 2 --lr 0.0001 --min-lr 0.0001 \
  --lr-decay-style constant --eval-iters 0 --eval-interval 1000 --log-interval 1 \
  --mock-data --sft --no-create-attention-mask-in-dataloader \
  --tokenizer-type NullTokenizer --vocab-size 256 --null-tokenizer-pad-id 0 \
  --position-embedding-type rope --moe-aux-loss-input-token-normalization
```

The command uses DP=4. Add `--context-parallel-size 2` for CP=2/DP=2, or
`--tensor-model-parallel-size 2 --sequence-parallel` for TP=2/DP=2. The packed
SFT path supplies sequence boundaries rather than a dense attention mask; the
`--no-create-attention-mask-in-dataloader` option is necessary for a consistent
TP broadcast protocol. The synthetic data checks execution; the unit test
above provides the masked-gradient correctness comparison.
