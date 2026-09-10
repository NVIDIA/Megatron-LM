<!-- Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->

# Reproduce balanced DSA partial-graph EP overlap

Run from the Megatron-LM repository root in Bash. These commands use mock THD
data, fixed CP2 and a fixed 512-token local packing capacity. They cover partial
TE graphs; delayed weight gradients and full-iteration graphs are out of scope.

Tested GB200 environment: PyTorch `2.12.0a0+0291f960b6.nv26.04.48445190`,
TE `2.16.0.dev0+01aef4fc`, cuDNN Frontend `1.26.0`, and CUTLASS DSL `4.5.0`.
The Frontend DSA API must expose `q_causal_offsets`; use FlashMLA built for the
target GPU and the model's 64 query heads. H100 BF16 protocol tests used PyTorch
`2.11.0a0+a6c236b9fd.nv26.03.46836102` and TE `2.14.1+366798ef`.
H100 DSA PP4 training also used Frontend `1.26.0`, CUTLASS DSL `4.5.0`,
FlashMLA revision `b7643bd54521f563b839b98289b5cd048c062ba2`, and Hadamard `1.1.0`.
Use GB200 for MXFP8 coverage. Completed training and numerical parity are recorded separately.

The GPU runs and frozen-weight comparisons below were collected on the follow-up
history ending at `d0ce976bf`, before rebasing onto #6058 at `61475056e`. That
dependency update changes compact indexer scoring and sparse-loss prediction
precision. The rebase includes local protocol/liveness checks and a regression
for the relocated shared backward-weight wrapper; GPU training and numerical
comparisons have not been rerun on the rebased code.

## Four-GPU ordinary PP2 training

```bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
common=(
  --hidden-size 256 --num-attention-heads 64 --ffn-hidden-size 512
  --max-position-embeddings 1024 --tokenizer-type NullTokenizer
  --vocab-size 1024 --make-vocab-size-divisible-by 128
  --normalization RMSNorm --swiglu --disable-bias-linear
  --untie-embeddings-and-output-weights --position-embedding-type rope --rotary-base 10000
  --multi-latent-attention --q-lora-rank 128 --kv-lora-rank 448
  --qk-pos-emb-head-dim 64 --v-head-dim 512 --o-groups 8 --o-lora-rank 128
  --experimental-attention-variant dsv4_hybrid --csa-window-size 128
  --dsa-indexer-n-heads 64 --dsa-indexer-head-dim 128 --dsa-indexer-topk 128
  --dsa-indexer-loss-coeff 0.01 --dsa-indexer-use-sparse-loss
  --dsa-kernel-backend cudnn --dsa-cp-balance-indexer
  --num-experts 4 --moe-ffn-hidden-size 256 --moe-router-topk 2 --moe-router-dtype fp32
  --moe-grouped-gemm --moe-permute-fusion --moe-token-dispatcher-type alltoall
  --tensor-model-parallel-size 1 --context-parallel-size 2
  --expert-model-parallel-size 2 --expert-tensor-parallel-size 1 --cp-partition-mode contiguous
  --bf16 --attention-dropout 0 --hidden-dropout 0 --micro-batch-size 1 --global-batch-size 8
  --seq-length 1024 --use-varlen-dataset --mock-data
  --varlen-mock-dataset-config-json '{"mode":"distribution","type":"lognormal","format":"thd","min_seq_len":128,"max_seq_len":1024,"mean_seq_len":512,"lognormal_sigma":0.8}'
  --sequence-packing-scheduler dp_balanced --max-seqlen-per-dp-cp-rank 512
  --pad-packed-seq-alignment max --thd-max-packed-sequences 8 --calculate-per-token-loss
  --split 99,1,0 --eval-iters 1 --eval-interval 10 --eval-global-batch-size 8
  --num-workers 2 --no-create-attention-mask-in-dataloader --log-interval 1
  --logging-level 20 --enable-experimental --te-rng-tracker --distributed-timeout-minutes 3
)
train=(--train-iters 35 --lr 0.00001 --min-lr 0.000001 --lr-decay-iters 40 --lr-warmup-iters 1)
gpt=(--num-layers 8 --csa-compress-ratios '[4,128,4,0,4,128,4,0]' --pipeline-model-parallel-size 2)
pcg=(--cuda-graph-impl transformer_engine --cuda-graph-modules attn
     --cuda-graph-dynamic-microbatches --cuda-graph-warmup-steps 2)
torchrun --standalone --nproc-per-node=4 pretrain_gpt.py \
  "${common[@]}" "${train[@]}" "${gpt[@]}" "${pcg[@]}" --overlap-moe-expert-parallel-comm
```

Check capture after iteration 2, changing physical microbatch counts, 35 completed
iterations, and evaluation after iterations 10/20/30 followed by resumed training.
The physical microbatch count comes from packing; it need not equal global batch size.

## Configuration variants

- **GPT PP1:** replace `gpt` with `(--num-layers 4 --csa-compress-ratios '[4,128,4,0]' --pipeline-model-parallel-size 1)`.
- **GPT PP2/VPP2:** retain the PP2 recipe and add `--num-virtual-stages-per-pipeline-rank 2`; optionally replace graph modules with `attn moe_router moe_preprocess`.
- **Hybrid PP2/VPP2:** use `pretrain_hybrid.py`, eight layers, ratios `[4,0,128,0,0,0,0,0]`, and `--hybrid-layer-pattern 'CE|HE|W-|E-' --spec megatron.core.models.hybrid.hybrid_layer_specs hybrid_dsv4_stack_spec --enable-hyper-connections`. The four pattern chunks derive VPP2 for PP2; do not add the GPT VPP flag. Exercise both graph scopes above.
- **Hybrid ordinary PP4:** keep that eight-layer Hybrid pattern and set PP4 with eight total GPUs. Four chunks now mean one chunk per PP rank. Attention-only capture leaves the final `E-` stage entirely eager, exercising capture-capacity agreement on a rank with no graphs. On one eight-GPU node use `--nproc-per-node=8`; on two four-GPU nodes use standard torchrun `--nnodes=2 --node-rank=0/1 --master-addr=<rank0-host> --master-port=29567` without `--standalone`.
- **Short schedules:** set `--global-batch-size 1` on PP2/PP4, retaining `--eval-global-batch-size 8`.
- **MXFP8/mHC:** on GB200 add `--fp8-format e4m3 --fp8-recipe mxfp8 --enable-hyper-connections`; omit the last flag to test without mHC.

## Focused regressions and evidence

```bash
torchrun --standalone --nproc-per-node=8 -m pytest \
  tests/unit_tests/a2a_overlap/test_pipeline_schedule.py --experimental -q --tb=short
torchrun --standalone --nproc-per-node=2 -m pytest \
  tests/unit_tests/a2a_overlap/test_hybrid_schedule.py --experimental -q --tb=short
python -m pytest tests/unit_tests/transformer/moe/test_moe_padding_mask.py \
  tests/unit_tests/transformer/test_thd_cuda_graph.py::TestCaptureReset --experimental -q
torchrun --standalone --nproc-per-node=2 -m pytest \
  tests/unit_tests/transformer/test_thd_cuda_graph.py \
  -k 'GraphDynamicRouteMetadataArena or DynamicMicrobatchSlots or balanced_dynamic' --experimental -q
```

Recorded execution includes GPT ordinary PP2 and PP2/VPP2 BF16 for 35 iterations,
Hybrid PP1 and PP2/VPP2 mHC BF16 for 35 iterations, and GPT/Hybrid PP1 MXFP8
cases for five iterations. On eight H100s, Hybrid ordinary PP4/CP2/EP2 with mHC
completes 35 iterations, including an eager-only pipeline stage. Hybrid VPP2
also completes five iterations with attention-only capture and eager-only chunks.
The H100 ordinary-attention PP2/PP4 regression passes loss/every-local-gradient
comparisons for fixed and varying shapes, including N1 and steady-state schedules.
Four GB200s also pass the six combined Hybrid regressions for sequence-parallel
RNG behavior, FP8/BF16 input lifetimes and non-final pipeline output deallocation,
with mHC enabled/disabled where applicable. The TP2/SP tests use ordinary E/dense
stacks; DSv4 attention itself remains restricted to TP1.
These results do not establish strict native DSv4 gradient parity. Native dump
comparisons have differences under investigation. In fixed-weight GPT BF16 controls,
all parameter dumps are byte-identical and every parameter passes the L2 check;
the strict elementwise check still reports differences before and after capture.

A separate four-GB200 Nsight Systems `2026.2.1` trace confirms actual EP/PCG
concurrency. CUDA correlation IDs and enclosing MoE NVTX ranges identify EP
kernels independently of CP collectives; graph compute requires a nonzero graph
node ID. EP overlaps graph compute by 278.015 and 376.928 microseconds on two
GPUs. The conservative filter observes no such overlap on the other two GPUs;
these intervals establish concurrency, not throughput improvement.

## Isolate numerics with unchanged weights

Use the PP1 `gpt` array above and fresh output directories. Replace `train` with:

```bash
gpt=(--num-layers 4 --csa-compress-ratios '[4,128,4,0]' --pipeline-model-parallel-size 1)
train=(--train-iters 3 --lr 0 --min-lr 0 --weight-decay 0 --seed 1234
       --lr-decay-iters 40 --lr-warmup-iters 1 --save-params-interval 1
       --save-wgrads-interval 1 --save-interval 1000 --no-save-optim --no-save-rng)
eager=(--cuda-graph-impl none --cuda-graph-modules attn
       --cuda-graph-dynamic-microbatches --cuda-graph-warmup-steps 2)
torchrun --standalone --nproc-per-node=4 pretrain_gpt.py "${common[@]}" "${train[@]}" "${gpt[@]}" \
  "${eager[@]}" --save results/fixed-eager
torchrun --standalone --nproc-per-node=4 pretrain_gpt.py "${common[@]}" "${train[@]}" "${gpt[@]}" \
  "${eager[@]}" --overlap-moe-expert-parallel-comm --save results/fixed-overlap
torchrun --standalone --nproc-per-node=4 pretrain_gpt.py "${common[@]}" "${train[@]}" "${gpt[@]}" \
  "${pcg[@]}" --overlap-moe-expert-parallel-comm --save results/fixed-pcg
```

Native `params` dumps are written after the optimizer step and `wgrads` before it;
LR0/WD0 keeps weights fixed. Verify identical named parameter tensors first, then
compare named gradients at each iteration, separating indexer, attention, router,
experts and mHC parameters. Iteration 3 is the first PCG replay. Training completion
and similar scalar losses are insufficient to declare this comparison passing.

The diagnostic uses unchanged `rtol=0.02`, `atol=1e-6`, requiring both elementwise
agreement and a per-parameter bound
`||candidate-reference|| <= rtol*||reference|| + atol*sqrt(numel)`.
Frozen runs have byte-identical parameter dumps at all three iterations.
Independent repeats use the same eager configuration, inputs and unchanged weights.
These TP1 runs, without BF16 exclusions inside FP8 models, used `7be9cfa79`.
The later SP/BF16-boundary fixes are covered by the dedicated regressions above.

| Model and precision | Step-3 overlap vs. PCG global relative L2 | Step-3 eager repeat global relative L2 |
| --- | ---: | ---: |
| GPT BF16, mHC off | 0.0010591 | 0.0010153 |
| Hybrid BF16, mHC on | 0.0009042 | 0.0009260 |
| GPT MXFP8, mHC on | 0.0046102 | 0.0046625 |
| Hybrid MXFP8, mHC off | 0.0031514 | 0.0036473 |
| Hybrid MXFP8, mHC on | 0.0028532 | Not measured |

All these comparisons, including the independent eager repeats, fail the strict
elementwise criterion. GPT BF16 and Hybrid MXFP8 without mHC pass every
per-parameter L2 bound; near-zero mHC scalar gradients can also fail that bound
in both execution modes. For
example, Hybrid BF16's first-layer `alpha_pre` differs by `6.82e-6` between
overlap and PCG, versus `1.88e-6` in one eager repeat. Identical initial residual
streams followed by RMSNorm make this a cancellation-sensitive scaling direction;
the experiment does not isolate the source of run variability or establish a
statistical bound from one repeat. These diagnostic failures remain visible and
must not be reported as strict numerical parity.
