# Shared-parameter looped decoder training

`TransformerBlock` can execute a contiguous middle segment repeatedly while
registering each physical decoder layer only once. This provides a limited
training execution primitive, not full Dory architecture support.

## Configuration

```python
import torch
from megatron.core.transformer import TransformerConfig

config = TransformerConfig(
    num_layers=4,                 # physical layers, not effective compute depth
    hidden_size=256,
    num_attention_heads=4,
    bf16=True,
    params_dtype=torch.bfloat16,
    decoder_repeat_start=1,        # inclusive, zero based
    decoder_repeat_end=3,          # exclusive
    decoder_num_repeats=3,         # total executions, including the first pass
)
assert config.get_decoder_layer_execution_order() == (0, 1, 2, 1, 2, 1, 2, 3)
```

The corresponding training CLI options are `--decoder-repeat-start`,
`--decoder-repeat-end`, and `--decoder-num-repeats`. These fields are exposed by
the existing TransformerConfig argument generator. The default repeat count is
one and preserves ordinary execution; range endpoints can both be omitted in
that case. Repeated execution requires both endpoints, with
`0 <= start < end <= num_layers`. Empty prefixes and suffixes are allowed.

`layers.1` and `layers.2` in this example keep their original Parameter objects
and physical layer numbers on every invocation. Autograd sums all loop
contributions into their gradients. DDP registers hooks once per physical
parameter, and the optimizer updates each parameter once per training step.
Final layer normalization runs once, after the suffix.

Parameter count, optimizer state allocation, model checkpoint keys and sharded
checkpoint layer axes depend on physical depth, not repetition count. Changing
the repeat count alone does not create new weights or optimizer states. Save
and restore the execution configuration with the checkpoint: weight keys alone
do not encode the repeat count or range. Resuming the same training computation
requires the same execution configuration.

## Initial scope

The implementation supports dense BF16 TransformerBlock training with
TP=PP=CP=EP=1 and DP=1 or 2. Ordinary Megatron DDP and its distributed optimizer
are supported with gradient reduction and parameter gathering overlap disabled.
The ordinary eager decoder forward path owns this execution plan. Stateless
evaluation through the same forward path is allowed.

Configurations outside this initial scope fail early: recomputation, FP8/FP4
and per-module quantization recipes, Kitchen, CPU/fine-grained activation
offloading, CUDA graphs, heterogeneous blocks, MLA, MTP, mHC/wide residuals,
virtual/custom pipeline layouts, weight sharding, FSDP, delayed weight-gradient
execution, schedule-plan execution, inference KV caches, and intermediate layer
extraction. HybridModel rejects these options because its stack does not execute
TransformerBlock's plan. Tensor-parallel/pipeline-parallel execution, cached
inference and additional precision/recompute combinations need separate work.

This path can be exercised through GPTModel, which upstream has deprecated for
new features. The API placement and supported boundary need maintainer agreement;
this change does not claim support for the preferred HybridModel API or import
of official Dory checkpoints.

## Validation

The unit tests compare real dense BF16 outputs, all physical parameter gradients
and an Adam update against a separately constructed flat model with tied
reference Parameters. They also cover two-microbatch accumulation, Megatron DDP
FP32 gradient buffers, FP32 master weights, optimizer state size, sharded model
keys, distributed checkpoint save/load, and equality after the next update.
The numerical unit cases require a one-rank GPU invocation:

```bash
python -m torch.distributed.run --standalone --nproc-per-node=1 -m pytest -q \
  tests/unit_tests/transformer/test_looped_decoder.py
```

Run the rank-distinct global-batch acceptance harness on one GPU, then two
**distinct** GPUs:

```bash
PYTHONPATH=. python -m torch.distributed.run --standalone --nproc-per-node=1 \
  tests/functional_tests/python_test_utils/looped_decoder_smoke.py \
  --output-dir /tmp/looped-dp1

PYTHONPATH=. python -m torch.distributed.run --standalone --nproc-per-node=2 \
  tests/functional_tests/python_test_utils/looped_decoder_smoke.py \
  --output-dir /tmp/looped-dp2
```

The harness tests repeat counts 1/2/3 with ordinary and distributed Adam, uses
rank-distinct causal language modeling samples, and compares DP gradients and
updates to a single-model reference accumulating the equivalent global batch.
For distributed Adam it checks only each rank's owned gradient shards and sums
optimizer moment sizes across DP. It then serializes a combined distributed
model/optimizer checkpoint and verifies the next resumed update exactly.

Development validation used one RTX 4090 with PyTorch 2.12.1+cu130 and the local
backend in an AutoDL container, rather than the project's pinned CI image.
Two physical GPU NCCL validation and Transformer Engine validation remain
required before claiming those environment qualifications. A two-process Gloo
smoke sharing one GPU, with an external adapter disabling CUDA communication
coalescing unsupported by Gloo, is supplemental diagnostic evidence only.
