# Mamba-based Language Models

## Introduction

This document is an entrypoint into the code used for
<em>[An Empirical Study of Mamba-based Language Models](https://arxiv.org/abs/2406.07887)</em>.

We are releasing the parameters for some of the models described in that
technical report via
[HuggingFace](https://huggingface.co/collections/nvidia/ssms-666a362c5c3bb7e4a6bcfb9c).
The code in the `main` branch is no longer compatible with the `Mamba2-*`
checkpoints. You can load them using the
[fixed snapshot of the code used for the technical report](https://github.com/NVIDIA/Megatron-LM/tree/ssm/examples/mamba).

## Installation

Create and run a Docker container using the [Dockerfile](./Dockerfile).

```
docker build -t your_image_name:your_tag .
docker run --gpus all -it --rm \
  -v /path/to/megatron:/workspace/megatron \
  -v /path/to/dataset:/workspace/dataset \
  -v /path/to/checkpoints:/workspace/checkpoints \
  -w /workspace/megatron/examples/mamba \
  your_image_name:your_tag
```

## Train

[`train.sh`](./train.sh) is an example pretraining script, showing how to run on
a single node. Select between 800M-scale and 8B-scale models by setting the
`MODEL_SCALE` variable. The 8B-scale hybrid model architecture is the same as
the one described in the technical report.

[`train_2B_A500M_moe_mda_r4.sh`](./train_2B_A500M_moe_mda_r4.sh) is the
`2B_A500M_moe` recipe with Multi-Decay FoX R4. It preserves the scaling-ladder
baseline's 12 Mamba and 12 MoE layers, replaces its three regular attention
layers with `#`, and uses four decay channels with `scaled_basis`, `query_mix`,
and the automatic training kernel. One of the four channels is an exact
no-decay NoPE channel, preserving the baseline attention's full-history view.
For example:

```bash
./train_2B_A500M_moe_mda_r4.sh \
    <per-split-data-args-path> <tokenizer-path>
```

By default, large outputs go under
`~/storage/megatron-lm-mda/2B_A500M_moe_mda_r4`; set `RUN_ROOT` to
override that location or `MDA_ROOT` to select a different `multi-decay-att`
checkout. The launcher defaults to the local four-GPU node layout with EP=4;
set `LOCAL_WORLD_SIZE` and `EXPERT_MODEL_PARALLEL_SIZE` for another allocation.

## Text Generation

Use [`run_text_gen_server_8b.sh`](./run_text_gen_server_8b.sh) to start a text
generation server using an 8B hybrid checkpoint. This is configured to run the
8B hybrid model described in the technical report, with tensor model parallel
set to 1.

The arguments in the script will need to be changed if using a checkpoint with a
different model parallel configuration or other differences, such as model
architecture. For example, to run the 8B pure Mamba-2 model, change
`--hybrid-layer-pattern` to use only `M` symbols (e.g., 56 `M`s for the 8B
model), or remove it entirely.

Use [`run_text_gen_server_8b_gpt3.sh`](./run_text_gen_server_8b_gpt3.sh) to start
a text generation server using the 8B reference Transformer checkpoint.

## Checkpoint Formats

For inference, the model must be configured to match the checkpoint file used,
including the hybrid layer configuration and model parallel configuration.

If you need to convert a hybrid checkpoint file to a different tensor parallel
or pipeline parallel size, use
[the hybrid conversion script](../../tools/checkpoint/hybrid_conversion.py).
There is an example run command at the end of that file.

Before running that script, you will need to set `PYTHONPATH` to include the
root directory of your Megatron-LM repository clone.

```
export PYTHONPATH=<path-to-megatron>:PYTHONPATH
```

## Hybrid Options

`--hybrid-layer-pattern PATTERN` specifies the layer type for every layer in
the model using a string of single-character symbols:

* `M` — Mamba layer
* `*` — Attention layer
* `#` — Multi-Decay FoX attention layer
* `-` — MLP layer
* `E` — MoE layer

The number of layers is derived from the pattern length, so `--num-layers`
should not be specified when `--hybrid-layer-pattern` is used.

For example, the 8B hybrid model described in the technical report uses:

```
--hybrid-layer-pattern "M-M-M--M-M*-M-M-M-M--M*-M-M-M-M-M*--M-M-M-M-M*-M--M-M-M-"
```

This is a 56-layer model with 4 attention layers, 28 MLP layers, and 24 Mamba
layers.

A pure Mamba model uses only `M` symbols (e.g., `MMMMMMMM` for 8 layers).
A pure transformer model uses only `*` and `-` symbols.

### Multi-Decay FoX hybrid layers

The `#` symbol builds Multi-Decay FoX from the companion `multi-decay-att`
checkout while preserving `*` for regular softmax attention. Put that checkout
before the packaged FLA installation on `PYTHONPATH`:

```bash
export PYTHONPATH="$HOME/multi-decay-att:$PYTHONPATH"
```

Then replace the desired attention symbols in the pattern and quote the pattern
so the shell treats `#` literally:

```bash
--position-embedding-type none \
--hybrid-layer-pattern 'M-M-M--M-M#-M-M-M-M--M#-' \
--no-create-attention-mask-in-dataloader \
--multi-decay-num-channels 8 \
--multi-decay-decay-generation scaled_basis \
--multi-decay-decay-type logsigmoid \
--multi-decay-aggregate-mode query_mix \
--multi-decay-training-kernel auto
```

Other controls are `--multi-decay-qkv-bias`, `--multi-decay-qk-norm`,
`--multi-decay-window-size`, `--no-multi-decay-decay-bias`,
`--multi-decay-use-output-gate`, and `--multi-decay-use-nope`. The integration
uses Megatron tensor-parallel projections and distributed checkpoint metadata
while delegating the attention algorithm and backend routing to the companion
`fla.ops.mda` implementation. All channel counts and aggregation modes use the
same SelfAttention-integrated path. In particular, R1 + NoPE omits auxiliary
decay and mixing parameters; with the optional output gate disabled, it retains
the regular attention parameter shell.

The current integration supports full-sequence training and evaluation.
It does not yet support context parallelism, packed sequences, padding/custom
attention masks, or KV-cache decoding.

### Pipeline parallelism

Use `|` to define pipeline stage boundaries for flexible virtual pipeline
parallelism (fVPP). For example, `M-M-|M-M*-|M-M-|M-M*-` defines 4 pipeline
segments. The number of segments must be evenly divisible by
`--pipeline-model-parallel-size`.

### Multi-Token Prediction (MTP)

Use `/` to append MTP layer patterns. Each pattern after the separator
represents one MTP prediction depth. For example, `M*M*/MM/MM` has main
pattern `M*M*` with MTP pattern `MM` repeated for 2 depths.

### Deprecated options

`--hybrid-override-pattern`, `--hybrid-attention-ratio`, and
`--hybrid-mlp-ratio` are deprecated. Use `--hybrid-layer-pattern` instead.

## Mamba vs Mamba-2

This codebase currently only supports Mamba-2, and not the original version of
Mamba. However, the
[fixed snapshot of the code used for the technical report](https://github.com/NVIDIA/Megatron-LM/tree/ssm/examples/mamba)
can be configured to run the original version of Mamba.
