"""SPMD contracts for Megatron operators and custom autograd functions."""

from __future__ import annotations

import inspect

import spmd_types as spmd

from megatron.core import parallel_state
from megatron.core import utils as megatron_utils
from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.fusions import fused_bias_gelu, fused_bias_swiglu, fused_cross_entropy
from megatron.core.tensor_parallel import cross_entropy as tp_cross_entropy
from megatron.core.tensor_parallel import layers as tp_layers
from megatron.core.tensor_parallel import mappings as tp_mappings
from megatron.core.transformer.moe import moe_utils


def rule_for(function):
    """Attach a type rule to an autograd function without changing its behavior."""

    def decorate(rule):
        function.spmd_typecheck = staticmethod(rule)
        return rule

    return decorate


@rule_for(tp_mappings._ReduceFromModelParallelRegion)
def _reduce_from_model_parallel_region(output, *, input_, group):
    """Type the all-reduce forward and identity backward."""
    spmd.assert_type(input_, {group: spmd.V})
    # I adds a gradient promise on top of R. A result with no gradient, such
    # as router token counts, is typed R so it can mix freely with V.
    output_type = spmd.I if output.requires_grad else spmd.R
    spmd.assert_local_type_like(output, input_, {group: output_type})


@rule_for(tp_mappings._ScatterToSequenceParallelRegion)
def _scatter_to_sequence_parallel_region(output, *, input_, group):
    """Type the sequence split and gather backward."""
    spmd.assert_type(input_, {group: spmd.I})
    spmd.assert_local_type_like(output, input_, {group: spmd.S(0)})


@rule_for(tp_mappings._GatherFromSequenceParallelRegion)
def _gather_from_sequence_parallel_region(output, *, input_, group, tensor_parallel_output_grad):
    """Type the sequence gather and its selectable gradient collective."""
    spmd.assert_type(input_, {group: spmd.S(0)})
    output_type = spmd.R if tensor_parallel_output_grad else spmd.I
    spmd.assert_local_type_like(output, input_, {group: output_type})


@rule_for(tp_mappings._ReduceScatterToSequenceParallelRegion)
def _reduce_scatter_to_sequence_parallel_region(output, *, input_, group):
    """Type the sequence reduce-scatter and gather backward."""
    spmd.assert_type(input_, {group: spmd.V})
    spmd.assert_local_type_like(output, input_, {group: spmd.S(0)})


@rule_for(tp_layers.LinearWithGradAccumulationAndAsyncCommunication)
def _linear_with_grad_accumulation(
    output, *, input, weight, bias, allreduce_dgrad, sequence_parallel, tp_group
):
    """Type the tensor-parallel contract of the completed trainable linear."""
    tp_axis = spmd.normalize_axis(tp_group)
    spmd.assert_type(weight, {tp_axis: spmd.V})
    if bias is not None:
        spmd.assert_type(bias, {tp_axis: spmd.V})
    if allreduce_dgrad:
        input_type = spmd.I
    elif sequence_parallel:
        input_type = spmd.S(0)
    else:
        input_type = spmd.V
    spmd.assert_type(input, {tp_axis: input_type})
    spmd.assert_local_type_like(output, input, {tp_axis: spmd.V})


@rule_for(tp_cross_entropy._VocabParallelCrossEntropy)
def _vocab_parallel_cross_entropy(loss, *, vocab_parallel_logits, target):
    """Type the vocab-sharded softmax, which all-reduces over the global TP group."""
    tp_group = parallel_state.get_tensor_model_parallel_group()
    spmd.assert_type(vocab_parallel_logits, {tp_group: spmd.S(-1)})
    spmd.assert_type(target, {tp_group: spmd.I})
    spmd.assert_local_type_like(loss, target)


@rule_for(fused_cross_entropy._VocabParallelCrossEntropy)
def _fused_vocab_parallel_cross_entropy(loss, *, vocab_parallel_logits, target, tp_group):
    """Type the fused vocab-sharded softmax, which all-reduces over ``tp_group``."""
    spmd.assert_type(vocab_parallel_logits, {tp_group: spmd.S(-1)})
    spmd.assert_type(target, {tp_group: spmd.I})
    spmd.assert_local_type_like(loss, target)


@rule_for(moe_utils.MoEAuxLossAutoScaler)
def _moe_aux_loss_autoscaler(result, *, output, aux_loss):
    """The forward is the identity on ``output``.

    ``aux_loss`` is saved for backward and does not affect the forward value.
    """
    spmd.rules.ignore(aux_loss)
    spmd.rules.output(result, output)


def _named_non_tensor_args(unpacker, values):
    """Name the values immediately unpacked into an implementation's locals."""
    first_local = len(inspect.signature(unpacker).parameters)
    names = unpacker.__code__.co_varnames[first_local : first_local + len(values)]
    if len(names) != len(values):
        raise NotImplementedError(
            f"Could not identify all non-tensor arguments for {unpacker.__qualname__}"
        )
    return dict(zip(names, values))


def _assert_te_linear_types(out, inp, weight, bias, weight_workspace, options):
    """State the TP contract shared by Transformer Engine linear functions."""
    tp_group = options["tp_group"]
    parallel_mode = options["parallel_mode"]
    # The input workspace is a cached, possibly quantized copy of ``weight``.
    spmd.rules.ignore(weight_workspace)
    if parallel_mode is None:
        # Megatron passes None when it performs TP communication itself.
        # There is no TE-owned TP weight layout to assert in this mode.
        spmd.assert_type(inp, {})
        spmd.assert_type(weight, {})
        bias_type = {}
        spmd.assert_local_type_like(out, inp)
    elif parallel_mode == "column":
        input_type = spmd.S(0) if options["sequence_parallel"] else spmd.I
        spmd.assert_type(inp, {tp_group: input_type})
        spmd.assert_type(weight, {tp_group: spmd.S(0)})
        bias_type = {tp_group: spmd.S(0)}
        spmd.assert_local_type_like(out, inp, {tp_group: spmd.S(-1)})
    elif parallel_mode == "row":
        output_type = spmd.S(0) if options["sequence_parallel"] else spmd.I
        spmd.assert_type(inp, {tp_group: spmd.S(-1)})
        spmd.assert_type(weight, {tp_group: spmd.S(1)})
        # The bias is added after the reduction, to each rank's sequence shard
        # under sequence parallelism.
        bias_type = {tp_group: spmd.R if options["sequence_parallel"] else spmd.I}
        spmd.assert_local_type_like(out, inp, {tp_group: output_type})
    else:
        raise NotImplementedError(f"Unsupported Transformer Engine linear mode {parallel_mode!r}")
    # TE passes an empty tensor when the layer has no bias.
    if bias is not None and bias.numel() > 0:
        spmd.assert_type(bias, bias_type)
    else:
        spmd.rules.ignore(bias)


def _te_linear(output, *, weight, weight_workspace, inp, bias, non_tensor_args):
    from transformer_engine.pytorch.module import linear as te_linear

    out, weight_workspace_out = output
    options = _named_non_tensor_args(te_linear._linear_forward_impl, non_tensor_args)
    _assert_te_linear_types(out, inp, weight, bias, weight_workspace, options)
    if weight_workspace_out is not None:
        spmd.assert_local_type_like(weight_workspace_out, weight)


def _te_layernorm_linear(
    output, *, inp, ln_weight, ln_bias, weight, weight_workspace, bias, non_tensor_args
):
    from transformer_engine.pytorch.module.layernorm_linear import _LayerNormLinear

    out, ln_out, weight_workspace_out = output
    options = _named_non_tensor_args(_LayerNormLinear.forward, non_tensor_args)
    for norm_parameter in (ln_weight, ln_bias):
        if norm_parameter is not None:
            if options["parallel_mode"] is None:
                spmd.assert_type(norm_parameter, {})
            else:
                # With sequence parallelism each rank normalizes its own sequence
                # shard, so the norm's gradient is partial until Megatron sums it.
                norm_type = spmd.R if options["sequence_parallel"] else spmd.I
                spmd.assert_type(norm_parameter, {options["tp_group"]: norm_type})
    _assert_te_linear_types(out, inp, weight, bias, weight_workspace, options)
    if ln_out is not None:
        if options["return_layernorm_output_gathered"]:
            spmd.assert_local_type_like(ln_out, inp, {options["tp_group"]: spmd.R})
        else:
            spmd.assert_local_type_like(ln_out, inp)
    if weight_workspace_out is not None:
        spmd.assert_local_type_like(weight_workspace_out, weight)


def _te_checkpoint(outputs, *, distribute_saved_activations, tp_group, args):
    """Type a checkpointed region as preserving its first input's layout.

    The checker does not see inside ``run_function``. Megatron checkpoints whole
    transformer layers, whose output hidden states are laid out like their input.

    Distributing saved activations keeps only this rank's chunk of ``args[0]``
    and all-gathers it over ``tp_group`` before recomputing, which restores the
    input only if it was the same on every rank of that group.
    """
    if distribute_saved_activations:
        spmd.assert_type(args[0], {tp_group: spmd.I})
    for out in outputs if isinstance(outputs, tuple) else (outputs,):
        spmd.assert_local_type_like(out, args[0])
    spmd.rules.ignore(*args)


LOCAL_AUTOGRAD_FUNCTIONS = [
    megatron_utils.MakeViewlessTensor,
    fused_bias_gelu.GeLUFunction,
    fused_bias_swiglu.SwiGLUFunction,
    moe_utils.RouterGatingLinearFunction,
]

if HAVE_TE:
    from transformer_engine.pytorch.attention.dot_product_attention.backends import FusedAttnFunc
    from transformer_engine.pytorch.attention.dot_product_attention.softmax import (
        ScaledMaskedSoftmax,
    )
    from transformer_engine.pytorch.distributed import _CheckpointFunction
    from transformer_engine.pytorch.module.layernorm_linear import _LayerNormLinear
    from transformer_engine.pytorch.module.linear import _Linear
    from transformer_engine.pytorch.ops.fuser import _OperationFuserAutogradFunction

    rule_for(_Linear)(_te_linear)
    rule_for(_LayerNormLinear)(_te_layernorm_linear)
    rule_for(_CheckpointFunction)(_te_checkpoint)
    LOCAL_AUTOGRAD_FUNCTIONS += [
        FusedAttnFunc,
        ScaledMaskedSoftmax,
        _OperationFuserAutogradFunction,
    ]

for function in LOCAL_AUTOGRAD_FUNCTIONS:
    spmd.register_local_autograd_function(function)
