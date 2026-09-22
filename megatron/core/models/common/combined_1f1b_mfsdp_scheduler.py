# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.module import FsdpModule
from megatron.core.utils import get_attr_wrapped_model


def _make_unshard_forward_hook(owner: FsdpModule):
    """Create a forward pre-hook that unshards the owning FSDP module before the submodule
    forward."""

    def hook(submodule, _args, _kwargs):
        if owner.is_root():
            context = owner.context
            context.allgather_stream.wait_stream(context.current_stream())
        owner.unshard()

    return hook


def _make_unshard_backward_hook(owner: FsdpModule):
    """Create a backward pre-hook that unshards the owning FSDP module before the submodule
    backward."""

    def hook(submodule, _grad_output):
        owner.unshard()

    return hook


def _module_post_backward_hook(module: FsdpModule) -> None:
    module.reshard()
    module._reduce_gradient_groups()


def reshard_fsdp_module(module: FsdpModule) -> None:
    """Reshard the FSDP module after fine-grained computation."""
    assert isinstance(module, FsdpModule), "Expected an FsdpModule."
    module.reshard()


def register_combined_1f1b_hooks(module: FsdpModule) -> None:
    """Install the sub-module hooks and gradient multiplicity for combined 1F1B."""

    # Nearest owning FSDP unit per module (first parent wins), recorded by the walk
    # below: unshard/reshard act on an FSDP *unit*, so a module that consumes another
    # module's parameter needs the unit owning that parameter, not its own nearest.
    owners = {}

    def register_hooks(submodule, owner):
        if isinstance(submodule, FsdpModule):
            owner = submodule  # BEFORE registering: an FSDP unit owns itself
        owners.setdefault(submodule, owner)
        if len(list(submodule.parameters(recurse=False))) > 0:
            submodule.register_forward_pre_hook(
                _make_unshard_forward_hook(owner), prepend=True, with_kwargs=True
            )
            submodule.register_full_backward_pre_hook(_make_unshard_backward_hook(owner))
        for child in submodule.children():
            register_hooks(child, owner)

    assert isinstance(module, FsdpModule), "Owner must be an FsdpModule."

    # This runs once per model chunk, so the declaration derives from that chunk's own
    # ``pre_process``/``mtp_process``. Those flags live on the inner model: MCore wraps
    # it in the mixed-precision wrapper before FSDP and ``Float16Module`` defines no
    # ``__getattr__``, hence the shared unwrapping helper instead of reading them off
    # the FSDP unit.
    model = get_attr_wrapped_model(module, "pre_process", return_model_obj=True)
    register_hooks(module, module)
    _register_borrowed_weight_unshard_hooks(model, owners)

    extras = _window_extras(model)
    for submodule in module.modules():
        if not isinstance(submodule, FsdpModule):
            continue
        submodule.param_grad_readiness.expected.update(_unit_grad_multiplicity(submodule, extras))
        submodule.register_post_backward_hook(_module_post_backward_hook)


def _register_borrowed_weight_unshard_hooks(model, owners) -> None:
    """Unshard the borrowed weight for a tied output projection that owns no parameter.

    ``gpt_model`` builds that projection with
    ``skip_weight_param_allocation=pre_process and share_embeddings_and_output_weights``
    and ``bias=False``, so the hook walk skips it and nothing re-unshards the embedding
    weight before the PostProcessNode backward, the first backward consumer of it. That
    backward then reads storage the post-forward reshard released:
    ``RuntimeError: The tensor has a non-zero number of elements, but its data is not
    allocated yet`` (tensor_parallel/layers.py:760). ``dbuffer.release_storage`` is not
    at fault -- it keeps the Storage object so a later reallocate restores saved aliases.

    Narrow on purpose: only a tied projection that really owns nothing gets the hooks,
    so the untied path registers exactly what it registered before.
    """
    if not model.share_embeddings_and_output_weights:
        return
    output_layer = getattr(model, 'output_layer', None)
    if output_layer is None or list(output_layer.parameters(recurse=False)):
        return
    embedding = getattr(model, 'embedding', None)
    owner = owners.get(getattr(embedding, 'word_embeddings', None) or embedding)
    if owner is None:
        return
    output_layer.register_forward_pre_hook(
        _make_unshard_forward_hook(owner), prepend=True, with_kwargs=True
    )
    output_layer.register_full_backward_pre_hook(_make_unshard_backward_hook(owner))


def _window_extras(model) -> list:
    """Extra backward contributions per weight for one backward window.

    A parameter accumulates its gradient once per schedule node that consumes it in its
    own GraphTask, and nearly every parameter has exactly one such node -- the base
    count of 1 in ``_unit_grad_multiplicity``. Three groups have more:

      1. The output projection: the ``post_process`` node and the ``mtp_post_process``
         node (one per MTP layer) each consume it. When the weights are tied it runs
         against the embedding weight instead of one of its own, so both land there.
      2. The embedding: one MTP pre-dispatch node per MTP layer consumes it
         (``_get_embeddings``).
      3. The second loss node's own weights: the MTP layer's
         ``enorm``/``hnorm``/``eh_proj``/``final_layernorm`` and the decoder's final
         layernorm. The last one is easy to miss -- an ordinary decoder parameter, but
         with MTP it feeds both the main loss and the MTP head.

    Groups 1 and 3 have a second consumer only on the interleaved schedule, where each
    node's backward runs as its own GraphTask: ``schedules.py:162-169`` selects it
    exactly when ``pipeline_model_parallel_size > 1``. At PP=1 the two loss nodes share
    one GraphTask and fire once between them.

    Entries are ``(weight, extra contributions)``; the matcher sums every entry a
    parameter matches.
    """
    tied = model.share_embeddings_and_output_weights and (model.pre_process or model.mtp_process)
    mtp_depth = _active_mtp_layers(model)
    interleaved = model.config.pipeline_model_parallel_size > 1

    embedding_weight = _resolve(model, 'embedding', 'word_embeddings', 'weight')
    # Group 1 consumes the embedding weight when tied, the projection's own otherwise.
    projection_weight = embedding_weight if tied else _resolve(model, 'output_layer', 'weight')

    extras = []
    if tied:
        extras.append((embedding_weight, 1))
    if mtp_depth:
        extras.append((embedding_weight, mtp_depth))
    if mtp_depth and interleaved:
        extras.extend(
            (weight, 1)
            for weight in (
                projection_weight,
                *_mtp_layer_weights(model),
                _resolve(model, 'decoder', 'final_layernorm', 'weight'),
            )
            if weight is not None
        )
    return extras


def _resolve(obj, *path):
    """``getattr`` along ``path``; any missing link resolves to ``None``."""
    for name in path:
        obj = getattr(obj, name, None)
        if obj is None:
            return None
    return obj


def _active_mtp_layers(module) -> int:
    """Return whether THIS pipeline stage runs MTP (0 or 1)."""
    depth = getattr(module.config, 'mtp_num_layers', None) or 0
    if depth == 0 or not module.mtp_process:
        return 0
    assert depth == 1, (
        "overlap_moe_expert_parallel_comm requires mtp_num_layers <= 1 "
        "(transformer_config.py:3485-3489); per-parameter multiplicity does not "
        f"model deeper MTP (got {depth})."
    )
    return 1


def _matches_fsdp_parameter(fsdp_parameter, weight) -> bool:
    """Return whether ``fsdp_parameter`` is ``weight``."""
    if weight is None:
        return False
    return fsdp_parameter.unsharded is weight or fsdp_parameter.sharded is weight


def _mtp_layer_weights(model):
    """Return the weights ``MultiTokenPredictionLayer`` owns directly.

    Its MTP-specific modules (``enorm``/``hnorm``/``eh_proj``/``final_layernorm``) minus
    ``mtp_model_layer``, the ordinary transformer layer it borrows: that layer's weights
    are consumed once per window like every other decoder layer's.
    """
    from megatron.core.transformer.multi_token_prediction import MultiTokenPredictionLayer

    return tuple(
        param
        for submodule in model.modules()
        if isinstance(submodule, MultiTokenPredictionLayer)
        for name, child in submodule.named_children()
        if name != 'mtp_model_layer'
        for param in child.parameters()
    )


def _unit_grad_multiplicity(unit, extras) -> dict:
    """Per-parameter backward contribution counts for the combined 1F1B path.

    ``1`` per parameter plus every extra contribution of the weights it is
    (``_window_extras``). Contributions add rather than being alternatives, so a weight
    that is both the embedding and the projection's target picks up both. The extras
    cannot be derived from the FSDP unit alone -- they depend on this chunk's stage
    flags and the pipeline size -- which is why the caller passes them in.
    """
    return {
        fsdp_parameter.fqns: 1
        + sum(n for weight, n in extras if _matches_fsdp_parameter(fsdp_parameter, weight))
        for fsdp_parameter in unit._trainable_fsdp_parameters()
    }
