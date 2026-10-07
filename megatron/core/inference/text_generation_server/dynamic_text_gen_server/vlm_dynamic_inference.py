# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Helpers for VLM dynamic-batching inference.

Exposes the small surface that the dynamic text generation server needs to
support multimodal checkpoints:

  * :func:`add_vlm_inference_args` — argparse group for VLM-specific args
  * :func:`_detect_vlm_from_checkpoint` — peek at saved training args and
    decide GPT-vs-VLM, with CLI > checkpoint > parser-default precedence
  * :func:`_print_resolved_args` — diagnostic dump of the args namespace
    *after* the late checkpoint resolution above
  * :func:`get_model` — build and load either a GPT or LLaVA model

The image-preprocessing helpers live in :mod:`.image_preprocessing` and are
re-exported here for backwards compatibility with older standalone callers.
"""

import json
import os
import re
from functools import partial

import torch

from megatron.core import dist_checkpointing
from megatron.core.models.vision.encoder_registry import get_spec
from megatron.core.transformer.module import MegatronModule
from megatron.core.utils import unwrap_model
from megatron.inference.checkpointing import load_checkpoint_for_inference
from megatron.training import get_args
from megatron.training import get_model as _get_model
from megatron.training import print_rank_0
from megatron.training.checkpointing import (
    get_checkpoint_name,
    get_checkpoint_tracker_filename,
    get_loaded_iteration,
    load_args_from_checkpoint,
    read_metadata,
)

# NOTE: ``get_model`` below does a ``from model import model_provider`` for the
# ``examples/multimodal/model.py`` file, whose siblings use bare imports like
# ``from config import ...``. The *caller* of this module (typically
# ``tools/run_dynamic_text_generation_server.py``) is expected to have already
# added the repo root and ``examples/multimodal/`` to ``sys.path`` before
# invoking ``get_model``. ``megatron/core`` does not mutate ``sys.path`` here.


def add_vlm_inference_args(parser):
    """Add VLM-specific inference arguments on top of the standard inference args."""
    from megatron.inference.utils import add_inference_args

    parser = add_inference_args(parser)
    group = parser.add_argument_group(title="VLM dynamic inference")
    group.add_argument(
        "--input-image-path",
        type=str,
        default=None,
        help="Path to input image(s). Can be a single image or directory.",
    )
    group.add_argument(
        "--input-prompts-json",
        type=str,
        default=None,
        help="Path to JSON file with prompts and image paths. "
        'Format: [{"prompt": "...", "image": "path/to/image.jpg"}, ...]',
    )
    return parser


_MISSING = object()


def _jsonable_arg_value(value):
    if value is _MISSING:
        return "<MISSING>"
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable_arg_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable_arg_value(val) for key, val in value.items()}
    return repr(value)


def _arg_value_changed(before, after):
    return _jsonable_arg_value(before) != _jsonable_arg_value(after)


def _resolution_record(attr, source, parser_value, checkpoint_value, resolved_value, note):
    """Return the provenance record of one resolved arg, as _print_resolved_args reports it."""
    return {
        "attr": attr,
        "source": source,
        "parser_value": parser_value,
        "checkpoint_value": checkpoint_value,
        "resolved_value": resolved_value,
        "parser_changed_by_resolution": _arg_value_changed(parser_value, resolved_value),
        "checkpoint_overridden": (
            source == "cli"
            and checkpoint_value is not _MISSING
            and checkpoint_value is not None
            and _arg_value_changed(checkpoint_value, resolved_value)
        ),
        "note": note,
    }


def _print_resolved_args(title, args):
    """Print args after late checkpoint resolution.

    Megatron's standard args table is emitted during parse/initialize, before
    this server copies VLM-only fields out of the checkpoint. Print a second
    table at the point where the values are the ones model construction will
    actually consume, followed by a per-attr provenance dump showing where
    each VLM-relevant value came from (CLI, checkpoint, parser default).
    """
    print_rank_0(f'------------------------ {title} ------------------------')
    str_list = []
    for arg in vars(args):
        if arg.startswith("_"):
            continue
        dots = '.' * (48 - len(arg))
        str_list.append('  {} {} {}'.format(arg, dots, getattr(args, arg)))
    for arg in sorted(str_list, key=lambda x: x.lower()):
        print_rank_0(arg)
    print_rank_0(f'-------------------- end of {title} ---------------------')

    resolution = getattr(args, "_vlm_arg_resolution", None)
    if not resolution:
        return

    print_rank_0("---------------- VLM argument provenance ----------------")
    for record in resolution:
        attr = record["attr"]
        final_value = getattr(args, attr, _MISSING)
        changed_after_resolution = _arg_value_changed(record["resolved_value"], final_value)
        payload = {
            "arg": attr,
            "source": record["source"],
            "parser": _jsonable_arg_value(record["parser_value"]),
            "checkpoint": _jsonable_arg_value(record["checkpoint_value"]),
            "resolved": _jsonable_arg_value(record["resolved_value"]),
            "final": _jsonable_arg_value(final_value),
            "parser_changed_by_resolution": record["parser_changed_by_resolution"],
            "checkpoint_overridden": record["checkpoint_overridden"],
            "changed_after_resolution": changed_after_resolution,
            "note": record["note"],
        }
        print_rank_0(f"[vlm_arg_provenance] {json.dumps(payload, sort_keys=True)}")
    print_rank_0("------------ end of VLM argument provenance -------------")


_MIMO_LANGUAGE_MODEL_PREFIX = 'language_model.module.module.'
# Captures (modality submodule prefix, encoder name) from a MIMO vision encoder key.
_MIMO_ENCODER_KEY = re.compile(r'(modality_submodules\.[^.]+\.(?:module\.)*)encoders\.([^.]+)\.')
# Training-only buffers that checkpoints may carry but inference models never build.
_TRAINING_ONLY_CHECKPOINT_SUFFIXES = ('router.qb_bin_bounds',)

# MIMO checkpoints don't record the LLaVA image geometry; it comes from the encoder registry.
_ENCODER_REGISTRY_ATTRS = {
    'patch_dim': 'patch_dim',
    'img_h': 'default_img_h',
    'img_w': 'default_img_w',
    'pixel_shuffle': 'pixel_shuffle',
    'conv_merging': 'conv_merging',
    'dynamic_resolution': 'dynamic_resolution',
    'dynamic_resolution_max_patches': 'dynamic_resolution_max_patches',
}


def _checkpoint_tensor_keys(args):
    """Return the tensor keys of the checkpoint iteration that --load resolves to."""
    if args.ckpt_step is not None:
        release = False
    else:
        _, release = read_metadata(get_checkpoint_tracker_filename(args.load))
    checkpoint_dir = get_checkpoint_name(
        args.load, get_loaded_iteration(), release, return_base_dir=True
    )
    return dist_checkpointing.load_tensors_metadata(checkpoint_dir).keys()


def _mimo_checkpoint_prefix_map(args):
    """Map LLaVAModel key prefixes to the MIMO checkpoint prefixes they load from.

    Returns None for a checkpoint without a vision encoder.
    """
    encoders = set()
    for key in _checkpoint_tensor_keys(args):
        match = _MIMO_ENCODER_KEY.match(key)
        if match is not None:
            encoders.add(match.groups())
    if not encoders:
        return None
    if len(encoders) > 1:
        raise ValueError(f"Expected one vision encoder in the MIMO checkpoint, found {encoders}")
    ((modality_prefix, encoder_name),) = encoders
    return {
        'language_model.': _MIMO_LANGUAGE_MODEL_PREFIX,
        'vision_model.': f'{modality_prefix}encoders.{encoder_name}.',
        'vision_projection.': f'{modality_prefix}input_projections.0.',
    }


def _enable_checkpoint_expert_bias(args):
    """Route with the trained router expert biases when the checkpoint carries them.

    Routing adds each MoE router's expert_bias to the scores when selecting experts. A checkpoint
    can hold trained biases while its saved args disable them (e.g. quantile balancing that
    estimates expert_bias from global-batch histograms), and --use-checkpoint-args would then
    build routers without them.
    """
    if any(key.endswith('router.expert_bias') for key in _checkpoint_tensor_keys(args)):
        if not args.moe_router_enable_expert_bias:
            print_rank_0(
                'Enabling moe_router_enable_expert_bias: the checkpoint has trained router '
                'expert biases'
            )
        args.moe_router_enable_expert_bias = True


# Image token roles a tokenizer_config.json can declare, by the arg they fill.
_TOKENIZER_IMAGE_TOKENS = {
    'image_token_id': 'image_token',
    'image_break_token_id': 'image_break_token',
    'image_end_token_id': 'image_end_token',
}


def _tokenizer_image_token_ids(args):
    """Return the image token IDs declared by the tokenizer's tokenizer_config.json, by arg name.

    The declarations may be nested (e.g. under a processor section); each declared token is mapped
    to its ID through added_tokens_decoder.
    """
    tokenizer_dir = getattr(args, 'tokenizer_model', None)
    config_path = os.path.join(tokenizer_dir, 'tokenizer_config.json') if tokenizer_dir else None
    if config_path is None or not os.path.isfile(config_path):
        return {}
    with open(config_path) as f:
        config = json.load(f)
    token_ids = {
        token['content']: int(token_id)
        for token_id, token in config.get('added_tokens_decoder', {}).items()
    }
    declared = {}

    def find(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key in _TOKENIZER_IMAGE_TOKENS.values() and isinstance(value, str):
                    declared.setdefault(key, value)
                find(value)
        elif isinstance(node, list):
            for value in node:
                find(value)

    find(config)
    return {
        arg: token_ids[declared[role]]
        for arg, role in _TOKENIZER_IMAGE_TOKENS.items()
        if declared.get(role) in token_ids
    }


def _resolve_mimo_vision_args(args, checkpoint_args, user_passed_attrs):
    """Fill the LLaVA vision args of a MIMO checkpoint, recording where each value came from."""
    if 'vision_model_type' not in user_passed_attrs:
        raise ValueError("MIMO checkpoint inference requires --vision-model-type")
    spec = get_spec(args.vision_model_type)
    sources = {
        attr: ('encoder registry', getattr(spec, spec_attr))
        for attr, spec_attr in _ENCODER_REGISTRY_ATTRS.items()
    }
    if not spec.dynamic_resolution_max_patches:  # 0 leaves the patch budget to the arguments.
        del sources['dynamic_resolution_max_patches']
    # Image tokens: the tokenizer's declared tokens, else the token the checkpoint recorded.
    checkpoint_image_token_id = getattr(checkpoint_args, 'image_token_id', None)
    sources['image_token_id'] = ('checkpoint', checkpoint_image_token_id)
    tokenizer_ids = _tokenizer_image_token_ids(args)
    for attr, token_id in tokenizer_ids.items():
        sources[attr] = ('tokenizer', token_id)
    tokenizer_image_token_id = tokenizer_ids.get('image_token_id')
    if None not in (tokenizer_image_token_id, checkpoint_image_token_id) and (
        tokenizer_image_token_id != checkpoint_image_token_id
    ):
        print_rank_0(
            f"WARNING: the tokenizer declares image token {tokenizer_image_token_id} but the "
            f"checkpoint recorded {checkpoint_image_token_id}; using the tokenizer's."
        )

    resolution = []
    for attr, (source, value) in sources.items():
        parser_value = getattr(args, attr, _MISSING)
        checkpoint_value = getattr(checkpoint_args, attr, _MISSING)
        if attr in user_passed_attrs:
            source, note = 'cli', 'explicit CLI value preserved'
        else:
            setattr(args, attr, value)
            note = f'copied from {source}'
        resolved_value = getattr(args, attr)
        resolution.append(
            _resolution_record(attr, source, parser_value, checkpoint_value, resolved_value, note)
        )
    args._vlm_arg_resolution = resolution


def _check_mimo_checkpoint_fully_loaded(args, model):
    """Fail if the checkpoint holds language or vision tensors that the model did not load.

    A MIMO checkpoint also stores modules this model never builds, so the load itself cannot
    reject unused checkpoint tensors; a module the model omits (e.g. from a config default that
    differs from training) would otherwise be dropped silently. Compares against the keys the
    load requested, which the model records while building its sharded state dict with the
    checkpoint's metadata.
    """
    requested = unwrap_model(model).sharded_checkpoint_keys
    assert requested is not None, "the model has not built a sharded state dict for the load"
    gathered = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(gathered, sorted(requested))

    unloaded = []
    if torch.distributed.get_rank() == 0:
        requested = set().union(*gathered)
        checkpoint_prefixes = tuple(args.mimo_checkpoint_prefix_map.values())
        unloaded = sorted(
            key
            for key in _checkpoint_tensor_keys(args)
            if key.startswith(checkpoint_prefixes)
            and not key.endswith(_TRAINING_ONLY_CHECKPOINT_SUFFIXES)
            # Factories (e.g. fused in_proj) expand into sub-keys of the requested key.
            and key not in requested
            and not any(key.startswith(f"{requested_key}.") for requested_key in requested)
        )
    # Every rank must fail together, not just the one that inspected the checkpoint.
    shared = [unloaded]
    torch.distributed.broadcast_object_list(shared, src=0)
    unloaded = shared[0]
    if unloaded:
        raise RuntimeError(
            f"{len(unloaded)} checkpoint model tensors were not loaded; the model config does "
            f"not match the checkpoint: {unloaded[:20]}"
        )


def _detect_vlm_from_checkpoint(args, user_passed_attrs=None):
    """Peek at the checkpoint's saved training args to detect VLM vs GPT.

    Returns True if the checkpoint was trained as a VLM (has
    ``language_model_type``, or is a MIMO checkpoint), False otherwise. As a side-effect, copies
    VLM-specific args from the checkpoint into the current args namespace
    so the multimodal model_provider can access them, and records resolution
    provenance on ``args._vlm_arg_resolution`` for the diagnostic dump.

    Precedence for each attr is CLI > checkpoint > parser default. Callers
    pass ``user_passed_attrs`` to indicate which attribute names the user
    actually typed on the command line; those values are left alone.
    """
    user_passed_attrs = user_passed_attrs or set()
    result = load_args_from_checkpoint(args)
    if not isinstance(result, tuple):
        return False

    _, checkpoint_args = result
    # MIMO training records its module-grid layout (--mimo-llm-*), and its checkpoints nest each
    # module under its own prefix; load a vision checkpoint into a LLaVAModel that uses those
    # key names.
    if hasattr(checkpoint_args, 'mimo_llm_tp'):
        _enable_checkpoint_expert_bias(args)
        prefix_map = _mimo_checkpoint_prefix_map(args)
        if prefix_map is not None:
            args.mimo_checkpoint_prefix_map = prefix_map
            _resolve_mimo_vision_args(args, checkpoint_args, user_passed_attrs)
            return True
        # Without a vision encoder, run text inference on the language model alone as a
        # HybridModel, loading only its keys.
        if 'model_provider' not in user_passed_attrs:
            args.model_provider = 'hybrid'
        args.checkpoint_model_prefix = _MIMO_LANGUAGE_MODEL_PREFIX
        return False
    if not hasattr(checkpoint_args, 'language_model_type'):
        return False
    if checkpoint_args.language_model_type is None:
        return False

    vlm_attrs = [
        'language_model_type',
        'vision_model_type',
        'vision_projection_type',
        'decoder_seq_length',
        'use_te',
        'disable_vision_class_token',
        'pixel_shuffle',
        'use_tile_tags',
        'max_num_tiles',
        'use_thumbnail',
        'use_tiling',
        'tokenizer_prompt_format',
        'recompute_vision',
        'num_frames',
        'freeze_LM',
        'freeze_ViT',
        'allow_missing_vision_projection_checkpoint',
        'pixel_mean',
        'pixel_std',
        'use_area_weighted_aspect_ratio',
        'dynamic_resolution',
        'dynamic_resolution_min_patches',
        'dynamic_resolution_max_patches',
        'class_token_len',
        'radio_force_cpe_eval_mode',
        'radio_force_eval_mode',
        'radio_interpolate_only_cpe',
        'radio_cpe_aspect_ratio_select',
        'radio_disable_cpe',
        'spec',
        'transformer_impl',
        'is_hybrid_model',
        'hybrid_override_pattern',
        'num_experts',
    ]
    resolution = []
    for attr in vlm_attrs:
        parser_value = getattr(args, attr, _MISSING)
        checkpoint_value = getattr(checkpoint_args, attr, _MISSING)
        if attr in user_passed_attrs:
            source = "cli"
            resolved_value = getattr(args, attr, _MISSING)
            note = "explicit CLI value preserved"
        elif checkpoint_value is not _MISSING and checkpoint_value is not None:
            source = "checkpoint"
            setattr(args, attr, checkpoint_value)
            resolved_value = checkpoint_value
            note = "copied from checkpoint"
        elif checkpoint_value is None:
            source = "default"
            resolved_value = getattr(args, attr, _MISSING)
            note = "checkpoint value is None; kept parser/default value"
        else:
            source = "default"
            resolved_value = getattr(args, attr, _MISSING)
            note = "not present in checkpoint; kept parser/default value"

        resolution.append(
            _resolution_record(attr, source, parser_value, checkpoint_value, resolved_value, note)
        )

    args._vlm_arg_resolution = resolution

    return True


def get_model(is_vlm: bool) -> MegatronModule:
    """Build and load the model; dispatches to the right model_provider."""
    args = get_args()

    is_mimo = getattr(args, 'mimo_checkpoint_prefix_map', None) is not None
    if is_vlm and is_mimo:
        from mimo_checkpoint_model import model_provider  # examples/multimodal

        model = _get_model(model_provider, wrap_with_ddp=False)
    elif is_vlm:
        from model import model_provider  # examples/multimodal/model.py

        model = _get_model(partial(model_provider), wrap_with_ddp=False)
    else:
        from gpt_builders import gpt_builder  # examples/inference/gpt
        from model_provider import model_provider

        model = _get_model(partial(model_provider, gpt_builder), wrap_with_ddp=False)

    assert args.load is not None
    args.exit_on_missing_checkpoint = True
    load_checkpoint_for_inference(model, strict=not args.inference_ckpt_non_strict)
    if is_mimo:
        _check_mimo_checkpoint_fully_loaded(args, model[0])

    assert len(model) == 1, "Virtual PP not supported for VLM inference"
    model = model[0]
    model.eval()
    return model
