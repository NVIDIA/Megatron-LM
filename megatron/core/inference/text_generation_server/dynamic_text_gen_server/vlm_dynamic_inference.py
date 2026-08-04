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
import sys
from functools import partial

# ``examples/multimodal/model.py`` and its siblings (``config.py``,
# ``layer_specs.py``) use bare imports like ``from config import ...``, so
# they must be importable as top-level modules. The script that calls into
# this module is expected to put the repo root on sys.path; we add the
# multimodal subdirectory here so callers don't have to.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# dynamic_text_gen_server -> text_generation_server -> inference -> core -> megatron -> ROOT
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, *(os.path.pardir,) * 5))
_EXAMPLES_MULTIMODAL = os.path.join(_REPO_ROOT, "examples", "multimodal")
if _EXAMPLES_MULTIMODAL not in sys.path:
    sys.path.append(_EXAMPLES_MULTIMODAL)

from megatron.core.transformer.module import MegatronModule
from megatron.inference.utils import add_inference_args
from megatron.training import get_args
from megatron.training import get_model as _get_model
from megatron.training import print_rank_0
from megatron.training.checkpointing import load_args_from_checkpoint, load_checkpoint


def add_vlm_inference_args(parser):
    """Add VLM-specific inference arguments on top of the standard inference args."""
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


def _detect_vlm_from_checkpoint(args, user_passed_attrs=None):
    """Peek at the checkpoint's saved training args to detect VLM vs GPT.

    Returns True if the checkpoint was trained as a VLM (has
    ``language_model_type``), False otherwise. As a side-effect, copies
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
            {
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
        )

    args._vlm_arg_resolution = resolution

    return True


def get_model(is_vlm: bool) -> MegatronModule:
    """Build and load the model; dispatches to the right model_provider."""
    args = get_args()

    if is_vlm:
        from model import model_provider  # examples/multimodal/model.py

        model = _get_model(partial(model_provider), wrap_with_ddp=False)
    else:
        from gpt_builders import gpt_builder  # examples/inference/gpt
        from model_provider import model_provider

        model = _get_model(partial(model_provider, gpt_builder), wrap_with_ddp=False)

    assert args.load is not None
    args.exit_on_missing_checkpoint = True
    load_checkpoint(
        ddp_model=model,
        optimizer=None,
        opt_param_scheduler=None,
        strict=not args.inference_ckpt_non_strict,
    )

    assert len(model) == 1, "Virtual PP not supported for VLM inference"
    model = model[0]
    model.eval()
    return model
