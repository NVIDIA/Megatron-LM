# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Resolve the MIMO encoder's geometry without constructing its model."""

from dataclasses import replace

from megatron.core.models.vision.encoder_registry import get_spec

VISION_MODEL_TYPES = ("radio", "pixtral-vit", "pixtral-vit-large", "pixtral-vit-1p6b")


def add_vision_args(parser):
    """Register tower selection and optional native-encoder initialization."""
    group = parser.add_argument_group("MIMO vision encoder")
    group.add_argument(
        "--vision-model-type",
        choices=VISION_MODEL_TYPES,
        default="radio",
        help="Vision tower preset (default: RADIO). Native Pixtral requires dynamic "
        "resolution, dropped class tokens and the Energon prepared-data application.",
    )
    group.add_argument(
        "--load-vision-from",
        metavar="DIRECTORY",
        help="Optional native Pixtral initialization from a PyTorch distributed-checkpoint "
        "architecture export. Requires every parameter under model.vision_model., "
        "BF16 tensors and encoder TP1. Loaded before DDP/optimizer construction. "
        "Full model --load takes precedence only with --exit-on-missing-checkpoint. "
        "--pretrained-checkpoint also skips this export.",
    )
    return parser


def resolve_encoder_spec(args):
    """Return the authoritative preset with explicit image dimensions applied.

    RADIO retains its existing MIMO defaults, including patch16 and 224-pixel
    construction dimensions. Native Pixtral derives omitted geometry from its
    preset. An explicit patch size must agree with that native architecture.
    This function performs no model, device, checkpoint or data initialization.
    """
    name = getattr(args, "vision_model_type", "radio")
    if name not in VISION_MODEL_TYPES:
        raise ValueError(f"unsupported MIMO vision model: {name}")
    spec = get_spec(name)
    patch = getattr(args, "patch_dim", None)
    height = getattr(args, "img_h", None)
    width = getattr(args, "img_w", None)
    if name == "radio":
        return replace(
            spec,
            patch_dim=16 if patch is None else patch,
            default_img_h=224 if height is None else height,
            default_img_w=224 if width is None else width,
            class_token_len=getattr(args, "class_token_len", 8),
            pixel_shuffle=bool(getattr(args, "pixel_shuffle", False)),
        )
    if patch is not None and patch != spec.patch_dim:
        raise ValueError(f"{name} requires patch_dim={spec.patch_dim}, got {patch}")
    if getattr(args, "pixel_shuffle", False):
        raise ValueError("native Pixtral uses its preset merger, not RADIO pixel shuffle")
    height = spec.default_img_h if height is None else height
    width = spec.default_img_w if width is None else width
    stride = spec.patch_dim * spec.spatial_reduction
    if any(type(value) is not int or value <= 0 or value % stride for value in (height, width)):
        raise ValueError(f"native Pixtral image bounds must be positive multiples of {stride}")
    return replace(spec, default_img_h=height, default_img_w=width)


def configure_vision_args(args):
    """Resolve omitted MIMO geometry before stock argument validation."""
    spec = resolve_encoder_spec(args)
    if spec.name != "radio":
        vision_initialization_checkpoint(args)
        if not args.dynamic_resolution or not args.disable_vision_class_token:
            raise ValueError("native Pixtral requires dynamic resolution and dropped class tokens")
        if getattr(args, "dataset_provider", "mock") != "energon":
            raise ValueError(
                "native Pixtral currently requires the Energon prepared-data application"
            )
    elif getattr(args, "load_vision_from", None):
        raise ValueError("--load-vision-from currently supports native Pixtral, not RADIO")
    args.patch_dim = spec.patch_dim
    args.img_h = spec.default_img_h
    args.img_w = spec.default_img_w
    return spec


def vision_initialization_checkpoint(args):
    """Select warm-start or strict full resume without probing checkpoint storage."""
    checkpoint = getattr(args, "load_vision_from", None)
    if checkpoint and getattr(args, "load", None):
        if not getattr(args, "exit_on_missing_checkpoint", False):
            raise ValueError(
                "combining --load with --load-vision-from requires "
                "--exit-on-missing-checkpoint to prevent a frozen random encoder"
            )
        return None
    if getattr(args, "pretrained_checkpoint", None):
        return None
    return checkpoint
