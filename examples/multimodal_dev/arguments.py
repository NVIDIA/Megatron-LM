# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Extra CLI arguments for multimodal_dev standalone training."""


def validate_multimodal_args(args):
    """Validate multimodal_dev-specific argument combinations."""
    if getattr(args, "packing_buffer_size", None) is not None and not getattr(
        args, "use_packed_sequence", False
    ):
        raise ValueError(
            "multimodal_dev --packing-buffer-size requires --use-packed-sequence, "
            "because Energon pre-packing returns logical segment lists consumed "
            "by the THD forward path."
        )
    if getattr(args, "use_packed_sequence", False) and int(args.micro_batch_size) != 1:
        raise ValueError(
            "multimodal_dev THD packing (--use-packed-sequence) currently requires "
            f"--micro-batch-size 1, got {args.micro_batch_size}. Increase sequence length "
            "or global batch size instead."
        )


def add_multimodal_args(parser):
    """Add multimodal-specific arguments to the Megatron argument parser."""
    group = parser.add_argument_group(
        "Multimodal", "Multimodal model arguments",
    )

    group.add_argument(
        "--model-arch",
        type=str,
        default="qwen35_vl",
        help="Model architecture. Available: qwen35_vl",
    )
    group.add_argument(
        "--model-variant",
        type=str,
        default="proxy",
        help="Model variant (size). E.g. proxy, 9b, 397b_a17b",
    )
    group.add_argument(
        "--dataset-provider",
        type=str,
        default="mock",
        help="Dataset provider: mock, cord_v2, energon",
    )
    group.add_argument(
        "--image-token-id",
        type=int,
        default=248056,
        help="Token ID for image placeholder tokens",
    )
    group.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Image size (height and width) for mock data",
    )
    group.add_argument(
        "--total-seq-length",
        type=int,
        default=1024,
        help="Total sequence length for mock data",
    )
    group.add_argument(
        "--image-seq-length",
        type=int,
        default=256,
        help="Number of image tokens in mock data",
    )
    group.add_argument(
        "--vision-num-layers",
        type=int,
        default=None,
        help=(
            "Override for vision backbone depth. "
            "Useful for proxy perf runs."
        ),
    )
    group.add_argument(
        "--hf-processor-path",
        type=str,
        default=None,
        help=(
            "HuggingFace processor path for real VLM datasets "
            "(e.g. Qwen/Qwen2.5-VL-7B-Instruct)"
        ),
    )
    group.add_argument(
        "--recompute-vision",
        action="store_true",
        default=False,
        help=(
            "Enable full activation recomputation for vision encoder layers. "
            "Uses uniform method and recomputes every layer. "
            "Independent of the decoder --recompute-* flags."
        ),
    )
    group.add_argument(
        "--use-packed-sequence",
        action="store_true",
        default=False,
        help=(
            "Pack variable-length sequences into THD format to eliminate "
            "padding waste."
        ),
    )
    group.add_argument(
        "--use-vanilla-collate-fn",
        action="store_true",
        default=False,
        help=(
            "Use vanilla collate function to collate the data."
        ),
    )
    group.add_argument(
        "--dataloader-save",
        type=str,
        default=None,
        help="Energon dataloader state save path.",
    )
    group.add_argument(
        "--packing-buffer-size",
        type=int,
        default=None,
        help=(
            "Enable Energon sample pre-packing with this buffer size. "
            "Final Megatron THD packing still uses --use-packed-sequence."
        ),
    )
    group.add_argument(
        "--packing-seq-length",
        type=int,
        default=0,
        help=(
            "Maximum token length for Energon sample pre-packing. "
            "Defaults to --total-seq-length/--seq-length when unset."
        ),
    )
    group.add_argument(
        "--packing-pad-to-multiple",
        type=int,
        default=1,
        help=(
            "Alignment budget used when Energon pre-packs samples. Final token "
            "padding is applied by the multimodal forward step."
        ),
    )
    group.add_argument(
        "--energon-virtual-epoch-length",
        type=int,
        default=1000,
        help="Virtual epoch length passed to get_train_dataset.",
    )
    group.add_argument(
        "--energon-max-samples-per-sequence",
        type=int,
        default=100,
        help="Maximum samples per sequence passed to get_train_dataset.",
    )
    group.add_argument(
        "--energon-shuffle-buffer-size",
        type=int,
        default=100,
        help="Shuffle buffer size passed to get_train_dataset.",
    )
    group.add_argument(
        "--energon-worker-debug-path",
        type=str,
        default=None,
        help="Optional Energon worker debug path.",
    )
    group.add_argument(
        "--energon-worker-log-level",
        type=int,
        default=0,
        help="Energon worker log level.",
    )
    group.add_argument(
        "--qwen-vl-min-pixels",
        type=int,
        default=256 * 28 * 28,
        help="Minimum image pixel budget passed to the Qwen VL processor.",
    )
    group.add_argument(
        "--qwen-vl-max-pixels",
        type=int,
        default=1280 * 28 * 28,
        help="Maximum image pixel budget passed to the Qwen VL processor.",
    )

    return parser
