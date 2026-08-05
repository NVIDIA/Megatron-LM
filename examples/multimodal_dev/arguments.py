# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Extra CLI arguments for multimodal_dev standalone training."""


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
        help=(
            "Dataset provider: mock, mock_varlen, or cord_v2. Every provider "
            "emits the multimodal sample contract; a pure-text corpus is "
            "mock_varlen with a complete explicit profile whose every "
            'component has images_per_document {"counts":[0],"weights":[1]} '
            "(image_sizes still has to carry one valid resolution bucket, "
            "which nothing then draws from)."
        ),
    )
    group.add_argument(
        "--multimodal-varlen-mock-dataset-config-json",
        type=str,
        default=None,
        help=(
            "Config for --dataset-provider mock_varlen (inline JSON or a "
            "JSON-file path). OPTIONAL: omitted, the context-scaled default "
            "for the final --seq-length applies (domain [4096, 131072]). An "
            "explicit JSON must be a COMPLETE four-key profile (components / "
            "image_sizes / plan_pool_windows / plan_seed); partial or unknown-key "
            "configs are rejected, with the resolved default embedded in the "
            "error to copy and edit. "
            "Requires --pad-packed-seq-alignment max and "
            "--max-seqlen-per-dp-cp-rank. Distinct from the core "
            "--varlen-mock-dataset-config-json flag. Schema and provenance: "
            "the packed_document README section."
        ),
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
        help=(
            "Total sequence length for the fixed-shape providers (mock, "
            "cord_v2); ignored by mock_varlen (--seq-length is the sole "
            "capacity authority there)."
        ),
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
            "Enable full activation recomputation for vision encoder layers, "
            "as per-layer uniform blocks. Independent of the decoder "
            "--recompute-* flags. See --recompute-vision-whole-tower to trade "
            "the saved per-layer inputs for a larger backward spike."
        ),
    )
    group.add_argument(
        "--recompute-vision-whole-tower",
        action="store_true",
        default=False,
        help=(
            "With --recompute-vision, configure the tower as ONE uniform block "
            "spanning all layers: only the patch-embed output is saved, but "
            "backward re-materializes every layer's activations at once. "
            "Recompute FLOPs are unchanged. Wins when the per-layer saves "
            "(raw_patches x vision_hidden x num_layers) dominate vision memory, "
            "as in the 128K long-window qualification; off by default because a "
            "lighter payload can instead be dominated by the backward spike."
        ),
    )
    group.add_argument(
        "--use-packed-sequence",
        action="store_true",
        default=False,
        help=(
            "Pack variable-length sequences into THD format to reduce "
            "padding waste."
        ),
    )
    group.add_argument(
        "--max-vision-patches-per-microbatch",
        type=int,
        default=None,
        help=(
            "Fail fast when one microbatch's vision payload exceeds this many "
            "raw patches (verified on the TP source before the batch is staged or broadcast; the verdict reaches every rank of the TP group through the pack-status handshake). The vision tower's "
            "packed attention workspace scales stepwise with total raw patches, "
            "so exceeding the memory envelope otherwise surfaces as an opaque "
            "CUDA OOM. Unset by default."
        ),
    )
    group.add_argument(
        "--max-vision-patches-per-image",
        type=int,
        default=None,
        help=(
            "Fail fast when any single image exceeds this many raw patches "
            "(verified on the TP source before staging, then propagated to the "
            "TP group). For --dataset-provider mock_varlen the default is "
            "derived from the bucket table (the largest drawable bucket's "
            "raw-patch count, ignoring weight-0 buckets — an exact invariant "
            "of the data); unset otherwise."
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

    return parser
