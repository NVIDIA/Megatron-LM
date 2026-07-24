# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Extra CLI arguments for multimodal_dev standalone training."""


def add_multimodal_args(parser):
    """Add multimodal-specific arguments to the Megatron argument parser."""
    group = parser.add_argument_group("Multimodal", "Multimodal model arguments")

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
        help="Dataset provider: mock, mock_varlen, or cord_v2",
    )
    group.add_argument(
        "--image-token-id", type=int, default=248056, help="Token ID for image placeholder tokens"
    )
    group.add_argument(
        "--image-size", type=int, default=224, help="Image size (height and width) for mock data"
    )
    group.add_argument(
        "--mock-image-size-config-json",
        type=str,
        default=None,
        help=(
            "Processed-image resolution buckets for mock_varlen (required in "
            "packed_window mode). Accepts inline JSON or a JSON-file path with "
            'schema {"mode":"buckets","resolutions":[[224,224],[448,224]],'
            '"weights":[3,1]}. Each resolution must be divisible by '
            "patch_size*spatial_merge_size; weights are optional categorical "
            "bucket weights. The generator itself is configured through the "
            "core --varlen-mock-dataset-config-json flag with "
            '{"mode":"packed_window",...}; see the packed_window README '
            "section for the schema."
        ),
    )
    group.add_argument(
        "--total-seq-length", type=int, default=1024, help="Total sequence length for mock data"
    )
    group.add_argument(
        "--image-seq-length", type=int, default=256, help="Number of image tokens in mock data"
    )
    group.add_argument(
        "--vision-num-layers",
        type=int,
        default=None,
        help=("Override for vision backbone depth. " "Useful for proxy perf runs."),
    )
    group.add_argument(
        "--hf-processor-path",
        type=str,
        default=None,
        help=(
            "HuggingFace processor path for real VLM datasets " "(e.g. Qwen/Qwen2.5-VL-7B-Instruct)"
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
        help=("Pack variable-length sequences into THD format to eliminate " "padding waste."),
    )
    group.add_argument(
        "--pad-packed-seq-by-appending-dummy-seq",
        dest="pad_packed_seq_by_appending_dummy_seq",
        action="store_true",
        default=True,
        help=(
            "Positive compatibility alias for multimodal THD recipes. "
            "Dummy-tail representation is enabled by default; core also "
            "provides --no-pad-packed-seq-by-appending-dummy-seq to disable it."
        ),
    )
    group.add_argument(
        "--use-vanilla-collate-fn",
        action="store_true",
        default=False,
        help=("Use vanilla collate function to collate the data."),
    )

    return parser
