# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
from megatron.core.models.multimodal.llava_model import IMAGE_TOKEN


def add_multimodal_extra_args(parser):
    """Extra arguments."""
    group = parser.add_argument_group(title='multimodal arguments')
    group.add_argument('--dataset-config', type=str, default=None)
    group.add_argument("--prompt-path", type=str, default=None)
    group.add_argument('--freeze-LM', action='store_true', default=False)
    group.add_argument('--freeze-ViT', action='store_true', default=False)
    group.add_argument('--freeze-sound-model', action='store_true', default=False)
    group.add_argument('--language-model-type', type=str, required=True)
    group.add_argument('--vision-model-type', type=str, default="clip")
    group.add_argument('--sound-model-type', type=str, default=None)
    group.add_argument("--disable-vision-class-token", action="store_true", default=False)
    group.add_argument(
        "--allow-missing-vision-projection-checkpoint", action="store_true", default=False
    )
    group.add_argument(
        "--allow-missing-sound-projection-checkpoint", action="store_true", default=False
    )
    group.add_argument(
        "--allow-missing-sound-model-checkpoint", action="store_true", default=False
    )
    group.add_argument("--use-te", action="store_true", default=False)
    group.add_argument(
        "--dataloader-save", type=str, default=None, help="Energon dataloader state save path"
    )
    group.add_argument(
        "--use-tiling", action="store_true", default=False, help="Use input image tiling"
    )
    group.add_argument("--max-num-tiles", type=int, default=1, help="Maximum number of image tiles")
    group.add_argument(
        "--use-thumbnail", action="store_true", default=False, help="Add image thumbnail as a tile"
    )
    group.add_argument(
        "--thumbnail-area-threshold", type=float, default=0.8,
        help="Maximum area percentage (0.0-1.0) of resized image relative to thumbnail area for which to add thumbnail. Default 0.8 (80%)"
    )
    group.add_argument(
        "--dataloader-seq-length",
        type=int,
        help="Make dataloader to produce sequences of specific length.",
    )
    group.add_argument(
        "--dataloader-seed",
        type=int,
        default=0,
        help="The seed for the dataloader to use for training.",
    )
    group.add_argument(
        "--lr-data-range-start",
        type=float,
        default=0,
        help="Start of the learning rate range as percentage (0-100) of the full training schedule. 0% means start from the beginning of the training schedule. E.g. setting to 10, means start at 10% of the training schedule (the dataloader still starts from the beginning of the dataset, but assume that corresponds to 10% of the training schedule)."
    )
    group.add_argument(
        "--lr-data-range-end",
        type=float,
        default=100,
        help="End of the learning rate range as percentage (0-100) of the full training schedule. 100% means the end of the training schedule. E.g. setting to 90, means end at 90% of the training schedule (the dataloader still ends at the end of the dataset, but assume that corresponds to 90% of the training schedule)."
    )
    group.add_argument(
        "--num-frames",
        type=int,
        default=1,
        help="Number of frames to regularly sample from the video as input to the model.",
    )
    group.add_argument(
        "--online-evaluation-config", type=str, help="Config file for online evaluation."
    )
    group.add_argument(
        "--special-tokens",
        nargs="*",
        default=[IMAGE_TOKEN],
        help="Special tokens used in the multimodal model",
    )
    group.add_argument(
        "--tokenizer-prompt-format",
        type=str,
        choices=["mistral", "llama3", "chatml", "nvlm-yi-34b", "qwen2p0", "qwen2p5", "llama3p1", "nemotron5",
                 "nemotron5-aligned", "llama_nemotron_8b", "nemotron-h-reasoning", "nemotron-h-5p5-reasoning",
                 "nemotron-h-5p5-reasoning-inference", "llama-nemotron-super", "llama-nemotron-super-1p5",
                 "nemotron6-moe"],
        required=True,
        help="Prompt format to use with the tokenizer.",
    )
    group.add_argument("--pixel-shuffle", action="store_true", default=False)
    group.add_argument(
        "--image-tag-type",
        type=str,
        choices=["nvlm", "internvl", ""],
        default="",  # Default: Image tag not used.
        help="Surround image tokens with tags.",
    )
    group.add_argument("--use-tile-tags", action="store_true", default=False, help="Use tile tags")
    group.add_argument("--class-token-len", type=int, default=None, help="Length of class token. If not set, uses model-specific defaults (radio: 8, radio-g: 5, cradio-g: 8). FP8 overrides to 16.")
    group.add_argument(
        "--packing-buffer-size",
        type=int,
        default=None,   # Packing is disabled by default.
        help="Enable sample packing by setting the buffer size to > 0",
    )
    group.add_argument(
        "--packing-seq-length", type=int, default=0, help="Packing sequence length. Must be > 0 if using packing."
    )
    group.add_argument(
        "--packing-knapsack-algorithm", type=str, default="greedy_knapsack", help="Knapsack algorithm to use for packing."
    )
    group.add_argument(
        "--recompute-vision", action="store_true", default=False, help="Enable activation checkpointing in the vision model"
    )
    group.add_argument(
        "--recompute-sound", action="store_true", default=False, help="Enable activation checkpointing in the sound model"
    )
    group.add_argument(
        "--use-loss-scaling", action="store_true", default=False, help="Scale loss based on conversation turn length (in tokens)."
    )
    group.add_argument(
        "--force-system-message", action="store_true", default=False, help="Force a specific system message"
    )
    group.add_argument("--eos-id", type=int, help="termination id for MultiModal Tokenizer")
    group.add_argument(
        "--use-area-weighted-aspect-ratio", action="store_true", default=False,
        help=(
            "When --use-tiling is True, find the aspect ratio to use based on the original ",
            "image aspect ratio and the area covered by the tiles.")
    )
    group.add_argument("--use-mcore-inference", action="store_true", default=False, help="Use the MCore inference API")
    group.add_argument("--use-vision-backbone-fp8-arch", action="store_true", default=False, help="Use the FP8 arch in the vision backbone. This is used to load the FP8 checkpoint when running inference.")
    group.add_argument(
        "--dynamic-resolution", action="store_true", default=False, help="Use input image dynamic resolution"
    )
    group.add_argument(
        "--dynamic-resolution-min-patches", type=int, default=0, help="Minimum number of patches per image for dynamic resolution"
    )
    group.add_argument(
        "--dynamic-resolution-max-patches", type=int, default=0, help="Maximum number of patches per image for dynamic resolution"
    )
    group.add_argument(
        "--dynamic-resolution-min-side", type=int, default=None, help="Minimum side length for dynamic resolution"
    )
    group.add_argument(
        "--match-tiling-dynamic-resolution", action="store_true", default=False,
        help="Use match-tiling dynamic resolution strategy that combines tiling logic with dynamic resolution processing"
    )
    group.add_argument(
        "--masked-tiling-dynamic-resolution", action="store_true", default=False,
        help="Use masked-tiling dynamic resolution strategy that isolates tiles as separate packed samples"
    )
    group.add_argument(
        "--image-break-token", type=str, default=None, help="Token to use for image break tokens, must be added to --special-tokens as well"
    )
    group.add_argument("--conv-merging", action="store_true", default=False, help="Use convolution merging which uses a convolution to merge tokens after the vision encoder")
    group.add_argument(
        "--allow-missing-conv-merge-checkpoint", action="store_true", default=False
    )
    group.add_argument(
        "--video-min-num-frames", type=int, default=8, help="Minimum number of frames to sample from the video as input to the model.",
    )
    group.add_argument(
        "--video-max-num-frames", type=int, default=32, help="Maximum number of frames to sample from the video as input to the model.",
    )
    group.add_argument(
        "--video-default-fps", type=int, default=2, help="Default frames per second to sample from the video as input to the model.",
    )
    group.add_argument(
        "--video-frame-temporal-jitter", action="store_true", default=False, help="Enable temporal jittering of the frames to sample from the video as input to the model.",
    )
    group.add_argument(
        "--video-target-img-size", type=int, default=None,
        help="Target image size (pixels) for video frames with dynamic resolution. "
             "Default None, must specify this or video_target_num_patches."
    )
    group.add_argument(
        "--video-target-num-patches", type=int, default=None,
        help=(
            "Target number of patches for video frames. Default None, must specify this or video_target_img_size."
        )
    )
    group.add_argument(
        "--video-maintain-aspect-ratio", action="store_true", default=False,
        help="Match video native aspect ratio while respecting target patch budget."
    )
    # Temporal compression arguments
    group.add_argument(
        "--video-temporal-patch-size", type=int, default=1,
        help="Temporal patch size for video frames. Default 1 (no temporal compression). "
             "Set to 2 to group pairs of frames into 3D tubelets for temporal compression."
    )
    group.add_argument(
        "--allow-checkpoint-without-temporal-compression", action="store_true", default=False,
        help="Allow loading a checkpoint without temporal compression into a model with temporal compression. "
             "When set, the embedder weights will be duplicated along the temporal dimension if needed."
    )
    group.add_argument(
        "--separate-video-embedder", action="store_true", default=False,
        help="Use separate embedders for images and videos. When set, the image embedder (self.embedder) "
             "expects C*P*P input, and a separate video embedder (self.video_embedder) expects C*T*P*P input. "
             "This avoids duplicating image patches along the temporal dimension. "
             "Only relevant when --video-temporal-patch-size > 1."
    )
    group.add_argument(
        "--video-prompt-version", type=int, default=2,
        help="Video prompt format version."
             "1 = each frame on its own line, <image> at tubelet boundaries. "
             "2 = group T frames with 'and', one <image> per group (generalization of 1 to support temporal compression)"
    )
    group.add_argument(
        "--enable-fusions", action="store_true", default=True, help="Enable fusions in the model."
    )
    group.add_argument(
        "--optimize-broadcast", action="store_true", default=True, help="Optimize the broadcast of data.",
    )
    group.add_argument(
        "--recompute-vision-num-layers", type=int, default=0, help="Number of layers to recompute in the vision model."
    )
    group.add_argument(
        "--recompute-granularity-vision", type=str, default=None, help="Granularity to recompute in the vision model.",
        choices=["full", "selective"],
    )
    group.add_argument(
        "--recompute-method-vision", type=str, default=None,
        choices=['uniform', 'block'], help="Method to recompute in the vision model.",
    )
    group.add_argument(
        "--recompute-vision-projection", action="store_true", default=False, help="Enable activation checkpointing in the vision projection layer."
    )
    group.add_argument(
        "--recompute-sound-projection", action="store_true", default=False, help="Enable activation checkpointing in the sound projection layer."
    )
    group.add_argument(
        "--allow-large-videos", action="store_true", default=False, help="Allow large videos to be loaded into the model."
    )
    group.add_argument(
        "--efficient-video-sampling-variant", type=str, default=None, help="The EVS variant. Read docstring on EVSHelper"
    )
    group.add_argument(
        "--sound-target-rate",
        type=int,
        default=16000,
        help="Target rate of sound clips to regularly sample from the audio as input to the model.",
    )
    group.add_argument(
        "--sound-embedding-size",
        type=int,
        default=750,
        help="Size of the sound embedding.",
    )
    group.add_argument(
        "--sound-clip-duration",
        type=int,
        default=30,
        help="Sound model clip duration in seconds."
    )
    group.add_argument(
        "--sound-min-duration",
        type=float,
        default=0.1,
        help="We will pad the audio clip to at least this duration (in seconds), even when sound-pad-to-clip-duration is False."
    )
    group.add_argument(
        "--sound-pad-to-clip-duration",
        action="store_true",
        default=False,
        help="Pad every audio clip to the clip duration (introduces potentially many padding tokens in the LLM input)."
    )
    group.add_argument(
        "--sound-batch-split",
        type=int,
        default=1,
        help="Splits the sound batch into this many chunks to avoid OOMs. Not necessary when using bucketing; use this only when --sound-pad-to-clip-duration is not specified and bucketing is not enabled."
    )
    group.add_argument(
        "--use-new-dataloader-path", action="store_true", default=False, help="Use the new dataloader path."
    )
    group.add_argument(
        "--decoder-tp-comm-overlap", action="store_true", default=False, help="Enable tensor parallel communication overlap in the decoder."
    )
    group.add_argument(
        "--freeze-vision-projection", action="store_true", default=False, help="Freeze the vision projection module."
    )
    group.add_argument(
        "--freeze-sound-projection", action="store_true", default=False, help="Freeze the sound projection module."
    )
    group.add_argument(
        "--relax-sender-check", action="store_true", default=False, help="Relax the sender check in the dataloader to allow other role than user and assistant."
    )
    group.add_argument(
        "--relax-thinking-trace-check", action="store_true", default=False, help="Relax the checks in the dataloader which ensure the thinking trace is well formatted."
    )
    group.add_argument(
        "--allow-cross-sample-attention", action="store_true", default=False, help="Allow cross sample attention when using sample packing."
    )
    group.add_argument(
        "--only-keep-samples-with-img", action="store_true", default=False, help="Discard samples that do not have an image."
    )
    group.add_argument(
        "--unfreeze-router", action="store_true", default=False, help="Unfreeze MoE router weights."
    )
    group.add_argument(
        "--log-moe-routing-diagnostics", action="store_true", default=False,
        help="Log MoE routing diagnostics (expert utilization, dead experts, bias stats) "
             "to tensorboard. Logged at tensorboard_log_interval."
    )
    group.add_argument(
        "--apply-data-augment", action="store_true", default=False, help="Apply data augmentation to the image."
    )
    group.add_argument(
        "--radio-force-eval-mode",
        action="store_true",
        default=False,
        help="Force RADIO to stay in eval mode (eval-mode CPE, no dropout). Recommended for pre-training."
    )
    group.add_argument(
        "--radio-force-cpe-eval-mode",
        action="store_true",
        default=False,
        help="Force RADIO to use CPE (cropped position embeddings) in eval mode. Recommended for SFT."
    )
    group.add_argument(
        "--radio-interpolate-only-cpe",
        action="store_true",
        default=False,
        help="Interpolate the position embeddings to input size, without any cropping."
    )
    group.add_argument(
        "--radio-cpe-aspect-ratio-select",
        action="store_true",
        default=False,
        help="Select position embeddings based on aspect ratio so long edge always mapped to 1."
    )
    group.add_argument(
        "--radio-disable-cpe",
        action="store_true",
        default=False,
        help="Disable cropped position embeddings in the radio model."
    )
    group.add_argument(
        "--no-calculate-per-token-loss", action="store_true", default=False,
        help="Disable calculating per-token loss."
    )
    group.add_argument(
        "--no-load-balancing-sequence-scaling", action="store_true", default=False,
        help="Disable scaling the load balancing gradient by the number of tokens."
    )
    group.add_argument(
        "--tokenizer-keep-history-thinking", action="store_true", default=False,
        help="Keep the history thinking in the tokenizer."
    )
    group.add_argument(
        "--log-model-grad-norms", action="store_true", default=False, help="Log the gradient norms of the model components."
    )
    group.add_argument(
        "--log-model-act-norms", action="store_true", default=False, help="Log the activation norms of the model components."
    )

    return parser
