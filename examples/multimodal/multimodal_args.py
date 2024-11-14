# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
from megatron.core.models.multimodal.llava_model import IMAGE_TOKEN


def add_multimodal_extra_args(parser):
    """Extra arguments."""
    group = parser.add_argument_group(title='multimodal arguments')
    group.add_argument('--dataset-config', type=str, default=None)
    group.add_argument("--prompt-path", type=str, default=None)
    group.add_argument('--freeze-LM', action='store_true', default=False)
    group.add_argument('--freeze-ViT', action='store_true', default=False)
    group.add_argument('--language-model-type', type=str, required=True)
    group.add_argument('--vision-model-type', type=str, default="clip")
    group.add_argument("--disable-vision-class-token", action="store_true", default=False)
    group.add_argument(
        "--allow-missing-vision-projection-checkpoint", action="store_true", default=False
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
        "--dataloader-seq-length",
        type=int,
        help="Make dataloader to produce sequences of specific length.",
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
        choices=["mistral", "llama3", "chatml"],
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

    return parser
