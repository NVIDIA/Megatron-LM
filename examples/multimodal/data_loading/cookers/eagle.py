# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
import re

from data_loading.conversation_sample import ConversationSample, Message
from PIL import Image

from megatron.core.models.multimodal.llava_model import IMAGE_TOKEN
from megatron.energon import CachePool, FileStore, MockLazy, basic_sample_keys, cooker, stateless
from megatron.training import get_args


@stateless
@cooker(need_cache=True)
def cook_eagle(sample: dict, cache: CachePool, media_source: FileStore) -> ConversationSample:
    args = get_args()
    data = sample["json"]
    raw_imgs = data.get("images", [])
    image_sizes = data.get("image_sizes", [])
    assert all(isinstance(img, str) for img in raw_imgs) or (
        all(
            not isinstance(img, str) and isinstance(img, (tuple, list)) and len(img) == 2
            for img in raw_imgs
        )
    ), f"Expected all images to be either paths or tuples of (path, int): {raw_imgs}"
    images = []
    for img in raw_imgs:
        if isinstance(img, str):
            images.append(cache.get_lazy(media_source, img))
        else:
            images.append((cache.get_lazy(media_source, img[0]), img[1]))

    # If this is a text-only sample and we are freezing the LM,
    # then use a dummy input image.
    if len(images) == 0 and args.freeze_LM:
        image_sizes.append((args.img_w, args.img_h))
        images.append(
            MockLazy(
                "empty_img", lambda x: Image.new("RGB", (args.img_w, args.img_h), (255, 255, 255))
            )
        )

    # Note: Some tokenizers may ignore the system prompt.
    conversation = [Message(sender="system", fragments=["Answer the questions."])]

    # Format the conversation as a list of "user" / "assistant" turns.
    for turn in data["conversations"]:
        assert turn["from"] in [
            "human",
            "gpt",
        ], f"Unexpected role {turn['from']} in {data['conversations']}"
        # TODO: Split the text into fragments.
        conversation.append(
            Message(
                sender="user" if turn["from"] == "human" else "assistant", fragments=[turn["value"]]
            )
        )

    # Replace the image tags <image-idx> with IMAGE_TOKEN and count the number of image tags
    number_image_tags = 0
    image_tag_ids_list = []
    for turn in conversation:
        if turn["role"] == "user":
            image_tag_ids = [int(x) - 1 for x in re.findall(r"<image-(\d+)>", turn["content"])]
            image_tag_ids_list.extend(image_tag_ids)
            turn["content"] = re.sub(r"<image-\d+>", IMAGE_TOKEN, turn["content"])
            number_image_tags += turn["content"].count(IMAGE_TOKEN)

    # We re-order the images in sample.images according to how they appear in the conversation.
    if len(image_tag_ids_list) > 0:
        try:
            sample.images = [sample.images[idx] for idx in image_tag_ids_list]
        except Exception as e:
            print(
                f"failed to find image tag in images. images {sample.images} and image_tag_ids_list {image_tag_ids_list}"
            )
            raise e
    # If there is only one image, but several image tags, we assume all the tags refer to the
    # same image and duplicate the image:
    if len(sample.images) == 1 and number_image_tags > 1:
        sample.images = sample.images * number_image_tags

    # We currently only support one video per sample.
    number_of_images = len(sample.images)
    # Fail if there are more image or video tags than image or videos:
    assert (
        number_image_tags <= number_of_images
    ), f"Found {number_image_tags} image tags for {number_of_images} images. {sample.texts}"

    # If there are less image of video tags than image or videos, prepend the tags to the first
    # user message:
    if number_image_tags < number_of_images:
        for turn in conversation:
            if turn["role"] == "user":
                turn["content"] = (
                    IMAGE_TOKEN * (number_of_images - number_image_tags) + "\n" + turn["content"]
                )
                break

    conversation = []
    for conv in data["conversations"]:
        fragments = []
        for fragment in conv:
            if fragment["from"] == "human":
                fragments.append(fragment["value"])
            else:
                fragments.append(fragment["value"])

    return ConversationSample(**basic_sample_keys(sample), conversation=conversation)
