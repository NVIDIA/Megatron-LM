def _all_audio_clips() -> bool:
    try:
        from megatron.training import get_args

        return bool(getattr(get_args(), "all_audio_clips", False))
    except Exception:
        return False


def reject_clip_list(audio, all_clips: bool) -> None:
    if isinstance(audio, (list, tuple)) and not all_clips:
        raise ValueError("encode_sample rejects a list of audio clips")


def _collapse_extra_audio(context: str) -> str:
    if context.count("<audio>") > 1:
        parts = context.split("<audio>")
        return parts[0] + "<audio>" + "".join(parts[1:])
    return context


def sample_loader(raw: dict) -> dict:
    """
    Load the sample from the raw data.
    Example content of wds data:
        - Example of wds data
            100_100.img
            100_100.json
            100_100.synthesized_683016ff-bb67-4a12-a873-a7e5d4132903.wav.flac
            100_100.synthesized_6a7b3a1c-05a9-4720-b7e6-f7028415dc71.wav.flac
            100_100.synthesized_7c4b39ba-20ef-4319-bd94-291c94ad362a.wav.flac
            100_100.synthesized_932f067b-2b9a-425d-b8e6-f74ca83335a2.wav.flac
            Content of 100_100.json: 
                {
                    "num_image": 7,
                    "length": 2049,
                    "label_length": 157,
                    "conversations": [
                        {
                            "from": "human",
                            "value": "<image><audio>"
                        },
                        {
                            "from": "gpt",
                            "value": "The stop sign is red, while the one-way signs are typically black and white."
                        },
                        ...
                        {
                            "from": "human",
                            "value": "<audio>"
                        },
                        {
                            "from": "gpt",
                            "value": "The purpose of these street signs is to communicate..."
                        }
                    ],
                    "audios": [
                        "100_100.synthesized_6a7b3a1c-05a9-4720-b7e6-f7028415dc71.wav.flac",
                        "100_100.synthesized_932f067b-2b9a-425d-b8e6-f74ca83335a2.wav.flac",
                        "100_100.synthesized_7c4b39ba-20ef-4319-bd94-291c94ad362a.wav.flac",
                        "100_100.synthesized_683016ff-bb67-4a12-a873-a7e5d4132903.wav.flac"
                    ],
                    "images": [
                        "100_100.img"
                    ]
                }
        - Example of read raw data for Energon
            raw["json"] -> json content
            raw["img"] -> bytes image content
            raw["synthesized_6a7b3a1c-05a9-4720-b7e6-f7028415dc71.wav.flac"] -> bytes audio content
            raw["synthesized_932f067b-2b9a-425d-b8e6-f74ca83335a2.wav.flac"] -> bytes audio content
            raw["synthesized_683016ff-bb67-4a12-a873-a7e5d4132903.wav.flac"] -> bytes audio content
    """

    jsn_content = raw["json"]
    conversation = jsn_content["conversations"]

    # Get raw image bytes
    image_bytes = raw["img"]

    # example: "100_100.synthesized_....wav.flac" -> "synthesized_....wav"
    audio_names = [audio_name.split(".", 1)[1] for audio_name in jsn_content["audios"]]

    if not _all_audio_clips():
        # First file and first turn. Extra "<audio>" tags in that turn collapse.
        context = _collapse_extra_audio(conversation[0]["value"])
        answers = conversation[1]["value"]
        return dict(
            __key__=raw["__key__"],
            context=context,
            answers=answers,
            image=image_bytes,
            audio=raw[audio_names[0]],
        )

    if len(conversation) % 2 != 0:
        raise ValueError(f"conversations are not paired for {raw['__key__']}")
    contexts = []
    answers = []
    user_tags = 0
    for turn in range(0, len(conversation), 2):
        user = conversation[turn]["value"]
        assistant = conversation[turn + 1]["value"]
        if "<audio>" in assistant:
            raise ValueError(f"assistant turn contains <audio> for {raw['__key__']}")
        user_tags += user.count("<audio>")
        contexts.append(user)
        answers.append(assistant)
    if user_tags != len(audio_names):
        raise ValueError(
            f"audio tag count {user_tags} != file count {len(audio_names)} "
            f"for {raw['__key__']}"
        )
    blobs = []
    for name in audio_names:
        if name not in raw:
            raise ValueError(f"missing audio blob {name} for {raw['__key__']}")
        blobs.append(raw[name])
    return dict(
        __key__=raw["__key__"],
        context=contexts,
        answers=answers,
        image=image_bytes,
        audio=blobs,
    )

def part_filter(part: str) -> bool:
    return True