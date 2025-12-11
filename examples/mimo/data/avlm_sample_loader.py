import io

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

    # Get raw audio bytes
    audio_names = [audio_name.split('.', 1)[1] for audio_name in jsn_content["audios"]] # example of audio_name file: "100_100.synthesized_683016ff-bb67-4a12-a873-a7e5d4132903.wav.flac" -> "synthesized_683016ff-bb67-4a12-a873-a7e5d4132903.wav"
    audio_name = audio_names[0]  # for now, take only the first audio
    audio_bytes = raw[audio_name]

    # process conversation, take only the first turn
    context = conversation[0]["value"]
    answers = conversation[1]["value"]

    # check that context only contains one "<audio>"
    # if contains more than one, replace all occurrences with a single one
    if context.count("<audio>") > 1:
        parts = context.split("<audio>")
        context = parts[0] + "<audio>" + "".join(parts[1:])

    return dict(
        __key__=raw["__key__"],
        context=context,
        answers=answers,
        image=image_bytes,
        audio=audio_bytes,
    )

def part_filter(part: str) -> bool:
    return True