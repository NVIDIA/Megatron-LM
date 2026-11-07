import math
import torch
from types import SimpleNamespace

# Model-specific audio processing parameters
AUDIO_MODEL_PARAMS = {
    "openai/whisper-base": {
        "model_type": "whisper",
        "sample_rate": 16000,  # 16kHz
        "window_stride": 0.01,  # 10ms
        "encoder_down_sampling": 2,
        "d_model": 512,
        "max_length_seconds": 30.0,
    },
}


def calculate_num_mel_frames(audio_length, sample_rate, window_stride, window_length=None):
    """
    Calculate the number of mel frames from an audio signal.

    Parameters:
    - audio_length (int): Total number of audio samples.
    - sample_rate (int or float): Sampling rate of the audio (samples per second).
    - window_stride (float): The time (in seconds) between successive frames.
    - window_length (float, optional): Window length in seconds. If provided, this function
      uses the standard formula: floor((N - window_length_in_samples) / hop_length) + 1.
      Otherwise, it uses the simplified calculation based on the window stride only.

    Returns:
    - int: The number of mel frames.
    """
    hop_length_samples = int(window_stride * sample_rate)

    if window_length is None:
        num_frames = math.ceil((audio_length + 1) / hop_length_samples)
    else:
        window_length_samples = int(window_length * sample_rate)
        num_frames = math.floor((audio_length - window_length_samples) / hop_length_samples) + 1

    return num_frames


def calculate_num_audio_tokens(audio_tensor, model_name):

    # Get audio length in samples
    audio_length = audio_tensor.shape[1]

    # Get model parameters
    if model_name in AUDIO_MODEL_PARAMS:
        model_params = SimpleNamespace(**AUDIO_MODEL_PARAMS[model_name])
        model_type = model_params.model_type
    else:
        raise ValueError(f"Unsupported model name: {model_name}")


    if model_type == "whisper":
        num_mel_frames = calculate_num_mel_frames(
            audio_length, model_params.sample_rate, model_params.window_stride
        )
        encoder_seq_length = math.ceil(num_mel_frames / model_params.encoder_down_sampling)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return max(1, int(encoder_seq_length))


def capped_whisper_lengths(audio_length, model_name="openai/whisper-base", feature_time=None):
    """Mel frames and encoder tokens after the 30 s Whisper cap.

    ``calculate_num_mel_frames`` is unchanged. Mel is then capped at
    ``max_length_seconds * sample_rate // hop`` and at ``feature_time``.
    """
    if model_name not in AUDIO_MODEL_PARAMS:
        raise ValueError(f"Unsupported model name: {model_name}")
    model_params = SimpleNamespace(**AUDIO_MODEL_PARAMS[model_name])
    hop = int(model_params.window_stride * model_params.sample_rate)
    mel_cap = int(model_params.max_length_seconds * model_params.sample_rate) // hop
    mel = calculate_num_mel_frames(
        audio_length, model_params.sample_rate, model_params.window_stride
    )
    mel = min(int(mel), mel_cap)
    if feature_time is not None:
        mel = min(mel, int(feature_time))
    tokens = max(1, math.ceil(mel / model_params.encoder_down_sampling))
    return mel, tokens


def expand_audio_placeholders(prompt: str, counts) -> str:
    """Replace each ``<audio>`` once. Do not use ``str.replace`` on the token."""
    parts = prompt.split("<audio>")
    counts = list(counts)
    if len(parts) != len(counts) + 1:
        raise ValueError(
            f"audio placeholder count {len(parts) - 1} != clip count {len(counts)}"
        )
    return parts[0] + "".join("<audio>" * n + parts[i + 1] for i, n in enumerate(counts))


def stack_or_cat(values):
    """Stack a shared clip shape. Concatenate a variable number of clips."""
    first = values[0]
    if first.ndim in (1, 3):
        return torch.cat(values, dim=0)
    return torch.stack(values, dim=0)
