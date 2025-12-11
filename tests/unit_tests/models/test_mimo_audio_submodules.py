# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

'''
WORLD_SIZE=1 LOCAL_RANK=0 python -m pytest tests/unit_tests/models/test_mimo_audio_submodules.py 
'''
import math
import random

import numpy as np
import pytest
import torch
from transformers import (
    ASTConfig,
    ASTFeatureExtractor,
    ASTModel,
    Wav2Vec2FeatureExtractor,
    WavLMConfig,
    WavLMModel,
    WhisperConfig,
    WhisperFeatureExtractor,
    WhisperModel,
)

from megatron.core.models.mimo.submodules.audio import AudioModalitySubmodules
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from tests.unit_tests.test_utilities import Utils

pytest.importorskip("modelopt", minversion="0.25")
# modelopt version < 0.27 breaks HF AutoModel.from_pretrained API
# so we need to skip the tests unitl versions are bumped in pyt LTS CI container

# Model-specific audio processing parameters
AUDIO_MODEL_PARAMS = {
    "openai/whisper-base": {
        "sample_rate": 16000,  # 16kHz
        "window_stride": 0.01,  # 10ms
        "encoder_down_sampling": 2,
        "d_model": 512,
        "max_length_seconds": 30.0,
    },
    # WavLM models
    "patrickvonplaten/wavlm-libri-clean-100h-base-plus": {
        "sample_rate": 16000,  # 16kHz
        "window_stride": 0.02,  # 20ms
        # Note: WavLM uses a series of convolutional layers with different kernels and strides
        # rather than a single downsampling factor. The overall effect is approximately 320x,
        # but we calculate it precisely using the conv_kernel and conv_stride parameters:
        # conv_kernel = [10, 3, 3, 3, 3, 2, 2]
        # conv_stride = [5, 2, 2, 2, 2, 2, 2]
        "encoder_down_sampling": 1,  # Placeholder, not used for WavLM
        "d_model": 768,
        "max_length_seconds": 30.0,
    },
    # AST model
    "MIT/ast-finetuned-audioset-10-10-0.4593": {
        "sample_rate": 16000,  # 16kHz
        "window_stride": 0.01,  # 10ms for spectrogram creation
        # AST uses fixed-size mel spectrograms and processes with patches
        "max_spectrogram_length": 1024,  # Maximum spectrogram length in frames
        "num_mel_bins": 128,  # Number of mel bins
        "patch_size": 16,  # Size of each patch
        "time_stride": 10,  # Stride for time dimension
        "frequency_stride": 10,  # Stride for frequency dimension
        "d_model": 768,  # Hidden size
        "max_length_seconds": 10.0,  # Reasonable maximum for testing
    },
}


class AudioEncoderWrapper(torch.nn.Module):
    """Generic wrapper for audio encoder models that extracts last_hidden_state."""

    def __init__(self, encoder, model_type="whisper"):
        super().__init__()
        self.encoder = encoder
        self.model_type = model_type

    def forward(self, input_features, seq_lengths=None):
        with torch.no_grad():
            hidden = self.encoder(input_features).last_hidden_state  # [b, s, h]
            if seq_lengths is not None:
                seq_len = hidden.shape[1]
                # breakpoint()
                mask = torch.arange(seq_len, device=hidden.device)[None, :] < seq_lengths[:, None]
                hidden = hidden[mask]
            return hidden


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


class TestAudioSubmodule:
    """Test the AudioModalitySubmodules class with forward passes."""

    def setup_method(self, method, model_name="openai/whisper-base"):
        '''setup env'''
        # Initialize distributed environment
        try:
            Utils.initialize_model_parallel(1, 1)
        except Exception as e:
            print(f"Warning: Could not initialize model parallel: {e}")

        model_parallel_cuda_manual_seed(123)
        random.seed(123)  # For reproducible random test cases

        # Get model-specific parameters
        if model_name not in AUDIO_MODEL_PARAMS:
            raise ValueError(
                f"Model {model_name} not supported. Available models: {list(AUDIO_MODEL_PARAMS.keys())}"
            )

        model_params = AUDIO_MODEL_PARAMS[model_name]

        # Audio processing parameters
        self.sample_rate = model_params["sample_rate"]
        self.window_stride = model_params.get("window_stride", 0.01)
        self.sample_per_mel_frame = int(self.window_stride * self.sample_rate)
        self.encoder_down_sampling = model_params.get("encoder_down_sampling", 1)
        self.max_length_seconds = model_params["max_length_seconds"]

        # For AST model
        self.max_spectrogram_length = model_params.get("max_spectrogram_length", None)
        self.num_mel_bins = model_params.get("num_mel_bins", None)
        self.patch_size = model_params.get("patch_size", None)
        self.time_stride = model_params.get("time_stride", None)
        self.frequency_stride = model_params.get("frequency_stride", None)

        self.audio_token_id = 50000

        # Keep name for logs
        self.model_name = model_name

        # Decide model type
        if "whisper" in model_name:
            self.model_type = "whisper"
            config = WhisperConfig()
            model = WhisperModel(config)
            raw_encoder = model.encoder
            self.processor = WhisperFeatureExtractor()
        elif "wavlm" in model_name:
            self.model_type = "wavlm"
            config = WavLMConfig()
            model = WavLMModel(config)
            raw_encoder = model
            self.processor = Wav2Vec2FeatureExtractor()
        elif "ast" in model_name.lower():
            self.model_type = "ast"
            config = ASTConfig(
                num_mel_bins=self.num_mel_bins,
                patch_size=self.patch_size,
                fstride=self.frequency_stride,
                tstride=self.time_stride,
            )
            model = ASTModel(config)
            raw_encoder = model
            self.processor = ASTFeatureExtractor()
        else:
            raise ValueError(f"Unsupported model type: {model_name}")

        self.encoder = AudioEncoderWrapper(raw_encoder, self.model_type)
        if hasattr(model.config, "d_model"):
            self.d_model = model.config.d_model
        else:
            self.d_model = model_params["d_model"]
        self.projection = torch.nn.Linear(self.d_model, 768)
        self.audio_module = AudioModalitySubmodules(
            encoders={"encoder": self.encoder}, input_projections=[self.projection]
        )

    def teardown_method(self, method):
        '''teardown env'''
        try:
            Utils.destroy_model_parallel()
        except Exception as e:
            print(f"Warning: Could not destroy model parallel: {e}")

    def _create_sample_audio(self, duration_seconds, sample_rate=None):
        """Create a sample audio waveform.

        Args:
            duration_seconds (float): Duration of audio in seconds
            sample_rate (int, optional): Sample rate in Hz. Defaults to self.sample_rate.

        Returns:
            torch.Tensor: Audio waveform of shape [1, samples]
        """
        sample_rate = sample_rate or self.sample_rate

        # Create a time array
        t = np.linspace(0, duration_seconds, int(duration_seconds * sample_rate), endpoint=False)

        # Create a simple sine wave at 440 Hz (A4)
        frequency = 440.0
        waveform = 0.5 * np.sin(2 * np.pi * frequency * t)

        # Convert to torch tensor
        return torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)

    def _calculate_seq_length(self, audio_tensor):

        # Get audio length in samples
        audio_length = audio_tensor.shape[1]

        if self.model_type in ["whisper"]:

            num_mel_frames = calculate_num_mel_frames(
                audio_length, self.sample_rate, self.window_stride
            )
            encoder_seq_length = math.ceil(num_mel_frames / self.encoder_down_sampling)

        elif self.model_type == "wavlm":
            # For WavLM, use the exact convolutional calculation logic
            # WavLM uses a series of convolutional layers with different kernels and strides
            conv_kernel = [10, 3, 3, 3, 3, 2, 2]
            conv_stride = [5, 2, 2, 2, 2, 2, 2]

            # Function to calculate output length of 1D convolution
            def _conv_out_length(input_length, kernel_size, stride):
                # 1D convolutional layer output length formula taken
                # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
                return torch.div(input_length - kernel_size, stride, rounding_mode="floor") + 1

            # Start with the original input length
            input_length = audio_length

            # Apply each convolutional layer
            for kernel_size, stride in zip(conv_kernel, conv_stride):
                input_length = _conv_out_length(input_length, kernel_size, stride)

            # The result is the encoder sequence length
            encoder_seq_length = input_length

        elif self.model_type == "ast":
            # AST uses a fixed-size spectrogram and divides it into patches
            # The exact formula is based on how CNN output dimensions are calculated
            # See: https://cs231n.github.io/convolutional-networks/#conv
            frequency_out_dimension = (
                self.num_mel_bins - self.patch_size
            ) // self.frequency_stride + 1
            time_out_dimension = (
                self.max_spectrogram_length - self.patch_size
            ) // self.time_stride + 1

            # Number of patches is the product of these dimensions
            num_patches = frequency_out_dimension * time_out_dimension

            # Add 2 for the cls_token and distillation_token
            encoder_seq_length = num_patches + 2

            print(
                f"AST patches: freq_dim={frequency_out_dimension}, time_dim={time_out_dimension}, "
                f"patches={num_patches}, total={encoder_seq_length}"
            )

        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        return max(1, int(encoder_seq_length))

    def _create_batch(self, batch_size=3, min_duration=1.0, max_duration=1.5):
        """
        Create a simple batch with mixed text and audio content.
        """
        # Use default parameters if not provided
        sample_rate = self.sample_rate

        # Randomly choose 1-4 audio segments per sample
        num_segments_per_sample = [random.randint(1, 4) for _ in range(batch_size)]
        total_segments = sum(num_segments_per_sample)
        audio_samples = [
            self._create_sample_audio(random.uniform(min_duration, max_duration), sample_rate)
            for _ in range(total_segments)
        ]

        processor_kwargs = {"sampling_rate": sample_rate, "return_tensors": "pt"}
        # processor for whisper (30 sec) and ast pads (1024 framesto max length
        # for wavlm lets pad to longest in the batch
        if self.model_type in ["wavlm"]:
            processor_kwargs["padding"] = "longest"
        processed = self.processor(
            [sample.squeeze().numpy() for sample in audio_samples], **processor_kwargs
        )

        if self.model_type == "whisper":
            processed_features = processed.input_features
        elif self.model_type in ["ast", "wavlm"]:
            processed_features = processed.input_values
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        # Calculate sequence lengths for audio tokens
        seq_lengths = torch.tensor(
            [self._calculate_seq_length(sample) for sample in audio_samples], dtype=torch.long
        )

        max_seq_len = 4096  # Arbitrary length that's enough for test
        input_ids = torch.zeros((batch_size, max_seq_len), dtype=torch.long)

        # Keep track of which audio segment we're using
        segment_idx = 0

        # Fill input_ids with text and audio tokens
        for i in range(batch_size):
            pos = 0
            num_segments = num_segments_per_sample[i]

            for _ in range(num_segments):
                # Random text
                text_len = random.randint(3, 8)
                input_ids[i, pos : pos + text_len] = torch.randint(1, 30000, (text_len,))
                pos += text_len

                # Audio segment
                audio_len = seq_lengths[segment_idx].item()
                input_ids[i, pos : pos + audio_len] = self.audio_token_id
                pos += audio_len
                segment_idx += 1

            # Final text
            text_len = random.randint(3, 8)
            if pos + text_len < max_seq_len:
                input_ids[i, pos : pos + text_len] = torch.randint(1, 30000, (text_len,))
                pos += text_len

            # Padding
            if pos < max_seq_len:
                input_ids[i, pos:] = 0

        return {
            'audio': processed_features,
            'input_ids': input_ids,
            'modality_seq_lengths': {'audio': seq_lengths},
        }

    @pytest.mark.parametrize(
        "model_name,batch_size",
        [
            # Test with batch_size=1
            pytest.param("openai/whisper-base", 1, id="whisper-base-batch1"),
            pytest.param("patrickvonplaten/wavlm-libri-clean-100h-base-plus", 1, id="wavlm-batch1"),
            pytest.param("MIT/ast-finetuned-audioset-10-10-0.4593", 1, id="ast-batch1"),
            # Test with batch_size=2
            pytest.param("openai/whisper-base", 2, id="whisper-base-batch2"),
            pytest.param("patrickvonplaten/wavlm-libri-clean-100h-base-plus", 2, id="wavlm-batch2"),
            pytest.param("MIT/ast-finetuned-audioset-10-10-0.4593", 2, id="ast-batch2"),
        ],
    )
    def test_multiple_audio_encoders(self, model_name, batch_size):
        '''Test the forward pass with different audio encoder models and batch sizes'''
        self.setup_method(None, model_name=model_name)

        batch = self._create_batch(batch_size=batch_size, min_duration=1.0, max_duration=3.0)

        feature_key = "input_features"

        # Create encoder inputs dictionary with named encoder
        seq_lengths = batch['modality_seq_lengths']['audio']
        encoder_inputs = {"encoder": {feature_key: batch['audio'], "seq_lengths": seq_lengths}}

        # Call forward with new interface
        embeddings = self.audio_module.forward(encoder_inputs)

        num_audio_tokens = (batch['input_ids'] == self.audio_token_id).sum().item()

        # Verify number of embeddings matches number of audio tokens
        assert embeddings.shape[0] == num_audio_tokens

        # Verify embeddings have expected dimension (768 is our target dimension)
        assert embeddings.shape[1] == 768

        print(
            f"Model {model_name} (d_model={self.d_model}) successfully processed audio and projected to dimension 768"
        )
